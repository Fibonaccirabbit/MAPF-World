"""Training transitions: LaCAM demonstrations, DDG and world-error collection."""

import argparse
import ctypes
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from tqdm import tqdm

from model import tokenizer as cost2go
from model.tokenizer import MOVES, Encoder, InputParameters, ObservationEncoder
from train.dataset import SCHEMA
from utils.scenarios import (
    ScenarioSuite,
    UnrollWrapper,
    make_training_env,
    run_directory,
    sha256,
    write_json,
)

# Adapted from MAPF-GPT's LaCAM wrapper.
# Solver source: https://github.com/Kei18/lacam3.


_LOCK = threading.Lock()
LOG = logging.getLogger(__name__)


class LacamExpert:
    def __init__(self, library, time_limit=10.0):
        if not np.isfinite(time_limit) or time_limit <= 0:
            raise ValueError("Expert time limit must be finite and positive")
        self.path = Path(library).resolve(strict=True)
        self.library = ctypes.CDLL(str(self.path))
        self.library.run_lacam.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_float,
        ]
        self.library.run_lacam.restype = ctypes.c_char_p
        self.time_limit = time_limit

    def plan(self, observations):
        grid = np.asarray(observations[0]["global_obstacles"])
        starts = np.asarray([obs["global_xy"] for obs in observations], dtype=int)
        goals = np.asarray([obs["global_target_xy"] for obs in observations], dtype=int)
        if len({tuple(p) for p in goals}) != len(goals):
            raise ValueError("LaCAM requires distinct goals")
        height, width = grid.shape
        map_text = f"type octile\nheight {height}\nwidth {width}\nmap\n"
        map_text += "\n".join("".join("@" if c else "." for c in row) for row in grid)
        scene = "version 1\n" + "\n".join(
            f"{i}\ttmp.map\t{width}\t{height}\t{s[1]}\t{s[0]}\t{g[1]}\t{g[0]}\t1"
            for i, (s, g) in enumerate(zip(starts, goals))
        )
        with _LOCK:
            raw = self.library.run_lacam(
                map_text.encode(), scene.encode(), len(starts), self.time_limit
            )
            result = raw.decode() if raw else "ERROR_NULL"
        if result.startswith("ERROR"):
            return None
        path = np.asarray(
            [
                [tuple(map(int, cell.split(",")))[::-1] for cell in line.split("|") if cell]
                for line in result.strip().splitlines()
            ],
            dtype=int,
        )
        if (
            path.shape[1:] != starts.shape
            or not np.array_equal(path[0], starts)
            or not np.array_equal(path[-1], goals)
        ):
            raise RuntimeError("LaCAM returned inconsistent endpoints")
        matches = (np.diff(path, axis=0)[:, :, None, :] == MOVES[None, None, :, :]).all(-1)
        if not matches.any(-1).all():
            raise RuntimeError("LaCAM returned non-adjacent moves")
        for step, positions in enumerate(path):
            if (
                len({tuple(p) for p in positions}) != len(starts)
                or grid[positions[:, 0], positions[:, 1]].any()
            ):
                raise RuntimeError("LaCAM returned a vertex or obstacle collision")
            if step:
                previous = {tuple(p): i for i, p in enumerate(path[step - 1])}
                for i, p in enumerate(positions):
                    j = previous.get(tuple(p))
                    if j is not None and j != i and np.array_equal(positions[j], path[step - 1, i]):
                        raise RuntimeError("LaCAM returned an edge-swap collision")
        return path, matches.argmax(-1).astype(np.int8)


def build_lacam_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="LaCAM3 checkout with src/ and include/"
    )
    parser.add_argument("--output", type=Path, required=True, help="New shared-library file")
    parser.add_argument("--compiler", default=os.environ.get("CXX", "c++"))
    args = parser.parse_args(argv)
    source = args.source.resolve()
    if (source / "lacam3/include/lacam.hpp").is_file():
        source = source / "lacam3"
    files = sorted((source / "src").glob("*.cpp"))
    if not files or not (source / "include/lacam.hpp").is_file():
        parser.error("Expected a LaCAM3 source checkout, not a build directory")
    if args.output.exists():
        parser.error("Output already exists")
    compiler = shutil.which(args.compiler)
    if not compiler:
        parser.error("A C++17 compiler is required")
    bridge = Path(__file__).resolve().parent / "native/lacam_bridge.cpp"
    if not bridge.is_file():
        parser.error("LaCAM bridge source is missing from this installation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        compiler,
        "-std=c++17",
        "-O3",
        "-fPIC",
        "-pthread",
        "-dynamiclib" if sys.platform == "darwin" else "-shared",
        "-I",
        str(source / "include"),
        str(bridge),
        *map(str, files),
        "-o",
        str(args.output.resolve()),
    ]
    print(f"Building LaCAM bridge from {len(files)} solver sources...", flush=True)
    subprocess.run(command, check=True)
    print(f"Built {args.output.resolve()}")


# Adapted from MAPF-World generate_dataset.py.


class TransitionWriter:
    def __init__(self, output, rows_per_shard):
        if rows_per_shard < 1:
            raise ValueError("rows_per_shard must be positive")
        self.output, self.capacity = output, rows_per_shard
        self.rows, self.files, self.total = [], [], 0

    def add(self, obs, next_obs, actions):
        for before, after, action in zip(obs, next_obs, actions):
            self.rows.append(
                {"obs": before.tolist(), "next_obs": after.tolist(), "next_action": int(action)}
            )
            if len(self.rows) == self.capacity:
                self.flush()

    def flush(self):
        if not self.rows:
            return
        name = f"train-{len(self.files):05d}.arrow"
        temporary = self.output / (name + ".tmp")
        with pa.OSFile(str(temporary), "wb") as sink:
            with ipc.new_file(sink, SCHEMA) as writer:
                writer.write_table(pa.Table.from_pylist(self.rows, schema=SCHEMA))
        temporary.replace(self.output / name)
        self.files.append(name)
        self.total += len(self.rows)
        self.rows.clear()


def generate(
    suite,
    expert,
    output,
    limit=None,
    rows_per_shard=10000,
    cache_mb=128,
    coordinate_encoding="clip",
):
    if suite.identity["split"] != "train":
        raise ValueError("Dataset generation accepts training suites only")
    if rows_per_shard < 1 or cache_mb < 1 or (limit is not None and limit < 1):
        raise ValueError("Shard size, cache size and limit must be positive")
    count = suite.total if limit is None else min(limit, suite.total)
    metadata = {
        **suite.identity,
        "selected_episodes": count,
        "expert_library_sha256": sha256(expert.path),
        "expert_time_limit": expert.time_limit,
        "coordinate_encoding": coordinate_encoding,
        "rows_per_shard": rows_per_shard,
        "terminal_padding": False,
        "wait_downsampling": False,
    }
    with run_directory(output, metadata) as (output, log):
        writer = TransitionWriter(output, rows_per_shard)
        solved, skipped = 0, 0
        log.info("Generating training transitions from %d/%d scenarios", count, suite.total)
        with (output / "episodes.jsonl").open("x") as ledger:
            for index, episode in enumerate(
                tqdm(suite.episodes(limit), total=count, desc="Training data", unit="episode")
            ):
                env = suite.create_env(episode, animation=False)
                try:
                    observations, _ = env.reset()
                    plan = expert.plan(observations)
                    reason = None
                    if plan is None:
                        reason = "expert_failed"
                    elif len(plan[1]) == 0:
                        reason = "already_solved"
                    elif len(plan[1]) > env.grid_config.max_episode_steps:
                        reason = "expert_exceeds_episode_horizon"
                    if reason:
                        skipped += 1
                        ledger.write(
                            json.dumps({"episode": index, "environment": episode, "status": reason})
                            + "\n"
                        )
                        log.warning("Skipping episode %d: %s", index, reason)
                        continue
                    path, actions = plan
                    # Validate the full trajectory against the actual collision rules
                    # before exporting any of its rows.
                    for step, joint_action in enumerate(actions):
                        observations, _, terminated, truncated, infos = env.step(
                            joint_action.tolist()
                        )
                        actual = np.asarray([obs["global_xy"] for obs in observations])
                        if not np.array_equal(actual, path[step + 1]):
                            raise RuntimeError(
                                "Expert replay disagrees with POGEMA; no manifest published"
                            )
                        if all(terminated) or all(truncated):
                            if step + 1 != len(actions):
                                raise RuntimeError("Expert replay ended before its final step")
                    if infos[0].get("metrics", {}).get("CSR") != 1.0:
                        raise RuntimeError("Expert trajectory did not jointly solve the scenario")
                    observations, _ = env.reset()
                    if not np.array_equal([obs["global_xy"] for obs in observations], path[0]):
                        raise RuntimeError("Scenario reset is not reproducible")
                    encoder = ObservationEncoder(cache_mb, coordinate_encoding)
                    before = encoder.encode(observations)
                    for joint_action in actions:
                        observations, *_ = env.step(joint_action.tolist())
                        after = encoder.encode(observations)
                        writer.add(before, after, joint_action)
                        before = after
                    solved += 1
                    ledger.write(
                        json.dumps(
                            {
                                "episode": index,
                                "environment": episode,
                                "status": "solved",
                                "rows": int(actions.size),
                                "steps": len(actions),
                            }
                        )
                        + "\n"
                    )
                    ledger.flush()
                    log.info(
                        "Episode %d/%d | solved=%d | skipped=%d | rows=%d",
                        index + 1,
                        count,
                        solved,
                        skipped,
                        writer.total + len(writer.rows),
                    )
                finally:
                    env.close()
        writer.flush()
        if not writer.total:
            raise RuntimeError(
                "No successful transitions generated; no training manifest published"
            )
        manifest = {
            "split": "train",
            "files": writer.files,
            "rows": writer.total,
            "solved_episodes": solved,
            "skipped_episodes": skipped,
            "coordinate_encoding": coordinate_encoding,
        }
        write_json(output / "train.json", manifest)
        log.info(
            "Training manifest: %s (%d rows, %d shards)",
            output / "train.json",
            writer.total,
            len(writer.files),
        )
        return manifest


# DDG/world-error algorithms adapted from MAPF-World finetuning/transition_collection.py.

ACTION_TO_CHAR = {
    -1: "n",
    0: "w",
    1: "u",
    2: "d",
    3: "l",
    4: "r",
}


class TransitionEncoder:
    def __init__(self, cfg=None):
        self.cfg = cfg or InputParameters()
        self.encoder = Encoder(self.cfg)
        self.cost2go_data = None
        self.actions_history = None

    def reset(self, observations, previous_actions=None):
        self.cost2go_data = cost2go.precompute_cost2go(
            observations[0]["global_obstacles"].copy().astype(int).tolist(),
            self.cfg.cost2go_radius,
        )
        num_agents = len(observations)
        self.actions_history = [
            ["n" for _ in range(self.cfg.num_previous_actions)] for _ in range(num_agents)
        ]
        if previous_actions:
            for actions in previous_actions[-self.cfg.num_previous_actions :]:
                self.update_actions(actions)

    def update_actions(self, actions):
        for agent_idx, action in enumerate(actions):
            self.actions_history[agent_idx].append(ACTION_TO_CHAR[int(action)])
            self.actions_history[agent_idx] = self.actions_history[agent_idx][
                -self.cfg.num_previous_actions :
            ]

    def encode(self, observations):
        next_actions = []
        for obs in observations:
            next_action = ""
            for move in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                new_pos = (obs["global_xy"][0] + move[0], obs["global_xy"][1] + move[1])
                if (
                    self.cost2go_data[obs["global_target_xy"]][new_pos[0]][new_pos[1]] >= 0
                    and self.cost2go_data[obs["global_target_xy"]][obs["global_xy"][0]][
                        obs["global_xy"][1]
                    ]
                    > self.cost2go_data[obs["global_target_xy"]][new_pos[0]][new_pos[1]]
                ):
                    next_action += "1"
                else:
                    next_action += "0"
            next_actions.append(next_action)

        encoded = []
        global_xy = [obs["global_xy"] for obs in observations]
        for agent_idx, obs in enumerate(observations):
            distances = []
            for other_idx, pos in enumerate(global_xy):
                distance = self.cost2go_data[tuple(global_xy[agent_idx])][pos[0]][pos[1]]
                if distance >= 0:
                    distances.append((other_idx, distance))
            distances.sort(key=lambda item: (item[1], item[0]))

            agents_info = []
            for other_idx, _ in distances[: self.cfg.num_agents]:
                relative_xy = (
                    observations[other_idx]["global_xy"][0] - obs["global_xy"][0],
                    observations[other_idx]["global_xy"][1] - obs["global_xy"][1],
                )
                if (
                    -self.cfg.agents_radius <= relative_xy[0] <= self.cfg.agents_radius
                    and -self.cfg.agents_radius <= relative_xy[1] <= self.cfg.agents_radius
                ):
                    relative_goal = (
                        observations[other_idx]["global_target_xy"][0] - obs["global_xy"][0],
                        observations[other_idx]["global_target_xy"][1] - obs["global_xy"][1],
                    )
                    agents_info.append(
                        {
                            "relative_pos": relative_xy,
                            "relative_goal": relative_goal,
                            "previous_actions": self.actions_history[other_idx],
                            "next_action": next_actions[other_idx],
                        }
                    )

            encoded.append(
                self.encoder.encode(
                    {
                        "agents": agents_info,
                        "cost2go": cost2go.generate_cost2go_obs(
                            self.cost2go_data[obs["global_target_xy"]],
                            obs["global_xy"],
                            self.cfg.cost2go_radius,
                            self.cfg.cost2go_value_limit,
                            self.cfg.mask_cost2go,
                        ),
                    }
                )
            )
        return encoded


def run_recorded_episode(env, algo):
    algo.reset_states()
    observations, _ = env.reset()
    infos = None
    while True:
        actions = algo.act(observations)
        observations, _, terminated, truncated, infos = env.step(actions)
        if all(terminated) or all(truncated):
            break
    return infos[0].get("metrics", {}) if infos else {}


def run_solver_from_unroll(env, unroll_steps, time_limit, expert_factory):
    env = deepcopy(env)
    env.set_unroll_steps(unroll_steps)
    solver = expert_factory(time_limit)
    metrics = run_recorded_episode(env, solver)
    metrics["step"] = unroll_steps
    return metrics


def select_delta_steps(env, ep_length, cfg, expert_factory):
    unroll_steps = list(range(0, ep_length, cfg.steps_delta))
    if len(unroll_steps) < 2:
        return unroll_steps if cfg.collect_on_success else [], {}

    fast_results = {
        step: run_solver_from_unroll(env, step, cfg.fast_time_limit, expert_factory)
        for step in unroll_steps
    }
    diffs = []
    for prev_step, curr_step in zip(unroll_steps[:-1], unroll_steps[1:]):
        prev_makespan = int(fast_results[prev_step].get("makespan", ep_length))
        curr_makespan = int(fast_results[curr_step].get("makespan", ep_length))
        diffs.append(curr_makespan - prev_makespan)

    if not diffs:
        return [], {"fast_solver_results": fast_results, "diffs": diffs}

    max_diff_idx = int(np.argmax(diffs))
    max_diff = diffs[max_diff_idx]
    if max_diff > cfg.diff_threshold:
        selected_steps = [unroll_steps[max_diff_idx]]
    elif cfg.collect_on_success:
        selected_steps = unroll_steps
    else:
        selected_steps = []

    return selected_steps, {
        "fast_solver_results": fast_results,
        "diffs": diffs,
        "selected_steps": selected_steps,
    }


def collect_transitions_with_solver(env, start_step, steps_to_collect, chosen_agents, expert_algo):
    expert_algo.reset_states()
    observations, _ = env.reset()
    previous_actions = [env.get_actions_at_step(step) for step in range(start_step - 5, start_step)]
    encoder = TransitionEncoder()
    encoder.reset(observations, previous_actions=previous_actions)

    obs_tokens = []
    next_obs_tokens = []
    next_actions = []
    infos = None
    for _ in range(steps_to_collect):
        current_tokens = encoder.encode(observations)
        actions = expert_algo.act(observations)
        if hasattr(expert_algo, "solved") and not expert_algo.solved:
            LOG.debug(f"Expert failed from step {start_step}")
            return (
                [],
                [],
                [],
                {"ISR": 0.0, "CSR": 0.0, "ep_length": 256, "SoC": -1, "makespan": 256},
            )
        next_observations, _, terminated, truncated, infos = env.step(actions)
        encoder.update_actions(actions)
        next_tokens = encoder.encode(next_observations)

        for agent_idx in chosen_agents:
            obs_tokens.append(current_tokens[agent_idx])
            next_obs_tokens.append(next_tokens[agent_idx])
            next_actions.append(actions[agent_idx])

        observations = next_observations
        if all(terminated) or all(truncated):
            break

    metrics = infos[0].get("metrics", {}) if infos else {}
    LOG.debug(f"Collected {len(obs_tokens)} transitions from step {start_step}")
    return obs_tokens, next_obs_tokens, next_actions, metrics


@dataclass
class TransitionDDGConfig:
    steps_delta: int = 16
    steps_saved: int = 32
    diff_threshold: int = 3
    fast_time_limit: float = 2.0
    expert_time_limit: float = 10.0
    collect_on_success: bool = False

    def validate(self):
        if self.steps_delta < 1 or self.steps_saved < 1 or self.diff_threshold < 0:
            raise ValueError("DDG steps must be positive and diff_threshold nonnegative")
        if any(
            not np.isfinite(t) or t <= 0 for t in (self.fast_time_limit, self.expert_time_limit)
        ):
            raise ValueError("Solver time limits must be finite and positive")


def collect_ddg_transitions(env, learnable_algo, cfg: TransitionDDGConfig, expert_factory):
    from pogema.wrappers.metrics import RuntimeMetricWrapper

    cfg.validate()
    env = UnrollWrapper(RuntimeMetricWrapper(env))
    gpt_metrics = run_recorded_episode(env, learnable_algo)
    ep_length = int(gpt_metrics.get("ep_length", 0))
    if ep_length <= 0:
        ep_length = int(env.grid_config.max_episode_steps)

    selected_steps, delta_logs = select_delta_steps(env, ep_length, cfg, expert_factory)
    if not selected_steps:
        return (
            {"obs": [], "next_obs": [], "next_action": []},
            {"gpt_results": gpt_metrics, **delta_logs},
        )

    expert = expert_factory(cfg.expert_time_limit)
    chosen_agents = list(range(env.grid_config.num_agents))
    all_obs = []
    all_next_obs = []
    all_actions = []
    expert_logs = []

    for start_step in selected_steps:
        env.set_unroll_steps(start_step)
        obs, next_obs, actions, metrics = collect_transitions_with_solver(
            env,
            start_step,
            cfg.steps_saved,
            chosen_agents,
            expert,
        )
        all_obs.extend(obs)
        all_next_obs.extend(next_obs)
        all_actions.extend(actions)
        expert_logs.append({"step": start_step, **metrics})

    return (
        {
            "obs": np.asarray(all_obs, dtype=np.int8),
            "next_obs": np.asarray(all_next_obs, dtype=np.int8),
            "next_action": np.asarray(all_actions, dtype=np.int8),
        },
        {
            "gpt_results": gpt_metrics,
            **delta_logs,
            "expert_results": expert_logs,
        },
    )


GREEDY_POSITIONS = [121 + i * 10 + 9 for i in range(13)]


def compute_greedy_error(predicted, actual):
    error = 0
    for pos in GREEDY_POSITIONS:
        if actual[pos] == 66:
            continue
        if int(predicted[pos]) != int(actual[pos]):
            error += 1
    return error


@dataclass
class WorldErrorDDGConfig:
    max_steps: int = 256
    max_transitions: int = 512

    def validate(self):
        if self.max_steps < 1 or self.max_transitions < 1:
            raise ValueError("World-error steps and transition limit must be positive")


def collect_world_error_transitions(env, learnable_algo, cfg: WorldErrorDDGConfig):
    from pogema.wrappers.metrics import RuntimeMetricWrapper

    cfg.validate()
    env = RuntimeMetricWrapper(env)
    learnable_algo.reset_states()
    observations, _ = env.reset()

    encoder = TransitionEncoder()
    encoder.reset(observations)

    step_data = []

    for step in range(cfg.max_steps):
        current_tokens = encoder.encode(observations)
        actions = learnable_algo.act(observations)
        predicted_tokens = getattr(learnable_algo, "last_predicted_tokens", None)
        if predicted_tokens is None:
            predicted_tokens = learnable_algo.predict_next_tokens()
        if np.asarray(predicted_tokens).shape != (len(observations), 256):
            raise ValueError("World prediction must contain one 256-token row per agent")

        next_observations, _, terminated, truncated, infos = env.step(actions)
        encoder.update_actions(actions)
        actual_next_tokens = encoder.encode(next_observations)

        for agent_idx in range(len(observations)):
            error = compute_greedy_error(predicted_tokens[agent_idx], actual_next_tokens[agent_idx])
            if error > 0:
                step_data.append(
                    {
                        "obs": current_tokens[agent_idx],
                        "next_obs": actual_next_tokens[agent_idx],
                        "action": actions[agent_idx],
                        "error": error,
                    }
                )

        observations = next_observations
        if all(terminated) or all(truncated):
            break

    metrics = infos[0].get("metrics", {}) if infos else {}
    step_data.sort(key=lambda x: x["error"], reverse=True)
    selected = step_data[: cfg.max_transitions]

    if not selected:
        return (
            {"obs": [], "next_obs": [], "next_action": []},
            {"metrics": metrics, "total_candidates": 0, "ep_length": step + 1},
        )

    return (
        {
            "obs": np.asarray([s["obs"] for s in selected], dtype=np.int8),
            "next_obs": np.asarray([s["next_obs"] for s in selected], dtype=np.int8),
            "next_action": np.asarray([s["action"] for s in selected], dtype=np.int8),
        },
        {
            "metrics": metrics,
            "total_candidates": len(step_data),
            "selected": len(selected),
            "max_error": selected[0]["error"] if selected else 0,
            "ep_length": step + 1,
        },
    )


class LacamCollectionPolicy:
    """LaCAM expert adapter for collection episodes."""

    def __init__(self, library, time_limit):
        self.expert = LacamExpert(library, time_limit)
        self.reset_states()

    def reset_states(self):
        self.actions = None
        self.step = 0
        self.solved = False

    def act(self, observations):
        if self.actions is None:
            plan = self.expert.plan(observations)
            self.solved = plan is not None
            self.actions = [] if plan is None else plan[1]
        if self.step < len(self.actions):
            actions = self.actions[self.step].tolist()
            self.step += 1
            return actions
        return [0] * len(observations)


class ActionCollectionPolicy:
    """Action-policy adapter for DDG collection."""

    # Adapted from MAPF-GPT gpt/inference.py.

    def __init__(self, checkpoint, device):
        import torch

        from model.world import GPT, GPTConfig

        if not isinstance(checkpoint.get("model_args"), dict):
            raise ValueError("Action checkpoint requires model_args")
        self.device = torch.device(device)
        self.model = GPT(GPTConfig(**checkpoint["model_args"]))
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.to(self.device).eval()
        self.reset_states()

    def reset_states(self):
        self.encoder = TransitionEncoder()
        self.previous = None

    def act(self, observations):
        import torch

        positions = np.asarray([o["global_xy"] for o in observations])
        if self.previous is None:
            self.encoder.reset(observations)
        else:
            delta = positions - self.previous
            matches = (delta[:, None, :] == MOVES[None, :, :]).all(-1)
            if not matches.any(-1).all():
                raise ValueError("Invalid executed displacement")
            self.encoder.update_actions(matches.argmax(-1))
        self.previous = positions.copy()
        tokens = torch.tensor(
            self.encoder.encode(observations), device=self.device, dtype=torch.long
        )
        with torch.no_grad():
            return self.model.act(tokens).reshape(-1).cpu().tolist()


def load_collection_policy(path, mode, device, trust_checkpoint=False):
    import torch

    from inference.run import (
        MAPFWorldInference,
        MAPFWorldInferenceConfig,
        require_action_checkpoint,
    )

    dev = torch.device(device)
    if dev.type not in ("cpu", "cuda") or (dev.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Collection requires an available CPU or CUDA device")
    checkpoint = torch.load(path, map_location="cpu", weights_only=not trust_checkpoint)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("Expected a checkpoint containing a model state dictionary")
    require_action_checkpoint(checkpoint)
    if not any(k.startswith("shared_backbone.") for k in checkpoint["model"]):
        if mode != "ddg":
            raise ValueError("World-error collection requires a complete World checkpoint")
        return ActionCollectionPolicy(checkpoint, device)
    return MAPFWorldInference(
        MAPFWorldInferenceConfig(
            world_weights=str(path),
            device=device,
            is_slow=mode == "world-error",
            trust_checkpoint=trust_checkpoint,
        )
    )


def filter_duplicate_obs(obs, next_obs, next_actions):
    """Deduplicate transitions by current observation within an episode."""
    # Adapted from MAPF-World transition_worker.py. Byte keys avoid hash collisions.
    seen, keep = set(), []
    for index, row in enumerate(obs):
        key = np.asarray(row, dtype=np.int8).tobytes()
        if key not in seen:
            seen.add(key)
            keep.append(index)
    return tuple(np.asarray(x, dtype=np.int8)[keep] for x in (obs, next_obs, next_actions))


def collect_training_data(
    mode,
    policy,
    output,
    *,
    checkpoint,
    library=None,
    seeds=32,
    map_seed=0,
    scenario_seed=0,
    num_agents=(32, 64, 96, 128),
    rows=102400,
    rows_per_shard=10000,
    max_episode_steps=256,
    ddg=None,
    world_error=None,
    seed=1337,
):
    """Collect transitions and write Arrow shards with a training manifest."""
    import torch

    if mode not in ("ddg", "world-error") or seeds < 1 or rows_per_shard < 1:
        raise ValueError("Invalid collection mode, seed count or shard size")
    if rows < (5 if mode == "ddg" else 50) or max_episode_steps < 1:
        raise ValueError("Require at least 5 DDG rows / 50 world-error rows and a positive horizon")
    if not num_agents or any(n < 1 for n in num_agents) or min(map_seed, scenario_seed, seed) < 0:
        raise ValueError("Agent counts must be positive and seeds nonnegative")
    ddg, world_error = ddg or TransitionDDGConfig(), world_error or WorldErrorDDGConfig()
    ddg.validate()
    world_error.validate()
    if mode == "ddg" and library is None:
        raise ValueError("DDG requires an explicit LaCAM library")
    metadata = dict(
        split="train",
        mode=mode,
        checkpoint=str(Path(checkpoint).resolve()),
        checkpoint_sha256=sha256(checkpoint),
        seeds=seeds,
        map_seed=map_seed,
        scenario_seed=scenario_seed,
        num_agents=list(num_agents),
        requested_rows=rows,
        max_episode_steps=max_episode_steps,
        seed=seed,
        collection=asdict(ddg if mode == "ddg" else world_error),
        prediction_decoder="sample_dream" if mode == "world-error" else None,
    )
    if library is not None:
        metadata["expert_library_sha256"] = sha256(library)
    torch.manual_seed(seed)
    np.random.seed(seed)

    def factory(limit):
        return LacamCollectionPolicy(library, limit)

    buffers = {
        kind: {"obs": [], "next_obs": [], "next_action": [], "counts": [0] * 5}
        for kind in (("ddg",) if mode == "ddg" else ("maze", "random"))
    }
    episode_count = 0
    with run_directory(output, metadata) as (output, log):
        # DDG alternates maze/random for each seed. World-error fills the maze
        # pool before the random pool, with a 9:1 quota ratio.
        episodes = (
            [(kind, i) for i in range(seeds) for kind in ("maze", "random")]
            if mode == "ddg"
            else [(kind, i) for kind in ("maze", "random") for i in range(seeds)]
        )
        with (
            (output / "episodes.jsonl").open("x") as ledger,
            tqdm(total=len(episodes), desc=f"Collect {mode}", unit="episode") as bar,
        ):
            for kind, index in episodes:
                buffer = buffers["ddg" if mode == "ddg" else kind]
                quota = rows // 5 if mode == "ddg" else rows * (9 if kind == "maze" else 1) // 50
                done = (
                    min(buffer["counts"]) >= quota
                    if mode == "ddg"
                    else len(buffer["obs"]) >= quota * 5
                )
                if done:
                    bar.update(1)
                    continue
                ms = map_seed + index
                ss = scenario_seed + index if mode == "ddg" else scenario_seed
                agents = num_agents[(index if mode == "ddg" else ms) % len(num_agents)]
                env = make_training_env(kind, agents, ms, ss, max_episode_steps)
                try:
                    data, details = (
                        collect_ddg_transitions(env, policy, ddg, factory)
                        if mode == "ddg"
                        else collect_world_error_transitions(env, policy, world_error)
                    )
                finally:
                    env.close()
                batch = (data["obs"], data["next_obs"], data["next_action"])
                if mode == "ddg" and len(batch[0]):
                    batch = filter_duplicate_obs(*batch)
                cap = quota if mode == "ddg" else quota * 3
                for before, after, action in zip(*batch):
                    action = int(action)
                    if not 0 <= action < 5:
                        raise ValueError("Collector produced an invalid action label")
                    if buffer["counts"][action] >= cap:
                        continue
                    buffer["counts"][action] += 1
                    buffer["obs"].append(before)
                    buffer["next_obs"].append(after)
                    buffer["next_action"].append(action)
                episode_count += 1
                ledger.write(
                    json.dumps(
                        dict(
                            kind=kind,
                            map_seed=ms,
                            scenario_seed=ss,
                            num_agents=agents,
                            counts=buffer["counts"],
                            **details,
                        )
                    )
                    + "\n"
                )
                ledger.flush()
                bar.update(1)
                bar.set_postfix(rows=sum(len(b["obs"]) for b in buffers.values()))
                log.info(
                    "%s seed=%d scenario=%d candidates=%d retained=%d actions=%s",
                    kind,
                    ms,
                    ss,
                    len(batch[0]),
                    len(buffer["obs"]),
                    buffer["counts"],
                )
        writer = TransitionWriter(output, rows_per_shard)
        exported_counts = [0] * 5
        remaining = rows
        for buffer in buffers.values():
            count = min(remaining, len(buffer["obs"]))
            writer.add(
                buffer["obs"][:count], buffer["next_obs"][:count], buffer["next_action"][:count]
            )
            for action in buffer["next_action"][:count]:
                exported_counts[action] += 1
            remaining -= count
        writer.flush()
        if writer.total == 0:
            raise RuntimeError(
                "No transitions selected; no train.json published. Inspect episode logs or increase seeds"
            )
        manifest = dict(
            split="train",
            files=writer.files,
            rows=writer.total,
            mode=mode,
            episodes=episode_count,
            action_counts=exported_counts,
            requested_rows=rows,
            quota_met=remaining == 0,
            coordinate_encoding="clip",
        )
        write_json(output / "train.json", manifest)
        if remaining:
            log.warning("Seed budget exhausted: exported %d/%d rows", writer.total, rows)
        log.info("Training manifest: %s (%d rows)", output / "train.json", writer.total)
        return manifest


def collect_from_config(checkpoint, output, config, device, round_index=0):
    """Collect one training round from a model snapshot."""
    options = dict(config)
    mode = options.pop("mode")
    options["map_seed"] = options.get("map_seed", 0) + round_index * options.get("seeds", 32)
    options["seed"] = options.get("seed", 1337) + round_index
    if "ddg" in options:
        options["ddg"] = TransitionDDGConfig(**options["ddg"])
    if "world_error" in options:
        options["world_error"] = WorldErrorDDGConfig(**options["world_error"])
    policy = load_collection_policy(checkpoint, mode, device)
    return collect_training_data(mode, policy, output, checkpoint=checkpoint, **options)


def collection_main(mode, argv):
    parser = argparse.ArgumentParser(description=f"Collect {mode} training transitions")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a trained action or World checkpoint",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory")
    parser.add_argument("--device", default="cuda", help="cpu or cuda[:index]")
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="Allow pickle loading only for a checkpoint you trust",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=32 if mode == "ddg" else 1000,
        help="Maximum map seeds per environment family",
    )
    parser.add_argument("--map-seed", type=int, default=0)
    parser.add_argument("--scenario-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337, help="Policy sampling RNG seed")
    parser.add_argument(
        "--num-agents", type=int, nargs="+", default=[32, 64, 96, 128] if mode == "ddg" else [32]
    )
    parser.add_argument(
        "--rows", type=int, default=102400, help="Target export rows; seed budget may produce fewer"
    )
    parser.add_argument("--rows-per-shard", type=int, default=10000)
    parser.add_argument("--max-episode-steps", type=int, default=256)
    if mode == "ddg":
        parser.add_argument("--lacam-library", type=Path, required=True)
        parser.add_argument("--steps-delta", type=int, default=16, help="Rollback probe stride")
        parser.add_argument(
            "--steps-saved", type=int, default=32, help="Expert rollout steps per selected state"
        )
        parser.add_argument(
            "--diff-threshold",
            type=int,
            default=3,
            help="Select the preceding state when maximum makespan increase exceeds this",
        )
        parser.add_argument(
            "--fast-time-limit", type=float, default=2.0, help="Seconds per rollback probe"
        )
        parser.add_argument("--expert-time-limit", type=float, default=10.0)
        parser.add_argument(
            "--collect-on-success",
            action="store_true",
            help="Also collect all probe states when no increase exceeds the threshold",
        )
    else:
        parser.add_argument(
            "--max-transitions",
            type=int,
            default=512,
            help="Highest-error agent transitions retained per episode",
        )
    args = parser.parse_args(argv)
    try:
        if args.output_dir.exists():
            raise FileExistsError("Output directory already exists")
        policy = load_collection_policy(args.checkpoint, mode, args.device, args.trust_checkpoint)
        ddg = (
            TransitionDDGConfig(
                args.steps_delta,
                args.steps_saved,
                args.diff_threshold,
                args.fast_time_limit,
                args.expert_time_limit,
                args.collect_on_success,
            )
            if mode == "ddg"
            else None
        )
        return collect_training_data(
            mode,
            policy,
            args.output_dir,
            checkpoint=args.checkpoint,
            library=getattr(args, "lacam_library", None),
            seeds=args.seeds,
            map_seed=args.map_seed,
            scenario_seed=args.scenario_seed,
            num_agents=args.num_agents,
            rows=args.rows,
            rows_per_shard=args.rows_per_shard,
            max_episode_steps=args.max_episode_steps,
            ddg=ddg,
            world_error=WorldErrorDDGConfig(
                args.max_episode_steps, getattr(args, "max_transitions", 512)
            ),
            seed=args.seed,
        )
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "build-lacam":
        build_lacam_main(argv[1:])
        return 0
    if argv and argv[0] in ("ddg", "world-error"):
        collection_main(argv[0], argv[1:])
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Training YAML declaring split: train; evaluation configs are rejected",
    )
    parser.add_argument(
        "--lacam-library",
        type=Path,
        required=True,
        help="Explicit shared library built by python -m generator.generate_dataset build-lacam",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New output directory for Arrow shards, train.json and logs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Generate an explicit episode subset; omitted means the full training grid",
    )
    parser.add_argument(
        "--expert-time-limit", type=float, default=10.0, help="LaCAM planning seconds per episode"
    )
    parser.add_argument(
        "--rows-per-shard",
        type=int,
        default=10000,
        help="Maximum buffered rows and rows per Arrow shard",
    )
    parser.add_argument("--cache-mb", type=int, default=128, help="BFS cache memory bound")
    parser.add_argument(
        "--coordinate-encoding",
        choices=("clip", "buckets"),
        default="clip",
        help="clip matches inference; buckets reproduces legacy dataset coordinate overflow",
    )
    args = parser.parse_args(argv)
    try:
        suite = ScenarioSuite(args.config, "train")
        expert = LacamExpert(args.lacam_library, args.expert_time_limit)
        generate(
            suite,
            expert,
            args.output_dir,
            args.limit,
            args.rows_per_shard,
            args.cache_mb,
            args.coordinate_encoding,
        )
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
