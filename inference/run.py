"""MAPF-World inference and episode execution."""

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from pogema_toolbox.algorithm_config import AlgoBase
from pogema_toolbox.registry import ToolboxRegistry
from pydantic import Extra
from tqdm import tqdm

from model.tokenizer import (
    Encoder,
    InputParameters,
    ObservationEncoder,
    generate_cost2go_obs,
    precompute_cost2go,
)
from model.world import FastAndSlow_B_2, FastAndSlow_S_2
from utils.scenarios import ScenarioSuite, run_directory, sha256, write_json

# Adapted from MAPF-World world/Inference.py.


LOG = logging.getLogger(__name__)


def require_action_checkpoint(checkpoint):
    """Reject known world-only training artifacts before fast/slow execution."""
    version = checkpoint.get("format_version")
    if version == 2:
        raise ValueError("This legacy world-head-only checkpoint has no supervised fast lm_head")
    if version is not None and version != 2:
        if (
            version != 3
            or checkpoint.get("objective") != "joint_action_world"
            or not {"action", "world"}.issubset(checkpoint.get("supervised_heads", []))
        ):
            raise ValueError(
                "Unsupported checkpoint or missing joint action/world supervision metadata"
            )


class WorldPolicy:
    """Episode-local tokenization and batched inference.

    `world` exposes the legacy world-token action readout. `fast` uses lm_head;
    `hybrid` uses a fast/half-dream three-step cycle. The latter
    two modes require a checkpoint with a trained action decoder.
    """

    def __init__(
        self,
        checkpoint,
        device="cpu",
        mode="fast",
        batch_size=256,
        sample=False,
        model_size="auto",
        cache_mb=128,
        trust_checkpoint=False,
        coordinate_encoding="clip",
    ):
        if mode not in ("world", "fast", "hybrid") or batch_size < 1:
            raise ValueError("Invalid policy mode or inference batch size")
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Supported inference devices are cpu and cuda[:index]")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        self.checkpoint = Path(checkpoint).resolve(strict=True)
        content = torch.load(self.checkpoint, map_location="cpu", weights_only=not trust_checkpoint)
        if not isinstance(content, dict) or not isinstance(content.get("model"), dict):
            raise ValueError("Expected a full checkpoint containing a model state dictionary")
        saved_size = content.get("config", {}).get("model_size")
        if model_size == "auto":
            if saved_size not in ("small", "base"):
                raise ValueError(
                    "Legacy checkpoint has no model_size; pass --model-size explicitly"
                )
            model_size = saved_size
        if model_size not in ("small", "base") or (saved_size and saved_size != model_size):
            raise ValueError("Checkpoint model_size does not match the requested architecture")
        if mode != "world":
            require_action_checkpoint(content)
        self.model = (FastAndSlow_S_2 if model_size == "small" else FastAndSlow_B_2)()
        self.model.load_state_dict(content["model"], strict=True)
        self.model.to(self.device).eval()
        self.mode, self.batch_size, self.sample = mode, batch_size, sample
        self.encoder = ObservationEncoder(cache_mb, coordinate_encoding)
        self.reset(0)

    def reset(self, seed=0):
        self.encoder.reset()
        self.previous_intent = None
        self.step = 0
        # fork_rng prevents a policy from consuming the caller's global CPU/CUDA RNG.
        devices = (
            [self.device.index if self.device.index is not None else torch.cuda.current_device()]
            if self.device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            self.cpu_rng = torch.get_rng_state()
            if devices:
                with torch.cuda.device(self.device):
                    torch.cuda.manual_seed(seed)
                self.cuda_rng = torch.cuda.get_rng_state(self.device)

    def act(self, observations):
        return self.act_tokens(self.encoder.encode(observations))

    @torch.inference_mode()
    def act_tokens(self, tokens):
        tokens = np.asarray(tokens)
        if tokens.ndim != 2 or tokens.shape[1] != 256 or len(tokens) == 0:
            raise ValueError("Expected a nonempty [agents, 256] token matrix")
        if not np.issubdtype(tokens.dtype, np.integer) or np.any((tokens < 0) | (tokens > 66)):
            raise ValueError("Tokens must be integers in [0, 66]")
        if self.previous_intent is not None and len(self.previous_intent) != len(tokens):
            raise ValueError("Agent count changed; reset the policy first")
        devices = (
            [self.device.index if self.device.index is not None else torch.cuda.current_device()]
            if self.device.type == "cuda"
            else []
        )
        actions, intents = [], []
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(self.cpu_rng)
            if devices:
                torch.cuda.set_rng_state(self.cuda_rng, self.device)
            for start in range(0, len(tokens), self.batch_size):
                idx = torch.as_tensor(
                    tokens[start : start + self.batch_size], dtype=torch.long, device=self.device
                )
                if self.mode == "world":
                    logits = self.model(idx)[:, 129, 45:50]
                    result = (
                        torch.multinomial(logits.softmax(-1), 1)
                        if self.sample
                        else logits.argmax(-1)
                    )
                elif self.mode == "fast" or self.step % 3 == 0:
                    result = self.model.fast_inference(idx, do_sample=self.sample)
                else:
                    previous = (
                        None
                        if self.previous_intent is None
                        else self.previous_intent[start : start + len(idx)]
                    )
                    intent, result = self.model.half_dream(idx, previous, do_sample=self.sample)
                    intents.append(intent.reshape(len(idx), 13))
                actions.extend(result.reshape(-1).cpu().tolist())
            self.cpu_rng = torch.get_rng_state()
            if devices:
                self.cuda_rng = torch.cuda.get_rng_state(self.device)
        self.previous_intent = torch.cat(intents) if intents else None
        self.step += 1
        return actions


class MAPFWorldInferenceConfig(AlgoBase, extra=Extra.forbid):
    name: Literal["MAPF-World"] = "MAPF-World"
    num_agents: int = 13
    num_previous_actions: int = 5
    cost2go_value_limit: int = 20
    agents_radius: int = 5
    cost2go_radius: int = 5
    path_to_weights: Optional[str] = None
    trust_checkpoint: bool = False
    world_weights: Optional[str] = "weights/mapf-world-3M.pt"
    model_size: Literal["auto", "small", "base"] = "auto"
    device: str = "cuda"
    context_size: int = 256
    mask_actions_history: bool = False
    mask_goal: bool = False
    mask_cost2go: bool = False
    mask_greed_action: bool = False
    batch_size: int = 512
    is_slow: bool = False
    sparse_attn: bool = False
    sparse_attn_window: int = 0
    sparse_attn_mode: Literal["sliding", "structured"] = "sliding"


class MAPFWorldInference:
    def __init__(self, cfg: MAPFWorldInferenceConfig, net=None):
        self.cfg: MAPFWorldInferenceConfig = cfg
        self.cost2go_data = None
        self.actions_history = None
        self.position_history = None
        self.env_step = 0
        self.encoder = Encoder(
            InputParameters(
                num_agents=cfg.num_agents,
                num_previous_actions=cfg.num_previous_actions,
                cost2go_value_limit=cfg.cost2go_value_limit,
                agents_radius=cfg.agents_radius,
                cost2go_radius=cfg.cost2go_radius,
                context_size=cfg.context_size,
                mask_actions_history=cfg.mask_actions_history,
                mask_cost2go=cfg.mask_cost2go,
                mask_goal=cfg.mask_goal,
                mask_greed_action=cfg.mask_greed_action,
            )
        )

        self.array_of_queues = []
        if "cuda" in self.cfg.device and not torch.cuda.is_available():
            ToolboxRegistry.warning(f"{self.cfg.device} is not available, using cpu instead!")
            self.cfg.device = "cpu"
        elif self.cfg.device == "mps" and not torch.backends.mps.is_available():
            ToolboxRegistry.warning(f"{self.cfg.device} is not available, using cpu instead!")
            self.cfg.device = "cpu"

        ckpt = torch.load(
            Path(self.cfg.world_weights),
            map_location=self.cfg.device,
            weights_only=not self.cfg.trust_checkpoint,
        )
        require_action_checkpoint(ckpt)
        state = ckpt["model"]
        saved_size = ckpt.get("config", {}).get("model_size")
        if saved_size not in ("small", "base"):
            layers = {key.split(".")[1] for key in state if key.startswith("world_branch.")}
            saved_size = {4: "small", 8: "base"}.get(len(layers))
        size = saved_size if self.cfg.model_size == "auto" else self.cfg.model_size
        if size not in ("small", "base") or (saved_size and saved_size != size):
            raise ValueError("World checkpoint architecture does not match model_size")
        factory = FastAndSlow_S_2 if size == "small" else FastAndSlow_B_2
        self.world = factory(
            sparse_attn=self.cfg.sparse_attn,
            sparse_attn_window=self.cfg.sparse_attn_window,
            sparse_attn_mode=self.cfg.sparse_attn_mode,
        ).to(self.cfg.device)
        self.world.load_state_dict(state, strict=True)
        self.world.eval()

        self.results = [0] * 10000
        self.is_first = True
        self.steps = [0] * 10000
        self.pre_agents = [None] * 10000

        self.episode_dreams = {}
        self.global_map_grid = None
        self.agent_goals = None

    def generate_input(self, observations):
        num = len(observations)
        global_xy = [obs["global_xy"] for obs in observations]
        R = self.cfg.agents_radius

        next_actions = [""] * num
        for agent_idx, obs in enumerate(observations):
            target = obs["global_target_xy"]
            cx, cy = obs["global_xy"]
            c2g_target = self.cost2go_data[target]
            cur_cost = c2g_target[cx][cy]
            bits = []
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb_cost = c2g_target[cx + dx][cy + dy]
                bits.append("1" if nb_cost >= 0 and cur_cost > nb_cost else "0")
            next_actions[agent_idx] = "".join(bits)

        spatial_index = {}
        for idx, (x, y) in enumerate(global_xy):
            key = (x, y)
            if key in spatial_index:
                spatial_index[key].append(idx)
            else:
                spatial_index[key] = [idx]

        inputs = []
        for agent_idx, obs in enumerate(observations):
            ax, ay = global_xy[agent_idx]
            c2g_src = self.cost2go_data[(ax, ay)]

            candidates = []
            for dx in range(-R, R + 1):
                for dy in range(-R, R + 1):
                    bucket = spatial_index.get((ax + dx, ay + dy))
                    if bucket:
                        for j in bucket:
                            dist = c2g_src[global_xy[j][0]][global_xy[j][1]]
                            if dist >= 0:
                                candidates.append((j, dist))

            candidates.sort(key=lambda x: (x[1], x[0]))

            agents_info = []
            for n, _ in candidates[: self.cfg.num_agents]:
                rel_x = global_xy[n][0] - ax
                rel_y = global_xy[n][1] - ay
                agents_info.append(
                    {
                        "relative_pos": (rel_x, rel_y),
                        "relative_goal": (
                            observations[n]["global_target_xy"][0] - ax,
                            observations[n]["global_target_xy"][1] - ay,
                        ),
                        "previous_actions": self.actions_history[n],
                        "next_action": next_actions[n],
                    }
                )

            inputs.append(
                {
                    "agents": agents_info,
                    "cost2go": generate_cost2go_obs(
                        self.cost2go_data[obs["global_target_xy"]],
                        obs["global_xy"],
                        self.cfg.cost2go_radius,
                        self.cfg.cost2go_value_limit,
                        self.cfg.mask_cost2go,
                    ),
                }
            )

        return inputs

    def pad_zero(self, x, target_dim=256):
        B, T, C = x.shape  # C=121
        pad_size = target_dim - C
        padding = torch.zeros(B, T, pad_size, device=x.device, dtype=x.dtype)
        return torch.cat([x, padding], dim=-1)  # [B, T, 256]

    @torch.no_grad()
    def act(self, observations, return_probs=False):
        import time as _time

        self.last_predicted_tokens = None
        num_agents = len(observations)
        moves = {(0, 0): "w", (-1, 0): "u", (1, 0): "d", (0, -1): "l", (0, 1): "r"}
        if self.cost2go_data is None:
            global_obs = observations[0]["global_obstacles"].copy().astype(int).tolist()
            self.global_map_grid = observations[0]["global_obstacles"].copy().astype(int)
            self.agent_goals = [obs["global_target_xy"] for obs in observations]

            self.cost2go_data = precompute_cost2go(global_obs, self.cfg.cost2go_radius)
            self.actions_history = [
                ["n" for _ in range(self.cfg.num_previous_actions)] for _ in range(num_agents)
            ]
            self.position_history = [[obs["global_xy"]] for obs in observations]
        else:
            for i in range(num_agents):
                self.position_history[i].append(observations[i]["global_xy"])
                self.actions_history[i].append(
                    moves[
                        (
                            self.position_history[i][-1][0] - self.position_history[i][-2][0],
                            self.position_history[i][-1][1] - self.position_history[i][-2][1],
                        )
                    ]
                )
                self.actions_history[i] = self.actions_history[i][-self.cfg.num_previous_actions :]

        _t0 = _time.perf_counter()
        inputs = self.generate_input(observations)
        _t1 = _time.perf_counter()

        self.last_encoded_tokens = [self.encoder.encode(inp) for inp in inputs]
        _t2 = _time.perf_counter()

        self._profile_generate = _t1 - _t0
        self._profile_encode = _t2 - _t1

        if num_agents > self.cfg.batch_size:
            actions = []
            all_probs = []
            for batch_idx, i in enumerate(range(0, num_agents, self.cfg.batch_size)):
                batch_inputs = torch.tensor(
                    self.last_encoded_tokens[i : i + self.cfg.batch_size],
                    dtype=torch.long,
                    device=self.cfg.device,
                )
                if self.cfg.is_slow:
                    batch_result = self.half_dream(
                        batch_inputs, batch_idx, return_probs=return_probs
                    )
                    if return_probs:
                        batch_actions, batch_probs = batch_result
                        all_probs.extend(batch_probs)
                    else:
                        batch_actions = batch_result
                else:
                    if return_probs:
                        act_result, probs = self.world.fast_inference(
                            batch_inputs, return_probs=True
                        )
                        batch_actions = act_result.reshape(-1).detach().cpu().numpy().tolist()
                        all_probs.extend(probs.detach().cpu().numpy())
                    else:
                        act_result = self.world.fast_inference(batch_inputs)
                        batch_actions = act_result.reshape(-1).detach().cpu().numpy().tolist()
                actions.extend(batch_actions)

            if return_probs:
                self.last_action_probs = np.array(all_probs)
        else:
            obs = torch.tensor(
                self.last_encoded_tokens,
                dtype=torch.long,
                device=self.cfg.device,
            )
            if self.cfg.is_slow:
                result = self.half_dream(obs, 0, return_probs=return_probs)
                if return_probs:
                    actions, probs_out = result
                    self.last_action_probs = probs_out
                else:
                    actions = result
            else:
                if return_probs:
                    act_result, probs = self.world.fast_inference(obs, return_probs=True)
                    actions = act_result.reshape(-1).detach().cpu().numpy().tolist()
                    self.last_action_probs = probs.detach().cpu().numpy()
                else:
                    actions = self.world.fast_inference(obs)
                    actions = actions.reshape(-1).detach().cpu().numpy().tolist()

        if not isinstance(actions, list):
            actions = [actions]
        return actions

    def fast_inference_with_dream(self, tensor_obs, i):
        actions = self.world.fast_inference(tensor_obs)
        actions = actions.reshape(-1).detach().cpu().numpy().tolist()
        dream = self.encoder.decode(tensor_obs[0].detach().cpu().numpy().tolist())
        from utils.visualize import visualize_decoded_obs

        visualize_decoded_obs(dream, actions, save_path="images/gt/" + f"{self.env_step}.png")
        self.env_step += 1
        return actions

    def half_dream(self, tensor_obs, i, return_probs=False):
        if self.steps[i] % 3 == 0:
            self.steps[i] = 0
            self.pre_agents[i] = None
            self.last_predicted_tokens = None
            self.last_used_slow = False
            if return_probs:
                actions, probs = self.world.fast_inference(
                    tensor_obs, do_sample=False, return_probs=True
                )
                probs_np = probs.detach().cpu().numpy()
            else:
                actions = self.world.fast_inference(tensor_obs, do_sample=False)
        else:
            self.last_used_slow = True
            if return_probs:
                self.pre_agents[i], actions, probs, predicted_tokens = self.world.half_dream(
                    tensor_obs, self.pre_agents[i], return_probs=True
                )
                probs_np = probs.detach().cpu().numpy()
                self.last_predicted_tokens = predicted_tokens.detach().cpu().numpy()
            else:
                self.pre_agents[i], actions = self.world.half_dream(tensor_obs, self.pre_agents[i])
        self.steps[i] += 1
        actions = actions.reshape(-1).detach().cpu().numpy().tolist()
        if return_probs:
            return actions, probs_np
        return actions

    def full_dream(self, tensor_obs, i):
        if self.steps[i] % 2 == 0:
            self.results[i] = self.world.dreamer(tensor_obs, max_steps=2)
            self.steps[i] = 0
        actions = self.results[i][self.steps[i]]
        self.steps[i] += 1
        return actions

    def full_dream_with_visualize(self, tensor_obs, i):

        if self.steps[i] % 3 == 0:
            self.results[i], dreams = self.world.dreamer(tensor_obs, max_steps=3, return_dream=True)
            self.steps[i] = 0

            from utils.visualize import save_dream_animation

            target_agent_idx = 0
            decoded_dreams = [
                self.encoder.decode(d[target_agent_idx].detach().cpu().numpy().tolist())
                for d in dreams
            ]

            self.episode_dreams[self.env_step] = decoded_dreams

            save_dream_animation(decoded_dreams, self.env_step, target_agent_idx)

        actions = self.results[i][self.steps[i]]
        self.steps[i] += 1
        self.env_step += 1
        return actions

    def hybrid_inference(self, tensor_obs, i):
        predicted = self.world.hybrid_inference(tensor_obs)
        actions = predicted.detach().cpu().numpy().tolist()
        return actions

    @torch.no_grad()
    def predict_next_tokens(self):
        """Decode the world prediction for collection without changing action RNG."""
        if getattr(self, "last_encoded_tokens", None) is None:
            raise RuntimeError("Call act before requesting a world prediction")
        device = torch.device(self.cfg.device)
        devices = [device.index or 0] if device.type == "cuda" else []
        predictions = []
        with torch.random.fork_rng(devices=devices):
            for start in range(0, len(self.last_encoded_tokens), self.cfg.batch_size):
                tokens = torch.tensor(
                    self.last_encoded_tokens[start : start + self.cfg.batch_size],
                    dtype=torch.long,
                    device=device,
                )
                predicted, _ = self.world.sample_dream(self.world(tokens), idx=tokens)
                predictions.extend(predicted.cpu().tolist())
        self.last_predicted_tokens = np.asarray(predictions, dtype=np.int8)
        return self.last_predicted_tokens

    def reset_states(self):
        if (
            self.position_history is not None
            and self.episode_dreams
            and self.global_map_grid is not None
        ):
            try:
                from utils.visualize import save_coordinated_animation

                save_coordinated_animation(
                    self.global_map_grid,
                    self.position_history,
                    self.episode_dreams,
                    target_agent_idx=0,
                    goals=self.agent_goals,
                    save_dir="dreams",
                )
                print("Saved coordinated animation.")
            except Exception as e:
                print(f"Failed to save coordinated animation: {e}")
                import traceback

                traceback.print_exc()

        self.cost2go_data = None
        self.actions_history = None
        self.position_history = None
        self.episode_dreams = {}
        self.global_map_grid = None
        self.agent_goals = None
        self.steps = [0] * 10000
        self.pre_agents = [None] * 10000
        self.last_action_probs = None
        self.last_predicted_tokens = None
        self.last_encoded_tokens = None
        self.last_used_slow = False


__all__ = ["MAPFWorldInference", "MAPFWorldInferenceConfig"]


def evaluate(
    suite,
    output,
    policy=None,
    expert=None,
    limit=None,
    animation=None,
    seed=0,
    save_actions=False,
    metadata=None,
):
    if (policy is None) == (expert is None):
        raise ValueError("Select exactly one policy or expert")
    count = suite.total if limit is None else min(suite.total, limit)
    if count < 1:
        raise ValueError("limit must be positive")
    metadata = {
        **suite.identity,
        **(metadata or {}),
        "selected_episodes": count,
        "subset": count < suite.total,
        "policy_seed": seed,
        "animation": suite.animation if animation is None else animation,
    }
    with run_directory(output, metadata) as (output, log):
        records = []
        log.info(
            "Benchmark: %d/%d episodes; policy=%s",
            count,
            suite.total,
            "LaCAM" if expert else policy.mode,
        )
        with (output / "episodes.jsonl").open("x") as result_file:
            for index, episode in enumerate(
                tqdm(suite.episodes(limit), total=count, desc="Benchmark", unit="episode")
            ):
                env = suite.create_env(episode, animation)
                actions_file = None
                try:
                    observations, _ = env.reset()
                    if save_actions:
                        actions_file = (output / f"actions-{index:06d}.jsonl").open("x")
                    start = time.perf_counter()
                    planning_seconds = 0.0
                    plan = None
                    if expert:
                        plan = expert.plan(observations)
                        planning_seconds = time.perf_counter() - start
                    else:
                        policy.reset(seed + index)
                    policy_seconds, steps = 0.0, 0
                    metrics = {}
                    while True:
                        tick = time.perf_counter()
                        if expert:
                            # Record expert failures in episode metrics.
                            actions = (
                                plan[1][steps].tolist()
                                if plan is not None and steps < len(plan[1])
                                else [0] * len(observations)
                            )
                        else:
                            actions = policy.act(observations)
                        policy_seconds += time.perf_counter() - tick
                        observations, _, terminated, truncated, infos = env.step(actions)
                        steps += 1
                        if actions_file:
                            actions_file.write(
                                json.dumps({"step": steps, "actions": actions}) + "\n"
                            )
                        if all(terminated) or all(truncated):
                            metrics = {
                                key: float(value)
                                for key, value in infos[0].get("metrics", {}).items()
                            }
                            break
                    if not {"ISR", "CSR"} <= metrics.keys():
                        raise RuntimeError("POGEMA did not produce terminal success metrics")
                    record = {
                        "episode": index,
                        "environment": episode,
                        "metrics": metrics,
                        "steps": steps,
                        "seconds": time.perf_counter() - start,
                        "policy_seconds": policy_seconds,
                        "planning_seconds": planning_seconds,
                        "expert_solved": plan is not None if expert else None,
                    }
                    result_file.write(json.dumps(record, allow_nan=False) + "\n")
                    result_file.flush()
                    records.append(record)
                    if metadata["animation"]:
                        env.save_animation(str(output / f"episode-{index:06d}.svg"))
                    log.info(
                        "Episode %d/%d | map=%s | agents=%d | CSR=%.3f | ISR=%.3f | steps=%d",
                        index + 1,
                        count,
                        episode["map_name"],
                        len(observations),
                        metrics["CSR"],
                        metrics["ISR"],
                        steps,
                    )
                finally:
                    if actions_file:
                        actions_file.close()
                    env.close()
        summary = {
            "episodes": len(records),
            "total_episodes": suite.total,
            "subset": count < suite.total,
            "aggregation": "unweighted mean over episodes",
            "metrics": {},
        }
        metric_names = sorted(set.intersection(*(set(record["metrics"]) for record in records)))
        for key in metric_names:
            summary["metrics"][key] = float(np.mean([record["metrics"][key] for record in records]))
        write_json(output / "summary.json", summary)
        with (output / "episodes.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["episode", "map_name", "num_agents", "seed", "seconds", *metric_names],
            )
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "episode": record["episode"],
                        "seconds": record["seconds"],
                        **{
                            key: record["environment"].get(key)
                            for key in ("map_name", "num_agents", "seed")
                        },
                        **{key: record["metrics"][key] for key in metric_names},
                    }
                )
        log.info("Completed %d episodes: %s", len(records), summary["metrics"])
        return summary


def main(argv=None):
    """Run one World-policy scenario."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy", choices=("world",), default="world")
    parser.add_argument("--model-size", choices=("auto", "small", "base"), default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--mode",
        choices=("world", "fast", "hybrid"),
        default="fast",
        help="fast uses lm_head; hybrid uses the fast/slow cycle",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-mb", type=int, default=128)
    parser.add_argument("--coordinate-encoding", choices=("clip", "buckets"), default="clip")
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="Allow pickle loading only for a trusted checkpoint",
    )
    parser.add_argument("--animation", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-actions", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    try:
        suite = ScenarioSuite(args.config, "eval")
        policy = WorldPolicy(
            args.checkpoint,
            args.device,
            args.mode,
            args.batch_size,
            args.sample,
            args.model_size,
            args.cache_mb,
            args.trust_checkpoint,
            args.coordinate_encoding,
        )
        metadata = {
            "checkpoint_sha256": sha256(args.checkpoint),
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
        evaluate(
            suite,
            args.output_dir,
            policy=policy,
            limit=1,
            animation=args.animation,
            seed=args.seed,
            save_actions=args.save_actions,
            metadata=metadata,
        )
        return 0
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
