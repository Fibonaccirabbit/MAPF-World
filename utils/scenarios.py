"""Shared scenario loading and run outputs for generation and inference."""

import hashlib
import importlib.metadata
import itertools
import json
import logging
import math
import platform
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml
from gymnasium import Wrapper

# Scenario grids adapted from MAPF-GPT / POGEMA configs.


LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_map(value):
    if isinstance(value, str):
        rows = value.strip().splitlines()
        if not rows or len({len(row) for row in rows}) != 1:
            raise ValueError("Map rows must be nonempty and rectangular")
        if set("".join(rows)) - set(".@#TOSGW"):
            raise ValueError("Unsupported map symbol")
        result = np.array([[int(c not in ".GS") for c in row] for row in rows])
    else:
        result = np.asarray(value)
    if result.ndim != 2 or not result.size or not np.isin(result, [0, 1]).all():
        raise ValueError("Expected a nonempty binary obstacle grid")
    return result.astype(np.int8).tolist()


# Replay wrapper adapted from MAPF-World finetuning/wrappers.py.
class UnrollWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._unroll_steps = 0
        self._recorded_actions = []
        self._recording_episode = None

    def step(self, actions):
        if self._recording_episode:
            self._recorded_actions.append(list(actions))
        return self.env.step(actions)

    def get_actions_at_step(self, step):
        if step < 0:
            return [-1 for _ in range(self.env.grid_config.num_agents)]
        if step < len(self._recorded_actions):
            return self._recorded_actions[step]
        return [-1 for _ in range(self.env.grid_config.num_agents)]

    def set_unroll_steps(self, num_steps):
        self._unroll_steps = max(0, int(num_steps))

    def reset(self, seed=None, **kwargs):
        self._recording_episode = True if self._recording_episode is None else False
        if seed is None:
            seed = self.env.grid_config.seed
        observations, infos = self.env.reset(seed=seed, **kwargs)

        if self._unroll_steps and self._recorded_actions:
            for idx in range(min(self._unroll_steps, len(self._recorded_actions))):
                observations, _, terminated, truncated, infos = self.env.step(
                    self._recorded_actions[idx]
                )
                if all(terminated) or all(truncated):
                    break
        return observations, infos


def make_training_env(kind, num_agents, map_seed, scenario_seed, max_episode_steps=256):
    """Seeded maze and random-map training scenarios."""
    # Adapted from MAPF-World finetuning/scenario_generators.py (MAPF-GPT Toolbox).
    from pogema import pogema_v0
    from pogema_toolbox.create_env import Environment
    from pogema_toolbox.generators.maze_generator import MazeGenerator, MazeRangeSettings
    from pogema_toolbox.generators.random_generator import MapRangeSettings, generate_map

    if kind == "maze":
        settings = MazeRangeSettings(
            width_min=17,
            width_max=21,
            height_min=17,
            height_max=21,
            wall_components_min=4,
            wall_components_max=8,
        )
        grid = MazeGenerator.generate_maze(**settings.sample(seed=map_seed))
    elif kind == "random":
        settings = MapRangeSettings(width_min=17, width_max=21, height_min=17, height_max=21)
        grid = generate_map(settings.sample(map_seed))
    else:
        raise ValueError("Training environment must be maze or random")
    cfg = Environment(
        num_agents=num_agents,
        observation_type="MAPF",
        max_episode_steps=max_episode_steps,
        map=grid,
        with_animation=False,
        on_target="nothing",
        seed=scenario_seed,
        collision_system="soft",
    )
    cfg.map_name = f"{kind}-seed-{map_seed}-scenario-{scenario_seed}"
    return pogema_v0(cfg)


class ScenarioSuite:
    """Expand only explicit grid_search fields; map files are config-relative."""

    def __init__(self, filename, split):
        self.path = Path(filename).resolve()
        content = yaml.load(self.path.read_text(), Loader=LOADER)
        declared_split = content.get("split", "eval") if isinstance(content, dict) else None
        if declared_split != split:
            raise ValueError(f"Scenario config must declare split={split!r}")
        if set(content) - {"split", "maps", "environment", "algorithms", "results_views"}:
            raise ValueError("Unknown scenario configuration fields")
        if split == "train" and any(
            part in {"eval_configs", "eval_cities_configs", "test_configs", "validation_configs"}
            for part in self.path.parts
        ):
            raise ValueError("Evaluation configs cannot be used for training generation")
        self.map_path = (self.path.parent / content.get("maps", "maps.yaml")).resolve()
        self.maps = yaml.load(self.map_path.read_text(), Loader=LOADER)
        if not isinstance(self.maps, dict) or not self.maps:
            raise ValueError("Map collection must be a nonempty name-to-grid mapping")
        environment = dict(content["environment"])
        if environment.pop("name", "Environment") != "Environment":
            raise ValueError("Only the POGEMA Environment is supported")
        self.animation = environment.pop("with_animation", False)
        if type(self.animation) is not bool:
            raise ValueError("with_animation must be boolean")
        from pogema import GridConfig

        if set(environment) - set(GridConfig.__fields__):
            raise ValueError(
                f"Unknown environment fields: {set(environment) - set(GridConfig.__fields__)}"
            )
        self.keys, self.values = [], []
        for key, value in environment.items():
            if isinstance(value, dict):
                if set(value) != {"grid_search"} or not isinstance(value["grid_search"], list):
                    raise ValueError(f"Invalid grid_search for {key}")
                choices = value["grid_search"]
            else:
                choices = [value]
            if not choices:
                raise ValueError(f"Empty grid_search for {key}")
            self.keys.append(key)
            self.values.append(choices)
        if "map_name" not in self.keys:
            raise ValueError("An explicit map_name is required")
        names = self.values[self.keys.index("map_name")]
        missing = set(names) - self.maps.keys()
        if missing:
            raise ValueError(f"Missing maps: {sorted(missing)[:10]}")
        self.total = math.prod(map(len, self.values))
        self.identity = {
            "config": str(self.path),
            "config_sha256": sha256(self.path),
            "maps_sha256": sha256(self.map_path),
            "split": split,
            "total_episodes": self.total,
        }

    def episodes(self, limit=None):
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        count = self.total if limit is None else min(limit, self.total)
        for values in itertools.islice(itertools.product(*self.values), count):
            yield dict(zip(self.keys, values))

    def create_env(self, episode, animation=None):
        from pogema import GridConfig, pogema_v0

        cfg = dict(episode)
        cfg["map"] = read_map(self.maps[cfg["map_name"]])
        cfg.setdefault("obs_radius", 5)
        if cfg.get("observation_type", "MAPF") != "MAPF" or cfg["obs_radius"] != 5:
            raise ValueError("The published tokenizer requires MAPF observations and obs_radius=5")
        cfg["observation_type"] = "MAPF"
        if cfg.get("on_target", "nothing") != "nothing":
            raise ValueError("These pipelines require on_target=nothing (joint completion)")
        cfg["on_target"] = "nothing"
        env = pogema_v0(GridConfig(**cfg))
        if self.animation if animation is None else animation:
            from pogema import AnimationConfig, AnimationMonitor

            env = AnimationMonitor(env, AnimationConfig(save_every_idx_episode=None))
        return env


def write_json(path, content):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def run_directory(output, metadata):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger(f"mapf_world.run.{output}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = [logging.FileHandler(output / "run.log"), logging.StreamHandler()]
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    versions = {}
    for name in ("numpy", "pogema", "PyYAML", "torch", "pyarrow", "mapf-world"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    metadata = {
        **metadata,
        "python": platform.python_version(),
        "versions": versions,
        "status": "running",
    }
    try:
        write_json(output / "run.json", metadata)
        yield output, logger
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "run.json", metadata)
        logger.exception("Run failed; partial files are not a completed result")
        raise
    else:
        metadata["status"] = "completed"
        write_json(output / "run.json", metadata)
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
