"""Run MAPF-World benchmarks from YAML configurations."""

import argparse
import json
import os
from pathlib import Path

import yaml
from pogema import AnimationConfig, AnimationMonitor, pogema_v0
from pogema.wrappers.metrics import AgentsDensityWrapper, RuntimeMetricWrapper
from pogema_toolbox.create_env import Environment, MultiMapWrapper
from pogema_toolbox.registry import ToolboxRegistry

from inference.run import MAPFWorldInference, MAPFWorldInferenceConfig

# Adapted from MAPF-World benchmark.py and create_env.py.


def create_eval_env(config):
    env = pogema_v0(grid_config=config)
    env = AgentsDensityWrapper(env)
    env = MultiMapWrapper(env)
    if config.with_animation:
        env = AnimationMonitor(env, AnimationConfig(save_every_idx_episode=None))
    return RuntimeMetricWrapper(env)


BASE_PATH = Path("inference/configs")
PROJECT_NAME = "Benchmark"


def register_methods():
    ToolboxRegistry.register_env("Environment", create_eval_env, Environment)
    ToolboxRegistry.register_algorithm("MAPF-World", MAPFWorldInference, MAPFWorldInferenceConfig)


def load_evaluation_config(config_path, algorithms=None):
    config_path = Path(config_path).resolve(strict=True)
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict) or not config.get("algorithms"):
        raise ValueError("Evaluation YAML must contain environment and algorithms sections")
    if config.get("split", "eval") != "eval":
        raise ValueError("Training configs are not benchmark inputs")
    if algorithms:
        missing = set(algorithms) - config["algorithms"].keys()
        if missing:
            raise ValueError(f"Unknown YAML method labels: {sorted(missing)}")
        config["algorithms"] = {
            label: cfg for label, cfg in config["algorithms"].items() if label in algorithms
        }
    maps_path = config_path.parent / config.get("maps", "maps.yaml")
    maps = yaml.safe_load(maps_path.read_text())
    ToolboxRegistry._maps = {}
    ToolboxRegistry.register_maps(maps)
    from pogema_toolbox.config_variant_generator import generate_variants

    for _, environment in generate_variants(config["environment"]):
        Environment(**environment)
        if environment.get("map_name") not in maps:
            raise ValueError(f"Missing map: {environment.get('map_name')}")
    for label, method in config["algorithms"].items():
        if method["name"] != "MAPF-World":
            raise ValueError(f"Only MAPF-World is supported, got {method['name']!r} in {label!r}")
        if method["name"] not in ToolboxRegistry.get_state()["algorithms"]:
            raise ValueError(f"Unregistered method {method['name']!r} in {label!r}")
        ToolboxRegistry.create_algorithm_config(method["name"], **method)
    return config


def ensure_weights(eval_config):
    """Fail before evaluation if a selected local checkpoint is missing."""
    for method in eval_config["algorithms"].values():
        cfg = ToolboxRegistry.create_algorithm_config(method["name"], **method)
        for key in ("world_weights", "path_to_weights", "model_path", "lacam_lib_path"):
            value = getattr(cfg, key, None)
            if value and (key != "path_to_weights" or not getattr(cfg, "world_weights", None)):
                if not Path(value).is_file():
                    raise FileNotFoundError(f"{method['name']}: missing {key}: {value}")


def run_benchmark(config_path, output_dir, algorithms=None):
    from pogema_toolbox.evaluator import evaluation

    register_methods()
    config = load_evaluation_config(config_path, algorithms)
    # Retain invocation-relative weight paths while confining renders to output.
    for method in config["algorithms"].values():
        cfg = ToolboxRegistry.create_algorithm_config(method["name"], **method)
        for key in (
            "world_weights",
            "path_to_weights",
            "model_path",
            "lacam_lib_path",
            "planner_dir",
        ):
            value = getattr(cfg, key, None)
            if value:
                method[key] = str(Path(value).resolve())
    ensure_weights(config)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    previous = Path.cwd()
    state = {"status": "running", "config": str(Path(config_path).resolve())}
    log_sink = None
    try:
        os.chdir(output)
        ToolboxRegistry.setup_logger()
        # Workers recreate stderr locally; stream objects cannot be pickled.
        ToolboxRegistry._logger_config["sink"] = None
        log_sink = ToolboxRegistry._logger.add(str(output / "benchmark.log"), level="INFO")
        results = evaluation(config, eval_dir=output)
        state.update(status="completed", results=len(results))
        return results
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        os.chdir(previous)
        if log_sink is not None:
            try:
                ToolboxRegistry._logger.remove(log_sink)
            except ValueError:
                pass  # The Toolbox may reconfigure logging in a backend.
        (output / "run.json").write_text(json.dumps(state, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BASE_PATH / "06-europe/06-europe.yaml")
    parser.add_argument("--output-dir", type=Path, help="New local result directory")
    parser.add_argument(
        "--algorithm", action="append", help="Optional YAML method label; repeat to select several"
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate without loading weights or running episodes",
    )
    args = parser.parse_args(argv)
    try:
        if args.check_config:
            register_methods()
            config = load_evaluation_config(args.config, args.algorithm)
            print(
                json.dumps(
                    {
                        "algorithms": config["algorithms"],
                        "results_views": list(config.get("results_views", {})),
                    },
                    indent=2,
                )
            )
        else:
            output = args.output_dir or Path("outputs/benchmark") / args.config.stem
            run_benchmark(args.config, output, args.algorithm)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
