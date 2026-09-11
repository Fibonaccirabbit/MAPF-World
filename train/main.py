"""MAPF-World training and distributed runtime."""

import argparse
import hashlib
import json
import logging
import math
import os
import random
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from train.dataset import ArrowBatchStream, Manifest, MixedBatchStream, validate_disjoint


@dataclass
class TrainConfig:
    train_manifest: str = ""
    validation_manifest: str = ""
    output_dir: str = "runs/world"
    resume: str = ""
    init_checkpoint: str = ""
    ddg_manifest: str = ""
    ddg_collection_config: str = ""
    ddg_interval: int = 500
    ddg_ratio: float = 0.25
    model_size: str = "small"
    batch_size: int = 256
    gradient_accumulation_steps: int = 16
    max_steps: int = 30000
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    weight_decay_mode: str = "legacy"
    loss_normalization: str = "microbatch"
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_steps: int = 2000
    lr_decay_steps: int = 30000
    min_lr: float = 6e-5
    costmap_weight: float = 1.0
    pos_weight: float = 2.0
    goal_weight: float = 1.0
    hist_weight: float = 0.1
    greedy_weight: float = 3.0
    action_weight: float = 1.0
    world_weight: float = 0.5
    validation_interval: int = 500
    validation_batches: int = 40
    checkpoint_interval: int = 500
    log_interval: int = 1
    seed: int = 1337
    device: str = "cuda"
    precision: str = "bfloat16"
    compile: bool = True
    trust_checkpoint: bool = False

    def validate(self, world_size=1):
        if type(world_size) is not int or world_size < 1:
            raise ValueError("WORLD_SIZE must be a positive integer")
        for name in (
            "batch_size",
            "gradient_accumulation_steps",
            "max_steps",
            "lr_decay_steps",
            "validation_interval",
            "validation_batches",
            "checkpoint_interval",
            "log_interval",
            "ddg_interval",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.model_size not in ("small", "base"):
            raise ValueError("model_size must be small or base")
        if self.weight_decay_mode not in ("legacy", "matrix"):
            raise ValueError("weight_decay_mode must be legacy or matrix")
        if self.loss_normalization not in ("microbatch", "global_tokens"):
            raise ValueError("loss_normalization must be microbatch or global_tokens")
        if self.precision not in ("float32", "bfloat16", "float16"):
            raise ValueError("precision must be float32, bfloat16 or float16")
        if self.gradient_accumulation_steps % world_size:
            raise ValueError("gradient_accumulation_steps must be divisible by WORLD_SIZE")
        if not 0 <= self.warmup_steps < self.lr_decay_steps:
            raise ValueError("Require 0 <= warmup_steps < lr_decay_steps")
        if not 0 <= self.min_lr <= self.learning_rate or self.learning_rate <= 0:
            raise ValueError("Require 0 <= min_lr <= learning_rate and learning_rate > 0")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("Adam betas must be in [0, 1)")
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
            if field.name.endswith("_weight") or field.name in ("grad_clip", "weight_decay"):
                if value < 0:
                    raise ValueError(f"{field.name} cannot be negative")
        if not self.train_manifest:
            raise ValueError("An explicit train_manifest is required")
        if self.action_weight <= 0 or self.world_weight <= 0:
            raise ValueError("Joint training requires positive action_weight and world_weight")
        if self.resume and not Path(self.resume).is_file():
            raise ValueError("resume checkpoint does not exist")
        if self.init_checkpoint and self.resume:
            raise ValueError("Use init_checkpoint for a new stage or resume for exact continuation")
        if self.init_checkpoint and not Path(self.init_checkpoint).is_file():
            raise ValueError("init_checkpoint does not exist")
        if not 0 < self.ddg_ratio < 1:
            raise ValueError("ddg_ratio must be between zero and one")
        if self.ddg_manifest or self.ddg_collection_config:
            if not self.resume and not self.init_checkpoint:
                raise ValueError("DDG training requires init_checkpoint or resume")
            if not 1 <= int(self.batch_size * self.ddg_ratio) < self.batch_size:
                raise ValueError("batch_size * ddg_ratio must allocate rows to both sources")
        if self.ddg_collection_config:
            read_collection_config(self.ddg_collection_config)
        if (
            self.validation_manifest
            and Path(self.validation_manifest).resolve() == Path(self.train_manifest).resolve()
        ):
            raise ValueError("Training and validation manifests must be different")
        return self

    def to_dict(self):
        return asdict(self)


def load_config(path=None, overrides=None):
    content = json.loads(Path(path).read_text()) if path else {}
    if not isinstance(content, dict):
        raise ValueError("Configuration must be a JSON object")
    content.update({k: v for k, v in (overrides or {}).items() if v is not None})
    if overrides and overrides.get("resume") and overrides.get("init_checkpoint") is None:
        content["init_checkpoint"] = ""
    known = {field.name for field in fields(TrainConfig)}
    if set(content) - known:
        raise ValueError(f"Unknown configuration keys: {sorted(set(content) - known)}")
    defaults = TrainConfig()
    for key, value in content.items():
        expected = type(getattr(defaults, key))
        if expected is float and type(value) in (int, float):
            content[key] = float(value)
        elif type(value) is not expected:
            raise ValueError(f"{key} must have type {expected.__name__}")
    return TrainConfig(**content)


def learning_rate_at(step, config):
    if not config.decay_lr:
        return config.learning_rate
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    if step >= config.lr_decay_steps:
        return config.min_lr
    ratio = (step - config.warmup_steps) / (config.lr_decay_steps - config.warmup_steps)
    return config.min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (
        config.learning_rate - config.min_lr
    )


# Regional losses adapted from MAPF-World train_world.py.


COSTMAP_POS = list(range(121))
POS_POS = [121 + i * 10 + j for i in range(13) for j in range(2)]
GOAL_POS = [121 + i * 10 + j for i in range(13) for j in range(2, 4)]
HIST_POS = [121 + i * 10 + j for i in range(13) for j in range(4, 9)]
GREEDY_POS = [121 + i * 10 + 9 for i in range(13)]
REGIONS = ("costmap", "pos", "goal", "hist", "greedy", "action")


def region_targets(obs, target_obs, target_action):
    """Select targets for loss and accuracy calculation."""
    specs = (
        (COSTMAP_POS, 0, 44),
        (POS_POS, 0, 5),
        (GOAL_POS, 0, 44),
        (HIST_POS, 44, 50),
        (GREEDY_POS, 50, 66),
        ([-1], 0, 5),
    )
    for name, (positions, start, end) in zip(REGIONS, specs):
        if name == "action":
            targets = target_action.reshape(-1, 1)
        elif name == "pos":
            current = obs[:, positions]
            future = target_obs[:, positions]
            targets = future.long() - current.long() + 2
            invalid = (current == 66) | (future == 66) | (targets < 0) | (targets > 4)
            targets = targets.masked_fill(invalid, -100)
        else:
            future = target_obs[:, positions]
            targets = future.long() - start
            invalid = (future == 66) | (targets < 0) | (targets >= end - start)
            targets = targets.masked_fill(invalid, -100)
        yield positions, start, end, targets.reshape(-1)


def valid_counts(obs, target_obs, target_action):
    return torch.stack(
        [(targets != -100).sum() for *_, targets in region_targets(obs, target_obs, target_action)]
    )


def loss_statistics(logits, obs, target_obs, target_action, *, accuracy=False):
    """FP32 CE sums and integer counts; an empty region has zero loss/gradient."""
    world_logits, action_logits = split_joint_logits(logits)
    sums, counts, correct = [], [], []
    for name, (positions, start, end, targets) in zip(
        REGIONS, region_targets(obs, target_obs, target_action)
    ):
        source = action_logits if name == "action" else world_logits
        scores = source[:, positions, start:end].reshape(-1, end - start).float()
        valid = targets != -100
        sums.append(F.cross_entropy(scores, targets, ignore_index=-100, reduction="sum"))
        counts.append(valid.sum())
        if accuracy:
            correct.append(((scores.argmax(dim=-1) == targets) & valid).sum())
    return torch.stack(sums), torch.stack(counts), torch.stack(correct) if accuracy else None


def normalized_loss(sums, denominators, config, scale=1.0):
    means = sums / denominators.clamp_min(1) * scale
    components = dict(zip(REGIONS, means.unbind()))
    return combine_losses(components, config)


def split_joint_logits(logits):
    if not isinstance(logits, tuple) or len(logits) != 2:
        raise ValueError(
            "Joint loss requires (world_logits, action_logits); call model(..., joint=True)"
        )
    return logits


def combine_losses(components, config):
    """Compute the weighted action and world losses."""
    world_loss = sum(
        getattr(config, f"{name}_weight") * components[name] for name in REGIONS if name != "action"
    )
    total = config.action_weight * components["action"] + config.world_weight * world_loss
    return total, {**components, "world": world_loss}


def safe_cross_entropy(logits, targets):
    """An entirely ignored region contributes differentiable zero, not NaN."""
    valid = targets != -100
    if not valid.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], targets[valid])


def region_cross_entropy(logits, targets, positions, vocab_start, vocab_end):
    region_logits = logits[:, positions, vocab_start:vocab_end]
    region_targets = targets[:, positions]
    local_targets = region_targets - vocab_start
    invalid = (
        (region_targets == 66) | (local_targets < 0) | (local_targets >= vocab_end - vocab_start)
    )
    local_targets = local_targets.masked_fill(invalid, -100)
    return safe_cross_entropy(
        region_logits.reshape(-1, vocab_end - vocab_start), local_targets.reshape(-1)
    )


def position_delta_ce(logits, obs, target_obs, positions=POS_POS):
    delta_class = target_obs[:, positions].long() - obs[:, positions].long() + 2
    invalid = (
        (target_obs[:, positions] == 66)
        | (obs[:, positions] == 66)
        | (delta_class < 0)
        | (delta_class > 4)
    )
    return safe_cross_entropy(
        logits[:, positions, :5].reshape(-1, 5), delta_class.masked_fill(invalid, -100).reshape(-1)
    )


def compute_loss(logits, obs, target_obs, target_action, config, *, denominators=None, scale=1.0):
    if denominators is not None:
        sums, _, _ = loss_statistics(logits, obs, target_obs, target_action)
        return normalized_loss(sums, denominators, config, scale)
    logits, action_logits = split_joint_logits(logits)
    components = {
        "costmap": region_cross_entropy(logits, target_obs, COSTMAP_POS, 0, 44),
        "pos": position_delta_ce(logits, obs, target_obs),
        "goal": region_cross_entropy(logits, target_obs, GOAL_POS, 0, 44),
        "hist": region_cross_entropy(logits, target_obs, HIST_POS, 44, 50),
        "greedy": region_cross_entropy(logits, target_obs, GREEDY_POS, 50, 66),
        "action": F.cross_entropy(action_logits[:, -1, :5], target_action.reshape(-1)),
    }
    return combine_losses(components, config)


def build_optimizer(model, config):
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if config.weight_decay_mode == "legacy":
        groups = [{"params": [p for _, p in named], "weight_decay": config.weight_decay}]
    else:
        # Exclude embedding tables and scalar/vector parameters (biases and norms).
        embeddings = {
            id(p)
            for module in model.modules()
            if isinstance(module, torch.nn.Embedding)
            for p in module.parameters(recurse=False)
        }
        decay, no_decay = [], []
        for name, p in named:
            destination = decay if p.ndim >= 2 and id(p) not in embeddings else no_decay
            destination.append(p)
        groups = [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        groups = [group for group in groups if group["params"]]
    return torch.optim.AdamW(groups, lr=config.learning_rate, betas=(config.beta1, config.beta2))


def load_checkpoint(path, trust_pickle=False):
    # Pickle loading requires trust=True.
    return torch.load(path, map_location="cpu", weights_only=not trust_pickle)


def build_model(config):
    from model.world import FastAndSlow_B_2, FastAndSlow_S_2

    factory = FastAndSlow_S_2 if config.model_size == "small" else FastAndSlow_B_2
    return factory()


def atomic_save(content, destination):
    destination = Path(destination)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(content, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def rng_state(device):
    numpy_state = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "cuda": [torch.cuda.get_rng_state(device)] if device.type == "cuda" else [],
    }


def restore_rng(state, device):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    array = state["numpy"]
    np.random.set_state((array[0], np.asarray(array[1], dtype=np.uint32), *array[2:]))
    if state["cuda"] and device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"][0], device)


def autocast(config, device):
    if config.precision == "float32":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=getattr(torch, config.precision))


@torch.no_grad()
def evaluate(model, manifest, config, device):
    """Validate on rank zero using the unwrapped model."""
    stream = ArrowBatchStream(manifest, config.batch_size, config.seed, shuffle=False)
    was_training = model.training
    totals, count = {}, 0
    token_sums = torch.zeros(6, dtype=torch.float64, device=device)
    token_counts = torch.zeros(6, dtype=torch.int64, device=device)
    token_correct = torch.zeros_like(token_counts)
    action_confusion = torch.zeros((5, 5), dtype=torch.int64, device=device)
    model.eval()
    try:
        for _ in range(config.validation_batches):
            size = min(config.batch_size, manifest.total_rows - count)
            if size <= 0:
                break
            obs, target, action = stream.next_batch(device, size=size)
            with autocast(config, device):
                logits = model(obs, joint=True)
                loss, components = compute_loss(logits, obs, target, action, config)
                numerators, counts, correct = loss_statistics(
                    logits, obs, target, action, accuracy=True
                )
            token_sums += numerators.double()
            token_counts += counts
            token_correct += correct
            predictions = logits[1][:, -1, :5].argmax(dim=-1)
            action_confusion += torch.bincount(
                action.reshape(-1) * 5 + predictions, minlength=25
            ).reshape(5, 5)
            for key, value in {"total": loss, **components}.items():
                totals[key] = totals.get(key, 0.0) + float(value) * size
            count += size
        result = {key: value / count for key, value in totals.items()}
        token_loss, token_components = normalized_loss(token_sums, token_counts, config)
        if config.loss_normalization == "global_tokens":
            result = {
                "total": float(token_loss),
                **{k: float(v) for k, v in token_components.items()},
            }
        result["token_total"] = float(token_loss)
        result["samples"] = count
        support = action_confusion.sum(dim=1)
        predicted = action_confusion.sum(dim=0)
        recall = action_confusion.diag().double() / support.clamp_min(1)
        result["action_balanced_accuracy"] = float(recall.sum() / (support > 0).sum().clamp_min(1))
        result["action_nonwait_accuracy"] = float(
            action_confusion.diag()[1:].sum() / support[1:].sum().clamp_min(1)
        )
        for action_id in range(5):
            result[f"action_{action_id}_support"] = int(support[action_id])
            result[f"action_{action_id}_predicted"] = int(predicted[action_id])
            result[f"action_{action_id}_recall"] = float(recall[action_id])
        for index, name in enumerate(REGIONS):
            result[f"{name}_valid_tokens"] = int(token_counts[index])
            result[f"{name}_accuracy"] = float(
                token_correct[index] / token_counts[index].clamp_min(1)
            )
        if not all(np.isfinite(value) for value in result.values()):
            raise FloatingPointError("Non-finite validation loss")
        return result
    finally:
        model.train(was_training)
        stream.close()


def read_collection_config(filename):
    from generator.generate_dataset import TransitionDDGConfig, WorldErrorDDGConfig

    content = json.loads(Path(filename).read_text())
    allowed = {
        "mode",
        "library",
        "seeds",
        "map_seed",
        "scenario_seed",
        "num_agents",
        "rows",
        "rows_per_shard",
        "max_episode_steps",
        "ddg",
        "world_error",
        "seed",
    }
    if not isinstance(content, dict) or set(content) - allowed:
        raise ValueError("Invalid DDG collection configuration fields")
    if content.get("mode") not in ("ddg", "world-error"):
        raise ValueError("Collection mode must be ddg or world-error")
    for key, default in {
        "seeds": 32,
        "rows": 102400,
        "rows_per_shard": 10000,
        "max_episode_steps": 256,
    }.items():
        value = content.get(key, default)
        if type(value) is not int or value < 1:
            raise ValueError(f"Collection {key} must be a positive integer")
    for key in ("map_seed", "scenario_seed", "seed"):
        if type(content.get(key, 0)) is not int or content.get(key, 0) < 0:
            raise ValueError(f"Collection {key} must be a nonnegative integer")
    agents = content.get("num_agents", [32, 64, 96, 128])
    if (
        not isinstance(agents, list)
        or not agents
        or any(type(n) is not int or n < 1 for n in agents)
    ):
        raise ValueError("Collection num_agents must be a nonempty list of positive integers")
    minimum = 5 if content["mode"] == "ddg" else 50
    if content.get("rows", 102400) < minimum:
        raise ValueError(f"Collection requires at least {minimum} requested rows")
    if content["mode"] == "ddg" and not Path(content.get("library", "")).is_file():
        raise ValueError("DDG collection requires an existing LaCAM library")
    try:
        TransitionDDGConfig(**content.get("ddg", {})).validate()
        WorldErrorDDGConfig(**content.get("world_error", {})).validate()
    except TypeError as exc:
        raise ValueError(f"Invalid collector options: {exc}") from exc
    return content


def collection_signature(config):
    if not config.ddg_collection_config:
        return ""
    content = read_collection_config(config.ddg_collection_config)
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


# Round scheduling and offline mixing follow MAPF-World train_ddg_world.py.
def refresh_ddg(model, stream, config, output, step, device, rank, distributed, collection_config):
    from generator.generate_dataset import collect_from_config
    from utils.scenarios import write_json

    outcome = [None, None]
    if rank == 0:
        logging.getLogger("mapf_world.train").info(
            "DDG collection starting: step=%d mode=%s round=%d existing_rows=%d",
            step,
            collection_config["mode"],
            step // config.ddg_interval,
            stream.ddg.manifest.total_rows if stream.ddg else 0,
        )
        saved_rng = rng_state(device)
        try:
            root = output / "ddg"
            root.mkdir(exist_ok=True)
            index = root / f"step-{step:08d}.json"
            signature = hashlib.sha256(
                json.dumps(collection_config, sort_keys=True).encode()
            ).hexdigest()
            if index.exists():
                metadata = json.loads(index.read_text())
                if metadata["collection_signature"] != signature:
                    raise ValueError("Cached DDG round was collected with different settings")
                snapshot = load_checkpoint(metadata["source_checkpoint"])["model"]
                live = model.state_dict()
                if snapshot.keys() != live.keys():
                    raise ValueError("Cached DDG snapshot does not match this model")
                for key, value in snapshot.items():
                    torch.testing.assert_close(
                        value, live[key].cpu(), rtol=0, atol=0, equal_nan=True
                    )
            else:
                directory = Path(tempfile.mkdtemp(prefix=f"step-{step:08d}-", dir=root))
                snapshot_path = directory / "model.pt"
                atomic_save(
                    {
                        "format_version": 3,
                        "model_layout": "shared_backbone_v1",
                        "objective": "joint_action_world",
                        "supervised_heads": ["action", "world"],
                        "model": model.state_dict(),
                        "config": {"model_size": config.model_size},
                    },
                    snapshot_path,
                )
                collection = directory / "data"
                collect_from_config(
                    snapshot_path,
                    collection,
                    collection_config,
                    str(device),
                    step // config.ddg_interval,
                )
                generated = Manifest(collection / "train.json", "train")
                files = (stream.ddg.manifest.files if stream.ddg else []) + generated.files
                metadata = {
                    "split": "train",
                    "files": files,
                    "source_step": step,
                    "source_checkpoint": str(snapshot_path),
                    "collection_signature": signature,
                }
                write_json(index, metadata)
            outcome[0] = str(index)
        except Exception as exc:
            outcome[1] = f"{type(exc).__name__}: {exc}"
        finally:
            restore_rng(saved_rng, device)
    if distributed:
        dist.broadcast_object_list(outcome, src=0)
    if outcome[1]:
        raise RuntimeError(f"DDG collection failed: {outcome[1]}")
    stream.set_ddg(outcome[0])
    stream.last_refresh = step
    if rank == 0:
        logging.getLogger("mapf_world.train").info(
            "DDG refreshed at step=%d: rows=%d, samples_per_batch=%d/%d, manifest=%s",
            step,
            stream.ddg.manifest.total_rows,
            stream.ddg_batch_size,
            config.batch_size,
            outcome[0],
        )


def check_resume(config, checkpoint, world_size):
    if checkpoint.get("format_version") != 3 or checkpoint.get("objective") != "joint_action_world":
        raise ValueError(
            "Exact resume requires a format_version=3 joint-action/world checkpoint; "
            "older world-head-only optimizer states are not equivalent"
        )
    if checkpoint["world_size"] != world_size:
        raise ValueError("Exact resume requires the same WORLD_SIZE")
    if checkpoint.get("collection_signature", "") != collection_signature(config):
        raise ValueError("DDG collection settings changed during resume")
    allowed = {
        "resume",
        "output_dir",
        "max_steps",
        "log_interval",
        "checkpoint_interval",
        "validation_interval",
        "validation_batches",
        "trust_checkpoint",
        "init_checkpoint",
    }
    for key, value in config.to_dict().items():
        if (
            key not in allowed
            and checkpoint["config"].get(key, getattr(TrainConfig(), key)) != value
        ):
            raise ValueError(f"Resume configuration mismatch: {key}")
    if config.max_steps < checkpoint["step"]:
        raise ValueError("max_steps is less than the number of completed checkpoint steps")


def train(config):
    collection_config = (
        read_collection_config(config.ddg_collection_config)
        if config.ddg_collection_config
        else None
    )
    collection_hash = (
        hashlib.sha256(json.dumps(collection_config, sort_keys=True).encode()).hexdigest()
        if collection_config
        else ""
    )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    config.validate(world_size)
    device = torch.device(config.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA requested but unavailable; explicitly use --device cpu --precision float32"
            )
        device = torch.device(f"cuda:{local_rank}" if world_size > 1 else config.device)
        torch.cuda.set_device(device)
        if config.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 unsupported on this GPU; use --precision float16")
    elif device.type != "cpu" or config.precision == "float16":
        raise ValueError("Supported devices: CUDA, or CPU with float32/bfloat16")
    train_manifest = Manifest(config.train_manifest, "train")
    validation = (
        Manifest(config.validation_manifest, "validation") if config.validation_manifest else None
    )
    validate_disjoint(train_manifest, validation)
    stream = ArrowBatchStream(train_manifest, config.batch_size, config.seed, rank, world_size)
    if config.ddg_manifest or config.ddg_collection_config:
        stream = MixedBatchStream(stream, config.ddg_ratio, validation, rank, world_size)
    distributed = world_size > 1
    output = Path(config.output_dir).resolve()
    lock = None
    handler = None
    logger = logging.getLogger("mapf_world.train")
    logger.setLevel(logging.INFO)
    if distributed:
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo",
            timeout=timedelta(hours=2) if config.ddg_collection_config else timedelta(minutes=30),
        )
    try:
        error = [None]
        if rank == 0:
            try:
                output.mkdir(parents=True, exist_ok=True)
                if not config.resume and any(output.iterdir()):
                    raise ValueError(
                        "New training requires an empty output directory; use --resume for an existing run"
                    )
                lock = output / ".training.lock"
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(fd)
            except (OSError, ValueError) as exc:
                lock = None
                error[0] = str(exc)
        if distributed:
            dist.broadcast_object_list(error, src=0)
        if error[0]:
            raise ValueError(error[0])
        if rank == 0:
            handler = logging.FileHandler(output / "train.log", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(handler)
            with (output / ("resume_config.json" if config.resume else "config.json")).open(
                "w"
            ) as file:
                json.dump(config.to_dict(), file, indent=2)
            logger.info(
                "Source rows=%d files=%d; effective batch=%d; world_size=%d",
                train_manifest.total_rows,
                len(train_manifest.files),
                config.batch_size * config.gradient_accumulation_steps,
                world_size,
            )
        random.seed(config.seed + rank)
        np.random.seed(config.seed + rank)
        torch.manual_seed(config.seed + rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        checkpoint = (
            load_checkpoint(config.resume, config.trust_checkpoint) if config.resume else None
        )
        if checkpoint:
            check_resume(config, checkpoint, world_size)
        raw_model = build_model(config).to(device)
        if checkpoint:
            raw_model.load_state_dict(checkpoint["model"], strict=True)
        elif config.init_checkpoint:
            initial = load_checkpoint(config.init_checkpoint, config.trust_checkpoint)
            if initial.get("format_version") == 2:
                raise ValueError("DDG initialization requires a supervised fast head")
            raw_model.load_state_dict(initial["model"], strict=True)
            del initial
            if rank == 0:
                logger.info(
                    "Initialized model from %s; optimizer and step start fresh",
                    config.init_checkpoint,
                )
        if config.ddg_manifest and not checkpoint:
            stream.set_ddg(config.ddg_manifest)
            if collection_config:
                # The initial pool supplies round zero; collect after training advances.
                stream.last_refresh = 0
        optimizer = build_optimizer(raw_model, config)
        if rank == 0:
            logger.info(
                "Algorithm loss_normalization=%s weight_decay_mode=%s optimizer_groups=%s",
                config.loss_normalization,
                config.weight_decay_mode,
                [
                    {
                        "parameters": sum(p.numel() for p in group["params"]),
                        "weight_decay": group["weight_decay"],
                    }
                    for group in optimizer.param_groups
                ],
            )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=device.type == "cuda" and config.precision == "float16"
        )
        model = torch.compile(raw_model) if config.compile else raw_model
        if distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                find_unused_parameters=True,
                broadcast_buffers=False,
            )
        step, best = 0, float("inf")
        if checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            step, best = checkpoint["step"], checkpoint["best_validation_loss"]
            stream.load_state_dict(checkpoint["ranks"][rank]["stream"])
            restore_rng(checkpoint["ranks"][rank]["rng"], device)
            checkpoint = None
        if rank == 0 and isinstance(stream, MixedBatchStream):
            logger.info(
                "DDG pool: rows=%d samples_per_batch=%d/%d last_refresh=%d",
                stream.ddg.manifest.total_rows if stream.ddg else 0,
                stream.ddg_batch_size,
                config.batch_size,
                stream.last_refresh,
            )
            if collection_config:
                next_refresh = (
                    (step + config.ddg_interval - 1) // config.ddg_interval
                ) * config.ddg_interval
                if next_refresh == stream.last_refresh:
                    next_refresh += config.ddg_interval
                logger.info(
                    "Online DDG enabled: mode=%s interval=%d next_refresh=%d; "
                    "collection and training alternate synchronously",
                    collection_config["mode"],
                    config.ddg_interval,
                    next_refresh,
                )
            else:
                logger.warning("Online DDG disabled: training uses only the supplied fixed pool")
        model.train()
        accumulation = config.gradient_accumulation_steps // world_size
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(
            total=config.max_steps, initial=step, desc="Training", unit="step", disable=rank != 0
        )
        skipped_updates = 0
        try:
            while step < config.max_steps:
                if (
                    config.ddg_collection_config
                    and step % config.ddg_interval == 0
                    and step != stream.last_refresh
                ):
                    refresh_ddg(
                        raw_model,
                        stream,
                        config,
                        output,
                        step,
                        device,
                        rank,
                        distributed,
                        collection_config,
                    )
                started = time.monotonic()
                lr = learning_rate_at(step, config)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                sums = torch.zeros(len(REGIONS) + 2, device=device)
                batches, denominators = None, None
                if config.loss_normalization == "global_tokens":
                    # Stage integer inputs on CPU to count valid tokens across ranks.
                    batches = [stream.next_batch(torch.device("cpu")) for _ in range(accumulation)]
                    denominators = sum(valid_counts(*batch) for batch in batches).to(device)
                    if distributed:
                        dist.all_reduce(denominators)
                for micro in range(accumulation):
                    if batches is None:
                        obs, target, action = stream.next_batch(device)
                    else:
                        obs, target, action = (tensor.to(device) for tensor in batches[micro])
                    sync = (
                        model.no_sync()
                        if distributed and micro < accumulation - 1
                        else nullcontext()
                    )
                    with sync:
                        with autocast(config, device):
                            loss, components = compute_loss(
                                model(obs, joint=True),
                                obs,
                                target,
                                action,
                                config,
                                denominators=denominators,
                                scale=accumulation * world_size,
                            )
                        finite = torch.isfinite(loss).to(torch.int32)
                        if distributed:
                            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                        if not finite.item():
                            raise FloatingPointError(f"Non-finite loss at completed step {step}")
                        scaler.scale(loss / accumulation).backward()
                    sums += (
                        torch.stack([loss.detach(), *[x.detach() for x in components.values()]])
                        / accumulation
                    )
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    raw_model.parameters(), config.grad_clip if config.grad_clip else float("inf")
                )
                finite = torch.isfinite(grad_norm).to(torch.int32)
                if distributed:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    if scaler.is_enabled() and skipped_updates < 10:
                        scaler.update(new_scale=scaler.get_scale() / 2)
                        optimizer.zero_grad(set_to_none=True)
                        skipped_updates += 1
                        if rank == 0:
                            logger.warning(
                                "FP16 overflow: skipped update, reduced scale to %s",
                                scaler.get_scale(),
                            )
                        continue
                    raise FloatingPointError(
                        "Non-finite gradients; stopping before an optimizer update"
                    )
                skipped_updates = 0
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if distributed:
                    dist.all_reduce(sums)
                    sums /= world_size
                elapsed = time.monotonic() - started
                values = dict(
                    zip(
                        ("loss", *REGIONS, "world"),
                        sums.tolist(),
                    )
                )
                if rank == 0:
                    progress.update(1)
                    progress.set_postfix(loss=f"{values['loss']:.4f}", lr=f"{lr:.2g}")
                    if step % config.log_interval == 0 or step == config.max_steps:
                        record = {
                            "step": step,
                            "lr": lr,
                            "seconds": elapsed,
                            "samples_per_second": config.batch_size
                            * config.gradient_accumulation_steps
                            / max(elapsed, 1e-9),
                            "grad_norm": float(grad_norm),
                            "action_weight": config.action_weight,
                            "world_weight": config.world_weight,
                            "ddg_rows": stream.ddg.manifest.total_rows
                            if isinstance(stream, MixedBatchStream) and stream.ddg
                            else 0,
                            "ddg_samples_per_batch": stream.ddg_batch_size
                            if isinstance(stream, MixedBatchStream) and stream.ddg
                            else 0,
                            "clip_coefficient": min(
                                1.0, config.grad_clip / (float(grad_norm) + 1e-6)
                            )
                            if config.grad_clip
                            else 1.0,
                            **values,
                        }
                        if denominators is not None:
                            record["valid_tokens"] = dict(zip(REGIONS, denominators.tolist()))
                        logger.info(json.dumps(record))
                        with (output / "metrics.jsonl").open("a") as file:
                            file.write(json.dumps(record) + "\n")
                improved = False
                if validation and (
                    step % config.validation_interval == 0 or step == config.max_steps
                ):
                    result = [None, None]
                    if rank == 0:
                        try:
                            result[0] = evaluate(raw_model, validation, config, device)
                        except Exception as exc:
                            result[1] = f"{type(exc).__name__}: {exc}"
                    if distributed:
                        dist.broadcast_object_list(result, src=0)
                    if result[1]:
                        raise RuntimeError(result[1])
                    improved = result[0]["total"] < best
                    best = min(best, result[0]["total"])
                    if rank == 0:
                        logger.info("Validation step=%d %s", step, json.dumps(result[0]))
                        with (output / "validation.jsonl").open("a") as file:
                            file.write(json.dumps({"step": step, **result[0]}) + "\n")
                if step % config.checkpoint_interval == 0 or step == config.max_steps or improved:
                    local_state = {"stream": stream.state_dict(), "rng": rng_state(device)}
                    ranks = [None] * world_size
                    if distributed:
                        dist.all_gather_object(ranks, local_state)
                    else:
                        ranks[0] = local_state
                    error = [None]
                    if rank == 0:
                        try:
                            payload = {
                                "format_version": 3,
                                "model_layout": "shared_backbone_v1",
                                "objective": "joint_action_world",
                                "supervised_heads": ["action", "world"],
                                "model": raw_model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scaler": scaler.state_dict(),
                                "step": step,
                                "iter_num": step,
                                "best_validation_loss": best,
                                "config": config.to_dict(),
                                "collection_signature": collection_hash,
                                "world_size": world_size,
                                "ranks": ranks,
                            }
                            atomic_save(payload, output / "last.pt")
                            if improved:
                                atomic_save(payload, output / "best.pt")
                            logger.info("Checkpoint saved at completed step %d", step)
                        except Exception as exc:
                            error[0] = f"{type(exc).__name__}: {exc}"
                    if distributed:
                        dist.broadcast_object_list(error, src=0)
                    if error[0]:
                        raise RuntimeError(error[0])
        finally:
            progress.close()
        return {"step": step, "output_dir": str(output), "train_rows": train_manifest.total_rows}
    except BaseException:
        if rank == 0 and handler is not None:
            logger.exception(
                "Training stopped before normal completion; latest successful checkpoint retained"
            )
        raise
    finally:
        stream.close()
        if handler:
            logger.removeHandler(handler)
            handler.close()
        if lock is not None and rank == 0:
            lock.unlink(missing_ok=True)
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", help="JSON configuration; CLI flags override fields")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifests/config and report the plan; no model, output directory or training",
    )
    defaults = TrainConfig()
    for key, value in defaults.to_dict().items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(
                flag,
                action=argparse.BooleanOptionalAction,
                default=None,
                help=f"Config field {key}; default {value}",
            )
        else:
            parser.add_argument(
                flag, type=type(value), default=None, help=f"Config field {key}; default {value!r}"
            )
    args = vars(parser.parse_args())
    filename, dry_run = args.pop("config"), args.pop("dry_run")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_config(filename, args)
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        config.validate(world_size)
        if dry_run:
            train_manifest = Manifest(config.train_manifest, "train")
            validation = (
                Manifest(config.validation_manifest, "validation")
                if config.validation_manifest
                else None
            )
            validate_disjoint(train_manifest, validation)
            if config.ddg_manifest:
                ddg_manifest = Manifest(config.ddg_manifest, "train")
                validate_disjoint(ddg_manifest, train_manifest)
                validate_disjoint(ddg_manifest, validation)
                if len(ddg_manifest.files) < world_size:
                    raise ValueError("Need at least one DDG Arrow file per global rank")
            if len(train_manifest.files) < world_size:
                raise ValueError("Need at least one training Arrow file per global rank")
            print(
                json.dumps(
                    {
                        "mode": "dry-run",
                        "config": config.to_dict(),
                        "train_rows": train_manifest.total_rows,
                        "train_files": len(train_manifest.files),
                        "validation_rows": validation.total_rows if validation else 0,
                        "global_batch": config.batch_size * config.gradient_accumulation_steps,
                        "micro_steps_per_rank": config.gradient_accumulation_steps // world_size,
                    },
                    indent=2,
                )
            )
        else:
            print(json.dumps(train(config)))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
