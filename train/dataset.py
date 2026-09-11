"""Arrow manifests and resumable minibatch loading."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch

HELD_OUT = re.compile(r"(^|[/_.-])(eval|test|validation)(?=$|[/_.-])", re.I)
SCHEMA = pa.schema(
    [
        ("obs", pa.list_(pa.int8(), 256)),
        ("next_obs", pa.list_(pa.int8(), 256)),
        ("next_action", pa.int8()),
    ]
)


class Manifest:
    def __init__(self, filename, split):
        self.path = Path(filename).resolve()
        content = json.loads(self.path.read_text())
        if not isinstance(content, dict) or content.get("split") != split:
            raise ValueError(f"Manifest must declare split={split!r}: {filename}")
        entries = content.get("files")
        if (
            not isinstance(entries, list)
            or not entries
            or not all(isinstance(x, str) and x for x in entries)
        ):
            raise ValueError(
                "Manifest files must be a nonempty list of explicit paths (no globbing)"
            )
        self.files = []
        self.rows = {}
        identities, signature = set(), []
        for name in entries:
            path = (self.path.parent / name).resolve()
            if split == "train" and (HELD_OUT.search(name) or HELD_OUT.search(path.name)):
                raise ValueError(f"Held-out filename is not accepted as training input: {name}")
            if path.suffix != ".arrow" or not path.is_file():
                raise ValueError(f"Expected an existing Arrow file: {path}")
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity in identities:
                raise ValueError(f"Duplicate data file or hardlink: {path}")
            identities.add(identity)
            with pa.memory_map(str(path)) as source:
                reader = ipc.open_file(source)
                if not reader.schema.equals(SCHEMA, check_metadata=False):
                    raise ValueError(f"Arrow schema mismatch in {path}: expected {SCHEMA}")
                rows = sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches))
            if rows == 0:
                raise ValueError(f"Empty Arrow file: {path}")
            self.files.append(str(path))
            self.rows[str(path)] = rows
            signature.append((str(path), stat.st_size, stat.st_mtime_ns, rows))
        self.identities = identities
        self.total_rows = sum(self.rows.values())
        self.fingerprint = hashlib.sha256(json.dumps(signature).encode()).hexdigest()


def validate_disjoint(train, validation):
    if validation is not None and train.identities & validation.identities:
        raise ValueError("Training and validation reference the same file (including hardlinks)")


class ArrowBatchStream:
    """One memory-mapped record batch on CPU; only yielded minibatches reach GPU.

    File-level sharding uses global RANK, not LOCAL_RANK. Epoch/file/record/row
    cursor and seed fully determine the next batch on resume.
    """

    def __init__(self, manifest, batch_size, seed=1337, rank=0, world_size=1, shuffle=True):
        if not 0 <= rank < world_size or batch_size < 1:
            raise ValueError("Invalid stream rank/world_size/batch_size")
        self.manifest = manifest
        self.batch_size, self.seed, self.shuffle = batch_size, seed, shuffle
        shards, sizes = [[] for _ in range(world_size)], [0] * world_size
        for file in sorted(manifest.files, key=lambda f: (-manifest.rows[f], f)):
            selected = min(range(world_size), key=lambda r: sizes[r])
            shards[selected].append(file)
            sizes[selected] += manifest.rows[file]
        self.files = shards[rank]
        if not self.files:
            raise ValueError("Need at least one training Arrow file per global rank")
        self.epoch = self.file_index = self.record_index = self.row_index = 0
        self._source = self._reader = self._record = self._cache_key = None
        self._order_epoch = None

    def _order(self):
        if not self.shuffle:
            return self.files
        if self._order_epoch != self.epoch:
            order = np.random.default_rng(self.seed + self.epoch).permutation(len(self.files))
            self._file_order = [self.files[i] for i in order]
            self._order_epoch = self.epoch
        return self._file_order

    def close(self):
        self._record = self._reader = self._cache_key = None
        if self._source is not None:
            self._source.close()
            self._source = None

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "file_index": self.file_index,
            "record_index": self.record_index,
            "row_index": self.row_index,
            "files": self.files,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "fingerprint": self.manifest.fingerprint,
        }

    def load_state_dict(self, state):
        for key in ("files", "seed", "batch_size", "shuffle", "fingerprint"):
            if self.state_dict()[key] != state[key]:
                raise ValueError(f"Data stream resume mismatch: {key}")
        for key in ("epoch", "file_index", "record_index", "row_index"):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError(f"Invalid stream cursor: {key}")
            setattr(self, key, state[key])
        if self.file_index >= len(self.files):
            raise ValueError("Invalid stream file index")
        self.close()

    def next_batch(self, device="cpu", size=None):
        pieces, needed = [], self.batch_size if size is None else size
        if needed < 1:
            raise ValueError("Batch size must be positive")
        while needed:
            file = self._order()[self.file_index]
            key = (self.epoch, file, self.record_index)
            if self._cache_key != key:
                self.close()
                self._source = pa.memory_map(file)
                self._reader = ipc.open_file(self._source)
                if self.record_index >= self._reader.num_record_batches:
                    self.record_index = self.row_index = 0
                    self.file_index += 1
                    if self.file_index == len(self.files):
                        self.file_index = 0
                        self.epoch += 1
                    continue
                self._record = self._reader.get_batch(self.record_index)
                order_seed = int.from_bytes(
                    hashlib.sha256(
                        f"{self.seed}:{self.epoch}:{file}:{self.record_index}".encode()
                    ).digest()[:8],
                    "little",
                )
                self._indices = (
                    np.random.default_rng(order_seed).permutation(self._record.num_rows)
                    if self.shuffle
                    else np.arange(self._record.num_rows)
                )
                self._cache_key = key
            available = self._record.num_rows - self.row_index
            if available < 0:
                raise ValueError("Resume row cursor exceeds the record batch size")
            count = min(needed, available)
            if count:
                rows = self._record.take(
                    pa.array(self._indices[self.row_index : self.row_index + count])
                )
                if any(
                    column.null_count
                    or (pa.types.is_fixed_size_list(column.type) and column.values.null_count)
                    for column in rows.columns
                ):
                    raise ValueError(f"Null values in training data: {file}")
                obs = rows.column(0).values.to_numpy(zero_copy_only=False).reshape(count, 256)
                future = rows.column(1).values.to_numpy(zero_copy_only=False).reshape(count, 256)
                actions = rows.column(2).to_numpy(zero_copy_only=False)
                if (
                    (obs < 0).any()
                    or (obs > 66).any()
                    or (future < 0).any()
                    or (future > 66).any()
                    or (actions < 0).any()
                    or (actions > 4).any()
                ):
                    raise ValueError(f"Out-of-range observation/action tokens: {file}")
                pieces.append((obs, future, actions))
                needed -= count
                self.row_index += count
            if self.row_index >= self._record.num_rows:
                self.record_index += 1
                self.row_index = 0
        batches = tuple(
            torch.from_numpy(np.concatenate([p[i] for p in pieces]).astype(np.int64))
            for i in range(3)
        )
        if torch.device(device).type == "cuda":
            return tuple(batch.pin_memory().to(device, non_blocking=True) for batch in batches)
        return batches


class MixedBatchStream:
    """Offline/DDG minibatches with independently resumable data cursors."""

    def __init__(self, offline, ratio, validation=None, rank=0, world_size=1):
        self.offline, self.ratio, self.validation = offline, ratio, validation
        self.rank, self.world_size = rank, world_size
        self.ddg = None
        self.last_refresh = -1
        self.ddg_batch_size = int(offline.batch_size * ratio)
        if not 1 <= self.ddg_batch_size < offline.batch_size:
            raise ValueError("DDG ratio must allocate at least one row to each data source")

    def set_ddg(self, filename):
        manifest = Manifest(filename, "train")
        if len(manifest.files) < self.world_size:
            raise ValueError(
                "Need at least one DDG Arrow file per global rank; reduce rows_per_shard"
            )
        validate_disjoint(manifest, self.offline.manifest)
        validate_disjoint(manifest, self.validation)
        replacement = ArrowBatchStream(
            manifest, self.ddg_batch_size, self.offline.seed + 1, self.rank, self.world_size
        )
        if self.ddg is not None:
            self.ddg.close()
        self.ddg = replacement

    def next_batch(self, device="cpu"):
        if self.ddg is None:
            return self.offline.next_batch(device)
        offline = self.offline.next_batch(device, self.offline.batch_size - self.ddg_batch_size)
        online = self.ddg.next_batch(device)
        return tuple(torch.cat([a, b]) for a, b in zip(offline, online))

    def state_dict(self):
        return {
            "offline": self.offline.state_dict(),
            "ddg": self.ddg.state_dict() if self.ddg else None,
            "ddg_manifest": str(self.ddg.manifest.path) if self.ddg else "",
            "ratio": self.ratio,
            "last_refresh": self.last_refresh,
        }

    def load_state_dict(self, state):
        if state["ratio"] != self.ratio:
            raise ValueError("DDG mixing ratio changed during resume")
        self.offline.load_state_dict(state["offline"])
        if state["ddg"] is not None:
            self.set_ddg(state["ddg_manifest"])
            self.ddg.load_state_dict(state["ddg"])
        self.last_refresh = state["last_refresh"]

    def close(self):
        self.offline.close()
        if self.ddg is not None:
            self.ddg.close()
