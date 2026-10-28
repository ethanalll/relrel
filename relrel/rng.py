"""Stable, independently keyed random streams for relational generation.

The existing factories consume process-global Python, NumPy and PyTorch RNGs.
``scoped_rng`` adapts them without coupling independently sampled components.
"""

import dataclasses
import hashlib
import json
import math
import random
from collections.abc import Mapping
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

RNG_VERSION = "keyed-sha256-v1"


def _canonical(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            return {
                "__mapping__": sorted(
                    [(_canonical(key), _canonical(item)) for key, item in value.items()],
                    key=lambda pair: json.dumps(pair[0], sort_keys=True),
                )
            }
        return {key: _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_canonical(item) for item in value), key=lambda x: json.dumps(x, sort_keys=True)
        )
    if isinstance(value, Enum):
        return {
            "__enum__": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": _canonical(value.value),
        }
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _canonical(value.item())
    if isinstance(value, np.ndarray):
        return {
            "__array__": value.dtype.str,
            "shape": list(value.shape),
            "values": _canonical(value.tolist()),
        }
    if isinstance(value, torch.Tensor):
        return {
            "__tensor__": str(value.dtype),
            "shape": list(value.shape),
            "values": _canonical(value.detach().cpu().tolist()),
        }
    if isinstance(value, torch.distributions.Beta):
        return {
            "__distribution__": "Beta",
            "concentration1": _canonical(value.concentration1),
            "concentration0": _canonical(value.concentration0),
        }
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if callable(value):
        # Lambda bytecode/constants distinguish callable configuration values
        # in SCMParams without relying on addresses or source checkout paths.
        name = f"{getattr(value, '__module__', type(value).__module__)}.{getattr(value, '__qualname__', type(value).__qualname__)}"
        code = getattr(value, "__code__", None)
        result = {"__callable__": name}
        if code is not None:
            result["bytecode"] = code.co_code.hex()
            result["constants"] = [
                _canonical(item)
                if isinstance(item, (str, int, float, bool, type(None), tuple))
                else type(item).__qualname__
                for item in code.co_consts
            ]
            result["defaults"] = _canonical(getattr(value, "__defaults__", None))
            result["keyword_defaults"] = _canonical(getattr(value, "__kwdefaults__", None))
            cells = getattr(value, "__closure__", None)
            if cells:
                if any(cell.cell_contents is value for cell in cells):
                    raise ValueError(
                        "Recursive callable closures cannot be configuration identities"
                    )
                result["closure"] = [_canonical(cell.cell_contents) for cell in cells]
        elif hasattr(value, "__dict__"):
            result["state"] = _canonical(vars(value))
        return result
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite values cannot be fingerprinted")
        return value
    raise TypeError(f"Unsupported canonical identity type: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Canonical JSON suitable for persistent configuration identities."""
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def keyed_seed(seed: int, *identities: Any) -> int:
    """Derive a stable 63-bit seed; never use Python's randomized ``hash``."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    payload = canonical_json([RNG_VERSION, int(seed), identities]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


@contextmanager
def scoped_rng(seed: int):
    """Temporarily seed legacy factories and restore every touched RNG state.

    CUDA state is preserved only when CUDA was already initialized, so CPU
    generation does not initialize a device as a side effect. Like the existing
    global-RNG factories this adapter is process-safe, not thread-safe.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        random.seed(int(seed))
        np.random.seed(int(seed) % (1 << 32))
        # torch.manual_seed also queues lazy CUDA seeding when no device is
        # initialized. Seed the CPU generator directly to avoid leaking that
        # queued state outside a CPU-only scope.
        torch.random.default_generator.manual_seed(int(seed))
        if cuda_states is not None:
            torch.cuda.manual_seed_all(int(seed))
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
