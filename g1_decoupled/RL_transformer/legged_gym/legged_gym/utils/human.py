import pickle
import sys
from collections import OrderedDict
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

try:
    import yaml
except ImportError:  # pragma: no cover - depends on env packages
    yaml = None


# TWIST2 keeps these numpy compatibility shims in its motion loader because some
# motion pickles were serialized against older numpy internals.
class FakeModule(ModuleType):
    def __init__(self, name, real=None):
        super().__init__(name)
        if real is not None:
            self.__dict__.update(real.__dict__)


sys.modules["numpy._core"] = FakeModule("numpy._core", np.core if hasattr(np, "core") else np)
sys.modules["numpy._core.multiarray"] = FakeModule(
    "numpy._core.multiarray", getattr(np.core, "multiarray", None)
)


# Old single-file H1-style loader kept for reference:
# import numpy as np
# import torch
#
# def load_target_jt(device, file, offset):
#     one_target_jt = np.load(f"data/{file}").astype(np.float32)
#     one_target_jt = torch.from_numpy(one_target_jt).to(device)
#     target_jt = one_target_jt.unsqueeze(0)
#     target_jt += offset
#
#     size = torch.tensor([one_target_jt.shape[0]]).to(device)
#     return target_jt, size
#
# Legacy helper kept for reference only:
# def _load_npy_motion(path, offset):
#     one_target_jt = np.load(path).astype(np.float32)
#     one_target_jt = torch.from_numpy(one_target_jt)
#     target_jt = one_target_jt + offset.cpu()
#     return target_jt, one_target_jt.shape[0]


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _PACKAGE_ROOT / "data"
_MOTION_CONFIG_DIR = _PACKAGE_ROOT / "motion_data_configs"


def _candidate_twist2_roots():
    roots = []
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "TWIST2_full"
        if candidate.exists():
            roots.append(candidate)
    explicit = Path("/home/shuaiyyy/humanoids_repo/TWIST2_full")
    if explicit.exists():
        roots.append(explicit)

    uniq = []
    seen = set()
    for root in roots:
        root = root.resolve()
        if root not in seen:
            uniq.append(root)
            seen.add(root)
    return uniq


def _resolve_source_path(file):
    source = Path(file).expanduser()
    candidates = []

    if source.is_absolute():
        candidates.append(source)
    else:
        candidates.extend([
            _DATA_DIR / source,
            _MOTION_CONFIG_DIR / source,
            _PACKAGE_ROOT / source,
            Path.cwd() / source,
        ])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Could not find motion source: {file}")


def _resolve_yaml_motion_path(yaml_path, root_path, rel_path):
    rel_path = Path(rel_path)
    candidates = []

    if root_path is not None:
        candidates.append(root_path / rel_path)
    candidates.append(yaml_path.parent / rel_path)

    for twist2_root in _candidate_twist2_roots():
        candidates.append(twist2_root / rel_path)
        candidates.append(twist2_root / "twist1_to_twist2" / rel_path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Could not resolve motion file '{rel_path}' from yaml '{yaml_path}'")


def _load_pkl_motion(path, expected_dofs):
    with open(path, "rb") as file_obj:
        motion_data = pickle.load(file_obj)

    if "dof_pos" not in motion_data:
        raise KeyError(f"Motion file '{path}' does not contain 'dof_pos'")

    one_target_jt = np.asarray(motion_data["dof_pos"], dtype=np.float32)
    if one_target_jt.ndim != 2:
        raise ValueError(f"Expected 2D dof_pos in '{path}', got shape {one_target_jt.shape}")
    if one_target_jt.shape[1] != expected_dofs:
        raise ValueError(
            f"Motion file '{path}' has {one_target_jt.shape[1]} dofs, expected {expected_dofs}"
        )

    fps = motion_data.get("fps", 30)
    return torch.from_numpy(one_target_jt), one_target_jt.shape[0], float(fps)


# Old eager-preload helper kept for reference:
# def _fetch_motion_files(source_path):
#     suffix = source_path.suffix.lower()
#
#     if source_path.is_dir():
#         return sorted(source_path.rglob("*.pkl"))
#
#     if suffix == ".yaml" or suffix == ".yml":
#         if yaml is None:
#             raise ImportError("PyYAML is required to load motion dataset yaml files")
#
#         with open(source_path, "r", encoding="utf-8") as yaml_file:
#             config = yaml.safe_load(yaml_file)
#
#         root_path = config.get("root_path")
#         root_path = Path(root_path).expanduser() if root_path else None
#         motions = config.get("motions", [])
#
#         motion_files = []
#         missing_motion_files = []
#         for motion in motions:
#             if "file" not in motion:
#                 continue
#             try:
#                 motion_files.append(_resolve_yaml_motion_path(source_path, root_path, motion["file"]))
#             except FileNotFoundError:
#                 missing_motion_files.append(motion["file"])
#
#         if missing_motion_files:
#             print(
#                 f"Skipping {len(missing_motion_files)} missing motion files from '{source_path.name}'. "
#                 f"First missing entry: {missing_motion_files[0]}"
#             )
#         return motion_files
#
#     if suffix == ".pkl":
#         return [source_path]
#
#     raise ValueError(
#         f"Unsupported motion source '{source_path}'. Use a TWIST2 .pkl, a folder of .pkl files, or a yaml motion list."
#     )


def _fetch_motion_entries(source_path):
    suffix = source_path.suffix.lower()

    if source_path.is_dir():
        return [(path.resolve(), 1.0) for path in sorted(source_path.rglob("*.pkl"))]

    if suffix == ".yaml" or suffix == ".yml":
        if yaml is None:
            raise ImportError("PyYAML is required to load motion dataset yaml files")

        with open(source_path, "r", encoding="utf-8") as yaml_file:
            config = yaml.safe_load(yaml_file)

        root_path = config.get("root_path")
        root_path = Path(root_path).expanduser() if root_path else None
        motions = config.get("motions", [])

        motion_entries = []
        missing_motion_files = []
        for motion in motions:
            if "file" not in motion:
                continue
            try:
                motion_entries.append(
                    (
                        _resolve_yaml_motion_path(source_path, root_path, motion["file"]),
                        float(motion.get("weight", 1.0)),
                    )
                )
            except FileNotFoundError:
                missing_motion_files.append(motion["file"])

        if missing_motion_files:
            print(
                f"Skipping {len(missing_motion_files)} missing motion files from '{source_path.name}'. "
                f"First missing entry: {missing_motion_files[0]}"
            )
        return motion_entries

    if suffix == ".pkl":
        return [(source_path.resolve(), 1.0)]

    raise ValueError(
        f"Unsupported motion source '{source_path}'. Use a TWIST2 .pkl, a folder of .pkl files, or a yaml motion list."
    )


class TargetJTMotionLib:
    def __init__(self, motion_file, device, expected_dofs, max_cached_motions=256):
        self._device = device
        self._expected_dofs = expected_dofs
        self._max_cached_motions = max_cached_motions
        self._cache = OrderedDict()

        source_path = _resolve_source_path(motion_file)
        motion_entries = _fetch_motion_entries(source_path)
        if not motion_entries:
            raise RuntimeError(f"No valid motion sequences found in source '{source_path}'")

        self._motion_files = [entry[0] for entry in motion_entries]
        self._motion_weights = torch.tensor(
            [entry[1] for entry in motion_entries], dtype=torch.float32, device=self._device
        )
        weight_sum = torch.sum(self._motion_weights)
        if weight_sum <= 0:
            raise ValueError(f"Motion weights must sum to a positive value for '{source_path}'")
        self._motion_weights /= weight_sum

        print(f"Registered {len(self._motion_files)} target motions from '{source_path.name}'")

    def num_motions(self):
        return len(self._motion_files)

    def sample_motions(self, n):
        return torch.multinomial(self._motion_weights, num_samples=n, replacement=True)

    def _evict_if_needed(self):
        while len(self._cache) > self._max_cached_motions:
            self._cache.popitem(last=False)

    def _load_motion(self, motion_id):
        motion_file = self._motion_files[motion_id]
        sequence, seq_len, fps = _load_pkl_motion(motion_file, self._expected_dofs)
        motion = {
            "sequence": sequence.to(self._device),
            "seq_len": seq_len,
            "fps": fps,
        }
        self._cache[motion_id] = motion
        self._cache.move_to_end(motion_id)
        self._evict_if_needed()
        return motion

    def _get_motion(self, motion_id):
        motion_id = int(motion_id)
        motion = self._cache.get(motion_id)
        if motion is None:
            motion = self._load_motion(motion_id)
        else:
            self._cache.move_to_end(motion_id)
        return motion

    def get_motion_lengths(self, motion_ids):
        motion_lengths = torch.zeros_like(motion_ids, dtype=torch.long, device=self._device)
        for motion_id in torch.unique(motion_ids).tolist():
            motion = self._get_motion(motion_id)
            motion_lengths[motion_ids == motion_id] = motion["seq_len"]
        return motion_lengths

    def get_motion_fps(self, motion_ids):
        motion_fps = torch.zeros_like(motion_ids, dtype=torch.float32, device=self._device)
        for motion_id in torch.unique(motion_ids).tolist():
            motion = self._get_motion(motion_id)
            motion_fps[motion_ids == motion_id] = motion["fps"]
        return motion_fps

    def get_frames(self, motion_ids, frame_ids):
        frames = torch.zeros(
            motion_ids.shape[0], self._expected_dofs, dtype=torch.float32, device=self._device
        )
        for motion_id in torch.unique(motion_ids).tolist():
            env_mask = motion_ids == motion_id
            motion = self._get_motion(motion_id)
            seq = motion["sequence"]
            motion_frame_ids = frame_ids[env_mask].clamp(min=0, max=motion["seq_len"] - 1)
            frames[env_mask] = seq[motion_frame_ids]
        return frames


# Old eager loader kept for reference:
# def load_target_jt(device, file, offset):
#     source_path = _resolve_source_path(file)
#     motion_files = _fetch_motion_files(source_path)
#     expected_dofs = offset.shape[-1]
#
#     sequences = []
#     lengths = []
#
#     for motion_file in motion_files:
#         suffix = motion_file.suffix.lower()
#         if suffix == ".pkl":
#             sequence, seq_len, _ = _load_pkl_motion(motion_file, expected_dofs)
#         else:
#             raise ValueError(
#                 f"Unsupported motion file '{motion_file}'. Legacy .npy loading is disabled; use TWIST2 .pkl motions."
#             )
#
#         if sequence.shape[1] != expected_dofs:
#             raise ValueError(
#                 f"Loaded motion '{motion_file}' with dof dim {sequence.shape[1]}, expected {expected_dofs}"
#             )
#
#         sequences.append(sequence)
#         lengths.append(seq_len)
#
#     if not sequences:
#         raise RuntimeError(f"No valid motion sequences found in source '{source_path}'")
#
#     target_jt_seq, target_jt_seq_len = _pack_motion_bank(sequences, lengths, device)
#     return target_jt_seq, target_jt_seq_len
