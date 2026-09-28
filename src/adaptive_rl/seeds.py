"""Validated, deterministic allocation of environment reset seeds."""

from __future__ import annotations

from collections.abc import Sequence

# A portable unsigned 32-bit domain is accepted by Gymnasium and common NumPy
# seed consumers; use it for every reset seed rather than relying on backend-
# specific support for arbitrarily large Python integers.
MAX_SEED = (1 << 32) - 1
CUSTOM_SEED_START = 0
CUSTOM_SEED_END = MAX_SEED + 1
TRAIN_SEED_START = 0
TRAIN_SEED_END = 1000
TEST_SEED_START = 1000
TEST_SEED_END = 1200

SEED_ALLOCATION_PROTOCOL = "sorted-group-fixed-block"
SEED_ALLOCATION_VERSION = 1

SPLIT_INTERVALS: dict[str, tuple[int, int]] = {
    "custom": (CUSTOM_SEED_START, CUSTOM_SEED_END),
    "train": (TRAIN_SEED_START, TRAIN_SEED_END),
    "test": (TEST_SEED_START, TEST_SEED_END),
}


def validate_seed(seed: object, *, label: str = "Seed") -> int:
    """Return a Gymnasium/NumPy-compatible unsigned 32-bit seed."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"{label} must be an integer in [0, {MAX_SEED}], got {seed!r}.")
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"{label} must be an integer in [0, {MAX_SEED}], got {seed!r}.")
    return seed


def validate_seed_groups(seeds: Sequence[int], *, split: str | None = None) -> list[int]:
    """Validate and canonicalize group identifiers."""
    if not seeds:
        raise ValueError("Evaluation group seeds must not be empty.")
    groups = [validate_seed(seed, label="Evaluation group seed") for seed in seeds]
    if len(groups) != len(set(groups)):
        raise ValueError("Evaluation group seeds must be unique; duplicate groups are not allowed.")
    groups.sort()

    if split is not None:
        if split not in SPLIT_INTERVALS or split == "custom":
            raise ValueError(f"Invalid evaluation split {split!r}; expected 'train' or 'test'.")
        start, end = SPLIT_INTERVALS[split]
        outside = [seed for seed in groups if not start <= seed < end]
        if outside:
            raise ValueError(
                f"Evaluation group seeds {outside} do not belong to the {split!r} split "
                f"[{start}, {end})."
            )
    return groups


def validate_episodes_per_group(episodes_per_seed: object) -> int:
    """Validate the requested positive integer episode count."""
    if isinstance(episodes_per_seed, bool) or not isinstance(episodes_per_seed, int):
        raise ValueError(
            f"Evaluation episodes per group must be a positive integer, got {episodes_per_seed!r}."
        )
    if episodes_per_seed <= 0:
        raise ValueError(
            f"Evaluation episodes per group must be a positive integer, got {episodes_per_seed!r}."
        )
    return episodes_per_seed


def allocate_episode_reset_seeds(
    seeds: Sequence[int], episodes_per_seed: int, *, split: str = "custom"
) -> dict[int, list[int]]:
    """Allocate disjoint, deterministic reset-seed blocks to sorted groups.

    Block width depends on the seed-space capacity and the canonical group set,
    not on ``episodes_per_seed``. Extending the episode count therefore keeps
    existing (group, episode-index) assignments stable. Adding/removing a group
    changes block ranks and may change every group's assigned reset seeds.
    """
    if not isinstance(split, str):
        raise ValueError("Evaluation split must be one of 'custom', 'train', or 'test'.")
    clean_split = split.strip().lower()
    if clean_split not in SPLIT_INTERVALS:
        raise ValueError("Evaluation split must be one of 'custom', 'train', or 'test'.")
    groups = validate_seed_groups(seeds, split=None if clean_split == "custom" else clean_split)
    episode_count = validate_episodes_per_group(episodes_per_seed)

    start, end = SPLIT_INTERVALS[clean_split]
    capacity = end - start
    maximum_episodes = capacity // len(groups)
    total_required = len(groups) * episode_count
    if episode_count > maximum_episodes:
        raise ValueError(
            f"Reset-seed allocation for split {clean_split!r} cannot fit: "
            f"requested groups={groups}, episodes per group={episode_count}, "
            f"total reset seeds required={total_required}, available capacity={capacity}, "
            f"maximum feasible episodes per group={maximum_episodes}. "
            "Reduce --episodes or use fewer evaluation groups."
        )

    return {
        group: [
            start + rank * maximum_episodes + episode_index
            for episode_index in range(episode_count)
        ]
        for rank, group in enumerate(groups)
    }


__all__ = [
    "MAX_SEED",
    "SEED_ALLOCATION_PROTOCOL",
    "SEED_ALLOCATION_VERSION",
    "allocate_episode_reset_seeds",
    "validate_episodes_per_group",
    "validate_seed",
    "validate_seed_groups",
]
