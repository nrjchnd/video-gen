"""Runtime policy decisions for forced API mode."""

from __future__ import annotations


def decide_force_api_generations(system: str, cuda_available: bool, vram_gb: int | None) -> bool:
    """Return whether API-only generation must be forced for this runtime."""
    if system == "Darwin":
        return True

    if system in ("Windows", "Linux"):
        if not cuda_available:
            return True
        if vram_gb is None:
            return True
        # LTX-2 requires significant VRAM for local inference.
        # We allow it if >= 12GB (covering most high-end laptop GPUs).
        return vram_gb < 12

    # Fail closed for other non-target platforms.
    return True
