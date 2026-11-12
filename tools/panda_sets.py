"""Versioned Panda reference sets (stage 5): where each set's retarget output, motion packs and
q0/alpha live, and the Hydra overrides that point training / eval at them.

  v1   the stage-3/4 set (CHOSEN cone a0=30/amax=50, deployed and tested on 2026-11-12).
       Paths are the original, un-suffixed ones; q0/alpha are the constants in robot.py.
  v2, v1b, cand_*, ...   new sets: runs/panda_retarget_<set>/, lafan1_panda_<set>_{dual,left}.pt,
       lafan1_panda_<set>.q0alpha.json (tools/derive_q0_alpha.py).

Writers must name the set explicitly; v1 is frozen (tag stage4-v1) and is never rewritten
unless --allow-v1 is passed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MOTIONS = REPO / "src" / "unisupermimic" / "assets" / "motions" / "panda"


@dataclass(frozen=True)
class RefSet:
    name: str

    @property
    def ret(self) -> Path:
        return REPO / "runs" / ("panda_retarget" if self.name == "v1" else f"panda_retarget_{self.name}")

    def pack(self, variant: str) -> Path:
        assert variant in ("dual", "left")
        stem = "lafan1_panda" if self.name == "v1" else f"lafan1_panda_{self.name}"
        return MOTIONS / f"{stem}_{variant}.pt"

    @property
    def q0alpha(self) -> Path | None:
        return None if self.name == "v1" else MOTIONS / f"lafan1_panda_{self.name}.q0alpha.json"

    def overrides(self, variant: str) -> list[str]:
        """Hydra overrides for unisupermimic-train / eval on this set (variant: left | dual)."""
        if self.name == "v1":
            return []
        rel = self.pack(variant).relative_to(REPO)
        return [f"env.commands.motion.motion_files=[{rel}]",
                f"env.commands.motion.q0_alpha_file={self.q0alpha.relative_to(REPO)}"]


def get(name: str, write: bool = False, allow_v1: bool = False) -> RefSet:
    if write and name == "v1" and not allow_v1:
        raise SystemExit("v1 is frozen (tag stage4-v1); pass --allow-v1 to overwrite it on purpose")
    return RefSet(name)


def add_args(ap, write: bool = False) -> None:
    ap.add_argument("--set", required=write, default=None if write else "v1",
                    help="reference set (v1 = frozen stage-4 set, v2, v1b, cand_A, ...)")
    if write:
        ap.add_argument("--allow-v1", action="store_true", help="really overwrite the frozen v1 files")
