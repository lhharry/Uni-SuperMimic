"""Register the fixed-base Panda SuperMimic task with UniLab."""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg

from unisupermimic.tasks.motion_tracking.panda.supermimic.env import (
    SUPPORTED_BACKENDS,
    make_panda_supermimic_env,
)

TASK_NAME = "PandaSuperMimic"

registry.register_env_config(TASK_NAME, ManagerBasedRlEnvCfg)
for backend in SUPPORTED_BACKENDS:
    registry.register_env(TASK_NAME, make_panda_supermimic_env, sim_backend=backend)

__all__ = ["TASK_NAME"]
