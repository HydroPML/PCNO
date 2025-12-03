from .oned.datapipes.kuramotosivashinsky1d import (
    onestep_test_datapipe_ks,
    onestep_valid_datapipe_ks,
    train_datapipe_ks,
    trajectory_test_datapipe_ks,
    trajectory_valid_datapipe_ks,
)

DATAPIPE_REGISTRY = {}

DATAPIPE_REGISTRY["KuramotoSivashinsky1D"] = {}
DATAPIPE_REGISTRY["KuramotoSivashinsky1D"]["train"] = train_datapipe_ks
DATAPIPE_REGISTRY["KuramotoSivashinsky1D"]["valid"] = [onestep_valid_datapipe_ks, trajectory_valid_datapipe_ks]
DATAPIPE_REGISTRY["KuramotoSivashinsky1D"]["test"] = [onestep_test_datapipe_ks, trajectory_test_datapipe_ks]
