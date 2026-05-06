import torch

PRESETS = {
    "jasper": dict(
        y_key="Y", gt_key="A", em_key="M",
        p=4, h=100, w=100,
        epochs=2000, lr=5e-3
    ),
    "cuprite": dict(
        y_key="X", gt_key=None, em_key="M",
        p=4, h=512, w=614,
        epochs=3000, lr=5e-3
    ),
    "chandrayaan": dict(
        y_key="Y", gt_key=None, em_key=None,
        h=2995, w=304,
        epochs=3000, lr=5e-3
    )
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"