import numpy as np
import pytest

from geom_utils import normalized_point_to_world


def test_normalized_point_to_world_center_downward():
    # Camera at (0,0,1) looking straight down (optical axis -Z)
    fx = fy = 1000.0
    cx = cy = 500.0
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    R = np.diag([1, 1, -1])  # maps camera z to world -z
    t = np.array([0, 0, 1])
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = R
    extrinsics[:3, 3] = t

    world_xy = normalized_point_to_world([500, 500], 1000, 1000, intrinsics, extrinsics)
    assert np.allclose(world_xy, [0.0, 0.0], atol=1e-6)


def test_normalized_point_to_world_raises_parallel():
    intrinsics = np.eye(3)
    # Camera optical axis lies in the ground plane (z component zero)
    R = np.array([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]])  # column3 = [1,0,0] -> ray z = 0 when x=0
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = R
    # Center pixel -> x=0 after normalization/intrinsics => ray_dir z ~ 0
    with pytest.raises(ValueError):
        normalized_point_to_world([0, 0], 1, 1, intrinsics, extrinsics)
