import numpy as np


def normalized_point_to_world(pt_norm, img_width, img_height, intrinsics, extrinsics):
    """
    Convert normalized image point [y,x] in 0-1000 space to world (x,y) on z=0 plane.
    intrinsics: 3x3 camera matrix (fx, fy, cx, cy)
    extrinsics: 4x4 matrix [R|t] from world to camera
    """
    y_norm, x_norm = pt_norm
    pixel_y = y_norm / 1000.0 * img_height
    pixel_x = x_norm / 1000.0 * img_width

    img_point = np.array([pixel_x, pixel_y, 1.0], dtype=float)
    cam_point = np.linalg.inv(intrinsics) @ img_point

    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]

    ray_dir = R.T @ cam_point
    cam_origin = -R.T @ t

    if abs(ray_dir[2]) < 1e-6:
        raise ValueError("Ray parallel to ground plane; cannot intersect z=0.")

    s = -cam_origin[2] / ray_dir[2]
    world_point = cam_origin + s * ray_dir
    return world_point[:2]
