# warp_ray_sensor.py

import torch
import math
from isaaclab.utils import math as math_utils


class WarpRaySensor:
    """
    A lightweight raycast distance sensor for IsaacLab pip mode.
    Uses scene.ray_query() to compute distances to obstacles.
    
    Supports:
    - multi-env
    - multi-agent
    - gpu batched raycasts
    """

    def __init__(
        self,
        num_rays=9,
        fov_degrees=80.0,
        max_distance=10.0,
        ray_origin_offset=(0.1, 0.0, 0.0),
        device="cuda:0",
    ):
        self.num_rays = num_rays
        self.fov_degrees = fov_degrees
        self.max_distance = max_distance
        self.ray_origin_offset = torch.tensor(ray_origin_offset, device=device)
        self.device = device

        # Precompute ray directions in BODY frame (single agent)
        angles = torch.linspace(-0.5, 0.5, steps=num_rays) * (fov_degrees * math.pi / 180.0)

        dirs = torch.stack(
            [torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)], dim=-1
        )  # (num_rays, 3)

        self.rays_body = dirs.to(device)  # (num_rays, 3)

    # ---------------------------------------------------------------------
    # MAIN METHOD: compute ray distances for a given robot articulation
    # ---------------------------------------------------------------------
    def compute(self, robot, scene, num_envs):
        """
        Compute ray distances for a given robot.

        Returns:
            distances: (num_envs, num_rays)
        """
        # 1. Get robot position/orientation
        pos_w = robot.data.root_pos_w  # (num_envs, 3)
        quat_w = robot.data.root_quat_w  # (num_envs, 4)

        # 2. Expand ray directions for all envs
        rays_body = self.rays_body.unsqueeze(0).expand(num_envs, -1, -1)  # (num_envs, num_rays, 3)

        # 3. Rotate rays from body frame → world frame
        # Reshape for quat_apply: (num_envs * num_rays, 3) and (num_envs * num_rays, 4)
        num_rays = rays_body.shape[1]
        
        # Expand quaternions to match each ray: (num_envs, num_rays, 4)
        quat_w_expanded = quat_w.unsqueeze(1).expand(-1, num_rays, -1)  # (num_envs, num_rays, 4)
        
        # Flatten for quat_apply
        quat_flat = quat_w_expanded.reshape(-1, 4)  # (num_envs * num_rays, 4)
        rays_flat = rays_body.reshape(-1, 3)  # (num_envs * num_rays, 3)
        
        # Apply rotation
        directions_w_flat = math_utils.quat_apply(quat_flat, rays_flat)  # (num_envs * num_rays, 3)
        
        # Reshape back to (num_envs, num_rays, 3)
        directions_w = directions_w_flat.reshape(num_envs, num_rays, 3)

        # 4. Compute ray origins (expand offset for all rays)
        # Expand ray_origin_offset to match num_envs: (num_envs, 3)
        ray_origin_offset_expanded = self.ray_origin_offset.unsqueeze(0).expand(num_envs, -1)  # (num_envs, 3)
        
        # Rotate offset from body frame to world frame
        ray_offset_w = math_utils.quat_apply(quat_w, ray_origin_offset_expanded)  # (num_envs, 3)
        
        # Add offset to robot position and expand for all rays
        origins_w = (pos_w + ray_offset_w).unsqueeze(1).expand(-1, num_rays, -1)  # (num_envs, num_rays, 3)

        origins_flat     = origins_w.reshape(-1, 3)
        directions_flat  = directions_w.reshape(-1, 3)

        # 5. Call IsaacLab's ray query (Warp)
        out = scene.sim.physics_sim_view.raycast(
            origins_flat,
            directions_flat,
            max_dist=self.max_distance,
        )
        distances = out["distance"].view(num_envs, self.num_rays)
        # Return only distances
        return distances  # (num_envs, num_rays)
