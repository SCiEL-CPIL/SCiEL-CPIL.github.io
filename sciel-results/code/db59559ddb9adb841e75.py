"""Kinematic closure: 12-hexagon honeycomb cluster, optimize global + inner rotation."""
import numpy as np
from scipy.optimize import minimize_scalar

N = 12

# Honeycomb cluster centers for 12 unit hexagons (side=1)
# In a honeycomb, adjacent hex centers are 2*sqrt(3)/2 = sqrt(3) apart
# Use axial coordinates: center at (q, r) -> (sqrt(3)*(q + r/2), 1.5*r)
# 12 hexagons in a compact cluster: center + 6 neighbors + 5 more
_honeycomb_axial = [
    (0, 0),
    (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (0, 1),  # center + 6 neighbors
    (2, -1), (2, -2), (1, -2), (-1, -2), (-2, -1),  # 5 more to make 12
]

def _axial_to_xy(q, r):
    x = np.sqrt(3.0) * (q + r / 2.0)
    y = 1.5 * r
    return x, y

# Precompute cluster centers
_cluster_centers = np.array([_axial_to_xy(q, r) for q, r in _honeycomb_axial])

# Six outward normal directions of a regular hexagon (pointing to vertices)
# For a hexagon with flat-top orientation, normals are at 0, 60, 120, 180, 240, 300 degrees
_normal_angles = np.deg2rad(np.arange(6) * 60.0)
_normals = np.stack([np.cos(_normal_angles), np.sin(_normal_angles)], axis=1)  # shape (6,2)

def _hexagon_support(angle_deg, normals):
    """Max projection of a unit regular hexagon (side=1) in each normal direction.
    A unit hexagon has vertices at distance 1 from center (circumradius = side = 1).
    The support in direction n is max over vertices v of v·n.
    Vertices are at angles angle_deg + 60*k, radius 1."""
    a = np.deg2rad(angle_deg)
    vertex_angles = a + np.deg2rad(np.arange(6) * 60.0)
    vertices = np.stack([np.cos(vertex_angles), np.sin(vertex_angles)], axis=1)  # (6,2)
    # support_k = max_j vertices[j] · normals[k]
    # = max over j of (cos(vj)·cos(nk) + sin(vj)·sin(nk))
    # For a regular hexagon, support in direction n is cos(angle_diff) where angle_diff
    # is the minimum angular distance from n to the nearest vertex direction.
    # Actually: support = max_j cos(vj - nk) = cos(min_j |vj - nk|)
    diff = np.abs(np.mod(vertex_angles[:, None] - normals_angles[None, :] + np.pi, 2*np.pi) - np.pi)
    return np.cos(diff.max(axis=0))  # Wait, we want max of cos, which is cos of min diff
    # Correction: max_j cos(vj - nk) = cos(min_j |vj - nk|) since cos is decreasing on [0, pi]

def _hex_support_fast(angle_deg, normals):
    """Support of unit hexagon in 6 normal directions."""
    a = np.deg2rad(angle_deg)
    vertex_angles = a + np.deg2rad(np.arange(6) * 60.0)
    # For each normal direction nk, find min angular distance to any vertex
    nk = np.deg2rad(np.arange(6) * 60.0)
    # diff[i,j] = angular distance between vertex i and normal j
    diff = np.abs(vertex_angles[:, None] - nk[None, :])
    diff = np.minimum(diff, 2*np.pi - diff)
    min_diff = diff.min(axis=0)  # (6,)
    return np.cos(min_diff)

def _outer_side(cluster_rotation_deg, inner_rotation_deg):
    """Compute the outer hexagon side length for given rotations."""
    theta = np.deg2rad(cluster_rotation_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_centers = _cluster_centers @ R.T  # (12, 2)
    
    # Support of cluster in each normal direction
    # cluster_support_k = max_i (rotated_centers[i] · normals[k])
    proj = rotated_centers @ _normals.T  # (12, 6)
    cluster_support = proj.max(axis=0)  # (6,)
    
    # Support of each inner hexagon in each normal direction
    inner_sup = _hex_support_fast(inner_rotation_deg, _normals)  # (6,)
    
    # Outer hexagon support in direction k = cluster_support_k + inner_sup_k
    # (since the hexagon centered at rotated_centers[i] extends to rotated_centers[i]·n + inner_sup)
    # The max over i is cluster_support_k + inner_sup_k (inner_sup is same for all since same rotation)
    outer_support = cluster_support + inner_sup  # (6,)
    
    # For a regular hexagon centered at origin with side S, support in each of the 6 
    # normal directions (aligned with the hexagon) is S.
    # The minimum S such that S >= outer_support_k for all k is max(outer_support)
    return outer_support.max()

def _outer_side_with_center(cluster_rotation_deg, inner_rotation_deg):
    """Compute outer side and optimal center offset."""
    theta = np.deg2rad(cluster_rotation_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_centers = _cluster_centers @ R.T  # (12, 2)
    
    proj = rotated_centers @ _normals.T  # (12, 6)
    cluster_support = proj.max(axis=0)  # (6,)
    
    inner_sup = _hex_support_fast(inner_rotation_deg, _normals)  # (6,)
    
    outer_support = cluster_support + inner_sup  # (6,)
    return outer_support.max()

# Optimize over cluster rotation and inner hexagon rotation
def _objective(params):
    cluster_rot, inner_rot = params
    return _outer_side(cluster_rot, inner_rot)

# Use grid search + local refinement for the 2D problem
def _optimize():
    best_val = np.inf
    best_params = (0.0, 0.0)
    
    # Coarse grid
    for cr in np.arange(0, 60, 2.0):
        for ir in np.arange(0, 60, 2.0):
            val = _outer_side(cr, ir)
            if val < best_val:
                best_val = val
                best_params = (cr, ir)
    
    # Refine around best
    for _ in range(3):
        cr0, ir0 = best_params
        for dc in np.arange(-2, 2.01, 0.25):
            for di in np.arange(-2, 2.01, 0.25):
                cr = (cr0 + dc) % 60
                ir = (ir0 + di) % 60
                val = _outer_side(cr, ir)
                if val < best_val:
                    best_val = val
                    best_params = (cr, ir)
    
    return best_params, best_val

_best_params, _best_val = _optimize()
_best_cluster_rot, _best_inner_rot = _best_params

def construct():
    """Return 12 inner hexagons, outer hexagon center, side, and rotation."""
    cluster_rot, inner_rot = _best_cluster_rot, _best_inner_rot
    theta = np.deg2rad(cluster_rot)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_centers = _cluster_centers @ R.T
    
    # Inner hexagons: each at rotated center with rotation = inner_rot
    inner = []
    for i in range(N):
        inner.append([float(rotated_centers[i, 0]), float(rotated_centers[i, 1]), float(inner_rot)])
    
    # Outer hexagon: center at origin, side = best_val, rotation = 0 (normals aligned)
    outer_center = (0.0, 0.0)
    outer_side = float(_best_val)
    outer_rotation = 0.0
    
    return inner, outer_center, outer_side, outer_rotation

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
