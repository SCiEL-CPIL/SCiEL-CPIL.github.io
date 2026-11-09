"""
Strategy: Differentiable SDF-based optimization via L-BFGS-B.
Instead of discrete SAT checks or force-directed physics with fixed parameters,
we model the inner hexagons using Signed Distance Fields (SDFs).
We define a smooth objective function based on the maximum SDF overlap between
inner hexagons and the boundary violations against the outer hexagon.
We then use L-BFGS-B (a quasi-Newton method) to minimize this smooth objective.
This allows the optimizer to exploit gradients through near-contact states,
often finding tighter local minima than discrete or physics-based methods.
"""

import numpy as np
import time
import random
import math
import sys
from scipy.optimize import minimize

N = 11
DEADLINE = 145.0  # Safety margin (150s limit)
START_TIME = time.time()

def _remaining_time():
    return DEADLINE - (time.time() - START_TIME)

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    angle_rad = np.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + k * np.pi / 3
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)

def _get_face_normals(angle_deg):
    """Return unit normals for the 6 faces of a hexagon at given angle."""
    angle_rad = np.radians(angle_deg)
    normals = []
    for k in range(6):
        a = angle_rad + np.pi/6 + k * np.pi / 3
        normals.append([np.cos(a), np.sin(a)])
    return np.array(normals)

def _min_enclosing_side(inner_positions, outer_center, outer_angle):
    """
    Calculate the minimum side length required to contain all inner hexagons.
    """
    cx, cy = outer_center
    face_normals = _get_face_normals(outer_angle)
    
    max_proj = 0.0
    for (ix, iy, iangle) in inner_positions:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        iverts_rel = iverts - np.array(outer_center)
        
        for n in face_normals:
            projs = iverts_rel @ n
            m = np.max(projs)
            if m > max_proj:
                max_proj = m
                
    return 2.0 * max_proj / np.sqrt(3)

def _check_validity(inner_positions, outer_center, outer_side, outer_angle, tol=1e-5):
    """
    Strictly check if the configuration is valid:
    1. All inner hexagons are inside the outer hexagon.
    2. No inner hexagons overlap.
    """
    cx, cy = outer_center
    outer_normals = _get_face_normals(outer_angle)
    inradius = outer_side * np.sqrt(3) / 2.0
    
    all_verts = []
    for i in range(N):
        ix, iy, iangle = inner_positions[i]
        verts = _hex_vertices(ix, iy, 1.0, iangle)
        all_verts.append(verts)
        
    # 1. Check Containment
    for i in range(N):
        verts = all_verts[i]
        verts_rel = verts - np.array(outer_center)
        for n in outer_normals:
            projs = verts_rel @ n
            if np.max(projs) > inradius + tol:
                return False
                
    # 2. Check Disjointness (SAT)
    for i in range(N):
        for j in range(i + 1, N):
            v1 = all_verts[i]
            v2 = all_verts[j]
            
            n1 = _get_face_normals(inner_positions[i][2])
            n2 = _get_face_normals(inner_positions[j][2])
            
            disjoint = False
            for n in np.vstack([n1, n2]):
                proj1 = v1 @ n
                proj2 = v2 @ n
                
                if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                    disjoint = True
                    break
            
            if not disjoint:
                return False
                
    return True

def _sdf_hexagon(points, cx, cy, side, angle_deg):
    """
    Compute Signed Distance Field for a set of points relative to a regular hexagon.
    Negative inside, positive outside.
    """
    angle_rad = np.radians(angle_deg)
    # Rotate points into local frame
    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    
    ca = np.cos(-angle_rad)
    sa = np.sin(-angle_rad)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    # Hexagon defined by 6 half-planes: n_k . p <= inradius
    # n_k are face normals in local frame (0, 30, 90, ...)
    # inradius = side * sqrt(3) / 2
    inradius = side * np.sqrt(3) / 2.0
    
    # Local face normals
    local_normals = np.array([
        [np.cos(np.pi/6 + k * np.pi/3), np.sin(np.pi/6 + k * np.pi/3)] for k in range(6)
    ])
    
    # Projections: (6, M)
    # proj[k, m] = n_k . p_m
    proj = lx[None, :] * local_normals[:, 0][:, None] + ly[None, :] * local_normals[:, 1][:, None]
    
    # Penetration depth for each point is the max over normals of (proj - inradius)
    # If max(proj - inradius) > 0, point is outside.
    # SDF approx: max(proj - inradius)
    # This is a valid SDF for convex polygons (Chebyshev distance to boundary in normal directions)
    # Actually, for a regular hexagon, the Euclidean distance to the boundary is exactly 
    # max(0, max_k (n_k . p - inradius)) ? 
    # No, that's the distance to the supporting lines. The true Euclidean distance to the 
    # polygon boundary is the minimum distance to any edge.
    # However, for optimization purposes, the "max of linear functions" (support function 
    # inverse) provides a smooth (almost) gradient that pushes points out.
    # It is a valid distance metric for convex sets in the sense that it grows linearly 
    # away from the set.
    
    sdf = np.max(proj - inradius, axis=0)
    return sdf

def _objective(params, outer_center, outer_side, outer_angle):
    """
    Objective function for L-BFGS-B.
    Minimizes the maximum overlap (inner-inner and inner-outer).
    params: flattened array of [x0, y0, a0, x1, y1, a1, ...]
    """
    pos = params.reshape(N, 3)
    
    # Pre-compute vertices for all inner hexagons
    all_verts = []
    for i in range(N):
        ix, iy, iangle = pos[i]
        verts = _hex_vertices(ix, iy, 1.0, iangle)
        all_verts.append(verts)
        
    # 1. Boundary Violation
    # For each inner hex, compute SDF of its vertices against outer hex.
    # Outer hex is centered at outer_center, side outer_side, angle outer_angle.
    # We want the SDF to be <= 0 (inside).
    # Penalty is sum of squares of positive parts (penetrations).
    
    boundary_penalty = 0.0
    for i in range(N):
        verts = all_verts[i]
        sdf_vals = _sdf_hexagon(verts, outer_center[0], outer_center[1], outer_side, outer_angle)
        # sdf_vals > 0 means outside
        # We want to minimize max(sdf_vals) or sum(sdf_vals^2)
        # Using sum of squares is differentiable everywhere.
        # To encourage tight packing, we might penalize the max more heavily, 
        # but sum of squares is standard for soft constraints.
        boundary_penalty += np.sum(np.maximum(sdf_vals, 0.0) ** 2)
        
    # 2. Inner-Inner Overlap
    # For each pair, check if they overlap.
    # We can use the SDF of vertices of one hex against the other.
    # If any vertex of hex A is inside hex B (SDF < 0), they might overlap.
    # But SDF < 0 just means inside. We need to ensure they are disjoint.
    # A robust smooth metric: For each pair (i, j), compute the SDF of vertices of i 
    # relative to j. If min(SDF) < 0, they are in contact/overlap.
    # We want to push them apart. 
    # Let d_ij = min(SDF(verts_i, hex_j)).
    # If d_ij < 0, they overlap. Penalty = max(0, -d_ij)^2.
    # This is equivalent to saying: the "depth" of penetration is -d_ij.
    
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            # SDF of vertices of i relative to j
            sdf_ij = _sdf_hexagon(all_verts[i], pos[j, 0], pos[j, 1], 1.0, pos[j, 2])
            min_sdf_ij = np.min(sdf_ij)
            
            # SDF of vertices of j relative to i
            sdf_ji = _sdf_hexagon(all_verts[j], pos[i, 0], pos[i, 1], 1.0, pos[i, 2])
            min_sdf_ji = np.min(sdf_ji)
            
            # If either is negative, they overlap.
            # Use the minimum of the two "min SDFs" to capture the deepest penetration.
            # Actually, if they overlap, both min SDFs will likely be negative.
            # We penalize the "amount" of overlap.
            # Let penetration = max(0, -min_sdf_ij, -min_sdf_ji)
            pen_ij = max(0.0, -min_sdf_ij)
            pen_ji = max(0.0, -min_sdf_ji)
            pen = max(pen_ij, pen_ji)
            
            overlap_penalty += pen ** 2
            
    return boundary_penalty + overlap_penalty

def _objective_and_grad(params, outer_center, outer_side, outer_angle):
    """
    Objective and numerical gradient (L-BFGS-B can do numerical grad, 
    but providing it is faster. However, numerical grad of SDF is tricky.
    Let's rely on scipy's numerical differentiation first for robustness,
    then optimize if needed.
    """
    val = _objective(params, outer_center, outer_side, outer_angle)
    return val

def _initial_config_lattice(outer_side):
    """Generate a lattice-like initial configuration."""
    pos = []
    pos.append([0.0, 0.0, 0.0])
    
    # Hexagonal lattice spacing for side 1 is 2*sqrt(3)
    dist = 2.0 * np.sqrt(3) * 0.95
    
    for k in range(6):
        a = np.radians(60 * k)
        pos.append([dist * np.cos(a), dist * np.sin(a), 0.0])
        
    dist2 = dist * 1.5
    for k in range(4):
        a = np.radians(30 + 60 * k)
        pos.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])
        
    return pos

def _initial_config_random(outer_side):
    """Generate a random initial configuration inside the outer hexagon."""
    pos = []
    pos.append([0.0, 0.0, 0.0])
    
    for i in range(1, N):
        while True:
            x = random.uniform(-outer_side, outer_side)
            y = random.uniform(-outer_side, outer_side)
            
            inside = True
            for n in _get_face_normals(0.0):
                if x * n[0] + y * n[1] > outer_side * np.sqrt(3) / 2:
                    inside = False
                    break
            
            if inside:
                break
                
        angle = random.uniform(0, 360)
        pos.append([x, y, angle])
        
    return pos

def construct():
    """
    Main entry point.
    1. Generate diverse initial configurations.
    2. For each, run L-BFGS-B to minimize the SDF-based penalty.
    3. Check validity of the result.
    4. Keep the best valid configuration (minimizing outer side).
    """
    random.seed(42)
    np.random.seed(42)
    
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    best_side = float('inf')
    best_pos = None
    
    # We will try to find a valid configuration for a given side length S.
    # We start with a loose S and shrink it.
    # Or we can fix S and see if we can pack.
    # Better: Fix S to a target, optimize positions.
    # If valid, try smaller S.
    
    # Search space for S: 4.5 to 6.0
    # We'll use a coarse-to-fine search on S.
    
    # Initial candidates for S
    s_candidates = [5.5, 5.2, 5.0, 4.8, 4.6, 4.5]
    
    # For each S, try multiple initializations
    inits_per_s = [
        _initial_config_lattice(s),
        _initial_config_random(s),
        _initial_config_random(s)
    ]
    
    for s in s_candidates:
        if _remaining_time() < 10.0:
            break
            
        if s >= best_side:
            continue
            
        for init in inits_per_s:
            if _remaining_time() < 5.0:
                break
                
            params = np.array(init).flatten()
            
            # Define bounds for L-BFGS-B
            # x, y in [-s, s], angle in [0, 360]
            bounds = []
            for i in range(N):
                bounds.append((-s, s))
                bounds.append((-s, s))
                bounds.append((0, 360))
                
            # Optimize
            # Use L-BFGS-B
            # maxiter: limit based on time
            # We can't easily limit by time inside minimize, 
            # so we set a reasonable maxiter and check time afterwards.
            
            try:
                res = minimize(
                    _objective_and_grad, 
                    params, 
                    args=(outer_center, s, outer_angle),
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-6}
                )
            except Exception as e:
                continue
                
            if res.success:
                pos = res.x.reshape(N, 3)
                # Ensure angles are in [0, 360)
                pos[:, 2] = pos[:, 2] % 360
                
                # Check validity
                if _check_validity(pos, outer_center, s, outer_angle, tol=1e-5):
                    if s < best_side:
                        best_side = s
                        best_pos = pos.tolist()
                        # If we found a valid config for this s, we can break 
                        # and try smaller s in the next loop iteration (if any)
                        # But our loop is over fixed s_candidates.
                        # We can break inner loop.
                        break
                        
    # If we didn't find a valid config with the fixed candidates, 
    # or if we want to try to improve further, we can do a local search 
    # around the best found.
    
    # Fallback if no valid config found
    if best_pos is None:
        # Return a very loose configuration
        pos = []
        pos.append([0.0, 0.0, 0.0])
        for k in range(10):
            a = 2 * np.pi * k / 10
            r = 4.0
            pos.append([r * np.cos(a), r * np.sin(a), 0.0])
        side = _min_enclosing_side(pos, outer_center, outer_angle)
        return pos, outer_center, side, outer_angle
        
    return best_pos, outer_center, best_side, outer_angle

if __name__ == "__main__":
    res = construct()
    print(res)
