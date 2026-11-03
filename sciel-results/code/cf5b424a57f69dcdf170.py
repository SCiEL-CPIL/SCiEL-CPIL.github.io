import math
import time
import random
import numpy as np
from scipy.optimize import minimize_scalar, differential_evolution

N = 12
DEADLINE = 140.0  # Safety margin before 150s wall-clock kill

# Precompute unit hexagon vertices and edge normals (angle 0)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i + 1) % 6, 0] - _UNIT_HEX_VERTS[_i, 1]
    _ey = _UNIT_HEX_VERTS[(_i + 1) % 6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx * _nx + _ny * _ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

_UX = _UNIT_HEX_VERTS[:, 0]
_UY = _UNIT_HEX_VERTS[:, 1]
_ENX = _EDGE_NORMALS[:, 0]
_ENY = _EDGE_NORMALS[:, 1]
_SQRT3 = math.sqrt(3)


def _hexagons_disjoint_fast(h1, h2):
    """Check if two hexagons are disjoint (non-overlapping)."""
    dx = h1[0] - h2[0]
    dy = h1[1] - h2[1]
    dist_sq = dx * dx + dy * dy
    if dist_sq > 4.5:  # Maximum distance where hexagons could overlap
        return True
    a1 = h1[2]
    a2 = h2[2]
    c1, s1 = math.cos(a1), math.sin(a1)
    c2, s2 = math.cos(a2), math.sin(a2)
    
    # Rotate hexagon 1 vertices and normals
    v1x = h1[0] + _UX * c1 - _UY * s1
    v1y = h1[1] + _UX * s1 + _UY * c1
    en1x = _ENX * c1 - _ENY * s1
    en1y = _ENX * s1 + _ENY * c1
    
    # Rotate hexagon 2 vertices and normals
    v2x = h2[0] + _UX * c2 - _UY * s2
    v2y = h2[1] + _UX * s2 + _UY * c2
    en2x = _ENX * c2 - _ENY * s2
    en2y = _ENX * s2 + _ENY * c2
    
    # Check separation along all 12 potential separating axes
    for i in range(6):
        nx, ny = en1x[i], en1y[i]
        p1 = v1x * nx + v1y * ny
        p2 = v2x * nx + v2y * ny
        if p1.max() < p2.min() or p2.max() < p1.min():
            return True
    for i in range(6):
        nx, ny = en2x[i], en2y[i]
        p1 = v1x * nx + v1y * ny
        p2 = v2x * nx + v2y * ny
        if p1.max() < p2.min() or p2.max() < p1.min():
            return True
    return False


def _is_valid(positions, angles_deg):
    """Check if all hexagons are pairwise disjoint."""
    angles_rad = np.radians(angles_deg)
    for i in range(N):
        for j in range(i + 1, N):
            h1 = (positions[i, 0], positions[i, 1], angles_rad[i])
            h2 = (positions[j, 0], positions[j, 1], angles_rad[j])
            if not _hexagons_disjoint_fast(h1, h2):
                return False
    return True


def _compute_min_outer_side(positions, angles_deg):
    """Compute the minimum side length of an outer hexagon enclosing all inner hexagons."""
    angles_rad = np.radians(angles_deg)
    c = np.cos(angles_rad)
    s = np.sin(angles_rad)
    
    # Rotate all hexagon vertices
    rx = _UX[None, :] * c[:, None] - _UY[None, :] * s[:, None]
    ry = _UX[None, :] * s[:, None] + _UY[None, :] * c[:, None]
    verts_x = rx + positions[:, 0:1]
    verts_y = ry + positions[:, 1:2]
    
    # Flatten all vertices
    verts_flat = np.stack((verts_x, verts_y), axis=-1).reshape(-1, 2)
    center = np.mean(verts_flat, axis=0)
    offsets = verts_flat - center
    
    def eval_phi(phi_deg):
        phi = np.radians(phi_deg)
        normal_angles = phi + np.pi / 6 + np.arange(6) * np.pi / 3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / _SQRT3) * max_projs
        return np.max(side_vals)
    
    # Vectorized coarse scan over phi from 0 to 60 degrees
    phi_vals = np.radians(np.arange(0, 60, 1))
    normal_angles = phi_vals[:, None] + np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
    abs_proj = np.abs(proj)
    max_projs = np.max(abs_proj, axis=0)  # Shape: (60, 6)
    side_vals = (2.0 / _SQRT3) * max_projs
    best_side = np.max(side_vals).min()
    best_idx = np.argmin(np.max(side_vals, axis=1))
    best_phi = float(best_idx)
    
    # Fine tune around best
    try:
        lo = max(0.0, best_phi - 1.0)
        hi = min(60.0, best_phi + 1.0)
        res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                              options={'xatol': 1e-10, 'maxiter': 100})
        if res.success and res.fun < best_side:
            best_side = float(res.fun)
            best_phi = float(res.x)
    except Exception:
        pass
    
    return float(best_side), float(best_phi), float(center[0]), float(center[1])


def _objective(params):
    """Objective function for differential evolution: outer hexagon side + overlap penalty."""
    # params: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11]
    positions = np.column_stack((params[0::3], params[1::3]))
    angles_deg = params[2::3]
    
    # Compute outer side length
    side, _, _, _ = _compute_min_outer_side(positions, angles_deg)
    
    # Compute overlap penalty
    penalty = 0.0
    angles_rad = np.radians(angles_deg)
    for i in range(N):
        for j in range(i + 1, N):
            h1 = (positions[i, 0], positions[i, 1], angles_rad[i])
            h2 = (positions[j, 0], positions[j, 1], angles_rad[j])
            if not _hexagons_disjoint_fast(h1, h2):
                # They overlap; add a penalty based on penetration depth
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                # Approximate penetration: assume ~sqrt(3) is the "touching" distance for parallel hexes
                penetration = max(0, _SQRT3 - dist)
                penalty += 50.0 * penetration * penetration
    
    return side + penalty


def _make_honeycomb():
    """Create a honeycomb-like initial configuration."""
    a1 = np.array([_SQRT3, 0.0])
    a2 = np.array([_SQRT3 / 2.0, 1.5])
    
    # 12 hexagons in a compact honeycomb pattern
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    for idx, (i, j) in enumerate(lattice_coords):
        pos[idx] = i * a1 + j * a2
        ang[idx] = 0.0
    return pos, ang


def _make_ring():
    """Create a ring configuration with 6 inner and 6 outer hexagons."""
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    for i in range(6):
        theta = i * np.pi / 3
        pos[i, 0] = 2.5 * _SQRT3 * np.cos(theta)
        pos[i, 1] = 2.5 * _SQRT3 * np.sin(theta)
    for i in range(6):
        theta = (i + 0.5) * np.pi / 3
        pos[6 + i, 0] = 1.2 * _SQRT3 * np.cos(theta)
        pos[6 + i, 1] = 1.2 * _SQRT3 * np.sin(theta)
    return pos, ang


def _make_two_rows():
    """Create a two-row configuration."""
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    for i in range(6):
        pos[i, 0] = i * _SQRT3
        pos[i, 1] = 0.0
    for i in range(6):
        pos[6 + i, 0] = i * _SQRT3 + _SQRT3 / 2
        pos[6 + i, 1] = 1.5
    return pos, ang


def _make_compact_cluster():
    """Create a compact cluster of 12 hexagons."""
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    # Central hexagon
    pos[0] = [0, 0]
    # 6 around it
    for i in range(6):
        theta = i * np.pi / 3
        pos[i + 1, 0] = _SQRT3 * np.cos(theta)
        pos[i + 1, 1] = _SQRT3 * np.sin(theta)
    # 5 more in a compact arrangement
    positions_extra = [
        [2 * _SQRT3, 0],
        [-_SQRT3, 3 * 1.5],
        [_SQRT3, 3 * 1.5],
        [-2 * _SQRT3, 0],
        [0, -3 * 1.5],
    ]
    for i, p in enumerate(positions_extra):
        pos[i + 7] = p
    return pos, ang


def _make_random():
    """Create a random initial configuration."""
    pos = np.random.uniform(-3, 3, (N, 2))
    ang = np.random.uniform(-30, 30, N)
    return pos, ang


def _local_search(pos, ang, cur_side, max_iters=500, time_budget=5.0):
    """Simple local search to polish a configuration."""
    t_start = time.time()
    step = 0.15
    best_pos = pos.copy()
    best_ang = ang.copy()
    best_side = cur_side
    
    for _ in range(max_iters):
        if time.time() - t_start > time_budget:
            break
        improved = False
        for idx in range(N):
            for dim in range(3):
                for direction in [1, -1]:
                    new_pos = pos.copy()
                    new_ang = ang.copy()
                    if dim == 0:
                        new_pos[idx, 0] += direction * step
                    elif dim == 1:
                        new_pos[idx, 1] += direction * step
                    else:
                        new_ang[idx] += direction * step
                    
                    # Check validity
                    if _is_valid(new_pos, new_ang):
                        new_side, new_phi, new_cx, new_cy = _compute_min_outer_side(new_pos, new_ang)
                        if new_side < best_side - 1e-10:
                            pos = new_pos
                            ang = new_ang
                            cur_side = new_side
                            best_pos = pos.copy()
                            best_ang = ang.copy()
                            best_side = cur_side
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if not improved:
            step *= 0.85
            if step < 0.005:
                break
    
    return best_pos, best_ang, best_side


def construct():
    t0 = time.time()
    
    # Try multiple initializations
    inits = [
        _make_honeycomb(),
        _make_ring(),
        _make_two_rows(),
        _make_compact_cluster(),
        _make_random(),
        _make_random(),
    ]
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    for init_idx, (pos0, ang0) in enumerate(inits):
        if time.time() - t0 > DEADLINE - 15.0:
            break
        
        # Check initial validity
        if not _is_valid(pos0, ang0):
            # Perturb to make valid
            for _ in range(200):
                if time.time() - t0 > DEADLINE - 15.0:
                    break
                pos0 = pos0 + np.random.uniform(-0.15, 0.15, (N, 2))
                ang0 = ang0 + np.random.uniform(-5, 5, N)
                if _is_valid(pos0, ang0):
                    break
        
        if not _is_valid(pos0, ang0):
            continue
        
        # Evaluate initial
        s0, phi0, cx0, cy0 = _compute_min_outer_side(pos0, ang0)
        if s0 < best_side:
            best_side = s0
            best_positions = pos0.copy()
            best_angles = ang0.copy()
            best_phi = phi0
            best_cx = cx0
            best_cy = cy0
        
        # Use differential evolution to improve
        if time.time() - t0 > DEADLINE - 25.0:
            break
        
        # Flatten parameters
        params0 = np.zeros(N * 3)
        for i in range(N):
            params0[i * 3] = pos0[i, 0]
            params0[i * 3 + 1] = pos0[i, 1]
            params0[i * 3 + 2] = ang0[i]
        
        # Bounds
        bounds = [(-5, 5), (-5, 5), (-45, 45)] * N
        
        # Run differential evolution with time limit
        t_de_start = time.time()
        max_de_time = min(15.0, DEADLINE - 20.0 - (time.time() - t0))
        
        if max_de_time > 5.0:
            try:
                res = differential_evolution(
                    _objective, bounds, 
                    maxiter=1000, tol=1e-6,
                    popsize=15, seed=42 + init_idx,
                    updating='immediate',
                    polish=False,
                    init='sobol',
                    maxfees=2000
                )
                if time.time() - t0 > DEADLINE - 15.0:
                    break
                    
                # Extract solution
                new_pos = np.column_stack((res.x[0::3], res.x[1::3]))
                new_ang = res.x[2::3]
                
                # Check validity
                if _is_valid(new_pos, new_ang):
                    new_side, new_phi, new_cx, new_cy = _compute_min_outer_side(new_pos, new_ang)
                    if new_side < best_side:
                        best_side = new_side
                        best_positions = new_pos.copy()
                        best_angles = new_ang.copy()
                        best_phi = new_phi
                        best_cx = new_cx
                        best_cy = new_cy
            except Exception:
                pass
        
        # Local search polish
        if time.time() - t0 > DEADLINE - 10.0:
            break
        
        if best_positions is not None and best_side < 100.0:
            t_ls_start = time.time()
            max_ls_time = min(5.0, DEADLINE - 8.0 - (time.time() - t0))
            if max_ls_time > 2.0:
                pos_ls, ang_ls, side_ls = _local_search(
                    best_positions, best_angles, best_side, 
                    max_iters=300, time_budget=max_ls_time
                )
                if side_ls < best_side:
                    best_side = side_ls
                    best_positions = pos_ls.copy()
                    best_angles = ang_ls.copy()
                    # Recompute phi, center
                    _, best_phi, best_cx, best_cy = _compute_min_outer_side(best_positions, best_angles)
    
    # Final verification
    if best_positions is None or not _is_valid(best_positions, best_angles):
        pos0, ang0 = _make_honeycomb()
        best_positions = pos0.copy()
        best_angles = ang0.copy()
        best_side, best_phi, best_cx, best_cy = _compute_min_outer_side(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
    
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
