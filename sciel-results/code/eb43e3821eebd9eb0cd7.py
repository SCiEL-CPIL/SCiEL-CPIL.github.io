# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_START_TIME = time.time()
_TIME_LIMIT = 580.0  # safety margin before 630s kill


def _hex_vertices(cx, cy, s, a_deg):
    """Return (6,2) array of vertices for a regular hexagon."""
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(th), cy + s * np.sin(th)])


def _edge_normals(vertices):
    """Return (6,2) array of outward-pointing unit edge normals for a convex polygon."""
    n = len(vertices)
    norms = np.zeros((n, 2))
    centroid = np.mean(vertices, axis=0)
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        normal /= np.linalg.norm(normal)
        mid = (p1 + p2) / 2.0
        if np.dot(normal, mid - centroid) < 0:
            normal = -normal
        norms[i] = normal
    return norms


def _min_gap(v1, v2, n1, n2):
    """
    Minimum separation between two convex hexagons using SAT.
    v1, v2: (6,2) vertex arrays
    n1, n2: (6,2) outward normal arrays
    Returns the minimum gap (positive if separated, 0 if touching/overlapping).
    """
    all_norms = np.vstack([n1, n2])  # (12, 2)
    min_gap = float('inf')
    for k in range(12):
        n = all_norms[k]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2:
            gap = min2 - max1
        elif max2 < min1:
            gap = min1 - max2
        else:
            gap = 0.0
        if gap < min_gap:
            min_gap = gap
    return min_gap


def _separation(h1, h2):
    """Return min gap between two unit hexagons given as (x, y, angle_deg)."""
    v1 = _hex_vertices(h1[0], h1[1], 1.0, h1[2])
    v2 = _hex_vertices(h2[0], h2[1], 1.0, h2[2])
    n1 = _edge_normals(v1)
    n2 = _edge_normals(v2)
    return _min_gap(v1, v2, n1, n2)


def _check_separations(inner):
    """Check all pairs. Return (min_gap, valid) where valid means all gaps > 1e-6."""
    n = len(inner)
    min_gap = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            gap = _separation(inner[i], inner[j])
            if gap < min_gap:
                min_gap = gap
    valid = min_gap > 1e-6
    return min_gap, valid


def _outer_side_for(inner, outer_center=np.array([0.0, 0.0]), outer_angle=0.0):
    """
    Find minimum outer side length such that all inner hex vertices are inside.
    Outer hexagon is centered at outer_center with given angle.
    """
    normal_angles = np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6,2)
    
    all_verts = []
    for h in inner:
        v = _hex_vertices(h[0], h[1], 1.0, h[2])
        v = v - outer_center
        all_verts.append(v)
    all_verts = np.vstack(all_verts)  # (66, 2)
    
    proj = all_verts @ normals.T  # (66, 6)
    max_proj_per_normal = proj.max(axis=0)  # (6,)
    required_s = (2.0 / np.sqrt(3.0)) * max_proj_per_normal
    return float(required_s.max())


def _objective(x, n=11):
    """
    Objective: minimize outer side length + penalty for overlaps.
    x: flat array of 11*3 values
    """
    inner = x.reshape(n, 3)
    
    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            gap = _separation(inner[i], inner[j])
            if gap < 1e-6:
                pen += (1e-6 - gap) ** 2 * 1000.0
    
    side = _outer_side_for(inner)
    return side + pen


def _initial_packing():
    """
    Analytical initial packing: 1 center + 6 ring + 4 in the 'corner' gaps.
    Unit hexagons in a hexagonal lattice.
    """
    d = np.sqrt(3.0)  # center-to-center distance for touching hexagons
    angles = np.deg2rad(60.0 * np.arange(6))
    dirs = np.column_stack([d * np.cos(angles), d * np.sin(angles)])
    
    inner = np.zeros((11, 3))
    inner[0] = [0.0, 0.0, 0.0]
    for k in range(6):
        inner[k + 1] = [dirs[k, 0], dirs[k, 1], 0.0]
    
    r2 = 2.0 * d
    ring2_angles = np.deg2rad([30.0, 90.0, 150.0, 270.0])
    for k in range(4):
        idx = 7 + k
        inner[idx, 0] = r2 * np.cos(ring2_angles[k])
        inner[idx, 1] = r2 * np.sin(ring2_angles[k])
        inner[idx, 2] = 0.0
    
    return inner


def _try_optimize(s0, best_state, n, max_iter=2000):
    """Try to optimize from start point s0, update best_state if improved."""
    try:
        res = minimize(_objective, s0.flatten(), method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-5, 'fatol': 1e-7,
                                'adaptive': True})
        x = res.x.reshape(n, 3)
        
        min_gap, valid = _check_separations(x)
        if valid:
            side = _outer_side_for(x)
            if side < best_state['side']:
                best_state['side'] = side
                best_state['x'] = x.copy()
    except Exception:
        pass
    return best_state


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    n = 11
    inner0 = _initial_packing()
    
    min_gap, valid = _check_separations(inner0)
    if not valid:
        inner0 = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.165, 0],
            [1.25, 2.165, 0],
            [-1.25, -2.165, 0],
            [1.25, -2.165, 0],
            [-3.75, 2.165, 0],
            [3.75, 2.165, 0],
            [-3.75, -2.165, 0],
            [3.75, -2.165, 0],
        ])
        min_gap, valid = _check_separations(inner0)
    
    best_state = {'x': inner0.copy(), 'side': _outer_side_for(inner0)}
    
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points with angle perturbations
    starts = [inner0.copy()]
    for i in range(12):
        jitter = rng.normal(0, 0.5, size=(n, 3))
        angle_jitter = rng.uniform(-30, 30, size=n)
        jitter[:, 2] = angle_jitter
        starts.append(inner0 + jitter)
    
    # Try each start
    for s0 in starts:
        if time.time() - _START_TIME > _TIME_LIMIT - 60:
            break
        best_state = _try_optimize(s0, best_state, n)
    
    # Additional refinement: try to optimize angles of best solution
    if time.time() - _START_TIME < _TIME_LIMIT - 30:
        x = best_state['x'].copy()
        for _ in range(5):
            if time.time() - _START_TIME > _TIME_LIMIT - 30:
                break
            x_perturbed = x.copy()
            x_perturbed[:, 2] += rng.normal(0, 5.0, size=n)
            best_state = _try_optimize(x_perturbed, best_state, n)
    
    # Final verification
    min_gap, valid = _check_separations(best_state['x'])
    if not valid:
        best_state['x'] = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.165, 0],
            [1.25, 2.165, 0],
            [-1.25, -2.165, 0],
            [1.25, -2.165, 0],
            [-3.75, 2.165, 0],
            [3.75, 2.165, 0],
            [-3.75, -2.165, 0],
            [3.75, -2.165, 0],
        ])
        best_state['side'] = 8.0
    
    side = best_state['side']
    outer = np.array([0.0, 0.0, 0.0])
    return best_state['x'], outer, float(side)


# EVOLVE-BLOCK-END
