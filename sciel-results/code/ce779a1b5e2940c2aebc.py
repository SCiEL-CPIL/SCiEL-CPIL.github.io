# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 630.0
_SAFETY_MARGIN = 30.0


def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with side d and angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _precompute_base():
    """Precompute base vertices and normals for unit hexagon at angle 0."""
    base_verts = _hex_vertices(0, 0, 1.0, 0)  # (6, 2)
    # Edge normals: for edge from v[k] to v[(k+1)%6], normal is perpendicular
    # The outward normal for a regular hexagon at angle 0
    # Vertex angles: 0, 60, 120, 180, 240, 300
    # Edge k is from vertex k to vertex k+1
    # The outward normal for edge k points at angle 30 + 60*k degrees
    normal_angles = np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)
    
    # For each normal k, compute min and max projection of base_verts onto it
    # base_dots[v, k] = dot(base_verts[v], normals[k])
    base_dots = base_verts @ normals.T  # (6, 6)
    base_min = base_dots.min(axis=0)  # (6,)
    base_max = base_dots.max(axis=0)  # (6,)
    
    return base_verts, normals, base_min, base_max


# Precompute once
_BASE_VERTS, _NORMALS, _BASE_MIN, _BASE_MAX = _precompute_base()


def _check_separation_fast(positions):
    """
    Check if all pairs of unit hexagons at given positions are separated.
    Uses SAT with vectorized computation. Returns True if all separated.
    """
    n = len(positions)
    # center_dots[i, k] = dot(positions[i], normals[k])
    center_dots = positions @ _NORMALS.T  # (n, 6)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Projection intervals for hex i onto each normal
            proj_i_min = center_dots[i] + _BASE_MIN  # (6,)
            proj_i_max = center_dots[i] + _BASE_MAX  # (6,)
            proj_j_min = center_dots[j] + _BASE_MIN  # (6,)
            proj_j_max = center_dots[j] + _BASE_MAX  # (6,)
            
            # Check if any normal separates them
            sep1 = proj_i_max < proj_j_min - 1e-6
            sep2 = proj_j_max < proj_i_min - 1e-6
            if np.any(sep1) or np.any(sep2):
                continue
            else:
                return False
    return True


def _separation_violation_fast(positions):
    """
    Compute total separation violation using vectorized SAT.
    Returns 0 if all pairs are separated, positive value otherwise.
    """
    n = len(positions)
    center_dots = positions @ _NORMALS.T  # (n, 6)
    
    total_violation = 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            proj_i_min = center_dots[i] + _BASE_MIN
            proj_i_max = center_dots[i] + _BASE_MAX
            proj_j_min = center_dots[j] + _BASE_MIN
            proj_j_max = center_dots[j] + _BASE_MAX
            
            # Overlap on each axis
            overlap = np.minimum(proj_i_max, proj_j_max) - np.maximum(proj_i_min, proj_j_min)
            max_overlap = np.max(overlap)
            
            if max_overlap > 0:
                total_violation += max_overlap ** 2
    
    return total_violation


def _min_containing_hex_side_fast(centers, angle_deg=0.0):
    """
    Compute the minimum side length s of a regular hexagon centered at (0,0) 
    with given angle that contains all unit hexagons with given centers.
    """
    a_rad = np.radians(angle_deg)
    # Rotate centers to align with the outer hexagon's frame
    # The outer hexagon has normals at angles a_rad + 30 + 60*k
    # We need to project all vertices onto these normals and find max projection
    
    # All vertices: centers[i] + base_verts[v]
    # We need to project onto normals at angles a_rad + 30 + 60*k
    normal_angles = a_rad + np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)
    
    # For each vertex of each hexagon: proj = dot(center + base_vert, normal)
    # = dot(center, normal) + dot(base_vert, normal)
    center_dots = centers @ normals.T  # (12, 6)
    base_dots = _BASE_VERTS @ normals.T  # (6, 6)
    
    # All projections: (12, 6, 6) -> center_dots[:, k, :] + base_dots[:, k]
    # Max over vertices for each hexagon and each normal
    # proj[i, k, v] = center_dots[i, k] + base_dots[v, k]
    all_proj = center_dots[:, :, np.newaxis] + base_dots[np.newaxis, :, :]  # (12, 6, 6)
    max_proj_per_hex_normal = np.max(all_proj, axis=2)  # (12, 6)
    max_proj = np.max(max_proj_per_hex_normal)  # scalar
    
    s = (2.0 / np.sqrt(3.0)) * max_proj
    return s


def _objective(params):
    """
    Objective: minimize outer hexagon side length with penalty for overlaps.
    params: (25,) array. First 24 are (x1, y1, ..., x12, y12). Last is angle_deg.
    """
    positions = params[:24].reshape(12, 2)
    angle_deg = params[24]
    
    # Check separation - if violated, return large penalty
    if not _check_separation_fast(positions):
        # Compute violation magnitude for gradient information
        violation = _separation_violation_fast(positions)
        return 100.0 + 10.0 * violation
    
    # Compute min containing hexagon side
    s = _min_containing_hex_side_fast(positions, angle_deg)
    return s


def _make_hex_cluster():
    """
    Create a compact hexagonal cluster of 12 unit hexagons.
    Using the natural hexagonal lattice arrangement.
    """
    # For flat-top hexagons with side 1:
    # Horizontal spacing: 1.5, Vertical spacing: sqrt(3)
    # This creates a honeycomb-like arrangement
    sqrt3 = np.sqrt(3)
    
    positions = np.array([
        [0.0, 0.0],           # center
        [1.5, sqrt3/2],       # ring 1
        [1.5, -sqrt3/2],
        [0.0, sqrt3],
        [0.0, -sqrt3],
        [-1.5, sqrt3/2],
        [-1.5, -sqrt3/2],
        # ring 2 (partial for 12 total)
        [3.0, 0.0],
        [2.25, 3*sqrt3/2],
        [0.75, 2*sqrt3],
        [-1.5, 3*sqrt3/2],
        [-3.0, 0.0],
    ])
    
    return positions


def _make_circular(r=2.0):
    """Generate positions on a circle."""
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    positions = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    return positions


def _make_spread():
    """A more spread out initial configuration."""
    sqrt3 = np.sqrt(3)
    positions = np.array([
        [0.0, 0.0],
        [-2.0, 0.0],
        [2.0, 0.0],
        [-1.0, 1.732],
        [1.0, 1.732],
        [-1.0, -1.732],
        [1.0, -1.732],
        [-3.0, 1.732],
        [3.0, 1.732],
        [-3.0, -1.732],
        [3.0, -1.732],
        [0.0, -3.464],
    ])
    return positions


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), (x, y, angle_deg)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_deg)
        outer_hex_side_length: float
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Emergency fallback
        centers = _make_hex_cluster()
        inner = np.zeros((12, 3))
        inner[:, 0] = centers[:, 0]
        inner[:, 1] = centers[:, 1]
        inner[:, 2] = 0.0
        outer = np.array([0.0, 0.0, 0.0])
        s = _min_containing_hex_side_fast(centers, 0.0)
        return inner, outer, s
    
    rng = np.random.default_rng(42)
    
    best_s = np.inf
    best_params = None
    
    # Initial configurations
    inits = []
    
    # Hexagonal cluster (should be valid)
    inits.append(_make_hex_cluster())
    
    # Spread configuration
    inits.append(_make_spread())
    
    # Circular arrangements
    for r in [1.8, 2.2, 2.5]:
        inits.append(_make_circular(r))
    
    # Perturbed hex cluster
    base = _make_hex_cluster()
    for scale in [0.3, 0.5, 0.8]:
        inits.append(base + rng.normal(0, scale, size=base.shape))
    
    # Random
    for _ in range(3):
        inits.append(rng.normal(0, 2.5, size=(12, 2)))
    
    # Try each initial config with multiple angle starts
    angle_starts = [0, 30, 15, -15, 10, -10]
    
    for init_idx, init_pos in enumerate(inits):
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - _SAFETY_MARGIN:
            break
        
        for init_angle in angle_starts:
            elapsed = time.time() - _START_TIME
            if elapsed > _DEADLINE - _SAFETY_MARGIN:
                break
            
            init_params = np.zeros(25)
            init_params[:24] = init_pos.flatten()
            init_params[24] = init_angle
            
            # Quick check: is this initial config valid?
            if _check_separation_fast(init_pos):
                s_init = _min_containing_hex_side_fast(init_pos, init_angle)
                if s_init < best_s:
                    best_s = s_init
                    best_params = init_params.copy()
            
            # Run Nelder-Mead optimization
            try:
                result = minimize(
                    _objective,
                    init_params,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 5000,
                        'maxfev': 50000,
                        'xatol': 1e-9,
                        'fatol': 1e-11,
                        'adaptive': True,
                    }
                )
            except Exception:
                continue
            
            if result is not None and result.fun < best_s:
                candidate = result.x
                positions = candidate[:24].reshape(12, 2)
                angle = candidate[24]
                if _check_separation_fast(positions):
                    s_cand = _min_containing_hex_side_fast(positions, angle)
                    if s_cand < best_s:
                        best_s = s_cand
                        best_params = candidate.copy()
    
    # If we haven't found a valid solution, use fallback
    if best_params is None or best_s > 100:
        best_pos = _make_hex_cluster()
        best_angle = 0.0
        best_s = _min_containing_hex_side_fast(best_pos, best_angle)
        best_params = np.zeros(25)
        best_params[:24] = best_pos.flatten()
        best_params[24] = best_angle
    
    # Final high-precision polish pass
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - _SAFETY_MARGIN:
        try:
            result_final = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 15000,
                    'maxfev': 150000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True,
                }
            )
        except Exception:
            result_final = None
        
        if result_final is not None:
            candidate = result_final.x
            positions = candidate[:24].reshape(12, 2)
            angle = candidate[24]
            if _check_separation_fast(positions):
                s_cand = _min_containing_hex_side_fast(positions, angle)
                if s_cand < best_s:
                    best_s = s_cand
                    best_params = candidate.copy()
    
    # Second polish with tighter tolerance if time permits
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - _SAFETY_MARGIN and best_s < 100:
        try:
            result_polish2 = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 20000,
                    'maxfev': 200000,
                    'xatol': 1e-11,
                    'fatol': 1e-13,
                    'adaptive': True,
                }
            )
        except Exception:
            result_polish2 = None
        
        if result_polish2 is not None:
            candidate = result_polish2.x
            positions = candidate[:24].reshape(12, 2)
            angle = candidate[24]
            if _check_separation_fast(positions):
                s_cand = _min_containing_hex_side_fast(positions, angle)
                if s_cand < best_s:
                    best_s = s_cand
                    best_params = candidate.copy()
    
    # Build output
    positions = best_params[:24].reshape(12, 2)
    angle_deg = best_params[24]
    
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = positions[i, 0]
        inner_hex_data[i, 1] = positions[i, 1]
        inner_hex_data[i, 2] = 0.0  # All inner hexagons at angle 0
    
    outer_hex_data = np.array([0.0, 0.0, angle_deg])
    outer_hex_side_length = best_s
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
