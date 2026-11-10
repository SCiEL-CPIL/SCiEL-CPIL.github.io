# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 630.0
_SAFETY_MARGIN = 30.0


def _hex_vertices(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _min_containing_hex_side(centers):
    """
    Compute the minimum side length s of a regular hexagon centered at (0,0), angle 0,
    that contains all unit hexagons with given centers.
    """
    n = len(centers)
    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)
    
    # All vertices for all hexagons: (n, 6, 2)
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        verts[i, :, 0] = centers[i, 0] + base_cos
        verts[i, :, 1] = centers[i, 1] + base_sin

    # Outward normals at angles 30, 90, 150, 210, 270, 330
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)
    
    verts_flat = verts.reshape(-1, 2)  # (36, 2)
    projections = normals @ verts_flat.T  # (6, 36)
    max_proj_per_normal = np.max(projections, axis=1)  # (6,)
    max_proj = np.max(max_proj_per_normal)  # scalar
    
    s = (2.0 / np.sqrt(3.0)) * max_proj
    return s


def _separation_violation(centers):
    """
    Compute total separation violation (how much pairs overlap).
    Returns 0 if all pairs are separated, positive value otherwise.
    """
    n = len(centers)
    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)
    
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)
    
    all_verts = np.zeros((n, 6, 2))
    for i in range(n):
        all_verts[i, :, 0] = centers[i, 0] + base_cos
        all_verts[i, :, 1] = centers[i, 1] + base_sin
    
    total_violation = 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            vi = all_verts[i]
            vj = all_verts[j]
            
            proj_i = vi @ normals.T
            proj_j = vj @ normals.T
            
            min_i = np.min(proj_i, axis=1)
            max_i = np.max(proj_i, axis=1)
            min_j = np.min(proj_j, axis=1)
            max_j = np.max(proj_j, axis=1)
            
            overlaps = np.minimum(max_i, max_j) - np.maximum(min_i, min_j)
            max_overlap = np.max(overlaps)
            
            if max_overlap > 0:
                total_violation += max_overlap ** 2
    
    return total_violation


def _check_separation(centers, tol=1e-6):
    """Check if all pairs of unit hexagons are separated using SAT."""
    n = len(centers)
    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)
    
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    all_verts = np.zeros((n, 6, 2))
    for i in range(n):
        all_verts[i, :, 0] = centers[i, 0] + base_cos
        all_verts[i, :, 1] = centers[i, 1] + base_sin
    
    for i in range(n):
        for j in range(i + 1, n):
            vi = all_verts[i]
            vj = all_verts[j]
            
            proj_i = vi @ normals.T
            proj_j = vj @ normals.T
            
            min_i = np.min(proj_i, axis=1)
            max_i = np.max(proj_i, axis=1)
            min_j = np.min(proj_j, axis=1)
            max_j = np.max(proj_j, axis=1)
            
            separated = np.any(max_i < min_j - tol) or np.any(max_j < min_i - tol)
            if not separated:
                return False
    return True


def _force_directed_relax(centers, n_iters=500, dt=0.05, repulsion_strength=1.0):
    """
    Force-directed relaxation: push overlapping hexagons apart using a simple
    repulsive force model based on separation violation gradients.
    """
    centers = centers.copy().astype(float)
    n = len(centers)
    
    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)
    
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    for it in range(n_iters):
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - _SAFETY_MARGIN:
            break
        
        forces = np.zeros_like(centers)
        
        for i in range(n):
            for j in range(i + 1, n):
                vi = centers[i] + np.column_stack([base_cos, base_sin])
                vj = centers[j] + np.column_stack([base_cos, base_sin])
                
                proj_i = vi @ normals.T
                proj_j = vj @ normals.T
                
                min_i = np.min(proj_i, axis=1)
                max_i = np.max(proj_i, axis=1)
                min_j = np.min(proj_j, axis=1)
                max_j = np.max(proj_j, axis=1)
                
                # For each normal axis, compute overlap and force direction
                for k in range(6):
                    overlap = min(max_i[k], max_j[k]) - max(min_i[k], min_j[k])
                    if overlap > 0:
                        # Push apart along this normal
                        n_dir = normals[k]
                        # Determine which hex is "on top"
                        if max_i[k] > max_j[k]:
                            forces[i] += overlap * n_dir * repulsion_strength
                            forces[j] -= overlap * n_dir * repulsion_strength
                        else:
                            forces[j] += overlap * n_dir * repulsion_strength
                            forces[i] -= overlap * n_dir * repulsion_strength
        
        centers += dt * forces
        # Decay step size
        dt *= 0.995
    
    return centers


def _barycentric_scale(centers, alpha):
    """Scale centers toward their centroid by factor alpha."""
    centroid = np.mean(centers, axis=0)
    return centroid + alpha * (centers - centroid)


def _make_hex_lattice_12():
    """Create a hexagonal lattice arrangement of 12 hexagons."""
    sqrt3 = np.sqrt(3)
    # Standard hex packing: center + 6 in ring 1 + 5 in ring 2
    return np.array([
        [0.0, 0.0],
        [1.5, 0.866025],
        [1.5, -0.866025],
        [0.0, 1.732051],
        [0.0, -1.732051],
        [-1.5, 0.866025],
        [-1.5, -0.866025],
        [3.0, 0.0],
        [2.25, 2.598076],
        [0.75, 3.464102],
        [-1.5, 2.598076],
        [-3.0, 0.0],
    ])


def _make_compact_initial():
    """A more compact initial arrangement."""
    sqrt3 = np.sqrt(3)
    return np.array([
        [0.0, 0.0],
        [sqrt3, 0.0],
        [sqrt3/2, 1.5],
        [-sqrt3/2, 1.5],
        [-sqrt3, 0.0],
        [-sqrt3/2, -1.5],
        [sqrt3/2, -1.5],
        [1.5*sqrt3, 1.5],
        [1.5*sqrt3, -1.5],
        [0.0, 3.0],
        [-1.5*sqrt3, 1.5],
        [-1.5*sqrt3, -1.5],
    ])


def _make_spread_initial():
    """A spread-out initial arrangement."""
    return np.array([
        [0.0, 0.0],
        [-2.5, 0.0],
        [2.5, 0.0],
        [-1.25, 2.165],
        [1.25, 2.165],
        [-1.25, -2.165],
        [1.25, -2.165],
        [-3.75, 2.165],
        [3.75, 2.165],
        [-3.75, -2.165],
        [3.75, -2.165],
        [0.0, -4.33],
    ])


def _objective(params):
    """Objective: minimize the containing hexagon side length with penalty for overlaps."""
    centers = params.reshape(12, 2)
    s = _min_containing_hex_side(centers)
    violation = _separation_violation(centers)
    penalty = 100.0 * violation
    return s + penalty


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), (x, y, angle_deg)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_deg)
        outer_hex_side_length: float
    """
    best_centers = None
    best_s = np.inf
    
    rng = np.random.default_rng(42)
    
    initial_configs = [
        _make_hex_lattice_12(),
        _make_compact_initial(),
        _make_spread_initial(),
    ]
    
    for init_idx, init_centers in enumerate(initial_configs):
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - _SAFETY_MARGIN:
            break
        
        # Phase 1: Force-directed relaxation to resolve overlaps
        relaxed = _force_directed_relax(init_centers, n_iters=800, dt=0.08, repulsion_strength=1.0)
        
        if _check_separation(relaxed):
            s_relaxed = _min_containing_hex_side(relaxed)
            if s_relaxed < best_s:
                best_s = s_relaxed
                best_centers = relaxed.copy()
        
        # Phase 2: Barycentric scaling + re-relaxation
        # Try multiple scale factors
        for alpha in [0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.85, 0.82, 0.80]:
            elapsed = time.time() - _START_TIME
            if elapsed > _DEADLINE - _SAFETY_MARGIN:
                break
            
            scaled = _barycentric_scale(relaxed, alpha)
            # Re-relax after scaling
            re_relaxed = _force_directed_relax(scaled, n_iters=400, dt=0.05, repulsion_strength=0.8)
            
            if _check_separation(re_relaxed):
                s_scaled = _min_containing_hex_side(re_relaxed)
                if s_scaled < best_s:
                    best_s = s_scaled
                    best_centers = re_relaxed.copy()
        
        # Phase 3: Nelder-Mead refinement from best found so far
        if best_centers is not None:
            elapsed = time.time() - _START_TIME
            if elapsed < _DEADLINE - _SAFETY_MARGIN:
                x0 = best_centers.flatten()
                try:
                    result = minimize(
                        _objective,
                        x0,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 8000,
                            'maxfev': 80000,
                            'xatol': 1e-10,
                            'fatol': 1e-12,
                            'adaptive': True,
                        }
                    )
                except Exception:
                    result = None
                
                if result is not None:
                    candidate = result.x.reshape(12, 2)
                    if _check_separation(candidate):
                        s_cand = _min_containing_hex_side(candidate)
                        if s_cand < best_s:
                            best_s = s_cand
                            best_centers = candidate.copy()
        
        # Phase 4: Random restarts with force-directed + scaling
        n_restarts = 3
        for r in range(n_restarts):
            elapsed = time.time() - _START_TIME
            if elapsed > _DEADLINE - _SAFETY_MARGIN:
                break
            
            noise_scale = 0.3 * (1.0 + r * 0.5)
            perturbed = init_centers + rng.normal(0, noise_scale, size=init_centers.shape)
            
            relaxed_r = _force_directed_relax(perturbed, n_iters=500, dt=0.06, repulsion_strength=1.0)
            
            if _check_separation(relaxed_r):
                s_r = _min_containing_hex_side(relaxed_r)
                if s_r < best_s:
                    best_s = s_r
                    best_centers = relaxed_r.copy()
            
            # Scale and re-relax
            for alpha in [0.95, 0.92, 0.89, 0.86]:
                elapsed = time.time() - _START_TIME
                if elapsed > _DEADLINE - _SAFETY_MARGIN:
                    break
                
                scaled_r = _barycentric_scale(relaxed_r, alpha)
                re_relaxed_r = _force_directed_relax(scaled_r, n_iters=300, dt=0.04, repulsion_strength=0.7)
                
                if _check_separation(re_relaxed_r):
                    s_sr = _min_containing_hex_side(re_relaxed_r)
                    if s_sr < best_s:
                        best_s = s_sr
                        best_centers = re_relaxed_r.copy()
    
    # Fallback
    if best_centers is None:
        best_centers = _make_spread_initial().copy()
        best_s = _min_containing_hex_side(best_centers)
    
    # Final Nelder-Mead polish
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - _SAFETY_MARGIN:
        x0_final = best_centers.flatten()
        try:
            result_final = minimize(
                _objective,
                x0_final,
                method='Nelder-Mead',
                options={
                    'maxiter': 12000,
                    'maxfev': 120000,
                    'xatol': 1e-11,
                    'fatol': 1e-13,
                    'adaptive': True,
                }
            )
        except Exception:
            result_final = None
        
        if result_final is not None:
            candidate = result_final.x.reshape(12, 2)
            if _check_separation(candidate):
                s_cand = _min_containing_hex_side(candidate)
                if s_cand < best_s:
                    best_s = s_cand
                    best_centers = candidate.copy()
    
    # Build output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_centers[i, 0]
        inner_hex_data[i, 1] = best_centers[i, 1]
        inner_hex_data[i, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = best_s
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
