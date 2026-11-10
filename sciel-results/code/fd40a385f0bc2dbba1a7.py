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


def _min_containing_hex_side(centers, ocx, ocy, oangle_deg):
    """
    Compute the minimum side length s of a regular hexagon centered at (ocx, ocy), angle oangle_deg,
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
        
    # Outer hexagon normals
    # The vertices of the outer hexagon are at angles oangle + 60*i.
    # The edges are perpendicular to the normals at angles oangle + 60*i + 30.
    # Wait, let's verify.
    # Vertices: v_i = (cos(a + 60i), sin(a + 60i))
    # Edge from v_i to v_{i+1}. Midpoint angle is a + 60i + 30.
    # Normal points outward at angle a + 60i + 30.
    # So normals are at angles: oangle_deg + 30 + 60*i for i=0..5.
    
    oangle_rad = np.deg2rad(oangle_deg)
    normal_angles = oangle_rad + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)
    
    verts_flat = verts.reshape(-1, 2)  # (36, 2)
    
    # We need s such that for all vertices v and all normals n_k:
    # (v - center) dot n_k <= s * apothem
    # Apothem of hexagon with side s is s * sqrt(3)/2.
    # So (v - center) dot n_k <= s * sqrt(3)/2
    
    rel_verts = verts_flat - np.array([ocx, ocy])
    projections = rel_verts @ normals.T  # (36, 6)
    max_proj = np.max(projections)
    
    if max_proj <= 0:
        return 0.0
        
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


def _objective(params):
    """
    Objective: minimize the containing hexagon side length with penalty for overlaps.
    params: [cx1, cy1, ..., cx12, cy12, ocx, ocy, oangle_deg]
    """
    centers = params[:24].reshape(12, 2)
    ocx = params[24]
    ocy = params[25]
    oangle_deg = params[26]
    
    s = _min_containing_hex_side(centers, ocx, ocy, oangle_deg)
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
    best_params = None
    best_s = np.inf
    
    rng = np.random.default_rng(42)
    
    initial_configs = [
        _make_hex_lattice_12(),
        _make_compact_initial(),
    ]
    
    for init_idx, init_centers in enumerate(initial_configs):
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - _SAFETY_MARGIN:
            break
        
        # Phase 1: Force-directed relaxation to resolve overlaps
        relaxed = _force_directed_relax(init_centers, n_iters=800, dt=0.08, repulsion_strength=1.0)
        
        if _check_separation(relaxed):
            # Initialize outer hexagon to contain the relaxed set
            # Center at centroid, angle 0
            centroid = np.mean(relaxed, axis=0)
            ocx, ocy = centroid[0], centroid[1]
            oangle = 0.0
            s_relaxed = _min_containing_hex_side(relaxed, ocx, ocy, oangle)
            
            if s_relaxed < best_s:
                best_s = s_relaxed
                best_params = np.concatenate([relaxed.flatten(), [ocx, ocy, oangle]])
        
        # Phase 2: Nelder-Mead optimization on centers + outer center/angle
        if best_params is not None:
            elapsed = time.time() - _START_TIME
            if elapsed < _DEADLINE - _SAFETY_MARGIN:
                x0 = best_params.copy()
                try:
                    result = minimize(
                        _objective,
                        x0,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 10000,
                            'maxfev': 100000,
                            'xatol': 1e-10,
                            'fatol': 1e-12,
                            'adaptive': True,
                        }
                    )
                except Exception:
                    result = None
                
                if result is not None:
                    candidate_params = result.x
                    candidate_centers = candidate_params[:24].reshape(12, 2)
                    candidate_ocx = candidate_params[24]
                    candidate_ocy = candidate_params[25]
                    candidate_oangle = candidate_params[26]
                    
                    if _check_separation(candidate_centers):
                        s_cand = _min_containing_hex_side(candidate_centers, candidate_ocx, candidate_ocy, candidate_oangle)
                        if s_cand < best_s:
                            best_s = s_cand
                            best_params = candidate_params.copy()
        
        # Phase 3: Random restarts
        n_restarts = 3
        for r in range(n_restarts):
            elapsed = time.time() - _START_TIME
            if elapsed > _DEADLINE - _SAFETY_MARGIN:
                break
            
            noise_scale = 0.3 * (1.0 + r * 0.5)
            perturbed = init_centers + rng.normal(0, noise_scale, size=init_centers.shape)
            
            relaxed_r = _force_directed_relax(perturbed, n_iters=500, dt=0.06, repulsion_strength=1.0)
            
            if _check_separation(relaxed_r):
                centroid_r = np.mean(relaxed_r, axis=0)
                ocx_r, ocy_r = centroid_r[0], centroid_r[1]
                oangle_r = 0.0
                s_r = _min_containing_hex_side(relaxed_r, ocx_r, ocy_r, oangle_r)
                
                if s_r < best_s:
                    best_s = s_r
                    best_params = np.concatenate([relaxed_r.flatten(), [ocx_r, ocy_r, oangle_r]])
    
    # Fallback
    if best_params is None:
        init_centers = _make_hex_lattice_12()
        relaxed = _force_directed_relax(init_centers, n_iters=500)
        centroid = np.mean(relaxed, axis=0)
        best_params = np.concatenate([relaxed.flatten(), [centroid[0], centroid[1], 0.0]])
        best_s = _min_containing_hex_side(relaxed, centroid[0], centroid[1], 0.0)
    
    # Final Polish
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - _SAFETY_MARGIN:
        x0_final = best_params.copy()
        try:
            result_final = minimize(
                _objective,
                x0_final,
                method='Nelder-Mead',
                options={
                    'maxiter': 15000,
                    'maxfev': 150000,
                    'xatol': 1e-11,
                    'fatol': 1e-13,
                    'adaptive': True,
                }
            )
        except Exception:
            result_final = None
        
        if result_final is not None:
            candidate_params = result_final.x
            candidate_centers = candidate_params[:24].reshape(12, 2)
            candidate_ocx = candidate_params[24]
            candidate_ocy = candidate_params[25]
            candidate_oangle = candidate_params[26]
            
            if _check_separation(candidate_centers):
                s_cand = _min_containing_hex_side(candidate_centers, candidate_ocx, candidate_ocy, candidate_oangle)
                if s_cand < best_s:
                    best_s = s_cand
                    best_params = candidate_params.copy()
    
    # Build output
    best_centers = best_params[:24].reshape(12, 2)
    best_ocx = best_params[24]
    best_ocy = best_params[25]
    best_oangle = best_params[26]
    
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_centers[i, 0]
        inner_hex_data[i, 1] = best_centers[i, 1]
        inner_hex_data[i, 2] = 0.0
    
    outer_hex_data = np.array([best_ocx, best_ocy, best_oangle])
    outer_hex_side_length = best_s
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
