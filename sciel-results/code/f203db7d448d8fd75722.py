# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

# Precompute sofa points at module level
_SOFA_U = np.arange(-3.0, 1.5, 0.01)
_SOFA_V = np.arange(-1.5, 1.5, 0.01)
_SOFA_X, _SOFA_Y = np.meshgrid(_SOFA_U, _SOFA_V, indexing='ij')
_SOFA_PTS = np.stack([_SOFA_X.ravel(), _SOFA_Y.ravel()], axis=1)  # shape (N, 2)
_N_SOFa = _SOFA_PTS.shape[0]

# Number of waypoints to optimize
N_WAYPOINTS = 12

def _generate_placements(waypoints):
    """
    Generate the full list of placements including start (0,0,0) and end (-2,0,90).
    Each placement is (x, y, a_degrees). Linearly interpolate between consecutive
    waypoints with m = max(1, ceil(|a2 - a1|)) steps.
    """
    P0 = np.array([0.0, 0.0, 0.0])
    PN = np.array([-2.0, 0.0, 90.0])
    full = [P0] + [np.array(p, dtype=np.float64) for p in waypoints] + [PN]
    
    placements = [full[0].copy()]
    for i in range(len(full) - 1):
        p1 = full[i]
        p2 = full[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da - 1e-9)))
        for k in range(1, m + 1):
            t = k / m
            placements.append(p1 + t * (p2 - p1))
    return placements

def _point_in_base(u, v):
    """Check if point (u, v) is in the base L-shaped corridor."""
    in_h = (u >= -1e-12) & (u <= 1 + 1e-12) & (v >= -3 - 1e-12) & (v <= 1 + 1e-12)
    in_v = (u >= -3 - 1e-12) & (u <= 1 + 1e-12) & (v >= -1e-12) & (v <= 1 + 1e-12)
    return in_h | in_v

def _compute_sofa_mask(waypoints):
    """
    Compute which sofa points are inside ALL placements along the path.
    Returns a boolean mask of shape (N_SOFa,).
    """
    placements = _generate_placements(waypoints)
    
    inside_all = np.ones(_N_SOFa, dtype=bool)
    
    for (x, y, a_deg) in placements:
        a_rad = np.radians(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        dx = _SOFA_PTS[:, 0] - x
        dy = _SOFA_PTS[:, 1] - y
        
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        in_base = _point_in_base(u, v)
        inside_all &= in_base
        
        if not inside_all.any():
            break
    
    return inside_all

def _compute_area(waypoints):
    """Compute the area of the sofa for a given path."""
    mask = _compute_sofa_mask(waypoints)
    return mask.sum() * 0.0001

def _compute_smooth_area(waypoints, beta=20.0):
    """
    Compute a smooth surrogate for the sofa area.
    For each sofa point, compute a smooth measure of how well it fits in all placements.
    Uses a soft-min approach: for each point, compute the minimum "inside-ness" across
    all placements, smoothed with a sigmoid.
    """
    placements = _generate_placements(waypoints)
    n_placements = len(placements)
    
    # For each placement, compute a smooth inside indicator in [0, 1]
    # Using a signed distance approximation: positive inside, negative outside
    # Smooth with sigmoid(beta * margin)
    
    min_score = np.ones(_N_SOFa, dtype=np.float64)
    
    for (x, y, a_deg) in placements:
        a_rad = np.radians(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        dx = _SOFA_PTS[:, 0] - x
        dy = _SOFA_PTS[:, 1] - y
        
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        # Compute distance to boundary of L-shaped corridor
        # For horizontal arm: {0 <= u <= 1, -3 <= v <= 1}
        # For vertical arm: {-3 <= u <= 1, 0 <= v <= 1}
        
        # Distance to horizontal arm boundary
        du_h = np.maximum(np.maximum(0 - u, u - 1), 0)
        dv_h = np.maximum(np.maximum(0 - v, v - 1), 0)
        dist_h = np.maximum(du_h, dv_h)
        inside_h = (u >= -1e-9) & (u <= 1 + 1e-9) & (v >= -3 - 1e-9) & (v <= 1 + 1e-9)
        
        # Distance to vertical arm boundary
        du_v = np.maximum(np.maximum(-3 - u, u - 1), 0)
        dv_v = np.maximum(np.maximum(0 - v, v - 1), 0)
        dist_v = np.maximum(du_v, dv_v)
        inside_v = (u >= -3 - 1e-9) & (u <= 1 + 1e-9) & (v >= -1e-9) & (v <= 1 + 1e-9)
        
        # Combined: inside if in either arm
        inside = inside_h | inside_v
        
        # Signed distance: positive if inside, negative if outside
        # For inside points, use 0 (or small positive)
        # For outside points, use negative of distance to nearest boundary
        margin = np.where(inside, 0.0, -np.minimum(dist_h, dist_v))
        
        # Smooth indicator: sigmoid(beta * margin)
        # margin >= 0 -> sigmoid >= 0.5, margin < 0 -> sigmoid < 0.5
        # Scale to [0, 1]: use 0.5 * (1 + sigmoid(beta * margin))
        smooth_ind = 0.5 * (1.0 + np.tanh(beta * margin))
        
        min_score = np.minimum(min_score, smooth_ind)
    
    # Sum over all points, scale to approximate area
    return np.sum(min_score) * 0.0001

def _neg_area(params, deadline):
    """Negative area for minimization."""
    if time.time() > deadline:
        return 1e6
    waypoints = params.reshape(N_WAYPOINTS, 3)
    area = _compute_area(waypoints)
    return -area

def _neg_smooth_area(params, deadline, beta=20.0):
    """Negative smooth area for minimization."""
    if time.time() > deadline:
        return 1e6
    waypoints = params.reshape(N_WAYPOINTS, 3)
    area = _compute_smooth_area(waypoints, beta)
    return -area

def _generate_initial_guesses():
    """
    Generate diverse initial guesses.
    """
    guesses = []
    n = N_WAYPOINTS
    
    # Template 1: Standard arc (semi-circular)
    angles = np.linspace(10, 85, n)
    t = np.linspace(0, 1, n)
    x = -0.1 - 1.4 * t
    y = 0.35 * np.sin(np.pi * t) + 0.05
    g1 = np.column_stack((x, y, angles))
    guesses.append(g1)
    
    # Template 2: Higher arc
    g2 = g1.copy()
    g2[:, 1] += 0.12
    guesses.append(g2)
    
    # Template 3: Lower arc
    g3 = g1.copy()
    g3[:, 1] -= 0.08
    guesses.append(g3)
    
    # Template 4: Shifted left
    g4 = g1.copy()
    g4[:, 0] -= 0.12
    guesses.append(g4)
    
    # Template 5: Different angle distribution (S-curve)
    angles5 = np.linspace(8, 87, n)
    # Apply S-curve transformation
    angles5_s = 45 - 42 * np.sin(np.pi * (angles5 - 45) / 42)
    angles5_s = np.clip(angles5_s, 5, 88)
    # Ensure monotonic
    for i in range(1, n):
        if angles5_s[i] < angles5_s[i-1]:
            angles5_s[i] = angles5_s[i-1] + 1
    g5 = np.column_stack((x, y, angles5_s))
    guesses.append(g5)
    
    # Template 6: More gradual at ends
    angles6 = np.linspace(5, 88, n)
    angles6 = 45 - 43 * np.sin(np.pi * (angles6 - 45) / 43)
    angles6 = np.clip(angles6, 3, 89)
    for i in range(1, n):
        if angles6[i] < angles6[i-1]:
            angles6[i] = angles6[i-1] + 1
    x6 = -0.15 - 1.3 * t
    y6 = 0.3 * np.sin(np.pi * t) + 0.08
    g6 = np.column_stack((x6, y6, angles6))
    guesses.append(g6)
    
    # Template 7: Asymmetric y profile
    y7 = 0.25 * np.sin(np.pi * t**0.7) + 0.1
    g7 = np.column_stack((x, y7, angles))
    guesses.append(g7)
    
    # Random perturbations of top templates
    rng = np.random.RandomState(42)
    for base_idx in [0, 1, 2, 5]:
        for _ in range(6):
            g = guesses[base_idx].copy()
            g += rng.normal(0, 0.06, g.shape)
            # Ensure monotonic angles
            g[:, 2] = np.maximum.accumulate(g[:, 2])
            g[:, 2] = np.clip(g[:, 2], 2, 88)
            guesses.append(g)
    
    return guesses

def _perturb_and_refine(params, deadline, n_trials=20, scales=None):
    """
    Local perturbation search around a given parameter set.
    """
    if scales is None:
        scales = [0.05, 0.03, 0.02, 0.01]
    
    best_params = params.copy()
    best_area = _compute_area(params.reshape(N_WAYPOINTS, 3))
    best_wpts = params.reshape(N_WAYPOINTS, 3)
    
    rng = np.random.RandomState(12345)
    
    for scale in scales:
        for _ in range(n_trials):
            if time.time() > deadline:
                break
            perturbed = best_wpts + rng.normal(0, scale, best_wpts.shape)
            # Ensure monotonic angles
            perturbed[:, 2] = np.maximum.accumulate(perturbed[:, 2])
            perturbed[:, 2] = np.clip(perturbed[:, 2], 1, 89)
            p_area = _compute_area(perturbed)
            if p_area > best_area:
                best_area = p_area
                best_params = perturbed.ravel().copy()
                best_wpts = perturbed
    
    return best_params, best_area

def moving_sofa_2d() -> np.ndarray:
    """
    Find the path that maximizes the sofa area through the L-shaped corridor.
    """
    start_time = time.time()
    deadline = start_time + 570  # Safety margin
    
    initial_guesses = _generate_initial_guesses()
    
    best_params = None
    best_area = -1.0
    
    # Phase 1: Evaluate all initial guesses
    scored_guesses = []
    for g in initial_guesses:
        if time.time() > deadline - 240:
            break
        area = _compute_area(g)
        scored_guesses.append((area, g))
        if area > best_area:
            best_area = area
            best_params = g.ravel().copy()
    
    if not scored_guesses or best_params is None:
        # Fallback: return a reasonable default
        n = N_WAYPOINTS
        angles = np.linspace(10, 85, n)
        t = np.linspace(0, 1, n)
        x = -0.1 - 1.4 * t
        y = 0.35 * np.sin(np.pi * t) + 0.05
        return np.column_stack((x, y, angles))
    
    scored_guesses.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 2: Smooth optimization on top candidates (faster, guides to good regions)
    top_n = min(4, len(scored_guesses))
    for i in range(top_n):
        if time.time() > deadline - 120:
            break
        g = scored_guesses[i][1]
        
        try:
            result = minimize(
                _neg_smooth_area,
                g.ravel(),
                args=(deadline, 20.0),
                method='Nelder-Mead',
                options={
                    'maxiter': 150,
                    'maxfev': 150,
                    'xatol': 1e-3,
                    'fatol': 1e-4,
                    'adaptive': True,
                }
            )
            
            cand_area = _compute_area(result.x.reshape(N_WAYPOINTS, 3))
            if cand_area > best_area:
                best_area = cand_area
                best_params = result.x.ravel().copy()
                
        except Exception:
            continue
    
    # Phase 3: Exact optimization on top candidates
    # Re-evaluate top candidates with exact area
    scored_exact = []
    for i in range(top_n):
        if time.time() > deadline - 60:
            break
        g = scored_guesses[i][1]
        area = _compute_area(g)
        scored_exact.append((area, g))
    
    # Also add the best from smooth optimization
    if best_params is not None:
        scored_exact.append((best_area, best_params.reshape(N_WAYPOINTS, 3)))
    
    scored_exact.sort(key=lambda x: x[0], reverse=True)
    top_exact = min(3, len(scored_exact))
    
    for i in range(top_exact):
        if time.time() > deadline - 45:
            break
        g = scored_exact[i][1]
        
        try:
            result = minimize(
                _neg_area,
                g.ravel(),
                args=(deadline,),
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'maxfev': 200,
                    'xatol': 1e-4,
                    'fatol': 1e-5,
                    'adaptive': True,
                }
            )
            
            cand_area = _compute_area(result.x.reshape(N_WAYPOINTS, 3))
            if cand_area > best_area:
                best_area = cand_area
                best_params = result.x.ravel().copy()
                
        except Exception:
            continue
    
    # Phase 4: Final refinement with tighter tolerance
    if best_params is not None and time.time() < deadline - 20:
        try:
            result = minimize(
                _neg_area,
                best_params,
                args=(deadline,),
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'maxfev': 300,
                    'xatol': 1e-5,
                    'fatol': 1e-6,
                    'adaptive': True,
                }
            )
            cand_area = _compute_area(result.x.reshape(N_WAYPOINTS, 3))
            if cand_area > best_area:
                best_area = cand_area
                best_params = result.x.ravel().copy()
        except Exception:
            pass
    
    # Phase 5: Local perturbation search on best
    if best_params is not None and time.time() < deadline - 10:
        best_params, best_area = _perturb_and_refine(
            best_params, deadline - 5, n_trials=15, scales=[0.04, 0.02, 0.01]
        )
    
    if best_params is None:
        return scored_guesses[0][1]
    
    return best_params.reshape(N_WAYPOINTS, 3)

# EVOLVE-BLOCK-END
