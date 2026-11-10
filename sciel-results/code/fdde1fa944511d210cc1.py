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

# Precompute for fast mask computation
_SOFA_X_1D = _SOFA_PTS[:, 0]
_SOFA_Y_1D = _SOFA_PTS[:, 1]

# Precompute unit vectors for sofa points for faster rotation
_SOFA_PTS_F = _SOFA_PTS.astype(np.float64)


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
    # Base: {0 <= u <= 1, -3 <= v <= 1} UNION {-3 <= u <= 1, 0 <= v <= 1}
    in_h = (u >= -1e-12) & (u <= 1 + 1e-12) & (v >= -3 - 1e-12) & (v <= 1 + 1e-12)
    in_v = (u >= -3 - 1e-12) & (u <= 1 + 1e-12) & (v >= -1e-12) & (v <= 1 + 1e-12)
    return in_h | in_v


def _compute_sofa_mask(waypoints):
    """
    Compute which sofa points are inside ALL placements along the path.
    Returns a boolean mask of shape (N_SOFa,).
    """
    placements = _generate_placements(waypoints)
    
    # Start with all True
    inside_all = np.ones(_N_SOFa, dtype=bool)
    
    for (x, y, a_deg) in placements:
        a_rad = np.radians(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        # Transform sofa points into corridor frame: R(-a)(w - (x, y))
        dx = _SOFA_X_1D - x
        dy = _SOFA_Y_1D - y
        
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        in_base = _point_in_base(u, v)
        inside_all &= in_base
        
        # Early exit if no points remain
        if not inside_all.any():
            break
    
    return inside_all


def _compute_area(waypoints):
    """Compute the area of the sofa for a given path."""
    mask = _compute_sofa_mask(waypoints)
    return mask.sum() * 0.0001


def _smooth_inside_value(u, v, softness=0.05):
    """
    Compute a smooth indicator of whether (u,v) is inside the base L-corridor.
    Returns values in [0, 1], with 1 for well-inside and 0 for well-outside.
    Uses smoothstep-like transitions near boundaries.
    """
    # For the horizontal arm: 0 <= u <= 1, -3 <= v <= 1
    # For the vertical arm: -3 <= u <= 1, 0 <= v <= 1
    
    # Define smooth min/max using softplus-like functions
    def smooth_min(a, b, s=softness):
        """Smooth minimum of two arrays"""
        return -np.logaddexp(-a/s, -b/s) * s
    
    def smooth_max(a, b, s=softness):
        """Smooth maximum of two arrays"""
        return np.logaddexp(a/s, b/s) * s
    
    # Distance to boundaries for horizontal arm
    # u in [0, 1], v in [-3, 1]
    hu_min = np.maximum(0.0, np.minimum(u, 1.0 - u))  # distance to u-boundaries
    hv_min = np.maximum(-3.0, np.minimum(v, 1.0 - v))  # this isn't right
    
    # Let me use a different approach: define margin from each boundary
    # For horizontal arm:
    #   u >= 0: margin_u_h = u
    #   u <= 1: margin_u_h = 1 - u
    #   v >= -3: margin_v_h = v + 3
    #   v <= 1: margin_v_h = 1 - v
    # Point is in horizontal arm if all margins >= 0
    
    mu_h1 = u  # u >= 0
    mu_h2 = 1.0 - u  # u <= 1
    mv_h1 = v + 3.0  # v >= -3
    mv_h2 = 1.0 - v  # v <= 1
    
    # For vertical arm:
    #   u >= -3: margin_u_v = u + 3
    #   u <= 1: margin_u_v = 1 - u
    #   v >= 0: margin_v_v = v
    #   v <= 1: margin_v_v = 1 - v
    
    mu_v1 = u + 3.0  # u >= -3
    mu_v2 = 1.0 - u  # u <= 1
    mv_v1 = v  # v >= 0
    mv_v2 = 1.0 - v  # v <= 1
    
    # Smooth min of the four margins for each arm
    min_h = np.minimum(np.minimum(mu_h1, mu_h2), np.minimum(mv_h1, mv_h2))
    min_v = np.minimum(np.minimum(mu_v1, mu_v2), np.minimum(mv_v1, mv_v2))
    
    # Smooth max of the two arm margins
    min_union = np.maximum(min_h, min_v)
    
    # Convert to [0,1] with smooth transition
    # Use tanh or sigmoid
    val = np.tanh(min_union / softness)
    return (val + 1.0) / 2.0


def _compute_smooth_value(waypoints, softness=0.05):
    """
    Compute a smooth approximation of the sofa area.
    For each point, compute the minimum smooth-inside-value over all placements.
    Sum these values and multiply by 0.0001.
    """
    placements = _generate_placements(waypoints)
    
    # Start with large values (so min works)
    min_vals = np.full(_N_SOFa, 1.0)
    
    for (x, y, a_deg) in placements:
        a_rad = np.radians(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        dx = _SOFA_X_1D - x
        dy = _SOFA_Y_1D - y
        
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        vals = _smooth_inside_value(u, v, softness)
        min_vals = np.minimum(min_vals, vals)
        
        # Early exit if all values are very small
        if min_vals.max() < 0.01:
            break
    
    return min_vals.sum() * 0.0001


def _neg_smooth_area(params, softness=0.05):
    """Negative smooth area for minimization."""
    n_wp = params.shape[0] // 3
    waypoints = params.reshape(n_wp, 3)
    val = _compute_smooth_value(waypoints, softness)
    return -val


def _neg_area(params):
    """Negative area for minimization. params is a flat array of waypoint coordinates."""
    n_wp = params.shape[0] // 3
    waypoints = params.reshape(n_wp, 3)
    area = _compute_area(waypoints)
    return -area


def _generate_initial_guesses(n_wp=12):
    """Generate diverse initial guesses for the optimizer."""
    guesses = []
    
    ts = np.linspace(0, 1, n_wp)
    a = 90.0 * ts
    
    # Path 1: Classic arc
    x = -1.2 * ts + 0.1 * np.sin(np.pi * ts)
    y = 0.4 * np.sin(np.pi * ts)
    guesses.append(np.column_stack([x, y, a]))
    
    # Path 2: Higher arc
    x = -1.2 * ts + 0.05 * np.sin(np.pi * ts)
    y = 0.55 * np.sin(np.pi * ts)
    guesses.append(np.column_stack([x, y, a]))
    
    # Path 3: Lower arc
    x = -1.2 * ts + 0.05 * np.sin(np.pi * ts)
    y = 0.25 * np.sin(np.pi * ts)
    guesses.append(np.column_stack([x, y, a]))
    
    # Path 4: S-curve
    x = -1.2 * ts + 0.15 * np.sin(2 * np.pi * ts)
    y = 0.35 * np.sin(np.pi * ts)
    guesses.append(np.column_stack([x, y, a]))
    
    # Path 5: Conservative
    x = -1.1 * ts
    y = 0.3 * np.sin(np.pi * ts)
    guesses.append(np.column_stack([x, y, a]))
    
    # Path 6: Aggressive turn early
    x = -1.2 * ts**1.5
    y = 0.4 * np.sin(np.pi * ts)
    guesses.append(np.column_stack([x, y, a]))
    
    # Path 7: Aggressive turn late
    x = -1.2 * ts**0.7
    y = 0.4 * np.sin(np.pi * ts)
    guesses.append(np.column_stack([x, y, a]))
    
    # Path 8: Minimal y
    x = -1.2 * ts
    y = 0.15 * np.sin(np.pi * ts)
    guesses.append(np.column_stack([x, y, a]))
    
    # Path 9: Quadratic x profile
    x = -1.2 * ts**2
    y = 0.35 * np.sin(np.pi * ts)
    guesses.append(np.column_stack([x, y, a]))
    
    # Path 10: Cubic x profile
    x = -1.2 * (3*ts**2 - 2*ts**3)
    y = 0.4 * np.sin(np.pi * ts)
    guesses.append(np.column_stack([x, y, a]))
    
    # Path 11: Very high arc
    x = -1.2 * ts
    y = 0.7 * np.sin(np.pi * ts)
    guesses.append(np.column_stack([x, y, a]))
    
    # Path 12: Very low arc
    x = -1.2 * ts
    y = 0.1 * np.sin(np.pi * ts)
    guesses.append(np.column_stack([x, y, a]))
    
    # Path 13: Pivot-like motion (stay near origin longer)
    x = -1.2 * np.clip((ts - 0.2) / 0.8, 0, 1)**1.3
    y = 0.35 * np.sin(np.pi * np.clip((ts - 0.2) / 0.8, 0, 1))
    guesses.append(np.column_stack([x, y, a]))
    
    # Path 14: Delayed start
    x = -1.2 * np.clip((ts - 0.15) / 0.85, 0, 1)**1.2
    y = 0.3 * np.sin(np.pi * np.clip((ts - 0.15) / 0.85, 0, 1))
    guesses.append(np.column_stack([x, y, a]))
    
    # Path 15: Asymmetric y
    x = -1.2 * ts
    y = 0.3 * np.sin(np.pi * ts) + 0.1 * np.sin(2*np.pi*ts)
    guesses.append(np.column_stack([x, y, a]))
    
    # Path 16: Bump in middle
    x = -1.2 * ts + 0.2 * np.exp(-((ts-0.5)/0.15)**2)
    y = 0.35 * np.sin(np.pi * ts)
    guesses.append(np.column_stack([x, y, a]))
    
    return guesses


def _perturb_params(params, n_wp, scale=0.1, rng=None):
    """Generate perturbed versions of params."""
    if rng is None:
        rng = np.random.default_rng()
    n_pert = 8
    perturbations = []
    for _ in range(n_pert):
        p = params.copy()
        p += rng.normal(0, scale, size=p.shape)
        perturbations.append(p)
    return perturbations


def moving_sofa_2d() -> np.ndarray:
    """
    Find the path that maximizes the sofa area through the L-shaped corridor.
    
    Returns:
        path: np.ndarray of shape (n, 3); each row is (x, y, angle_degrees) for
            the corridor's inner corner.
    """
    start_time = time.time()
    deadline = start_time + 570  # Safety margin before 630s limit
    
    N_WAYPOINTS = 12
    
    # Generate initial guesses
    start_points = _generate_initial_guesses(N_WAYPOINTS)
    
    # Evaluate all initial guesses
    best_params = start_points[0].ravel().copy()
    best_area = _compute_area(start_points[0])
    
    for sp in start_points[1:]:
        if time.time() > deadline - 120:
            break
        area = _compute_area(sp)
        if area > best_area:
            best_area = area
            best_params = sp.ravel().copy()
    
    # Phase 1: Coarse optimization with smooth objective for top start points
    softness = 0.08
    scored_points = []
    for sp in start_points:
        area = _compute_area(sp)
        scored_points.append((area, sp))
    scored_points.sort(key=lambda x: -x[0])
    
    # Optimize top 8 start points with smooth objective
    for i, (sp_area, sp) in enumerate(scored_points[:8]):
        if time.time() > deadline - 90:
            break
        
        try:
            result = minimize(
                lambda p: _neg_smooth_area(p, softness),
                sp.ravel(),
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'maxfev': 200,
                    'xatol': 1e-3,
                    'fatol': 1e-4,
                    'adaptive': True,
                }
            )
            
            candidate = result.x.reshape(N_WAYPOINTS, 3)
            cand_area = _compute_area(candidate)
            if cand_area > best_area:
                best_area = cand_area
                best_params = candidate.ravel().copy()
                
        except Exception:
            continue
    
    # Phase 2: Multi-start perturbation around best solution
    if time.time() < deadline - 60:
        rng = np.random.default_rng(42)
        perturbations = _perturb_params(best_params, N_WAYPOINTS, scale=0.08, rng=rng)
        
        for pert in perturbations:
            if time.time() > deadline - 45:
                break
            try:
                result = minimize(
                    _neg_area,
                    pert,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 150,
                        'maxfev': 150,
                        'xatol': 1e-4,
                        'fatol': 1e-5,
                        'adaptive': True,
                    }
                )
                
                candidate = result.x.reshape(N_WAYPOINTS, 3)
                cand_area = _compute_area(candidate)
                if cand_area > best_area:
                    best_area = cand_area
                    best_params = candidate.ravel().copy()
                    
            except Exception:
                continue
    
    # Phase 3: Fine optimization with exact objective - multiple rounds
    for round_num in range(6):
        if time.time() > deadline - 30:
            break
        
        try:
            result = minimize(
                _neg_area,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 250,
                    'maxfev': 250,
                    'xatol': 1e-4,
                    'fatol': 1e-5,
                    'adaptive': True,
                }
            )
            
            candidate = result.x.reshape(N_WAYPOINTS, 3)
            cand_area = _compute_area(candidate)
            if cand_area > best_area:
                best_area = cand_area
                best_params = candidate.ravel().copy()
            else:
                break  # No improvement, stop early
                
        except Exception:
            break
    
    # Phase 4: Final polish with very tight tolerance
    if time.time() < deadline - 10:
        try:
            result = minimize(
                _neg_area,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'maxfev': 500,
                    'xatol': 1e-5,
                    'fatol': 1e-6,
                    'adaptive': True,
                }
            )
            
            candidate = result.x.reshape(N_WAYPOINTS, 3)
            cand_area = _compute_area(candidate)
            if cand_area > best_area:
                best_area = cand_area
                best_params = candidate.ravel().copy()
        except Exception:
            pass
    
    waypoints = best_params.reshape(N_WAYPOINTS, 3)
    
    # Validate: ensure all values are finite
    if not np.all(np.isfinite(waypoints)):
        # Fallback to a simple path
        ts = np.linspace(0, 1, N_WAYPOINTS)
        x = -1.2 * ts
        y = 0.3 * np.sin(np.pi * ts)
        a = 90.0 * ts
        waypoints = np.column_stack([x, y, a])
    
    return waypoints


# EVOLVE-BLOCK-END
