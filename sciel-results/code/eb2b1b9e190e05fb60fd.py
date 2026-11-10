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
N_WAYPOINTS = 8

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
        # R(-a) = [[cos(a), sin(a)], [-sin(a), cos(a)]]
        dx = _SOFA_PTS[:, 0] - x
        dy = _SOFA_PTS[:, 1] - y
        
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

def _neg_area(params, deadline):
    """Negative area for minimization. params is a flat array of waypoint coordinates."""
    if time.time() > deadline:
        return 1e6
    waypoints = params.reshape(N_WAYPOINTS, 3)
    area = _compute_area(waypoints)
    return -area

def _generate_initial_guesses():
    """
    Generate a diverse set of initial guesses for the optimizer.
    The path generally goes from (0,0,0) to (-2,0,90).
    We parameterize the intermediate points to follow a rough arc.
    """
    guesses = []
    
    # Baseline from current program, expanded to 8 points
    base_6 = np.array([
        [-0.1, 0.1, 15.0],
        [-0.3, 0.3, 30.0],
        [-0.5, 0.3, 45.0],
        [-0.7, 0.3, 60.0],
        [-0.9, 0.2, 75.0],
        [-1.1, 0.1, 85.0],
    ])
    
    # Interpolate base_6 to 8 points
    angles_8 = np.linspace(15, 85, 8)
    xs_6 = base_6[:, 0]
    ys_6 = base_6[:, 1]
    as_6 = base_6[:, 2]
    
    # Simple linear interpolation for x, y based on angle
    xs_8 = np.interp(angles_8, as_6, xs_6)
    ys_8 = np.interp(angles_8, as_6, ys_6)
    
    guess1 = np.column_stack((xs_8, ys_8, angles_8))
    guesses.append(guess1)
    
    # Variant 2: Wider arc
    guess2 = guess1.copy()
    guess2[:, 1] += 0.2 # Push y higher
    guesses.append(guess2)
    
    # Variant 3: Tighter arc
    guess3 = guess1.copy()
    guess3[:, 1] -= 0.1
    guesses.append(guess3)
    
    # Variant 4: Shifted x
    guess4 = guess1.copy()
    guess4[:, 0] -= 0.2
    guesses.append(guess4)
    
    # Random perturbations
    for i in range(20):
        g = guess1 + np.random.normal(0, 0.1, guess1.shape)
        # Ensure angles are roughly monotonic
        g[:, 2] = np.sort(g[:, 2])
        guesses.append(g)
        
    return guesses

def moving_sofa_2d() -> np.ndarray:
    """
    Find the path that maximizes the sofa area through the L-shaped corridor.
    
    Returns:
        path: np.ndarray of shape (n, 3); each row is (x, y, angle_degrees) for
            the corridor's inner corner.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    initial_guesses = _generate_initial_guesses()
    
    best_params = None
    best_area = -1.0
    
    # Phase 1: Quick evaluation of initial guesses
    scored_guesses = []
    for g in initial_guesses:
        if time.time() > deadline - 120:
            break
        area = _compute_area(g)
        scored_guesses.append((area, g))
        
    if not scored_guesses:
        # Fallback to a very simple path
        return np.array([
            [-0.1, 0.1, 15.0],
            [-0.3, 0.3, 30.0],
            [-0.5, 0.3, 45.0],
            [-0.7, 0.3, 60.0],
            [-0.9, 0.2, 75.0],
            [-1.1, 0.1, 85.0],
            [-1.3, 0.1, 87.0],
            [-1.5, 0.1, 88.0]
        ])

    # Sort by area descending
    scored_guesses.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 5 for detailed optimization
    top_candidates = [g for _, g in scored_guesses[:5]]
    
    # Phase 2: Nelder-Mead optimization on top candidates
    for g in top_candidates:
        if time.time() > deadline - 60:
            break
            
        try:
            result = minimize(
                _neg_area,
                g.ravel(),
                args=(deadline,),
                method='Nelder-Mead',
                options={
                    'maxiter': 150,
                    'maxfev': 150,
                    'xatol': 1e-3,
                    'fatol': 1e-4,
                    'adaptive': True,
                }
            )
            
            if -result.fun > best_area:
                candidate = result.x.reshape(N_WAYPOINTS, 3)
                # Verify the candidate
                cand_area = _compute_area(candidate)
                if cand_area > best_area:
                    best_area = cand_area
                    best_params = candidate.ravel().copy()
                    
        except Exception:
            continue
            
    # Phase 3: Final refinement on the best found
    if best_params is not None and time.time() < deadline - 30:
        try:
            result = minimize(
                _neg_area,
                best_params,
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
            if -result.fun > best_area:
                candidate = result.x.reshape(N_WAYPOINTS, 3)
                cand_area = _compute_area(candidate)
                if cand_area > best_area:
                    best_area = cand_area
                    best_params = candidate.ravel().copy()
        except Exception:
            pass

    if best_params is None:
        # Should not happen, but fallback
        return initial_guesses[0]

    return best_params.reshape(N_WAYPOINTS, 3)

# EVOLVE-BLOCK-END
