# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

# Precompute sofa points
_SOFA_U = np.arange(-3.0, 1.5, 0.01)
_SOFA_V = np.arange(-1.5, 1.5, 0.01)
_SOFA_X, _SOFA_Y = np.meshgrid(_SOFA_U, _SOFA_V, indexing='ij')
_N_SOFA = _SOFA_X.size
_SOFA_X_1D = _SOFA_X.ravel()
_SOFA_Y_1D = _SOFA_Y.ravel()

# Precompute base corridor vertices for SAT
# Base corridor: union of {0<=u<=1, -3<=v<=1} and {-3<=u<=1, 0<=v<=1}
# Convex hull of this L-shape:
# The L-shape is non-convex, but we can work with its convex hull for a fast rejection test,
# then do exact point-in-polygon for the actual L-shape.
# Actually, for SAT we need convex shapes. The L-shape is not convex.
# Let's use a different approach: precompute the 8 vertices of the L-shape boundary
# and use point-in-polygon (which is fast) rather than SAT.

# L-shape vertices (counterclockwise):
# (0,-3), (1,-3), (1,1), (0,1), (-3,1), (-3,0), (0,0), (0,-3)
_L_VERTS = np.array([
    [0.0, -3.0],
    [1.0, -3.0],
    [1.0, 1.0],
    [0.0, 1.0],
    [-3.0, 1.0],
    [-3.0, 0.0],
    [0.0, 0.0],
])
_N_L_VERTS = len(_L_VERTS)

# For point-in-polygon with the L-shape, we can use the fact that it's a simple polygon.
# But the vectorized numpy approach checking all 145000 points is actually quite fast.
# Let's profile: each placement check does ~145000 * (few ops) = fast.
# The bottleneck is the number of placements and optimization iterations.

def _generate_placements(waypoints):
    """Generate all placements including start, waypoints, end, with linear interpolation."""
    P0 = np.array([0.0, 0.0, 0.0])
    PN = np.array([-2.0, 0.0, 90.0])
    full = [P0] + [np.asarray(p, dtype=np.float64) for p in waypoints] + [PN]
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


def _compute_area(waypoints):
    """Compute the area of the sofa intersection. Returns count * 0.0001."""
    placements = _generate_placements(waypoints)
    inside_all = np.ones(_N_SOFA, dtype=bool)
    for (x, y, a_deg) in placements:
        a_rad = np.radians(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        dx = _SOFA_X_1D - x
        dy = _SOFA_Y_1D - y
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        in_h = (u >= -1e-12) & (u <= 1 + 1e-12) & (v >= -3 - 1e-12) & (v <= 1 + 1e-12)
        in_v = (u >= -3 - 1e-12) & (u <= 1 + 1e-12) & (v >= -1e-12) & (v <= 1 + 1e-12)
        in_base = in_h | in_v
        inside_all &= in_base
        if not inside_all.any():
            break
    return inside_all.sum() * 0.0001


def _neg_area(params):
    n_wp = params.shape[0] // 3
    waypoints = params.reshape(n_wp, 3)
    return -_compute_area(waypoints)


def _generate_initial_candidates(n_wp, seed=42):
    """Generate diverse initial candidate paths."""
    rng = np.random.RandomState(seed)
    ts = np.linspace(0, 1, n_wp)
    candidates = []
    
    # S-curve variations
    for amp in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for x_scale in [1.0, 1.2, 1.5, 1.8, 2.0]:
            candidates.append(np.column_stack([
                -x_scale * ts,
                amp * np.sin(np.pi * ts),
                90.0 * ts
            ]))
    
    # Early turn
    for p in [1.2, 1.5, 2.0, 2.5, 3.0]:
        candidates.append(np.column_stack([
            -1.5 * ts**p,
            0.4 * np.sin(np.pi * ts),
            90.0 * ts
        ]))
    
    # Late turn
    for p in [0.3, 0.5, 0.7, 0.8, 1.0]:
        candidates.append(np.column_stack([
            -1.5 * ts**p,
            0.4 * np.sin(np.pi * ts),
            90.0 * ts
        ]))
    
    # Arc-like paths
    for r in [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
        angles = np.linspace(0, np.pi/2, n_wp)
        x = r * (1 - np.cos(angles)) - 1.2
        y = r * np.sin(angles)
        a = 90.0 * angles / (np.pi/2)
        candidates.append(np.column_stack([x, y, a]))
    
    # Sine wave variations
    for freq in [0.5, 1.0, 1.5, 2.0]:
        for amp in [0.2, 0.4, 0.6]:
            candidates.append(np.column_stack([
                -1.5 * ts,
                amp * np.sin(freq * np.pi * ts),
                90.0 * ts
            ]))
    
    # Random perturbations around a base path
    base_x = -1.5 * ts
    base_y = 0.4 * np.sin(np.pi * ts)
    base_a = 90.0 * ts
    for _ in range(10):
        px = base_x + rng.randn(n_wp) * 0.1
        py = base_y + rng.randn(n_wp) * 0.1
        pa = base_a + rng.randn(n_wp) * 2.0
        pa = np.clip(pa, 0, 90)
        candidates.append(np.column_stack([px, py, pa]))
    
    return candidates


def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    
    N_WAYPOINTS = 15
    
    # Generate initial candidates
    candidates = _generate_initial_candidates(N_WAYPOINTS)
    
    # Evaluate all candidates
    scored = []
    for c in candidates:
        if time.time() > deadline - 300:
            break
        area = _compute_area(c)
        scored.append((area, c))
    
    if not scored:
        # Fallback
        ts = np.linspace(0, 1, N_WAYPOINTS)
        waypoints = np.column_stack([
            -1.5 * ts,
            0.4 * np.sin(np.pi * ts),
            90.0 * ts
        ])
        return waypoints
    
    scored.sort(key=lambda x: -x[0])
    
    best_params = scored[0][1].ravel().copy()
    best_area = scored[0][0]
    
    # Optimize top candidates with Nelder-Mead
    top_k = min(8, len(scored))
    for i in range(top_k):
        if time.time() > deadline - 200:
            break
        try:
            result = minimize(
                _neg_area,
                scored[i][1].ravel(),
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'maxfev': 500,
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
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 100:
        try:
            result = minimize(
                _neg_area,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'maxfev': 1000,
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
    
    # Second round of optimization from slightly perturbed best
    if time.time() < deadline - 50:
        try:
            rng = np.random.RandomState(123)
            for trial in range(5):
                if time.time() > deadline - 20:
                    break
                perturbed = best_params + rng.randn(*best_params.shape) * 0.02
                result = minimize(
                    _neg_area,
                    perturbed,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 400,
                        'maxfev': 400,
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
    
    # Third round: differential evolution-like exploration
    if time.time() < deadline - 30:
        try:
            rng = np.random.RandomState(456)
            for trial in range(3):
                if time.time() > deadline - 10:
                    break
                # Larger perturbation
                perturbed = best_params + rng.randn(*best_params.shape) * 0.1
                result = minimize(
                    _neg_area,
                    perturbed,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 300,
                        'maxfev': 300,
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
            pass
    
    waypoints = best_params.reshape(N_WAYPOINTS, 3)
    
    if not np.all(np.isfinite(waypoints)):
        ts = np.linspace(0, 1, N_WAYPOINTS)
        waypoints = np.column_stack([
            -1.5 * ts,
            0.4 * np.sin(np.pi * ts),
            90.0 * ts
        ])
    
    return waypoints

# EVOLVE-BLOCK-END
