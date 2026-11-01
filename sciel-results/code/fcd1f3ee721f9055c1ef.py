"""Circle packing in unit square n=32 - NLP interior-point approach."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 32
DEADLINE = 150.0  # wall-clock limit with margin


def _hex_initialization(n):
    """Hexagonal close-packing initialization."""
    pts = []
    row = 0
    while len(pts) < n:
        x_off = (row % 2) * 0.5 * 0.55
        for col in range(20):
            x = 0.05 + col * 0.55 + x_off
            y = 0.05 + row * 0.476
            if 0 < x < 1 and 0 < y < 1:
                pts.append((x, y))
            if len(pts) >= n:
                break
        row += 1
    return np.array(pts[:n])


def _random_initialization(n, seed):
    """Random initialization with minimum spacing."""
    rng = np.random.RandomState(seed)
    pts = []
    attempts = 0
    while len(pts) < n and attempts < 5000:
        x, y = rng.uniform(0.05, 0.95, 2)
        ok = True
        for px, py in pts:
            if (x - px)**2 + (y - py)**2 < 0.01:
                ok = False
                break
        if ok:
            pts.append((x, y))
        attempts += 1
    # Fill remaining randomly
    while len(pts) < n:
        pts.append(tuple(rng.uniform(0.05, 0.95, 2)))
    return np.array(pts[:n])


def _grid_initialization(n):
    """Perturbed grid initialization."""
    side = int(np.ceil(np.sqrt(n)))
    pts = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            pts.append((x, y))
            if len(pts) >= n:
                break
        if len(pts) >= n:
            break
    return np.array(pts[:n])


def _softmin_objective(vars_vec, n, beta):
    """
    Objective: minimize -R + penalty for constraint violations.
    vars_vec = [x0,y0,...,x_{n-1},y_{n-1}, R]
    Constraints:
      - Pairwise: ||p_i - p_j|| >= 2R  (soft)
      - Boundary: R <= min(x_i, 1-x_i, y_i, 1-y_i) for all i
    """
    pts = vars_vec[:2*n].reshape(n, 2)
    R = vars_vec[2*n]
    
    # Pairwise distance violations: need dist_ij >= 2R
    # soft-min of (dist_ij - 2R) over all pairs
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    violations = 2*R - d  # positive when violated
    # Soft-min: -logsumexp(-violations)/beta ... actually we want to penalize max violation
    # Use: sum of exp(beta * max(0, violation)) as penalty
    pos_viol = np.maximum(violations, 0)
    pair_penalty = np.sum(np.exp(beta * pos_viol))
    
    # Boundary violations: R - min(x_i, 1-x_i, y_i, 1-y_i)
    margins = np.minimum(np.minimum(pts[:, 0], 1 - pts[:, 0]),
                         np.minimum(pts[:, 1], 1 - pts[:, 1]))
    bound_viol = np.maximum(R - margins, 0)
    bound_penalty = np.sum(np.exp(beta * bound_viol))
    
    # Objective: -R (we maximize R) + penalties
    obj = -R + 0.01 * (pair_penalty + bound_penalty)
    return obj


def _pack_with_nlp(pts0, R0, beta, time_limit, tol):
    """Run NLP optimization from initial point."""
    n = len(pts0)
    x0 = np.concatenate([pts0.flatten(), [R0]])
    
    def obj(v):
        return _softmin_objective(v, n, beta)
    
    def obj_grad(v):
        # Numerical gradient for reliability
        eps = 1e-7
        g = np.zeros_like(v)
        for i in range(len(v)):
            vp = v.copy(); vp[i] += eps
            vm = v.copy(); vm[i] -= eps
            g[i] = (obj(vp) - obj(vm)) / (2 * eps)
        return g
    
    result = minimize(
        obj, x0, jac=obj_grad, method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': tol, 'gtol': tol, 'maxfun': 2000}
    )
    
    if result.success or result.fun < 0:
        pts = result.x[:2*n].reshape(n, 2)
        R = result.x[2*n]
        # Clamp to valid range
        R = max(0.001, min(R, 0.5))
        pts = np.clip(pts, [R, R], [1-R, 1-R])
        return pts, R
    return None, None


def _validate(pts, R):
    """Check if configuration is valid."""
    n = len(pts)
    if np.any(pts < 0) or np.any(pts > 1):
        return False
    if R <= 0:
        return False
    # Boundary
    if np.any(pts[:, 0] < R) or np.any(pts[:, 0] > 1 - R):
        return False
    if np.any(pts[:, 1] < R) or np.any(pts[:, 1] > 1 - R):
        return False
    # Pairwise
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 2 * R - 1e-9:
        return False
    return True


def _compute_score(pts, R):
    """Score = sum of radii = N * R for uniform packing."""
    return N * R


def construct():
    t0 = time.time()
    best_score = -np.inf
    best_result = None
    
    # Initial configurations
    inits = []
    inits.append(('hex', _hex_initialization(N)))
    inits.append(('grid', _grid_initialization(N)))
    inits.append(('rand0', _random_initialization(N, 42)))
    inits.append(('rand1', _random_initialization(N, 123)))
    inits.append(('rand2', _random_initialization(N, 999)))
    
    # Estimate initial R for each
    def estimate_R(pts):
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        min_d = np.min(d)
        margins = np.minimum(np.minimum(pts[:, 0], 1 - pts[:, 0]),
                             np.minimum(pts[:, 1], 1 - pts[:, 1]))
        return max(0.01, 0.5 * min(min_d, np.min(margins)) * 0.95)
    
    # Phase 1: Coarse optimization with moderate beta
    for name, pts0 in inits:
        if time.time() - t0 > DEADLINE - 30:
            break
        R0 = estimate_R(pts0)
        pts, R = _pack_with_nlp(pts0, R0, beta=5.0, time_limit=20, tol=1e-6)
        if pts is not None and _validate(pts, R):
            score = _compute_score(pts, R)
            if score > best_score:
                best_score = score
                best_result = (pts.copy(), R)
    
    # Phase 2: Refinement with higher beta on best result
    if best_result is not None:
        pts_best, R_best = best_result
        if time.time() - t0 < DEADLINE - 15:
            # Try higher beta for sharper constraints
            for beta in [10.0, 20.0, 50.0]:
                if time.time() - t0 > DEADLINE - 10:
                    break
                pts, R = _pack_with_nlp(pts_best, R_best, beta=beta, time_limit=15, tol=1e-8)
                if pts is not None and _validate(pts, R):
                    score = _compute_score(pts, R)
                    if score > best_score:
                        best_score = score
                        best_result = (pts.copy(), R)
                        pts_best, R_best = pts, R
    
    # Phase 3: Final polish with very tight tolerance
    if best_result is not None and time.time() - t0 < DEADLINE - 5:
        pts_best, R_best = best_result
        pts, R = _pack_with_nlp(pts_best, R_best, beta=100.0, time_limit=10, tol=1e-10)
        if pts is not None and _validate(pts, R):
            score = _compute_score(pts, R)
            if score > best_score:
                best_score = score
                best_result = (pts.copy(), R)
    
    # Fallback to seed if nothing worked
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    pts, R = best_result
    # Build output: (x, y, radius)
    result = np.column_stack([pts[:, 0], pts[:, 1], np.full(N, R)])
    return result


# EVOLVE-BLOCK-START
# (The entire construct() and helper functions above constitute the evolve block)
# EVOLVE-BLOCK-END
