import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 50
_DEADLINE = 140.0  # safety margin before 150s kill

def _score(centers):
    """Compute the radius for a set of centers. Returns -inf if invalid."""
    if centers.shape != (N, 2):
        return -np.inf
    if not np.all(np.isfinite(centers)):
        return -np.inf
    # Check inside square
    if np.any(centers[:, 0] < -0.5) or np.any(centers[:, 0] > 0.5):
        return -np.inf
    if np.any(centers[:, 1] < -0.5) or np.any(centers[:, 1] > 0.5):
        return -np.inf
    # Compute all pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    # Boundary distance
    boundary = min(0.5 - np.max(centers[:, 0]), 0.5 + np.min(centers[:, 0]),
                   0.5 - np.max(centers[:, 1]), 0.5 + np.min(centers[:, 1]))
    return min(min_pair_dist / 2.0, boundary)

def _soft_min_neg(centers, alpha=50.0):
    """Smooth surrogate: -soft_min of all pairwise distances and boundary distances.
    Lower is better (we minimize this)."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-30)
    np.fill_diagonal(dists, np.inf)
    # Soft min of pairwise distances
    log_sum = np.log(np.sum(np.exp(-alpha * dists)) + 1e-30)
    soft_min_pair = -np.log(np.sum(np.exp(-alpha * dists)) + 1e-30) / alpha
    # Boundary distances
    bx = 0.5 - np.abs(centers[:, 0])
    by = 0.5 - np.abs(centers[:, 1])
    bdist = np.concatenate([bx, by])
    soft_min_boundary = -np.log(np.sum(np.exp(-alpha * bdist)) + 1e-30) / alpha
    # Combined soft min
    combined = -np.log(np.exp(-alpha * soft_min_pair) + np.exp(-alpha * soft_min_boundary)) / alpha
    return -combined  # We want to maximize min, so minimize negative

def _neg_score(centers, alpha=50.0):
    """Objective to minimize: negative of the radius (with smooth surrogate for interior)."""
    c = centers.reshape(-1)
    if not np.all(np.isfinite(c)):
        return 1e6
    # Hard constraint penalty
    penalty = 0.0
    if np.any(c[0::2] < -0.5) or np.any(c[0::2] > 0.5) or np.any(c[1::2] < -0.5) or np.any(c[1::2] > 0.5):
        penalty = 10.0
    r = _score(centers)
    if r < 0:
        return 1e6 + penalty
    return -r + penalty

def _neg_score_smooth(centers, alpha=50.0):
    """Smooth objective for L-BFGS-B."""
    c = centers.reshape(-1)
    if not np.all(np.isfinite(c)):
        return 1e6
    # Small penalty for being outside
    penalty = 0.0
    out = (c[0::2] < -0.5) | (c[0::2] > 0.5) | (c[1::2] < -0.5) | (c[1::2] > 0.5)
    if np.any(out):
        penalty = 5.0 * np.sum(np.abs(np.clip(c[0::2], -0.5, 0.5) - c[0::2])**2 + 
                               np.abs(np.clip(c[1::2], -0.5, 0.5) - c[1::2])**2)
    val = _soft_min_neg(centers, alpha)
    return val + penalty

def _push_apart(centers, alpha=50.0, n_iters=500, lr=0.01):
    """Gradient-based refinement: push apart closest pairs, pull toward center."""
    c = centers.copy()
    for it in range(n_iters):
        if time.time() > _DEADLINE - 5:
            break
        # Compute forces
        forces = np.zeros_like(c)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-30)
        np.fill_diagonal(dists, 1e30)
        
        # Repulsion: stronger for closer pairs (soft-min gradient)
        # Gradient of soft_min is weighted average of unit vectors
        weights = np.exp(-alpha * dists)
        weights /= (np.sum(weights, axis=1, keepdims=True) + 1e-30)
        # Force on i from j: -w_ij * (c_i - c_j) / |c_i - c_j|
        unit = diff / (dists[:, :, np.newaxis] + 1e-30)
        forces -= np.sum(weights[:, :, np.newaxis] * unit, axis=1)
        
        # Boundary repulsion
        for k in range(N):
            for dim in range(2):
                if c[k, dim] > 0.5:
                    forces[k, dim] -= (c[k, dim] - 0.5) * 10
                elif c[k, dim] < -0.5:
                    forces[k, dim] -= (c[k, dim] + 0.5) * 10
        
        # Update
        c += lr * forces
        # Clip to square
        c[:, 0] = np.clip(c[:, 0], -0.5, 0.5)
        c[:, 1] = np.clip(c[:, 1], -0.5, 0.5)
    
    return c

def _make_hex_init(n_rows=8):
    """Create initial hexagonal-like arrangement."""
    centers = []
    if n_rows == 8:
        counts = [7, 7, 7, 7, 7, 7, 7, 1]
    elif n_rows == 7:
        counts = [8, 7, 7, 7, 7, 7, 7]
    else:
        counts = [7] * (N // 7) + [N % 7]
        n_rows = len(counts)
    
    dy = 1.0 / n_rows
    for i, n in enumerate(counts):
        if n == 0:
            continue
        dx = 1.0 / n
        y = -0.5 + (i + 0.5) * dy
        for j in range(n):
            x = -0.5 + (j + 0.5) * dx
            # Offset alternating rows
            if i % 2 == 1:
                x += dx / 2.0
            x = np.clip(x, -0.5, 0.5)
            centers.append([x, y])
    
    centers = np.array(centers[:N])
    if len(centers) < N:
        # Pad with random
        extra = np.random.uniform(-0.4, 0.4, (N - len(centers), 2))
        centers = np.vstack([centers, extra])
    return centers

def _hill_climb(centers, n_restarts=10, alpha=50.0):
    """Local refinement using L-BFGS-B on smooth objective."""
    best = centers.copy()
    best_score = _score(best)
    
    for restart in range(n_restarts):
        if time.time() > _DEADLINE - 10:
            break
        c = best.copy()
        # Add small perturbation for restarts > 0
        if restart > 0:
            c += np.random.normal(0, 0.01, c.shape)
            c[:, 0] = np.clip(c[:, 0], -0.49, 0.49)
            c[:, 1] = np.clip(c[:, 1], -0.49, 0.49)
        
        # L-BFGS-B optimization
        try:
            res = minimize(
                _neg_score_smooth,
                c.flatten(),
                args=(alpha,),
                method='L-BFGS-B',
                bounds=[(-0.499, 0.499)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            c_opt = res.x.reshape(N, 2)
            c_opt[:, 0] = np.clip(c_opt[:, 0], -0.5, 0.5)
            c_opt[:, 1] = np.clip(c_opt[:, 1], -0.5, 0.5)
            
            s = _score(c_opt)
            if s > best_score:
                best_score = s
                best = c_opt.copy()
        except Exception:
            pass
        
        # Push-apart refinement
        c_pushed = _push_apart(best, alpha=alpha, n_iters=300, lr=0.005)
        s_push = _score(c_pushed)
        if s_push > best_score:
            best_score = s_push
            best = c_pushed.copy()
    
    return best

def construct():
    start_time = time.time()
    
    # Phase 1: Generate diverse initial solutions
    inits = []
    
    # Hexagonal patterns
    for n_rows in [7, 8]:
        inits.append(_make_hex_init(n_rows))
    
    # Random initializations
    for _ in range(3):
        c = np.random.uniform(-0.45, 0.45, (N, 2))
        inits.append(c)
    
    # Grid-like
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
    inits.append(grid)
    
    # Phase 2: Optimize each initial solution
    best_centers = None
    best_radius = 0
    
    for i, init in enumerate(inits):
        if time.time() > _DEADLINE - 15:
            break
        
        # Hill climb from this init
        result = _hill_climb(init, n_restarts=5, alpha=50.0)
        r = _score(result)
        
        if r > best_radius:
            best_radius = r
            best_centers = result.copy()
    
    # Phase 3: Global optimization with differential_evolution on smooth objective
    if time.time() < _DEADLINE - 30:
        try:
            # Use a reduced parameterization: center + scale + rotation
            def _param_to_centers(params):
                cx, cy, scale, angle = params
                # Base hex pattern
                c = _make_hex_init(8)
                c = (c - np.array([cx, cy])) * scale
                # Rotate
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                c = c @ R.T + np.array([cx, cy])
                c[:, 0] = np.clip(c[:, 0], -0.5, 0.5)
                c[:, 1] = np.clip(c[:, 1], -0.5, 0.5)
                return c
            
            def _obj(params):
                c = _param_to_centers(params)
                return _neg_score_smooth(c, alpha=30.0)
            
            bounds = [(-0.1, 0.1), (-0.1, 0.1), (0.8, 1.3), (-0.3, 0.3)]
            de_result = differential_evolution(
                _obj, bounds, maxiter=50, popsize=12, seed=42,
                tol=1e-6, polish=False
            )
            c_de = _param_to_centers(de_result.x)
            # Refine with hill climb
            c_de = _hill_climb(c_de, n_restarts=3, alpha=50.0)
            r_de = _score(c_de)
            if r_de > best_radius:
                best_radius = r_de
                best_centers = c_de.copy()
        except Exception:
            pass
    
    # Phase 4: Final polish
    if best_centers is not None and time.time() < _DEADLINE - 5:
        final = _push_apart(best_centers, alpha=80.0, n_iters=500, lr=0.003)
        r_final = _score(final)
        if r_final > best_radius:
            best_radius = r_final
            best_centers = final
    
    if best_centers is not None:
        return best_centers
    
    # Fallback
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]
