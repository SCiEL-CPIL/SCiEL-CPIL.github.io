"""Combined packing strategy: Langevin dynamics + smooth-min refinement + binary search."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE_MARGIN = 18.0  # seconds before hard limit


def _pairwise_indices(n):
    """Return indices for all pairs i < j."""
    return np.triu_indices(n, 1)


def _score_vec(points):
    """Compute the packing radius for given centers."""
    n = len(points)
    radii = 1.0 - np.linalg.norm(points, axis=1)
    r = np.min(radii)
    i_idx, j_idx = _pairwise_indices(n)
    pair_dists = np.linalg.norm(points[i_idx] - points[j_idx], axis=1) / 2.0
    r = min(r, np.min(pair_dists))
    return r


def _softmin_logsumexp(values, beta):
    """Numerically-stable soft-min via log-sum-exp."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0
    neg = -values * beta
    neg_max = np.max(neg)
    if not np.isfinite(neg_max):
        return 0.0
    exp_sum = np.sum(np.exp(neg - neg_max))
    if exp_sum <= 0.0:
        return 0.0
    return -(neg_max + np.log(exp_sum)) / beta


def _smooth_objective(x, beta):
    """Smooth objective for optimization: minimize negative soft-min of margins."""
    pts = x.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms
    i_idx, j_idx = _pairwise_indices(N)
    diffs = pts[i_idx] - pts[j_idx]
    pair_half = np.linalg.norm(diffs, axis=1) / 2.0
    all_vals = np.concatenate([pair_half, boundary])
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) == 0:
        return 1e10
    return -_softmin_logsumexp(all_vals, beta)


def _smooth_grad(x, beta):
    """Gradient of the smooth objective."""
    pts = x.reshape(N, 2)
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms
    i_idx, j_idx = _pairwise_indices(N)
    diffs = pts[i_idx] - pts[j_idx]
    dists = np.linalg.norm(diffs, axis=1)
    pair_half = dists / 2.0
    all_vals = np.concatenate([pair_half, boundary])
    
    neg = -all_vals * beta
    neg_max = np.max(neg)
    exp_terms = np.exp(neg - neg_max)
    sum_exp = np.sum(exp_terms)
    
    if sum_exp <= 0:
        return np.zeros_like(x)
    
    # d(-softmin)/d(all_vals) = exp_terms / sum_exp
    weights = exp_terms / sum_exp
    
    grad = np.zeros_like(pts)
    
    # Boundary terms
    for i in range(N):
        if norms[i] > 1e-10:
            grad[i] += weights[N + i] * (-pts[i] / norms[i])
    
    # Pairwise terms
    for k in range(len(i_idx)):
        i, j = i_idx[k], j_idx[k]
        d = dists[k]
        if d > 1e-10:
            factor = weights[k] / (2.0 * d)
            grad[i] += factor * diffs[k]
            grad[j] -= factor * diffs[k]
    
    return grad.flatten()


def _anneal_points(pts, deadline, max_betas=10):
    """Multi-stage annealing optimization on full point coordinates."""
    current = pts.copy()
    current_score = _score_vec(current)
    
    betas = [5.0, 15.0, 40.0, 100.0, 250.0, 600.0, 1500.0, 4000.0, 10000.0, 25000.0]
    betas = betas[:max_betas]
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(
                _smooth_objective,
                current.ravel(),
                args=(beta,),
                jac=lambda x, b: _smooth_grad(x, b),
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'ftol': 1e-16,
                    'gtol': 1e-14,
                    'maxfun': 20000
                }
            )
            new_pts = res.x.reshape(N, 2)
            if not np.all(np.isfinite(new_pts)):
                continue
            s = _score_vec(new_pts)
            if s > current_score:
                current = new_pts
                current_score = s
        except Exception:
            pass
            
    return current, current_score


def _langevin_dynamics(pts, r_target, deadline, steps=20000, dt=0.001, gamma=0.1):
    """
    Langevin dynamics for hard spheres in a harmonic potential.
    Uses velocity Verlet with thermostat (Langevin) noise.
    """
    n = len(pts)
    x = pts.copy()
    v = np.zeros_like(x)
    
    # Parameters
    k_wall = 1000.0  # Harmonic wall potential stiffness
    k_repel = 5000.0  # Repulsion stiffness
    
    best_x = x.copy()
    best_score = _score_vec(x)
    
    for step in range(steps):
        if time.time() > deadline - 3:
            break
        
        # Compute forces
        f = np.zeros_like(x)
        
        # Wall repulsion: push inward if outside radius (1 - r_target)
        norms = np.linalg.norm(x, axis=1)
        wall_r = 1.0 - r_target
        for i in range(n):
            if norms[i] > wall_r:
                # Force pulling back toward center
                direction = x[i] / (norms[i] + 1e-15)
                excess = norms[i] - wall_r
                f[i] -= k_wall * excess * direction
        
        # Pair repulsion
        i_idx, j_idx = _pairwise_indices(n)
        diffs = x[i_idx] - x[j_idx]
        dists = np.linalg.norm(diffs, axis=1)
        min_dist = 2.0 * r_target
        for k in range(len(i_idx)):
            d = dists[k]
            if d < min_dist and d > 1e-15:
                i, j = i_idx[k], j_idx[k]
                direction = diffs[k] / d
                excess = min_dist - d
                f[i] += k_repel * excess * direction
                f[j] -= k_repel * excess * direction
        
        # Langevin update (velocity Verlet with thermostat)
        # v_new = v + f*dt + random_noise
        noise = np.random.randn(n, 2) * np.sqrt(2 * gamma * dt)
        v = v + f * dt + noise
        v *= (1.0 - gamma * dt)  # Damping
        
        # Position update
        x = x + v * dt
        
        # Project back to valid region (hard constraint)
        norms = np.linalg.norm(x, axis=1)
        for i in range(n):
            if norms[i] > 1.0 - r_target + 1e-10:
                x[i] *= (1.0 - r_target) / (norms[i] + 1e-15)
        
        # Track best
        s = _score_vec(x)
        if s > best_score:
            best_score = s
            best_x = x.copy()
        
        # Convergence check
        if best_score >= r_target + 1e-10:
            break
    
    return best_x, best_score


def _constraint_push(pts, r_target, deadline, max_iters=300):
    """Polish solution by iteratively pushing violating constraints."""
    pts = pts.copy()
    for _ in range(max_iters):
        if time.time() > deadline - 2:
            break
        
        norms = np.linalg.norm(pts, axis=1)
        boundary_margins = 1.0 - norms
        
        i_idx, j_idx = _pairwise_indices(len(pts))
        diffs = pts[i_idx] - pts[j_idx]
        pair_margins = np.linalg.norm(diffs, axis=1) / 2.0
        
        min_pair = np.min(pair_margins)
        min_bound = np.min(boundary_margins)
        
        if min_pair < min_bound - 1e-14:
            # Pair constraint is binding
            k = int(np.argmin(pair_margins))
            i, j = i_idx[k], j_idx[k]
            di = pts[i] - pts[j]
            dist = np.linalg.norm(di)
            if dist < 1e-15:
                break
            di /= dist
            push = (2.0 * r_target - dist) / 2.0
            if push > 1e-15:
                pts[i] += push * di
                pts[j] -= push * di
        else:
            # Boundary constraint is binding
            i = int(np.argmin(boundary_margins))
            pi = pts[i]
            dist = np.linalg.norm(pi)
            if dist < 1e-15:
                pts[i] += 1e-4 * np.array([1.0, 0.0])
            else:
                di = pi / dist
                push = (1.0 - r_target - dist)
                if push > 1e-15:
                    pts[i] += push * di
        
        # Check convergence
        s = _score_vec(pts)
        if s >= r_target - 1e-12:
            break
    
    return pts


def _binary_search_polish(pts, deadline):
    """Binary search for maximum feasible radius with projection polish."""
    current = pts.copy()
    low = _score_vec(current) - 1e-8
    high = _score_vec(current) + 1e-4
    
    for _ in range(25):
        if time.time() > deadline - 2:
            break
        mid = (low + high) / 2.0
        polished = _constraint_push(current, mid, deadline, max_iters=150)
        s = _score_vec(polished)
        if s >= mid - 1e-10:
            low = mid
            current = polished
        else:
            high = mid
    
    return current, _score_vec(current)


def _ring_init():
    """Create initial ring configuration: 1 + 6 + 12 + 18 + 13."""
    ring_sizes = [1, 6, 12, 18, 13]
    ring_radii = [0.0, 0.25, 0.45, 0.65, 0.85]
    
    points = [np.array([0.0, 0.0])]
    for i in range(1, len(ring_sizes)):
        r = ring_radii[i]
        n = ring_sizes[i]
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        ring_pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        points.append(ring_pts)
    
    return np.vstack(points)


def _hex_init(scale=0.13):
    """Generate hexagonal lattice points and filter to fit in unit disk."""
    points = []
    for i in range(-12, 13):
        for j in range(-12, 13):
            x = scale * (i + 0.5 * j)
            y = scale * (np.sqrt(3) / 2 * j)
            if x**2 + y**2 <= 0.85:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) == 0:
        return _ring_init()
    
    dists = np.sqrt(np.sum(points**2, axis=1))
    idx = np.argsort(dists)
    points = points[idx[:N]]
    return points


def construct():
    t0 = time.time()
    deadline = t0 + 120.0 - DEADLINE_MARGIN
    
    best_points = None
    best_score = -1.0
    
    # ---- Phase 1: Multiple initializations ----
    candidates = []
    
    ring_pts = _ring_init()
    ring_score = _score_vec(ring_pts)
    if ring_score > 0:
        candidates.append((ring_pts.copy(), ring_score))
    
    hex_pts = _hex_init(scale=0.13)
    hex_score = _score_vec(hex_pts)
    if hex_score > 0:
        candidates.append((hex_pts.copy(), hex_score))
    
    # Slightly perturbed ring
    ring_pert = _ring_init() + np.random.randn(N, 2) * 0.01
    ring_pert_score = _score_vec(ring_pert)
    if ring_pert_score > 0:
        candidates.append((ring_pert.copy(), ring_pert_score))
    
    # Select best candidate
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_points = candidates[0][0].copy()
        best_score = candidates[0][1]
    else:
        best_points = _ring_init()
        best_score = _score_vec(best_points)
    
    # ---- Phase 2: Langevin dynamics to find good packing ----
    if time.time() < deadline - 40:
        # Use current best score as target radius for Langevin
        r_target = best_score * 1.05  # Slightly above current
        lang_pts, lang_score = _langevin_dynamics(
            best_points, r_target, deadline - 40,
            steps=15000, dt=0.0005, gamma=0.05
        )
        if lang_score > best_score:
            best_score = lang_score
            best_points = lang_pts.copy()
    
    # ---- Phase 3: Smooth-min annealing refinement ----
    if time.time() < deadline - 20:
        annealed_pts, annealed_score = _anneal_points(
            best_points, deadline - 20, max_betas=8
        )
        if annealed_score > best_score:
            best_score = annealed_score
            best_points = annealed_pts.copy()
    
    # ---- Phase 4: Binary search polish ----
    if time.time() < deadline - 5:
        polished_pts, polished_score = _binary_search_polish(
            best_points, deadline - 5
        )
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_pts.copy()
    
    # ---- Phase 5: Multi-start with Langevin from different inits ----
    if time.time() < deadline - 10:
        for init_pts, init_score in candidates:
            if time.time() > deadline - 10:
                break
            r_target = max(init_score * 1.03, best_score * 0.98)
            lang_pts, lang_score = _langevin_dynamics(
                init_pts, r_target, deadline - 10,
                steps=8000, dt=0.0005, gamma=0.05
            )
            if lang_score > best_score:
                best_score = lang_score
                best_points = lang_pts.copy()
                # Quick polish
                if time.time() < deadline - 5:
                    polished_pts, polished_score = _binary_search_polish(
                        best_points, deadline - 5
                    )
                    if polished_score > best_score:
                        best_score = polished_score
                        best_points = polished_pts.copy()
    
    # Final validity check
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = _ring_init()
    
    return best_points
