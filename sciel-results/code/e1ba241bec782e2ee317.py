"""Packing 50 equal circles in a unit circle via ring-based parametric optimization + smooth polish + constraint push + binary search."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
SEED = 42
DEADLINE_MARGIN = 15.0


def _pairwise_distances(pts):
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _score(pts):
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    min_boundary = np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1)))
    return max(min(min_pair, min_boundary), 0.0)


def _softmin_logsumexp(values, beta):
    if len(values) == 0:
        return 0.0
    neg = -values * beta
    neg_max = np.max(neg)
    if not np.isfinite(neg_max):
        return 0.0
    exp_sum = np.sum(np.exp(neg - neg_max))
    if exp_sum == 0:
        return 0.0
    return -(neg_max + np.log(exp_sum)) / beta


def _smooth_objective(x, beta=50.0):
    pts = x.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    boundary = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    pair_half = d / 2.0
    all_vals = np.concatenate([pair_half.ravel(), boundary])
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) == 0:
        return 1e10
    soft_min = _softmin_logsumexp(all_vals, beta)
    return -soft_min


def _smooth_grad(x, beta=50.0):
    """Analytical gradient of the smooth (soft-min) objective."""
    pts = x.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return np.zeros_like(x)
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    boundary = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    pair_half = d / 2.0
    all_vals = np.concatenate([pair_half.ravel(), boundary])
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) == 0:
        return np.zeros_like(x)

    # Soft-min gradient: d(soft_min)/d(val_k) = -exp(-beta * val_k) / sum(exp(-beta * val_j))
    neg = -all_vals * beta
    neg_max = np.max(neg)
    exp_vals = np.exp(neg - neg_max)
    exp_sum = np.sum(exp_vals)
    if exp_sum == 0:
        return np.zeros_like(x)
    weights = exp_vals / exp_sum  # sum to 1, weights for each constraint

    # Gradient of soft_min w.r.t. each value
    # soft_min = -(neg_max + log(exp_sum)) / beta
    # d(soft_min)/d(val_k) = -exp(-beta*val_k) / (beta * sum(exp(-beta*val_j)))
    # = -weights[k] / beta  ... wait, let me re-derive
    # soft_min = -max_j(-beta*val_j) / beta - log(sum(exp(-beta*val_j - max(-beta*val)))) / beta
    # Actually: soft_min = -log(sum(exp(-beta*val_j))) / beta + max_j(val_j) ... no
    # soft_min = -(max(neg) + log(sum(exp(neg - max(neg)))))/beta
    # d(soft_min)/d(val_k) = -d/d(val_k)[(max(neg) + log(sum(exp(neg-max(neg)))))/beta]
    # = -(1/beta) * d/d(val_k)[max(neg) + log(sum(exp(neg-max(neg))))]
    # max(neg) is piecewise linear, ignore its derivative (0 almost everywhere)
    # d/d(val_k) log(sum(exp(neg_j - max(neg)))) = d/d(val_k) log(sum(exp(-beta*val_j - max(neg))))
    # = (-beta * exp(-beta*val_k - max(neg))) / sum(exp(-beta*val_j - max(neg)))
    # = -beta * weights[k]
    # So d(soft_min)/d(val_k) = -(1/beta) * (-beta * weights[k]) = weights[k]
    # Wait that gives positive. Let me check: we want to MAXIMIZE soft_min, objective is -soft_min
    # d(-soft_min)/d(val_k) = -weights[k]
    # So gradient of objective w.r.t. val_k is -weights[k]

    grad = np.zeros_like(x)
    n_pairs = len(pair_half.ravel())

    # Pairwise contributions
    for idx in range(n_pairs):
        i = idx // N
        j = idx % N
        if i >= j:
            continue
        w = weights[idx]
        # d(pair_half)/d(pts[i]) = (pts[i] - pts[j]) / (2 * |pts[i]-pts[j]|)
        # d(pair_half)/d(pts[j]) = (pts[j] - pts[i]) / (2 * |pts[i]-pts[j]|)
        diff = pts[i] - pts[j]
        dist = np.linalg.norm(diff)
        if dist < 1e-15:
            continue
        unit = diff / dist
        # objective = -soft_min, d(obj)/d(pts[i]) = -w * d(pair_half)/d(pts[i]) = -w * unit / 2
        grad[2*i] += -w * unit[0] / 2.0
        grad[2*i+1] += -w * unit[1] / 2.0
        grad[2*j] += -w * (-unit[0]) / 2.0
        grad[2*j+1] += -w * (-unit[1]) / 2.0

    # Boundary contributions
    for idx in range(N):
        w = weights[n_pairs + idx]
        # boundary_i = 1 - |pts[i]|
        # d(boundary_i)/d(pts[i]) = -pts[i] / |pts[i]|
        norm_i = np.linalg.norm(pts[idx])
        if norm_i < 1e-15:
            continue
        unit = pts[idx] / norm_i
        # d(obj)/d(pts[i]) = -w * d(boundary_i)/d(pts[i]) = -w * (-unit) = w * unit
        grad[2*idx] += w * unit[0]
        grad[2*idx+1] += w * unit[1]

    return grad


def _local_search_polish(pts, deadline, max_iters=200):
    """Local search: iteratively move the most constrained point to improve the minimum constraint."""
    pts = pts.copy()
    current_score = _score(pts)

    for it in range(max_iters):
        if time.time() > deadline - 3:
            break

        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        pair_margins = d / 2.0
        boundary_margins = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))

        min_pair_margin = np.min(pair_margins)
        min_boundary_margin = np.min(boundary_margins)

        if min_pair_margin < min_boundary_margin - 1e-14:
            i, j = np.unravel_index(np.argmin(pair_margins), pair_margins.shape)
            if i == j:
                break
            di = pts[i] - pts[j]
            dist = np.linalg.norm(di)
            if dist < 1e-15:
                break
            di /= dist
            # Try small moves in random directions for both points
            best_move = None
            best_margin = min_pair_margin
            rng = np.random.RandomState(it + 42)
            for trial in range(20):
                angle = rng.uniform(0, 2 * np.pi)
                perp = np.array([-di[1], di[0]])
                for step in [0.005, 0.01, 0.02, 0.04]:
                    delta_i = step * (0.5 * di + 0.5 * perp * np.sin(angle))
                    delta_j = step * (0.5 * (-di) + 0.5 * (-perp) * np.sin(angle))
                    new_i = pts[i] + delta_i
                    new_j = pts[j] + delta_j
                    # Check if move helps
                    new_dij = np.linalg.norm(new_i - new_j) / 2.0
                    if new_dij > best_margin + 1e-12:
                        # Also check boundary
                        if np.linalg.norm(new_i) < 1.0 - 1e-10 and np.linalg.norm(new_j) < 1.0 - 1e-10:
                            # Quick check against all other points
                            ok = True
                            for k in range(N):
                                if k == i or k == j:
                                    continue
                                if np.linalg.norm(new_i - pts[k]) < 2 * best_margin - 1e-10:
                                    ok = False
                                    break
                                if np.linalg.norm(new_j - pts[k]) < 2 * best_margin - 1e-10:
                                    ok = False
                                    break
                            if ok:
                                best_margin = new_dij
                                best_move = (i, j, delta_i, delta_j)
            if best_move is not None:
                i, j, di_move, dj_move = best_move
                pts[i] += di_move
                pts[j] += dj_move
        else:
            i = int(np.argmin(boundary_margins))
            pi = pts[i]
            dist = np.linalg.norm(pi)
            if dist < 1e-15:
                pts[i] += 1e-4 * np.array([1.0, 0.0])
            else:
                di = pi / dist
                # Move inward
                for step in [0.005, 0.01, 0.02, 0.04]:
                    new_i = pi - step * di
                    if np.linalg.norm(new_i) < 1.0 - 1e-10:
                        ok = True
                        for k in range(N):
                            if k == i:
                                continue
                            if np.linalg.norm(new_i - pts[k]) < 2 * min_boundary_margin - 1e-10:
                                ok = False
                                break
                        if ok:
                            pts[i] = new_i
                            break

        new_score = _score(pts)
        if new_score > current_score + 1e-14:
            current_score = new_score
        elif it > 50:
            # No improvement for a while, try a different strategy
            break

    return pts, current_score


def _ring_to_points(params, ring_counts):
    """
    Convert ring parameters to points.
    params: [r1, r2, ..., rk, theta1, theta2, ..., thetak]
    Radii are sorted to avoid permutation issues.
    """
    n_rings = len(ring_counts)
    radii = params[:n_rings]
    offsets = params[n_rings:]
    radii = np.maximum(radii, 0.01)
    radii_sorted = np.sort(radii)
    
    centers = []
    for idx, (count, radius) in enumerate(zip(ring_counts, radii_sorted)):
        if count == 1 and radius < 1e-10:
            centers.append(np.array([0.0, 0.0]))
        else:
            offset = offsets[idx] if idx < len(offsets) else 0.0
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + offset
            xs = radius * np.cos(angles)
            ys = radius * np.sin(angles)
            for i in range(count):
                centers.append(np.array([xs[i], ys[i]]))
    
    pts = np.array(centers[:N])
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    for i in range(len(pts)):
        if norms[i] > 0.999:
            pts[i] *= 0.999 / norms[i]
    return pts


def _ring_objective(params, ring_counts):
    pts = _ring_to_points(params, ring_counts)
    return -_score(pts)


def _de_ring_search(ring_counts, deadline):
    """Differential evolution over ring parameters with manual implementation for time control."""
    n_rings = len(ring_counts)
    
    def neg_score(params):
        pts = _ring_to_points(params, ring_counts)
        return -_score(pts)
    
    # Bounds: radii in [0.05, 0.95], offsets in [0, 2*pi]
    bounds = []
    for i in range(n_rings):
        bounds.append((0.05, 0.95))
    for i in range(n_rings):
        bounds.append((0, 2 * np.pi))
    
    rng = np.random.RandomState(SEED)
    pop_size = 35
    max_gen = 80
    F = 0.8
    CR = 0.9
    
    pop = []
    for _ in range(pop_size):
        params = np.array([rng.uniform(b[0], b[1]) for b in bounds])
        pop.append(params)
    pop = np.array(pop)
    
    scores = np.array([neg_score(p) for p in pop])
    best_idx = np.argmin(scores)
    best_params = pop[best_idx].copy()
    best_score_val = scores[best_idx]
    
    for gen in range(max_gen):
        if time.time() > deadline - 30:
            break
        
        for i in range(pop_size):
            a_idx, b_idx, c_idx = rng.choice(pop_size, 3, replace=False)
            v = pop[a_idx] + F * (pop[b_idx] - pop[c_idx])
            for j in range(len(v)):
                v[j] = np.clip(v[j], bounds[j][0], bounds[j][1])
            
            trial = pop[i].copy()
            j_rand = rng.randint(len(v))
            for j in range(len(v)):
                if rng.rand() < CR or j == j_rand:
                    trial[j] = v[j]
            
            trial_score = neg_score(trial)
            if trial_score < scores[i]:
                pop[i] = trial
                scores[i] = trial_score
                if trial_score < best_score_val:
                    best_score_val = trial_score
                    best_params = trial.copy()
    
    return best_params, best_score_val


def _optimize_ring_params(params0, ring_counts, deadline):
    """Optimize ring parameters using L-BFGS-B."""
    try:
        result = minimize(
            _ring_objective,
            params0,
            args=(ring_counts,),
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        if result.fun < _ring_objective(params0, ring_counts):
            return result.x
    except Exception:
        pass
    return params0


def _anneal_points(pts, deadline, max_beta_stages=6):
    """Multi-stage annealing optimization from initial points with analytical gradient."""
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    betas = [5.0, 15.0, 40.0, 100.0, 250.0, 600.0]
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
        x0 = current_pts.ravel()
        try:
            result = minimize(
                _smooth_objective,
                x0,
                args=(beta,),
                jac=lambda x, b=beta: _smooth_grad(x, b),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-11, 'maxfun': 8000}
            )
            new_pts = result.x.reshape(N, 2)
            if not np.all(np.isfinite(new_pts)):
                continue
            new_score = _score(new_pts)
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
        except Exception:
            pass
    
    # High-precision polish with analytical gradient
    for beta in [1000.0, 2000.0, 5000.0, 10000.0]:
        if time.time() > deadline - 5:
            break
        x0 = current_pts.ravel()
        try:
            result = minimize(
                _smooth_objective,
                x0,
                args=(beta,),
                jac=lambda x, b=beta: _smooth_grad(x, b),
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxfun': 50000}
            )
            new_pts = result.x.reshape(N, 2)
            if not np.all(np.isfinite(new_pts)):
                continue
            new_score = _score(new_pts)
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
        except Exception:
            pass
    
    # Local search polish for final fine-tuning
    if time.time() < deadline - 3:
        current_pts, current_score = _local_search_polish(current_pts, deadline, max_iters=100)
    
    return current_pts, current_score


def _constraint_push(pts, r_target, deadline, max_iters=500):
    """Push points to achieve target radius via constraint satisfaction."""
    pts = pts.copy()
    
    for _ in range(max_iters):
        if time.time() > deadline - 3:
            break
        
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        pair_margins = d / 2.0
        boundary_margins = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
        
        min_pair_margin = np.min(pair_margins)
        min_boundary_margin = np.min(boundary_margins)
        
        if min_pair_margin < min_boundary_margin - 1e-14:
            i, j = np.unravel_index(np.argmin(pair_margins), pair_margins.shape)
            if i == j:
                break
            di = pts[i] - pts[j]
            dist = np.linalg.norm(di)
            if dist < 1e-15:
                break
            di /= dist
            push = (2 * r_target - dist) / 2.0
            if push > 1e-15:
                pts[i] += push * di
                pts[j] -= push * di
        else:
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
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        r = min(np.min(d) / 2.0, np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1))))
        if r >= r_target - 1e-12:
            break
    
    return pts


def _feasibility_check(pts, r):
    """Check if all circles of radius r fit with given centers."""
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    min_boundary = np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1)))
    return min(min_pair, min_boundary) >= r - 1e-10


def _binary_search_polish(pts, r_low, deadline, max_iters=30):
    """Binary search on radius with smooth optimization at each step."""
    current_pts = pts.copy()
    current_r = r_low
    
    r_high = r_low * 1.05  # Start with 5% higher target
    
    for _ in range(max_iters):
        if time.time() > deadline - 10:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        # Try to optimize to achieve r_mid
        polished_pts, polished_score = _anneal_points(current_pts, deadline)
        
        if polished_score >= r_mid - 1e-10:
            # Feasible, try higher
            r_low = r_mid
            current_pts = polished_pts
            current_r = polished_score
        else:
            # Not feasible, try lower
            r_high = r_mid
        
        # If convergence is tight, stop
        if r_high - r_low < 1e-8:
            break
    
    return current_pts, current_r


def _ring_initialization():
    """Generate ring-based initial configuration."""
    points = np.zeros((N, 2))
    points[0] = [0.0, 0.0]
    r1 = 0.25
    for i in range(6):
        angle = i * 2 * np.pi / 6
        points[1 + i] = [r1 * np.cos(angle), r1 * np.sin(angle)]
    r2 = 0.5
    for i in range(12):
        angle = (i + 0.5) * 2 * np.pi / 12
        points[7 + i] = [r2 * np.cos(angle), r2 * np.sin(angle)]
    r3 = 0.72
    for i in range(12):
        angle = i * 2 * np.pi / 12
        points[19 + i] = [r3 * np.cos(angle), r3 * np.sin(angle)]
    r4 = 0.88
    for i in range(19):
        angle = (i + 0.5) * 2 * np.pi / 19
        points[31 + i] = [r4 * np.cos(angle), r4 * np.sin(angle)]
    return points


def _hexagonal_initial():
    """Generate initial hexagonal lattice packing."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points, radius = points[mask], radius[mask]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    np.random.seed(SEED)
    
    best_points = None
    best_score = -1.0
    
    # Multiple ring topologies to try
    ring_topologies = [
        [1, 6, 12, 12, 19],
        [1, 6, 12, 13, 18],
        [1, 7, 12, 12, 18],
        [1, 6, 13, 12, 18],
    ]
    
    # Strategy 1: Differential Evolution over ring parameters (try multiple topologies)
    for ring_counts in ring_topologies:
        if time.time() > deadline - 50.0:
            break
        try:
            de_params, de_score_val = _de_ring_search(ring_counts, deadline)
            best_params = _optimize_ring_params(de_params, ring_counts, deadline)
            de_points = _ring_to_points(best_params, ring_counts)
            de_score = _score(de_points)
            
            if de_score > best_score:
                best_score = de_score
                best_points = de_points.copy()
        except Exception:
            pass
    
    # Strategy 2: Ring-based initialization + multi-stage smooth optimization
    if time.time() < deadline - 40.0:
        initial = _ring_initialization()
        result_points, result_score = _anneal_points(initial, deadline)
        
        if result_score > best_score:
            best_score = result_score
            best_points = result_points.copy()
    
    # Strategy 3: Hexagonal initialization + multi-stage smooth optimization
    if time.time() < deadline - 30.0:
        initial = _hexagonal_initial()
        result_points, result_score = _anneal_points(initial, deadline)
        
        if result_score > best_score:
            best_score = result_score
            best_points = result_points.copy()
    
    # Strategy 4: Random restarts with smooth optimization
    for restart in range(4):
        if time.time() > deadline - 25.0:
            break
        
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.random.uniform(0.05, 0.85, N)
        initial = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        result_points, result_score = _anneal_points(initial, deadline)
        
        if result_score > best_score:
            best_score = result_score
            best_points = result_points.copy()
    
    # Strategy 5: Binary search polish on the best solution
    if best_points is not None and time.time() < deadline - 15.0:
        r_target = best_score
        bs_points, bs_score = _binary_search_polish(best_points, r_target, deadline)
        if bs_score > best_score:
            best_score = bs_score
            best_points = bs_points.copy()
    
    # Final constraint push to improve the best solution
    if best_points is not None and time.time() < deadline - 5.0:
        r_target = best_score
        pushed = _constraint_push(best_points, r_target, deadline, max_iters=300)
        pushed_score = _score(pushed)
        if pushed_score > best_score:
            best_score = pushed_score
            best_points = pushed.copy()
        
        # One more round of smooth polish after push
        if time.time() < deadline - 3.0:
            final_points, final_score = _anneal_points(best_points, deadline)
            if final_score > best_score:
                best_score = final_score
                best_points = final_points.copy()
    
    # Final local search polish
    if best_points is not None and time.time() < deadline - 2.0:
        final_pts, final_score = _local_search_polish(best_points, deadline, max_iters=150)
        if final_score > best_score:
            best_score = final_score
            best_points = final_pts.copy()
    
    # Ensure valid output
    if best_points is None:
        best_points = _hexagonal_initial()
    
    best_points = np.array(best_points, dtype=np.float64)
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_points[mask] *= (1.0 - 1e-12) / norms[mask]
    
    return best_points
