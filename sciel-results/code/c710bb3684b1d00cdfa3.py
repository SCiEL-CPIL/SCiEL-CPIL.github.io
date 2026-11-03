"""Packing 50 equal circles in a unit circle using soft-min energy with ring structures."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 138.0  # safety margin before 150s kill


def _score(P):
    """Calculate the radius score for a given set of points."""
    P = np.asarray(P, dtype=np.float64)
    if P.shape != (N, 2) or not np.all(np.isfinite(P)):
        return 0.0
    d_bound = 1.0 - np.linalg.norm(P, axis=1)
    r_bound = np.min(d_bound)
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    r_pair = min_pair_dist / 2.0
    return max(0.0, min(r_pair, r_bound))


def _softmin_energy(P, r, beta=50.0):
    """
    Smooth energy using log-sum-exp soft-min of constraint violations.
    Low beta = smoother, high beta = closer to true min.
    """
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    dist_origin = np.linalg.norm(P, axis=1)
    
    # Violations: we want dist_ij >= 2r and dist_origin <= 1-r
    cc_viol = 2.0 * r - dist  # positive means violation
    cb_viol = dist_origin - (1.0 - r)  # positive means violation
    
    # Soft-min of violations (we want to minimize the maximum violation)
    # Use: -log(sum(exp(-beta * viol))) / beta  approximates min(viol)
    # But we want to penalize violations, so we use:
    # If viol > 0, we penalize. If viol <= 0, no penalty.
    # Smooth approximation: softplus(beta * viol) / beta
    
    # For circle-circle: penalize when cc_viol > 0
    cc_pen = np.sum(np.log1p(np.exp(beta * cc_viol)) / beta)
    # For circle-boundary: penalize when cb_viol > 0
    cb_pen = np.sum(np.log1p(np.exp(beta * cb_viol)) / beta)
    
    return cc_pen + cb_pen


def _softmin_energy_grad(P, r, beta=50.0):
    """Gradient of the soft-min energy."""
    grad = np.zeros_like(P)
    eps = 1e-15
    
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    dist_origin = np.linalg.norm(P, axis=1)
    
    cc_viol = 2.0 * r - dist
    cb_viol = dist_origin - (1.0 - r)
    
    # Gradient of log1p(exp(beta*v))/beta w.r.t. P
    # d/dP [log1p(exp(beta*v))/beta] = exp(beta*v)/(1+exp(beta*v)) * dv/dP
    # = sigmoid(beta*v) * dv/dP  (where sigmoid(x) = exp(x)/(1+exp(x)))
    
    def sigmoid(x):
        # Numerically stable sigmoid
        out = np.zeros_like(x)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        exp_neg = np.exp(x[~pos])
        out[~pos] = exp_neg / (1.0 + exp_neg)
        return out
    
    # Circle-circle gradient
    sig_cc = sigmoid(beta * cc_viol)
    inv_dist = np.zeros_like(dist)
    np.divide(1.0, dist, out=inv_dist, where=dist > eps)
    
    # d cc_viol / d P_i = -d dist_ij / d P_i = (P_j - P_i) / dist_ij
    # So d energy / d P_i = sum_j sig_cc[i,j] * (P_j - P_i) / dist_ij
    W_cc = sig_cc * inv_dist
    # For each i: sum_j W_cc[i,j] * (P_j - P_i)
    # = sum_j W_cc[i,j] * P_j - P_i * sum_j W_cc[i,j]
    sum_W_cc = np.sum(W_cc, axis=1)
    term1 = (P.T @ W_cc).T
    term2 = P * sum_W_cc[:, np.newaxis]
    grad += term1 - term2
    
    # Circle-boundary gradient
    sig_cb = sigmoid(beta * cb_viol)
    inv_dist_origin = np.zeros_like(dist_origin)
    np.divide(1.0, dist_origin, out=inv_dist_origin, where=dist_origin > eps)
    
    # d cb_viol / d P_i = P_i / dist_origin_i
    grad += (sig_cb * inv_dist_origin)[:, np.newaxis] * P
    
    return grad


def _ring_to_points(radii, offsets, counts):
    """Convert ring parameters to point positions."""
    P = np.zeros((N, 2))
    count = 0
    for i, n_pts in enumerate(counts):
        if n_pts == 1:
            P[count] = [0.0, 0.0]
            count += 1
        else:
            R = max(0.0, radii[i])
            offset = offsets[i]
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False) + offset
            P[count:count+n_pts, 0] = R * np.cos(angles)
            P[count:count+n_pts, 1] = R * np.sin(angles)
            count += n_pts
    return P


def _hexagonal_init():
    """Generate hexagonal lattice points inside unit circle."""
    spacing = 2.0 / (np.sqrt(N) + 1.5)
    while True:
        k = int(1.0 / spacing) + 3
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        margin = spacing * 0.5
        mask = radius <= 1.0 - margin
        points = points[mask]
        radius = radius[mask]
        if len(points) >= N:
            indices = np.argsort(radius, kind="stable")
            return points[indices[:N]]
        spacing *= 0.97


def _anneal_points(P, r, beta_start=10.0, beta_end=200.0, steps=500, lr=0.005, time_limit=10.0):
    """
    Simulated-annealing-style gradient descent with increasing beta.
    Uses Adam-like momentum for better convergence.
    """
    t_start = time.time()
    P = P.copy()
    m = np.zeros_like(P)
    v = np.zeros_like(P)
    beta = beta_start
    lr_schedule = lr
    
    for step in range(steps):
        if time.time() - t_start > time_limit:
            break
        
        # Linearly increase beta
        beta = beta_start + (beta_end - beta_start) * (step / max(1, steps - 1))
        
        E = _softmin_energy(P, r, beta)
        G = _softmin_energy_grad(P, r, beta)
        
        # Simple momentum update
        m = 0.9 * m + 0.1 * G
        v = 0.99 * v + 0.01 * G**2
        P -= lr_schedule * m / (np.sqrt(v) + 1e-8)
        
        # Project points to stay within unit circle
        norms = np.linalg.norm(P, axis=1)
        max_norm = 1.0 - r + 0.01
        mask = norms > max_norm
        if np.any(mask):
            P[mask] *= max_norm / norms[mask, np.newaxis]
    
    return P


def _constraint_push(P, r):
    """
    Directly resolve constraint violations by pushing points apart.
    """
    P = P.copy()
    for _ in range(20):
        # Check circle-circle
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        min_dist = np.min(dist)
        if min_dist >= 2.0 * r - 1e-10:
            break
        
        # Find closest pair and push them apart
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        if i > j:
            i, j = j, i
        
        d = P[i] - P[j]
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-10:
            d = np.array([1.0, 0.0])
            d_norm = 1.0
        d_hat = d / d_norm
        
        # Push apart by half the overlap each
        overlap = 2.0 * r - d_norm
        if overlap > 0:
            P[i] += 0.5 * overlap * d_hat
            P[j] -= 0.5 * overlap * d_hat
        
        # Check boundary
        dist_origin = np.linalg.norm(P, axis=1)
        for k in range(N):
            if dist_origin[k] > 1.0 - r:
                scale = (1.0 - r) / dist_origin[k]
                P[k] *= scale
    
    return P


def _optimize_for_r(P_init, r, time_limit=8.0):
    """
    Optimize positions for a fixed target radius r.
    Returns (P_opt, energy).
    """
    P = P_init.copy()
    best_P = P.copy()
    best_E = _softmin_energy(P, r, beta=50.0)
    
    # Phase 1: Anneal with smooth energy
    P_annealed = _anneal_points(P, r, beta_start=10.0, beta_end=100.0, steps=300, lr=0.004, time_limit=time_limit * 0.5)
    E_annealed = _softmin_energy(P_annealed, r, beta=100.0)
    if E_annealed < best_E:
        best_P = P_annealed
        best_E = E_annealed
    
    # Phase 2: Constraint push to resolve remaining violations
    P_pushed = _constraint_push(P_annealed, r)
    E_pushed = _softmin_energy(P_pushed, r, beta=200.0)
    if E_pushed < best_E:
        best_P = P_pushed
        best_E = E_pushed
    
    # Phase 3: Fine anneal with high beta
    if time.time() - time_limit < time_limit * 0.3:
        P_fine = _anneal_points(best_P, r, beta_start=50.0, beta_end=300.0, steps=200, lr=0.001, time_limit=time_limit * 0.3)
        E_fine = _softmin_energy(P_fine, r, beta=300.0)
        if E_fine < best_E:
            best_P = P_fine
            best_E = E_fine
        
        # Final constraint push
        P_final = _constraint_push(P_fine, r)
        E_final = _softmin_energy(P_final, r, beta=300.0)
        if E_final < best_E:
            best_P = P_final
            best_E = E_final
    
    return best_P, best_E


def _binary_search_polish(P_init, r_low, r_high, max_iters=10, time_limit=60.0):
    """
    Binary search on r, optimizing positions for each trial r.
    Returns best P and score found.
    """
    t_start = time.time()
    best_P = P_init.copy()
    best_score = _score(P_init)
    
    for _ in range(max_iters):
        if time.time() - t_start > time_limit:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        P_opt, E = _optimize_for_r(P_init, r_mid, time_limit=min(6.0, time_limit - (time.time() - t_start)))
        
        score_opt = _score(P_opt)
        
        if E < 1e-6 and score_opt >= r_mid - 1e-9:
            # Feasible: r_mid is achievable
            if score_opt > best_score:
                best_P = P_opt.copy()
                best_score = score_opt
            r_low = score_opt
        else:
            # Not feasible: r_mid too large
            r_high = r_mid
        
        P_init = best_P.copy()
    
    return best_P, best_score


def construct():
    t_start = time.time()
    
    # Start with hexagonal initialization
    P_best = _hexagonal_init()
    score_best = _score(P_best)
    
    # Define ring structures for N=50
    ring_structures = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 14, 17],
        [1, 6, 12, 16, 15],
        [1, 6, 12, 15, 16],
        [1, 6, 10, 14, 19],
        [1, 6, 10, 12, 21],
        [1, 6, 12, 17, 14],
        [1, 6, 13, 13, 17],
    ]
    
    # Try ring-based initializations
    for counts in ring_structures:
        if time.time() - t_start > DEADLINE - 40:
            break
        
        n_rings = len(counts)
        # Initial radii: evenly spaced
        radii0 = np.linspace(0, 0.85, n_rings)
        radii0[0] = 0.0
        offsets0 = np.zeros(n_rings)
        
        P_ring = _ring_to_points(radii0, offsets0, counts)
        score_ring = _score(P_ring)
        
        if score_ring > score_best:
            P_best = P_ring.copy()
            score_best = score_ring
    
    # Binary search polish starting from current best
    r_low = score_best
    r_high = min(0.14, score_best * 1.15)
    
    remaining_time = DEADLINE - (time.time() - t_start)
    if remaining_time > 20:
        P_best, score_best = _binary_search_polish(
            P_best, r_low, r_high,
            max_iters=8,
            time_limit=remaining_time - 15
        )
    
    # Final polish: direct constraint resolution and fine-tuning
    if time.time() - t_start < DEADLINE - 5:
        r_current = _score(P_best)
        P_polished = _constraint_push(P_best, r_current * 0.999)
        score_polished = _score(P_polished)
        if score_polished > score_best:
            P_best = P_polished.copy()
            score_best = score_polished
        
        # One more anneal pass with very high beta
        P_fine = _anneal_points(P_best, score_best * 0.999, beta_start=100.0, beta_end=500.0, steps=100, lr=0.0005, time_limit=5.0)
        P_fine = _constraint_push(P_fine, score_best * 0.999)
        score_fine = _score(P_fine)
        if score_fine > score_best:
            P_best = P_fine.copy()
            score_best = score_fine
    
    # Ensure validity: clamp points
    r_final = _score(P_best)
    if r_final > 0:
        r2 = np.hypot(P_best[:, 0], P_best[:, 1])
        for i in range(N):
            if r2[i] > 1 - r_final + 1e-10:
                scale = (1 - r_final) / r2[i]
                P_best[i] *= scale
    
    return P_best.reshape(N, 2)
