"""Packing 50 equal circles in a unit circle using multi-stage optimization."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # safety margin before 150s kill


def _pairwise_indices(n):
    """Return array of (i, j) pairs for i < j."""
    iu = np.triu_indices(n, k=1)
    return np.column_stack((iu[0], iu[1]))


def _compute_score(P):
    """Calculate the radius score for a given set of points."""
    P = np.asarray(P)
    if P.shape != (N, 2) or not np.all(np.isfinite(P)):
        return 0.0
    
    # Distance to boundary
    d_bound = 1.0 - np.linalg.norm(P, axis=1)
    r_bound = np.min(d_bound)
    
    # Pairwise distances
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    r_pair = min_pair_dist / 2.0
    
    return max(0.0, min(r_pair, r_bound))


def _softmin_energy(P, r, beta=50.0):
    """
    Smooth energy using log-sum-exp soft-min of constraint violations.
    Lower energy = better packing. Returns 0 if all constraints satisfied.
    """
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    dist_origin = np.linalg.norm(P, axis=1)
    
    # Violations: how much each constraint is violated
    cc_viol = 2.0 * r - dist  # positive when overlapping
    cb_viol = dist_origin + r - 1.0  # positive when outside
    
    # Soft-min of violations using log-sum-exp
    # Use: softplus(beta * violation) / beta as smooth max(violation, 0)
    def smooth_max(v, b):
        return np.logaddexp(0.0, b * v) / b
    
    penalty_cc = np.sum(smooth_max(cc_viol, beta))
    penalty_cb = np.sum(smooth_max(cb_viol, beta))
    
    return penalty_cc + penalty_cb


def _softmin_grad(P, r, beta=50.0):
    """
    Gradient of the soft-min energy function.
    """
    grad = np.zeros_like(P)
    
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    dist_origin = np.linalg.norm(P, axis=1)
    
    cc_viol = 2.0 * r - dist
    cb_viol = dist_origin + r - 1.0
    
    # Gradient of softplus(beta * v) / beta = sigmoid(beta * v)
    # For cc_viol: d/dP_i [sigmoid(beta*(2r - dist_ij))] * (P_i - P_j)/dist_ij
    sig_cc = 1.0 / (1.0 + np.exp(-beta * cc_viol))
    sig_cc[~np.isfinite(sig_cc)] = 0.0
    
    inv_dist = np.zeros_like(dist)
    np.divide(1.0, dist, out=inv_dist, where=dist > 1e-12)
    
    W_cc = sig_cc * inv_dist
    sum_W_cc = np.sum(W_cc, axis=1)
    term1_cc = (P.T @ W_cc).T
    term2_cc = P * sum_W_cc[:, np.newaxis]
    grad += term1_cc - term2_cc
    
    # For cb_viol: d/dP_i [sigmoid(beta*(|P_i| + r - 1))] * P_i/|P_i|
    sig_cb = 1.0 / (1.0 + np.exp(-beta * cb_viol))
    sig_cb[~np.isfinite(sig_cb)] = 0.0
    
    inv_dist_origin = np.zeros_like(dist_origin)
    np.divide(1.0, dist_origin, out=inv_dist_origin, where=dist_origin > 1e-12)
    
    grad += (sig_cb * inv_dist_origin)[:, np.newaxis] * P
    
    return grad


def _get_ring_initialization(params, ring_counts):
    """
    Generate points based on concentric rings.
    params: [R1, R2, ..., Rk, offset1, offset2, ..., offsetk]
    ring_counts: list of number of points in each ring.
    """
    P = np.zeros((N, 2))
    k = len(ring_counts)
    
    # Radii are the first k params
    radii = params[:k]
    # Offsets are the next k params
    offsets = params[k:2*k]
    
    count = 0
    for i, n_pts in enumerate(ring_counts):
        if n_pts == 1:
            # Center point
            P[count] = [0.0, 0.0]
            count += 1
        else:
            R = max(0.0, radii[i])
            offset = offsets[i]
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False) + offset
            P[count:count+n_pts, 0] = R * np.cos(angles)
            P[count:count+n_pts, 1] = R * np.sin(angles)
            count += 1 if n_pts == 1 else n_pts
            
    return P


def _optimize_positions_soft(P_init, r_target, time_limit=5.0, beta=50.0):
    """
    Optimize positions P using soft-min energy for a fixed r_target.
    Returns (P_opt, energy).
    """
    start_time = time.time()
    
    def objective(P_flat):
        P = P_flat.reshape(N, 2)
        return _softmin_energy(P, r_target, beta)
    
    def grad(P_flat):
        P = P_flat.reshape(N, 2)
        return _softmin_grad(P, r_target, beta).flatten()
        
    try:
        res = minimize(objective, P_init.flatten(), jac=grad, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
        P_opt = res.x.reshape(N, 2)
        min_energy = objective(P_opt)
        return P_opt, min_energy
    except Exception:
        return P_init, objective(P_init.flatten())


def _optimize_positions(P_init, r_target, time_limit=5.0):
    """
    Optimize positions P to minimize soft-min energy for a fixed r_target.
    Returns (P_opt, min_energy).
    """
    return _optimize_positions_soft(P_init, r_target, time_limit, beta=50.0)


def _get_hex_initial_points():
    """Fallback initialization using hexagonal lattice."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        margin = spacing / 2.0 
        points, radius = points[radius <= 1.0 - margin], radius[radius <= 1.0 - margin]
        if len(points) >= N:
            indices = np.argsort(radius, kind="stable")
            return points[indices[:N]]
        spacing *= 0.98


def _optimize_ring_structure(radii0, counts, angles_offset0=None, time_limit=30.0):
    """Optimize ring radii and angles for fixed counts using Nelder-Mead."""
    if angles_offset0 is None:
        angles_offset0 = [0.0] * len(counts)
    
    n_rings = len(counts)
    
    def build_and_score(params):
        radii = np.maximum(params[:n_rings], 0.0)
        angles = params[n_rings:]
        P = _get_ring_initialization(np.concatenate([radii, angles]), counts)
        return -_compute_score(P)  # minimize negative score
    
    t_start = time.time()
    try:
        x0 = np.concatenate([radii0, angles_offset0])
        res = minimize(build_and_score, x0, method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
        
        if res.fun < build_and_score(x0):
            radii_opt = np.maximum(res.x[:n_rings], 0.0)
            angles_opt = res.x[n_rings:]
            return radii_opt, angles_opt
    except Exception:
        pass
    
    return radii0, angles_offset0


def _polish_l_bfgs(P, r, time_limit=10.0):
    """High-precision polish pass using L-BFGS-B with tight tolerances."""
    t_start = time.time()
    
    def objective(P_flat):
        P = P_flat.reshape(N, 2)
        return _softmin_energy(P, r, beta=100.0)
    
    def grad(P_flat):
        P = P_flat.reshape(N, 2)
        return _softmin_grad(P, r, beta=100.0).flatten()
    
    try:
        res = minimize(objective, P.flatten(), jac=grad, method='L-BFGS-B',
                       options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13})
        P_opt = res.x.reshape(N, 2)
        return P_opt
    except Exception:
        return P


def _anneal_langevin(P_init, r, time_limit=20.0, n_steps=30000, dt=0.0005, T0=0.05, Tend=0.0005, seed=42):
    """
    Run Langevin dynamics with temperature annealing to find ground state.
    """
    t_start = time.time()
    rng = np.random.RandomState(seed)
    
    P = P_init.copy()
    V = np.zeros_like(P)
    
    best_score = _compute_score(P)
    best_P = P.copy()
    
    for step in range(n_steps):
        if time.time() - t_start > time_limit:
            break
        
        # Linear temperature annealing
        alpha = step / max(n_steps - 1, 1)
        kT = T0 * (Tend / T0) ** alpha
        
        # Compute forces
        F = np.zeros_like(P)
        
        # Pairwise repulsion
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, 1e12)
        
        k_rep = 500.0
        for i in range(N):
            for j in range(i + 1, N):
                d = dist[i, j]
                if d < 2.5 * r:
                    if d < 1e-10:
                        d = 1e-10
                        diff_ij = np.array([1.0, 0.0])
                    else:
                        diff_ij = diff[i, j] / d
                    if d < 2.0 * r:
                        f_mag = k_rep * (2.0 * r - d)
                        F[i] += f_mag * diff_ij
                        F[j] -= f_mag * diff_ij
        
        # Wall potential
        dist_origin = np.linalg.norm(P, axis=1)
        k_wall = 50.0
        for i in range(N):
            if dist_origin[i] > (1.0 - r) and dist_origin[i] > 1e-10:
                F[i] += -k_wall * P[i]
        
        # Langevin update
        m = 1.0
        gamma = 1.0
        noise = rng.randn(N, 2)
        V = V + (F / m - gamma * V) * dt + np.sqrt(2.0 * gamma * kT / m) * np.sqrt(dt) * noise
        P = P + V * dt
        
        # Keep inside container with hard wall reflection
        dist_origin = np.linalg.norm(P, axis=1)
        for i in range(N):
            if dist_origin[i] > (1.0 - r):
                normal = P[i] / dist_origin[i]
                v_normal = np.dot(V[i], normal)
                if v_normal > 0:
                    V[i] -= 2.0 * v_normal * normal
                P[i] = (1.0 - r) * normal
        
        # Check score periodically
        if step % 500 == 0:
            score = _compute_score(P)
            if score > best_score:
                best_score = score
                best_P = P.copy()
        
        # If all circles are well-separated and inside, we can reduce noise
        if best_score > r * 0.999:
            break
    
    return best_P


def construct():
    t_start = time.time()
    
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
    
    # Start with hexagonal initialization
    P_best = _get_hex_initial_points()
    score_best = _compute_score(P_best)
    
    # For each ring structure, optimize and refine
    for counts in ring_structures:
        elapsed = time.time() - t_start
        if elapsed > DEADLINE - 30:
            break
        
        n_rings = len(counts)
        
        # Initial radii: evenly spaced
        if n_rings == 1:
            radii0 = np.array([0.0])
        else:
            radii0 = np.linspace(0, 0.85, n_rings)
            radii0[0] = 0.0
        
        angles0 = [0.0] * n_rings
        
        # Optimize ring structure parameters
        remaining = DEADLINE - (time.time() - t_start) - 35
        if remaining < 5.0:
            continue
        radii_opt, angles_opt = _optimize_ring_structure(
            radii0, counts, angles0,
            time_limit=min(15.0, remaining)
        )
        
        # Build centers from optimized ring structure
        P_candidate = _get_ring_initialization(
            np.concatenate([radii_opt, angles_opt]), counts
        )
        
        score_candidate = _compute_score(P_candidate)
        
        if score_candidate > score_best:
            P_best = P_candidate.copy()
            score_best = score_candidate
        
        # Binary search on r with soft-min optimization
        r_low = score_best * 0.98
        r_high = min(0.135, score_best * 1.05)
        
        # Ensure r_low is feasible
        remaining = DEADLINE - (time.time() - t_start) - 20
        if remaining < 3.0:
            continue
        P_opt, E = _optimize_positions(P_candidate, r_low, time_limit=min(4.0, remaining * 0.3))
        if E < 1e-8:
            score = _compute_score(P_opt)
            if score > score_best:
                P_best = P_opt.copy()
                score_best = score
            r_low = score
        
        # Binary search for better r
        for _ in range(10):
            remaining = DEADLINE - (time.time() - t_start) - 15
            if remaining < 3.0:
                break
            
            r_mid = (r_low + r_high) / 2.0
            P_init = P_best.copy()
            
            P_opt, E = _optimize_positions(P_init, r_mid, time_limit=min(3.5, remaining * 0.25))
            
            if E < 1e-8:
                score = _compute_score(P_opt)
                if score > score_best:
                    P_best = P_opt.copy()
                    score_best = score
                r_low = score
            else:
                r_high = r_mid
        
        # Langevin dynamics annealing to escape local minima
        remaining = DEADLINE - (time.time() - t_start) - 10
        if remaining > 5.0:
            r_target = min(r_high, score_best * 1.02)
            P_langevin = _anneal_langevin(
                P_best, r_target,
                time_limit=min(15.0, remaining),
                n_steps=30000,
                dt=0.0005,
                T0=0.05,
                Tend=0.0005,
                seed=123
            )
            score_langevin = _compute_score(P_langevin)
            if score_langevin > score_best:
                P_best = P_langevin.copy()
                score_best = score_langevin
        
        # Final high-precision polish with L-BFGS-B
        remaining = DEADLINE - (time.time() - t_start) - 5
        if remaining > 5.0:
            r_current = _compute_score(P_best)
            P_polished = _polish_l_bfgs(P_best, r_current * 0.999, time_limit=remaining)
            score_polished = _compute_score(P_polished)
            if score_polished > score_best:
                P_best = P_polished.copy()
                score_best = score_polished
    
    # Ensure validity: clamp points that might be slightly outside
    r_final = _compute_score(P_best)
    if r_final > 0:
        r2 = np.hypot(P_best[:, 0], P_best[:, 1])
        for i in range(N):
            if r2[i] > 1 - r_final + 1e-10:
                scale = (1 - r_final) / r2[i]
                P_best[i] *= scale
    
    return P_best.reshape(N, 2)
