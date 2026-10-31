"""Circle packing in unit square - 26 circles maximizing sum of radii.
Uses compact structural parameterization with multi-phase optimization.
Implements spectral graph formulation with distance matrix eigenvalue
proxy for global minimum pairwise separation, combined with DE global
search and high-precision local polish.
"""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(radii)
    for i in range(n):
        if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
            return False
        if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True


def _spectral_proxy(pos, radii, beta=40.0):
    """Smooth spectral proxy for minimum pairwise separation.
    Uses the log-sum-exp of negative squared distances as a smooth
    differentiable surrogate that is sensitive to the closest pair.
    """
    n = len(radii)
    d = pos[:, 0][:, None] - pos[:, 0][None, :]
    e = pos[:, 1][:, None] - pos[:, 1][None, :]
    dist_sq = d * d + e * e
    np.fill_diagonal(dist_sq, np.inf)
    
    # Negative distances (we want to maximize the minimum)
    neg_dist = -np.sqrt(dist_sq)
    # Replace inf with a large negative number
    neg_dist = np.where(np.isinf(neg_dist), -100.0, neg_dist)
    
    # Log-sum-exp of beta * neg_dist (smooth max of -dist, i.e., smooth min of dist)
    m = np.max(neg_dist)
    shifted = beta * (neg_dist - m)
    # Mask diagonal
    shifted[np.diag_indices(n)] = -1000.0
    
    lse = (np.log(np.sum(np.exp(shifted))) + beta * m) / beta
    return lse  # This is close to -min_dist, so negative


def _objective(x, deadline):
    """Objective: minimize negative sum of radii + smooth penalties.
    Uses a combined approach:
    - Quadratic penalty on positive violations
    - Spectral proxy for minimum pairwise separation
    """
    if time.time() > deadline:
        return 1e10
    n = N
    pos = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]

    # Boundary violations (vectorized)
    left = radii - pos[:, 0]
    right = pos[:, 0] - (1.0 - radii)
    bot = radii - pos[:, 1]
    top = pos[:, 1] - (1.0 - radii)

    # Overlap violations (vectorized via distance matrix)
    d = pos[:, 0][:, None] - pos[:, 0][None, :]
    e = pos[:, 1][:, None] - pos[:, 1][None, :]
    dist = np.sqrt(d * d + e * e)
    min_dist = radii[:, None] + radii[None, :]
    viol = min_dist - dist
    np.fill_diagonal(viol, 0.0)

    # Stack all violation magnitudes
    all_v = np.concatenate([left, right, bot, top, viol.ravel()])

    # Quadratic penalty on positive violations
    pos_viol = np.maximum(all_v, 0.0)
    quad_penalty = np.sum(pos_viol ** 2)

    # Spectral proxy: smooth measure of worst overlap
    # We want to penalize when min_dist < dist, i.e., viol > 0
    # Use log-sum-exp on the positive violations with high beta
    beta = 50.0
    m = np.max(pos_viol)
    if m > 0:
        shifted = beta * pos_viol
        lse = (np.log(np.sum(np.exp(shifted - beta * m))) + beta * m) / beta
    else:
        lse = 0.0

    # Combined penalty
    penalty = 40.0 * quad_penalty + 15.0 * lse

    return -np.sum(radii) + penalty


# ---------------- Compact structural parameterization ----------------
# p = [corner_r, center_r, cx, cy,
#      ring1_off, ring1_r, ring1_r2,
#      ring2_off, ring2_r, ring2_r2]
# ring1: 8 circles, ring2: 13 circles, plus 4 corners + 1 center = 26.

def _expand_struct(p):
    """Map a 10-dim structural vector to (pos, radii) of 26 circles."""
    corner_r = np.clip(p[0], 0.05, 0.30)
    center_r = np.clip(p[1], 0.05, 0.35)
    cx = np.clip(p[2], 0.2, 0.8)
    cy = np.clip(p[3], 0.2, 0.8)
    r1_off = np.clip(p[4], 0.15, 0.6)
    r1_r = np.clip(p[5], 0.03, 0.20)
    r1_r2 = np.clip(p[6], 0.03, 0.20)
    r2_off = np.clip(p[7], 0.25, 0.7)
    r2_r = np.clip(p[8], 0.02, 0.16)
    r2_r2 = np.clip(p[9], 0.02, 0.16)

    pos = np.zeros((N, 2))
    radii = np.zeros(N)

    # 4 corners
    pos[0] = [corner_r, corner_r]
    pos[1] = [1.0 - corner_r, corner_r]
    pos[2] = [corner_r, 1.0 - corner_r]
    pos[3] = [1.0 - corner_r, 1.0 - corner_r]
    radii[0:4] = corner_r

    # center
    pos[4] = [cx, cy]
    radii[4] = center_r

    # ring 1: 8 circles (indices 5..12)
    for i in range(8):
        ang = 2 * np.pi * i / 8.0
        pos[5 + i] = [cx + r1_off * np.cos(ang), cy + r1_off * np.sin(ang)]
        radii[5 + i] = r1_r if i % 2 == 0 else r1_r2

    # ring 2: 13 circles (indices 13..25)
    for i in range(13):
        ang = 2 * np.pi * i / 13.0 + 0.1
        pos[13 + i] = [cx + r2_off * np.cos(ang), cy + r2_off * np.sin(ang)]
        radii[13 + i] = r2_r if i % 2 == 0 else r2_r2

    return pos, radii


def _struct_objective(p, deadline):
    pos, radii = _expand_struct(p)
    x = np.concatenate([pos.flatten(), radii])
    return _objective(x, deadline)


# ---------------- Post-processing & polish ----------------

def _post_process(pos, radii, deadline):
    """Iteratively shrink circles to ensure validity, conservatively."""
    pos = pos.copy()
    radii = radii.copy()

    for _ in range(500):
        if time.time() > deadline:
            break
        valid = True
        for i in range(N):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r:
                radii[i] = max_r * 0.999
                valid = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    # Shrink the smaller circle more to preserve larger ones
                    if radii[i] <= radii[j]:
                        needed = (min_dist - dist) / (2.0 * radii[i] + 1e-12)
                        radii[i] *= max(0.5, 1.0 - needed * 0.999)
                    else:
                        needed = (min_dist - dist) / (2.0 * radii[j] + 1e-12)
                        radii[j] *= max(0.5, 1.0 - needed * 0.999)
                    valid = False
        if valid:
            break

    for i in range(N):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        radii[i] = max(1e-6, min(radii[i], max_r))

    return pos, radii


def _gradient_polish(pos, radii, deadline, max_iter=5000, step=0.001):
    """Gradient-based polish with adaptive step size.
    Uses numerical gradients of the smooth objective.
    """
    pos = pos.copy()
    radii = radii.copy()
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-4, 0.35))
    
    best_x = x0.copy()
    best_f = _objective(x0, deadline)
    
    # Adaptive step size
    h = step
    for it in range(max_iter):
        if time.time() > deadline - 2:
            break
        
        # Compute numerical gradient
        grad = np.zeros_like(x0)
        for i in range(len(x0)):
            xp = x0.copy()
            xp[i] += h
            xm = x0.copy()
            xm[i] -= h
            # Clamp to bounds
            lo, hi = bounds[i]
            xp[i] = max(lo, min(hi, xp[i]))
            xm[i] = max(lo, min(hi, xm[i]))
            grad[i] = (_objective(xp, deadline) - _objective(xm, deadline)) / (xp[i] - xm[i] + 1e-15)
        
        # Gradient step
        x_new = x0 - h * grad
        # Clamp to bounds
        for i in range(len(x0)):
            lo, hi = bounds[i]
            x_new[i] = max(lo, min(hi, x_new[i]))
        
        f_new = _objective(x_new, deadline)
        if f_new < best_f:
            best_x = x_new.copy()
            best_f = f_new
            # Reduce step slightly on success
            h *= 0.95
        else:
            # Increase step or try different direction
            h *= 1.1
            if h > 0.01:
                h = 0.01
            # Try a small random perturbation
            if it % 100 == 0:
                x_new = x0 + np.random.normal(0, 0.005, len(x0))
                for i in range(len(x0)):
                    lo, hi = bounds[i]
                    x_new[i] = max(lo, min(hi, x_new[i]))
                f_new = _objective(x_new, deadline)
                if f_new < best_f:
                    best_x = x_new.copy()
                    best_f = f_new
        
        x0 = best_x.copy()
    
    pos_opt = best_x[:2 * N].reshape(N, 2)
    radii_opt = best_x[2 * N:]
    pos_pp, radii_pp = _post_process(pos_opt, radii_opt, deadline)
    return pos_pp, radii_pp


def _polish(pos, radii, deadline):
    """High-precision L-BFGS-B polish with tight bounds and tolerances."""
    pos = pos.copy()
    radii = radii.copy()
    x0 = np.concatenate([pos.flatten(), radii])
    
    # Base bounds
    base_bounds = []
    for i in range(N):
        base_bounds.append((max(0.0, radii[i] * 0.5), min(1.0, 1.0 - radii[i] * 0.5)))
        base_bounds.append((max(0.0, radii[i] * 0.5), min(1.0, 1.0 - radii[i] * 0.5)))
    for i in range(N):
        base_bounds.append((1e-4, 0.35))
    
    best_result = None
    best_score = -np.inf
    
    # Try multiple perturbations to escape local minima
    rng = np.random.RandomState(42)
    num_tries = 3
    for try_idx in range(num_tries):
        if time.time() > deadline - 5:
            break
        
        if try_idx == 0:
            x_try = x0.copy()
        else:
            perturbation = rng.normal(0, 0.01, size=len(x0))
            x_try = x0 + perturbation
            x_try[2*N:] = np.maximum(x_try[2*N:], 1e-4)
            for i in range(N):
                x_try[2*i] = np.clip(x_try[2*i], 0.0, 1.0)
                x_try[2*i+1] = np.clip(x_try[2*i+1], 0.0, 1.0)
        
        try:
            result = minimize(lambda x: _objective(x, deadline), x_try, method='L-BFGS-B',
                              bounds=base_bounds, options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14})
            pos_opt = result.x[:2 * N].reshape(N, 2)
            radii_opt = result.x[2 * N:]
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt, deadline)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_result = (pos_pp, radii_pp)
        except Exception:
            pass
    
    if best_result is not None:
        return best_result
    return pos, radii


def _slsqp_polish(pos, radii, deadline):
    """Constraint-based SLSQP polish with vectorized constraint functions."""
    pos = pos.copy()
    radii = radii.copy()
    x0 = np.concatenate([pos.flatten(), radii])

    # Bounds: positions in [0,1], radii in [1e-4, 0.35]
    bounds = [(0.0, 1.0)] * (2 * N) + [(1e-4, 0.35)] * N

    # Vectorized constraint functions
    def boundary_cons(x):
        """All boundary constraints: x_i >= r_i, 1 - x_i >= r_i"""
        p = x[:2*N].reshape(N, 2)
        r = x[2*N:]
        left = p[:, 0] - r
        right = 1.0 - p[:, 0] - r
        bot = p[:, 1] - r
        top = 1.0 - p[:, 1] - r
        return np.concatenate([left, right, bot, top])
    
    def overlap_cons(x):
        """All overlap constraints: dist(i,j) >= r_i + r_j for all i < j"""
        p = x[:2*N].reshape(N, 2)
        r = x[2*N:]
        d = p[:, 0][:, None] - p[:, 0][None, :]
        e = p[:, 1][:, None] - p[:, 1][None, :]
        dist = np.sqrt(d * d + e * e)
        min_dist = r[:, None] + r[None, :]
        viol = min_dist - dist
        # Only take upper triangle
        idx = np.triu_indices(N, k=1)
        return (dist[idx] - min_dist[idx]).ravel()
    
    cons = [
        {'type': 'ineq', 'fun': boundary_cons},
        {'type': 'ineq', 'fun': overlap_cons}
    ]

    def neg_sum_rad(x):
        if time.time() > deadline:
            return 1e10
        return -np.sum(x[2 * N:])

    best_result = None
    best_score = -np.inf
    rng = np.random.RandomState(123)
    
    num_tries = 2
    for try_idx in range(num_tries):
        if time.time() > deadline - 3:
            break
        
        if try_idx == 0:
            x_try = x0.copy()
        else:
            perturbation = rng.normal(0, 0.005, size=len(x0))
            x_try = x0 + perturbation
            x_try[2*N:] = np.maximum(x_try[2*N:], 1e-4)
            for i in range(N):
                x_try[2*i] = np.clip(x_try[2*i], 0.0, 1.0)
                x_try[2*i+1] = np.clip(x_try[2*i+1], 0.0, 1.0)
        
        try:
            result = minimize(neg_sum_rad, x_try, method='SLSQP', bounds=bounds,
                              constraints=cons,
                              options={'maxiter': 300, 'ftol': 1e-10})
            if result.x is not None:
                pos_opt = result.x[:2 * N].reshape(N, 2)
                radii_opt = result.x[2 * N:]
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt, deadline)
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score:
                        best_score = score
                        best_result = (pos_pp, radii_pp)
        except Exception:
            pass
    
    if best_result is not None:
        return best_result
    return pos, radii


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s

    best_pos = None
    best_radii = None
    best_score = -np.inf

    def update_best(pos, radii):
        nonlocal best_pos, best_radii, best_score
        if time.time() > deadline:
            return
        pos_pp, radii_pp = _post_process(pos, radii, deadline)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()

    struct_bounds = [
        (0.06, 0.28),   # corner_r
        (0.06, 0.32),   # center_r
        (0.3, 0.7),     # cx
        (0.3, 0.7),     # cy
        (0.18, 0.45),   # ring1 offset
        (0.04, 0.16),   # ring1 r
        (0.04, 0.16),   # ring1 r2
        (0.30, 0.55),   # ring2 offset
        (0.03, 0.12),   # ring2 r
        (0.03, 0.12),   # ring2 r2
    ]

    # ---------- Track A: compact structural parameterization ----------
    # DE over 10-dim structural space, then expand + polish.

    # First DE run
    if time.time() < deadline - 80:
        try:
            de_res = differential_evolution(_struct_objective, struct_bounds,
                                            args=(deadline,),
                                            maxiter=160, popsize=28, seed=7,
                                            tol=1e-10, polish=False,
                                            mutation=(0.35, 1.0), recombination=0.75,
                                            updating='deferred')
            if de_res.x is not None:
                pos_s, radii_s = _expand_struct(de_res.x)
                update_best(pos_s, radii_s)
                if time.time() < deadline - 45:
                    pos_p, radii_p = _polish(pos_s, radii_s, deadline)
                    if _is_valid(pos_p, radii_p):
                        s = np.sum(radii_p)
                        if s > best_score:
                            best_score = s
                            best_pos = pos_p.copy()
                            best_radii = radii_p.copy()
        except Exception:
            pass

    # Second DE run with different parameters
    if time.time() < deadline - 60:
        try:
            de_res2 = differential_evolution(_struct_objective, struct_bounds,
                                             args=(deadline,),
                                             maxiter=100, popsize=22, seed=123,
                                             tol=1e-9, polish=False,
                                             mutation=(0.5, 1.1), recombination=0.85,
                                             updating='immediate')
            if de_res2.x is not None:
                pos_s2, radii_s2 = _expand_struct(de_res2.x)
                update_best(pos_s2, radii_s2)
                if time.time() < deadline - 35:
                    pos_p2, radii_p2 = _polish(pos_s2, radii_s2, deadline)
                    if _is_valid(pos_p2, radii_p2):
                        s2 = np.sum(radii_p2)
                        if s2 > best_score:
                            best_score = s2
                            best_pos = pos_p2.copy()
                            best_radii = radii_p2.copy()
        except Exception:
            pass

    # L-BFGS-B on structural vector from seeds
    struct_seeds = [
        [0.13, 0.16, 0.5, 0.5, 0.30, 0.11, 0.11, 0.42, 0.07, 0.07],
        [0.12, 0.18, 0.5, 0.5, 0.33, 0.10, 0.10, 0.45, 0.06, 0.06],
        [0.15, 0.14, 0.5, 0.5, 0.28, 0.12, 0.12, 0.40, 0.08, 0.08],
        [0.14, 0.15, 0.5, 0.5, 0.31, 0.105, 0.105, 0.43, 0.065, 0.065],
    ]
    for s0 in struct_seeds:
        if time.time() > deadline - 50:
            break
        try:
            res = minimize(_struct_objective, np.array(s0), args=(deadline,),
                           method='L-BFGS-B',
                           bounds=struct_bounds,
                           options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-13})
            pos_s, radii_s = _expand_struct(res.x)
            update_best(pos_s, radii_s)
        except Exception:
            pass

    # ---------- Track B: direct 78-dim from structural seed ----------
    if time.time() < deadline - 35:
        pos_s0, radii_s0 = _expand_struct(np.array(struct_seeds[0]))
        x0 = np.concatenate([pos_s0.flatten(), radii_s0])
        bounds = []
        for i in range(N):
            bounds.append((0.01, 0.99))
            bounds.append((0.01, 0.99))
        for i in range(N):
            bounds.append((0.01, 0.32))
        try:
            result = minimize(lambda x: _objective(x, deadline), x0, method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12})
            if result.success or result.fun < _objective(x0, deadline):
                pos_opt = result.x[:2 * N].reshape(N, 2)
                radii_opt = result.x[2 * N:]
                update_best(pos_opt, radii_opt)
        except Exception:
            pass

    # ---------- Track C: Gradient polish on best configuration ----------
    if best_pos is not None and time.time() < deadline - 20:
        try:
            pos_gp, radii_gp = _gradient_polish(best_pos, best_radii, deadline,
                                                 max_iter=3000, step=0.001)
            if _is_valid(pos_gp, radii_gp):
                s = np.sum(radii_gp)
                if s > best_score:
                    best_score = s
                    best_pos = pos_gp.copy()
                    best_radii = radii_gp.copy()
        except Exception:
            pass

    # ---------- Final high-precision polish of overall best ----------
    if best_pos is not None and time.time() < deadline - 10:
        pos_p, radii_p = _polish(best_pos, best_radii, deadline)
        if _is_valid(pos_p, radii_p):
            s = np.sum(radii_p)
            if s > best_score:
                best_score = s
                best_pos = pos_p.copy()
                best_radii = radii_p.copy()

    # ---------- Final SLSQP cross-check polish ----------
    if best_pos is not None and time.time() < deadline - 4:
        pos_q, radii_q = _slsqp_polish(best_pos, best_radii, deadline)
        if _is_valid(pos_q, radii_q):
            s = np.sum(radii_q)
            if s > best_score:
                best_score = s
                best_pos = pos_q.copy()
                best_radii = radii_q.copy()

    # Fallback
    if best_pos is None:
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        side = 5
        for i in range(side):
            for j in range(side):
                idx = i * side + j
                if idx < N:
                    pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                    radii[idx] = 0.5 / side - 0.005
        best_pos, best_radii = _post_process(pos, radii, deadline)

    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
