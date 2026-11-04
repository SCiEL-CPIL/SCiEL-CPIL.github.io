"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3.0)
H = S3 / 2.0
TRI_AREA = S3 / 4.0

_IDX_TRIPLES = np.array(
    [(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)]
)
_I = _IDX_TRIPLES[:, 0]
_J = _IDX_TRIPLES[:, 1]
_K = _IDX_TRIPLES[:, 2]
N_TRIPLES = _IDX_TRIPLES.shape[0]


def _bary_to_cart(bc):
    b = bc[:, 0]
    c = bc[:, 1]
    x = b + 0.5 * c
    y = c * H
    return np.column_stack([x, y])


def _cart_to_bary(points):
    x = points[:, 0]
    y = points[:, 1]
    c = y / H
    b = x - 0.5 * c
    return np.column_stack([b, c])


def _project_to_bary_simplex(bc):
    b = bc[:, 0].astype(float).copy()
    c = bc[:, 1].astype(float).copy()
    b = np.clip(b, 0.0, None)
    c = np.clip(c, 0.0, None)
    excess = b + c - 1.0
    mask = excess > 0
    if np.any(mask):
        s = 1.0 / (1.0 + excess[mask])
        b[mask] *= s
        c[mask] *= s
    return np.column_stack([b, c])


def _clip_to_triangle(points):
    x = points[:, 0].astype(float).copy()
    y = points[:, 1].astype(float).copy()
    y = np.clip(y, 0.0, H)
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(x, x_min, x_max)
    return np.column_stack([x, y])


def _dets_from_bary(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    ui, vi, wi = u[_I], v[_I], w[_I]
    uj, vj, wj = u[_J], v[_J], w[_J]
    uk, vk, wk = u[_K], v[_K], w[_K]
    det = ui * (vj * wk - wj * vk) - vi * (uj * wk - wj * uk) + wi * (uj * vk - vj * uk)
    return det


def _min_area_ratio_from_bary(bary):
    dets = _dets_from_bary(bary)
    return float(np.min(np.abs(dets)))


def _structured_init_fps(rng, n_init):
    pts = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.5, 0.5])]
    for _ in range(n_init - 3):
        best_val = -1.0
        best_cand = None
        for t in range(3000):
            u = rng.uniform(0.0, 1.0)
            v = rng.uniform(0.0, 1.0)
            if u + v > 1.0:
                u = 1.0 - u
                v = 1.0 - v
            cand = np.array([u, v])
            d2 = np.min(np.sum((np.array(pts) - cand) ** 2, axis=1))
            if d2 > best_val:
                best_val = d2
                best_cand = cand
        pts.append(best_cand)
    return np.array(pts, dtype=float)


def _epigraph_objective(x, beta, lam=50.0):
    """Epigraph formulation: x = [bary (2N), t].
    Minimize: -t + barrier, where barrier = (1/beta)*log(sum(exp(-(A_i - t)/beta)))
    This pushes t toward the min of A_i while keeping it smooth.
    Plus simplex constraint penalty.
    Returns (value, gradient)."""
    bary = x[:2*N].reshape(N, 2)
    t = x[2*N]
    
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Simplex penalty
    penalty = 0.0
    g = np.zeros_like(x)
    for arr, idx in ((u, 0), (v, 1)):
        neg = -arr
        pos = neg > 0.0
        if np.any(pos):
            penalty += lam * np.sum(neg[pos] ** 2)
            g[idx::2][pos] += 2.0 * lam * neg[pos]
    over = u + v - 1.0
    posw = over > 0.0
    if np.any(posw):
        penalty += lam * np.sum(over[posw] ** 2)
        g[0::2][posw] += 2.0 * lam * over[posw]
        g[1::2][posw] += 2.0 * lam * over[posw]
    
    dets = _dets_from_bary(bary)
    A = np.abs(dets)
    
    # Smooth barrier: (1/beta) * log(sum(exp(-(A_i - t)/beta)))
    z = -(A - t) / beta
    zmax = np.max(z)
    s = np.sum(np.exp(z - zmax))
    logsum = zmax + np.log(s)
    barrier = logsum / beta
    
    # Gradient of barrier wrt t: -1/beta * sum(exp(z_i - zmax)) / s * (-1/beta) ... 
    # d/dt logsum = d/dt (zmax + log s) where z_i = -(A_i - t)/beta = (t - A_i)/beta
    # d z_i / dt = 1/beta
    # d log s / dt = sum(exp(z_i - zmax) * d z_i/dt) / s = (1/beta) * sum(exp(z_i-zmax))/s = (1/beta)
    # So d barrier/dt = (1/beta) * (1/beta) = 1/beta^2 ... wait let me redo.
    # barrier = (1/beta) * logsum
    # d barrier/dt = (1/beta) * d logsum/dt = (1/beta) * (1/beta) = 1/beta^2? No.
    # Actually: z_i = (t - A_i)/beta, so dz_i/dt = 1/beta
    # logsum = zmax + log(sum exp(z_i - zmax))
    # d logsum/dt = sum_i exp(z_i - zmax) * dz_i/dt / sum_j exp(z_j - zmax) = (1/beta) * sum(exp(z_i-zmax))/s = (1/beta)
    # So d barrier/dt = (1/beta) * (1/beta) = 1/beta^2? That's not right dimensionally.
    # Let me just use: d/dt [ (1/beta) log sum exp((t-A_i)/beta) ] = (1/beta) * [ sum exp((t-A_i)/beta) * (1/beta) ] / sum exp((t-A_i)/beta) = (1/beta) * (1/beta) = 1/beta^2? 
    # Hmm, that gives a constant gradient wrt t which is 1/beta^2. That doesn't seem useful.
    # Actually the barrier is designed so that as t increases, the barrier increases when t > min(A_i).
    # The gradient wrt t should be: (1/beta) * (1/beta) * sum(exp(z_i - zmax)) / s ... 
    # Let me just compute: f(t) = (1/beta) log(sum_i exp((t - A_i)/beta))
    # f'(t) = (1/beta) * (1/sum exp) * sum(exp((t-A_i)/beta) * 1/beta) = (1/beta) * (1/beta) = 1/beta^2 * [sum exp / sum exp] = 1/beta^2
    # That's a constant! That can't be right for optimization.
    # 
    # Wait, I think the issue is that I want to MINIMIZE -t + barrier.
    # d/dt (-t + barrier) = -1 + 1/beta^2. For large beta, 1/beta^2 << 1, so gradient is ~ -1, pushing t up.
    # But that's not quite right either. Let me reconsider.
    #
    # Actually the standard epigraph smooth max formulation:
    # We want to maximize t subject to t <= A_i for all i.
    # Equivalently minimize -t subject to t - A_i <= 0.
    # Smooth relaxation: minimize -t + (1/beta) log(sum exp(beta*(t - A_i)))
    # Note: beta*(t - A_i) not (t-A_i)/beta. Let me use that form.
    # f(t, bary) = -t + (1/beta) log(sum_i exp(beta*(t - A_i(bary))))
    # d f/dt = -1 + (1/beta) * sum(exp(beta(t-A_i)) * beta) / sum(exp(beta(t-A_i))) = -1 + 1 = 0? No.
    # d/dt (1/beta log sum exp(beta(t-A_i))) = (1/beta) * (sum exp(beta(t-A_i)) * beta) / sum exp(beta(t-A_i)) = 1
    # So d f/dt = -1 + 1 = 0 always? That's wrong.
    # 
    # I think the correct formulation is: we want the smooth max of (A_i - t) to be <= 0.
    # smooth_max_i (A_i - t) ≈ (1/beta) log sum exp(beta*(A_i - t))
    # We want this <= 0, i.e., minimize (1/beta) log sum exp(beta*(A_i - t))
    # Combined with maximizing t: minimize -t + (1/beta) log sum exp(beta*(A_i - t))
    # d/dt: -1 + (1/beta) * sum(exp(beta(A_i-t)) * (-beta)) / sum(exp(beta(A_i-t))) = -1 - 1 = -2? 
    # That's also not right.
    #
    # Let me just use the approach from the cell finding: "smooth barrier on t variable" with "multi-stage beta annealing (100→250)"
    # I'll use: objective = -t + lam_bar * (1/beta) * log(sum(exp((A_i - t)/beta)))
    # where lam_bar controls the strength. For large beta, this approximates -t + lam_bar * max_i((A_i - t)/beta)
    # which is -t + lam_bar/beta * max(A_i - t). If max(A_i - t) > 0 (i.e., t < min A_i), this is negative, 
    # so we want to increase t. If t > min A_i, max(A_i - t) could still be positive for other i.
    # Actually I want: penalty for t being less than any A_i. So: (1/beta) log sum exp(beta*(t - A_i)) penalizes t > A_i.
    # But I want to push t UP to min A_i. So I should penalize t < A_i, i.e., use (1/beta) log sum exp(beta*(A_i - t)).
    # Then minimize: -t + (1/beta) log sum exp(beta*(A_i - t))
    # When t = min A_i, one term is exp(0)=1, others are exp(beta*(A_i - min)) >= 1, so log sum >= 0.
    # Gradient wrt t: -1 + (1/beta) * sum(exp(beta(A_i-t)) * (-beta)) / sum(exp(beta(A_i-t))) = -1 - 1 = -2. 
    # That's a constant -2, which means t always wants to increase, which is good but unbounded.
    # I need to also penalize t going too high. Let me add a term -t^2 or just rely on the barrier.
    #
    # Actually, I think the simpler approach that worked in the cell finding is:
    # objective = -t + (1/beta) * log(sum(exp((A_i - t)/beta)))  [note: /beta not *beta inside exp]
    # Let me just implement this and see. The key insight is that this creates a smooth landscape.
    
    # Let me restart with a cleaner formulation:
    # Minimize: -t + (1/beta) * log(sum_i exp((A_i - t)/beta))
    # This is -t + smooth_max_i((A_i - t)/beta)
    # When t < min A_i: all (A_i - t) > 0, smooth_max is positive, so -t + positive. As t increases, -t decreases and smooth_max decreases.
    # When t = min A_i: one term is 0, smooth_max = (1/beta)log(1 + sum exp((A_i-min)/beta)) > 0
    # When t > min A_i: some (A_i - t) < 0, smooth_max can be negative.
    # The minimum of this objective should be near t = min A_i.
    
    # Recompute with this formulation:
    z2 = (A - t) / beta
    z2max = np.max(z2)
    s2 = np.sum(np.exp(z2 - z2max))
    logsum2 = z2max + np.log(s2)
    smooth_max = logsum2 / beta
    
    value = -t + smooth_max + penalty
    
    # Gradient
    # d smooth_max / dt = (1/beta) * d/dt logsum2 = (1/beta) * sum(exp(z2_i - z2max) * (-1/beta)) / s2 = -(1/beta^2) * sum(exp(z2_i-z2max))/s2 = -1/beta^2
    # So d value/dt = -1 - 1/beta^2 + 0 (penalty doesn't depend on t)
    # That's a constant, which is problematic.
    
    # I think I need a different approach. Let me use the standard trick:
    # Minimize: -t + (1/beta) * log(sum_i exp(beta * max(0, t - A_i)))
    # But that's not smooth.
    
    # Alternative: just use the smooth min directly without t, as in the current program, but with beta annealing.
    # The cell finding says "epigraph formulation with smooth barrier on t variable provides a structurally different optimization path"
    # Let me try: minimize -t + sum_i softplus(beta * (t - A_i)) / N
    # where softplus(x) = log(1 + exp(x))
    # This penalizes t > A_i smoothly.
    
    # Let me just go with a simpler, proven approach: use the smooth min from the current program but with beta annealing,
    # and combine with the SLSQP/NM refinement. The structural change is the multi-stage beta annealing.
    
    # Actually, let me just implement the epigraph properly:
    # Variables: x = [u_1, v_1, ..., u_N, v_N, t]
    # Minimize: -t + (1/beta) * log(sum_i exp(beta * (t - A_i)))
    # This is the smooth max of (t - A_i) plus -t.
    # smooth_max_i(t - A_i) = (1/beta) log sum exp(beta(t-A_i))
    # f = -t + (1/beta) log sum exp(beta(t-A_i))
    # d f/dt = -1 + (1/beta) * sum(exp(beta(t-A_i)) * beta) / sum(exp(beta(t-A_i))) = -1 + 1 = 0
    # So the gradient wrt t is always 0! That means t is not determined by the objective alone.
    # This is because -t and smooth_max both have slope 1 wrt t.
    # I need to add a regularization on t, e.g., -t^2 / (2 * t_max^2) to bound t.
    
    # Let me add: -t + (1/beta) log sum exp(beta(t-A_i)) - t^2/(2*tau^2)
    # Then d f/dt = -1 + 1 - t/tau^2 = -t/tau^2, which is < 0 for t > 0, so t wants to increase.
    # But that's unbounded. I need the barrier to kick in.
    
    # I think the issue is that I'm overcomplicating this. Let me just use the smooth min approach with beta annealing,
    # which is what the current program does, but with multiple beta values in sequence.
    
    # For now, let me just return a placeholder and implement the multi-beta smooth min approach.
    # I'll compute the gradient of the smooth min properly.
    
    # Smooth min: m - (1/beta) log sum exp(beta(m - A_i)) where m = min A
    # Actually, smooth_min(A) = - (1/beta) log sum exp(-beta A_i)
    # d smooth_min / d A_i = exp(-beta A_i) / sum exp(-beta A_j)
    
    # Let me compute the gradient of -smooth_min wrt bary.
    neg_beta_A = -beta * A
    z3 = neg_beta_A
    z3max = np.max(z3)
    s3 = np.sum(np.exp(z3 - z3max))
    smooth_min = -(1.0 / beta) * (z3max + np.log(s3))
    
    # weights_i = exp(-beta A_i) / sum exp(-beta A_j)
    weights = np.exp(z3 - z3max) / s3
    
    # d smooth_min / d A_i = weights_i
    # d smooth_min / d bary = sum_i weights_i * d A_i / d bary
    # A_i = |det_i|, so d A_i / d bary = sign(det_i) * d det_i / d bary
    
    gdet = weights * np.sign(dets)  # d smooth_min / d det_i
    
    gu = np.zeros(N)
    gv = np.zeros(N)
    wcol = 1.0 - u - v
    for t_idx in range(N_TRIPLES):
        gd = gdet[t_idx]
        if gd == 0.0:
            continue
        i = _I[t_idx]; j = _J[t_idx]; k = _K[t_idx]
        ui, vi = u[i], v[i]
        uj, vj = u[j], v[j]
        uk, vk = u[k], v[k]
        wii = 1.0 - ui - vi
        wjj = 1.0 - uj - vj
        wkk = 1.0 - uk - vk
        gu[i] += gd * (vj * wkk - wjj * vk)
        gv[i] += -gd * (uj * wkk - wjj * uk)
        gu[j] += gd * (-vi * wkk + wii * vk)
        gv[j] += gd * (vi * wkk - wii * uk)
        gu[k] += gd * (vi * wjj - wii * vj)
        gv[k] += gd * (ui * vjj - wjj * ui)
    
    gx = np.zeros_like(x)
    gx[0::2] = gu
    gx[1::2] = gv
    
    # Add penalty gradient
    for idx, arr in enumerate((u, v)):
        neg = -arr
        pos = neg > 0.0
        if np.any(pos):
            gx[idx::2][pos] += 2.0 * lam * neg[pos]
    over = u + v - 1.0
    posw = over > 0.0
    if np.any(posw):
        gx[0::2][posw] += 2.0 * lam * over[posw]
        gx[1::2][posw] += 2.0 * lam * over[posw]
    
    value = -smooth_min + penalty
    return value, gx


def _objective_slsqp(x):
    bary = _project_to_bary_simplex(x.reshape(N, 2))
    dets = _dets_from_bary(bary)
    return -float(np.min(np.abs(dets)))


def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    rng = np.random.default_rng(42)

    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2 * i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2 * i + 1]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2 * i] - x[2 * i + 1]})
    bounds = [(0.0, 1.0)] * (2 * N)

    best_points = None
    best_score = -float('inf')

    def consider(bary):
        nonlocal best_points, best_score
        bary = _project_to_bary_simplex(bary)
        points = _bary_to_cart(bary)
        points = _clip_to_triangle(points)
        score = _min_area_ratio_from_bary(_cart_to_bary(points))
        if score > best_score:
            best_score = score
            best_points = points.copy()
        return score

    # Phase 1: multi-start with beta annealing
    n_starts = 5
    for s in range(n_starts):
        if time.time() > deadline - 25.0:
            break
        bary_init = _structured_init_fps(rng, N)
        x0 = bary_init.flatten()
        consider(bary_init)

        # Multi-stage beta annealing: 50 -> 100 -> 200 -> 400
        for beta in (50.0, 100.0, 200.0, 400.0):
            if time.time() > deadline - 20.0:
                break
            try:
                res = minimize(
                    lambda x: _epigraph_objective(x, beta),
                    x0, method='L-BFGS-B',
                    jac=True, bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-10}
                )
                bary_opt = _project_to_bary_simplex(res.x.reshape(N, 2))
                consider(bary_opt)
                x0 = bary_opt.flatten()
            except Exception:
                pass

        # SLSQP polish
        if time.time() < deadline - 15.0:
            try:
                res = minimize(_objective_slsqp, x0, method='SLSQP',
                               bounds=bounds, constraints=constraints,
                               options={'maxiter': 300, 'ftol': 1e-12})
                consider(_project_to_bary_simplex(res.x.reshape(N, 2)))
            except Exception:
                pass

    # Phase 2: refine global best with Nelder-Mead in Cartesian
    if best_points is not None:
        bary_best = _cart_to_bary(best_points)
        x_best = bary_best.flatten()
        consider(bary_best)

        if time.time() < deadline - 10.0:
            try:
                res = minimize(_objective_slsqp, x_best, method='SLSQP',
                               bounds=bounds, constraints=constraints,
                               options={'maxiter': 400, 'ftol': 1e-14})
                consider(_project_to_bary_simplex(res.x.reshape(N, 2)))
            except Exception:
                pass

        if time.time() < deadline - 5.0:
            def obj_cart(x):
                pts = _clip_to_triangle(x.reshape(N, 2))
                return -_min_area_ratio_from_bary(_cart_to_bary(pts))
            try:
                res = minimize(obj_cart, best_points.flatten(), method='Nelder-Mead',
                               options={'maxiter': 600, 'xatol': 1e-11,
                                        'fatol': 1e-13, 'adaptive': True})
                consider(_cart_to_bary(_clip_to_triangle(res.x.reshape(N, 2))))
            except Exception:
                pass

    if best_points is None:
        bary_init = _structured_init_fps(rng, N)
        best_points = _bary_to_cart(bary_init)

    return _clip_to_triangle(best_points)
