"""Structural change: treat the packing as a continuous optimal-control problem
with a low-dimensional control parameterisation of the circle-centre configuration.

State: 200 circle centres.  Control: a small set of global "shape modes" (Fourier-like
boundary deformations) that modulate a base hexagonal lattice.  The adjoint method
gives exact gradients of a smooth surrogate for the minimum clearance with respect
to these low-dimensional controls, enabling precise gradient-based refinement.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize, L_BFGS_B
from scipy.spatial import cKDTree

N = 200

_T0 = time.time()
_DEADLINE = _T0 + 140.0


def _time_left():
    return _DEADLINE - time.time()


def _compute_radius(pos):
    """Exact packing radius in double precision."""
    n = len(pos)
    if n == 0:
        return 0.0
    dist_boundary = np.min(0.5 - np.abs(pos), axis=1)
    min_boundary = float(np.min(dist_boundary))
    if min_boundary <= 0.0:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair = math.sqrt(float(np.min(dist_sq)))
    return min(min_pair * 0.5, min_boundary)


def _base_hex_positions(n, spacing):
    """Generate a hexagonal lattice of n points centred in the unit square."""
    dy = spacing * math.sqrt(3.0) / 2.0
    rows = []
    y = -0.5
    row = 0
    while y <= 0.5:
        x_off = 0.0 if row % 2 == 0 else spacing / 2.0
        x = -0.5 + x_off
        while x <= 0.5:
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                rows.append((x, y))
            x += spacing
        y += dy
        row += 1
    pts = np.array(rows, dtype=float)
    if len(pts) < n:
        # Pad with jittered points
        extra = np.random.uniform(-0.49, 0.49, (n - len(pts), 2))
        pts = np.vstack([pts, extra])
    else:
        pts = pts[:n]
    return pts


def _shape_modes_apply(base, params):
    """
    Apply low-dimensional shape-mode deformations to base positions.
    params: array of shape (n_params,)
    Modes:
      0: global scale (x)
      1: global scale (y)
      2: shear xy
      3: shear yx
      4: quadratic x^2
      5: quadratic y^2
      6: quadratic xy
      7: sine x (period 1)
      8: sine y (period 1)
      9: cosine x (period 1)
      10: cosine y (period 1)
    """
    pos = base.copy()
    if len(params) == 0:
        return pos
    x = pos[:, 0]
    y = pos[:, 1]
    # Global scales
    if len(params) > 0:
        pos[:, 0] *= (1.0 + params[0])
    if len(params) > 1:
        pos[:, 1] *= (1.0 + params[1])
    # Shears
    if len(params) > 2:
        pos[:, 1] += params[2] * x
    if len(params) > 3:
        pos[:, 0] += params[3] * y
    # Quadratics
    if len(params) > 4:
        pos[:, 0] += params[4] * (x * x - 1.0 / 12.0)
    if len(params) > 5:
        pos[:, 1] += params[5] * (y * y - 1.0 / 12.0)
    if len(params) > 6:
        pos[:, 0] += params[6] * (x * y)
    # Sinusoidal modes
    if len(params) > 7:
        pos[:, 0] += params[7] * np.sin(2.0 * np.pi * x)
    if len(params) > 8:
        pos[:, 1] += params[8] * np.sin(2.0 * np.pi * y)
    if len(params) > 9:
        pos[:, 0] += params[9] * np.cos(2.0 * np.pi * x)
    if len(params) > 10:
        pos[:, 1] += params[10] * np.cos(2.0 * np.pi * y)
    # Clip to interior
    pos = np.clip(pos, -0.49999, 0.49999)
    return pos


def _soft_min_objective(params, base, beta):
    """
    Smooth surrogate: -log(sum(exp(-c_i / beta))) where c_i are the clearance
    constraints (pair distances / 2 and boundary distances). Minimising this
    maximises the minimum clearance.
    """
    pos = _shape_modes_apply(base, params)
    n = len(pos)
    # Boundary clearances
    bx = 0.5 - np.abs(pos[:, 0])
    by = 0.5 - np.abs(pos[:, 1])
    # Pairwise clearances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist_full = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    pair_clear = dist_full[iu] / 2.0
    # Combine all clearances
    all_clear = np.concatenate([bx, by, pair_clear])
    # Soft min: -beta * log(mean(exp(-c / beta)))
    # For numerical stability, use log-sum-exp
    m = np.min(all_clear)
    log_sum = m + beta * np.log(np.mean(np.exp(-(all_clear - m) / beta)))
    obj = -log_sum
    return obj


def _soft_min_gradient(params, base, beta):
    """
    Adjoint-style gradient: differentiate the soft-min objective with respect
    to the control parameters using the chain rule through the shape modes.
    """
    pos = _shape_modes_apply(base, params)
    n = len(pos)
    bx = 0.5 - np.abs(pos[:, 0])
    by = 0.5 - np.abs(pos[:, 1])
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist_full = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    pair_clear = dist_full[iu] / 2.0
    all_clear = np.concatenate([bx, by, pair_clear])
    m = np.min(all_clear)
    exp_terms = np.exp(-(all_clear - m) / beta)
    weights = exp_terms / np.sum(exp_terms)
    # Gradient of soft-min w.r.t. pos
    grad_pos = np.zeros_like(pos)
    # Boundary contributions
    w_bx = weights[:n]
    w_by = weights[n:2 * n]
    grad_pos[:, 0] += w_bx * np.sign(pos[:, 0]) / beta
    grad_pos[:, 1] += w_by * np.sign(pos[:, 1]) / beta
    # Pair contributions
    w_pair = weights[2 * n:]
    # Build full weight matrix
    wmat = np.zeros((n, n))
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        wmat[i, j] = w_pair[idx]
        wmat[j, i] = w_pair[idx]
    # grad_pos[i] += sum_j wmat[i,j] * (pos[i] - pos[j]) / (dist_ij * beta)
    for i in range(n):
        for j in range(n):
            if i != j and wmat[i, j] > 0:
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                if dist > 1e-12:
                    grad_pos[i] += wmat[i, j] * d / (dist * beta)
    # Chain rule through shape modes
    grad_params = np.zeros(len(params))
    x = base[:, 0]
    y = base[:, 1]
    if len(params) > 0:
        grad_params[0] = np.sum(grad_pos[:, 0] * pos[:, 0])
    if len(params) > 1:
        grad_params[1] = np.sum(grad_pos[:, 1] * pos[:, 1])
    if len(params) > 2:
        grad_params[2] = np.sum(grad_pos[:, 1] * x)
    if len(params) > 3:
        grad_params[3] = np.sum(grad_pos[:, 0] * y)
    if len(params) > 4:
        grad_params[4] = np.sum(grad_pos[:, 0] * (x * x - 1.0 / 12.0))
    if len(params) > 5:
        grad_params[5] = np.sum(grad_pos[:, 1] * (y * y - 1.0 / 12.0))
    if len(params) > 6:
        grad_params[6] = np.sum(grad_pos[:, 0] * (x * y))
    if len(params) > 7:
        grad_params[7] = np.sum(grad_pos[:, 0] * np.sin(2.0 * np.pi * x))
    if len(params) > 8:
        grad_params[8] = np.sum(grad_pos[:, 1] * np.sin(2.0 * np.pi * y))
    if len(params) > 9:
        grad_params[9] = np.sum(grad_pos[:, 0] * np.cos(2.0 * np.pi * x))
    if len(params) > 10:
        grad_params[10] = np.sum(grad_pos[:, 1] * np.cos(2.0 * np.pi * y))
    return -grad_params  # We're minimising, so negate


def _local_refine_positions(pos, max_iter=200):
    """
    Direct local refinement: iteratively adjust the closest pair and boundary
    violators to increase the minimum clearance.
    """
    pos = np.array(pos, dtype=float, copy=True)
    n = len(pos)
    for it in range(max_iter):
        if _time_left() < 8.0:
            break
        r = _compute_radius(pos)
        if r <= 0:
            break
        # Find closest pair
        tree = cKDTree(pos)
        pairs = tree.query_pairs(r=r * 2.2)
        if len(pairs) == 0:
            break
        # Find the closest pair
        min_dist = np.inf
        min_pair = None
        for i, j in pairs:
            d = np.linalg.norm(pos[i] - pos[j])
            if d < min_dist:
                min_dist = d
                min_pair = (i, j)
        # Find closest boundary point
        dist_b = np.min(0.5 - np.abs(pos), axis=1)
        min_b_idx = int(np.argmin(dist_b))
        min_b_val = dist_b[min_b_idx]
        # Decide which constraint is active
        if min_dist / 2.0 <= min_b_val:
            i, j = min_pair
            d = pos[i] - pos[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                d = np.random.randn(2) * 1e-8
                dist = np.linalg.norm(d)
            d_norm = d / dist
            step = 0.002 * max(0, dist - 2 * r)
            if step > 1e-12:
                pos[i] += d_norm * step / 2
                pos[j] -= d_norm * step / 2
        else:
            p = pos[min_b_idx]
            if np.linalg.norm(p) > 1e-12:
                step = 0.002 * max(0, min_b_val - r)
                if step > 1e-12:
                    pos[min_b_idx] = p - p / np.linalg.norm(p) * step
        pos = np.clip(pos, -0.49999, 0.49999)
    return pos


def _optimize_with_controls(base, n_params=11, beta_start=100.0, beta_end=500.0):
    """
    Optimize the low-dimensional control parameters using L-BFGS-B,
    with the soft-min objective and adjoint gradients.
    """
    x0 = np.zeros(n_params)

    def obj_and_grad(p):
        obj = _soft_min_objective(p, base, beta_start)
        grad = _soft_min_gradient(p, base, beta_start)
        return obj, grad

    res = minimize(
        lambda p: _soft_min_objective(p, base, beta_start),
        x0,
        jac=lambda p: _soft_min_gradient(p, base, beta_start),
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8},
    )
    best_params = res.x if res.x is not None else x0
    best_pos = _shape_modes_apply(base, best_params)
    best_r = _compute_radius(best_pos)
    return best_pos, best_r


def construct():
    best_pos = None
    best_r = 0.0

    # Generate base hexagonal lattice
    spacing = 0.036
    base = _base_hex_positions(N, spacing)
    base = np.clip(base, -0.49999, 0.49999)

    # Initial radius
    r0 = _compute_radius(base)
    if r0 > best_r:
        best_r = r0
        best_pos = base.copy()

    # Optimize with shape-mode controls
    if _time_left() > 30.0:
        pos_opt, r_opt = _optimize_with_controls(base, n_params=11)
        if r_opt > best_r:
            best_r = r_opt
            best_pos = pos_opt

    # Direct local refinement
    if _time_left() > 20.0:
        pos_ref = _local_refine_positions(best_pos, max_iter=300)
        r_ref = _compute_radius(pos_ref)
        if r_ref > best_r:
            best_r = r_ref
            best_pos = pos_ref

    # Second round: perturb and re-optimize
    if _time_left() > 25.0:
        rng = np.random.RandomState(123)
        perturbed = best_pos + rng.randn(N, 2) * 0.003
        perturbed = np.clip(perturbed, -0.49999, 0.49999)
        # Re-derive a base from the perturbed positions
        base2 = perturbed.copy()
        pos_opt2, r_opt2 = _optimize_with_controls(base2, n_params=11)
        if r_opt2 > best_r:
            best_r = r_opt2
            best_pos = pos_opt2
        pos_ref2 = _local_refine_positions(best_pos, max_iter=200)
        r_ref2 = _compute_radius(pos_ref2)
        if r_ref2 > best_r:
            best_r = r_ref2
            best_pos = pos_ref2

    # Final polish
    if _time_left() > 10.0:
        pos_pol = _local_refine_positions(best_pos, max_iter=200)
        r_pol = _compute_radius(pos_pol)
        if r_pol > best_r:
            best_r = r_pol
            best_pos = pos_pol

    # Safety
    best_pos = np.asarray(best_pos, dtype=float)
    if best_pos.shape != (N, 2):
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_pos = np.array([(x, y) for y in ticks for x in ticks])[:N]
    best_pos = np.clip(best_pos, -0.49999, 0.49999)
    if not np.all(np.isfinite(best_pos)):
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_pos = np.array([(x, y) for y in ticks for x in ticks])[:N]

    return best_pos
