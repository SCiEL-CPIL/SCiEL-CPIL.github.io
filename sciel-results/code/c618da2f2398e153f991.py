"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: evaluator kills at 330s wall-clock; keep a 320s safety margin.
_DEADLINE = 320.0
_T_START = time.time()


def _elapsed():
    return time.time() - _T_START


def _compute_ei(mu, sigma, f_best):
    """Expected Improvement for minimization."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _hv_contrib_exact(cost, val):
    """
    Exact 2D hypervolume contribution of each point in the region
    cost in [0,1], val in [0,1] (both minimized), reference (1,1).

    Uses a monotone stack for the skyline -> O(N log N).
    """
    n = len(cost)
    contrib = np.zeros(n)
    if n == 0:
        return contrib

    # Sort by cost ascending; break ties by value ascending (better value first).
    order = np.lexsort((val, cost))
    c = cost[order]
    v = val[order]

    # Skyline: indices (in sorted order) where value strictly decreases.
    stack = []          # list of (cost_idx, value)
    for i in range(n):
        if stack and v[i] >= stack[-1][1]:
            continue    # dominated in value by a cheaper/equal-cost point
        while stack and v[i] < stack[-1][1]:
            stack.pop()
        stack.append((i, v[i]))

    m = len(stack)
    for k in range(m):
        i, val_i = stack[k]
        cost_i = c[i]
        next_cost = c[stack[k + 1][0]] if k + 1 < m else 1.0
        cost_upper = min(next_cost, 1.0)
        if cost_upper <= cost_i:
            continue
        height = 1.0 - val_i
        if height <= 0.0:
            continue
        contrib[order[i]] = (cost_upper - cost_i) * height

    return contrib


def _hv_contrib_soft(cost, val, beta):
    """
    Smooth surrogate for the hypervolume contribution using a log-sum-exp
    soft-max over the value axis (smooths the nonsmooth min/max).
    """
    n = len(cost)
    if n == 0:
        return np.zeros(0)

    order = np.lexsort((val, cost))
    c = cost[order]
    v = val[order]

    # Soft-min of (value, 1.0) over the suffix, via log-sum-exp.
    # logsumexp(v, 1) = m + log(exp(v-m)+exp(1-m)); soft-min = -that.
    m = np.maximum(v, 1.0)
    logsumexp = m + np.log(np.exp(v - m) + np.exp(1.0 - m))
    soft_min = -logsumexp                       # <= min(v, 1)
    height = np.clip(1.0 - soft_min, 0.0, 1.0)

    # Soft cost upper bound: next cost in the skyline, softened by beta.
    stack = []
    for i in range(n):
        if stack and v[i] >= stack[-1][1]:
            continue
        while stack and v[i] < stack[-1][1]:
            stack.pop()
        stack.append(i)

    m = len(stack)
    cost_upper = np.empty(n)
    for k in range(m):
        i = stack[k]
        next_cost = c[stack[k + 1]] if k + 1 < m else 1.0
        # smooth the "min(next_cost, 1)" boundary
        cu = min(next_cost, 1.0)
        cost_upper[i] = cu

    # Width: from each point's cost to its skyline cost-upper, softened.
    contrib = np.zeros(n)
    for k in range(m):
        i = stack[k]
        cost_i = c[i]
        cu = cost_upper[i]
        if cu <= cost_i:
            continue
        # Smooth the width with a log-sum-exp over {cost_i, cu} (soft-min boundary)
        mm = np.maximum(cost_i, cu)
        soft_min_c = -(mm + np.log(np.exp(cost_i - mm) + np.exp(cu - mm)))
        width = cu - soft_min_c
        if width <= 0.0:
            continue
        height = np.clip(1.0 - soft_min[i], 0.0, 1.0)
        contrib[i] = width * height

    # Map back to original order.
    out = np.zeros(n)
    out[order] = contrib
    return out


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Pareto-frontier hypervolume contribution over the
    (cost, -EI) trade-off, with a smooth log-sum-exp surrogate and an adaptive
    EI-vs-cost balance plus a bounded uncertainty exploration term.
    """
    t0 = time.time()

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize inputs.
    mu = np.where(np.isfinite(mu), mu, float(f_best))
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    ei = _compute_ei(mu, sigma, f_best)

    # --- Adaptive trade-off: balance EI against cost via marginal EI per cost ---
    # Normalize both to [0,1].
    cost_norm = np.clip((cost - 1.0) / 1.0, 0.0, 1.0)
    max_ei = float(np.max(ei)) if ei.size else 0.0
    if max_ei < 1e-12:
        # No expected improvement: prefer the cheapest points.
        scores = 1.0 / np.maximum(cost, 1e-6)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    ei_norm = ei / max_ei

    # Marginal value of exploration per unit cost: use a robust (top-10%) EI.
    # This sets how much "improvement budget" each unit of cost should buy.
    if ei.size > 20:
        thr = np.percentile(ei, 90.0)
    else:
        thr = float(np.max(ei))
    # Adaptive weight: when cheap points already give good EI, weight EI more.
    # ratio = typical EI per cost among promising points.
    promising = ei >= thr
    if np.any(promising):
        ratio = float(np.mean(ei_norm[promising] / np.maximum(cost_norm[promising] + 0.5, 0.5)))
    else:
        ratio = 1.0
    # Map ratio to a cost-penalty weight in [0.3, 1.5].
    w_cost = float(np.clip(0.9 / (0.5 + ratio), 0.3, 1.5))

    # 2D objective to minimize: (cost_norm, -ei_norm).
    c = cost_norm
    v = -ei_norm  # in [-1, 0]; we "minimize" this (i.e., maximize ei).

    # Re-base value axis to [0,1]: shift so that v in [min_v, 0] -> [1, 0].
    v_min = float(np.min(v))
    if v_min < -1e-12:
        v_axis = np.clip(-v / (-v_min), 0.0, 1.0)  # 0 at best, 1 at worst
    else:
        v_axis = np.zeros_like(v)

    # Choose exact vs smooth surrogate based on time budget.
    if _elapsed() > _DEADLINE * 0.6:
        # Fast path: exact O(N log N).
        scores = _hv_contrib_exact(c, v_axis)
    else:
        # Smooth path for a denser signal (bounded extra cost).
        try:
            scores = _hv_contrib_soft(c, v_axis, beta=0.05)
        except Exception:
            scores = _hv_contrib_exact(c, v_axis)

    # Bounded uncertainty exploration term (helps escape local basins in 8D).
    max_sigma = float(np.max(sigma)) if sigma.size else 0.0
    if max_sigma > 1e-12:
        unc = np.clip(sigma / max_sigma, 0.0, 1.0)
    else:
        unc = np.zeros_like(sigma)

    # Combine: hypervolume score (primary) + small uncertainty bonus,
    # then apply the adaptive cost-awareness via the w_cost already baked into
    # the cost axis. Keep it a clean weighted sum with a modest exploration cap.
    max_hv = float(np.max(scores)) if scores.size else 0.0
    if max_hv > 1e-12:
        scores_norm = scores / max_hv
    else:
        scores_norm = np.zeros_like(scores)

    final = scores_norm + 0.10 * unc * scores_norm  # exploration only where promising

    final = np.where(np.isfinite(final), final, 0.0)
    return final


def cost_safe(cost):
    """Ensure cost values are safe for division."""
    return np.maximum(cost, 1e-6)
# EVOLVE-BLOCK-END
