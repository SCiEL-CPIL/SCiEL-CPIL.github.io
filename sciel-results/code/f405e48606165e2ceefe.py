# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _compute_area(x, n=64, M=20000):
    """
    Compute the area of the union of n triangles using quadrature at M heights.
    x: array of shape (n,) with base positions.
    Returns: area (float).
    """
    a = (np.arange(n) + 1.0) / n  # a_i = (i+1)/n
    # y_m = (m + 0.5) / M for m = 0..M-1
    y = (np.arange(M) + 0.5) / M  # shape (M,)
    
    # L_i = x_i + a_i * y  -> shape (n, M)
    # w = (1 - y) / n     -> shape (M,)
    # We need to sort L_i along axis=0 for each y, then compute union length.
    
    # L[i, m] = x[i] + a[i] * y[m]
    L = x[:, None] + a[:, None] * y[None, :]  # shape (n, M)
    w = (1.0 - y) / n  # shape (M,)
    
    # Sort L along axis 0 (for each column m)
    L_sorted = np.sort(L, axis=0)  # shape (n, M)
    
    # Union length at each height m:
    # w + sum_{j=0}^{n-2} min(L_sorted[j+1, m] - L_sorted[j, m], w[m])
    diffs = L_sorted[1:, :] - L_sorted[:-1, :]  # shape (n-1, M)
    union_lengths = w[None, :] + np.sum(np.minimum(diffs, w[None, :]), axis=0)  # shape (M,)
    
    area = np.mean(union_lengths)
    return area


def _compute_area_fast(x, n=64, M=20000):
    """
    Optimized area computation.

    Uses the identity:
        union_length = max(L) - min(L) + w*(n-1) - sum_j max(gap_j, w)
    where gap_j = L_(j+1) - L_(j) over the sorted L.  This avoids a full
    sort: we only need max, min, and the sum of positive excess gaps.
    """
    a = (np.arange(n) + 1.0) / n
    y = (np.arange(M) + 0.5) / M

    L = x[:, None] + a[:, None] * y[None, :]
    w = (1.0 - y) / n

    L_sorted = np.sort(L, axis=0)
    diffs = L_sorted[1:, :] - L_sorted[:-1, :]
    # union = w + sum(min(diffs, w)) = w + sum(diffs) - sum(max(diffs - w, 0))
    # sum(diffs) = max(L) - min(L)
    union_lengths = w + (L_sorted[-1, :] - L_sorted[0, :]) - np.sum(np.maximum(diffs - w[None, :], 0.0), axis=0)
    # Note: w + (max-min) - sum(max(diffs-w,0))
    # Verify: w + sum(min(diffs,w)) = w + sum(diffs) - sum(max(diffs-w,0)) = w + (max-min) - sum(max(diffs-w,0))  ✓

    area = np.mean(union_lengths)
    return area


def _compute_area_softmin(x, n=64, M=20000, beta=50.0):
    """
    Smooth surrogate for the union area using a soft-min (log-sum-exp based).

    The exact union length at height m is:
        w + sum_j min(L_(j+1) - L_(j), w)
    A smooth approximation replaces min(a, b) with -softmax(-a, -b) scaled,
    but a simpler and effective approach: use the identity
        union = w + sum(min(diffs, w))
    and approximate min(d, w) via a smooth min:
        smin(d, w) = -logsumexp(-d, -w) / beta  (for large beta, approaches min)
    Actually: smin(a,b) = -log(exp(-beta*a) + exp(-beta*b))/beta
    For large beta this approaches min(a,b).

    However, we still need sorted L to get the gaps.  The sort is the expensive
    part.  Instead, we use a different smooth formulation:

    union_length = max_i L_i - min_i L_i + w*(n-1) - sum_j max(gap_j, w)

    The max(gap_j, w) term is the non-smooth part.  We approximate:
        max(g, w) ≈ softplus(g - w) / beta + w  (for large beta)
    Actually max(g,w) = g * 1[g>w] + w * 1[g<=w]
    Smooth: max(g,w) ≈ (g + w + softplus(beta*(g-w)))/2 ... not quite.

    Let's use: max(g, w) ≈ (g + w)/2 + (1/beta)*log(cosh(beta*(g-w)/2))  -- no, that's not right either.

    Simplest correct smooth max:  smax(g, w) = (1/beta)*log(exp(beta*g) + exp(beta*w))
    For large beta, smax -> max(g, w).

    So: union ≈ w + (max L - min L) - sum_j smax(gap_j, w) / 1 ... wait, we subtract sum(max(gap_j, w)).

    union ≈ w + (maxL - minL) - sum_j smax(gap_j, w)

    where smax(g, w) = (1/beta)*log(exp(beta*g) + exp(beta*w))

    But maxL and minL are also non-smooth.  We can smooth them too:
    smax_all(L) = (1/beta)*logsumexp(beta*L)
    smin_all(L) = -(1/beta)*logsumexp(-beta*L)

    This gives a fully smooth objective.  The cost is computing logsumexp over n values
    at each of M heights, which is O(n*M) with no sort -- same as before but smoother.
    """
    a = (np.arange(n) + 1.0) / n
    y = (np.arange(M) + 0.5) / M

    L = x[:, None] + a[:, None] * y[None, :]  # (n, M)
    w = (1.0 - y) / n  # (M,)

    # Smooth max and min over axis 0
    bL = beta * L  # (n, M)
    # logsumexp along axis 0
    bL_max = np.max(bL, axis=0, keepdims=True)  # (1, M)
    logsumexp_max = np.log(np.sum(np.exp(bL - bL_max), axis=0, keepdims=True)) + bL_max  # (1, M)
    smax_L = logsumexp_max[0, :] / beta  # (M,)

    bL_neg = -beta * L
    bL_neg_max = np.max(bL_neg, axis=0, keepdims=True)
    logsumexp_min = np.log(np.sum(np.exp(bL_neg - bL_neg_max), axis=0, keepdims=True)) + bL_neg_max
    smin_L = -logsumexp_min[0, :] / beta  # (M,)

    # For gaps: we still need sorted L.  But for the smooth version, let's just
    # use the exact sort for the gap computation and only smooth the max(gap, w) part.
    # This is a hybrid: sort is O(n log n * M) but the smoothing removes the kinks
    # in the objective that cause CD/NM to stall.

    L_sorted = np.sort(L, axis=0)
    diffs = L_sorted[1:, :] - L_sorted[:-1, :]  # (n-1, M)

    # Smooth max(diffs, w): smax(d, w) = (1/beta)*log(exp(beta*d) + exp(beta*w))
    # = (d + w)/2 + (1/beta)*log(cosh(beta*(d-w)/2)) ... let's use the log-sum-exp form
    bd = beta * diffs  # (n-1, M)
    bw = beta * w[None, :]  # (1, M)
    # smax = (1/beta) * logsumexp(bd, bw) along axis 0
    tmp = np.stack([bd, bw], axis=0)  # (2, n-1, M)
    tmp_max = np.max(tmp, axis=0, keepdims=True)  # (1, n-1, M)
    logsumexp_smax = np.log(np.sum(np.exp(tmp - tmp_max), axis=0, keepdims=True)) + tmp_max  # (1, n-1, M)
    smax_gaps = logsumexp_smax[0, :, :] / beta  # (n-1, M)

    sum_smax_gaps = np.sum(smax_gaps, axis=0)  # (M,)

    union_lengths = w + (smax_L - smin_L) - sum_smax_gaps
    area = np.mean(union_lengths)
    return area


def _keich_initial(n=64):
    """
    Keich construction for initial positions.
    """
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)


def _breakpoint_cd_1d(x, i, n=64, M=20000, deadline=None):
    """
    Exact 1-D coordinate descent for coordinate i.

    The area as a function of x_i is piecewise linear (or piecewise smooth in
    the soft-min version).  The breakpoints occur when the ordering of L values
    changes, i.e., when x_i + a_i * y_m = x_j + a_j * y_m for some j, m.
    For fixed j and m, this gives:
        x_i = x_j + (a_j - a_i) * y_m
    So the breakpoints in x_i are at x_j + (a_j - a_i) * y_m for all j != i, m.

    We collect all such breakpoints, sort them, and evaluate the area at each
    interval midpoint and at the breakpoints themselves.  The minimum of a
    piecewise-linear function over an interval occurs at an endpoint.
    """
    a = (np.arange(n) + 1.0) / n
    y = (np.arange(M) + 0.5) / M

    # Collect breakpoints: x_i = x_j + (a_j - a_i) * y_m
    # For efficiency, we only need breakpoints within a reasonable range of x[i]
    x_i_current = x[i]
    # Breakpoints: for each j != i, x_j + (a_j - a_i) * y_m
    # This is a lot: (n-1) * M breakpoints.  For n=64, M=20000, that's ~1.28M.
    # We can filter to a window around x_i_current.

    # Compute all breakpoints efficiently
    j_idx = np.arange(n)
    j_idx = j_idx[j_idx != i]
    a_j = a[j_idx]  # (n-1,)
    x_j = x[j_idx]  # (n-1,)

    # breakpoints[j, m] = x_j[j] + (a_j[j] - a[i]) * y[m]
    diff_a = a_j - a[i]  # (n-1,)
    # Shape: (n-1, M)
    breakpoints = x_j[:, None] + diff_a[:, None] * y[None, :]

    # Filter to a window: [x_i_current - 2.0, x_i_current + 2.0]
    lo_bound = x_i_current - 2.0
    hi_bound = x_i_current + 2.0
    mask = (breakpoints >= lo_bound) & (breakpoints <= hi_bound)
    bp_filtered = breakpoints[mask]

    if len(bp_filtered) == 0:
        # No breakpoints in range; the function is linear here.
        # Evaluate at a few points to determine direction.
        test_pts = np.array([x_i_current - 0.1, x_i_current, x_i_current + 0.1])
        areas = []
        for tp in test_pts:
            x_trial = x.copy()
            x_trial[i] = tp
            areas.append(_compute_area_fast(x_trial, n, M))
        best_k = int(np.argmin(areas))
        new_xi = test_pts[best_k]
        return new_xi, areas[best_k]

    # Sort unique breakpoints
    bp_sorted = np.unique(np.round(bp_filtered, decimals=10))

    # Candidate points: breakpoints + midpoints between consecutive breakpoints + boundaries
    candidates = list(bp_sorted)
    if len(bp_sorted) > 1:
        mids = (bp_sorted[:-1] + bp_sorted[1:]) / 2.0
        candidates.extend(mids.tolist())
    # Add boundaries
    candidates.append(lo_bound)
    candidates.append(hi_bound)
    candidates.append(x_i_current)

    candidates = np.array(candidates)
    candidates = np.unique(np.round(candidates, decimals=10))

    # Evaluate area at each candidate
    best_val = np.inf
    best_xi = x_i_current
    for tp in candidates:
        if deadline is not None and time.time() > deadline:
            break
        x_trial = x.copy()
        x_trial[i] = tp
        val = _compute_area_fast(x_trial, n, M)
        if val < best_val:
            best_val = val
            best_xi = tp

    return best_xi, best_val


def _coordinate_descent(x, n=64, M=20000, max_iters=500, tol=1e-12, deadline=None):
    """
    Coordinate descent using exact breakpoint-aware 1-D minimization.
    """
    current_area = _compute_area_fast(x, n, M)
    best_x = x.copy()
    best_area = current_area

    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break

            new_xi, new_area = _breakpoint_cd_1d(x, i, n, M, deadline)

            if new_area < current_area - tol:
                x[i] = new_xi
                current_area = new_area
                improved = True

        if not improved:
            break

        if current_area < best_area:
            best_area = current_area
            best_x = x.copy()

    return best_x, best_area


def _nelder_mead_softmin(x, n=64, M=20000, deadline=None, max_iter=200):
    """
    Nelder-Mead on the smooth soft-min surrogate to escape local minima
    of the piecewise-linear objective.
    """
    from scipy.optimize import minimize

    beta = 50.0

    def objective(x_vec):
        return _compute_area_softmin(x_vec, n, M, beta=beta)

    x0 = x.copy()
    res = minimize(
        objective, x0, method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10}
    )
    x_nm = res.x
    # Evaluate with the exact (non-smooth) objective
    area_exact = _compute_area_fast(x_nm, n, M)
    return x_nm, area_exact


def _simulated_annealing(x, n=64, M=20000, deadline=None, max_steps=50000):
    """
    Simulated annealing to escape local minima.
    """
    current_area = _compute_area_fast(x, n, M)
    best_x = x.copy()
    best_area = current_area

    T = 0.01  # initial temperature
    cooling = 0.999

    rng = np.random.default_rng(42)

    step = 0
    while step < max_steps:
        if deadline is not None and time.time() > deadline:
            break

        # Perturb one or a few coordinates
        n_perturb = int(rng.integers(1, 4))
        indices = rng.choice(n, size=n_perturb, replace=False)
        x_trial = x.copy()
        x_trial[indices] += rng.normal(0, T, size=n_perturb)

        trial_area = _compute_area_fast(x_trial, n, M)

        delta = trial_area - current_area
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            x = x_trial
            current_area = trial_area
            if current_area < best_area:
                best_area = current_area
                best_x = x.copy()

        T *= cooling
        step += 1

    return best_x, best_area


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem.
    """
    n = 64
    M = 20000

    start_time = time.time()
    deadline = start_time + 600.0  # 600s wall-clock limit

    # Start with Keich construction
    x = _keich_initial(n)
    current_area = _compute_area_fast(x, n, M)
    best_x = x.copy()
    best_area = current_area

    # Phase 1: Breakpoint-aware coordinate descent
    x_cd, area_cd = _coordinate_descent(x, n, M, max_iters=100, tol=1e-12, deadline=deadline - 60)
    if area_cd < best_area:
        best_area = area_cd
        best_x = x_cd.copy()

    # Phase 2: Nelder-Mead on smooth surrogate to escape local minima
    if time.time() < deadline - 60:
        x_nm, area_nm = _nelder_mead_softmin(best_x, n, M, deadline=deadline - 30, max_iter=150)
        if area_nm < best_area:
            best_area = area_nm
            best_x = x_nm.copy()

    # Phase 3: Coordinate descent refinement after NM
    if time.time() < deadline - 30:
        x_cd2, area_cd2 = _coordinate_descent(best_x, n, M, max_iters=100, tol=1e-12, deadline=deadline - 20)
        if area_cd2 < best_area:
            best_area = area_cd2
            best_x = x_cd2.copy()

    # Phase 4: Simulated annealing for global exploration
    if time.time() < deadline - 20:
        x_sa, area_sa = _simulated_annealing(best_x, n, M, deadline=deadline - 10, max_steps=15000)
        if area_sa < best_area:
            best_area = area_sa
            best_x = x_sa.copy()

    # Phase 5: Final coordinate descent polish
    if time.time() < deadline - 5:
        x_final, area_final = _coordinate_descent(best_x, n, M, max_iters=50, tol=1e-14, deadline=deadline - 1)
        if area_final < best_area:
            best_area = area_final
            best_x = x_final.copy()

    # Ensure validity
    if not np.all(np.isfinite(best_x)) or best_area <= 0:
        return _keich_initial(n)

    return best_x


# EVOLVE-BLOCK-END
