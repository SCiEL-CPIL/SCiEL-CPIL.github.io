# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    delta = 1.0 / N
    deadline = time.time() + 570  # safety margin before 630s kill

    # ---------- Keich construction (initial guess) ----------
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    # ---------- Exact 3D union volume (separable structure) ----------
    # Tube (i,j) at height c is the square
    #   [x_i + c*i/N, x_i + c*i/N + 1/N] x [y_j + c*j/N, y_j + c*j/N + 1/N]
    # Union at c is a union of N^2 axis-aligned squares.
    # Area(c) = (1/N) * U_x(c) * U_y(c)  where U_x is the 1D union length in x,
    # and similarly for y.  This identity holds because the union of products
    # of 1D sets equals the product of the unions.
    # Volume = ∫_0^1 Area(c) dc = (1/N) * ∫ U_x(c) U_y(c) dc.

    def union_length_1d(c, x):
        """Exact 1D union length of intervals [x_i + c*i/N, x_i + c*i/N + 1/N]."""
        starts = x + c * np.arange(N) / N
        ends = starts + delta
        order = np.argsort(starts)
        s = starts[order]
        e = ends[order]
        total = 0.0
        cs = s[0]
        ce = e[0]
        for k in range(1, N):
            if s[k] > ce:
                total += ce - cs
                cs = s[k]
                ce = e[k]
            else:
                if e[k] > ce:
                    ce = e[k]
        total += ce - cs
        return total

    def volume_exact(x, y, n_c=200):
        """Exact volume via dense quadrature over c (piecewise-linear, so dense trapezoid is very accurate)."""
        cs = np.linspace(0.0, 1.0, n_c)
        ux = np.array([union_length_1d(c, x) for c in cs])
        uy = np.array([union_length_1d(c, y) for c in cs])
        # trapezoidal integration of ux*uy
        area = ux * uy
        integral = np.trapz(area, cs)
        return integral * delta

    # ---------- Smooth surrogate for gradient-based optimization ----------
    # Soft-min: -logsumexp(-x)/T approximates min(x).  We want to MINIMIZE the
    # union, but the union length is a max-based quantity.  Instead, we build a
    # smooth lower bound / surrogate that is differentiable.
    #
    # A clean differentiable surrogate for the 1D union length:
    #   U(c) = sum over merged intervals.  Hard to differentiate directly.
    #
    # Alternative: use the identity that the union length of intervals of
    # equal width w starting at positions s_i is:
    #   U = sum_i w - sum of overlaps ... (inclusion-exclusion, not smooth).
    #
    # Better: represent union length as integral of an indicator:
    #   U(c) = ∫ 1_{exists i: s_i <= t <= s_i + w} dt
    #        = ∫ max_i 1_{t in [s_i, s_i+w]} dt
    # The max of indicators is non-smooth.  Smooth it with a softmax over
    # "membership scores".  For each t, define m_i(t) = a smooth bump that is
    # 1 inside interval i and 0 outside.  Then softmax gives a smooth
    # approximation to the indicator of the union.
    #
    # To keep it tractable and fast, I'll use a different smooth surrogate:
    # minimize the mean of (U(c))^p via a soft representation of U(c) built
    # from pairwise overlaps.  The union length can be written as:
    #   U = w * M  where M = number of "clusters" ... not smooth either.
    #
    # Practical smooth surrogate that correlates strongly with the true union:
    #   S = (1/N) * sum_i w  +  (1/N^2) * sum_{i<j} overlap_ij
    # This is NOT the union but a smooth quantity that is small when intervals
    # overlap a lot (which is what we want: overlap reduces union).  Actually
    # we want to MINIMIZE union, which happens when intervals overlap maximally.
    # The pairwise overlap sum is a smooth proxy: more overlap -> smaller union.
    # So minimizing (w*N - total_overlap) ~ minimizing union.
    #
    # total_overlap(c) = sum_{i<j} max(0, min(e_i,e_j) - max(s_i,s_j))
    # This is piecewise-linear in c, hence non-smooth, but we can smooth it.
    #
    # I'll use a softplus-smoothed version:
    #   soft_overlap = softplus(min(e_i,e_j) - max(s_i,s_j)) / T  (approx max(0, .))
    # and minimize  N*w - total_soft_overlap.

    def soft_overlap_sum(c, x, T=0.01):
        """Smooth approximation of total pairwise overlap of the N intervals at height c."""
        starts = x + c * np.arange(N) / N
        ends = starts + delta
        # pairwise
        # min_ends - max_starts for each pair
        # Use broadcasting
        se = starts[None, :]  # (1, N)
        ee = ends[None, :]
        ss = starts[:, None]
        es = ends[:, None]
        # overlap_ij = min(e_i, e_j) - max(s_i, s_j)
        min_e = np.minimum(ee, es)  # (N, N)
        max_s = np.maximum(se, ss)
        raw = min_e - max_s  # (N, N)
        # softplus(raw)/T approximates max(0, raw)
        # softplus(z) = log(1 + exp(z))
        # For numerical stability: softplus(z) = z + log1p(exp(-|z|)) - ... 
        # use: np.logaddexp(0, z)
        sp = np.logaddexp(0.0, raw) / T
        # Only count i<j (upper triangle) and divide by 2 not needed since we take upper
        iu = np.triu_indices(N, k=1)
        return sp[iu].sum()

    def smooth_objective(params, T=0.01, n_c=40):
        """Smooth surrogate: mean over c of (N*w - soft_overlap).  Lower = more overlap = smaller union."""
        x = params[:N]
        y = params[N:]
        cs = np.linspace(0.05, 0.95, n_c)
        total = 0.0
        for c in cs:
            ox = soft_overlap_sum(c, x, T)
            oy = soft_overlap_sum(c, y, T)
            # union_x ~ N*w - ox  (lower bound-ish proxy)
            # We want to minimize union, so minimize (N*w - ox) + (N*w - oy)
            total += (N * delta - ox) + (N * delta - oy)
        return total / len(cs)

    # ---------- Optimization ----------
    best_params = np.concatenate([keich, keich])
    best_vol = volume_exact(keich, keich)

    # Phase 1: gradient-based (L-BFGS-B) on the smooth surrogate
    if time.time() < deadline - 200:
        res = minimize(smooth_objective, best_params, method='L-BFGS-B',
                       options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6})
        if res.fun < smooth_objective(best_params):
            cand = res.x
            v = volume_exact(cand[:N], cand[N:])
            if v < best_vol:
                best_vol = v
                best_params = cand

    # Phase 2: Nelder-Mead on the exact volume (polish)
    def exact_obj(params):
        return volume_exact(params[:N], params[N:])

    if time.time() < deadline - 120:
        res = minimize(exact_obj, best_params, method='Nelder-Mead',
                       options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-9,
                                'adaptive': True})
        if res.fun < best_vol:
            best_vol = res.fun
            best_params = res.x

    # Phase 3: multi-start local search if time permits
    rng_local = np.random.default_rng(7)
    while time.time() < deadline - 60:
        # random perturbation
        scale = 0.05 if rng_local.random() < 0.8 else 0.15
        x_try = best_params[:N] + rng_local.normal(0, scale, size=N)
        y_try = best_params[N:] + rng_local.normal(0, scale, size=N)
        params_try = np.concatenate([x_try, y_try])
        v_try = exact_obj(params_try)
        if v_try < best_vol:
            best_vol = v_try
            best_params = params_try
            # local polish
            if time.time() < deadline - 30:
                res = minimize(exact_obj, best_params, method='Nelder-Mead',
                               options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-10,
                                        'adaptive': True})
                if res.fun < best_vol:
                    best_vol = res.fun
                    best_params = res.x

    # ---------- Build the 64x2 positions array ----------
    x_opt = best_params[:N]
    y_opt = best_params[N:]
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    base_x = x_opt[i_flat]
    base_y = y_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)

    return positions

# EVOLVE-BLOCK-END
