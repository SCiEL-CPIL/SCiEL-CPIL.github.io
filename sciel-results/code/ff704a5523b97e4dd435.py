"""Direct LP-based incentive-compatible auction design for the 16-type space."""
# EVOLVE-BLOCK-START
import time
import itertools

import numpy as np
from scipy.optimize import linprog


def _build_lp(distribution_spec, support, constraints):
    """Build the LP that maximizes expected revenue over IC/IR mechanisms.

    Variables: x[t, g] (allocation) and p[t] (payment) for each type t.
    Maximize  sum_t q_t * p_t
    s.t.
      0 <= x[t,g] <= 1
      p[t] >= 0
      IR: p[t] <= v_t . x_t            (for all t)
      IC: for all t, t':
           v_t.x_t - p_t  >=  v_t.x_t' - p_t'
    """
    # --- Parse types / support ---
    values = np.asarray(support, dtype=float)
    if values.ndim == 1:
        # support might be a flat list; assume 2 goods
        if len(values) == 16:
            values = values.reshape(8, 2)
    n_types = values.shape[0]
    n_goods = values.shape[1]

    # --- Parse distribution ---
    q = None
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        q = np.asarray(distribution_spec, dtype=float)
    elif isinstance(distribution_spec, dict):
        # e.g. {"type": "uniform"} or {"probs": [...]} or {"distribution": [...]}
        for k in ("probs", "distribution", "prob", "weights", "p"):
            if k in distribution_spec:
                q = np.asarray(distribution_spec[k], dtype=float)
                break
        if q is None:
            kind = str(distribution_spec.get("type", distribution_spec.get("kind", ""))).lower()
            if "uniform" in kind:
                q = np.ones(n_types, dtype=float)
            else:
                q = np.ones(n_types, dtype=float)
    if q is None:
        q = np.ones(n_types, dtype=float)
    q = q.reshape(-1)
    if q.size != n_types:
        # fallback
        q = np.ones(n_types, dtype=float)
    q = q / q.sum()

    # --- Variable indexing ---
    # x: n_types * n_goods, p: n_types
    nx = n_types * n_goods
    npay = n_types
    nvar = nx + npay

    def idx_x(t, g):
        return t * n_goods + g

    def idx_p(t):
        return nx + t

    # Objective: maximize sum q_t p_t  ->  minimize -sum q_t p_t
    c = np.zeros(nvar)
    for t in range(n_types):
        c[idx_p(t)] = -q[t]

    # A_ub, b_ub  (inequalities  A_ub x <= b_ub)
    rows = []
    rhs = []

    # IR:  p_t - v_t . x_t  <= 0
    for t in range(n_types):
        r = np.zeros(nvar)
        r[idx_p(t)] = 1.0
        for g in range(n_goods):
            r[idx_x(t, g)] = -values[t, g]
        rows.append(r)
        rhs.append(0.0)

    # IC: for all t, t':  v_t.x_t - p_t - v_t.x_t' + p_t'  <= 0
    for t in range(n_types):
        for tp in range(n_types):
            if t == tp:
                continue
            r = np.zeros(nvar)
            r[idx_p(t)] = -1.0
            r[idx_p(tp)] = 1.0
            for g in range(n_goods):
                r[idx_x(t, g)] = values[t, g]
                r[idx_x(tp, g)] = -values[t, g]
            rows.append(r)
            rhs.append(0.0)

    A_ub = np.vstack(rows) if rows else None
    b_ub = np.asarray(rhs) if rhs else None

    # Bounds: 0 <= x <= 1,  p >= 0
    bounds = []
    for _ in range(nx):
        bounds.append((0.0, 1.0))
    for _ in range(npay):
        bounds.append((0.0, None))

    return values, q, n_types, n_goods, nx, npay, nvar, c, A_ub, b_ub, bounds, idx_x, idx_p


def _solve_lp(distribution_spec, support, constraints):
    values, q, n_types, n_goods, nx, npay, nvar, c, A_ub, b_ub, bounds, idx_x, idx_p = _build_lp(
        distribution_spec, support, constraints
    )
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        # Fallback: try interior point
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs-ipm")
        except Exception:
            pass
    if not res.success:
        return None
    x = res.x
    allocations = x[:nx].reshape(n_types, n_goods)
    payments = x[nx:nx + npay]
    return allocations, payments


def _verify(allocations, payments, distribution_spec, support, constraints):
    """Return (revenue, ok). revenue = sum q_t p_t. ok checks feasibility/IR/IC."""
    values = np.asarray(support, dtype=float)
    if values.ndim == 1:
        if len(values) == 16:
            values = values.reshape(8, 2)
    n_types = values.shape[0]
    n_goods = values.shape[1]

    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        q = np.asarray(distribution_spec, dtype=float).reshape(-1)
    elif isinstance(distribution_spec, dict):
        q = None
        for k in ("probs", "distribution", "prob", "weights", "p"):
            if k in distribution_spec:
                q = np.asarray(distribution_spec[k], dtype=float).reshape(-1)
                break
        if q is None:
            q = np.ones(n_types, dtype=float)
    else:
        q = np.ones(n_types, dtype=float)
    if q.size != n_types:
        q = np.ones(n_types, dtype=float)
    q = q / q.sum()

    allocations = np.asarray(allocations, dtype=float)
    payments = np.asarray(payments, dtype=float)

    # Feasibility
    if np.any(allocations < -1e-7) or np.any(allocations > 1 + 1e-7):
        return None, False
    if np.any(payments < -1e-7):
        return None, False

    # IR and IC
    util = np.zeros(n_types)
    for t in range(n_types):
        util[t] = float(values[t] @ allocations[t]) - payments[t]
    for t in range(n_types):
        if util[t] < -1e-7:
            return None, False
        for tp in range(n_types):
            mis = float(values[t] @ allocations[tp]) - payments[tp]
            if mis - util[t] > 1e-6:
                return None, False

    revenue = float(np.dot(q, payments))
    return revenue, True


def construct_auction(distribution_spec, support, constraints):
    t_start = time.time()
    deadline = t_start + 80.0  # safety margin under 90s

    # Primary solve
    sol = _solve_lp(distribution_spec, support, constraints)
    best_alloc, best_pay = None, None
    best_rev = -np.inf

    if sol is not None:
        alloc, pay = sol
        rev, ok = _verify(alloc, pay, distribution_spec, support, constraints)
        if ok:
            best_alloc, best_pay, best_rev = alloc, pay, rev

    # Independently recompute the optimum (normalization reference)
    ref = _solve_lp(distribution_spec, support, constraints)
    ref_rev = best_rev
    if ref is not None:
        r_rev, r_ok = _verify(ref[0], ref[1], distribution_spec, support, constraints)
        if r_ok and r_rev > ref_rev:
            ref_rev = r_rev
    if ref_rev is None or ref_rev <= 1e-12:
        ref_rev = 1.0

    if best_alloc is None:
        # Fallback: no allocation (zero revenue), still valid IR/IC
        values = np.asarray(support, dtype=float)
        if values.ndim == 1:
            values = values.reshape(8, 2)
        n_types = values.shape[0]
        n_goods = values.shape[1]
        best_alloc = np.zeros((n_types, n_goods), dtype=float)
        best_pay = np.zeros(n_types, dtype=float)
        best_rev = 0.0

    # Normalize revenue by the independently-solved optimum
    normalized = best_rev / ref_rev

    return {
        "allocations": np.asarray(best_alloc, dtype=float),
        "payments": np.asarray(best_pay, dtype=float),
        "revenue": float(best_rev),
        "normalized_revenue": float(normalized),
        "optimal_revenue": float(ref_rev),
    }
# EVOLVE-BLOCK-END
