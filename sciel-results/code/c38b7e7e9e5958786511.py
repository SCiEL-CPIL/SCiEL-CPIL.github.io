"""Incentive-compatible finite auction design.

Strategy: parameterize the mechanism by the allocation rule only.
For a single additive bidder the payment is determined by the allocation
rule via the payment-integral (revenue equivalence), which guarantees
incentive compatibility *by construction* (regret = 0).  The objective
(expected revenue) is then a linear function of the allocation variables,
so the whole problem is a linear program.  We solve it exactly with a
high-precision polish pass and a deadline guard.
"""
# EVOLVE-BLOCK-START
import time
import math

import numpy as np
from scipy.optimize import linprog


def _time_budget():
    return 75.0  # safety margin well under the 90s kill


def construct_auction(distribution_spec, support, constraints):
    t0 = time.time()
    budget = _time_budget()

    # ------------------------------------------------------------------
    # Enumerate the 16 value types: two goods, each in {0,1,2,3}.
    # ------------------------------------------------------------------
    vals = [0, 1, 2, 3]
    types = [(a, b) for a in vals for b in vals]   # 16 types, index i
    n = len(types)                                  # 16
    nG = 2

    # Map type -> index
    idx = {t: i for i, t in enumerate(types)}

    # ------------------------------------------------------------------
    # Distribution: distribution_spec is a dict/structure over the 16
    # types.  Accept a few common encodings and fall back to uniform.
    # ------------------------------------------------------------------
    p = np.full(n, 1.0 / n, dtype=float)
    try:
        if isinstance(distribution_spec, dict):
            for t, pr in distribution_spec.items():
                if t in idx:
                    p[idx[t]] = float(pr)
            s = p.sum()
            if s > 0:
                p = p / s
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            arr = np.asarray(distribution_spec, dtype=float).ravel()
            if arr.size == n:
                s = arr.sum()
                if s > 0:
                    p = arr / s
    except Exception:
        p = np.full(n, 1.0 / n, dtype=float)

    # ------------------------------------------------------------------
    # Variables: allocation probability x[i, g] for type i, good g.
    #   0 <= x[i,g] <= 1
    # Payment by revenue equivalence (additive, anchored at v=0):
    #   pay(v) = sum_g \int_0^{v_g} x_g(v_g', v_{-g}) dv_g'
    # For a grid {0,1,2,3} we integrate piecewise-linearly in v_g with
    # the other good fixed.  Because the grid is uniform (step 1), the
    # integral of x_g over [0, v_g] is the trapezoidal sum of the values
    # of x_g at the intermediate grid points.
    # ------------------------------------------------------------------
    # x is a function of (v1, v2).  For good g we integrate over v_g
    # holding v_{-g} fixed.  Build the linear coefficient of each
    # x[i,g] in the expected revenue.
    #
    # Revenue = E_v[ pay(v) ]
    #         = sum_v p_v * sum_g \int_0^{v_g} x_g(v_g', v_{-g}) dv_g'
    #
    # The integral \int_0^{v_g} f(v_g') dv_g' with f defined on grid
    # {0,1,2,3} (piecewise linear) is the trapezoidal rule:
    #   \int_0^{v} f = sum_{k=0}^{v-1} 0.5*(f(k)+f(k+1))
    # This is linear in the f values, so we can build coefficients.
    #
    # We treat x_g as a function of the full type; for the integral we
    # hold v_{-g} fixed and vary v_g along the grid.

    # coeff[i, g] = coefficient of x[i,g] in revenue
    coeff = np.zeros((n, nG), dtype=float)

    for vi in range(n):
        v = types[vi]
        for g in range(nG):
            vg = v[g]
            other = v[1 - g]
            # trapezoidal integral of x_g over [0, vg] holding other fixed
            # f(k) = x_g(type (k, other) if g==0 else (other, k))
            for k in range(vg):
                # each f(k) and f(k+1) appears with weight 0.5
                for kk in (k, k + 1):
                    if g == 0:
                        ttype = (kk, other)
                    else:
                        ttype = (other, kk)
                    j = idx[ttype]
                    coeff[j, g] += p[vi] * 0.5

    # ------------------------------------------------------------------
    # LP: maximize  sum_{i,g} coeff[i,g] * x[i,g]
    #   subject to 0 <= x[i,g] <= 1
    # (IC holds by construction; IR: pay(0)=0 and pay(v)>=0 since x>=0,
    #  and v>=0 so utility = v.x - pay >= 0 is guaranteed by the
    #  payment-integral structure for a single bidder with x(0)=0.)
    # ------------------------------------------------------------------
    # Flatten variable vector: var[i*nG + g]
    c = -coeff.ravel()  # linprog minimizes
    A_ub = None
    b_ub = None
    bounds = [(0.0, 1.0)] * (n * nG)

    best_x = np.ones((n, nG), dtype=float)  # fallback: all-allocate
    best_rev = None

    def _time_left():
        return budget - (time.time() - t0)

    # Initial solve (fast method), then high-precision polish.
    for method, tol in (("highs", 1e-9), ("highs", 1e-12)):
        if _time_left() < 5.0:
            break
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                          method=method,
                          options={"presolve": True, "time_limit": max(1.0, _time_left() - 3.0)})
            if res is not None and res.status in (0, 1) and res.x is not None:
                x = res.x.reshape(n, nG)
                rev = float(coeff.ravel() @ x.ravel())
                if best_rev is None or rev > best_rev + 1e-12:
                    best_rev = rev
                    best_x = x
        except Exception:
            pass

    x = np.clip(best_x, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Payments via the payment-integral (exact on the grid).
    # ------------------------------------------------------------------
    payments = np.zeros(n, dtype=float)
    for vi in range(n):
        v = types[vi]
        pay = 0.0
        for g in range(nG):
            vg = v[g]
            other = v[1 - g]
            for k in range(vg):
                # trapezoid 0.5*(f(k)+f(k+1))
                if g == 0:
                    fk = x[idx[(k, other)], g]
                    fk1 = x[idx[(k + 1, other)], g]
                else:
                    fk = x[idx[(other, k)], g]
                    fk1 = x[idx[(other, k + 1)], g]
                pay += 0.5 * (fk + fk1)
        payments[vi] = pay

    # Ensure nonnegative payments (numerical safety).
    payments = np.clip(payments, 0.0, None)

    allocations = x

    # ------------------------------------------------------------------
    # Verification pass: check IR, feasibility, and regret (IC) on the
    # declared support.  If any constraint is violated beyond tolerance,
    # fall back to the safe all-allocate truthful mechanism.
    # ------------------------------------------------------------------
    def _verify(alloc, pay):
        # feasibility
        if np.any(alloc < -1e-9) or np.any(alloc > 1 + 1e-9):
            return False
        if np.any(pay < -1e-9):
            return False
        # regret: for each true type v, best report v' of u(v; v')
        for vi in range(n):
            v = types[vi]
            best = -math.inf
            for rj in range(n):
                r = types[rj]
                util = v[0] * alloc[rj, 0] + v[1] * alloc[rj, 1] - pay[rj]
                if util > best:
                    best = util
            truth = v[0] * alloc[vi, 0] + v[1] * alloc[vi, 1] - pay[vi]
            if best - truth > 1e-8:
                return False
        # IR: utility at truth >= 0
        for vi in range(n):
            v = types[vi]
            util = v[0] * alloc[vi, 0] + v[1] * alloc[vi, 1] - pay[vi]
            if util < -1e-8:
                return False
        return True

    if not _verify(allocations, payments):
        # Safe fallback: truthful all-allocate (value >= 0 -> allocate),
        # which is IC and IR for a single additive bidder.
        allocations = np.ones((n, nG), dtype=float)
        payments = np.zeros(n, dtype=float)
        for vi in range(n):
            v = types[vi]
            pay = 0.0
            for g in range(nG):
                vg = v[g]
                other = v[1 - g]
                for k in range(vg):
                    # x_g = 1 everywhere for positive; at 0 it is 1 too
                    # (allocation prob 1 for all types in fallback)
                    pay += 0.5 * (1.0 + 1.0)
            payments[vi] = pay
        # Recompute fallback payments exactly with x=1:
        # integral of 1 over [0, vg] = vg
        payments = np.array([types[vi][0] + types[vi][1] for vi in range(n)], dtype=float)
        if not _verify(allocations, payments):
            # Last resort: allocate nothing, pay nothing (trivially valid).
            allocations = np.zeros((n, nG), dtype=float)
            payments = np.zeros(n, dtype=float)

    return {
        "allocations": np.asarray(allocations, dtype=float),
        "payments": np.asarray(payments, dtype=float),
    }
# EVOLVE-BLOCK-END
