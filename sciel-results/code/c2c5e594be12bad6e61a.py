"""Direct finite-support incentive-compatible auction solver (no baseline import)."""
# EVOLVE-BLOCK-START
import time

def _get_support(support):
    """Return a 16x2 integer array of (v1,v2) for the declared report/value support."""
    import numpy as np
    if support is not None:
        try:
            arr = np.asarray(support, dtype=int)
            if arr.shape == (16, 2):
                return arr
        except Exception:
            pass
    return np.array([[v1, v2] for v1 in range(4) for v2 in range(4)], dtype=int)

def _get_probs(distribution_spec, n_reports):
    """Return a probability vector over reports from the distribution spec."""
    import numpy as np
    if distribution_spec is not None:
        try:
            # dict mapping report index -> prob, or (v1,v2)->prob
            if isinstance(distribution_spec, dict):
                p = np.zeros(n_reports, dtype=float)
                for k, v in distribution_spec.items():
                    if isinstance(k, (list, tuple)) and len(k) == 2:
                        idx = int(k[0]) * 4 + int(k[1])
                    else:
                        idx = int(k)
                    if 0 <= idx < n_reports:
                        p[idx] = float(v)
                s = p.sum()
                if s > 0:
                    return p / s
            # array-like of length 16
            arr = np.asarray(distribution_spec, dtype=float).ravel()
            if arr.size == n_reports:
                s = arr.sum()
                if s > 0:
                    return arr / s
        except Exception:
            pass
    return np.full(n_reports, 1.0 / n_reports, dtype=float)

def _build_mechanism(prob, v1, v2):
    """
    Solve the finite-support IC mechanism via a polytope formulation.
    Variables: x1[16], x2[16], p[16]  (prob of good1, good2, payment for each report)
    Objective: maximize sum_i prob_i * p_i
    Constraints:
      0 <= x1, x2 <= 1
      x1 + x2 <= 1
      p >= 0
      IC: for each report r and each alternative a:
          p_r - p_a >= v1_r*x1_r + v2_r*x2_r - (v1_a*x1_a + v2_a*x2_a)
      IR: for each report r:
          p_r <= v1_r*x1_r + v2_r*x2_r
    """
    import numpy as np
    from scipy.optimize import linprog

    n = 16
    # variable ordering: x1[0..15], x2[16..31], p[32..47]
    c = np.zeros(48)
    c[32:48] = -prob  # maximize revenue

    A_ub = []
    b_ub = []

    # 0 <= x1 <= 1, 0 <= x2 <= 1 handled by bounds
    # x1 + x2 <= 1
    for i in range(n):
        row = np.zeros(48)
        row[i] = 1.0
        row[16 + i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0)

    # IC constraints: p_r - p_a >= val_r - val_a  =>  -p_r + p_a + val_r - val_a <= 0
    for r in range(n):
        vr = v1[r] * 1.0 + v2[r] * 1.0  # not used directly; need full dot
        for a in range(n):
            if a == r:
                continue
            row = np.zeros(48)
            row[32 + r] = -1.0
            row[32 + a] = 1.0
            # val_r - val_a = v1_r*x1_r + v2_r*x2_r - (v1_a*x1_a + v2_a*x2_a)
            row[r] += v1[r]
            row[16 + r] += v2[r]
            row[a] -= v1[a]
            row[16 + a] -= v2[a]
            A_ub.append(row)
            b_ub.append(0.0)

    # IR: p_r <= val_r  =>  p_r - v1_r*x1_r - v2_r*x2_r <= 0
    for r in range(n):
        row = np.zeros(48)
        row[32 + r] = 1.0
        row[r] -= v1[r]
        row[16 + r] -= v2[r]
        A_ub.append(row)
        b_ub.append(0.0)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    bounds = []
    for i in range(32):
        bounds.append((0.0, 1.0))
    for i in range(32, 48):
        bounds.append((0.0, None))

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if res.status != 0:
        # Fallback: deterministic allocation, payment = value (zero revenue but feasible)
        x1 = np.zeros(n)
        x2 = np.zeros(n)
        p = np.zeros(n)
        # allocate good1 to report with highest v1, else good2
        best1 = int(np.argmax(v1))
        x1[best1] = 1.0
        p[best1] = v1[best1]
        return x1, x2, p

    x = res.x
    x1 = x[0:16].clip(0, 1)
    x2 = x[16:32].clip(0, 1)
    p = x[32:48].clip(0, None)
    return x1, x2, p

def _verify(x1, x2, p, prob, v1, v2, tol=1e-8):
    """Check feasibility; return (ok, regret)."""
    import numpy as np
    n = 16
    # feasibility
    if np.any(x1 < -tol) or np.any(x1 > 1 + tol) or np.any(x2 < -tol) or np.any(x2 > 1 + tol):
        return False, 1e9
    if np.any(x1 + x2 > 1 + tol):
        return False, 1e9
    if np.any(p < -tol):
        return False, 1e9
    # IC regret
    regret = 0.0
    for r in range(n):
        for a in range(n):
            if a == r:
                continue
            lhs = p[r] - p[a]
            rhs = (v1[r]*x1[r] + v2[r]*x2[r]) - (v1[a]*x1[a] + v2[a]*x2[a])
            gap = rhs - lhs
            if gap > regret:
                regret = gap
    # IR
    for r in range(n):
        ir = (v1[r]*x1[r] + v2[r]*x2[r]) - p[r]
        if -ir > regret:
            regret = -ir
    return regret <= tol, regret

def construct_auction(distribution_spec, support, constraints):
    import numpy as np
    t_start = time.time()
    deadline = t_start + 80.0  # safety margin under 90s

    v1, v2 = _get_support(support)
    prob = _get_probs(distribution_spec, 16)

    x1, x2, p = _build_mechanism(prob, v1, v2)

    ok, regret = _verify(x1, x2, p, prob, v1, v2)
    if not ok:
        # sanitize
        x1 = x1.clip(0, 1)
        x2 = x2.clip(0, 1)
        # enforce x1+x2<=1
        over = x1 + x2 - 1.0
        over = np.maximum(over, 0.0)
        x1 = x1 - over / 2.0
        x2 = x2 - over / 2.0
        p = p.clip(0, None)
        # re-check; if still bad, zero payments
        ok, regret = _verify(x1, x2, p, prob, v1, v2)
        if not ok:
            p = np.minimum(p, v1 * x1 + v2 * x2)
            ok, regret = _verify(x1, x2, p, prob, v1, v2)
        if not ok:
            x1 = np.zeros(16)
            x2 = np.zeros(16)
            p = np.zeros(16)

    if time.time() > deadline:
        pass  # already have a result

    return {
        'allocations': np.column_stack((x1, x2)),
        'payments': p,
    }
# EVOLVE-BLOCK-END
