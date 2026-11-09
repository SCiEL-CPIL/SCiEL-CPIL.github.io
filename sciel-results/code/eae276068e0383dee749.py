# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def _build_constraints(K):
    """Build A_ub, b_ub for the M(x) <= 1 constraints.
    Variables: f(1), f(2), ..., f(K)  (K+1 variables)
    Constraint: for each x in 1..10K, sum_{k} f(k) * floor(x/k) <= 1
    """
    x_max = 10 * K
    # A_ub[i, j] = floor((i+1) / (j+1)) for i in 0..x_max-1, j in 0..K
    x_vals = np.arange(1, x_max + 1)          # shape (x_max,)
    k_vals = np.arange(1, K + 1)              # shape (K+1,)
    A_ub = np.floor(x_vals[:, None].astype(np.float64) / k_vals[None, :].astype(np.float64))
    b_ub = np.ones(x_max, dtype=np.float64)
    return A_ub, b_ub


def _clip_and_adjust(domain, values, K):
    """Apply the evaluator's clipping and f(1) adjustment.
    Returns adjusted values array aligned to 1..K (zeros where k not in domain).
    """
    # Clip values to [-10, 10]
    clipped = np.clip(values, -10.0, 10.0)

    # Build f array for k=1..K
    f = np.zeros(K + 1, dtype=np.float64)
    for i, k in enumerate(domain):
        if 1 <= k <= K:
            f[k] = clipped[i]

    # S = sum_k f(k)/k over the clipped values (only where k in domain)
    S = np.sum(clipped / domain.astype(np.float64))
    # f(1) is replaced by f(1) - S
    f[1] = f[1] - S
    return f


def _evaluate(f, K):
    """Compute M(x) for all x in 1..10K and the objective a.
    Returns (feasible, a).
    """
    x_max = 10 * K
    x_vals = np.arange(1, x_max + 1)
    k_vals = np.arange(1, K + 1)
    # M(x) = sum_k f(k) * floor(x/k)
    M = np.dot(np.floor(x_vals[:, None].astype(np.float64) / k_vals[None, :].astype(np.float64)), f[1:])
    feasible = bool(np.all(M <= 1.0001))
    if not feasible:
        return False, 0.0
    # a = -sum_k f(k) * log(k) / k
    log_k = np.log(k_vals)
    a = -np.sum(f[1:] * log_k / k_vals)
    return True, float(a)


def _solve_lp(K, time_limit):
    """Solve the LP: maximize -sum f(k) log(k)/k subject to M(x)<=1, f(k)<=10.
    f(1) is free (handled by the evaluator's adjustment), but we constrain f(2..K) <= 10.
    Actually, the evaluator clips to [-10,10] and adjusts f(1). For the LP, we can
    treat f(1) as a free variable (unbounded) since the evaluator will adjust it.
    But M(x) constraints involve f(1). Let's include f(1) as a variable with no bound
    (or very loose bounds) and f(2..K) bounded by [-10, 10].
    """
    n_vars = K + 1  # f(1), f(2), ..., f(K)
    # Objective: maximize -sum_{k=1}^{K} f(k) log(k)/k
    # c = -(-log(k)/k) = log(k)/k for minimization (linprog minimizes)
    k_vals = np.arange(1, K + 1)
    c = np.log(k_vals) / k_vals  # minimize sum c_j * f_j = sum f(k) log(k)/k
                                  # which is -a, so minimizing this maximizes a

    # Bounds: f(1) is free (the evaluator adjusts it), f(2..K) in [-10, 10]
    bounds = [(None, None)]  # f(1) free
    for k in range(2, K + 1):
        bounds.append((-10.0, 10.0))

    A_ub, b_ub = _build_constraints(K)

    t0 = time.time()
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        elapsed = time.time() - t0
        if elapsed > time_limit:
            return None
        if result.status != 0:
            return None
        f_opt = result.x
        # Build domain and values
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        # Use LP solution for k=2..K
        for k in range(2, K + 1):
            if abs(f_opt[k]) > 1e-9:
                # Add k to domain if not already present
                if k not in domain:
                    domain = np.append(domain, k)
                    values = np.append(values, f_opt[k])
                else:
                    idx = np.where(domain == k)[0][0]
                    values[idx] = f_opt[k]
        # Set f(1) from LP
        idx1 = np.where(domain == 1)[0]
        if len(idx1) > 0:
            values[idx1[0]] = f_opt[1]
        # Sort domain
        order = np.argsort(domain)
        domain = domain[order]
        values = values[order]
        return domain, values
    except Exception:
        return None


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    t_start = time.time()
    deadline = t_start + 600.0  # 600s budget

    # Seed fallback
    domain_seed = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values_seed = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    best_domain = domain_seed
    best_values = values_seed
    best_score = 1.0  # seed score

    # Try LP for increasing K
    for K in [20, 30, 40, 50, 60, 80, 100, 150, 200]:
        if time.time() > deadline - 10.0:
            break
        result = _solve_lp(K, time_limit=50.0)
        if result is not None:
            dom, val = result
            # Evaluate
            f = _clip_and_adjust(dom, val, K)
            feasible, a = _evaluate(f, K)
            if feasible:
                if a > best_score:
                    best_score = a
                    best_domain = dom
                    best_values = val

    # Final safety: return best found
    return best_domain, best_values


# EVOLVE-BLOCK-END
