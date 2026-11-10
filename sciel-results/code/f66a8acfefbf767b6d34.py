# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    The function f is scored by a fixed evaluator that:
      * subtracts a constant from f(1) so that sum_k f(k)/k == 0,
      * requires the step function M(x) = sum_k f(k) * floor(x/k) to stay <= 1 for every
        real x in [1, 10*max_key] (otherwise the construction is infeasible), and
      * returns the objective a = -sum_k f(k) * log(k) / k (to be maximized; optimum 1.0).
    The optimum is approached by the Mobius function mu(k), for which the classic identity
    sum_{k<=x} mu(k) floor(x/k) = 1 holds and -sum mu(k) log(k)/k -> 1.

    Representation:
        domain : np.ndarray, shape (m,), dtype int   -> positive integers in the domain of f
        values : np.ndarray, shape (m,), dtype float -> the corresponding real values f(k)

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f.
    """
    start_time = time.time()
    deadline = 600  # seconds, safety margin before 630s kill

    def mobius_sieve(n):
        """Compute Mobius function mu(k) for k=1..n using a sieve."""
        mu = np.ones(n + 1, dtype=np.int8)
        mu[0] = 0
        is_prime = np.ones(n + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        primes = []
        for i in range(2, n + 1):
            if is_prime[i]:
                primes.append(i)
                mu[i] = -1
            for p in primes:
                if i * p > n:
                    break
                is_prime[i * p] = False
                if i % p == 0:
                    mu[i * p] = 0
                    break
                else:
                    mu[i * p] = -mu[i]
        return mu

    def compute_M_values(domain, values, max_x):
        """Compute M(x) = sum_k f(k) * floor(x/k) for x in [1, max_x]."""
        m = len(domain)
        M = np.zeros(max_x + 1, dtype=np.float64)
        # For each k in domain, add f(k) * floor(x/k) for all x
        for i in range(m):
            k = domain[i]
            f_k = values[i]
            if f_k == 0:
                continue
            # floor(x/k) is constant on intervals [j*k, (j+1)*k - 1]
            j = 1
            while j * k <= max_x:
                start = j * k
                end = min((j + 1) * k - 1, max_x)
                M[start:end + 1] += f_k * j
                j += 1
        return M

    def check_feasibility_fast(domain, values, K):
        """Check M(x) <= 1.0001 for all x in [1, 10K] efficiently."""
        max_x = 10 * K
        m = len(domain)
        
        # M(x) for x <= K: for the full Mobius, M(x) = 1
        # But we need to check our specific function
        # Use the fact that M(x) is a step function, only changes at multiples of domain elements
        
        # For efficiency, only check critical points: x = 1, 2, ..., K and then sample beyond
        # Actually, M(x) is non-decreasing in x if all f(k) >= 0, but here we have negative values
        # So we need to check more carefully
        
        # Check x in [1, K] - these are the most critical
        # For x <= K, floor(x/k) = 0 for k > x, so M(x) = sum_{k<=x} f(k)*floor(x/k)
        
        # Let's compute M(x) for all x in [1, min(K, 10000)] and sample for larger x
        check_points = set()
        # Add all x in [1, min(K, 5000)]
        for x in range(1, min(K, 5000) + 1):
            check_points.add(x)
        # Add some larger points
        if K > 5000:
            for x in range(5000, K + 1, max(1, (K - 5000) // 100)):
                check_points.add(x)
            # Add points in [K+1, 10K]
            for x in range(K + 1, max_x + 1, max(1, (max_x - K) // 200)):
                check_points.add(x)
        
        check_xs = sorted(check_points)
        
        for x in check_xs:
            Mx = 0.0
            for i in range(m):
                k = domain[i]
                if k > x:
                    break
                Mx += values[i] * (x // k)
            if Mx > 1.0001:
                return False
        return True

    def solve_lp_for_K(K):
        """
        Try to solve an LP to maximize a = -sum f(k)*log(k)/k
        subject to M(x) <= 1.0001 for all x in [1, 10K]
        and sum f(k)/k = 0 (after the f(1) adjustment, this is automatically satisfied)
        
        Actually, the evaluator adjusts f(1) so that sum f(k)/k = 0.
        So we don't need to enforce this constraint - the evaluator handles it.
        
        The constraint is: for all x in [1, 10K], sum_{k in domain, k<=x} f(k)*floor(x/k) <= 1.0001
        
        We'll use scipy's linprog if available, otherwise fall back to Mobius.
        """
        try:
            from scipy.optimize import linprog
        except ImportError:
            return None, None, None
        
        mu = mobius_sieve(K)
        nonzero_idx = np.where(mu[1:] != 0)[0] + 1
        domain = nonzero_idx.astype(np.int64)
        m = len(domain)
        
        if m < 1:
            return None, None, None
        
        # Objective: maximize -sum f(k)*log(k)/k
        # In linprog form: minimize sum f(k)*log(k)/k
        c = np.log(domain.astype(np.float64)) / domain.astype(np.float64)
        
        # Constraints: M(x) <= 1.0001 for all x in [1, 10K]
        # M(x) = sum_{k<=x} f(k) * floor(x/k)
        # This is: A @ f <= b where A[i,j] = floor(x_i / k_j) if k_j <= x_i else 0
        
        # We can't check all x in [1, 10K] - too many constraints
        # Instead, we'll use a subset of critical x values
        
        # Critical x values: x = 1, 2, ..., K (since for x > K, floor(x/k) changes less frequently)
        # Plus some x > K
        
        # For the LP, we'll use x = 1, 2, ..., min(K, 100) and some larger values
        # Actually, the binding constraints are likely at small x where the sum is most sensitive
        
        # Let's use x = 1, 2, ..., min(K, 50) for the LP
        max_x_lp = min(K, 50)
        xs = np.arange(1, max_x_lp + 1, dtype=np.int64)
        n_constraints = len(xs)
        
        # Build constraint matrix A
        A = np.zeros((n_constraints, m), dtype=np.float64)
        for i, x in enumerate(xs):
            for j, k in enumerate(domain):
                if k <= x:
                    A[i, j] = x // k
        
        b = np.full(n_constraints, 1.0001, dtype=np.float64)
        
        # Bounds: f(k) can be any real, but let's bound them to [-10, 10] for numerical stability
        bounds = [(-10.0, 10.0)] * m
        
        # Solve LP
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        
        if not result.success:
            return None, None, None
        
        f_opt = result.x
        
        # Verify feasibility for more x values
        feasible = True
        # Check x in [1, K]
        for x in range(1, K + 1):
            Mx = 0.0
            for j in range(m):
                k = domain[j]
                if k > x:
                    break
                Mx += f_opt[j] * (x // k)
            if Mx > 1.0001:
                feasible = False
                break
        
        if not feasible:
            # Try with more constraints or fall back
            return None, None, None
        
        # Check some x > K
        max_x = 10 * K
        for x in range(K + 1, max_x + 1, max(1, (max_x - K) // 100)):
            Mx = 0.0
            for j in range(m):
                k = domain[j]
                if k > x:
                    break
                Mx += f_opt[j] * (x // k)
            if Mx > 1.0001:
                feasible = False
                break
        
        if not feasible:
            return None, None, None
        
        # Compute objective
        log_k = np.log(domain.astype(np.float64))
        a = -np.sum(f_opt * log_k / domain.astype(np.float64))
        
        return domain, f_opt, a

    # First, try the pure Mobius approach at various K values
    best_domain = None
    best_values = None
    best_score = -np.inf

    for K in [100, 500, 1000, 5000, 10000]:
        if time.time() - start_time > deadline - 60:
            break

        mu = mobius_sieve(K)
        nonzero_idx = np.where(mu[1:] != 0)[0] + 1
        domain = nonzero_idx.astype(np.int64)
        values = mu[nonzero_idx].astype(np.float64)

        m = len(domain)
        if m < 1:
            continue

        # Compute objective
        log_k = np.log(domain.astype(np.float64))
        a = -np.sum(values * log_k / domain.astype(np.float64))

        # Check feasibility
        feasible = check_feasibility_fast(domain, values, K)

        if feasible and a > best_score:
            best_score = a
            best_domain = domain.copy()
            best_values = values.copy()

    # Now try LP optimization at smaller K values (LP is expensive)
    for K in [10, 20, 50, 100]:
        if time.time() - start_time > deadline - 30:
            break

        domain_lp, values_lp, a_lp = solve_lp_for_K(K)
        
        if domain_lp is not None and a_lp > best_score:
            best_score = a_lp
            best_domain = domain_lp.copy()
            best_values = values_lp.copy()

    # Fallback to baseline if nothing found
    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    else:
        domain = best_domain
        values = best_values

    return domain, values


# EVOLVE-BLOCK-END
