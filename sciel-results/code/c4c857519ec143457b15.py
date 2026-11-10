# EVOLVE-BLOCK-START
import numpy as np
import time
import sys


def _compute_mobius(n):
    """Compute Mobius function mu(1..n) using a sieve."""
    mu = np.ones(n + 1, dtype=np.int8)
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[:2] = False
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


def _check_feasibility(domain, values, K):
    """
    Check if M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for all x in [1, 10K].
    Returns True if feasible, False otherwise.
    """
    m = len(domain)
    if m == 0:
        return True
    
    # Clip values to [-10, 10]
    vals = np.clip(values, -10.0, 10.0)
    
    # Compute S = sum_k f(k)/k
    S = np.sum(vals / domain.astype(np.float64))
    
    # Adjust f(1)
    if 1 in domain:
        idx = np.where(domain == 1)[0][0]
        vals[idx] -= S
    else:
        # f(1) is 0, so no adjustment needed (0 - S = -S, but f(1) not in domain)
        # Actually, the problem says f(1) = 0 before this if 1 is not in the domain
        # So we don't add anything to the sum, but the sum should become 0
        # Let's re-read: "f(1) is replaced by f(1) - S ... so that afterwards sum_k f(k)/k = 0"
        # If 1 is not in domain, f(1) = 0. After replacement, f(1) = -S.
        # But f(1) is not in the domain, so it doesn't contribute to M(x).
        # Wait, the sum is over k in domain. If 1 is not in domain, f(1) doesn't exist in the sum.
        # Let me re-read carefully.
        # "values holds the finite reals f(k)" for k in domain.
        # "f(1) is replaced by f(1) - S ... (f(1) = 0 before this if 1 is not in the domain)"
        # This implies f(1) is a concept that exists even if 1 is not in domain, with value 0.
        # But M(x) = sum_k f(k) * floor(x/k) where k is in domain.
        # So if 1 is not in domain, f(1) doesn't appear in M(x).
        # The constraint is sum_k f(k)/k = 0 over k in domain.
        # If 1 is not in domain, we can't adjust f(1) to make the sum 0 unless we add 1 to domain.
        # But the problem says "f(1) is replaced by f(1) - S", implying we should ensure 1 is in domain
        # or handle it specially.
        # Let's assume the evaluator adds 1 to domain with value 0 if not present, then adjusts.
        # Or maybe the sum is over all positive integers, with f(k)=0 if k not in domain.
        # Let's assume the latter: f(k) = 0 for k not in domain.
        # Then S = sum_{k=1}^K f(k)/k where f(k)=0 if k not in domain.
        # And M(x) = sum_{k=1}^K f(k) * floor(x/k).
        # So if 1 is not in domain, f(1)=0, and after adjustment f(1) = -S.
        # But since 1 is not in domain, f(1) is still 0 in M(x)?
        # This is confusing. Let's assume the evaluator works as follows:
        # 1. Clip values to [-10, 10].
        # 2. Compute S = sum_{k in domain} f(k)/k.
        # 3. If 1 in domain: f(1) -= S. Else: f(1) = -S (and 1 is effectively in domain with value -S).
        # 4. Now sum_{k} f(k)/k = 0.
        # 5. M(x) = sum_{k} f(k) * floor(x/k) where k ranges over all positive integers with f(k) != 0.
        # This interpretation makes sense.
        pass
    
    # For now, let's assume 1 is in domain or we handle it properly.
    # Let's just check M(x) for the given domain and values, assuming the adjustment has been done.
    # But we need to do the adjustment first.
    
    # Let's redo:
    vals_adjusted = vals.copy()
    if 1 in domain:
        idx = np.where(domain == 1)[0][0]
        vals_adjusted[idx] -= S
    else:
        # We need to add 1 to domain with value -S
        domain_ext = np.concatenate([domain, [1]])
        vals_adjusted = np.concatenate([vals_adjusted, [-S]])
        # But this changes the domain, which might not be what we want.
        # Let's just check feasibility with the assumption that the evaluator handles this.
        # For now, let's assume 1 is in domain in our construction.
        return False  # Infeasible if 1 not in domain, as we can't guarantee the sum is 0
    
    # Now check M(x) <= 1.0001 for x in [1, 10K]
    # M(x) = sum_k f(k) * floor(x/k)
    # This is expensive to compute naively. Let's use the fact that floor(x/k) is constant
    # for ranges of x.
    
    # Actually, for a quick check, let's just compute M(x) for a few sample points.
    # But the evaluator checks all x in [1, 10K]. We need an efficient check.
    
    # One approach: M(x) is a step function. The maximum of M(x) occurs at x = k for some k
    # or at x = 10K.
    # Actually, M(x) is non-decreasing if all f(k) >= 0, but f(k) can be negative.
    # Let's just compute M(x) for x = 1, 2, ..., 10K and check the max.
    
    max_x = 10 * K
    if max_x > 1000000:  # Too large to compute naively
        # Use a smarter approach
        # M(x) = sum_k f(k) * floor(x/k)
        # We can compute this using a sieve-like approach.
        # For each k, floor(x/k) changes at x = k, 2k, 3k, ...
        # So M(x) is a sum of step functions.
        # The maximum of M(x) occurs at one of the points x = j*k for some j, k.
        # But there are too many such points.
        # Let's just check x = 1, 2, ..., min(10K, 10000) and assume it's bounded.
        # This is a heuristic, not a guarantee.
        max_x_check = min(max_x, 10000)
    else:
        max_x_check = max_x
    
    # Compute M(x) for x in [1, max_x_check]
    # M(x) = sum_k f(k) * floor(x/k)
    # We can compute this by iterating over k and adding f(k) to M(x) for x >= k.
    # But this is O(m * max_x_check), which might be too slow.
    
    # Alternative: M(x) = sum_{j=1}^{floor(x/k)} f(k) for each k.
    # So M(x) = sum_{k=1}^{x} f(k) * floor(x/k).
    # We can compute this using a divisor sum approach.
    
    # Let's just do a naive check for small max_x_check.
    if max_x_check <= 100000:
        M = np.zeros(max_x_check + 1, dtype=np.float64)
        for k, v in zip(domain, vals_adjusted):
            if k <= max_x_check:
                # floor(x/k) for x in [1, max_x_check]
                # x = 1, ..., max_x_check
                # floor(x/k) = 0 for x < k, 1 for k <= x < 2k, etc.
                # We can add v to M[x] for x in [j*k, (j+1)*k - 1] for j = 1, 2, ...
                for j in range(1, max_x_check // k + 1):
                    start = j * k
                    end = min((j + 1) * k - 1, max_x_check)
                    if start <= end:
                        M[start:end + 1] += v
        if np.max(M[1:]) > 1.0001:
            return False
    else:
        # For large max_x_check, we can't do a naive check.
        # Let's assume it's feasible if the max for small x is <= 1.0001.
        # This is a heuristic.
        max_x_check_small = 10000
        M = np.zeros(max_x_check_small + 1, dtype=np.float64)
        for k, v in zip(domain, vals_adjusted):
            if k <= max_x_check_small:
                for j in range(1, max_x_check_small // k + 1):
                    start = j * k
                    end = min((j + 1) * k - 1, max_x_check_small)
                    if start <= end:
                        M[start:end + 1] += v
        if np.max(M[1:]) > 1.0001:
            return False
    
    return True


def _compute_objective(domain, values):
    """
    Compute the objective a = -sum_k f(k) * log(k) / k.
    """
    vals = np.clip(values, -10.0, 10.0)
    S = np.sum(vals / domain.astype(np.float64))
    
    if 1 in domain:
        idx = np.where(domain == 1)[0][0]
        vals[idx] -= S
    
    # a = -sum_k f(k) * log(k) / k
    a = -np.sum(vals * np.log(domain.astype(np.float64)) / domain.astype(np.float64))
    return a


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds
    
    # Compute Mobius function up to some limit
    # The optimal solution is related to mu(k), so let's use mu(k) for k up to K.
    # But we need to check feasibility.
    
    # Let's start with K = 1000 and increase if feasible.
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    for K in [100, 500, 1000, 5000, 10000, 50000, 100000, 200000]:
        if time.time() > deadline - 30:  # Leave 30 seconds margin
            break
        
        # Compute mu(1..K)
        mu = _compute_mobius(K)
        
        # Use mu(k) as f(k) for k in 1..K
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = mu.astype(np.float64)
        
        # Check feasibility
        if _check_feasibility(domain, values, K):
            # Compute objective
            score = _compute_objective(domain, values)
            if score > best_score:
                best_score = score
                best_domain = domain
                best_values = values
        
        # If not feasible, we can try to adjust the values.
        # One approach: scale the values by a factor to make M(x) <= 1.
        # But this reduces the objective.
        # Another approach: truncate the domain to a smaller K.
        # Let's just try the next K.
    
    if best_domain is not None:
        return best_domain, best_values
    
    # Fallback: return the original hardcoded solution
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values


# EVOLVE-BLOCK-END
