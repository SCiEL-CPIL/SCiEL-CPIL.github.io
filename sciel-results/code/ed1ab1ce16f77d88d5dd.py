# EVOLVE-BLOCK-START
import numpy as np
from sympy import mobius
import time


def _compute_mobius(K):
    """Compute Mobius function for 1..K using a simple sieve."""
    mu = np.ones(K + 1, dtype=np.int64)
    mu[0] = 0
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
    # Handle squares
    for p in primes:
        p2 = p * p
        if p2 > K:
            break
        for j in range(p2, K + 1, p2):
            mu[j] = 0
    return mu


def _evaluate_feasibility_and_score(domain, values, K_max):
    """
    Apply clipping, S-adjustment, check M(x) constraint, compute score.
    Returns (feasible, score).
    """
    # Clip to [-10, 10]
    vals = np.clip(values, -10.0, 10.0)
    
    # Compute S = sum_k f(k)/k
    S = np.sum(vals / domain.astype(np.float64))
    
    # Adjust f(1): if 1 is in domain, f(1) -= S; else f(1) = -S (but 1 not in domain means we add it? No, the task says f(1)=0 before this if 1 not in domain)
    # Actually, re-reading: "f(1) is replaced by f(1) - S with S = sum_k f(k)/k over the clipped values (f(1) = 0 before this if 1 is not in the domain)"
    # So if 1 is not in domain, we treat f(1)=0 for the purpose of computing S? No, S is computed over the domain.
    # Let me re-read: "f(1) is replaced by f(1) - S with S = sum_k f(k)/k over the clipped values (f(1) = 0 before this if 1 is not in the domain)"
    # I think this means: when computing S, if 1 is not in the domain, we still include the term f(1)/1 = 0/1 = 0, so S is just sum over domain of f(k)/k.
    # Then f(1) (which may or may not be in domain) is set to f(1) - S. If 1 was not in domain, f(1) was 0, so it becomes -S, and 1 is now in the domain?
    # Hmm, this is ambiguous. Let me assume that if 1 is not in the original domain, we add it with value -S.
    
    if 1 in domain:
        idx1 = np.where(domain == 1)[0][0]
        vals[idx1] -= S
    else:
        # Add 1 to domain with value -S
        domain = np.insert(domain, 0, 1)
        vals = np.insert(vals, 0, -S)
    
    # Now sum_k f(k)/k should be 0
    # Check M(x) for x in [1, 10*K_max]
    K = domain.max()
    x_max = 10 * K
    # M(x) = sum_k f(k) * floor(x/k)
    # For efficiency, compute for all x in [1, x_max]
    # M(x) = sum_{k in domain} f(k) * floor(x/k)
    # We can compute this as: for each k, f(k) contributes f(k) to M(x) for x >= k, in steps
    # More efficiently: M(x) = sum_k f(k) * floor(x/k)
    # Let's compute directly for x in [1, x_max]
    
    # Use vectorized approach
    # For each x, M(x) = sum_k f(k) * floor(x/k)
    # This is expensive if done naively. Let's use the fact that floor(x/k) changes only O(sqrt(x)) times.
    # But x_max can be up to 10*200000 = 2000000, and m can be up to K.
    # Let's just compute it directly if m and x_max are reasonable.
    
    # Actually, let's compute M(x) for all x in [1, x_max] using a different approach:
    # M(x) = sum_{d=1}^{x} (sum_{k|d, k in domain} f(k)) 
    # No wait, M(x) = sum_k f(k) floor(x/k) = sum_{n=1}^{x} sum_{k|n} f(k) = sum_{n=1}^{x} g(n) where g(n) = sum_{k|n, k in domain} f(k)
    
    # Let's compute g(n) for n in [1, x_max]
    # g(n) = sum_{k in domain, k|n} f(k)
    
    x_max = min(10 * K, 2000000)  # Cap for safety
    
    g = np.zeros(x_max + 1, dtype=np.float64)
    for i, k in enumerate(domain):
        if k > x_max:
            continue
        g[k::k] += vals[i]
    
    # M(x) = sum_{n=1}^{x} g(n) = cumsum(g)
    M = np.cumsum(g[1:])  # M(x) for x=1..x_max
    
    if np.max(M) > 1.0001:
        return False, -np.inf
    
    # Compute score a = -sum_k f(k) * log(k) / k
    score = -np.sum(vals * np.log(domain.astype(np.float64)) / domain.astype(np.float64))
    return True, score


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    The function f is scored by a fixed evaluator (see evaluator.py) that:
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
    deadline = start_time + 600  # 600 second deadline
    
    best_score = -np.inf
    best_domain = None
    best_values = None
    
    # Try increasing K, using Mobius function values
    for K in range(1, 200001):
        if time.time() > deadline - 30:  # Safety margin
            break
        
        mu = _compute_mobius(K)
        
        # Use mu(k) for k=1..K, but skip k where mu(k)=0 to keep domain small?
        # Actually, including k with mu(k)=0 doesn't change anything, but increases m.
        # Let's include all k=1..K for now, but only non-zero mu values to save memory.
        non_zero = np.where(mu[1:] != 0)[0] + 1
        if len(non_zero) == 0:
            continue
        
        domain = non_zero.astype(np.int64)
        values = mu[1:][mu[1:] != 0].astype(np.float64)
        
        feasible, score = _evaluate_feasibility_and_score(domain, values, K)
        
        if feasible and score > best_score:
            best_score = score
            best_domain = domain.copy()
            best_values = values.copy()
        
        # If score is already close to 1, we can stop
        if best_score > 0.999:
            break
    
    if best_domain is not None:
        return best_domain, best_values
    
    # Fallback to seed
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values


# EVOLVE-BLOCK-END
