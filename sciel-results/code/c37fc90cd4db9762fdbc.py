# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_mobius(K):
    """Compute the Mobius function mu(n) for n=1..K using a sieve."""
    mu = np.ones(K + 1, dtype=np.int8)
    primes = []
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            if i2 <= K:
                for j in range(i2, K + 1, i2):
                    mu[j] = 0
                    
    return mu

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
    deadline = start_time + 600
    
    # We want to use the Mobius function for k up to some K_max.
    # The constraint M(x) <= 1.0001 is crucial.
    # M(x) = sum_{k<=K} mu(k) * floor(x/k).
    # We know sum_{k<=x} mu(k) floor(x/k) = 1 for all integer x.
    # However, we are summing only up to K_max. Let x be up to 10*K_max.
    # If x > K_max, the sum is partial.
    # Actually, the identity sum_{k|x} mu(k) = 1 if x=1 else 0 is not what we use.
    # We use sum_{k<=N} mu(k) floor(N/k) = 1 for any N.
    # Wait, the identity is sum_{d|n} mu(d) = 1 if n=1 else 0.
    # Summing over n: sum_{n<=x} sum_{d|n} mu(d) = 1.
    # Swap sums: sum_{d<=x} mu(d) sum_{n<=x, d|n} 1 = sum_{d<=x} mu(d) floor(x/d) = 1.
    # So if we take domain = [1..K_max] and values = mu(k), then for any x,
    # M(x) = sum_{k<=K_max} mu(k) floor(x/k).
    # If x <= K_max, M(x) = 1.
    # If x > K_max, M(x) = sum_{k<=K_max} mu(k) floor(x/k).
    # This is NOT necessarily 1. It is the sum over k|n for n<=x, but restricted to k<=K_max.
    # The full sum is 1. The missing terms are k > K_max.
    # M(x) = 1 - sum_{k>K_max} mu(k) floor(x/k).
    # We need M(x) <= 1.0001.
    # So we need 1 - sum_{k>K_max} mu(k) floor(x/k) <= 1.0001
    # => - sum_{k>K_max} mu(k) floor(x/k) <= 0.0001
    # => sum_{k>K_max} mu(k) floor(x/k) >= -0.0001.
    
    # Let's test this for various K_max.
    # We can't test for all x up to 10*K_max easily in the function without taking too long,
    # but we can pick a K_max that is known to work or verify it quickly.
    
    # The error term is related to the Mertens function M(K_max).
    # Actually, let's just compute it for a candidate K_max and check the max violation.
    
    # The objective is a = -sum_{k=1}^{K_max} mu(k) log(k) / k.
    # This converges to 1.
    
    # Let's try K_max = 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000.
    # We need to check feasibility: max_{x in 1..10*K_max} M(x) <= 1.0001.
    
    # To speed up, we can just check a few x values or use the property.
    # M(x) is a step function. It changes value only when x is a multiple of some k.
    # But checking all x is O(K_max * 10*K_max) = O(10 K_max^2).
    # For K_max = 10000, that's 10^9 operations, which might be too slow in Python if done naively.
    # But we can vectorize or just pick a smaller K_max if it's close to 1.
    
    # Let's try K_max = 2000 first.
    # If it's feasible and gives a good score, we use it.
    
    # We will implement a helper to check feasibility and compute score for a given K_max.
    
    def check_and_score(K_max):
        mu = _compute_mobius(K_max)
        # domain is 1..K_max where mu(k) != 0? Or all 1..K_max?
        # The task says "distinct positive integers".
        # If we include mu(k)=0, they don't contribute to M(x) or a.
        # But they increase m. Does m matter? "m >= 1". No penalty for large m explicitly, 
        # except "cost about m * 10K" for evaluation. 
        # Wait, the evaluator computes M(x) for every x in [1, 10K].
        # If m is large, the evaluator is slower? "cost about m * 10K".
        # If K_max = 10000, m=10000, cost = 10000 * 100000 = 10^9. 
        # This might timeout the EVALUATOR if it's not optimized.
        # The prompt says "Each evaluation must finish within 600 s".
        # The evaluator is fixed. We don't control its speed.
        # If the evaluator is O(m * 10K), and K is max key, then for K=10000, 10K=100000.
        # 10000 * 100000 = 10^9 operations. In C/NumPy this is fast. In Python loops it's slow.
        # The evaluator is likely implemented in C/NumPy or optimized.
        # Let's assume the evaluator can handle K up to 200000 reasonably.
        
        # Let's filter to non-zero mu to reduce m?
        # "domain holds distinct positive integers".
        # If we omit zero terms, m is smaller.
        # Let's try omitting zeros.
        
        mask = mu[1:] != 0
        domain = np.where(mask)[0] + 1
        values = mu[domain].astype(np.float64)
        
        # Check feasibility
        # M(x) = sum_{k in domain} values[k-1] * floor(x/k)
        # We need max M(x) <= 1.0001 for x in 1..10*K_max.
        
        # We can compute M(x) for all x using convolution or loops.
        # Vectorized approach:
        # M = np.zeros(10*K_max + 1)
        # for k, v in zip(domain, values):
        #     M[k::k] += v
        # This is O(10K_max * sum(1/k)) ~ O(10K_max log K_max).
        # For K_max = 10000, 100000 * 10 ~ 10^6. Very fast.
        
        max_x = 10 * K_max
        M = np.zeros(max_x + 1, dtype=np.float64)
        # Iterate over non-zero terms
        for i in range(len(domain)):
            k = domain[i]
            v = values[i]
            if v != 0:
                M[k::k] += v
                
        max_M = np.max(M[1:]) # x starts from 1
        if max_M > 1.0001:
            return False, 0.0
            
        # Compute score
        # a = -sum f(k) log(k) / k
        # Note: f(1) is adjusted in evaluator.
        # Evaluator: f(1) <- f(1) - S, where S = sum f(k)/k.
        # Then a = -sum f_new(k) log(k)/k.
        # f_new(1) = f(1) - S.
        # a = - (f(1) - S) log(1)/1 - sum_{k>1} f(k) log(k)/k
        # log(1) = 0, so the f(1) term vanishes in the objective!
        # a = - sum_{k>1} f(k) log(k)/k.
        # So the adjustment of f(1) does not change the objective a, 
        # provided the constraint sum f(k)/k = 0 is satisfied by the adjustment.
        # The adjustment always makes sum f(k)/k = 0.
        # So we just need to compute - sum_{k>1} mu(k) log(k)/k.
        
        # However, we must ensure that the adjusted f(1) doesn't violate M(x) <= 1.0001?
        # M(x) depends on f(1).
        # M(x) = f(1) floor(x/1) + sum_{k>1} f(k) floor(x/k) = f(1) x + ...
        # Wait, floor(x/1) = x.
        # So M(x) is linear in f(1) with slope x.
        # The evaluator adjusts f(1) AFTER we return it?
        # "Then f(1) is replaced by f(1) - S ... so that afterwards sum f(k)/k = 0."
        # "M(x) ... is computed ... if any M(x) > 1.0001 ... infeasible".
        # So the feasibility check uses the ADJUSTED f(1).
        
        # Let's re-evaluate feasibility with the adjusted f(1).
        # S = sum_{k in domain} values[k-1] / k.
        # Note: values[0] corresponds to k=1.
        # If 1 is in domain:
        #   f(1) = values[0].
        #   S = f(1)/1 + sum_{k>1} f(k)/k.
        #   f_new(1) = f(1) - S = - sum_{k>1} f(k)/k.
        # If 1 is NOT in domain:
        #   f(1) = 0 (implicitly).
        #   S = sum_{k>1} f(k)/k.
        #   f_new(1) = -S = - sum_{k>1} f(k)/k.
        # In both cases, f_new(1) = - sum_{k>1} f(k)/k.
        
        # Let's compute f_new(1).
        sum_inv = np.sum(values[domain > 1] / domain[domain > 1])
        f1_new = -sum_inv
        
        # Now recompute M(x) with f1_new.
        # M(x) = f1_new * x + sum_{k>1} f(k) floor(x/k).
        
        # We already computed M_partial(x) = sum_{k>1} f(k) floor(x/k) above?
        # No, we included k=1 in the loop if it was in domain.
        # Let's recompute M for k > 1 only, then add f1_new * x.
        
        M_gt1 = np.zeros(max_x + 1, dtype=np.float64)
        mask_gt1 = domain > 1
        for i in np.where(mask_gt1)[0]:
            k = domain[i]
            v = values[i]
            if v != 0:
                M_gt1[k::k] += v
                
        # M(x) = f1_new * x + M_gt1(x)
        x_range = np.arange(1, max_x + 1)
        M_final = f1_new * x_range + M_gt1[1:]
        
        max_M_final = np.max(M_final)
        if max_M_final > 1.0001:
            return False, 0.0
            
        # Compute a
        # a = - sum_{k>1} f(k) log(k)/k
        # (f(1) term is 0)
        a = -np.sum(values[mask_gt1] * np.log(domain[mask_gt1]) / domain[mask_gt1])
        
        return True, a

    # Try increasing K_max
    best_score = 0.0
    best_domain = None
    best_values = None
    
    # Start with a small K to ensure we have a valid fallback
    K_candidates = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
    
    for K in K_candidates:
        if time.time() > deadline - 60: # Leave 60s margin
            break
            
        feasible, score = check_and_score(K)
        if feasible and score > best_score:
            best_score = score
            mu = _compute_mobius(K)
            mask = mu[1:] != 0
            best_domain = np.where(mask)[0] + 1
            best_values = mu[best_domain].astype(np.float64)
            
    if best_domain is None:
        # Fallback to the original hard-coded solution if something went wrong
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
        
    return best_domain, best_values

# EVOLVE-BLOCK-END
