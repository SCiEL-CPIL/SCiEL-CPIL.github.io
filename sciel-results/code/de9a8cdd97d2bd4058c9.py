# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = 600  # seconds, safety margin before 630s kill

    def mobius_sieve(n):
        """Compute Mobius function mu(k) for k=1..n using a linear sieve."""
        mu = np.zeros(n + 1, dtype=np.int8)
        mu[1] = 1
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

    def compute_M(f, K, max_x):
        """
        Compute M(x) = sum_{d|x} f(d) for x in [1, max_x].
        f is an array of length K+1 (1-indexed).
        """
        M = np.zeros(max_x + 1, dtype=np.float64)
        # For each d, add f[d] to M[d], M[2d], ...
        # We only care about d where f[d] != 0
        for d in range(1, K + 1):
            if f[d] != 0.0:
                M[d::d] += f[d]
        return M

    def compute_metric(f, K):
        """Compute a = -sum_k f(k) * log(k) / k for k=1..K"""
        k = np.arange(1, K + 1, dtype=np.float64)
        # f is 1-indexed, so f[1:] corresponds to k=1..K
        return -np.sum(f[1:] * np.log(k) / k)

    def check_feasibility(M, max_x):
        return np.max(M[1:max_x+1]) <= 1.0001

    # Use maximum K
    K = 200000
    max_x = 10 * K
    
    # 1. Compute Mobius function
    mu = mobius_sieve(K)
    
    # Initialize f with mu
    f = np.zeros(K + 1, dtype=np.float64)
    f[1:] = mu[1:].astype(np.float64)
    
    # Current metric
    current_a = compute_metric(f, K)
    
    # Check initial feasibility
    M = compute_M(f, K, max_x)
    if not check_feasibility(M, max_x):
        # Should not happen for mu, but fallback to small domain if it does
        domain = np.array([1], dtype=np.int64)
        values = np.array([0.0], dtype=np.float64)
        return domain, values

    # 2. Optimization Strategy:
    # We want to maximize a = -sum f(k) log(k)/k.
    # This is equivalent to minimizing sum f(k) log(k)/k.
    # For mu(k) = -1, term is -(-1)log(k)/k = +log(k)/k. To maximize, we want f(k) to be more negative (closer to -10).
    # For mu(k) = 1, term is -(1)log(k)/k = -log(k)/k. To maximize, we want f(k) to be more positive (closer to 10).
    
    # The constraint is M(x) = sum_{d|x} f(d) <= 1.0001.
    # For mu, M(1)=1, M(x>1)=0.
    # If we decrease f(k) (for mu(k)=-1) by delta, M(x) for x|k decreases by delta.
    # This relaxes the constraint for x>1 (since M(x) was 0, it becomes -delta, which is <= 1).
    # However, M(1) = f(1). f(1) is fixed by the sum constraint?
    # No, f(1) is part of f.
    # Constraint: sum f(k)/k = 0.
    # If we change f(k) for k>1, we must adjust f(1) to maintain sum=0.
    # f(1) = - sum_{k=2}^K f(k)/k.
    # M(1) = f(1). So M(1) = - sum_{k=2}^K f(k)/k.
    # We require M(1) <= 1.0001.
    # Also M(x) <= 1.0001 for all x.
    
    # Let's try to push f(k) for k in [K/2, K] towards the bounds.
    # These k have few multiples (only themselves and maybe 2k if 2k<=10K, which is true).
    # Changing f(k) affects M(k), M(2k), ..., M(10k).
    
    # Greedy approach:
    # Identify candidates k where mu(k) != 0.
    # Sort by potential gain in objective per unit change?
    # Gain for decreasing f(k) (mu=-1) by 1: +log(k)/k.
    # Gain for increasing f(k) (mu=1) by 1: +log(k)/k.
    
    # We will iterate a few times.
    # For each k, calculate how much we can push f(k) towards its bound (-10 or 10)
    # without violating any M(x) <= 1.0001 or the sum constraint (handled by f(1)).
    
    # Note: The sum constraint is global.
    # Let's define the change vector.
    # We will update f in batches.
    
    # To keep it simple and fast, we will try to set f(k) to its bound for a subset of k,
    # specifically those with largest log(k)/k, and check if constraints hold.
    
    # Actually, a simpler heuristic:
    # The "slack" in M(x) for x>1 is 1.0 (since M(x)=0).
    # We can use this slack to push f(k) away from 0.
    
    # Let's try a local search on the top 1000 indices by log(k)/k.
    # This is a heuristic.
    
    # Better: Use the fact that M(x) for x>1 is 0.
    # If we change f(k) to f(k) + delta, M(x) for x multiple of k changes by delta.
    # We need M(x) + delta <= 1.0001 for all x multiple of k.
    # Since M(x)=0, we need delta <= 1.0001.
    # Also, we need to update f(1).
    # f(1) = -sum_{j=2}^K f(j)/j.
    # If we change f(k) by delta, f(1) changes by -delta/k.
    # M(1) = f(1).
    # We need f(1) <= 1.0001.
    
    # Let's try to push all mu(k)=-1 to -10 and mu(k)=1 to 10?
    # No, that would violate M(1).
    
    # Let's stick to the pure Mobius function for K=200000.
    # The score is extremely close to 1.
    # If the seed is 1.0, and this is 1.0, it's a tie.
    # To improve, we need the finite correction.
    
    # Let's try adjusting f(k) for k > K/2.
    # For these k, the only multiple x in [1, 10K] that is "small" is k itself?
    # No, 2k, 3k... 10k are all <= 10K.
    
    # Let's try a simple optimization:
    # Increase f(k) for mu(k)=1 towards 10.
    # Decrease f(k) for mu(k)=-1 towards -10.
    # But we must keep M(x) <= 1.
    
    # Let's try setting f(k) = 10 * mu(k) for all k?
    # M(x) = 10 * [x=1]. M(1)=10 > 1. Infeasible.
    
    # So we are limited by M(1) <= 1.
    # f(1) <= 1.
    # f(1) = 1 - sum_{k=2}^K f(k)/k.
    # So sum_{k=2}^K f(k)/k >= 0.
    
    # For mu, sum_{k=2}^K mu(k)/k is small.
    
    # Let's just return the Mobius function for K=200000.
    # It is the best known approximation.
    
    # Construct domain and values
    nonzero_idx = np.where(mu[1:] != 0)[0] + 1
    domain = nonzero_idx.astype(np.int64)
    values = mu[nonzero_idx].astype(np.float64)
    
    # Ensure we meet the interface
    if len(domain) < 1:
        domain = np.array([1], dtype=np.int64)
        values = np.array([0.0], dtype=np.float64)
        
    return domain, values

# EVOLVE-BLOCK-END
