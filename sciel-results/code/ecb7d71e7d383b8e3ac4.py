# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def compute_mobius(n):
    """Compute Möbius function mu(k) for k = 0..n using linear sieve."""
    mu = np.zeros(n + 1, dtype=np.int64)
    mu[1] = 1
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
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

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    
    Strategy:
    1. Start with the Möbius function for a large K, as it naturally satisfies M(x) ~ 1.
    2. Use Greedy Coordinate Descent to optimize the objective a = -sum f(k)log(k)/k
       subject to M(x) <= 1 for all x in [1, 10K] and sum f(k)/k = 0.
    3. The evaluator adjusts f(1) to satisfy the sum constraint, so we focus on
       optimizing f(k) for k >= 2 and the M(x) constraints.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    # We need to choose K carefully.
    # Larger K gives more terms in the sum, potentially higher score.
    # But M(x) computation is O(K * 10K) = O(K^2) per full check, or O(K^2) total if vectorized.
    # Greedy steps require efficient updates.
    
    # Let's try a few K values.
    # The inspirations suggest LP works well up to K~200-500.
    # Greedy might handle larger K if we are smart about updates.
    
    # However, the constraint M(x) <= 1 involves floor(x/k).
    # M(x) = sum_{k=1}^K f(k) floor(x/k).
    # For a fixed k, changing f(k) by delta changes M(x) by delta * floor(x/k) for all x.
    # We need max_x (M(x) + delta * floor(x/k)) <= 1.
    
    # This is a large-scale optimization.
    # Let's stick to a moderate K where we can afford multiple iterations.
    # K = 1000 seems reasonable. 10K = 10000.
    
    K = 1000
    max_x = 10 * K
    
    # Initialize with Möbius
    mu = compute_mobius(K)
    f = np.zeros(K + 1, dtype=np.float64)
    # f[1] is handled by evaluator, but we can set it to 0 initially for our internal calc
    # Actually, the evaluator sets f(1) AFTER we return.
    # The constraint sum f(k)/k = 0 is enforced by evaluator.
    # So we only need to ensure M(x) <= 1.
    # Wait, the metric a = -sum f(k) log(k) / k.
    # Does f(1) contribute? log(1)=0, so f(1) doesn't affect 'a'.
    # But f(1) affects M(x) because floor(x/1) = x.
    # M(x) = f(1)*x + sum_{k>=2} f(k) floor(x/k).
    # The evaluator replaces f(1) with f(1) - S where S = sum f(k)/k.
    # So the actual f(1) used in M(x) is -S.
    # Let's denote the returned values as g(k).
    # g(1) is arbitrary in our output, but effectively becomes -sum_{k>=2} g(k)/k.
    # Let h(k) = g(k) for k >= 2.
    # Effective f(1) = -sum_{k>=2} h(k)/k.
    # M(x) = (-sum_{k>=2} h(k)/k) * x + sum_{k>=2} h(k) floor(x/k).
    # M(x) = sum_{k>=2} h(k) (floor(x/k) - x/k).
    # Note that floor(x/k) - x/k = - {x/k}, the negative fractional part.
    # So M(x) = - sum_{k>=2} h(k) {x/k}.
    # We need M(x) <= 1.
    # Since {x/k} >= 0, if we choose h(k) <= 0, then M(x) >= 0.
    # Wait, if h(k) is negative, -h(k) is positive, so M(x) is positive.
    # We want M(x) <= 1.
    
    # Let's re-read the contract carefully.
    # "values are clipped to [-10, 10]. Then f(1) is replaced by f(1) - S ... so that afterwards sum_k f(k)/k = 0."
    # "M(x) = sum_k f(k) * floor(x/k) is computed for every integer x in [1, 10K]".
    # "if any M(x) > 1.0001 the function is infeasible".
    
    # So the final f(1) is determined by the other values.
    # Let's define variables y_k = f(k) for k >= 2.
    # Final f(1) = - sum_{k=2}^K y_k / k.
    # M(x) = f(1)*x + sum_{k=2}^K y_k floor(x/k)
    # M(x) = -x * sum_{k=2}^K y_k/k + sum_{k=2}^K y_k floor(x/k)
    # M(x) = sum_{k=2}^K y_k (floor(x/k) - x/k)
    # M(x) = - sum_{k=2}^K y_k {x/k}
    
    # Objective: a = - sum_{k=1}^K f(k) log(k) / k.
    # f(1) term is 0.
    # a = - sum_{k=2}^K y_k log(k) / k.
    
    # We want to maximize a.
    # This is equivalent to minimizing sum_{k=2}^K y_k log(k) / k.
    # Since log(k)/k > 0 for k >= 2, we want y_k to be as negative as possible?
    # If y_k is negative, the term y_k log(k)/k is negative, so - (negative) is positive.
    # So we want large negative y_k.
    
    # Constraints:
    # 1. M(x) <= 1 for all x in [1, 10K].
    #    M(x) = - sum_{k=2}^K y_k {x/k}.
    #    If y_k < 0, let z_k = -y_k > 0.
    #    M(x) = sum_{k=2}^K z_k {x/k}.
    #    We need sum_{k=2}^K z_k {x/k} <= 1 for all x.
    # 2. Clipping: y_k in [-10, 10] => z_k in [-10, 10]. Since we expect z_k > 0, z_k in [0, 10].
    
    # So the problem is:
    # Maximize sum_{k=2}^K z_k log(k)/k
    # Subject to:
    #   sum_{k=2}^K z_k {x/k} <= 1 for all x in [1, 10K]
    #   0 <= z_k <= 10
    
    # This is a Linear Program!
    # Variables: z_k for k=2..K.
    # Objective: Max c^T z where c_k = log(k)/k.
    # Constraints: A z <= 1, where A_{x,k} = {x/k}.
    #             z >= 0, z <= 10.
    
    # This LP formulation is much cleaner.
    # The number of constraints is 10K. The number of variables is K-1.
    # For K=1000, we have 10000 constraints and 999 variables.
    # This is small enough for scipy.optimize.linprog.
    
    # Let's solve this LP.
    
    # Construct A matrix.
    # A is (10K) x (K-1).
    # Rows correspond to x = 1..10K.
    # Columns correspond to k = 2..K.
    # A[x-1, k-2] = (x % k) / k.
    
    # This matrix is dense. 10000 * 1000 = 10^7 floats. 80 MB. Acceptable.
    
    # Let's try K=2000. 20000 * 2000 = 4*10^7 floats. 320 MB. Might be tight but ok.
    # Let's stick to K=1000 first to ensure speed.
    
    # Actually, we can be smarter.
    # {x/k} depends only on x mod k.
    # But constructing the full matrix is straightforward.
    
    # Let's use scipy.optimize.linprog.
    # Method: 'highs' is default and fast.
    
    try:
        from scipy.optimize import linprog
    except ImportError:
        # Fallback to heuristic if scipy not available (unlikely)
        mu = compute_mobius(K)
        domain = np.nonzero(mu[1:K+1] != 0)[0] + 1
        values = mu[1:K+1][mu[1:K+1] != 0].astype(np.float64)
        return domain, values

    # Define K
    # We can try to increase K if time permits.
    # Let's start with K=500 to be safe, then try 1000.
    
    best_score = -np.inf
    best_domain = None
    best_values = None
    
    for K_trial in [500, 1000, 2000]:
        if time.time() > deadline - 30:
            break
            
        max_x = 10 * K_trial
        n_vars = K_trial - 1
        
        # Construct c (objective coefficients)
        # c_k = log(k)/k for k=2..K
        ks = np.arange(2, K_trial + 1, dtype=np.float64)
        c = np.log(ks) / ks
        # We want to maximize c^T z. linprog minimizes. So minimize -c^T z.
        c_min = -c
        
        # Construct A_ub (constraint matrix)
        # A_ub[i, j] = { (i+1) / (j+2) }
        # i in 0..10K-1, j in 0..K-2
        # Vectorized construction:
        # x_vals = np.arange(1, max_x + 1)
        # k_vals = np.arange(2, K_trial + 1)
        # A = (x_vals[:, None] % k_vals[None, :]) / k_vals[None, :]
        
        # This creates a large array.
        # For K=2000, 20000x2000 = 40M elements.
        # Might be slow to construct.
        
        # Let's check time.
        t_start = time.time()
        
        x_vals = np.arange(1, max_x + 1, dtype=np.int32)
        k_vals = np.arange(2, K_trial + 1, dtype=np.int32)
        
        # Use broadcasting
        # Note: modulo operation on large arrays can be slow.
        # A = (x_vals[:, None] % k_vals[None, :]).astype(np.float64) / k_vals[None, :].astype(np.float64)
        
        # Alternative:
        # We can compute this in chunks if memory is an issue, but 320MB is usually fine.
        
        A_ub = (x_vals[:, None] % k_vals[None, :]).astype(np.float64)
        A_ub /= k_vals[None, :].astype(np.float64)
        
        b_ub = np.ones(max_x, dtype=np.float64)
        
        # Bounds: 0 <= z_k <= 10
        bounds = [(0, 10)] * n_vars
        
        # Solve LP
        res = linprog(c_min, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        t_end = time.time()
        
        if res.success:
            z_opt = res.x
            # z_opt corresponds to k=2..K
            # y_k = -z_opt
            # f(k) = y_k for k>=2
            # f(1) is determined by evaluator, but we can set it to 0 in our output
            # because the evaluator will adjust it.
            # Wait, the output must be domain and values.
            # domain should include 1?
            # "domain holds distinct positive integers".
            # If 1 is not in domain, f(1)=0 before adjustment.
            # "f(1) = 0 before this if 1 is not in the domain".
            # So we can just exclude 1 from the domain.
            
            domain = np.arange(2, K_trial + 1, dtype=np.int64)
            values = -z_opt
            
            # Calculate the score 'a'
            # a = - sum f(k) log(k) / k
            # f(k) = -z_k
            # a = - sum (-z_k) log(k) / k = sum z_k log(k) / k
            # This is exactly -res.fun (since res.fun is min of -c^T z = -max c^T z)
            # Actually res.fun is the value of the objective function at the optimum.
            # Objective was min -c^T z. So res.fun = - (max c^T z).
            # So max c^T z = -res.fun.
            # And a = max c^T z.
            
            a = -res.fun
            
            if a > best_score:
                best_score = a
                best_domain = domain
                best_values = values
                
            # If we got a good score, we might stop
            if best_score > 1.5: # Just a guess, scale is unknown
                 # Let's check what a typical score is.
                 # For Möbius, a is small.
                 # LP should be better.
                 pass
        
        else:
            print(f"LP failed for K={K_trial}: {res.message}")
            
    # Fallback
    if best_domain is None:
        # Use Möbius
        K_fb = 100
        mu = compute_mobius(K_fb)
        mask = mu[1:K_fb + 1] != 0
        domain = np.nonzero(mask)[0] + 1
        values = mu[1:K_fb + 1][mask].astype(np.float64)
        return domain, values
        
    return best_domain, best_values

# EVOLVE-BLOCK-END
