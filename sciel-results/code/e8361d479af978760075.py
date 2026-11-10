# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def _compute_mobius(n):
    """Compute Mobius function mu(k) for k = 1..n using sieve."""
    mu = np.ones(n + 1, dtype=np.int8)
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
    mu[1] = 1
    return mu

def _evaluate_lp(K, time_limit=500):
    """
    Solve the LP problem for the optimal partial function f(k) for k in 1..K.
    Returns (domain, values) or None if infeasible or timeout.
    """
    start_time = time.time()
    
    # Variables: y_k = f(k) for k=1..K
    n_vars = K
    
    # Objective: maximize a = -sum_k f(k) * log(k) / k
    # linprog minimizes, so we minimize c = sum_k f(k) * log(k) / k
    k_range = np.arange(1, K + 1)
    c = np.log(k_range) / k_range
    
    # Constraints:
    # 1. M(x) <= 1.0001 for all x in [1, 10K]
    # M(x) = sum_k f(k) * floor(x/k)
    # Note: The evaluator adjusts f(1) by -S where S = sum f(k)/k.
    # Let f(k) = y_k for k>1, and f(1) = y_1 - S.
    # Actually, it's easier to work with the adjusted values directly or substitute.
    # Let's define variables y_k as the values BEFORE adjustment? No, the evaluator clips then adjusts.
    # Since we expect optimal values to be within [-10, 10], we can assume clipping doesn't bind initially.
    # Let's define variables v_k as the final values AFTER adjustment?
    # If we define v_k as final values, then sum v_k/k = 0 is a constraint.
    # And M(x) = sum v_k * floor(x/k) <= 1.0001.
    # Objective a = -sum v_k log(k)/k.
    # This is a standard LP.
    
    # Constraints A_ub @ v <= b_ub
    # Rows for M(x) <= 1.0001 for x = 1 .. 10K
    max_x = 10 * K
    # Matrix A is (10K) x K. A[x, k] = floor(x/k)
    # This matrix is huge if K=200000 (2M rows, 200k cols). 
    # However, we can exploit structure or just use a smaller K for LP.
    # Let's try K up to 5000 for LP, which is 50k rows x 5k cols. Manageable.
    
    # We also need sum v_k / k = 0. This is an equality constraint.
    # A_eq @ v = b_eq
    A_eq = (1.0 / k_range).reshape(1, -1)
    b_eq = np.array([0.0])
    
    # Bounds: -10 <= v_k <= 10
    bounds = [(-10, 10)] * n_vars
    
    # Construct A_ub
    # To save memory and time, we can sample x or use all x if K is small enough.
    # For K=2000, max_x=20000. A is 20000x2000. 40M floats ~ 320MB. OK.
    # For K=5000, max_x=50000. A is 50000x5000. 250M floats ~ 2GB. Might be too big.
    # Let's cap K for LP at 3000 or 4000.
    
    if K > 4000:
        return None

    A_ub = np.zeros((max_x, K))
    for x in range(1, max_x + 1):
        # Vectorized floor division
        A_ub[x-1, :] = x // k_range
        
    b_ub = np.ones(max_x) * 1.0001
    
    # Solve LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    except Exception:
        return None
        
    if not res.success:
        return None
        
    v = res.x
    
    # Verify feasibility and compute score
    # Check M(x)
    M_vals = A_ub @ v
    if np.max(M_vals) > 1.0001:
        return None
        
    # Check sum v/k
    if abs(np.sum(v / k_range)) > 1e-9:
        return None

    # Construct domain and values
    # The problem asks for a partial function. We can include all k where v_k is not 0.
    # However, including zeros doesn't hurt score but increases m. 
    # Let's include all k=1..K for simplicity, or just non-zeros.
    # The evaluator handles any subset.
    # To be safe and minimal, let's include all k.
    
    domain = k_range.astype(np.int64)
    values = v.astype(np.float64)
    
    # Calculate actual score to compare
    score = -np.sum(values * np.log(k_range) / k_range)
    
    # Check time
    if time.time() - start_time > time_limit:
        return None
        
    return domain, values, score

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 600s limit
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # Strategy:
    # 1. Try LP for small K (up to 4000) to find the true optimum for that K.
    # 2. If LP fails or times out, fallback to Mobius for larger K.
    # 3. Compare scores and return the best.
    
    # First, try Mobius for a range of K to establish a baseline quickly
    # Mobius is fast.
    mu_Ks = [100, 1000, 5000, 10000, 50000, 100000, 200000]
    for K in mu_Ks:
        if time.time() > deadline - 120: # Leave time for LP
            break
        mu = _compute_mobius(K)
        mask = mu[1:] != 0
        domain = np.where(mask)[0] + 1
        values = mu[1:][mask].astype(np.float64)
        
        # Quick evaluation of Mobius score (approx)
        # We know score approaches 1.
        # Let's just store it and evaluate properly if it beats LP
        # For now, we assume Mobius score is < 1.0 for finite K.
        # But let's keep the best Mobius result as fallback.
        
        # We need to check feasibility for Mobius too? 
        # M(x) for Mobius is 1 for x>=1 (except x=0). So M(x) <= 1.0001 is always true.
        # So Mobius is always feasible.
        # Score for Mobius: -sum mu(k) log(k)/k.
        k_float = domain.astype(np.float64)
        score_mu = -np.sum(values * np.log(k_float) / k_float)
        
        if score_mu > best_score:
            best_score = score_mu
            best_domain = domain.copy()
            best_values = values.copy()
            
    # Now try LP for increasing K, stopping when time is up
    lp_Ks = [100, 500, 1000, 2000, 3000, 4000]
    
    for K in lp_Ks:
        if time.time() > deadline - 60:
            break
            
        result = _evaluate_lp(K, time_limit=deadline - time.time() - 10)
        if result is not None:
            domain, values, score_lp = result
            if score_lp > best_score:
                best_score = score_lp
                best_domain = domain
                best_values = values
                
    if best_domain is None:
        # Fallback
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
        
    return best_domain, best_values

# EVOLVE-BLOCK-END
