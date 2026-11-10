# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Maximize a = -sum_k f(k)*log(k)/k subject to:
      M(x) = sum_k f(k)*floor(x/k) <= 1.0001 for all x in [1, 10K]
      sum_k f(k)/k = 0 (after f(1) adjustment)
      values clipped to [-10, 10]
    
    Strategy: Use LP over square-free numbers up to K.
    The key insight is that f(k) only matters for square-free k (mu(k) != 0),
    and the constraint M(x) <= 1 for all x in [1, 10K] can be reduced to
    checking x where the maximum occurs.
    
    For small K (up to ~200), we can solve the LP directly.
    For larger K, we use a greedy/heuristic approach.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    def elapsed():
        return time.time() - start_time
    
    # Compute Mobius function up to K_max
    K_max = 200000
    mu = np.zeros(K_max + 1, dtype=np.int64)
    mu[1] = 1
    is_composite = np.zeros(K_max + 1, dtype=bool)
    primes = []
    
    for i in range(2, K_max + 1):
        if not is_composite[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K_max:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    
    if elapsed() > deadline - 100:
        return np.array([1, 2, 3, 5, 30], dtype=np.int64), np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    
    # For LP approach, we need to work with square-free numbers only
    # Let's try different K values, starting from smaller ones where LP is feasible
    
    # The LP formulation:
    # Variables: f(k) for square-free k in [1, K]
    # Objective: maximize -sum f(k)*log(k)/k  (equivalently minimize sum f(k)*log(k)/k)
    # Constraints:
    #   M(x) = sum_k f(k)*floor(x/k) <= 1.0001 for all x in [1, 10K]
    #   sum_k f(k)/k = 0
    #   -10 <= f(k) <= 10
    
    # But M(x) for x <= K: M(x) = sum_{k=1}^{min(x,K)} f(k)*floor(x/k)
    # For x > K: M(x) = sum_{k=1}^K f(k)*floor(x/k)
    
    # The maximum of M(x) over x in [1, 10K] typically occurs at specific points.
    # Key observation: M(x) = sum_{k|j, j<=x} f(k) for each j, so
    # M(x) = sum_{j=1}^x g(j) where g(j) = sum_{d|j} f(d)
    # Wait, that's not right. Let me reconsider.
    # M(x) = sum_{k=1}^K f(k) * floor(x/k)
    # = sum_{k=1}^K f(k) * (number of multiples of k up to x)
    # = sum_{j=1}^x (sum_{k|j, k<=K} f(k))
    # = sum_{j=1}^x h(j) where h(j) = sum_{d|j, d<=K} f(d)
    
    # So M(x) = sum_{j=1}^x h(j), and we need M(x) <= 1.0001 for all x in [1, 10K].
    # This means the cumulative sum of h(j) must never exceed 1.0001.
    
    # For the LP, we need constraints on all x in [1, 10K], which is 10K constraints.
    # For K=200, that's 2000 constraints, which is manageable.
    
    # Let's try K values: 50, 100, 150, 200, 300, 500
    # For each K, solve the LP if feasible within time.
    
    best_score = 0.0
    best_result = None
    
    K_candidates = [50, 100, 150, 200, 300, 500, 1000]
    
    for K in K_candidates:
        if elapsed() > deadline - 60:
            break
        
        # Get square-free numbers up to K
        sf_indices = np.where(mu[1:K+1] != 0)[0] + 1
        n = len(sf_indices)
        
        if n == 0:
            continue
        
        # Objective: minimize sum f(k)*log(k)/k (to maximize -sum f(k)*log(k)/k)
        # c = log(k)/k for each square-free k
        c = np.log(sf_indices.astype(np.float64)) / sf_indices.astype(np.float64)
        
        # Constraints:
        # 1. M(x) <= 1.0001 for all x in [1, 10K]
        #    M(x) = sum_{k in sf} f(k) * floor(x/k)
        #    So row x of A has entries floor(x/k) for each square-free k
        # 2. sum f(k)/k = 0
        # 3. -10 <= f(k) <= 10 (bounds)
        
        max_x = 10 * K
        
        # Build constraint matrix A for M(x) <= 1.0001
        # A has shape (max_x, n)
        # A[x-1, j] = floor(x / sf_indices[j])
        
        # This could be large. For K=200, n~122, max_x=2000, so 2000x122 matrix.
        # That's about 244K entries, manageable.
        
        if n * max_x > 5_000_000:
            # Too large for direct LP, skip
            continue
        
        A = np.zeros((max_x, n), dtype=np.float64)
        for j, k in enumerate(sf_indices):
            A[:, j] = np.floor(np.arange(1, max_x + 1) / k)
        
        # Add equality constraint: sum f(k)/k = 0
        # This is an equality, so we use A_eq, b_eq
        A_eq = np.zeros((1, n), dtype=np.float64)
        A_eq[0, :] = 1.0 / sf_indices.astype(np.float64)
        b_eq = np.zeros(1, dtype=np.float64)
        
        # Bounds: -10 <= f(k) <= 10
        bounds = [(-10.0, 10.0)] * n
        
        # Inequality: A @ f <= 1.0001
        b_ineq = np.full(max_x, 1.0001, dtype=np.float64)
        
        # Solve LP: minimize c @ f subject to A @ f <= b_ineq, A_eq @ f = b_eq, bounds
        try:
            from scipy.optimize import linprog
            result = linprog(c, A_ub=A, b_ub=b_ineq, A_eq=A_eq, b_eq=b_eq, 
                            bounds=bounds, method='highs')
        except Exception:
            continue
        
        if not result.success:
            continue
        
        f_vals = result.x
        
        # Check feasibility more carefully
        # Compute M(x) for all x in [1, 10K]
        M_vals = A @ f_vals
        if np.max(M_vals) > 1.0001 + 1e-6:
            continue
        
        # Compute the score
        # First, clip values to [-10, 10]
        f_clipped = np.clip(f_vals, -10.0, 10.0)
        
        # Compute S = sum_k f(k)/k over clipped values
        S = np.sum(f_clipped / sf_indices.astype(np.float64))
        
        # f(1) adjustment: if 1 is in domain, f(1) = f(1) - S
        # Otherwise, f(1) = -S (but 1 must be in domain for this)
        # Actually, the problem says "f(1) is replaced by f(1) - S with S = sum_k f(k)/k"
        # If 1 is not in domain, f(1) = 0 before this step, so f(1) becomes -S
        # But then 1 must be added to domain? Let me re-read.
        
        # "f(1) is replaced by f(1) - S with S = sum_k f(k)/k over the clipped values 
        #  (f(1) = 0 before this if 1 is not in the domain), so that afterwards sum_k f(k)/k = 0"
        
        # So after adjustment, sum f(k)/k = 0.
        # The domain includes 1 if it was originally there, or we need to add it.
        # Actually, I think the evaluator handles this: it takes our domain and values,
        # clips them, computes S, adjusts f(1), and then checks feasibility and score.
        
        # So we just need to return domain and values, and the evaluator does the rest.
        # But we should make sure that after the adjustment, the feasibility holds.
        
        # Let's compute what the evaluator will do:
        # 1. Clip values to [-10, 10]
        # 2. Compute S = sum f(k)/k
        # 3. f(1) = f(1) - S (if 1 in domain) or f(1) = -S (add 1 to domain)
        # 4. Check M(x) <= 1.0001 for all x in [1, 10K]
        # 5. Score = -sum f(k)*log(k)/k
        
        # To be safe, let's include 1 in our domain and set f(1) such that after adjustment,
        # the sum is zero. But the evaluator will do the adjustment, so we just need
        # to ensure that after their adjustment, feasibility holds.
        
        # Let's compute the adjusted values:
        # If 1 is in sf_indices:
        if 1 in sf_indices:
            idx_1 = np.where(sf_indices == 1)[0][0]
            f_adj = f_clipped.copy()
            f_adj[idx_1] -= S
        else:
            # Need to add 1 to domain with value -S
            sf_indices_adj = np.concatenate([[1], sf_indices])
            f_adj = np.concatenate([[-S], f_clipped])
        
        # Now check feasibility with adjusted values
        # M(x) = sum_k f_adj(k) * floor(x/k)
        # Build new constraint matrix if we added 1
        if len(sf_indices_adj) != n:
            n_adj = len(sf_indices_adj)
            A_adj = np.zeros((max_x, n_adj), dtype=np.float64)
            for j, k in enumerate(sf_indices_adj):
                A_adj[:, j] = np.floor(np.arange(1, max_x + 1) / k)
            M_adj = A_adj @ f_adj
        else:
            M_adj = A @ f_adj
        
        if np.max(M_adj) > 1.0001 + 1e-6:
            # Infeasible after adjustment, skip
            continue
        
        # Compute score
        score = -np.sum(f_adj * np.log(sf_indices_adj.astype(np.float64)) / sf_indices_adj.astype(np.float64))
        
        if score > best_score:
            best_score = score
            best_result = (sf_indices_adj.astype(np.int64), f_adj.astype(np.float64))
    
    if best_result is not None:
        return best_result
    
    # Fallback: small feasible solution
    return np.array([1, 2, 3, 5, 30], dtype=np.int64), np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)

# EVOLVE-BLOCK-END
