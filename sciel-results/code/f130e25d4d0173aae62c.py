# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy import sparse
from scipy.optimize import linprog


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 580  # Leave 50s margin for evaluator overhead
    
    # Sieve for Mobius function up to K_max
    K_max = 200000
    mu = np.zeros(K_max + 1, dtype=np.int8)
    mu[1] = 1
    is_prime = np.ones(K_max + 1, dtype=bool)
    primes = []
    for i in range(2, K_max + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K_max:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    
    # Precompute mu(k)/k
    k_vals = np.arange(1, K_max + 1)
    mu_over_k = mu[1:] / k_vals.astype(np.float64)
    
    # Precompute log(k)/k for score calculation
    log_over_k = np.log(k_vals) / k_vals.astype(np.float64)
    
    # Try LP optimization with various K values
    K_candidates = [3000, 2000, 1500, 1000]
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    for K in K_candidates:
        if time.time() > deadline:
            break
        if K > K_max:
            continue
        
        # For LP: variables are f(1), f(2), ..., f(K)
        # Objective: maximize -sum f(k)*log(k)/k => minimize sum f(k)*log(k)/k
        # c = log(k)/k for k=1..K
        
        c = log_over_k[:K].copy()
        
        # Bounds: -10 <= f(k) <= 10
        bounds = [(-10, 10)] * K
        
        # Constraints:
        # 1. sum f(k)/k <= 1 (inequality form, but we want equality; use two inequalities)
        #    Actually, the evaluator handles the f(1) adjustment, so we don't need to enforce this.
        #    But including it might help. Let's skip it for now.
        
        # 2. M(x) = sum f(k)*floor(x/k) <= 1 for x in [1, 10K]
        x_max = 10 * K
        
        # Build sparse matrix A_ub for M(x) <= 1 constraints
        # A_ub has shape (x_max, K)
        # A_ub[i, k] = floor((i+1)/(k+1)) for x=i+1, variable index k (0-indexed)
        
        # Build column by column for memory efficiency
        rows = []
        cols = []
        data = []
        
        for k in range(1, K + 1):
            # For each x in [1, x_max], A[x-1, k-1] = floor(x/k)
            x_vals = np.arange(1, x_max + 1)
            floor_vals = x_vals // k
            # Only add non-zero entries
            nonzero_mask = floor_vals != 0
            if np.any(nonzero_mask):
                rows.extend(np.where(nonzero_mask)[0])
                cols.extend([k-1] * np.sum(nonzero_mask))
                data.extend(floor_vals[nonzero_mask])
        
        A_ub = sparse.csr_matrix((data, (rows, cols)), shape=(x_max, K))
        b_ub = np.ones(x_max)
        
        # Solve LP
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                            options={'presolve': True, 'time_limit': 300})
            
            if result.success:
                f = result.x
                f = np.clip(f, -10, 10)
                
                # Verify M(x) constraints
                x_range = np.arange(1, x_max + 1, dtype=np.int64)
                M_prime = np.zeros(x_max, dtype=np.float64)
                for k in range(1, K + 1):
                    v = f[k-1]
                    if v != 0.0:
                        M_prime += v * (x_range // k)
                
                max_M = np.max(M_prime)
                if max_M <= 1.0001:
                    score = -np.sum(f * log_over_k[:K])
                    if score > best_score:
                        best_score = score
                        best_domain = k_vals[:K].copy()
                        best_values = f.copy()
                        
        except Exception as e:
            pass
    
    if best_domain is None:
        # Fallback to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    else:
        domain = best_domain
        values = best_values
    
    return domain, values

# EVOLVE-BLOCK-END
