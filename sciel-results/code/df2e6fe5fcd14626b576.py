# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt
import scipy.sparse as sp

def _build_lp_matrices(K):
    """
    Build the constraint matrix A_ub and b_ub for the LP problem.
    A_ub has shape (10*K, K) and contains floor(x/k) for x in 1..10K, k in 1..K.
    b_ub is ones(10*K).
    """
    max_x = 10 * K
    n_rows = max_x
    n_cols = K
    
    # We will build the sparse matrix A_ub
    # A_ub[i, j] = floor((i+1) / (j+1))
    # Note: numpy indices are 0-based, but math uses 1-based.
    
    # Efficient way to fill sparse matrix:
    # For each column k (1-based), the values are floor(x/k) for x=1..10K.
    # This is a step function.
    
    rows = []
    cols = []
    data = []
    
    # Iterate over k (1-based)
    for k in range(1, K + 1):
        # For a fixed k, floor(x/k) is constant on intervals [j*k, (j+1)*k - 1].
        # The value is j.
        # j ranges from 0 to 10K // k.
        # If j=0, the value is 0, so we can skip adding to sparse matrix (unless we need explicit zeros, which we don't for linprog).
        # Actually, floor(x/k) for x < k is 0.
        # So we start j from 1.
        
        max_j = max_x // k
        if max_j < 1:
            continue
            
        for j in range(1, max_j + 1):
            # The interval of x where floor(x/k) == j is [j*k, min((j+1)*k - 1, max_x)]
            start_x = j * k
            end_x = min((j + 1) * k - 1, max_x)
            
            # Number of x values in this interval
            count = end_x - start_x + 1
            if count <= 0:
                continue
                
            # Indices for x (0-based for numpy)
            # x goes from start_x to end_x (1-based)
            # So 0-based indices are start_x - 1 to end_x - 1
            x_indices = np.arange(start_x - 1, end_x)
            
            # All these entries in column k-1 have value j
            rows.extend(x_indices)
            cols.extend([k - 1] * count)
            data.extend([j] * count)
            
    A_ub = sp.csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))
    b_ub = np.ones(n_rows)
    
    return A_ub, b_ub

def _solve_lp_for_K(K, time_limit):
    """
    Solve the LP for a given K.
    Returns (domain, values) or None.
    """
    start = time.time()
    
    # Objective: Minimize -sum f(k) * log(k) / k
    # c_k = -log(k) / k
    k_vals = np.arange(1, K + 1, dtype=np.float64)
    c = -np.log(k_vals) / k_vals
    
    # Constraints: M(x) <= 1
    A_ub, b_ub = _build_lp_matrices(K)
    
    # Equality constraint: sum f(k) / k = 0
    A_eq = (1.0 / k_vals).reshape(1, -1)
    b_eq = np.array([0.0])
    
    # Bounds: -10 <= f(k) <= 10
    bounds = [(-10.0, 10.0)] * K
    
    try:
        # Use HiGHS solver, which is efficient for LP
        res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                          bounds=bounds, method='highs')
        
        if not res.success:
            return None
            
        f_vals = res.x
        
        # Filter out very small values to reduce domain size? 
        # The problem allows any distinct positive integers. 
        # Keeping all K is fine, but if f(k) is 0, it doesn't contribute to M(x) or objective.
        # However, the evaluator sums over the domain provided.
        # If we provide domain 1..K, it's fine.
        
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = f_vals.astype(np.float64)
        
        return domain, values
        
    except Exception:
        return None

def _evaluate_candidate(domain, values):
    """
    Simulate the evaluator to check feasibility and compute score.
    Returns (is_feasible, score).
    """
    # 1. Clip values
    clipped_values = np.clip(values, -10.0, 10.0)
    
    # 2. Compute S = sum f(k)/k
    k_float = domain.astype(np.float64)
    S = np.sum(clipped_values / k_float)
    
    # 3. Adjust f(1)
    # "f(1) is replaced by f(1) - S ... (f(1) = 0 before this if 1 is not in the domain)"
    # We need to construct the final values array corresponding to the domain.
    # Since domain is sorted (from LP), we can find index of 1.
    # Note: The evaluator logic says "f(1) is replaced". It implies if 1 is in domain, 
    # we modify the value at index where domain==1. If 1 is not in domain, 
    # we insert 1 with value -S? 
    # "f(1) = 0 before this if 1 is not in the domain"
    # This implies the function is extended to include 1 if it wasn't there, 
    # or the sum S is calculated over the existing domain, and then 
    # the value at k=1 is set to (current f(1) if exists, else 0) - S.
    
    # Let's construct the final f array for all k in domain, plus potentially 1.
    # But the domain returned must be the domain of the function.
    # If 1 is not in domain, we must add it? 
    # "domain holds distinct positive integers... values holds ... f(k)"
    # The evaluator computes M(x) = sum_k f(k) floor(x/k).
    # If 1 is not in domain, f(1) is effectively 0 in the sum? 
    # No, the text says: "f(1) is replaced by f(1) - S ... (f(1) = 0 before this if 1 is not in the domain)".
    # This strongly suggests that the evaluator *adds* 1 to the domain if it's missing, 
    # or simply treats the term for k=1 in the sums as (f(1)_orig - S) where f(1)_orig=0 if missing.
    # However, the return interface is (domain, values). 
    # If we return a domain that doesn't include 1, does the evaluator add it?
    # "values are clipped... Then f(1) is replaced... so that afterwards sum f(k)/k = 0".
    # If 1 is not in the returned domain, the sum sum_k f(k)/k over the returned domain is S.
    # The evaluator then defines a new function g where g(k)=f(k) for k!=1, and g(1) = (f(1) if 1 in domain else 0) - S.
    # And then uses g for M(x) and objective.
    # So, effectively, we should ensure 1 is in our domain to control the value, 
    # OR we can return domain without 1, and the evaluator will handle it.
    # To be safe and consistent with the LP which includes 1, we will keep 1 in the domain.
    
    # For evaluation simulation:
    # Create a dictionary or array for final values.
    # Since K can be large, let's use a dense array for M(x) calculation if K is not too big,
    # or a sparse approach.
    
    # Let's create the final values for the domain.
    # If 1 is in domain:
    if 1 in domain:
        idx_1 = np.where(domain == 1)[0][0]
        final_values = clipped_values.copy()
        final_values[idx_1] = final_values[idx_1] - S
    else:
        # If 1 is not in domain, we need to add it for the calculation?
        # The evaluator says "f(1) = 0 before this".
        # So g(1) = -S.
        # The domain for M(x) calculation would effectively include 1.
        # But our returned domain doesn't.
        # For the sake of checking feasibility, we must account for the contribution of k=1.
        # M(x) = sum_{k in domain} f(k) floor(x/k) + g(1) * floor(x/1)
        # floor(x/1) = x.
        # So M(x) = sum_{k in domain, k!=1} f(k) floor(x/k) + (-S) * x.
        
        # Let's construct the effective domain and values for calculation.
        # We can just add 1 to the domain for calculation.
        calc_domain = np.concatenate(([1], domain))
        calc_values = np.concatenate(([-S], clipped_values))
        
        # But wait, if 1 is in domain, we already handled it.
        # Let's just use a helper to get the value for any k.
        # Since domain is sorted, we can use searchsorted or a dict.
        # For speed in simulation, let's stick to the case where 1 is in domain, 
        # which is what our LP produces.
        pass # We assume 1 is in domain for the LP solution.
    
    # Check Feasibility: M(x) <= 1.0001 for x in [1, 10*K_max]
    # K_max is max(domain).
    K_max = int(np.max(domain))
    max_x = 10 * K_max
    
    # We need to compute M(x) for all x in 1..max_x.
    # M(x) = sum_{k in domain} final_f(k) * floor(x/k)
    
    # We can compute this using the difference array trick or direct summation.
    # Direct summation is O(m * max_x) which is too slow for large K.
    # The difference array trick is O(m * (max_x/k)) which is O(m * max_x * log(max_x)) roughly?
    # Actually, the number of intervals for k is max_x/k. Sum over k is max_x * H(max_x).
    # This is efficient enough for K up to ~5000.
    
    diff = np.zeros(max_x + 2, dtype=np.float64)
    
    for idx in range(len(domain)):
        k = int(domain[idx])
        f_k = final_values[idx]
        if f_k == 0.0:
            continue
            
        max_j = max_x // k
        if max_j == 0:
            continue
            
        # For each j, interval [j*k, min((j+1)*k-1, max_x)] has value f_k * j
        # Start: j*k, End: (j+1)*k - 1
        # We add f_k*j to diff[start] and -f_k*j to diff[end+1]
        
        # Vectorized update
        starts = np.arange(1, max_j + 1, dtype=np.int64) * k
        ends = np.arange(2, max_j + 2, dtype=np.int64) * k
        ends = np.minimum(ends, max_x + 1)
        vals = f_k * np.arange(1, max_j + 1, dtype=np.float64)
        
        np.add.at(diff, starts, vals)
        np.add.at(diff, ends, -vals)
        
    M = np.cumsum(diff)
    M = M[:max_x + 1]
    
    # Check max M(x)
    if np.max(M) > 1.0001:
        return False, -np.inf
        
    # Compute objective a
    # a = -sum f(k) log(k) / k
    # Note: The objective uses the FINAL values (after f(1) adjustment).
    a = -np.sum(final_values * np.log(k_float) / k_float)
    
    return True, a

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = 600.0  # Safety margin before 630s kill
    
    best_result = None
    best_a = -np.inf
    
    # Try increasing K
    # K=200 is very fast. K=1000 is moderate. K=2000 might be slow.
    # We will try a sequence and stop if time runs out.
    
    K_list = [200, 500, 1000, 2000, 5000]
    
    for K in K_list:
        if time.time() - start_time > deadline - 120:
            break
            
        # Solve LP
        res = _solve_lp_for_K(K, deadline - 120 - (time.time() - start_time))
        
        if res is not None:
            domain, values = res
            
            # Evaluate
            is_feasible, a = _evaluate_candidate(domain, values)
            
            if is_feasible and a > best_a:
                best_a = a
                best_result = (domain, values)
                print(f"K={K}, a={a}, time={time.time()-start_time:.2f}s")
                
    if best_result is None:
        # Fallback to seed if all fail
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
        
    return best_result

# EVOLVE-BLOCK-END
