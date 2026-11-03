"""QAP Solver using Hungarian Algorithm Global Reassignment and 2-opt Polishing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta_vectorized(a_sym, b, p, i, j):
    """
    Compute delta for swapping positions i and j in permutation p.
    Vectorized version for speed.
    """
    x = p[i]
    y = p[j]
    
    # diff_a[k] = a_sym[i, k] - a_sym[j, k]
    diff_a = a_sym[i, :] - a_sym[j, :]
    
    # diff_b[k] = b[y, k] - b[x, k]
    diff_b = b[y, :] - b[x, :]
    
    # Calculate sum over all k: sum_k diff_a[k] * diff_b[p[k]]
    temp = diff_a * diff_b[p]
    
    # Subtract terms for k=i and k=j
    delta = np.sum(temp) - diff_a[i] * diff_b[x] - diff_a[j] * diff_b[y]
    
    return delta

def _hungarian_step(a, b, p):
    """
    Perform a global reassignment using the Hungarian algorithm.
    Linearize the QAP cost by fixing the current assignment p and calculating
    the marginal cost of assigning each facility i to each location x.
    """
    n = len(a)
    
    # C[i, x] = sum_j a[i, j] * b[x, p[j]]
    M = b[:, p]  # Shape (n, n), M[x, j] = b[x, p[j]]
    C = a @ M.T  # Shape (n, n), C[i, x] = sum_j a[i, j] b[x, p[j]]
    
    row_ind, col_ind = linear_sum_assignment(C)
    
    new_p = np.empty(n, dtype=np.int64)
    new_p[row_ind] = col_ind
    
    return new_p

def _vectorized_2opt_pass(a_sym, b, p):
    """
    Perform one full pass of 2-opt with vectorized delta computation.
    Returns (new_p, num_swaps, min_delta_found)
    """
    n = len(p)
    p = p.copy()
    num_swaps = 0
    min_delta = float('inf')
    
    # For each i, compute deltas for all j > i
    for i in range(n - 1):
        x = p[i]
        
        # diff_a for facility at position i vs all other positions
        # We need to compute delta for swapping i with each j > i
        
        # Precompute diff_b for position i vs all other positions
        # diff_b_j = b[p[j], :] - b[x, :] for j > i
        
        # Vectorized: compute all deltas at once
        # For j in range(i+1, n):
        #   diff_a = a_sym[i, :] - a_sym[j, :]  (but j is position, not facility)
        #   Wait, a_sym is indexed by facility, not position.
        #   p[i] is the facility at position i.
        
        # Let me reconsider: a_sym[facility1, facility2]
        # When we swap positions i and j, we swap facilities p[i] and p[j].
        # The delta depends on a_sym[p[i], :] and a_sym[p[j], :]
        
        # For fixed i, varying j:
        # diff_a_j = a_sym[p[i], :] - a_sym[p[j], :]
        # diff_b_j = b[p[j], :] - b[p[i], :]
        
        # We can precompute:
        # A_i = a_sym[p[i], :]  (shape n)
        # B_i = b[p[i], :]      (shape n)
        
        A_i = a_sym[p[i], :]
        B_i = b[p[i], :]
        
        # For each j > i:
        # diff_a_j = A_i - a_sym[p[j], :]
        # diff_b_j = b[p[j], :] - B_i
        # delta_j = sum_k (A_i[k] - a_sym[p[j], k]) * (b[p[j], k] - B_i[k]) - correction
        
        # Let's compute this more carefully
        # delta = sum_k [a_sym[x,k] - a_sym[y,k]] * [b[y,k] - b[x,k]] - a_sym[x,x]*b[x,x] - a_sym[y,y]*b[y,y]
        # where x = p[i], y = p[j]
        
        # Expand: sum_k a_sym[x,k]*b[y,k] - sum_k a_sym[x,k]*b[x,k] - sum_k a_sym[y,k]*b[y,k] + sum_k a_sym[y,k]*b[x,k]
        #         - a_sym[x,x]*b[x,x] - a_sym[y,y]*b[y,y]
        
        # Let S_xy = sum_k a_sym[x,k]*b[y,k]
        # Let S_xx = sum_k a_sym[x,k]*b[x,k]
        # Let S_yy = sum_k a_sym[y,k]*b[y,k]
        # Let S_yx = sum_k a_sym[y,k]*b[x,k] = S_xy (by symmetry of the product? No, not necessarily)
        
        # Actually S_yx = sum_k a_sym[y,k]*b[x,k], which is different from S_xy.
        
        # This is getting complex. Let me just use the original formula but vectorize over j.
        
        # For fixed i, compute delta for all j > i:
        js = np.arange(i + 1, n)
        if len(js) == 0:
            continue
            
        y_facs = p[js]  # facilities at positions j
        
        # diff_a for each j: shape (len(js), n)
        # diff_a[j, k] = a_sym[x, k] - a_sym[y_j, k]
        diff_a_mat = A_i[None, :] - a_sym[y_facs, :]  # shape (len(js), n)
        
        # diff_b for each j: shape (len(js), n)
        # diff_b[j, k] = b[y_j, k] - b[x, k]
        diff_b_mat = b[y_facs, :] - B_i[None, :]  # shape (len(js), n)
        
        # temp[j, k] = diff_a[j, k] * diff_b[j, k]
        temp = diff_a_mat * diff_b_mat  # shape (len(js), n)
        
        # sum_k temp[j, k]
        sums = np.sum(temp, axis=1)  # shape (len(js),)
        
        # Correction terms:
        # For each j: - diff_a[j, i] * diff_b[j, x] - diff_a[j, j] * diff_b[j, y_j]
        # diff_a[j, i] = A_i[i] - a_sym[y_j, i]
        # diff_b[j, x] = b[y_j, x] - B_i[x]
        # diff_a[j, j] = A_i[j] - a_sym[y_j, j]
        # diff_b[j, y_j] = b[y_j, y_j] - B_i[y_j]
        
        # This is still per-j. Let me just loop over j for the correction.
        
        # For now, let's find the best swap in this batch
        # We'll compute the full delta for each j
        best_j = -1
        best_delta = 0
        
        for idx, j in enumerate(js):
            y = p[j]
            diff_a = A_i - a_sym[y, :]
            diff_b = b[y, :] - B_i
            
            temp_full = diff_a * diff_b[p]
            delta = np.sum(temp_full) - diff_a[i] * diff_b[x] - diff_a[j] * diff_b[y]
            
            if delta < best_delta:
                best_delta = delta
                best_j = j
        
        if best_j >= 0 and best_delta < -1e-9:
            p[i], p[best_j] = p[best_j], p[i]
            num_swaps += 1
            min_delta = min(min_delta, best_delta)
    
    return p, num_swaps, min_delta

def _local_search_2opt(a, b, a_sym, p, max_iter=50, time_limit=None, start_time=None):
    """
    Perform 2-opt local search to refine the solution.
    Uses first-improvement strategy for speed.
    """
    n = len(a)
    current_cost = _qap_cost(a, b, p)
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
                
        # Do one full pass with first-improvement
        new_p, num_swaps, _ = _vectorized_2opt_pass(a_sym, b, p)
        
        if num_swaps > 0:
            p = new_p
            current_cost = _qap_cost(a, b, p)
            improved = True
    
    return p, current_cost

def _perturb(p, n_swaps=5):
    """Randomly swap n_swaps pairs to perturb the solution."""
    p = p.copy()
    n = len(p)
    for _ in range(n_swaps):
        i, j = random.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 118.0 # Safety margin (150s - 32s for overhead)
    
    # Precompute symmetric a for fast delta
    a_sym = a + a.T
    
    # Initial solution: Rank matching (greedy)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()
    
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Track stagnation
    stagnation_count = 0
    no_improve_threshold = 3
    
    # Main loop: Alternating Hungarian Global Reassignment and 2-opt Polishing
    while time.time() - start_time < time_limit:
        # 1. Hungarian Step
        new_p = _hungarian_step(a, b, p)
        
        # Check if it improved
        new_cost = _qap_cost(a, b, new_p)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_p = new_p.copy()
            p = new_p.copy()
            stagnation_count = 0
        else:
            # Accept if within 5% of best to allow diversification
            if new_cost <= best_cost * 1.05:
                p = new_p.copy()
                stagnation_count += 1
            else:
                stagnation_count += 1
        
        # 2. 2-opt Polishing
        polish_start = time.time()
        p, current_cost = _local_search_2opt(a, b, a_sym, p, max_iter=30, time_limit=time_limit - 8.0, start_time=start_time)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            stagnation_count = 0
        
        # 3. Perturbation if stuck
        if stagnation_count >= no_improve_threshold:
            if time.time() - start_time < time_limit - 15.0:
                perturbed_p = _perturb(p, n_swaps=random.randint(3, 8))
                perturbed_cost = _qap_cost(a, b, perturbed_p)
                
                # Only accept perturbation if not too bad
                if perturbed_cost < best_cost * 1.1:
                    p = perturbed_p
                    current_cost = perturbed_cost
                stagnation_count = 0
        
        # Check time
        if time.time() - start_time > time_limit - 10.0:
            break
            
    # Final polish with more iterations if time permits
    if time.time() - start_time < time_limit - 3.0:
        final_p, final_cost = _local_search_2opt(a, b, a_sym, best_p, max_iter=20, time_limit=time_limit - 1.0, start_time=start_time)
        if final_cost < best_cost:
            best_cost = final_cost
            best_p = final_p
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
