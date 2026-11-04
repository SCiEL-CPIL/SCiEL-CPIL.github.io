"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Ensure we are working with float64 for calculations to avoid overflow/int issues in intermediate steps
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    
    # Precompute interaction matrix C where C[i, j] = sum_k a[i, k] * b[j, p[k]]?
    # No, that depends on p.
    # Let's use the standard O(1) update for QAP swap (i, j).
    # Cost(p) = sum_{u,v} a[u,v] b[p[u], p[v]]
    # Let's define the cost contribution of a specific facility i at location loc:
    # But the cost is global.
    # 
    # Standard O(1) swap delta derivation:
    # Let p be the current permutation.
    # We swap facilities i and j.
    # Let u = p[i] and v = p[j].
    # 
    # The change in cost involves only terms where the facility index is i or j, 
    # OR the location index is u or v?
    # Actually, the term is a[x, y] * b[p[x], p[y]].
    # If we swap p[i] and p[j]:
    # 1. Terms where x=i or y=i:
    #    Old: sum_y a[i, y] b[u, p[y]] + sum_x a[x, i] b[p[x], u] - a[i, i] b[u, u] (diagonal counted twice in sum if not careful, but let's just sum all)
    #    Actually, it's easier to define:
    #    C_i_old = sum_{k != i} a[i, k] b[u, p[k]] + sum_{k != i} a[k, i] b[p[k], u]
    #    C_j_old = sum_{k != j} a[j, k] b[v, p[k]] + sum_{k != j} a[k, j] b[p[k], v]
    #    C_ij_old = a[i, j] b[u, v] + a[j, i] b[v, u]
    #    
    #    New (swap u and v):
    #    C_i_new = sum_{k != i} a[i, k] b[v, p[k]] + sum_{k != i} a[k, i] b[p[k], v]
    #    C_j_new = sum_{k != j} a[j, k] b[u, p[k]] + sum_{k != j} a[k, j] b[p[k], u]
    #    C_ij_new = a[i, j] b[v, u] + a[j, i] b[u, v]
    #    
    #    Note: In the sums for C_i_new and C_j_new, the term k=j and k=i are included in the "sum over k != i" and "sum over k != j" respectively?
    #    Wait, if I sum k != i, I include k=j.
    #    Let's refine the delta formula to be O(1) by precomputing vectors.
    #    
    #    Let A_i = a[i, :]
    #    Let B_u = b[u, :]
    #    
    #    We need to compute:
    #    Delta = (C_i_new + C_j_new + C_ij_new) - (C_i_old + C_j_old + C_ij_old)
    #    
    #    This is still O(n) if we compute the sums directly.
    #    To make it O(1), we can maintain auxiliary arrays:
    #    Let R[i, l] = sum_{k != i} a[i, k] b[l, p[k]]
    #    Let C[i, l] = sum_{k != i} a[k, i] b[p[k], l]
    #    
    #    If we maintain R and C, we can update them in O(n) when a swap happens.
    #    Then the delta for a swap (i, j) with locations u, v is:
    #    
    #    Old Cost parts involving i, j:
    #    Terms with i as 1st index in A: R[i, u] + a[i, j] b[u, v]  (Note: R[i, u] excludes k=i, but includes k=j? 
    #    If R[i, u] = sum_{k != i} a[i, k] b[u, p[k]], then for k=j, term is a[i, j] b[u, v].
    #    So R[i, u] includes the interaction with j.
    #    
    #    Let's define:
    #    S1[i, l] = sum_{k} a[i, k] b[l, p[k]]
    #    S2[i, l] = sum_{k} a[k, i] b[p[k], l]
    #    
    #    Then Cost = sum_i S1[i, p[i]] + sum_i S2[i, p[i]] - sum_i a[i,i] b[p[i], p[i]] ?
    #    No, Cost = sum_{i,j} a[i,j] b[p[i], p[j]].
    #    This is exactly sum_i sum_j a[i,j] b[p[i], p[j]].
    #    
    #    Let's stick to the O(n) delta calculation but optimize it with NumPy vectorization.
    #    For n=150, O(n) is 150 operations. 
    #    In Python, a loop of 150 ops is slow. But `np.dot` is fast.
    #    The previous program used `np.dot` for the delta, which is O(n) but implemented in C.
    #    The bottleneck in the previous program was likely the overhead of Python loop + random generation + function calls.
    #    
    #    Strategy: 
    #    1. Use a faster initialization (Spectral).
    #    2. Use Tabu Search with a short tabu list to prevent cycling.
    #    3. Vectorize the delta calculation as much as possible.
    #    4. Multiple restarts if time permits.
    
    def spectral_init():
        # Spectral method: Align eigenvectors of A and B
        try:
            w_a, v_a = np.linalg.eigh(a)
            w_b, v_b = np.linalg.eigh(b)
            
            # For minimization, we typically align the smallest eigenvalues?
            # Or largest? 
            # If A and B are cost matrices, we want to put high flow (large A) at low distance (small B).
            # Spectral methods often align the "centrality" vectors.
            # Let's try aligning the principal eigenvectors (largest eigenvalue) or smallest?
            # In QAP, usually we want to match the "shape" of the graphs.
            # Let's use the eigenvector corresponding to the largest eigenvalue for both, 
            # and sort facilities by their coordinate in v_a and locations by v_b.
            
            # Note: Eigenvectors can have arbitrary sign. We should ensure consistent orientation.
            # A common heuristic is to maximize the inner product of the sorted vectors.
            
            idx_a = np.argmax(w_a)
            idx_b = np.argmax(w_b)
            
            vec_a = v_a[:, idx_a]
            vec_b = v_b[:, idx_b]
            
            # Sort
            order_a = np.argsort(vec_a)
            order_b = np.argsort(vec_b)
            
            p = np.empty(n, dtype=np.int64)
            for i, fac in enumerate(order_a):
                p[fac] = order_b[i]
            
            return p
        except Exception:
            return np.arange(n)
    
    def calc_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))
    
    def get_delta(p, i, j):
        """
        Calculate delta cost for swapping facilities i and j.
        Uses vectorized operations.
        """
        u = p[i]
        v = p[j]
        
        # Rows of A
        ai = a[i]
        aj = a[j]
        
        # Rows of B permuted by p? 
        # We need sum_k a[i, k] b[u, p[k]] which is dot(ai, b[u, p])
        # b[u, p] is a vector.
        
        B_u = b[u, p]
        B_v = b[v, p]
        
        # Delta = 2 * (dot(ai, B_v) + dot(aj, B_u) - dot(ai, B_u) - dot(aj, B_v))
        # This assumes symmetry of A and B. 
        # If A and B are not symmetric, the formula is:
        # Old = sum_k a[i,k] b[u, p[k]] + sum_k a[k,i] b[p[k], u] + ...
        # New = sum_k a[i,k] b[v, p[k]] + sum_k a[k,i] b[p[k], v] + ...
        # 
        # Let's assume symmetry as is standard for QAPLIB. 
        # If not symmetric, we should use:
        # delta = (ai @ B_v + ai.T @ B_v.T ... ) -- wait.
        # 
        # General case:
        # Term 1: sum_k a[i, k] b[u, p[k]] -> ai @ b[u, p]
        # Term 2: sum_k a[k, i] b[p[k], u] -> a[:, i] @ b[p, u] = a[:, i] @ b[u, p] (since b is symmetric? No, b[p[k], u] is column u of b permuted. If b is symmetric, b[p[k], u] = b[u, p[k]].)
        # 
        # Assuming symmetry:
        delta = 2.0 * (np.dot(ai, B_v) + np.dot(aj, B_u) - np.dot(ai, B_u) - np.dot(aj, B_v))
        
        return delta

    # Tabu Search Parameters
    tabu_tenure = 10  # Keep last 10 moves in tabu list
    max_no_improve = 500  # Restart if no improvement for 500 steps
    
    best_p = None
    best_cost = float('inf')
    
    # Multi-start
    num_restarts = 0
    max_restarts = 5 # Will run as many as time permits
    
    while time.time() < deadline - 2.0:
        # Initialize
        p = spectral_init()
        # Random perturbation to break symmetry if spectral fails or is identical
        if num_restarts > 0:
            for _ in range(5):
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                if i != j:
                    p[i], p[j] = p[j], p[i]
        
        current_cost = calc_cost(p)
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
        tabu_list = []
        tabu_dict = {} # (i, j) -> expiration step
        step = 0
        no_improve_count = 0
        
        # Local Search Loop
        while time.time() < deadline - 2.0:
            # Check for restart condition
            if no_improve_count > max_no_improve:
                break
            
            # Select best non-tabu move
            best_delta = float('inf')
            best_i = -1
            best_j = -1
            
            # Sampling moves instead of full neighborhood for speed?
            # Full neighborhood is n*(n-1)/2 ~ 11k moves. 
            # 11k * O(n) dot products. 11k * 150 = 1.6M ops. 
            # In NumPy, this might take ~1-2 ms per iteration? 
            # 11k moves * 2ms = 22 seconds per full neighborhood scan. Too slow.
            
            # We need to sample moves or use a smarter selection.
            # Let's sample 100 random moves per step.
            
            candidates = 100
            for _ in range(candidates):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                
                # Check Tabu
                key = (min(i, j), max(i, j))
                if key in tabu_dict and tabu_dict[key] > step:
                    continue
                
                delta = get_delta(p, i, j)
                
                # Aspiration criterion: accept if it improves global best
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
                    best_key = key
            
            if best_i == -1:
                # No valid non-tabu move found in sample, or all worse?
                # If no move found, break or random move?
                # Let's just pick a random move ignoring tabu if stuck
                best_i = random.randint(0, n - 1)
                best_j = random.randint(0, n - 1)
                if best_i == best_j:
                    continue
                best_key = (min(best_i, best_j), max(best_i, best_j))
                best_delta = get_delta(p, best_i, best_j)
                
            # Perform move
            u = p[best_i]
            v = p[best_j]
            p[best_i], p[best_j] = v, u
            current_cost += best_delta
            
            # Update Tabu
            tabu_dict[best_key] = step + tabu_tenure
            # Clean up tabu dict occasionally to save memory/time
            if step % 100 == 0:
                tabu_dict = {k: v for k, v in tabu_dict.items() if v > step}
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            step += 1
            num_restarts += 1 # Rough count, actually this is steps in current run
            
            # If we've done enough steps in this run, restart
            if step > 2000:
                break
                
    return best_p.tolist()
# EVOLVE-BLOCK-END
