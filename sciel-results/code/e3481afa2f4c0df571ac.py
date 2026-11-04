"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Ensure matrices are numpy arrays
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    def compute_cost(p):
        p_arr = np.asarray(p, dtype=np.int64)
        # Cost = sum a[i, j] * b[p[i], p[j]]
        # Vectorized: B_perm[i, j] = b[p[i], p[j]]
        # We can do this as np.sum(a * b[np.ix_(p_arr, p_arr)])
        return int(np.sum(a * b[np.ix_(p_arr, p_arr)]))

    def delta_swap(i, j, p):
        """
        Compute cost change from swapping facilities i and j.
        Assumes a and b are symmetric.
        """
        # v[k] = a[i, k] - a[j, k]
        v = a[i] - a[j]
        # u[k] = b[p[j], k] - b[p[i], k]
        # We need u[p[k]] for the dot product
        # u_perm[k] = b[p[j], p[k]] - b[p[i], p[k]]
        # This is (b[p[j], :] - b[p[i], :])[p]
        u = b[p[j]] - b[p[i]]
        u_perm = u[p]
        
        # Sum over all k
        total_sum = np.dot(v, u_perm)
        
        # Subtract terms for k=i and k=j
        # term_i = v[i] * u_perm[i] = (a[i,i] - a[j,i]) * (b[p[j], p[i]] - b[p[i], p[i]])
        # term_j = v[j] * u_perm[j] = (a[i,j] - a[j,j]) * (b[p[j], p[j]] - b[p[i], p[j]])
        term_i = v[i] * u_perm[i]
        term_j = v[j] * u_perm[j]
        
        delta = 2.0 * (total_sum - term_i - term_j)
        return delta

    def local_search(p, max_passes=1000):
        p = p[:]
        cur_cost = compute_cost(p)
        best_cost = cur_cost
        best_p = p[:]
        
        for _ in range(max_passes):
            if time.time() > deadline - 5.0:
                break
            improved = False
            # Try to find any improving swap
            # We can iterate i and find best j
            for i in range(n):
                if time.time() > deadline - 5.0:
                    break
                best_j = -1
                best_delta = 0
                # For each i, find the best j > i to swap with
                # We can vectorize the delta calculation for all j > i?
                # v = a[i] - a[j] depends on j. So not easily vectorized across j.
                # But n=100, so loop is fine.
                for j in range(i + 1, n):
                    d = delta_swap(i, j, p)
                    if d < best_delta:
                        best_delta = d
                        best_j = j
                
                if best_j != -1:
                    p[i], p[best_j] = p[best_j], p[i]
                    cur_cost += best_delta
                    improved = True
                    if cur_cost < best_cost:
                        best_cost = cur_cost
                        best_p = p[:]
            
            if not improved:
                break
        
        return best_p, best_cost

    def simulated_annealing(p, start_temp=100000.0, end_temp=10.0):
        p = p[:]
        cur_cost = compute_cost(p)
        best_cost = cur_cost
        best_p = p[:]
        t = start_temp
        cool_factor = 0.999
        no_improve = 0
        restart_threshold = 5000
        
        while time.time() < deadline - 2.0:
            if no_improve > restart_threshold:
                p = best_p[:]
                cur_cost = best_cost
                t = start_temp * 0.5
                no_improve = 0
            
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            d = delta_swap(i, j, p)
            
            if d < 0 or (t > 1e-9 and random.random() < np.exp(-d / t)):
                p[i], p[j] = p[j], p[i]
                cur_cost += d
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_p = p[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
            
            t = max(end_temp, t * cool_factor)
        
        return best_p, best_cost

    # Initialization: Rank Matching
    # Heavy flows to central locations (low sum of distances)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.tolist()
    
    cur_cost = compute_cost(p)
    best_p = p[:]
    best_cost = cur_cost
    
    # Local Search
    p, cur_cost = local_search(p, max_passes=500)
    if cur_cost < best_cost:
        best_cost = cur_cost
        best_p = p[:]
        
    # Simulated Annealing
    p, cur_cost = simulated_annealing(p, start_temp=50000.0, end_temp=5.0)
    if cur_cost < best_cost:
        best_cost = cur_cost
        best_p = p[:]
        
    # Final Local Search
    if time.time() < deadline - 5.0:
        p, cur_cost = local_search(best_p, max_passes=100)
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_p = p[:]
            
    return best_p

# EVOLVE-BLOCK-END
