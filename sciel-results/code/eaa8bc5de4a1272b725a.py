"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    # Safety margin: 120s limit, aim to finish by 115s
    deadline = time.time() + 115.0

    # 1. Initialization: Rank-matching
    # Assign facilities with highest flow sum to locations with lowest distance sum
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    best_cost = compute_cost(p)
    best_p = p.copy()

    # Precompute transposes if needed, but a[:, i] is a view
    a_t = a.T

    def swap_delta(i, j, p):
        u = p[i]
        v = p[j]
        
        # Vectorized delta calculation
        # term1 = sum_k (a[i, k] - a[j, k]) * (b[v, p[k]] - b[u, p[k]])
        # term2 = sum_k (a[k, i] - a[k, j]) * (b[p[k], v] - b[p[k], u])
        
        diff_a_rows = a[i, :] - a[j, :]
        diff_b_rows = b[v, p] - b[u, p]
        term1 = np.dot(diff_a_rows, diff_b_rows)
        
        diff_a_cols = a_t[:, i] - a_t[:, j]
        diff_b_cols = b[p, v] - b[p, u]
        term2 = np.dot(diff_a_cols, diff_b_cols)
        
        return term1 + term2

    def steepest_descent_local_search(p, current_cost, deadline):
        improved = True
        # Limit iterations to prevent infinite loops in case of floating point issues (though ints)
        max_iter = 1000 
        iter_count = 0
        
        while improved and time.time() < deadline - 1.0 and iter_count < max_iter:
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1
            
            # Scan all pairs to find the steepest descent move
            for i in range(n - 1):
                if time.time() >= deadline - 1.0:
                    break
                for j in range(i + 1, n):
                    if time.time() >= deadline - 1.0:
                        break
                    d = swap_delta(i, j, p)
                    if d < best_delta:
                        best_delta = d
                        best_i = i
                        best_j = j
            
            if best_delta < 0:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
                improved = True
                iter_count += 1
            else:
                break
                
        return p, current_cost

    # Run initial local search
    p, current_cost = steepest_descent_local_search(p, best_cost, deadline)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()

    # 2. Iterated Local Search (ILS)
    # Use random perturbations to escape local minima
    # We keep the best solution found so far and perturb it
    
    # Set a safety margin for the final return to ensure we don't crash
    # We need to leave enough time for the final cost verification and return
    while time.time() < deadline - 5.0: 
        # Perturb the best solution
        p = best_p.copy()
        current_cost = best_cost
        
        # Perform random swaps
        # Number of swaps: typically between 5 and 15 for n=150
        num_swaps = np.random.randint(5, 16)
        for _ in range(num_swaps):
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            if i != j:
                p[i], p[j] = p[j], p[i]
        
        # Compute cost of perturbed solution
        current_cost = compute_cost(p)
        
        # Run local search from the perturbed state
        p, current_cost = steepest_descent_local_search(p, current_cost, deadline)
        
        # Update best if improved
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
        # If time is running out, we might want to stop early to ensure clean exit
        if time.time() > deadline - 10.0:
            break

    # Final verification
    final_cost = compute_cost(best_p)
    if final_cost < best_cost:
        best_cost = final_cost

    return best_p.tolist()

# EVOLVE-BLOCK-END
