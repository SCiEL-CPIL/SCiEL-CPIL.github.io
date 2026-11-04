"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_cost(a, b, p):
    """Compute the total cost of the assignment p."""
    # Use matrix multiplication for speed
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # This is equivalent to trace(A * B_p) or sum(A * B_p)
    b_permuted = b[np.ix_(p, p)]
    return np.sum(a * b_permuted)

def _local_search(a, b, p, deadline):
    """
    Perform 2-Opt local search (swap) on permutation p.
    Returns the best permutation and cost found.
    """
    n = len(p)
    p = p.copy()
    best_cost = _compute_cost(a, b, p)
    
    # Precompute row sums? No, not needed for delta.
    
    while time.time() < deadline:
        best_delta = 0
        best_i, best_j = -1, -1
        
        # Find the best swap
        for i in range(n):
            ai = a[i]
            for j in range(i + 1, n):
                aj = a[j]
                u = p[i]
                v = p[j]
                
                # Calculate delta using vectorized dot product
                # Delta = sum_{k != i,j} (a[i,k] - a[j,k]) * (b[v,p[k]] - b[u,p[k]])
                # We compute full dot product and subtract k=i and k=j terms
                
                diff_a = ai - aj
                diff_b = b[v] - b[u]
                dot_val = np.dot(diff_a, diff_b)
                
                # Correction terms for k=i and k=j
                # term_i corresponds to k=i in the sum: (a[i,i] - a[j,i]) * (b[v,u] - b[u,u])
                # term_j corresponds to k=j in the sum: (a[i,j] - a[j,j]) * (b[v,v] - b[u,v])
                
                # Assuming b[u,u] = 0 and b[v,v] = 0
                term_i = (a[i, i] - a[j, i]) * b[v, u]
                term_j = (a[i, j] - a[j, j]) * (-b[u, v]) # b[v,v] is 0, so -b[u,v]
                
                # Note: b[u,v] == b[v,u] for symmetric matrix
                # So term_j = (a[i,j] - a[j,j]) * (-b[v,u])
                
                delta = dot_val - term_i - term_j
                
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j
        
        if best_delta < 0:
            # Apply the best swap
            p[best_i], p[best_j] = p[best_j], p[best_i]
            best_cost += best_delta
        else:
            # No improvement found, local optimum reached
            break
            
    return p, best_cost

def solve(a, b):
    n = len(a)
    
    # 1. Initial Solution: Rank Matching
    # Sort facilities by total flow (descending)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    # Sort locations by total distance (ascending)
    central = np.argsort(b.sum(axis=1), kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # 2. Local Search
    # Set a deadline to ensure we return before the 120s limit
    # The evaluator kills at 150s, but we should aim for 120s.
    # We use 110s as a safety margin.
    deadline = time.time() + 110
    
    best_p, best_cost = _local_search(a, b, p, deadline)
    
    # 3. Optional: Restart with perturbation if time remains
    # Given the efficiency of 2-Opt for n=150, a single run from a good start is often sufficient.
    # However, we can add a simple restart loop if we have time.
    
    # For now, return the best found.
    return best_p.tolist()

# EVOLVE-BLOCK-END
