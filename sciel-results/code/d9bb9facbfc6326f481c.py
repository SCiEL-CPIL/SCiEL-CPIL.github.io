"""QAP solver for tho150 using First Improvement local search with efficient swap evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    """Solve QAP: assign facilities to locations to minimize sum a[i,j]*b[p[i],p[j]].
    
    Args:
        a: flow matrix (n x n) int64
        b: distance matrix (n x n) int64
    Returns:
        p: permutation list where facility i goes to location p[i]
    """
    n = len(a)
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    # Precompute row sums for initial heuristics
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Convert to float for faster arithmetic in delta computation
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    
    def compute_cost(p):
        """Compute exact cost of permutation p."""
        M = b[np.ix_(p, p)]
        return int(np.sum(a * M))
    
    def local_search_first_improvement(p, deadline):
        """First improvement local search. Returns improved permutation and cost."""
        cost = compute_cost(p)
        improved = True
        i_order = list(range(n))
        
        # Initialize M = b[np.ix_(p, p)]
        M = b_f[np.ix_(p, p)]
        
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(i_order)
            
            for i in i_order:
                if time.time() > deadline:
                    break
                
                pi = p[i]
                row_ai = a_f[i, :]
                col_ai = a_f[:, i]
                
                # Precompute vectors for i
                b_pi_p = b_f[pi, p]
                b_p_pi = b_f[p, pi]
                
                # Scan for first improving swap
                for j in range(n):
                    if j == i:
                        continue
                    
                    pj = p[j]
                    row_aj = a_f[j, :]
                    col_aj = a_f[:, j]
                    
                    # Vectorized delta calculation
                    # d = (row_ai - row_aj) @ (M[j, :] - b_pi_p) 
                    #   + (col_ai - col_aj) @ (M[:, j] - b_p_pi)
                    #   + (a[i,i]-a[j,j])*(b_f[pj,pj]-b_f[pi,pi])
                    
                    diff_row = row_ai - row_aj
                    diff_col = col_ai - col_aj
                    
                    d = np.dot(diff_row, M[j, :] - b_pi_p) + np.dot(diff_col, M[:, j] - b_p_pi)
                    d += (a_f[i, i] - a_f[j, j]) * (b_f[pj, pj] - b_f[pi, pi])
                    
                    if d < -1e-9: # Small epsilon to avoid floating point noise
                        # Accept swap
                        p[i], p[j] = p[j], p[i]
                        
                        # Update M
                        # M[i, :] = b_f[pj, p]
                        M[i, :] = b_f[pj, p]
                        # M[:, i] = b_f[p, pj]
                        M[:, i] = b_f[p, pj]
                        
                        # M[j, :] = b_f[pi, p]
                        M[j, :] = b_f[pi, p]
                        # M[:, j] = b_f[p, pi]
                        M[:, j] = b_f[p, pi]
                        
                        cost += int(d)
                        improved = True
                        break # Break j loop, continue to next i
        
        return p, cost

    def make_initial_solution(method, rng):
        """Create an initial permutation using different strategies."""
        if method == 0:
            # Rank matching: heavy flow -> central location
            heavy = np.argsort(-a_row_sums, kind='stable')
            central = np.argsort(b_row_sums, kind='stable')
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
        elif method == 1:
            # Reverse rank
            heavy = np.argsort(a_row_sums, kind='stable')
            central = np.argsort(-b_row_sums, kind='stable')
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
        elif method == 2:
            # Random
            p = rng.permutation(n).astype(np.int64)
        elif method == 3:
            # Sort by a_row_sums / b_row_sums ratio
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(b_row_sums > 0, a_row_sums / np.maximum(b_row_sums, 1), 0)
            fac_order = np.argsort(-ratio, kind='stable')
            loc_order = np.argsort(b_row_sums, kind='stable')
            p = np.empty(n, dtype=np.int64)
            p[fac_order] = loc_order
        else:
            p = rng.permutation(n).astype(np.int64)
        return p
    
    # Multiple initializations with ILS
    best_p = None
    best_cost = np.inf
    
    rng = np.random.default_rng(42)
    
    # Try several different initial solutions
    n_init = 12 # Increased restarts
    for init_idx in range(n_init):
        if time.time() > deadline:
            break
        
        method = init_idx % 4
        p = make_initial_solution(method, rng)
        
        # Local search
        p, cost = local_search_first_improvement(p, deadline)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
        
        # ILS: perturb and search again if time permits
        if time.time() < deadline - 20:
            for ils_iter in range(8): # More ILS iterations
                if time.time() > deadline:
                    break
                # Perturb: random swaps
                n_swaps = rng.integers(10, 30)
                p_perturbed = p.copy()
                for _ in range(n_swaps):
                    i, j = rng.integers(0, n, size=2)
                    if i != j:
                        p_perturbed[i], p_perturbed[j] = p_perturbed[j], p_perturbed[i]
                
                p_perturbed, cost_p = local_search_first_improvement(p_perturbed, deadline)
                
                if cost_p < best_cost:
                    best_cost = cost_p
                    best_p = p_perturbed.copy()
                elif cost_p < cost:
                    p = p_perturbed.copy()
                    cost = cost_p
    
    # Final verification
    if best_p is None:
        best_p = make_initial_solution(0, rng)
    
    final_cost = compute_cost(best_p)
    if final_cost < best_cost:
        best_cost = final_cost
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
