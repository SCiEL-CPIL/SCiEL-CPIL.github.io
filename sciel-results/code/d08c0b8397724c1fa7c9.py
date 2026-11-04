"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Helper functions
    def calc_cost(p):
        b_perm = b[p][:, p]
        return int(np.sum(a * b_perm))
    
    def swap_delta(i, j, p, Bp, M_cache):
        """Compute delta for swapping facilities i and j (locations p[i], p[j])"""
        u = p[i]
        v = p[j]
        # M_cache[i] = sum_k a[i,k] * b[:, p[k]]
        # M_cache[j] = sum_k a[j,k] * b[:, p[k]]
        # delta = (M_cache[i][v] - M_cache[i][u]) + (M_cache[j][u] - M_cache[j][v])
        return (M_cache[i][v] - M_cache[i][u]) + (M_cache[j][u] - M_cache[j][v])
    
    def compute_M(p, b):
        """M[i] = sum_k a[i,k] * b[:, p[k]]"""
        # M = a @ (b[:, p])^T? No.
        # M[i, loc] = sum_k a[i,k] * b[p[k], loc]
        # This is (a @ b[p])^T?
        # Let Bp = b[:, p] -> shape (n, n), Bp[k, loc] = b[p[k], loc] = b.loc, p[k]
        # M[i, loc] = sum_k a[i,k] * Bp[k, loc]
        # So M = a @ Bp
        Bp = b[:, p]
        return a @ Bp
    
    # 1. Initialization: Spectral / Linear Relaxation
    # We formulate: min Tr(P^T A P B)
    # Relaxed: min Tr(X^T A X B) s.t. X doubly stochastic
    # One approach: use eigendecomposition or SVD.
    # Another: solve a sequence of LAPs (linear assignment problems)
    # using a cost matrix derived from A and B.
    
    # Simple Heuristic Initialization: Rank Matching
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)
    
    # Sort facilities by descending flow (heavy first)
    heavy_order = np.argsort(-a_row_sums, kind="stable")
    # Sort locations by ascending distance (central first)
    central_order = np.argsort(b_col_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    
    best_p = p.copy()
    best_cost = calc_cost(best_p)
    
    # 2. Spectral Refinement of Initial Solution
    # Use eigenvalues/vectors to improve the starting point.
    # Idea: Minimize sum A[i,j] B[p[i], p[j]].
    # This is related to the quadratic form of A and B.
    # We can try to align the dominant eigenvectors of A and B.
    # However, for QAP, a direct spectral method is complex.
    # Let's stick with the rank matching but add a small random perturbation
    # and a quick local search to refine it immediately.
    
    # Quick Refinement: One pass of local search to fix obvious bad moves
    def quick_local_search(p, max_iters=1000, time_limit=5.0):
        current_cost = calc_cost(p)
        Bp = b[:, p]
        M = a @ Bp
        start = time.time()
        
        for _ in range(max_iters):
            if time.time() > start + time_limit:
                break
            
            # Find best swap
            best_delta = 0  # Only accept negative deltas (improvements)
            best_i, best_j = -1, -1
            
            # Sample random swaps to find a good move quickly?
            # Or exhaustive search is O(n^2). With n=150, 22500 ops. Fast.
            for i in range(n):
                for j in range(i + 1, n):
                    u = p[i]
                    v = p[j]
                    delta = (M[i][v] - M[i][u]) + (M[j][u] - M[j][v])
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
            
            if best_delta < 0:
                # Apply move
                i, j = best_i, best_j
                u, v = p[i], p[j]
                p[i], p[j] = v, u
                current_cost += best_delta
                
                # Update M
                # M[k] changes because Bp changed for positions i and j.
                # M[k, u] -= a[k, j] (since p[i] is now v, p[j] is now u? No.)
                # Old: p[i]=u, p[j]=v.
                # New: p[i]=v, p[j]=u.
                # Bp[i] was b[:, u], now b[:, v].
                # Bp[j] was b[:, v], now b[:, u].
                # M[k, :] = sum_q a[k,q] Bp[q, :]
                # M[k, u] = sum_q a[k,q] Bp[q, u]. 
                # Bp[q, u] is b[p[q], u].
                # Only q=i and q=j changed their contributions to the sum.
                # For q=i: a[k,i] * b[p[i], u]. Old b[u,u], New b[v,u].
                # For q=j: a[k,j] * b[p[j], u]. Old b[v,u], New b[u,u].
                # So M[k, u] += a[k,i](b[v,u]-b[u,u]) + a[k,j](b[u,u]-b[v,u])
                #            = (a[k,i]-a[k,j]) * (b[v,u]-b[u,u])
                # Similarly M[k, v] += (a[k,j]-a[k,i]) * (b[u,v]-b[v,v])
                # Note b is symmetric usually, so b[v,u]=b[u,v], b[u,u]=b[v,v] is false.
                
                diff_a = a[:, i] - a[:, j]
                delta_u = (diff_a) * (b[v, u] - b[u, u])
                delta_v = (-diff_a) * (b[u, v] - b[v, v])
                # Wait, b[v,u] == b[u,v] if symmetric.
                # M[:, u] += (a[:, i] - a[:, j]) * (b[v, u] - b[u, u])
                # M[:, v] += (a[:, j] - a[:, i]) * (b[u, v] - b[v, v])
                
                M[:, u] += diff_a * (b[v, u] - b[u, u])
                M[:, v] -= diff_a * (b[u, v] - b[v, v])
                
                # Also swap columns in Bp if we maintained it? 
                # We recompute M from scratch or update. Updating M is efficient.
                
            else:
                break # Local optimum found
                
        return p, current_cost

    # Run initial quick refinement
    p, cost = quick_local_search(p, max_iters=2000, time_limit=5.0)
    if cost < best_cost:
        best_cost = cost
        best_p = p.copy()

    # 3. Main Loop: Iterated Local Search with Tabu
    # Use Tabu Search to escape local optima.
    
    time_remaining = deadline - time.time()
    tabu_tenure = 10 + random.randint(0, 10)
    
    # We will run multiple restarts/re-phases
    # Phase 1: Strong Tabu Search
    # Phase 2: If time remains, try different initializations
    
    # Data structures for Tabu
    # tabu_until[i, j] = step number when move (i,j) leaves tabu
    # Using a dictionary is slow. Array is better.
    tabu_matrix = np.zeros((n, n), dtype=np.int64)
    
    p_curr = best_p.copy()
    curr_cost = best_cost
    Bp = b[:, p_curr]
    M = a @ Bp
    
    step = 0
    no_improve_count = 0
    max_no_improve = 500 # Restart logic
    
    while time.time() < deadline - 2.0: # Reserve 2s for final check/cleanup
        
        if no_improve_count > max_no_improve:
            # Perturb and restart
            # Random swaps
            for _ in range(5 + n // 50):
                i, j = np.random.choice(n, 2, replace=False)
                p_curr[i], p_curr[j] = p_curr[j], p_curr[i]
            curr_cost = calc_cost(p_curr)
            
            # Recompute M
            Bp = b[:, p_curr]
            M = a @ Bp
            
            no_improve_count = 0
            tabu_matrix.fill(0)
            step = 0
            continue
            
        # Find best non-tabu move (or best move if all are tabu and improve global)
        best_delta = 0 # Aspiration: only accept negative (improving) unless all tabu
        best_delta_aspir = 0
        best_i, best_j = -1, -1
        best_i_aspir, best_j_aspir = -1, -1
        
        # Efficiently finding the best move is O(n^2).
        # With n=150, this is fast.
        
        # Precompute potential improvements
        # Delta[i,j] = (M[i, p[j]] - M[i, p[i]]) + (M[j, p[i]] - M[j, p[j]])
        
        # Vectorized approach might be hard due to dynamic p[i] inside index.
        # Let's loop.
        
        min_val = np.inf
        min_val_aspir = np.inf
        
        for i in range(n):
            ui = p_curr[i]
            Mi = M[i]
            for j in range(i + 1, n):
                # Check tabu
                is_tabu = tabu_matrix[i, j] > step
                if is_tabu:
                    # Check aspiration: if it's better than best global?
                    # We need delta.
                    pass
                
                uj = p_curr[j]
                Mj = M[j]
                delta = (Mi[uj] - Mi[ui]) + (Mj[ui] - Mj[uj])
                
                if not is_tabu:
                    if delta < min_val:
                        min_val = delta
                        best_i, best_j = i, j
                else:
                    # Only consider if it improves the global best?
                    # Standard aspiration: if move improves the best-known solution
                    # (global best), allow it even if tabu.
                    # But we track curr_cost. If curr_cost + delta < best_cost, allow.
                    if delta < min_val_aspir:
                        min_val_aspir = delta
                        best_i_aspir, best_j_aspir = i, j
        
        # Decide which move to take
        if min_val < 0:
            i_move, j_move = best_i, best_j
            delta_take = min_val
        elif min_val_aspir < best_cost - curr_cost:
            # Aspiration: This move makes curr_cost better than best_cost?
            # Actually, standard aspiration is if it improves the GLOBAL BEST.
            # If curr_cost + delta < best_cost, then it improves global best.
            i_move, j_move = best_i_aspir, best_j_aspir
            delta_take = min_val_aspir
        else:
            # No improving moves (tabu or not)
            # Do a kick (random perturb)
            for _ in range(3):
                i, j = np.random.choice(n, 2, replace=False)
                p_curr[i], p_curr[j] = p_curr[j], p_curr[i]
            curr_cost = calc_cost(p_curr)
            Bp = b[:, p_curr]
            M = a @ Bp
            tabu_matrix.fill(0)
            step = 0
            continue
        
        # Apply move
        i, j = i_move, j_move
        u, v = p_curr[i], p_curr[j]
        p_curr[i], p_curr[j] = v, u
        curr_cost += int(delta_take)
        
        # Update Tabu
        tabu_matrix[i, j] = step + tabu_tenure
        tabu_matrix[j, i] = step + tabu_tenure
        
        # Update M
        # diff_a = a[:, i] - a[:, j]
        # M[:, u] += diff_a * (b[v, u] - b[u, u])
        # M[:, v] -= diff_a * (b[u, v] - b[v, v])
        diff_a = a[:, i] - a[:, j]
        M[:, u] += diff_a * (b[v, u] - b[u, u])
        M[:, v] -= diff_a * (b[u, v] - b[v, v])
        
        # Check improvement
        if curr_cost < best_cost:
            best_cost = curr_cost
            best_p = p_curr.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        step += 1
        
        # Periodically clear tabu list to avoid infinite loops in tabu zones?
        # Tabu tenure handles this naturally as step increases.
        
    # Final verification
    final_cost = calc_cost(best_p)
    if final_cost != best_cost:
        # Should not happen if updates are correct, but safety check
        if final_cost < best_cost:
            best_cost = final_cost
            # best_p is already optimal for this cost
        
    return best_p.tolist()

# EVOLVE-BLOCK-END
