"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit (150s kill)

    # Helper functions
    def calc_cost_full(perm):
        return int(np.sum(a * b[np.ix_(perm, perm)]))

    def two_opt_delta(p, i, j):
        """Compute delta cost for swapping facilities i and j in permutation p."""
        li = p[i]
        lj = p[j]
        if li == lj:
            return 0
        # Use in-place operations to minimize allocations
        d = b[lj, :] - b[li, :]
        d[p] = 0  # Zero out positions i and j via their location values
        a_diff = a[i, :] - a[j, :] + a[:, i] - a[:, j]
        a_diff[i] = 0
        a_diff[j] = 0
        return int(np.dot(a_diff, d))

    def linearized_cost_matrix(p):
        """
        Compute the linearized cost matrix for Hungarian algorithm.
        cost[i,l] = sum_j a[i,j] * b[l, p[j]] + sum_j a[j,i] * b[p[j], l]
        """
        C1 = b[:, p]  # C1[l, j] = b[l, p[j]]
        C2 = b[p, :]  # C2[j, l] = b[p[j], l]
        term1 = a @ C1.T  # term1[i,l] = sum_j a[i,j] * b[l, p[j]]
        term2 = a.T @ C2  # term2[i,l] = sum_j a[j,i] * b[p[j], l]
        return term1 + term2

    def two_opt_polish(p, max_time, best_cost_ref=None):
        """Greedy 2-opt local search with best-cost tracking. Returns (p, cost)."""
        cost = calc_cost_full(p)
        if best_cost_ref is not None:
            best_cost_ref[0] = min(best_cost_ref[0], cost)
            best_cost_ref[1] = p.copy()
        improved = True
        passes = 0
        while improved and time.time() < max_time:
            improved = False
            passes += 1
            # Use random order for diversity in later passes
            indices = list(range(n))
            if passes > 1:
                random.shuffle(indices)
            for i_idx in indices:
                i = i_idx
                if time.time() > max_time:
                    break
                for j_idx in indices[i_idx+1:]:
                    j = j_idx
                    if i == j:
                        continue
                    if time.time() > max_time:
                        break
                    delta = two_opt_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        if best_cost_ref is not None and cost < best_cost_ref[0]:
                            best_cost_ref[0] = cost
                            best_cost_ref[1] = p.copy()
                        break
                if improved:
                    break
        return p, cost

    # --- Phase 1: Initial Solution via Linear Assignment on Row Sums ---
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    cost_matrix = np.outer(a_row_sums, b_row_sums)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    p = p.copy()

    # Initial polish with best tracking
    best_track = [float('inf'), None]
    p, current_cost = two_opt_polish(p, max_time=start_time + 15.0, best_cost_ref=best_track)
    best_p = best_track[1].copy() if best_track[1] is not None else p.copy()
    best_cost = best_track[0]

    # Check symmetry
    a_sym = np.array_equal(a, a.T)
    b_sym = np.array_equal(b, b.T)

    if a_sym and b_sym:
        # --- Phase 2: Enhanced Tabu Search with Best-Improvement Sampling ---
        tabu_end_time = start_time + 80.0
        
        # Tabu Search Structures
        tabu = np.zeros((n, n), dtype=np.int64)
        tabu_tenure = 40  # Slightly shorter tenure for more exploration
        iteration = 0
        consecutive_no_improve = 0
        max_consecutive_no_improve = 8
        
        # Precompute row/col sums of a for faster delta estimation
        a_row_sums_cache = a.sum(axis=1)
        a_col_sums_cache = a.sum(axis=0)
        
        while time.time() < tabu_end_time:
            improved = False
            # Use best-improvement among sampled candidates for better move quality
            best_delta = 0
            best_i = -1
            best_j = -1
            
            # Sample a subset of moves to find the best one
            num_samples = min(200, n * n // 10)  # Adaptive sampling
            sampled_moves = set()
            for _ in range(num_samples):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                key = (min(i,j), max(i,j))
                if key in sampled_moves:
                    continue
                sampled_moves.add(key)
                
                # Check tabu
                is_tabu = tabu[i, j] > iteration or tabu[j, i] > iteration
                
                delta = two_opt_delta(p, i, j)
                
                # Aspiration: accept if it improves global best even if tabu
                if delta < 0 or (is_tabu and current_cost + delta < best_cost):
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            
            if best_i >= 0:
                # Perform the best move found
                u = p[best_i]
                v = p[best_j]
                p[best_i], p[best_j] = v, u
                current_cost += best_delta
                
                # Update tabu
                tabu[best_i, best_j] = iteration + tabu_tenure
                tabu[best_j, best_i] = iteration + tabu_tenure
                
                # Update best
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                    consecutive_no_improve = 0
                else:
                    consecutive_no_improve += 1
                
                improved = True
            else:
                consecutive_no_improve += 1
            
            iteration += 1
            
            if not improved or consecutive_no_improve >= max_consecutive_no_improve:
                # Local optimum or stuck - Diversification: Random kick
                if time.time() > tabu_end_time - 5.0:
                    break
                
                # Choose multiple random moves for stronger diversification
                num_kicks = random.randint(2, 5)
                for _ in range(num_kicks):
                    i_kick = random.randint(0, n - 1)
                    j_kick = random.randint(0, n - 1)
                    if i_kick != j_kick:
                        delta = two_opt_delta(p, i_kick, j_kick)
                        v = p[j_kick]
                        p[i_kick], p[j_kick] = v, p[i_kick]
                        current_cost += delta
                        
                        # Update tabu
                        tabu[i_kick, j_kick] = iteration + tabu_tenure
                        tabu[j_kick, i_kick] = iteration + tabu_tenure
                
                # Polish after kick
                polish_end = min(tabu_end_time - 2.0, time.time() + 3.0)
                p, current_cost = two_opt_polish(p, max_time=polish_end)
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                
                consecutive_no_improve = 0

    # --- Phase 3: Hungarian Reassignment Loop with Enhanced Polish ---
    # Use remaining time for global reassignment
    # This allows escaping local minima found by Tabu/2-opt
    
    hungarian_count = 0
    no_improve_rounds = 0
    max_no_improve = 4
    
    # Start from the best solution found so far
    p = best_p.copy()
    current_cost = best_cost
    
    while time.time() < deadline - 15.0:
        # Compute linearized cost and solve assignment
        cost_mat = linearized_cost_matrix(p)
        
        # Add structured perturbation to break ties and escape plateaus
        if hungarian_count > 0 and no_improve_rounds > 0:
            # Use log-sum-exp inspired smooth perturbation
            noise_scale = 0.5 + no_improve_rounds * 0.3
            # Add noise proportional to current cost magnitude
            cost_mag = np.abs(cost_mat).mean()
            noise = np.random.randn(n, n) * (noise_scale * cost_mag * 0.01)
            cost_mat = cost_mat + noise
        
        try:
            row_ind, col_ind = linear_sum_assignment(cost_mat)
        except Exception:
            break
            
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        
        new_cost = calc_cost_full(new_p)
        
        if new_cost < best_cost:
            p = new_p
            best_p = p.copy()
            best_cost = new_cost
            hungarian_count += 1
            no_improve_rounds = 0
            
            # Polish after a good Hungarian step with tighter convergence
            polish_end = min(deadline - 10.0, time.time() + 8.0)
            best_track = [best_cost, best_p.copy()]
            p, p_cost = two_opt_polish(p, max_time=polish_end, best_cost_ref=best_track)
            if best_track[0] < best_cost:
                best_cost = best_track[0]
                best_p = best_track[1].copy()
        else:
            no_improve_rounds += 1
            
            # If stuck, try a perturbed restart from best
            if no_improve_rounds >= max_no_improve:
                p = best_p.copy()
                # Randomly swap a few pairs to escape local minimum
                num_swaps = random.randint(4, 12)
                for _ in range(num_swaps):
                    i_s = random.randint(0, n - 1)
                    j_s = random.randint(0, n - 1)
                    if i_s != j_s:
                        p[i_s], p[j_s] = p[j_s], p[i_s]
                
                # Polish the perturbed solution with more time
                polish_end = min(deadline - 10.0, time.time() + 6.0)
                best_track = [best_cost, best_p.copy()]
                p, p_cost = two_opt_polish(p, max_time=polish_end, best_cost_ref=best_track)
                
                if best_track[0] < best_cost:
                    best_cost = best_track[0]
                    best_p = best_track[1].copy()
                    no_improve_rounds = 0
                else:
                    # Reset to best if perturbation didn't help
                    p = best_p.copy()
                    no_improve_rounds = 0 # Reset counter to try again

    # Final high-precision polish pass
    # Use remaining time for a final intensive local search
    final_polish_time = max(0, deadline - time.time() - 3.0)
    if final_polish_time > 2.0:
        p = best_p.copy()
        best_track = [best_cost, best_p.copy()]
        p, final_cost = two_opt_polish(p, max_time=time.time() + final_polish_time, best_cost_ref=best_track)
        if best_track[0] < best_cost:
            best_cost = best_track[0]
            best_p = best_track[1].copy()
    
    # Final safety check: ensure best_p is a valid permutation
    if sorted(best_p.tolist()) != list(range(n)):
        valid_p = np.arange(n, dtype=np.int64)
        return valid_p.tolist()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
