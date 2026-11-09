"""Capacitated facility location: Simulated Annealing with vectorized moves."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def assign(demands, capacities, open_cost, assign_cost):
    n_cust = len(demands)
    n_fac = len(capacities)
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    deadline = time.time() + 145.0

    def get_greedy():
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        order = np.argsort(-demands)
        for i in order:
            d_i = demands[i]
            feasible = np.where(load + d_i <= capacities)[0]
            if len(feasible) == 0:
                f = int(np.argmax(capacities - load))
                res[i] = f
                load[f] += d_i
            else:
                costs = assign_cost[i, feasible]
                f_idx = int(feasible[np.argmin(costs)])
                res[i] = f_idx
                load[f_idx] += d_i
        return res

    best_res = get_greedy()
    best_load = np.zeros(n_fac)
    for i in range(n_cust):
        best_load[best_res[i]] += demands[i]
    
    def calc_score(res, load):
        used = np.unique(res)
        return np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_cust), res])

    best_score = calc_score(best_res, best_load)

    def single_move_pass(res, load, temp):
        # Vectorized evaluation of all possible single moves
        # For each customer i, try moving to each facility j != res[i]
        # Feasibility: load[j] + demands[i] <= capacities[j]
        
        # We process customers in random order to break symmetry
        order = np.random.permutation(n_cust)
        
        # Precompute current assignment costs
        current_costs = assign_cost[np.arange(n_cust), res]
        
        # We will update res and load as we go, so we can't fully vectorize the whole pass
        # but we can vectorize the candidate evaluation for each customer
        for i in order:
            f_old = res[i]
            d_i = demands[i]
            
            # Feasible facilities
            feasible_mask = (load + d_i <= capacities)
            feasible_mask[f_old] = False # Cannot move to self
            
            if not np.any(feasible_mask):
                continue
                
            # Cost change for moving i to j
            # delta_assign = assign_cost[i, j] - assign_cost[i, f_old]
            # delta_open:
            #   if load[f_old] == d_i, f_old closes -> -open_cost[f_old]
            #   if load[j] == 0 and j != f_old, j opens -> +open_cost[j]
            
            delta_assign = assign_cost[i, :] - current_costs[i]
            
            delta_open = np.zeros(n_fac)
            if load[f_old] == d_i:
                delta_open[f_old] = -open_cost[f_old]
            # Note: if we move to j and load[j] was 0, we add open_cost[j].
            # But if we close f_old, and j was 0, we add open_cost[j].
            # If we don't close f_old, and j was 0, we add open_cost[j].
            # So delta_open[j] = open_cost[j] if load[j]==0 else 0.
            # However, we must be careful: if f_old closes, does it affect j? No.
            # So delta_open[j] = open_cost[j] * (load[j] == 0)
            # But wait, if j == f_old, we already excluded it.
            
            # Actually, the open cost change is:
            # New Open Set = Old Open Set - {f_old if load[f_old]==d_i} + {j if load[j]==0}
            # Delta = (open_cost[j] if load[j]==0 else 0) - (open_cost[f_old] if load[f_old]==d_i else 0)
            
            open_delta = np.where(load == 0, open_cost, 0.0)
            if load[f_old] == d_i:
                open_delta[f_old] -= open_cost[f_old] # This is handled by the "old" part?
                # Let's re-calculate delta_open properly.
                # Old open cost contribution: sum(open_cost[k] for k in Open)
                # New open cost contribution: sum(open_cost[k] for k in NewOpen)
                # Delta = New - Old
                # If f_old closes: -open_cost[f_old]
                # If j opens: +open_cost[j] (if it was closed)
                
                # So for a specific j:
                # delta_open_j = (open_cost[j] if load[j]==0 else 0) - (open_cost[f_old] if load[f_old]==d_i else 0)
                
                # Let's compute this vectorized for all j
                base_open_delta = np.where(load == 0, open_cost, 0.0)
                if load[f_old] == d_i:
                    base_open_delta -= open_cost[f_old]
            else:
                base_open_delta = np.where(load == 0, open_cost, 0.0)
                # No facility closes, so no subtraction
                
            # But wait, if j is the one we move to, and load[j] was 0, we add open_cost[j].
            # The vector `base_open_delta` already has open_cost[j] at index j if load[j]==0.
            # And if f_old closes, we subtract open_cost[f_old] from ALL j?
            # No, the closing of f_old is a global effect on the total cost, independent of which j we pick?
            # Yes, if f_old becomes empty, it closes. This happens regardless of where i moves (as long as i leaves f_old).
            # So the term `-open_cost[f_old]` (if closing) is constant for all j.
            # The term `+open_cost[j]` (if opening) depends on j.
            
            # So total_delta[j] = delta_assign[j] + base_open_delta[j]
            # Where base_open_delta[j] = (open_cost[j] if load[j]==0 else 0) - (open_cost[f_old] if load[f_old]==d_i else 0)
            
            # Let's refine:
            open_part = np.where(load == 0, open_cost, 0.0)
            if load[f_old] == d_i:
                open_part -= open_cost[f_old]
            
            total_delta = delta_assign + open_part
            total_delta[~feasible_mask] = np.inf
            total_delta[f_old] = np.inf
            
            if np.all(np.isinf(total_delta)):
                continue
                
            best_j = int(np.argmin(total_delta))
            best_delta = total_delta[best_j]
            
            # Simulated Annealing Acceptance
            if best_delta < 0:
                accept = True
            else:
                if temp > 1e-9:
                    accept = np.random.rand() < np.exp(-best_delta / temp)
                else:
                    accept = False
            
            if accept:
                res[i] = best_j
                load[f_old] -= d_i
                load[best_j] += d_i
                current_costs[i] = assign_cost[i, best_j]
                return True
        return False

    def swap_pass(res, load, temp):
        # Swap two customers i, k assigned to f_i, f_k
        # Feasibility:
        #   If f_i == f_k: no change in load, always feasible if valid indices.
        #   If f_i != f_k:
        #     New load f_i = load[f_i] - d_i + d_k <= cap[f_i]
        #     New load f_k = load[f_k] - d_k + d_i <= cap[f_k]
        
        # We will try a random number of swaps
        n_swaps = 50
        improved = False
        for _ in range(n_swaps):
            i, k = np.random.choice(n_cust, 2, replace=False)
            f_i = res[i]
            f_k = res[k]
            d_i = demands[i]
            d_k = demands[k]
            
            if f_i == f_k:
                # Swap doesn't change assignment effectively, skip
                continue
                
            # Check feasibility
            new_load_i = load[f_i] - d_i + d_k
            new_load_k = load[f_k] - d_k + d_i
            if new_load_i > capacities[f_i] or new_load_k > capacities[f_k]:
                continue
                
            # Calculate cost delta
            # Assign cost change:
            # Old: assign_cost[i, f_i] + assign_cost[k, f_k]
            # New: assign_cost[i, f_k] + assign_cost[k, f_i]
            delta_assign = (assign_cost[i, f_k] + assign_cost[k, f_i]) - (assign_cost[i, f_i] + assign_cost[k, f_k])
            
            # Open cost change
            # Check if f_i or f_k become empty or non-empty
            # f_i becomes empty if load[f_i] == d_i (and d_k != 0, which is true)
            # But wait, if we swap, f_i loses d_i and gains d_k.
            # So f_i becomes empty ONLY IF d_i == load[f_i] AND d_k == 0? No, d_k > 0.
            # So f_i never becomes empty if we swap with k != i?
            # Wait, if load[f_i] == d_i, then after swap load[f_i] = d_k. Since d_k > 0, f_i stays open.
            # Similarly, f_k stays open.
            # So open cost does NOT change in a swap between two customers at different facilities?
            # Unless... one of the facilities was open ONLY because of that customer?
            # If load[f_i] == d_i, f_i is open. After swap, load[f_i] = d_k. Still open.
            # So open cost is invariant under swaps between distinct facilities?
            # Yes, because both facilities remain non-empty.
            
            delta_total = delta_assign
            
            if delta_total < 0:
                accept = True
            else:
                if temp > 1e-9:
                    accept = np.random.rand() < np.exp(-delta_total / temp)
                else:
                    accept = False
                    
            if accept:
                res[i], res[k] = res[k], res[i]
                load[f_i] = new_load_i
                load[f_k] = new_load_k
                improved = True
        return improved

    # Simulated Annealing Loop
    T_start = 100.0
    T_end = 0.1
    alpha = 0.9995
    
    current_res = best_res.copy()
    current_load = best_load.copy()
    current_score = best_score
    
    T = T_start
    step = 0
    while time.time() < deadline - 5.0:
        step += 1
        # Alternate between single moves and swaps
        if step % 3 == 0:
            improved = swap_pass(current_res, current_load, T)
        else:
            improved = single_move_pass(current_res, current_load, T)
            
        # Recalculate score periodically or if improved
        # Actually, single_move_pass updates score implicitly? No, it returns bool.
        # We should track score carefully.
        # For simplicity, recalculate every N steps or if we suspect change.
        # Given vectorization, recalc is fast (100 ops).
        if improved:
            current_score = calc_score(current_res, current_load)
            if current_score < best_score:
                best_score = current_score
                best_res = current_res.copy()
                best_load = current_load.copy()
        
        # Cooling
        T *= alpha
        if T < T_end:
            T = T_end
            
        # Restart logic: if stuck for a long time, restart with a perturbed best solution?
        # For now, just continue.
        
        # Safety check: ensure validity
        # If load exceeds capacity, revert? 
        # The moves check feasibility, so it should be safe.
        
    # Final polish with deterministic local search (greedy improvement)
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i in range(n_cust):
            f_old = best_res[i]
            d_i = demands[i]
            feasible_mask = (best_load + d_i <= capacities)
            feasible_mask = feasible_mask & (np.arange(n_fac) != f_old)
            if not np.any(feasible_mask):
                continue
            delta_assign = assign_cost[i, :] - assign_cost[i, f_old]
            open_part = np.where(best_load == 0, open_cost, 0.0)
            if best_load[f_old] == d_i:
                open_part -= open_cost[f_old]
            total_delta = delta_assign + open_part
            total_delta[~feasible_mask] = np.inf
            total_delta[f_old] = np.inf
            best_f_new = int(np.argmin(total_delta))
            best_delta = total_delta[best_f_new]
            if best_delta < -1e-9:
                best_res[i] = best_f_new
                best_load[f_old] -= d_i
                best_load[best_f_new] += d_i
                improved = True

    return best_res
# EVOLVE-BLOCK-END
