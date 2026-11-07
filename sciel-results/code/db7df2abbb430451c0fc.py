# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility respecting capacities.
    Minimizes open_cost (for open facilities) + assign_cost.
    Strategy: Greedy init + Aggressive Local Search (single move + pair swap + 2-opt).
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    deadline = time.time() + 115.0  # safety margin
    
    # --- Step 1: Greedy Initialization ---
    # Sort customers by demand descending to place large ones first
    # For each customer, try facilities in order of cost (lowest first)
    # This gives a valid starting point usually.
    
    def _greedy_init(demands, capacities, assign_cost, open_cost):
        n_cust = len(demands)
        n_fac = len(capacities)
        result = np.full(n_cust, -1, dtype=np.int64)
        load = np.zeros(n_fac, dtype=np.int64)
        
        # Order customers by demand (largest first) to ensure capacity is reserved for big ones
        order = np.argsort(-demands)
        
        # Pre-compute facility order for each customer based on cost
        # To speed up, we can just iterate facilities sorted by cost for that customer
        # But since n_fac is small (20), sorting once per customer is fine.
        fac_order = np.argsort(assign_cost, axis=1)
        
        for i in order:
            assigned = False
            # Try facilities in order of increasing cost
            for f in fac_order[i]:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            if not assigned:
                # If no space, pick the facility with most remaining capacity (or just first with space)
                # This shouldn't happen if total capacity >= total demand
                # But just in case, find any valid facility
                for f in range(n_fac):
                    if load[f] + demands[i] <= capacities[f]:
                        load[f] += demands[i]
                        result[i] = f
                        assigned = True
                        break
                if not assigned:
                    # Fallback: assign to facility with max remaining capacity
                    remaining = capacities - load
                    f_best = np.argmax(remaining)
                    result[i] = f_best
                    load[f_best] += demands[i]
        return result, load

    result, load = _greedy_init(demands, capacities, assign_cost, open_cost)
    
    # --- Step 2: Local Search ---
    # Objective: Sum(assign_cost[i, f_i]) + Sum(open_cost[f] for f in used facilities)
    # We will perform iterative improvement using:
    # 1. Single moves (move customer i to facility j)
    # 2. Pair swaps (swap customers i and j)
    
    # Helper to compute objective
    def _objective(result, load, demands, capacities, open_cost, assign_cost):
        cost = np.sum(assign_cost[np.arange(len(result)), result])
        used = set(result.tolist())
        for f in used:
            cost += open_cost[f]
        return cost

    # We don't need to explicitly track load if we recompute or update carefully.
    # Let's track load for O(1) capacity checks.
    
    def _local_search(result, load, demands, capacities, open_cost, assign_cost, deadline):
        n_cust = len(demands)
        n_fac = len(capacities)
        
        # Precompute assignment costs
        ac = assign_cost
        
        # Current objective
        used_facs = set(result.tolist())
        cur_obj = np.sum(ac[np.arange(n_cust), result]) + sum(open_cost[f] for f in used_facs)
        
        # Precompute facility groups for fast pair swap access
        def _get_groups(result):
            groups = {f: [] for f in range(n_fac)}
            for i in range(n_cust):
                groups[result[i]].append(i)
            return groups
        
        groups = _get_groups(result)
        
        max_rounds = 200  # High enough to converge, but we have time guard
        for rnd in range(max_rounds):
            if time.time() > deadline - 5:
                break
            
            improved = False
            
            # --- Phase 1: Single Moves ---
            # For each customer i, check all other facilities j
            # Vectorized delta calculation:
            # delta(i, j) = ac[i, j] - ac[i, f_i] + open_cost_delta(i, j)
            # open_cost_delta:
            #   If j is not currently open: +open_cost[j]
            #   If i is the last customer at f_i: -open_cost[f_i]
            #   If j == f_i: 0 (but we skip j == f_i)
            
            cur_fac = result
            ac_cur = ac[np.arange(n_cust), cur_fac]
            
            # Cost to move i to j (assignment part)
            delta_assign = ac - ac_cur[:, np.newaxis]
            
            # Open cost part
            # Term 1: +open_cost[j] if j is currently unused
            # Term 2: -open_cost[f_i] if i is the last customer at f_i
            
            # Is facility j unused?
            used_mask = np.zeros(n_fac, dtype=bool)
            for f in range(n_fac):
                if load[f] > 0:
                    used_mask[f] = True
            
            # For each i, if we move i to j:
            # j becomes used if it wasn't.
            # f_i becomes unused if i was the only one there.
            
            # last_at_f_i: True if load[f_i] == demands[i]
            # Note: this is only true if demands[i] > 0.
            
            is_last = (load[cur_fac] == demands)
            
            # Delta open cost matrix: (n_cust, n_fac)
            # For customer i moving to facility j:
            # gain_open = open_cost[j] if not used_mask[j] else 0
            # loss_open = open_cost[cur_fac[i]] if is_last[i] else 0
            
            open_gain = open_cost.copy()
            open_gain[used_mask] = 0.0  # If already used, no extra cost
            
            open_loss = np.zeros(n_cust)
            open_loss[is_last] = open_cost[cur_fac[is_last]]
            
            # delta_open[i, j] = open_gain[j] - (open_loss[i] if j != cur_fac[i] else 0)
            # Actually, if j == cur_fac[i], no move happens, so we ignore.
            
            delta_open = open_gain[np.newaxis, :] - open_loss[:, np.newaxis]
            # If j == cur_fac[i], delta_open should be 0, but we mask it out later.
            # However, if is_last[i] is true and we move to j != f_i, we lose open_cost[f_i].
            # If j is unused, we gain open_cost[j].
            # So delta_open[i, j] = open_gain[j] - (open_loss[i] if j != f_i else 0)
            # Wait, if j == f_i, we don't move, so it's not a candidate.
            
            total_delta = delta_assign + delta_open
            
            # Mask invalid moves:
            # 1. j == cur_fac[i]
            # 2. load[j] + demands[i] > capacities[j]
            
            infeasible = (load[np.newaxis, :] + demands[:, np.newaxis]) > capacities[np.newaxis, :]
            total_delta[infeasible] = np.inf
            
            # Also mask j == cur_fac[i]
            # We can just set diagonal to inf if we were comparing to same, but cur_fac is not diagonal in this matrix.
            # cur_fac[i] is the column index for row i that is current.
            total_delta[np.arange(n_cust), cur_fac] = np.inf
            
            # Find best move for each customer
            best_f_new = np.argmin(total_delta, axis=1)
            best_delta = np.min(total_delta, axis=1)
            
            # Apply moves in order of most improvement (lowest delta)
            # We must be careful with simultaneous moves affecting capacity.
            # Simplest: apply one move at a time, recompute? Or apply sequentially and check capacity.
            
            # Sort customers by delta (most negative first)
            order = np.argsort(best_delta)
            
            # We will apply up to a certain number of moves in this round, or all non-overlapping?
            # Let's apply the single best move first, then re-evaluate? 
            # Or apply all valid non-conflicting moves?
            # Conflicting moves: moving i to j and k to j might exceed capacity.
            # To be safe and fast, let's apply moves greedily one by one, checking capacity each time.
            
            applied = False
            for i in order:
                if time.time() > deadline - 5:
                    break
                delta = best_delta[i]
                if delta >= -1e-9:
                    break  # No more improving moves for this customer
            
                f_old = result[i]
                f_new = best_f_new[i]
                
                if load[f_new] + demands[i] <= capacities[f_new]:
                    # Apply move
                    result[i] = f_new
                    load[f_old] -= demands[i]
                    load[f_new] += demands[i]
                    
                    # Update groups
                    groups[f_old].remove(i)
                    groups[f_new].append(i)
                    
                    # Update open_cost state
                    if load[f_old] == 0:
                        used_mask[f_old] = False
                    if load[f_new] > 0:
                        used_mask[f_new] = True
                    
                    cur_obj += delta
                    applied = True
                    # Since state changed, delta for other customers might be invalid.
                    # For simplicity in this loop, we just break and restart round?
                    # Or continue? If we continue, subsequent best_delta values might be stale.
                    # Let's break and restart the round to recalculate.
                    # Actually, applying one move and restarting is safer and often effective.
                    # But it might be slow if many moves.
                    # Let's apply all "safe" moves where we don't conflict?
                    # Too complex. Let's just apply one best move and restart.
                    break
            
            if not applied:
                # Try Pair Swaps
                # Iterate over pairs of facilities (fi, fj)
                # For each pair of customers (i in fi, j in fj)
                # Check if swap is feasible and improves cost
                
                found_swap = False
                for fi in range(n_fac):
                    if time.time() > deadline - 5:
                        break
                    if not groups[fi]:
                        continue
                    for fj in range(fi + 1, n_fac):
                        if time.time() > deadline - 5:
                            break
                        if not groups[fj]:
                            continue
                        
                        # Check capacity feasibility for any swap
                        # Max possible load change:
                        # load[fi] - min_demands_in_fi + max_demands_in_fj <= capacities[fi]
                        # This is a necessary condition, not sufficient.
                        
                        # Iterate pairs
                        for i in groups[fi]:
                            if time.time() > deadline - 5:
                                break
                            for j in groups[fj]:
                                # Check capacity
                                new_load_fi = load[fi] - demands[i] + demands[j]
                                new_load_fj = load[fj] - demands[j] + demands[i]
                                
                                if new_load_fi > capacities[fi] or new_load_fj > capacities[fj]:
                                    continue
                                
                                # Calculate delta
                                # Old: ac[i, fi] + ac[j, fj]
                                # New: ac[i, fj] + ac[j, fi]
                                # Open cost:
                                # If fi becomes empty (only i was there) and fj becomes empty (only j was there), etc.
                                # open_cost_delta:
                                #   If len(groups[fi]) == 1 and len(groups[fj]) == 1:
                                #       fi closes, fj closes. Delta = 0 (both lose cost) -> actually -open_cost[fi] - open_cost[fj] + 0?
                                #       Wait, if they swap, fi still has j, fj still has i. So neither closes.
                                #       Open cost remains same.
                                #   If len(groups[fi]) == 1: fi becomes empty (i leaves), but j arrives. So fi stays open (j is there).
                                #       Unless j arrives and i leaves, but j is already in fj.
                                #       So if len(groups[fi]) == 1, after swap, fi has j (1 cust). So fi stays open.
                                #       fj has i (1 cust). So fj stays open.
                                #       Open cost is unchanged in all swap cases where both facilities remain non-empty.
                                #       What if a facility becomes empty?
                                #       Swap i (from fi) and j (from fj).
                                #       fi loses i, gains j. If i was the only one, fi now has j. So fi is NOT empty.
                                #       So open cost NEVER changes in a swap!
                                #       Because both facilities gain one customer and lose one.
                                #       Exception: if i and j are the only customers, and they swap?
                                #       fi: had {i}, now {j}. Not empty.
                                #       So open cost is constant for swaps.
                                
                                delta = ac[i, fj] + ac[j, fi] - ac[i, fi] - ac[j, fj]
                                
                                if delta < -1e-9:
                                    # Apply swap
                                    result[i] = fj
                                    result[j] = fi
                                    
                                    load[fi] = new_load_fi
                                    load[fj] = new_load_fj
                                    
                                    # Update groups
                                    groups[fi].remove(i)
                                    groups[fi].append(j)
                                    groups[fj].remove(j)
                                    groups[fj].append(i)
                                    
                                    cur_obj += delta
                                    found_swap = True
                                    break
                            if found_swap:
                                break
                        if found_swap:
                            break
                    if found_swap:
                        break
                
                if found_swap:
                    improved = True
                    continue # Restart round
            
            # If no single move and no swap found, we are at local optimum
            if not applied and not improved:
                break
        
        return result, load

    result, load = _local_search(result, load, demands, capacities, open_cost, assign_cost, deadline)
    
    # Final validity check
    # Ensure all customers assigned
    if -1 in result:
        # Should not happen, but fix
        for i in range(len(result)):
            if result[i] == -1:
                for f in range(n_fac):
                    if load[f] + demands[i] <= capacities[f]:
                        result[i] = f
                        load[f] += demands[i]
                        break
    
    return result.astype(int)

# EVOLVE-BLOCK-END
