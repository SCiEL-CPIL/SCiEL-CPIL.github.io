"""
Structural Change: Multi-Start Greedy Construction with Aggressive Local Search.

This approach uses multiple restarts with different initialization strategies to
explore diverse regions of the solution space. Each restart starts with a greedy
construction that assigns customers to facilities in a way that minimizes immediate
cost while respecting capacity constraints. The solution is then refined using
aggressive local search with 1-moves and 2-swaps.

Key structural ideas:
1. Multiple restarts with different initialization strategies (random, cost-based, etc.)
2. Greedy construction that assigns customers to the best feasible facility
3. Aggressive local search with 1-moves and 2-swaps to refine the solution
4. Time guard to ensure we return the best solution found before the deadline
"""

import numpy as np
import time

DEADLINE = 145.0  # Safety margin before 150s kill


def _total_cost(assignment, demands, capacities, open_cost, assign_cost):
    """Calculate the total cost of an assignment."""
    used = np.zeros(len(capacities), dtype=bool)
    for f in assignment:
        used[f] = True
    open_sum = float(open_cost[used].sum())
    cust = np.arange(len(assignment))
    assign_sum = float(assign_cost[cust, assignment].sum())
    return open_sum + assign_sum


def _valid(assignment, demands, capacities):
    """Check if assignment is valid (integer indices, capacity constraints)."""
    if len(assignment) != len(demands):
        return False
    if np.any(assignment < 0) or np.any(assignment >= len(capacities)):
        return False
    if np.any(np.asarray(assignment, dtype=float) % 1 != 0):
        return False
    loads = np.zeros(len(capacities))
    np.add.at(loads, assignment, demands)
    return bool(np.all(loads <= capacities + 1e-9))


def _greedy_construct(demands, capacities, open_cost, assign_cost, init_type=0):
    """
    Greedy construction that assigns customers to the best feasible facility.
    Different initialization strategies:
    - init_type=0: Assign in order of increasing demand
    - init_type=1: Assign in order of decreasing demand
    - init_type=2: Random order
    - init_type=3: Assign based on lowest cost to any facility
    """
    n_c = len(demands)
    n_f = len(capacities)
    assignment = np.full(n_c, -1, dtype=int)
    loads = np.zeros(n_f)
    
    if init_type == 0:
        order = np.argsort(demands)
    elif init_type == 1:
        order = np.argsort(-demands)
    elif init_type == 2:
        order = np.random.permutation(n_c)
    elif init_type == 3:
        # Assign based on lowest cost to any facility
        min_costs = np.min(assign_cost, axis=1)
        order = np.argsort(min_costs)
    else:
        order = np.random.permutation(n_c)
    
    for i in order:
        d = demands[i]
        # Find feasible facilities
        feasible = loads + d <= capacities + 1e-9
        if not np.any(feasible):
            # Fallback: pick facility with most remaining space
            remaining = capacities - loads
            best_f = np.argmax(remaining)
            assignment[i] = best_f
            loads[best_f] += d
            continue
        
        # Among feasible facilities, pick the one with lowest total cost
        # (assign_cost + open_cost if facility is not yet used)
        in_use = loads > 0
        costs = assign_cost[i, :].copy()
        costs[~in_use] += open_cost[~in_use]
        costs[~feasible] = np.inf
        
        best_f = np.argmin(costs)
        assignment[i] = best_f
        loads[best_f] += d
    
    return assignment


def _local_search(assignment, demands, capacities, open_cost, assign_cost, deadline):
    """
    Aggressive local search with 1-moves and 2-swaps.
    Uses vectorized operations where possible to speed up.
    """
    n_c = len(assignment)
    n_f = len(capacities)
    
    if n_c == 0:
        return assignment
        
    assignment = assignment.copy()
    loads = np.zeros(n_f)
    np.add.at(loads, assignment, demands)
    
    in_use = np.zeros(n_f, dtype=bool)
    for f in assignment:
        in_use[f] = True
        
    t_start = time.time()
    
    # Precompute current costs for each customer
    current_costs = assign_cost[np.arange(n_c), assignment].copy()
    
    max_passes = 100
    for pass_num in range(max_passes):
        if time.time() >= deadline:
            break
            
        improved = False
        
        # --- 1-Move Search ---
        # For each customer, find the best alternative facility
        for i in range(n_c):
            if time.time() >= deadline:
                break
                
            cur_f = assignment[i]
            d = demands[i]
            cur_cost = current_costs[i]
            
            # Calculate cost delta for moving i to each f
            new_costs = assign_cost[i, :]
            
            # Open cost changes
            open_deltas = np.zeros(n_f)
            # If f is not in use, we pay open_cost[f]
            open_deltas[~in_use] = open_cost[~in_use]
            # If cur_f becomes unused, we save open_cost[cur_f]
            if loads[cur_f] == d:
                open_deltas -= open_cost[cur_f]
            
            total_deltas = new_costs - cur_cost + open_deltas
            
            # Mask infeasible facilities
            feasible = loads + d <= capacities + 1e-9
            feasible = feasible & (np.arange(n_f) != cur_f)
            
            if not np.any(feasible):
                continue
                
            total_deltas[~feasible] = np.inf
            
            best_f = np.argmin(total_deltas)
            gain = -total_deltas[best_f]
            
            if gain > 1e-9:
                # Perform move
                loads[cur_f] -= d
                loads[best_f] += d
                assignment[i] = best_f
                current_costs[i] = assign_cost[i, best_f]
                
                # Update in_use
                in_use[best_f] = True
                if loads[cur_f] == 0:
                    in_use[cur_f] = False
                    
                improved = True
        
        # --- 2-Swap Search ---
        if time.time() < deadline:
            for i in range(n_c):
                if time.time() >= deadline:
                    break
                fi = assignment[i]
                di = demands[i]
                ci = current_costs[i]
                
                for k in range(i + 1, n_c):
                    fk = assignment[k]
                    if fi == fk:
                        continue
                    dk = demands[k]
                    ck = current_costs[k]
                    
                    # Check capacity after swap
                    if loads[fi] - di + dk > capacities[fi] + 1e-9:
                        continue
                    if loads[fk] - dk + di > capacities[fk] + 1e-9:
                        continue
                    
                    # Calculate cost change
                    new_ci = assign_cost[i, fk]
                    new_ck = assign_cost[k, fi]
                    
                    delta_assign = (new_ci + new_ck) - (ci + ck)
                    
                    if delta_assign < -1e-9:
                        # Perform swap
                        loads[fi] = loads[fi] - di + dk
                        loads[fk] = loads[fk] - dk + di
                        assignment[i] = fk
                        assignment[k] = fi
                        current_costs[i] = new_ci
                        current_costs[k] = new_ck
                        
                        improved = True
                        break  # Restart i loop after a swap
                
                if improved and time.time() >= deadline:
                    break
        
        if not improved:
            break
    
    return assignment


def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_c = len(demands)
    n_f = len(capacities)
    t0 = time.time()
    
    # Track best solution found
    best_assign = None
    best_cost = np.inf
    
    # Multiple restarts with different initialization strategies
    num_restarts = 10
    init_types = [0, 1, 2, 3, 2, 0, 1, 3, 2, 1]
    
    for restart in range(num_restarts):
        if time.time() - t0 > DEADLINE - 5:
            break
            
        # Greedy construction with different initialization
        init_type = init_types[restart % len(init_types)]
        candidate = _greedy_construct(demands, capacities, open_cost, assign_cost, init_type)
        
        # Ensure validity
        if not _valid(candidate, demands, capacities):
            # Fallback to simple greedy
            candidate = np.zeros(n_c, dtype=int)
            loads = np.zeros(n_f)
            order = np.argsort(-demands)
            for i in order:
                for f in range(n_f):
                    if loads[f] + demands[i] <= capacities[f]:
                        candidate[i] = f
                        loads[f] += demands[i]
                        break
        
        candidate_cost = _total_cost(candidate, demands, capacities, open_cost, assign_cost)
        
        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_assign = candidate.copy()
        
        # Local search polish
        remaining_for_ls = min(t0 + DEADLINE - 5, t0 + 10 + restart * 5)
        if time.time() < remaining_for_ls - 2:
            polished = _local_search(candidate, demands, capacities, open_cost, 
                                      assign_cost, remaining_for_ls)
            if _valid(polished, demands, capacities):
                polished_cost = _total_cost(polished, demands, capacities, open_cost, assign_cost)
                if polished_cost < best_cost:
                    best_cost = polished_cost
                    best_assign = polished.copy()
    
    # Final polish with remaining time
    if best_assign is not None and time.time() - t0 < DEADLINE - 3:
        final_polished = _local_search(best_assign, demands, capacities, open_cost, 
                                        assign_cost, t0 + DEADLINE - 2)
        if _valid(final_polished, demands, capacities):
            final_cost = _total_cost(final_polished, demands, capacities, open_cost, assign_cost)
            if final_cost < best_cost:
                best_assign = final_polished
                best_cost = final_cost
    
    # Ensure we have a valid result
    if best_assign is None or not _valid(best_assign, demands, capacities):
        # Absolute fallback
        best_assign = np.zeros(n_c, dtype=int)
        loads = np.zeros(n_f)
        order = np.argsort(-demands)
        for i in order:
            for f in range(n_f):
                if loads[f] + demands[i] <= capacities[f]:
                    best_assign[i] = f
                    loads[f] += demands[i]
                    break
    
    return best_assign.astype(int)
