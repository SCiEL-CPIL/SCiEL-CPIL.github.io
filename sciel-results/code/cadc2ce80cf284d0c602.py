"""
Structural Change: Multi-Start Greedy Construction with Aggressive Local Search.

Instead of Lagrangian relaxation, we use:
1. Multiple diverse greedy construction heuristics (different customer orderings)
2. A powerful local search with 1-moves, 2-swaps, and 2-opt style exchanges
3. Vectorized cost evaluation where possible for speed

The key insight is that for capacitated facility location with 100 customers and 20 facilities,
a good initial solution combined with thorough local search can outperform sophisticated
relaxation methods, especially when we try multiple construction strategies.
"""

import numpy as np
import time

DEADLINE = 115.0  # Safety margin before 120s limit


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


def _greedy_construct(demands, capacities, assign_cost, order):
    """
    Greedy construction: assign customers in the given order to the cheapest
    feasible facility.
    """
    n_c = len(demands)
    n_f = len(capacities)
    
    assignment = np.full(n_c, -1, dtype=int)
    loads = np.zeros(n_f)
    
    for i in order:
        d = demands[i]
        # Find feasible facilities (sorted by cost for this customer)
        costs = assign_cost[i, :]
        # Get facilities sorted by cost
        sorted_f = np.argsort(costs)
        
        assigned = False
        for f in sorted_f:
            if loads[f] + d <= capacities[f] + 1e-9:
                assignment[i] = f
                loads[f] += d
                assigned = True
                break
        
        if not assigned:
            # Fallback: assign to facility with most remaining capacity
            best_f = np.argmax(capacities - loads)
            assignment[i] = best_f
            loads[best_f] += d
    
    return assignment


def _local_search(assignment, demands, capacities, open_cost, assign_cost, deadline):
    """
    Aggressive local search with 1-moves and 2-swaps.
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
    current_costs = assign_cost[np.arange(n_c), assignment]
    
    max_rounds = 50
    for round_num in range(max_rounds):
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
        if time.time() >= deadline:
            break
            
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
    
    # Generate multiple diverse orderings for greedy construction
    orderings = []
    
    # 1. Natural order
    orderings.append(np.arange(n_c))
    
    # 2. Random order (seeded for reproducibility within a call)
    rng = np.random.default_rng(42)
    for _ in range(5):
        orderings.append(rng.permutation(n_c))
    
    # 3. Order by demand (descending - large first)
    orderings.append(np.argsort(-demands))
    
    # 4. Order by demand (ascending - small first)
    orderings.append(np.argsort(demands))
    
    # 5. Order by sum of assignment costs (cheapest to assign first)
    sum_costs = assign_cost.sum(axis=1)
    orderings.append(np.argsort(sum_costs))
    
    # 6. Order by sum of assignment costs (most expensive to assign first)
    orderings.append(np.argsort(-sum_costs))
    
    # 7. Order by "regret" (difference between best and second-best cost)
    sorted_costs = np.sort(assign_cost, axis=1)
    regrets = sorted_costs[:, 1] - sorted_costs[:, 0]
    orderings.append(np.argsort(-regrets))
    
    # 8. Order by "regret" ascending
    orderings.append(np.argsort(regrets))
    
    # Try each ordering, keep the best
    best_assign = None
    best_cost = np.inf
    
    for order in orderings:
        if time.time() - t0 > DEADLINE - 30:
            break
            
        candidate = _greedy_construct(demands, capacities, assign_cost, order)
        
        if not _valid(candidate, demands, capacities):
            continue
        
        cost = _total_cost(candidate, demands, capacities, open_cost, assign_cost)
        
        if cost < best_cost:
            best_cost = cost
            best_assign = candidate.copy()
    
    # If no valid assignment was found (shouldn't happen), use a fallback
    if best_assign is None:
        best_assign = _greedy_construct(demands, capacities, assign_cost, np.arange(n_c))
        best_cost = _total_cost(best_assign, demands, capacities, open_cost, assign_cost)
    
    # Polish with local search
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 5:
        deadline_for_ls = t0 + DEADLINE - 2
        polished = _local_search(best_assign, demands, capacities, open_cost, assign_cost, deadline_for_ls)
        if _valid(polished, demands, capacities):
            polished_cost = _total_cost(polished, demands, capacities, open_cost, assign_cost)
            if polished_cost < best_cost:
                best_assign = polished
                best_cost = polished_cost
    
    # If we have time, try more diverse orderings and local search
    if time.time() - t0 < DEADLINE - 20:
        # Generate more random orderings
        rng2 = np.random.default_rng(123)
        for _ in range(10):
            if time.time() - t0 > DEADLINE - 10:
                break
                
            order = rng2.permutation(n_c)
            candidate = _greedy_construct(demands, capacities, assign_cost, order)
            
            if not _valid(candidate, demands, capacities):
                continue
            
            cost = _total_cost(candidate, demands, capacities, open_cost, assign_cost)
            
            if cost < best_cost:
                best_cost = cost
                best_assign = candidate.copy()
                
                # Polish the new best
                if time.time() - t0 < DEADLINE - 5:
                    deadline_for_ls = t0 + DEADLINE - 2
                    polished = _local_search(best_assign, demands, capacities, open_cost, assign_cost, deadline_for_ls)
                    if _valid(polished, demands, capacities):
                        polished_cost = _total_cost(polished, demands, capacities, open_cost, assign_cost)
                        if polished_cost < best_cost:
                            best_assign = polished
                            best_cost = polished_cost
    
    return best_assign.astype(int)
