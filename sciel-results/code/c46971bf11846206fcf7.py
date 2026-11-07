"""
Structural Change: Lagrangian Relaxation with Regret-Based Rounding and Efficient Local Search.

This approach formulates the problem by relaxing capacity constraints with Lagrange multipliers.
The "LP relaxation" insight is used to guide a deterministic rounding procedure:
1. Solve the Lagrangian dual to obtain facility prices (lambda).
2. Compute adjusted costs c_ij + lambda_j.
3. Rank customers by "regret" (gap between best and second-best adjusted cost).
4. Assign customers in order of decreasing regret (most confident first).
5. If capacity is violated, resolve conflicts by moving the customer with the smallest cost increase to an alternative facility.
6. Polish with an efficient local search (1-move and 2-swap) using vectorized operations where possible.
"""

import numpy as np
import time

DEADLINE = 115.0  # Safety margin


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


def _lagrangian_dual(demands, capacities, assign_cost, max_iter=200, time_limit=10.0):
    """
    Solve the Lagrangian dual to get facility prices (lambda).
    Returns lambda array and the best dual value.
    """
    n_f = len(capacities)
    lam = np.zeros(n_f)
    step_size = 1.0
    best_dual = -np.inf
    t_start = time.time()
    
    for _ in range(max_iter):
        if time.time() - t_start > time_limit:
            break
            
        # Subproblem: assign each customer to argmin_j (c_ij + lam_j)
        adjusted_costs = assign_cost + lam[None, :]
        best_facilities = np.argmin(adjusted_costs, axis=1)
        best_costs = adjusted_costs[np.arange(len(demands)), best_facilities]
        
        # Calculate load
        loads = np.zeros(n_f)
        np.add.at(loads, best_facilities, demands)
        
        # Dual value: sum of min costs + sum(lambda * capacity)
        dual_val = float(np.sum(best_costs)) + float(np.dot(lam, capacities))
        
        if dual_val > best_dual:
            best_dual = dual_val
            
        # Subgradient: s_j = capacity_j - load_j
        subgrad = capacities - loads
        
        # Update lambda
        norm_s = np.linalg.norm(subgrad)
        if norm_s < 1e-10:
            break
            
        # Polyak-like step size or constant step with decay
        step = step_size / norm_s
        lam = np.maximum(lam + step * subgrad, 0.0)
        
        # Decay step size
        step_size *= 0.95
        
    return lam


def _regret_rounding(demands, capacities, assign_cost, lam):
    """
    Deterministic rounding procedure guided by Lagrangian prices.
    1. Calculate adjusted costs.
    2. Calculate regret for each customer (best - second best).
    3. Sort customers by regret (descending).
    4. Assign in order, respecting capacity.
    5. If capacity exceeded, move the customer with smallest cost increase to next best facility.
    """
    n_c = len(demands)
    n_f = len(capacities)
    
    adjusted_costs = assign_cost + lam[None, :]
    
    # Calculate regret: difference between best and second best cost
    # Sort costs for each customer
    sorted_costs = np.sort(adjusted_costs, axis=1)
    regrets = sorted_costs[:, 1] - sorted_costs[:, 0]
    
    # Order customers by regret (high regret first = more "integral" in LP sense)
    order = np.argsort(-regrets)
    
    assignment = np.full(n_c, -1, dtype=int)
    loads = np.zeros(n_f)
    used = np.zeros(n_f, dtype=bool)
    
    # First pass: assign in order of regret
    for i in order:
        # Find feasible facilities
        feasible = []
        for f in range(n_f):
            if loads[f] + demands[i] <= capacities[f] + 1e-9:
                feasible.append(f)
        
        if not feasible:
            # No feasible facility with current load?
            # This shouldn't happen if total demand <= total capacity, but let's be safe.
            # If it happens, we'll fix it later or pick any.
            continue
            
        # Pick the one with lowest adjusted cost among feasible
        min_cost = np.inf
        best_f = -1
        for f in feasible:
            if adjusted_costs[i, f] < min_cost:
                min_cost = adjusted_costs[i, f]
                best_f = f
                
        assignment[i] = best_f
        loads[best_f] += demands[i]
        used[best_f] = True
        
    # Second pass: handle any unassigned customers (should be none if capacity is sufficient)
    # And repair any potential issues by moving customers if needed.
    # Since we assigned greedily, we might have left some customers unassigned if capacity was tight.
    # Let's check and re-assign if necessary.
    for i in range(n_c):
        if assignment[i] == -1:
            # Find any facility with space
            for f in range(n_f):
                if loads[f] + demands[i] <= capacities[f] + 1e-9:
                    assignment[i] = f
                    loads[f] += demands[i]
                    used[f] = True
                    break
            if assignment[i] == -1:
                # Fallback: assign to facility with most space (or just 0)
                # This indicates infeasibility or bug in logic
                max_space = -1
                best_f = 0
                for f in range(n_f):
                    space = capacities[f] - loads[f]
                    if space > max_space:
                        max_space = space
                        best_f = f
                assignment[i] = best_f
                loads[best_f] += demands[i]
                used[best_f] = True

    return assignment


def _local_search(assignment, demands, capacities, open_cost, assign_cost, deadline):
    """
    Efficient local search with 1-moves and 2-swaps.
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
    current_costs = assign_cost[np.arange(n_c), assignment]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # --- 1-Move Search ---
        # For each customer, find the best alternative facility
        # We can vectorize the cost calculation, but capacity check is tricky.
        # We'll iterate customers but use NumPy for finding best facility.
        
        for i in range(n_c):
            if time.time() >= deadline:
                break
                
            cur_f = assignment[i]
            d = demands[i]
            cur_cost = current_costs[i]
            
            # Calculate cost delta for moving i to each f
            # new_cost = assign_cost[i, f]
            # open_delta = (open_cost[f] if not in_use[f] else 0) - (open_cost[cur_f] if loads[cur_f] == d else 0)
            # total_delta = new_cost - cur_cost + open_delta
            
            # Vectorized calculation of potential gains
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
            feasible = feasible & (np.arange(n_f) != cur_f) # No point moving to same
            
            if not np.any(feasible):
                continue
                
            total_deltas[~feasible] = np.inf
            
            best_f = np.argmin(total_deltas)
            gain = -total_deltas[best_f] # We want to minimize cost, so gain is negative delta
            
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
        # This is O(N^2) so we limit it or do it less frequently
        if time.time() < deadline:
            # Sample a subset of pairs to speed up if N is large
            # For N=100, N^2=10000, which is fast enough in Python if optimized.
            # We'll iterate through pairs.
            
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
                    # load[fi] - di + dk <= cap[fi]
                    # load[fk] - dk + di <= cap[fk]
                    if loads[fi] - di + dk > capacities[fi] + 1e-9:
                        continue
                    if loads[fk] - dk + di > capacities[fk] + 1e-9:
                        continue
                    
                    # Calculate cost change
                    new_ci = assign_cost[i, fk]
                    new_ck = assign_cost[k, fi]
                    
                    # Open cost change
                    # Before: fi and fk are used (since i and k are assigned to them)
                    # After: fi and fk are still used (since k and i are assigned to them)
                    # Exception: if one facility had only one customer, it might become unused?
                    # No, because we swap. If fi only had i, now it has k. Still used.
                    # If fk only had k, now it has i. Still used.
                    # So open cost doesn't change in a swap of two customers between two used facilities.
                    # Wait, what if fi == fk? We skipped that.
                    # What if one facility becomes empty?
                    # load[fi] - di + dk. If load[fi] was di, new load is dk. If dk>0, still used.
                    # So open cost delta is 0.
                    
                    delta_assign = (new_ci + new_ck) - (ci + ck)
                    
                    if delta_assign < -1e-9:
                        # Perform swap
                        loads[fi] = loads[fi] - di + dk
                        loads[fk] = loads[fk] - dk + di
                        assignment[i] = fk
                        assignment[k] = fi
                        current_costs[i] = new_ci
                        current_costs[k] = new_ck
                        
                        # in_use doesn't change for swaps between two non-empty facilities
                        # But just in case, recompute if needed (though it shouldn't)
                        improved = True
                        break # Restart i loop after a swap? Or just continue?
                        # Continuing is usually fine, but restarting ensures we pick up new opportunities.
                        # For speed, we just continue to next i.
                    
                if improved and time.time() >= deadline:
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
    
    # 1. Get Lagrangian prices
    lam = _lagrangian_dual(demands, capacities, assign_cost, max_iter=100, time_limit=10.0)
    
    # 2. Regret-based rounding
    best_assign = _regret_rounding(demands, capacities, assign_cost, lam)
    
    # Ensure validity
    if not _valid(best_assign, demands, capacities):
        # Fallback to simple greedy if rounding failed (should not happen)
        best_assign = np.zeros(n_c, dtype=int)
        loads = np.zeros(n_f)
        for i in range(n_c):
            for f in range(n_f):
                if loads[f] + demands[i] <= capacities[f]:
                    best_assign[i] = f
                    loads[f] += demands[i]
                    break
    
    best_cost = _total_cost(best_assign, demands, capacities, open_cost, assign_cost)
    
    # 3. Local Search Polish
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 5:
        polished = _local_search(best_assign, demands, capacities, open_cost, assign_cost, t0 + DEADLINE - 2)
        if _valid(polished, demands, capacities):
            polished_cost = _total_cost(polished, demands, capacities, open_cost, assign_cost)
            if polished_cost < best_cost:
                best_assign = polished
                best_cost = polished_cost
    
    # 4. Optional: Second round of Lagrangian + Rounding with different parameters
    # If we have time, try again with more iterations or different step sizes
    if time.time() - t0 < DEADLINE - 15:
        lam2 = _lagrangian_dual(demands, capacities, assign_cost, max_iter=200, time_limit=10.0)
        # Perturb lam slightly to escape local minima?
        lam2 = lam2 + np.random.uniform(-0.1, 0.1, size=n_f)
        lam2 = np.maximum(lam2, 0)
        
        assign2 = _regret_rounding(demands, capacities, assign_cost, lam2)
        if _valid(assign2, demands, capacities):
            cost2 = _total_cost(assign2, demands, capacities, open_cost, assign_cost)
            if cost2 < best_cost:
                best_assign = assign2
                best_cost = cost2
                
                # Polish the new best
                if time.time() - t0 < DEADLINE - 5:
                    polished2 = _local_search(best_assign, demands, capacities, open_cost, assign_cost, t0 + DEADLINE - 2)
                    if _valid(polished2, demands, capacities):
                        cost2_p = _total_cost(polished2, demands, capacities, open_cost, assign_cost)
                        if cost2_p < best_cost:
                            best_assign = polished2
                            best_cost = cost2_p
    
    return best_assign.astype(int)
