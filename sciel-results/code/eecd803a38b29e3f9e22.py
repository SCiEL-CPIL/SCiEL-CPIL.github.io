# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def _greedy_assign(demands, capacities, assign_cost, open_cost, allowed_subset):
    """
    Greedy assignment: sort customers by demand descending, assign to cheapest feasible facility.
    Returns (assignment, total_cost) or (None, inf).
    """
    n_cust = len(demands)
    if len(allowed_subset) == 0:
        return None, float('inf')
    
    total_cap = sum(capacities[f] for f in allowed_subset)
    total_demand = sum(demands)
    if total_cap < total_demand:
        return None, float('inf')
    
    load = np.zeros(len(capacities))
    assignment = np.zeros(n_cust, dtype=int)
    
    # Sort customers by demand descending
    order = np.argsort(-demands)
    
    for c in order:
        # Find cheapest feasible facility
        best_f = -1
        best_cost = float('inf')
        for f in allowed_subset:
            if load[f] + demands[c] <= capacities[f]:
                cost = assign_cost[c, f]
                if cost < best_cost:
                    best_cost = cost
                    best_f = f
        if best_f == -1:
            return None, float('inf')
        assignment[c] = best_f
        load[best_f] += demands[c]
    
    # Calculate total cost
    used = set(assignment.tolist())
    open_cost_total = sum(open_cost[f] for f in used)
    assign_cost_total = sum(assign_cost[c, assignment[c]] for c in range(n_cust))
    total_cost = open_cost_total + assign_cost_total
    
    return assignment, total_cost


def _local_search(assignment, demands, capacities, assign_cost, open_cost, max_iters=200):
    """
    Local search with single moves and swaps.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Precompute loads
    load = np.zeros(n_fac)
    for c in range(n_cust):
        f = assignment[c]
        load[f] += demands[c]
    
    # Track which facilities are open
    used = set(assignment.tolist())
    
    improved = True
    iters = 0
    
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        # Single moves: try moving each customer to a different facility
        for c in range(n_cust):
            current_f = assignment[c]
            current_cost = assign_cost[c, current_f]
            
            # Find best alternative facility (must be feasible)
            best_f = -1
            best_delta = 0.0
            
            for f in range(n_fac):
                if f == current_f:
                    continue
                if load[f] + demands[c] > capacities[f]:
                    continue
                
                delta = assign_cost[c, f] - current_cost
                # If f is not currently open, add open cost
                if f not in used:
                    delta += open_cost[f]
                # If current_f becomes empty after this move, subtract its open cost
                # (but we don't know yet if it will become empty)
                
                if delta < best_delta:
                    best_delta = delta
                    best_f = f
            
            if best_f != -1:
                # Make the move
                load[current_f] -= demands[c]
                load[best_f] += demands[c]
                assignment[c] = best_f
                
                # Update used set
                if load[current_f] == 0:
                    used.discard(current_f)
                used.add(best_f)
                
                improved = True
        
        # Swaps: try swapping two customers
        for c1 in range(n_cust):
            f1 = assignment[c1]
            for c2 in range(c1 + 1, n_cust):
                f2 = assignment[c2]
                if f1 == f2:
                    continue
                
                # Check feasibility after swap
                new_load_f1 = load[f1] - demands[c1] + demands[c2]
                new_load_f2 = load[f2] - demands[c2] + demands[c1]
                
                if new_load_f1 > capacities[f1] or new_load_f2 > capacities[f2]:
                    continue
                
                cost_before = assign_cost[c1, f1] + assign_cost[c2, f2]
                cost_after = assign_cost[c1, f2] + assign_cost[c2, f1]
                
                # Check if opening/closing facilities changes cost
                # f1 still has load (at least demands[c2]), so still open
                # f2 still has load (at least demands[c1]), so still open
                # No change in open cost
                
                if cost_after < cost_before - 1e-9:
                    # Make the swap
                    load[f1] = new_load_f1
                    load[f2] = new_load_f2
                    assignment[c1] = f2
                    assignment[c2] = f1
                    improved = True
    
    return assignment


def _vectorized_local_search(assignment, demands, capacities, assign_cost, open_cost, max_iters=50):
    """
    Vectorized local search for faster convergence.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    load = np.zeros(n_fac)
    for c in range(n_cust):
        f = assignment[c]
        load[f] += demands[c]
    
    used = set(assignment.tolist())
    
    for iters in range(max_iters):
        improved = False
        
        # For each customer, find the best facility to move to
        best_moves = []
        for c in range(n_cust):
            current_f = assignment[c]
            current_cost = assign_cost[c, current_f]
            
            # Vectorized: compute cost delta for all facilities
            # cost_delta[f] = assign_cost[c, f] - current_cost + open_cost[f] if f not in used
            cost_delta = assign_cost[c, :] - current_cost
            # Add open cost for unopened facilities
            for f in range(n_fac):
                if f not in used:
                    cost_delta[f] += open_cost[f]
            
            # Check feasibility
            feasible = load + demands[c] <= capacities
            
            # Mask infeasible
            cost_delta[~feasible] = float('inf')
            cost_delta[current_f] = float('inf')  # Don't move to same facility
            
            best_f = np.argmin(cost_delta)
            if cost_delta[best_f] < -1e-9:
                best_moves.append((c, best_f, cost_delta[best_f]))
        
        # Apply best moves (greedily, checking feasibility each time)
        best_moves.sort(key=lambda x: x[2])
        for c, f, delta in best_moves:
            if load[f] + demands[c] <= capacities[f]:
                current_f = assignment[c]
                load[current_f] -= demands[c]
                load[f] += demands[c]
                assignment[c] = f
                
                if load[current_f] == 0:
                    used.discard(current_f)
                used.add(f)
                improved = True
        
        if not improved:
            break
    
    return assignment


def assign(demands, capacities, open_cost, assign_cost):
    """
    Capacitated facility location: assign 100 customers to 20 facilities.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (150s limit)
    
    demands = np.asarray(demands)
    capacities = np.asarray(capacities)
    open_cost = np.asarray(open_cost)
    assign_cost = np.asarray(assign_cost)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Precompute average assignment cost for each facility
    avg_assign_cost = assign_cost.mean(axis=0)
    
    # Generate multiple candidate subsets using different heuristics
    candidate_subsets = set()
    
    # 1. All facilities
    candidate_subsets.add(frozenset(range(n_fac)))
    
    # 2. Based on score = open_cost + avg_assign_cost
    scores = open_cost + avg_assign_cost
    sorted_by_score = np.argsort(scores)
    for k in [8, 10, 12, 15, 18]:
        subset = frozenset(sorted_by_score[:k].tolist())
        candidate_subsets.add(subset)
    
    # 3. Based on open cost only
    sorted_by_open = np.argsort(open_cost)
    for k in [8, 10, 12, 15]:
        subset = frozenset(sorted_by_open[:k].tolist())
        candidate_subsets.add(subset)
    
    # 4. Based on avg assign cost only
    sorted_by_assign = np.argsort(avg_assign_cost)
    for k in [8, 10, 12, 15]:
        subset = frozenset(sorted_by_assign[:k].tolist())
        candidate_subsets.add(subset)
    
    # 5. Random subsets
    random.seed(42)
    for _ in range(5):
        k = random.randint(8, 18)
        subset = frozenset(random.sample(range(n_fac), k))
        candidate_subsets.add(subset)
    
    best_result = None
    best_total = float('inf')
    
    # Try each candidate subset
    for subset in candidate_subsets:
        if time.time() > deadline - 10:
            break
        
        allowed = list(subset)
        result, total_cost = _greedy_assign(demands, capacities, assign_cost, open_cost, allowed)
        
        if result is not None:
            if total_cost < best_total:
                best_total = total_cost
                best_result = result.copy()
    
    # Fallback if no valid solution found
    if best_result is None:
        best_result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        for i in np.argsort(-demands):
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    best_result[i] = f
                    break
            else:
                # Should not happen
                pass
        used = set(best_result.tolist())
        best_total = sum(open_cost[f] for f in used) + sum(assign_cost[i, best_result[i]] for i in range(n_cust))
    
    # Local search improvement
    if time.time() < deadline - 5:
        used_facilities = set(best_result.tolist())
        best_result = _local_search(best_result, demands, capacities, assign_cost, open_cost, max_iters=100)
        
        # Recalculate cost
        used_facilities = set(best_result.tolist())
        final_cost = sum(open_cost[f] for f in used_facilities) + sum(assign_cost[i, best_result[i]] for i in range(n_cust))
    
    return best_result
# EVOLVE-BLOCK-END
