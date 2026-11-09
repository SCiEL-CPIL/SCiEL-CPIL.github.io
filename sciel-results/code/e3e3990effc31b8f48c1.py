# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
import math
from scipy.optimize import linprog


def _solve_assignment_mcf(demands, capacities, assign_cost, open_facilities):
    """
    Solve capacitated assignment as a min-cost flow using linprog.
    Variables: x[i,j] = 1 if customer i assigned to facility j (j in open_facilities)
    Constraints: sum_j x[i,j] = 1 for each i; sum_i x[i,j] * demands[i] <= cap[j] for each j
    Objective: minimize sum x[i,j] * assign_cost[i,j]
    Returns (assignment array, total_assign_cost) or (None, inf).
    """
    n_cust = len(demands)
    n_fac = len(open_facilities)
    if n_fac == 0:
        return None, float('inf')
    
    # Check total capacity
    total_cap = sum(capacities[f] for f in open_facilities)
    if total_cap < sum(demands):
        return None, float('inf')
    
    # Decision variables: x[i, j] for i in range(n_cust), j in range(n_fac)
    # Total variables: n_cust * n_fac
    n_vars = n_cust * n_fac
    
    # Objective: c = assign_cost[i, open_facilities[j]]
    c = np.array([assign_cost[i, open_facilities[j]] for i in range(n_cust) for j in range(n_fac)], dtype=np.float64)
    
    # Equality constraints: for each customer i, sum_j x[i,j] = 1
    A_eq = np.zeros((n_cust, n_vars), dtype=np.float64)
    b_eq = np.ones(n_cust, dtype=np.float64)
    for i in range(n_cust):
        for j in range(n_fac):
            A_eq[i, i * n_fac + j] = 1.0
    
    # Inequality constraints: for each facility j, sum_i x[i,j] * demands[i] <= cap[j]
    A_ub = np.zeros((n_fac, n_vars), dtype=np.float64)
    b_ub = np.array([capacities[f] for f in open_facilities], dtype=np.float64)
    for j in range(n_fac):
        for i in range(n_cust):
            A_ub[j, i * n_fac + j] = demands[i]
    
    # Bounds: 0 <= x <= 1
    bounds = [(0, 1)] * n_vars
    
    try:
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not res.success:
            return None, float('inf')
        
        # Extract assignment
        assignment = np.full(n_cust, -1, dtype=int)
        for i in range(n_cust):
            for j in range(n_fac):
                if res.x[i * n_fac + j] > 0.5:
                    assignment[i] = open_facilities[j]
                    break
            if assignment[i] == -1:
                return None, float('inf')
        
        total_cost = sum(assign_cost[i, assignment[i]] for i in range(n_cust))
        return assignment, total_cost
    
    except Exception:
        return None, float('inf')


def _greedy_assignment(demands, capacities, assign_cost, open_facilities):
    """
    Greedy assignment: process largest demands first, assign to cheapest feasible facility.
    Returns (assignment array, total_assign_cost) or (None, inf).
    """
    n_cust = len(demands)
    n_fac = len(open_facilities)
    if n_fac == 0:
        return None, float('inf')
    
    total_cap = sum(capacities[f] for f in open_facilities)
    if total_cap < sum(demands):
        return None, float('inf')
    
    order = np.argsort(-demands)
    load = np.zeros(n_fac)
    assignment = np.full(n_cust, -1, dtype=int)
    
    # Precompute sorted facilities by cost for each customer
    costs = assign_cost[:, open_facilities]
    sorted_facs = np.argsort(costs, axis=1)
    
    for i in order:
        placed = False
        for f_local in sorted_facs[i]:
            f_global = open_facilities[f_local]
            if load[f_local] + demands[i] <= capacities[f_global]:
                load[f_local] += demands[i]
                assignment[i] = f_global
                placed = True
                break
        if not placed:
            return None, float('inf')
    
    total_cost = sum(assign_cost[i, assignment[i]] for i in range(n_cust))
    return assignment, total_cost


def _local_search(assignment, demands, capacities, assign_cost, open_cost, open_facilities, deadline):
    """
    Local search with 1-move, 2-swap, and facility open/close moves.
    Returns (best_assignment, best_total_cost).
    """
    n_cust = len(demands)
    open_facilities = np.array(open_facilities)
    n_fac = len(open_facilities)
    fac_to_local = {f: idx for idx, f in enumerate(open_facilities)}
    
    # Build load
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[fac_to_local[assignment[i]]] += demands[i]
    
    # Current cost
    used_mask = np.zeros(n_fac, dtype=bool)
    for f in assignment:
        used_mask[fac_to_local[f]] = True
    open_cost_total = open_cost[open_facilities][used_mask].sum()
    assign_cost_total = sum(assign_cost[i, assignment[i]] for i in range(n_cust))
    current_cost = open_cost_total + assign_cost_total
    
    best_assignment = assignment.copy()
    best_cost = current_cost
    
    # Precompute cost matrix for open facilities
    cost_mat = assign_cost[:, open_facilities]  # shape (n_cust, n_fac)
    
    # Local search
    no_improve = 0
    max_no_improve = 1000
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        
        # 1-move: try moving each customer to a different facility
        for i in range(n_cust):
            if time.time() > deadline:
                break
            f_old = fac_to_local[assignment[i]]
            for f_new in range(n_fac):
                if f_new == f_old:
                    continue
                if load[f_new] + demands[i] > capacities[open_facilities[f_new]]:
                    continue
                
                # Delta cost
                delta_assign = cost_mat[i, f_new] - cost_mat[i, f_old]
                
                # Open cost change
                new_load_old = load[f_old] - demands[i]
                new_load_new = load[f_new] + demands[i]
                
                close_old = (new_load_old == 0)
                open_new = (new_load_new == 0)  # Was 0, now >0
                
                delta_open = 0
                if open_new:
                    delta_open += open_cost[open_facilities[f_new]]
                if close_old:
                    delta_open -= open_cost[open_facilities[f_old]]
                
                delta = delta_assign + delta_open
                if delta < -1e-9:
                    # Accept
                    load[f_old] = new_load_old
                    load[f_new] = new_load_new
                    assignment[i] = open_facilities[f_new]
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_assignment = assignment.copy()
                    break
        
        # 2-swap: try swapping two customers
        for i in range(n_cust):
            if time.time() > deadline:
                break
            f_i = fac_to_local[assignment[i]]
            for j in range(i + 1, n_cust):
                f_j = fac_to_local[assignment[j]]
                if f_i == f_j:
                    continue
                
                # Check capacity
                new_load_i_at_j = load[f_j] - demands[j] + demands[i]
                new_load_j_at_i = load[f_i] - demands[i] + demands[j]
                
                if new_load_i_at_j > capacities[open_facilities[f_j]] or new_load_j_at_i > capacities[open_facilities[f_i]]:
                    continue
                
                delta_assign = (cost_mat[i, f_j] + cost_mat[j, f_i]) - (cost_mat[i, f_i] + cost_mat[j, f_j])
                if delta_assign < -1e-9:
                    # Accept
                    load[f_i] = new_load_j_at_i
                    load[f_j] = new_load_i_at_j
                    assignment[i] = open_facilities[f_j]
                    assignment[j] = open_facilities[f_i]
                    current_cost += delta_assign
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_assignment = assignment.copy()
        
        if not improved:
            no_improve += 1
        else:
            no_improve = 0
    
    return best_assignment, best_cost


def _branch_and_bound_facility_selection(demands, capacities, open_cost, assign_cost, deadline):
    """
    Branch-and-bound over facility subsets.
    Uses lower bound: sum of min assign_cost per customer + min open_cost for needed facilities.
    """
    n_cust = len(demands)
    n_fac = len(open_cost)
    
    # Precompute min assign cost per customer
    min_assign_per_cust = assign_cost.min(axis=1)
    lb_assign = min_assign_per_cust.sum()
    
    # Sort facilities by open cost
    fac_order = np.argsort(open_cost)
    
    best_result = None
    best_total = float('inf')
    
    # Use a priority queue: (lower_bound, open_set, closed_set, node_id)
    # Lower bound: lb_assign + sum of open costs for open facilities
    pq = []
    heapq.heappush(pq, (lb_assign, frozenset(), frozenset(), 0))
    
    node_count = 0
    max_nodes = 5000
    
    while pq and time.time() < deadline and node_count < max_nodes:
        lb, open_set, closed_set, _ = heapq.heappop(pq)
        node_count += 1
        
        if lb >= best_total:
            continue
        
        # Determine undecided facilities
        undecided = [f for f in range(n_fac) if f not in open_set and f not in closed_set]
        
        if not undecided:
            # All decided: try to solve assignment
            open_list = list(open_set)
            if not open_list:
                continue
            
            # Quick capacity check
            total_cap = sum(capacities[f] for f in open_list)
            if total_cap < sum(demands):
                continue
            
            # Try greedy first
            result, assign_cost_total = _greedy_assignment(demands, capacities, assign_cost, open_list)
            if result is not None:
                used_mask = np.zeros(n_fac, dtype=bool)
                for f in result:
                    used_mask[f] = True
                open_cost_total = open_cost[used_mask].sum()
                total = open_cost_total + assign_cost_total
                
                if total < best_total:
                    best_total = total
                    best_result = result.copy()
                    # Run local search
                    ls_result, ls_cost = _local_search(result, demands, capacities, assign_cost, open_cost, open_list, deadline)
                    if ls_cost < best_total:
                        best_total = ls_cost
                        best_result = ls_result.copy()
            continue
        
        # Branch on the next undecided facility
        f = undecided[0]
        
        # Branch 1: open facility f
        new_open = open_set | {f}
        new_lb = lb + open_cost[f]
        if new_lb < best_total:
            heapq.heappush(pq, (new_lb, new_open, closed_set, node_count + 1))
        
        # Branch 2: close facility f
        new_closed = closed_set | {f}
        # Check if remaining capacity is sufficient
        remaining_open = [f2 for f2 in range(n_fac) if f2 in open_set or (f2 not in closed_set and f2 != f)]
        # Actually, we need to check if the open_set plus all undecided (except f) can cover demand
        # This is complex; just push and let it be pruned later
        heapq.heappush(pq, (lb, open_set, new_closed, node_count + 1))
    
    return best_result, best_total


def assign(demands, capacities, open_cost, assign_cost):
    """
    Capacitated facility location: assign 100 customers to 20 facilities.
    Uses branch-and-bound + min-cost flow + local search.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Phase 1: Try several facility subsets with min-cost flow
    best_result = None
    best_total = float('inf')
    
    # Generate candidate subsets
    candidates = []
    
    # All facilities
    candidates.append(list(range(n_fac)))
    
    # Sort by open cost
    by_open = np.argsort(open_cost)
    for k in [8, 10, 12, 15]:
        candidates.append(by_open[:k].tolist())
    
    # Sort by average assign cost
    avg_assign = assign_cost.mean(axis=0)
    by_assign = np.argsort(avg_assign)
    for k in [8, 10, 12, 15]:
        candidates.append(by_assign[:k].tolist())
    
    # Combined metric
    combined = open_cost + avg_assign
    by_combined = np.argsort(combined)
    for k in [8, 10, 12, 15]:
        candidates.append(by_combined[:k].tolist())
    
    # Remove duplicates
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = frozenset(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    # Evaluate each subset
    for subset in unique_candidates:
        if time.time() > deadline - 30:
            break
        
        # Try min-cost flow first
        result, assign_cost_total = _solve_assignment_mcf(demands, capacities, assign_cost, subset)
        
        if result is None:
            # Fall back to greedy
            result, assign_cost_total = _greedy_assignment(demands, capacities, assign_cost, subset)
        
        if result is None:
            continue
        
        # Compute total cost
        used_mask = np.zeros(n_fac, dtype=bool)
        for f in result:
            used_mask[f] = True
        open_cost_total = open_cost[used_mask].sum()
        total = open_cost_total + assign_cost_total
        
        if total < best_total:
            best_total = total
            best_result = result.copy()
    
    # Phase 2: Local search on best result
    if best_result is not None:
        used_facilities = sorted(set(best_result.tolist()))
        ls_result, ls_cost = _local_search(best_result, demands, capacities, assign_cost, open_cost, used_facilities, deadline - 10)
        if ls_cost < best_total:
            best_result = ls_result
            best_total = ls_cost
    
    # Phase 3: Try branch-and-bound if time remains
    if time.time() < deadline - 20:
        bb_result, bb_cost = _branch_and_bound_facility_selection(demands, capacities, open_cost, assign_cost, deadline - 10)
        if bb_result is not None and bb_cost < best_total:
            best_result = bb_result
            best_total = bb_cost
    
    # Fallback: if no valid result, use greedy on all facilities
    if best_result is None:
        best_result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        for i in np.argsort(-demands):
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    best_result[i] = f
                    break
    
    return best_result.astype(int)
# EVOLVE-BLOCK-END
