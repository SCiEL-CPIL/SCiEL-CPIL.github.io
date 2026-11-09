"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
import heapq
import itertools

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Precompute some values
    total_demand = np.sum(demands)
    
    # Helper to solve the LP relaxation for a set of available facilities
    def solve_lp_relaxation(avail_facilities):
        if not avail_facilities:
            return np.inf, None
            
        avail_list = list(avail_facilities)
        m = len(avail_list)
        
        # Variables x[c, f] for c in range(n_customers), f in avail_list
        # Total variables: n_customers * m
        n_vars = n_customers * m
        
        c_obj = np.zeros(n_vars)
        # Map (c, f_idx) to variable index
        # var_idx = c * m + f_idx
        
        for c in range(n_customers):
            for j, f in enumerate(avail_list):
                c_obj[c * m + j] = assign_cost[c, f]
                
        # Constraints:
        # 1. Sum_f x[c, f] = 1 for each c
        # 2. Sum_c demands[c] * x[c, f] <= capacities[f] for each f in avail_list
        
        A_eq = np.zeros((n_customers, n_vars))
        b_eq = np.ones(n_customers)
        for c in range(n_customers):
            for j in range(m):
                A_eq[c, c * m + j] = 1.0
                
        A_ub = np.zeros((m, n_vars))
        b_ub = np.zeros(m)
        for j, f in enumerate(avail_list):
            for c in range(n_customers):
                A_ub[j, c * m + j] = demands[c]
            b_ub[j] = capacities[f]
            
        bounds = [(0, 1)] * n_vars # x is between 0 and 1
        
        try:
            res = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                return res.fun, res.x
            else:
                return np.inf, None
        except Exception:
            return np.inf, None

    # Helper to solve the integer GAP for a set of open facilities
    def solve_gap(open_facilities):
        if not open_facilities:
            return np.inf, None
            
        open_list = list(open_facilities)
        m = len(open_list)
        n_vars = n_customers * m
        
        c_obj = np.zeros(n_vars)
        for c in range(n_customers):
            for j, f in enumerate(open_list):
                c_obj[c * m + j] = assign_cost[c, f]
                
        # Constraints
        A_eq = np.zeros((n_customers, n_vars))
        b_eq = np.ones(n_customers)
        for c in range(n_customers):
            for j in range(m):
                A_eq[c, c * m + j] = 1.0
                
        A_ub = np.zeros((m, n_vars))
        b_ub = np.zeros(m)
        for j, f in enumerate(open_list):
            for c in range(n_customers):
                A_ub[j, c * m + j] = demands[c]
            b_ub[j] = capacities[f]
            
        bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
        integrality = np.ones(n_vars)
        
        try:
            res = milp(c=c_obj, 
                       constraints=[LinearConstraint(A_eq, b_eq, b_eq),
                                    LinearConstraint(A_ub, -np.inf, b_ub)],
                       bounds=bounds, 
                       integrality=integrality)
            if res.status == 0:
                x = res.x.reshape(n_customers, m)
                # Extract assignment
                assignment = np.zeros(n_customers, dtype=int)
                for c in range(n_customers):
                    assigned_j = np.argmax(x[c])
                    assignment[c] = open_list[assigned_j]
                return res.fun, assignment
            else:
                return np.inf, None
        except Exception:
            return np.inf, None

    # Branch and Bound
    best_cost = np.inf
    best_assignment = None
    
    # Initial greedy solution to set an upper bound
    # Sort customers by demand descending
    order = np.argsort(-demands)
    loads = np.zeros(n_facilities)
    greedy_assign = np.zeros(n_customers, dtype=int)
    feasible = True
    for c in order:
        best_f = -1
        best_c = np.inf
        for f in range(n_facilities):
            if loads[f] + demands[c] <= capacities[f]:
                if assign_cost[c, f] < best_c:
                    best_c = assign_cost[c, f]
                    best_f = f
        if best_f == -1:
            feasible = False
            break
        greedy_assign[c] = best_f
        loads[best_f] += demands[c]
        
    if feasible:
        open_set_greedy = set(greedy_assign)
        cost_greedy = sum(open_cost[f] for f in open_set_greedy) + sum(assign_cost[c, greedy_assign[c]] for c in range(n_customers))
        best_cost = cost_greedy
        best_assignment = greedy_assign.copy()
        
    # B&B Stack: (lb, open_set, closed_set, undecided_set)
    # We use a priority queue to explore nodes with lowest LB first
    # State: (lb, open_mask, closed_mask, undecided_mask)
    # Using bitmasks for efficiency
    # n_facilities <= 20, so 32-bit int is enough
    
    initial_undecided = (1 << n_facilities) - 1
    stack = []
    
    # Calculate initial LB
    lb_init, _ = solve_lp_relaxation(set(range(n_facilities)))
    if lb_init < np.inf:
        heapq.heappush(stack, (lb_init, 0, 0, initial_undecided))
        
    node_count = 0
    while stack:
        lb, open_mask, closed_mask, undecided_mask = heapq.heappop(stack)
        node_count += 1
        
        if lb >= best_cost:
            continue
            
        if undecided_mask == 0:
            # Leaf node
            open_set = set(i for i in range(n_facilities) if (open_mask >> i) & 1)
            cost, assignment = solve_gap(open_set)
            if cost < np.inf and cost < best_cost:
                best_cost = cost
                best_assignment = assignment
            continue
            
        # Branch on a facility in undecided
        # Heuristic: pick facility with highest open cost or lowest index?
        # Let's pick the one with highest open cost to prune early?
        # Or the one most "central"?
        # For simplicity, pick lowest index in undecided
        f = (undecided_mask & -undecided_mask).bit_length() - 1
        
        # Branch 1: Open f
        new_open_mask = open_mask | (1 << f)
        new_undecided_mask = undecided_mask & ~(1 << f)
        
        # Available facilities for LB: Open + Undecided
        avail_set = set(i for i in range(n_facilities) if ((open_mask | new_undecided_mask) >> i) & 1)
        lb1, _ = solve_lp_relaxation(avail_set)
        if lb1 < np.inf:
            lb1 += sum(open_cost[i] for i in range(n_facilities) if (new_open_mask >> i) & 1)
            if lb1 < best_cost:
                heapq.heappush(stack, (lb1, new_open_mask, closed_mask, new_undecided_mask))
                
        # Branch 2: Close f
        new_closed_mask = closed_mask | (1 << f)
        new_undecided_mask2 = undecided_mask & ~(1 << f)
        
        avail_set2 = set(i for i in range(n_facilities) if ((open_mask | new_undecided_mask2) >> i) & 1)
        lb2, _ = solve_lp_relaxation(avail_set2)
        if lb2 < np.inf:
            lb2 += sum(open_cost[i] for i in range(n_facilities) if (open_mask >> i) & 1)
            if lb2 < best_cost:
                heapq.heappush(stack, (lb2, open_mask, new_closed_mask, new_undecided_mask2))
                
    if best_assignment is not None:
        return best_assignment
    else:
        # Fallback if B&B failed (should not happen)
        return greedy_assign

# EVOLVE-BLOCK-END
