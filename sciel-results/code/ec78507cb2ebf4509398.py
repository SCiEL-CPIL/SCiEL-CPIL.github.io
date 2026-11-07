"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog

def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Build a min-cost flow-like LP:
    # Nodes: source (0), customers (1..n_customers), facilities (1+n_customers..1+n_customers+n_facilities), sink (last)
    # But we can formulate directly as an LP with flow conservation.
    # Variables:
    #   f[i, j]: flow from customer i to facility j (amount of demand assigned)
    #   g[j]: flow from facility j to sink (total demand served by j)
    #   h[j, k]: surrogate for opening: we split open_cost into per-unit cost spread over capacity
    #     To approximate fixed charge: add cost per unit of demand assigned to facility j = open_cost[j] / max(capacities[j], 1)
    #     This makes the LP prefer facilities with lower per-unit open cost, encouraging fewer openings.
    #
    # Objective: minimize sum_{i,j} assign_cost[i,j] * f[i,j] + sum_j (open_cost[j] / max(capacities[j],1)) * g[j]
    #
    # Constraints:
    #   For each customer i: sum_j f[i,j] = demands[i]
    #   For each facility j: sum_i f[i,j] = g[j]
    #   0 <= f[i,j] <= demands[i]
    #   0 <= g[j] <= capacities[j]
    #
    # This is a transportation problem with an extra cost on facility usage.
    # Solve LP, then round to integer assignment.
    
    # To handle the fixed-charge better, we can use a two-phase approach:
    # Phase 1: Solve LP with per-unit open cost to get a good fractional solution
    # Phase 2: Use the LP solution as a guide for integer assignment
    
    # Build LP
    n_f = n_customers * n_facilities  # f variables
    n_g = n_facilities                 # g variables
    n_vars = n_f + n_g
    
    # Objective
    c = np.zeros(n_vars)
    c[:n_f] = assign_cost.flatten()
    # Per-unit open cost: open_cost[j] / capacities[j] (if capacities[j] > 0)
    per_unit_open = np.zeros(n_facilities)
    for j in range(n_facilities):
        cap_j = capacities[j] if capacities[j] > 0 else 1.0
        per_unit_open[j] = open_cost[j] / cap_j
    c[n_f:] = per_unit_open
    
    # Equality constraints:
    # 1. For each customer i: sum_j f[i,j] = demands[i]
    # 2. For each facility j: sum_i f[i,j] - g[j] = 0
    
    n_eq = n_customers + n_facilities
    A_eq = np.zeros((n_eq, n_vars))
    b_eq = np.zeros(n_eq)
    
    # Customer constraints
    for i in range(n_customers):
        for j in range(n_facilities):
            A_eq[i, i * n_facilities + j] = 1.0
        b_eq[i] = demands[i]
    
    # Facility balance constraints
    for j in range(n_facilities):
        for i in range(n_customers):
            A_eq[n_customers + j, i * n_facilities + j] = 1.0
        A_eq[n_customers + j, n_f + j] = -1.0
        b_eq[n_customers + j] = 0.0
    
    # Bounds
    bounds = []
    for i in range(n_customers):
        for j in range(n_facilities):
            bounds.append((0, demands[i]))
    for j in range(n_facilities):
        bounds.append((0, capacities[j]))
    
    # Solve LP
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if not res.success:
        # Fallback greedy
        return _greedy_fallback(demands, capacities, assign_cost)
    
    # Extract LP solution
    f_vals = res.x[:n_f].reshape(n_customers, n_facilities)
    g_vals = res.x[n_f:]
    
    # Round to integer assignment
    # Strategy: For each customer, assign to facility with highest fractional flow
    # Then repair capacity violations
    
    result = np.zeros(n_customers, dtype=int)
    
    # Sort customers by demand descending for better greedy repair
    order = np.argsort(-demands)
    
    # First pass: assign based on LP solution
    load = np.zeros(n_facilities)
    for i in order:
        # Find facility with highest LP flow
        best_j = np.argmax(f_vals[i])
        if load[best_j] + demands[i] <= capacities[best_j]:
            result[i] = best_j
            load[best_j] += demands[i]
        else:
            # Find alternative facility with enough capacity
            assigned = False
            for j in np.argsort(assign_cost[i]):
                if load[j] + demands[i] <= capacities[j]:
                    result[i] = j
                    load[j] += demands[i]
                    assigned = True
                    break
            if not assigned:
                # Should not happen if total capacity >= total demand
                result[i] = 0
    
    # Verify and repair if needed
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[result[i]] += demands[i]
    
    if np.any(load > capacities + 1e-6):
        # Repair: find customers that exceed capacity and reassign
        for j in range(n_facilities):
            excess = load[j] - capacities[j]
            if excess > 1e-6:
                # Find customers assigned to j with smallest demand first
                customers_at_j = np.where(result == j)[0]
                # Sort by demand ascending to move small ones first
                customers_at_j = customers_at_j[np.argsort(demands[customers_at_j])]
                for i in customers_at_j:
                    if excess <= 1e-6:
                        break
                    # Try to reassign i to another facility
                    for k in np.argsort(assign_cost[i]):
                        if k != j and load[k] + demands[i] <= capacities[k]:
                            load[j] -= demands[i]
                            load[k] += demands[i]
                            result[i] = k
                            excess -= demands[i]
                            break
    
    return result.astype(int)

def _greedy_fallback(demands, capacities, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    order = np.argsort(-demands)
    
    for i in order:
        best_j = None
        best_cost = np.inf
        for j in range(n_facilities):
            if load[j] + demands[i] <= capacities[j]:
                cost = assign_cost[i, j]
                if cost < best_cost:
                    best_cost = cost
                    best_j = j
        if best_j is not None:
            result[i] = best_j
            load[best_j] += demands[i]
        else:
            # Should not happen
            result[i] = 0
    
    return result
# EVOLVE-BLOCK-END
