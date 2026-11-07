"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def assign(demands, capacities, open_cost, assign_cost):
    """
    Capacitated facility location via LP relaxation + rounding + local search.
    """
    start_time = time.time()
    deadline = 110.0  # seconds, safety margin before 120s limit

    n_customers = len(demands)
    n_facilities = len(capacities)
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    # --- Step 1: Solve LP relaxation ---
    # Variables: x_{i,j} for i in customers, j in facilities (n_customers * n_facilities)
    #            y_j for j in facilities (n_facilities)
    # Total variables: n_customers * n_facilities + n_facilities
    n_vars = n_customers * n_facilities + n_facilities

    # Objective: minimize sum_{i,j} assign_cost[i,j] * x_{i,j} + sum_j open_cost[j] * y_j
    c = np.zeros(n_vars)
    for i in range(n_customers):
        for j in range(n_facilities):
            c[i * n_facilities + j] = assign_cost[i, j]
    for j in range(n_facilities):
        c[n_customers * n_facilities + j] = open_cost[j]

    # Constraints:
    # 1. Each customer assigned to exactly one facility: sum_j x_{i,j} = 1 for each i
    # 2. Capacity: sum_i demands[i] * x_{i,j} <= capacities[j] for each j
    # 3. Linking: x_{i,j} <= y_j for all i,j

    A_eq = []
    b_eq = []
    for i in range(n_customers):
        row = np.zeros(n_vars)
        for j in range(n_facilities):
            row[i * n_facilities + j] = 1.0
        A_eq.append(row)
        b_eq.append(1.0)
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)

    A_ub = []
    b_ub = []
    # Capacity constraints
    for j in range(n_facilities):
        row = np.zeros(n_vars)
        for i in range(n_customers):
            row[i * n_facilities + j] = demands[i]
        A_ub.append(row)
        b_ub.append(capacities[j])
    # Linking constraints: x_{i,j} - y_j <= 0
    for i in range(n_customers):
        for j in range(n_facilities):
            row = np.zeros(n_vars)
            row[i * n_facilities + j] = 1.0
            row[n_customers * n_facilities + j] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Bounds: 0 <= x <= 1, 0 <= y <= 1
    bounds = [(0, 1)] * n_vars

    # Solve LP
    lp_result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method='highs')

    if lp_result.success:
        x_lp = lp_result.x[:n_customers * n_facilities].reshape(n_customers, n_facilities)
        y_lp = lp_result.x[n_customers * n_facilities:]
    else:
        # Fallback: greedy
        x_lp = np.zeros((n_customers, n_facilities))
        y_lp = np.zeros(n_facilities)

    # --- Step 2: Rounding to get initial feasible solution ---
    # Use a deterministic rounding based on LP solution
    # First, determine which facilities to open based on y_lp
    # Then assign customers to open facilities respecting capacity

    # Sort facilities by LP y value (higher = more likely to be open)
    fac_order = np.argsort(-y_lp)
    open_facilities = []
    for j in fac_order:
        if y_lp[j] > 0.5:
            open_facilities.append(j)
    # If too few facilities open, add more
    if len(open_facilities) < 1:
        open_facilities = [fac_order[0]]

    # Assign customers greedily to open facilities, largest demand first
    result = np.full(n_customers, -1, dtype=int)
    load = np.zeros(n_facilities)

    cust_order = np.argsort(-demands)
    for i in cust_order:
        # Find feasible open facilities
        feasible = [j for j in open_facilities if load[j] + demands[i] <= capacities[j]]
        if not feasible:
            # Try all facilities
            feasible = [j for j in range(n_facilities) if load[j] + demands[i] <= capacities[j]]
        if not feasible:
            # Should not happen if capacities are sufficient
            result[i] = 0
            load[0] += demands[i]
            continue
        # Choose the one with minimum assign_cost
        best_j = min(feasible, key=lambda j: assign_cost[i, j])
        result[i] = best_j
        load[best_j] += demands[i]

    # Ensure all facilities in result are actually "open" (have at least one customer)
    # The cost will naturally account for this

    # --- Step 3: Local search with 1-swap and 2-swap moves ---
    def compute_cost(assignment):
        """Compute total cost: open costs + assignment costs."""
        total = 0.0
        used_facilities = set(assignment.tolist())
        for j in used_facilities:
            total += open_cost[j]
        for i in range(n_customers):
            total += assign_cost[i, assignment[i]]
        return total

    def is_feasible(assignment, loads):
        for i in range(n_customers):
            if loads[i] > capacities[i] + 1e-6:
                return False
        return True

    def get_loads(assignment):
        loads = np.zeros(n_facilities)
        for i in range(n_customers):
            loads[assignment[i]] += demands[i]
        return loads

    best_assignment = result.copy()
    best_cost = compute_cost(best_assignment)
    best_loads = get_loads(best_assignment)

    # 1-swap local search
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(n_customers):
            if time.time() > deadline:
                break
            current_f = best_assignment[i]
            # Try moving customer i to another facility
            for j in range(n_facilities):
                if j == current_f:
                    continue
                # Check capacity
                if best_loads[j] + demands[i] > capacities[j] + 1e-6:
                    continue
                # Compute delta cost
                # Remove i from current_f: if current_f becomes empty, save open_cost[current_f]
                # Add i to j: if j was empty, add open_cost[j]
                delta = assign_cost[i, j] - assign_cost[i, current_f]
                if best_loads[current_f] == demands[i]:
                    # current_f becomes empty
                    delta -= open_cost[current_f]
                if best_loads[j] == 0:
                    # j becomes non-empty
                    delta += open_cost[j]
                if delta < -1e-9:
                    # Accept the move
                    best_loads[current_f] -= demands[i]
                    best_loads[j] += demands[i]
                    best_assignment[i] = j
                    best_cost += delta
                    improved = True
                    break  # restart for this customer

    # 2-swap local search (swap two customers between facilities)
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(n_customers):
            if time.time() > deadline:
                break
            for k in range(i + 1, n_customers):
                if time.time() > deadline:
                    break
                fi = best_assignment[i]
                fk = best_assignment[k]
                if fi == fk:
                    continue
                # Check capacity after swap
                if best_loads[fk] - demands[k] + demands[i] > capacities[fk] + 1e-6:
                    continue
                if best_loads[fi] - demands[i] + demands[k] > capacities[fi] + 1e-6:
                    continue
                # Compute delta cost
                delta = (assign_cost[i, fk] + assign_cost[k, fi]) - \
                        (assign_cost[i, fi] + assign_cost[k, fk])
                # Check if facilities become empty
                if best_loads[fi] == demands[i]:
                    # fi becomes empty after removing i, but k moves in
                    # Actually, fi loses i and gains k, so it's not empty unless demands[i]==0 which it isn't
                    # Wait, fi loses i and gains k. So fi is not empty.
                    # fk loses k and gains i. So fk is not empty.
                    # So no open cost changes.
                    pass
                if best_loads[fk] == demands[k]:
                    pass
                # Actually, let's re-check: after swap, fi has load = best_loads[fi] - demands[i] + demands[k]
                # fk has load = best_loads[fk] - demands[k] + demands[i]
                # Neither becomes empty since demands > 0
                if delta < -1e-9:
                    # Accept the swap
                    best_loads[fi] = best_loads[fi] - demands[i] + demands[k]
                    best_loads[fk] = best_loads[fk] - demands[k] + demands[i]
                    best_assignment[i] = fk
                    best_assignment[k] = fi
                    best_cost += delta
                    improved = True
                    break
            if improved:
                break

    # Final feasibility check
    final_loads = get_loads(best_assignment)
    if np.any(final_loads > capacities + 1e-6):
        # Fallback to greedy if infeasible
        result = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities)
        for i in np.argsort(-demands):
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
        return result.astype(int)

    return best_assignment.astype(int)

# EVOLVE-BLOCK-END
