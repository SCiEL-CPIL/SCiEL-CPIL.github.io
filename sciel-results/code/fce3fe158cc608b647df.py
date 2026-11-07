"""Capacitated facility location: MILP via scipy.optimize.milp (HiGHS) with heuristic fallback."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix, vstack


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to one of 20 facilities respecting capacity constraints.
    Uses MILP formulation solved via HiGHS for near-optimal solutions.
    Falls back to greedy + local search if MILP fails or times out.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Safety deadline: 145s out of 150s
    deadline = time.time() + 145.0
    
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    # --- MILP Formulation ---
    # Variables:
    #   x[i, f] = 1 if customer i assigned to facility f (binary)
    #   y[f] = 1 if facility f is open (binary)
    # Variable ordering: [x[0,0],...,x[0,F-1], x[1,0],...,x[N-1,F-1], y[0],...,y[F-1]]
    
    n_x = n_customers * n_facilities
    n_y = n_facilities
    n_vars = n_x + n_y
    
    # Objective coefficients
    c = np.zeros(n_vars)
    for i in range(n_customers):
        for f in range(n_facilities):
            c[i * n_facilities + f] = assign_cost[i, f]
    for f in range(n_facilities):
        c[n_x + f] = open_cost[f]
    
    # Constraint 1: sum_f x[i,f] = 1 for each customer i (equality)
    rows1, cols1, vals1 = [], [], []
    for i in range(n_customers):
        for f in range(n_facilities):
            rows1.append(i)
            cols1.append(i * n_facilities + f)
            vals1.append(1.0)
    A1 = coo_matrix((vals1, (rows1, cols1)), shape=(n_customers, n_vars))
    lb1 = np.ones(n_customers)
    ub1 = np.ones(n_customers)
    
    # Constraint 2: sum_i demands[i] * x[i,f] <= capacities[f] for each facility f
    rows2, cols2, vals2 = [], [], []
    for f in range(n_facilities):
        for i in range(n_customers):
            rows2.append(f)
            cols2.append(i * n_facilities + f)
            vals2.append(demands[i])
    A2 = coo_matrix((vals2, (rows2, cols2)), shape=(n_facilities, n_vars))
    lb2 = np.full(n_facilities, -np.inf)
    ub2 = capacities.copy()
    
    # Constraint 3: x[i,f] <= y[f] for all i,f  =>  x[i,f] - y[f] <= 0
    rows3, cols3, vals3 = [], [], []
    for i in range(n_customers):
        for f in range(n_facilities):
            rows3.append(i * n_facilities + f)
            cols3.append(i * n_facilities + f)
            vals3.append(1.0)
            rows3.append(i * n_facilities + f)
            cols3.append(n_x + f)
            vals3.append(-1.0)
    A3 = coo_matrix((vals3, (rows3, cols3)), shape=(n_customers * n_facilities, n_vars))
    lb3 = np.full(n_customers * n_facilities, -np.inf)
    ub3 = np.zeros(n_customers * n_facilities)
    
    # Combine constraints
    A_all = vstack([A1, A2, A3])
    lb_all = np.concatenate([lb1, lb2, lb3])
    ub_all = np.concatenate([ub1, ub2, ub3])
    
    # All variables binary
    integrality = np.ones(n_vars)
    lb_vars = np.zeros(n_vars)
    ub_vars = np.ones(n_vars)
    
    constraints = LinearConstraint(A_all, lb_all, ub_all)
    bounds = Bounds(lb_vars, ub_vars)
    
    # Solve MILP with time limit
    try:
        result = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": 130, "mip_rel_gap": 0.01}
        )
        
        if result.x is not None and np.all(np.isfinite(result.x)):
            x_sol = result.x[:n_x].reshape(n_customers, n_facilities)
            assignment = np.argmax(x_sol, axis=1).astype(int)
            
            # Validate capacity
            load = np.zeros(n_facilities)
            for i in range(n_customers):
                load[assignment[i]] += demands[i]
            if np.all(load <= capacities + 1e-6):
                return assignment
            else:
                pass  # Fall through to heuristic
                
    except Exception:
        pass  # Fall through to heuristic
    
    # --- Fallback: Greedy + Local Search ---
    return _heuristic_fallback(demands, capacities, open_cost, assign_cost, deadline)


def _heuristic_fallback(demands, capacities, open_cost, assign_cost, deadline):
    """Greedy construction followed by local search (relocate + 2-swap)."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    # Greedy: largest demand first, cheapest feasible facility
    assignment = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities, dtype=np.float64)
    
    order = np.argsort(-demands)
    for i in order:
        fac_order = np.argsort(assign_cost[i])
        assigned = False
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                assignment[i] = f
                assigned = True
                break
        if not assigned:
            remaining = capacities - load
            f = np.argmax(remaining)
            assignment[i] = f
            load[f] += demands[i]
    
    # Local search: relocate moves
    fac_counts = np.zeros(n_facilities, dtype=int)
    for i in range(n_customers):
        fac_counts[assignment[i]] += 1
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i in range(n_customers):
            if time.time() > deadline - 2.0:
                break
            cf = assignment[i]
            for f in np.argsort(assign_cost[i]):
                if f == cf:
                    continue
                if load[f] + demands[i] > capacities[f]:
                    continue
                
                d_assign = assign_cost[i, f] - assign_cost[i, cf]
                d_open = 0.0
                if fac_counts[cf] == 1:
                    d_open -= open_cost[cf]
                if fac_counts[f] == 0:
                    d_open += open_cost[f]
                
                if d_assign + d_open < -1e-9:
                    assignment[i] = f
                    load[cf] -= demands[i]
                    load[f] += demands[i]
                    fac_counts[cf] -= 1
                    fac_counts[f] += 1
                    improved = True
                    break
    
    # 2-swap moves
    if time.time() < deadline - 2.0:
        swapped = True
        while swapped and time.time() < deadline - 2.0:
            swapped = False
            for i in range(n_customers):
                if time.time() > deadline - 2.0:
                    break
                for j in range(i + 1, n_customers):
                    if time.time() > deadline - 2.0:
                        break
                    fi = assignment[i]
                    fj = assignment[j]
                    if fi == fj:
                        continue
                    
                    if load[fi] - demands[i] + demands[j] > capacities[fi]:
                        continue
                    if load[fj] - demands[j] + demands[i] > capacities[fj]:
                        continue
                    
                    d_assign = (assign_cost[i, fj] + assign_cost[j, fi]) - \
                               (assign_cost[i, fi] + assign_cost[j, fj])
                    
                    if d_assign < -1e-9:
                        assignment[i] = fj
                        assignment[j] = fi
                        load[fi] = load[fi] - demands[i] + demands[j]
                        load[fj] = load[fj] - demands[j] + demands[i]
                        swapped = True
    
    # Final validation
    final_load = np.zeros(n_facilities)
    for i in range(n_customers):
        final_load[assignment[i]] += demands[i]
    
    if np.any(final_load > capacities + 1e-9):
        # Redo greedy as last resort
        assignment = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities, dtype=np.float64)
        order = np.argsort(-demands)
        for i in order:
            fac_order = np.argsort(assign_cost[i])
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    assignment[i] = f
                    break
            else:
                remaining = capacities - load
                f = np.argmax(remaining)
                assignment[i] = f
                load[f] += demands[i]
    
    return assignment.astype(int)
# EVOLVE-BLOCK-END
