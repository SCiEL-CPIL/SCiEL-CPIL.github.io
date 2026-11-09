"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linprog

def _solve_mcf(demands, capacities, open_cost, assign_cost, open_mask=None):
    """
    Solve the capacitated facility location as a Minimum Cost Flow problem.
    
    Network structure:
    - Source -> Customer nodes (capacity = demand, cost = 0)
    - Customer -> Facility nodes (capacity = demand, cost = assign_cost)
    - Facility -> Sink (capacity = capacity, cost = 0)
    - For facilities not in open_mask, we add a huge cost to discourage usage
    
    Actually, for MCF formulation of CFL:
    We need to decide which facilities to open. This is tricky with pure MCF.
    
    Alternative: Use the standard CFL formulation as an LP/ILP.
    Variables:
    - x[c, f]: 1 if customer c is assigned to facility f
    - y[f]: 1 if facility f is open
    
    Objective: sum_c,f x[c,f]*assign_cost[c,f] + sum_f y[f]*open_cost[f]
    Constraints:
    - sum_f x[c,f] = 1 for all c
    - sum_c demands[c]*x[c,f] <= capacities[f]*y[f] for all f
    - x[c,f] <= y[f] for all c,f
    - x, y binary (but LP relaxation works due to total unimodularity if we relax)
    
    Actually, the CFL is not totally unimodular in general. Let's use scipy's linprog
    with integer constraints via milp or just solve the LP and round.
    
    Better approach: Use scipy.optimize.milp for the exact MILP solution.
    """
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
    except ImportError:
        return None, float('inf')
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Variables: x[0..n_c*n_f-1] for assignment, y[n_c*n_f..n_c*n_f+n_f-1] for open
    n_x = n_customers * n_facilities
    n_y = n_facilities
    n_vars = n_x + n_y
    
    # Objective coefficients
    c_obj = np.zeros(n_vars)
    for ci in range(n_customers):
        for fi in range(n_facilities):
            c_obj[ci * n_facilities + fi] = assign_cost[ci, fi]
    for fi in range(n_facilities):
        c_obj[n_x + fi] = open_cost[fi]
    
    # Constraints:
    # 1. Each customer assigned to exactly one facility: sum_f x[c,f] = 1
    # 2. Capacity: sum_c demands[c]*x[c,f] <= capacities[f]*y[f]
    # 3. x[c,f] <= y[f]
    
    # Build constraint matrix
    n_cons = n_customers + n_facilities + n_x
    A = np.zeros((n_cons, n_vars))
    lb = np.zeros(n_cons)
    ub = np.zeros(n_cons)
    
    # Customer assignment constraints (equality)
    for ci in range(n_customers):
        for fi in range(n_facilities):
            A[ci, ci * n_facilities + fi] = 1.0
        lb[ci] = 1.0
        ub[ci] = 1.0
    
    # Capacity constraints: sum_c demands[c]*x[c,f] - capacities[f]*y[f] <= 0
    for fi in range(n_facilities):
        row = n_customers + fi
        for ci in range(n_customers):
            A[row, ci * n_facilities + fi] = demands[ci]
        A[row, n_x + fi] = -capacities[fi]
        lb[row] = -np.inf
        ub[row] = 0.0
    
    # x[c,f] <= y[f]: x[c,f] - y[f] <= 0
    for ci in range(n_customers):
        for fi in range(n_facilities):
            row = n_customers + n_facilities + ci * n_facilities + fi
            A[row, ci * n_facilities + fi] = 1.0
            A[row, n_x + fi] = -1.0
            lb[row] = -np.inf
            ub[row] = 0.0
    
    # Bounds: x in [0,1], y in [0,1]
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    
    # Integrality: all variables binary
    integrality = np.ones(n_vars)
    
    constraint = LinearConstraint(A, lb, ub)
    
    res = milp(c=c_obj, constraints=constraint, integrality=integrality, bounds=bounds)
    
    if res.status == 0 and res.fun < float('inf'):
        x_flat = res.x[:n_x]
        assignment = np.zeros(n_customers, dtype=int)
        for ci in range(n_customers):
            # Find the facility with x close to 1
            vals = x_flat[ci * n_facilities:(ci+1) * n_facilities]
            assignment[ci] = np.argmax(vals)
        
        # Verify feasibility
        loads = np.zeros(n_facilities)
        for ci in range(n_customers):
            loads[assignment[ci]] += demands[ci]
        if np.all(loads <= capacities + 1e-9):
            # Compute actual cost
            cost = 0.0
            for ci in range(n_customers):
                cost += assign_cost[ci, assignment[ci]]
            for fi in range(n_facilities):
                if loads[fi] > 0:
                    cost += open_cost[fi]
            return assignment, cost
    
    return None, float('inf')


def assign(demands, capacities, open_cost, assign_cost):
    """
    Capacitated facility location: assign customers to facilities minimizing
    open_cost + assign_cost, respecting capacities.
    
    Strategy: 
    1. Try to solve the exact MILP formulation.
    2. If that fails or takes too long, use heuristic + local search.
    3. Track best solution found within time budget.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    best_result = None
    best_cost = float('inf')
    
    def total_cost(assignment):
        """Compute total cost for an assignment."""
        loads = np.zeros(n_facilities)
        for ci in range(n_customers):
            loads[assignment[ci]] += demands[ci]
        if np.any(loads > capacities + 1e-9):
            return float('inf')
        cost = 0.0
        for ci in range(n_customers):
            cost += assign_cost[ci, assignment[ci]]
        for f in range(n_facilities):
            if loads[f] > 0:
                cost += open_cost[f]
        return cost
    
    # Try MILP first
    try:
        milp_result, milp_cost = _solve_mcf(demands, capacities, open_cost, assign_cost)
        if milp_result is not None:
            best_result = milp_result
            best_cost = milp_cost
    except Exception:
        pass
    
    # If MILP didn't solve, use heuristics
    if best_result is None or time.time() < deadline - 20:
        def local_search(assignment, max_iters=500):
            """Improve assignment via 1-swap local search."""
            loads = np.zeros(n_facilities)
            for ci in range(n_customers):
                loads[assignment[ci]] += demands[ci]
            
            current_cost = total_cost(assignment)
            if current_cost == float('inf'):
                return assignment, current_cost
            
            improved = True
            iter_count = 0
            while improved and iter_count < max_iters:
                improved = False
                iter_count += 1
                for ci in range(n_customers):
                    current_f = assignment[ci]
                    for f in range(n_facilities):
                        if f == current_f:
                            continue
                        # Check capacity
                        if loads[f] + demands[ci] <= capacities[f] + 1e-9:
                            # Compute cost delta
                            old_cost = assign_cost[ci, current_f]
                            new_cost = assign_cost[ci, f]
                            # Open cost change
                            open_delta = 0
                            if loads[current_f] == demands[ci]:
                                # This facility would be closed
                                open_delta -= open_cost[current_f]
                            if loads[f] == 0:
                                # This facility would be opened
                                open_delta += open_cost[f]
                            
                            delta = new_cost - old_cost + open_delta
                            if delta < -1e-9:
                                # Apply swap
                                loads[current_f] -= demands[ci]
                                loads[f] += demands[ci]
                                assignment[ci] = f
                                current_cost += delta
                                improved = True
                                break
            return assignment, current_cost
        
        def greedy_nearest(demands, capacities, open_cost, assign_cost):
            """Greedy: assign each customer to nearest feasible facility."""
            result = np.zeros(n_customers, dtype=int)
            load = np.zeros(n_facilities)
            order = np.argsort(-demands)
            for i in order:
                feasible = [f for f in range(n_facilities) if load[f] + demands[i] <= capacities[f]]
                if not feasible:
                    f = np.argmax(capacities - load)
                    result[i] = f
                    load[f] += demands[i]
                else:
                    f = min(feasible, key=lambda f: assign_cost[i, f])
                    result[i] = f
                    load[result[i]] += demands[i]
            return result
        
        def greedy_open_aware(demands, capacities, open_cost, assign_cost):
            """Greedy that considers open cost more prominently."""
            result = np.zeros(n_customers, dtype=int)
            load = np.zeros(n_facilities)
            order = np.argsort(-demands)
            for i in order:
                feasible = [f for f in range(n_facilities) if load[f] + demands[i] <= capacities[f]]
                if not feasible:
                    f = np.argmax(capacities - load)
                    result[i] = f
                    load[f] += demands[i]
                else:
                    # Compute effective cost: assign_cost + marginal open cost
                    eff_costs = []
                    for f in feasible:
                        if load[f] == 0:
                            eff_costs.append(assign_cost[i, f] + open_cost[f])
                        else:
                            eff_costs.append(assign_cost[i, f])
                    f = feasible[np.argmin(eff_costs)]
                    result[i] = f
                    load[result[i]] += demands[i]
            return result
        
        def greedy_randomized(demands, capacities, open_cost, assign_cost, rng):
            """Greedy with randomized facility ordering."""
            result = np.zeros(n_customers, dtype=int)
            load = np.zeros(n_facilities)
            order = np.argsort(-demands)
            for i in order:
                feasible = [f for f in range(n_facilities) if load[f] + demands[i] <= capacities[f]]
                if not feasible:
                    f = np.argmax(capacities - load)
                    result[i] = f
                    load[f] += demands[i]
                else:
                    costs = np.array([assign_cost[i, f] for f in feasible])
                    costs_noisy = costs + rng.uniform(0, 1) * np.max(costs) * 0.1
                    min_idx = np.argmin(costs_noisy)
                    result[i] = feasible[min_idx]
                    load[result[i]] += demands[i]
            return result
        
        # Generate diverse candidates
        candidates = []
        
        # Deterministic heuristics
        candidates.append(greedy_nearest(demands, capacities, open_cost, assign_cost))
        candidates.append(greedy_open_aware(demands, capacities, open_cost, assign_cost))
        
        # Randomized heuristics with different seeds
        for seed in range(10):
            rng_i = np.random.default_rng(seed)
            cand = greedy_randomized(demands, capacities, open_cost, assign_cost, rng_i)
            candidates.append(cand)
        
        # Evaluate and improve each candidate
        for cand in candidates:
            if time.time() > deadline - 5:
                break
            
            # Validate
            loads = np.zeros(n_facilities)
            valid = True
            for ci in range(n_customers):
                if cand[ci] < 0 or cand[ci] >= n_facilities:
                    valid = False
                    break
                loads[cand[ci]] += demands[ci]
            if not valid or np.any(loads > capacities + 1e-9):
                continue
            
            cost = total_cost(cand)
            if cost < best_cost:
                best_cost = cost
                best_result = cand.copy()
            
            # Local search to improve
            if time.time() < deadline - 10:
                improved, improved_cost = local_search(cand.copy(), max_iters=200)
                if improved_cost < best_cost:
                    best_cost = improved_cost
                    best_result = improved.copy()
        
        # If we still have time, do more local search iterations on the best solution
        if best_result is not None and time.time() < deadline - 5:
            while time.time() < deadline - 2:
                improved, improved_cost = local_search(best_result.copy(), max_iters=100)
                if improved_cost < best_cost - 1e-9:
                    best_cost = improved_cost
                    best_result = improved.copy()
                else:
                    break
    
    if best_result is None:
        # Fallback: simple greedy
        best_result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        order = np.argsort(-demands)
        for i in order:
            feasible = [f for f in range(n_facilities) if load[f] + demands[i] <= capacities[f]]
            if not feasible:
                f = np.argmax(capacities - load)
            else:
                f = min(feasible, key=lambda f: assign_cost[i, f])
            best_result[i] = f
            load[f] += demands[i]
    
    return best_result.astype(int)

# EVOLVE-BLOCK-END
