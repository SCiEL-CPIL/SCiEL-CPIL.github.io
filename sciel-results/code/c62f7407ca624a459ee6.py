"""
Capacitated Facility Location Problem (CFLP) solver using MIP with HiGHS backend.

Strategy:
1. Formulate the CFLP as a Mixed Integer Linear Program (MIP):
   - Binary variables x[c, f]: customer c assigned to facility f
   - Binary variables y[f]: facility f is open
   - Constraints: each customer assigned to exactly one facility, capacity limits,
     and facility opening linkage.
   - Objective: minimize sum of assignment costs + open costs.
2. Use scipy.optimize.milp with HiGHS backend for exact/near-exact solution.
3. Fallback to greedy + local search if MIP times out or fails.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility.
    
    Returns: integer NumPy array with one facility index per customer.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (evaluator kills at 150s)
    
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capabilities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    def compute_cost(assignment):
        """Compute total cost for an assignment, return inf if infeasible."""
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            f = int(assignment[c])
            if f < 0 or f >= n_facilities:
                return np.inf
            load[f] += demands[c]
        for f in range(n_facilities):
            if load[f] > capacities[f] + 1e-9:
                return np.inf
        
        cost = 0.0
        used = np.zeros(n_facilities, dtype=bool)
        for c in range(n_customers):
            f = int(assignment[c])
            cost += assign_cost[c, f]
            used[f] = True
        for f in range(n_facilities):
            if used[f]:
                cost += open_cost[f]
        return cost
    
    def solve_mip():
        """
        Solve the CFLP as a MIP using scipy.optimize.milp.
        
        Variables:
        - x[c, f] for c in 0..n_customers-1, f in 0..n_facilities-1 (binary)
        - y[f] for f in 0..n_facilities-1 (binary)
        
        Total variables: n_customers * n_facilities + n_facilities
        
        Objective: minimize sum_{c,f} assign_cost[c,f] * x[c,f] + sum_f open_cost[f] * y[f]
        
        Constraints:
        1. For each customer c: sum_f x[c,f] = 1
        2. For each facility f: sum_c demands[c] * x[c,f] <= capacities[f] * y[f]
        3. x[c,f] <= y[f] for all c, f (if customer assigned to f, f must be open)
        """
        # Variable layout: [x[0,0], x[0,1], ..., x[0,F-1], x[1,0], ..., x[nC-1,F-1], y[0], ..., y[F-1]]
        n_x = n_customers * n_facilities
        n_y = n_facilities
        n_vars = n_x + n_y
        
        # Objective coefficients
        c_obj = np.zeros(n_vars)
        for c in range(n_customers):
            for f in range(n_facilities):
                c_obj[c * n_facilities + f] = assign_cost[c, f]
        for f in range(n_facilities):
            c_obj[n_x + f] = open_cost[f]
        
        # Bounds: all binary [0, 1]
        lower_bounds = np.zeros(n_vars)
        upper_bounds = np.ones(n_vars)
        integrality = np.ones(n_vars)  # All variables are integer (binary)
        
        # Build constraint matrix A_ub x <= b_ub
        # Constraint 1: For each customer c: sum_f x[c,f] = 1 -> use equality constraints
        # Constraint 2: For each facility f: sum_c demands[c] * x[c,f] - capacities[f] * y[f] <= 0
        # Constraint 3: For each c,f: x[c,f] - y[f] <= 0
        
        # We'll use A_eq for constraint 1, A_ub for constraints 2 and 3
        
        # Equality constraints: n_customers constraints
        A_eq = np.zeros((n_customers, n_vars))
        b_eq = np.ones(n_customers)
        for c in range(n_customers):
            for f in range(n_facilities):
                A_eq[c, c * n_facilities + f] = 1.0
        
        # Inequality constraints:
        # Type 2: n_facilities constraints
        # Type 3: n_customers * n_facilities constraints
        n_ineq = n_facilities + n_customers * n_facilities
        A_ub = np.zeros((n_ineq, n_vars))
        b_ub = np.zeros(n_ineq)
        
        # Type 2: capacity constraints
        for f in range(n_facilities):
            row = f
            for c in range(n_customers):
                A_ub[row, c * n_facilities + f] = demands[c]
            A_ub[row, n_x + f] = -capacities[f]
            b_ub[row] = 0.0
        
        # Type 3: x[c,f] <= y[f]
        for c in range(n_customers):
            for f in range(n_facilities):
                row = n_facilities + c * n_facilities + f
                A_ub[row, c * n_facilities + f] = 1.0
                A_ub[row, n_x + f] = -1.0
                b_ub[row] = 0.0
        
        try:
            res = milp(
                c=c_obj,
                constraints=LinearConstraint(A_ub, -np.inf, b_ub),
                integrality=integrality,
                bounds=Bounds(lower_bounds, upper_bounds)
            )
            
            # Also need to handle equality constraints - scipy's milp handles them via constraints
            # Actually, let me restructure to put all constraints in one LinearConstraint
            
            return None  # Will redo below
        except Exception:
            return None
    
    def solve_mip_v2():
        """
        Solve the CFLP as a MIP using scipy.optimize.milp with proper constraint setup.
        """
        n_x = n_customers * n_facilities
        n_y = n_facilities
        n_vars = n_x + n_y
        
        # Objective coefficients
        c_obj = np.zeros(n_vars)
        for c in range(n_customers):
            for f in range(n_facilities):
                c_obj[c * n_facilities + f] = assign_cost[c, f]
        for f in range(n_facilities):
            c_obj[n_x + f] = open_cost[f]
        
        # Bounds: all binary [0, 1]
        lower_bounds = np.zeros(n_vars)
        upper_bounds = np.ones(n_vars)
        integrality = np.ones(n_vars)
        
        # All inequality constraints: A_ub x <= b_ub
        # Type 2: For each facility f: sum_c demands[c] * x[c,f] - capacities[f] * y[f] <= 0
        # Type 3: For each c,f: x[c,f] - y[f] <= 0
        # Type 1 (equality): For each customer c: sum_f x[c,f] = 1 -> convert to two inequalities
        #   sum_f x[c,f] <= 1 AND -sum_f x[c,f] <= -1
        
        n_ineq = n_facilities + n_customers * n_facilities + 2 * n_customers
        A_ub = np.zeros((n_ineq, n_vars))
        b_ub = np.zeros(n_ineq)
        
        row_idx = 0
        
        # Type 2: capacity constraints
        for f in range(n_facilities):
            for c in range(n_customers):
                A_ub[row_idx, c * n_facilities + f] = demands[c]
            A_ub[row_idx, n_x + f] = -capacities[f]
            b_ub[row_idx] = 0.0
            row_idx += 1
        
        # Type 3: x[c,f] <= y[f]
        for c in range(n_customers):
            for f in range(n_facilities):
                A_ub[row_idx, c * n_facilities + f] = 1.0
                A_ub[row_idx, n_x + f] = -1.0
                b_ub[row_idx] = 0.0
                row_idx += 1
        
        # Type 1a: sum_f x[c,f] <= 1
        for c in range(n_customers):
            for f in range(n_facilities):
                A_ub[row_idx, c * n_facilities + f] = 1.0
            b_ub[row_idx] = 1.0
            row_idx += 1
        
        # Type 1b: -sum_f x[c,f] <= -1
        for c in range(n_customers):
            for f in range(n_facilities):
                A_ub[row_idx, c * n_facilities + f] = -1.0
            b_ub[row_idx] = -1.0
            row_idx += 1
        
        try:
            res = milp(
                c=c_obj,
                constraints=LinearConstraint(A_ub, -np.inf, b_ub),
                integrality=integrality,
                bounds=Bounds(lower_bounds, upper_bounds)
            )
            
            if not res.success:
                return None
            
            # Extract solution
            x_vals = res.x[:n_x].reshape(n_customers, n_facilities)
            y_vals = res.x[n_x:n_x + n_y]
            
            # Build assignment
            assignment = np.full(n_customers, -1, dtype=int)
            for c in range(n_customers):
                best_f = np.argmax(x_vals[c])
                if x_vals[c, best_f] < 0.5:
                    return None
                assignment[c] = best_f
            
            # Verify feasibility
            load = np.zeros(n_facilities)
            for c in range(n_customers):
                load[assignment[c]] += demands[c]
            for f in range(n_facilities):
                if load[f] > capacities[f] + 1e-6:
                    return None
            
            return assignment, res.fun
        
        except Exception:
            return None
    
    def solve_greedy(open_facilities):
        """
        Greedy assignment to a given set of open facilities.
        """
        if not open_facilities:
            return None
        
        assignment = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities)
        
        # Sort customers by demand descending
        order = np.argsort(-demands)
        
        for c in order:
            best_f = -1
            best_cost = np.inf
            for f in open_facilities:
                if load[f] + demands[c] <= capacities[f]:
                    cost = assign_cost[c, f]
                    if cost < best_cost:
                        best_cost = cost
                        best_f = f
            if best_f == -1:
                return None
            assignment[c] = best_f
            load[best_f] += demands[c]
        
        return assignment
    
    def local_search(assignment, max_time=10.0):
        """
        Local search with single moves.
        """
        assignment = assignment.copy()
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[int(assignment[c])] += demands[c]
        
        current_cost = compute_cost(assignment)
        if np.isinf(current_cost):
            return assignment
        
        start_ls = time.time()
        improved = True
        while improved and (time.time() - start_ls) < max_time:
            improved = False
            
            for c in range(n_customers):
                if time.time() > deadline:
                    return assignment
                old_f = int(assignment[c])
                load[old_f] -= demands[c]
                
                best_new_f = old_f
                best_delta = 0.0
                
                for new_f in range(n_facilities):
                    if new_f == old_f:
                        continue
                    if load[new_f] + demands[c] > capacities[new_f]:
                        continue
                    
                    delta = assign_cost[c, new_f] - assign_cost[c, old_f]
                    
                    if load[new_f] == 0:
                        delta += open_cost[new_f]
                    if load[old_f] == 0:
                        delta -= open_cost[old_f]
                        
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_new_f = new_f
                        
                if best_new_f != old_f:
                    assignment[c] = best_new_f
                    load[best_new_f] += demands[c]
                    current_cost += best_delta
                    improved = True
                else:
                    load[old_f] += demands[c]
                    
        return assignment
    
    # --- Main Logic ---
    
    best_solution = None
    best_cost = np.inf
    
    # First, try MIP with time limit
    if time.time() < deadline - 30:
        mip_result = solve_mip_v2()
        if mip_result is not None:
            mip_assignment, mip_cost = mip_result
            best_solution = mip_assignment
            best_cost = mip_cost
    
    # If MIP didn't give a solution, use greedy + local search
    if best_solution is None or time.time() < deadline - 20:
        # Greedy with all facilities open
        greedy_sol = solve_greedy(list(range(n_facilities)))
        if greedy_sol is not None:
            greedy_cost = compute_cost(greedy_sol)
            if greedy_cost < best_cost:
                best_cost = greedy_cost
                best_solution = greedy_sol
            
            # Local search
            if time.time() < deadline - 10:
                ls_sol = local_search(best_solution, max_time=min(15.0, deadline - time.time() - 5))
                ls_cost = compute_cost(ls_sol)
                if ls_cost < best_cost:
                    best_cost = ls_cost
                    best_solution = ls_sol
    
    # Final polish with local search
    if best_solution is not None and time.time() < deadline - 5:
        final_sol = local_search(best_solution, max_time=deadline - time.time() - 2)
        final_cost = compute_cost(final_sol)
        if final_cost < best_cost:
            best_solution = final_sol
    
    if best_solution is None:
        # Ultimate fallback: assign all to facility 0 (may be infeasible but better than nothing)
        best_solution = np.zeros(n_customers, dtype=int)
    
    return best_solution.astype(int)

# EVOLVE-BLOCK-END
