import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using a MILP formulation.
    Falls back to a greedy heuristic if the solver fails or times out.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Fallback heuristic: Greedy Largest Demand First
    def greedy_fallback():
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        # Sort customers by demand descending
        sorted_idx = np.argsort(-demands)
        for i in sorted_idx:
            # Sort facilities by assignment cost for this customer
            sorted_fac = np.argsort(assign_cost[i])
            assigned = False
            for f in sorted_fac:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            if not assigned:
                # If no facility has capacity, assign to the one with most remaining capacity
                # This case implies infeasible instance or logic error in problem setup
                # Assuming feasible instances as per task description
                remaining_cap = capacities - load
                best_f = np.argmax(remaining_cap)
                if remaining_cap[best_f] >= 0: # Just in case
                    result[i] = best_f
                else:
                    result[i] = 0 # Arbitrary, should not happen
        return result

    # Try MILP
    try:
        # Variable ordering:
        # x[0 : n_customers * n_facilities] : x_{i,f}
        # x[n_customers * n_facilities : n_customers * n_facilities + n_facilities] : y_f
        
        n_x = n_customers * n_facilities
        n_y = n_facilities
        total_vars = n_x + n_y
        
        # Objective coefficients
        c_obj = np.zeros(total_vars)
        
        # Assignment costs
        # Reshape assign_cost to (n_customers, n_facilities)
        ac = np.asarray(assign_cost)
        c_obj[:n_x] = ac.flatten()
        
        # Opening costs
        c_obj[n_x:n_x+n_y] = np.asarray(open_cost)
        
        # Bounds
        # x_{i,f} are binary (0 or 1)
        # y_f are binary (0 or 1)
        lb = np.zeros(total_vars)
        ub = np.ones(total_vars)
        
        # Integrality
        integrality = np.ones(total_vars) # All variables are integer (binary)
        
        # Constraints
        # 1. Assignment: sum_f x_{i,f} = 1 for each i
        # Rows: n_customers
        # Cols: n_x + n_y
        A_assign = np.zeros((n_customers, total_vars))
        for i in range(n_customers):
            # x_{i,f} are at indices i*n_facilities to i*n_facilities + n_facilities - 1
            start = i * n_facilities
            end = start + n_facilities
            A_assign[i, start:end] = 1.0
        
        lb_assign = np.ones(n_customers)
        ub_assign = np.ones(n_customers)
        
        # 2. Capacity: sum_i d_i x_{i,f} <= capacities[f] for each f
        # Rows: n_facilities
        # Cols: n_x + n_y
        A_cap = np.zeros((n_facilities, total_vars))
        d = np.asarray(demands)
        for f in range(n_facilities):
            for i in range(n_customers):
                # Index of x_{i,f} is i*n_facilities + f
                idx = i * n_facilities + f
                A_cap[f, idx] = d[i]
        
        lb_cap = np.zeros(n_facilities)
        ub_cap = np.asarray(capacities)
        
        # 3. Linking: x_{i,f} <= y_f  =>  x_{i,f} - y_f <= 0
        # Rows: n_customers * n_facilities
        # Cols: n_x + n_y
        A_link = np.zeros((n_x, total_vars))
        for i in range(n_customers):
            for f in range(n_facilities):
                idx_x = i * n_facilities + f
                idx_y = n_x + f
                A_link[i * n_facilities + f, idx_x] = 1.0
                A_link[i * n_facilities + f, idx_y] = -1.0
        
        lb_link = np.zeros(n_x)
        ub_link = np.zeros(n_x)
        
        # Combine constraints
        # LinearConstraint(A, lb, ub)
        # We can combine them into one big matrix or use a list.
        # Using a list of LinearConstraint objects is cleaner in SciPy.
        
        constraints = [
            LinearConstraint(A_assign, lb_assign, ub_assign),
            LinearConstraint(A_cap, lb_cap, ub_cap),
            LinearConstraint(A_link, lb_link, ub_link)
        ]
        
        bounds = Bounds(lb, ub)
        
        # Solve
        # options={'presolve': True, 'time_limit': 100} might be useful but HiGHS defaults are usually fine for this size.
        # We don't have a direct timeout in the standard interface that guarantees return, 
        # but the problem is small (100x20 = 2000 binary vars + 20 binary vars).
        # HiGHS should solve this in milliseconds.
        
        res = milp(c=c_obj, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0 and res.x is not None:
            x_vals = res.x[:n_x].reshape(n_customers, n_facilities)
            # Find the facility with x=1 for each customer
            result = np.argmax(x_vals, axis=1)
            
            # Verify validity just in case
            load = np.zeros(n_facilities)
            for i in range(n_customers):
                f = result[i]
                load[f] += demands[i]
            
            if np.all(load <= capacities + 1e-6):
                return result
            else:
                # Solver returned a solution that violates capacity? Unlikely if constraints are correct.
                # Fall back to greedy
                return greedy_fallback()
        else:
            # Solver failed
            return greedy_fallback()

    except Exception:
        # Any error, fall back to greedy
        return greedy_fallback()
