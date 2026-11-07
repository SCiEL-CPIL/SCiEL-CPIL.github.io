# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility respecting capacities.
    Minimizes open_cost (for open facilities) + assign_cost.
    Strategy: Try MILP via scipy.optimize.milp. Fallback to Greedy + Local Search.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    deadline = time.time() + 115.0  # safety margin for 120s limit

    # Fallback Heuristic
    def _heuristic():
        fac_order = np.argsort(assign_cost, axis=1)
        load = np.zeros(n_fac, dtype=np.int64)
        result = np.full(n_cust, -1, dtype=np.int64)
        order = np.argsort(-demands)
        
        for i in order:
            assigned = False
            for f in fac_order[i]:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            if not assigned:
                for f in range(n_fac):
                    if load[f] + demands[i] <= capacities[f]:
                        load[f] += demands[i]
                        result[i] = f
                        assigned = True
                        break
            if not assigned:
                result[i] = 0
                load[0] += demands[i]
        
        # Simple local search improvement
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(n_cust):
                current_f = result[i]
                current_cost = assign_cost[i, current_f]
                # Try moving to a better facility
                best_f = current_f
                best_cost = current_cost
                # Check if moving away from current facility saves open cost (if it becomes empty)
                # This is complex to check quickly without full re-eval, so just check assignment cost + potential open cost change roughly?
                # For simplicity, just check assignment cost delta + open cost delta if facility becomes empty or was empty.
                load_check = load.copy()
                for f in fac_order[i]:
                    if f == current_f:
                        continue
                    if load_check[f] + demands[i] <= capacities[f]:
                        # Estimate delta
                        delta_assign = assign_cost[i, f] - current_cost
                        delta_open = 0.0
                        # If current facility becomes empty, save open_cost
                        if load_check[current_f] == demands[i]:
                            delta_open -= open_cost[current_f]
                        # If new facility was empty, pay open_cost
                        if load_check[f] == 0:
                            delta_open += open_cost[f]
                        
                        if delta_assign + delta_open < -1e-9:
                            # Apply move
                            load[current_f] -= demands[i]
                            load[f] += demands[i]
                            result[i] = f
                            improved = True
                            break
        return result

    # Try MILP
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # Variables: y[0..19], x[0..1999]
        # Total vars: 20 + 100*20 = 2020
        n_vars = n_fac + n_cust * n_fac
        
        # Objective coefficients
        c = np.zeros(n_vars)
        c[:n_fac] = open_cost
        c[n_fac:] = assign_cost.flatten()
        
        # Constraints
        constraints = []
        
        # 1. Each customer assigned to exactly one facility
        # For each i: sum_j x[i,j] = 1
        A_assign = np.zeros((n_cust, n_vars))
        for i in range(n_cust):
            start = n_fac + i * n_fac
            A_assign[i, start:start+n_fac] = 1
        constraints.append(LinearConstraint(A_assign, 1, 1))
        
        # 2. Capacity constraints: sum_i d_i x[i,j] - C_j y_j <= 0
        A_cap = np.zeros((n_fac, n_vars))
        for j in range(n_fac):
            A_cap[j, j] = -capacities[j]
            for i in range(n_cust):
                idx = n_fac + i * n_fac + j
                A_cap[j, idx] = demands[i]
        constraints.append(LinearConstraint(A_cap, -np.inf, 0))
        
        # Bounds
        bounds = Bounds(0, 1)
        
        # Integrality: all binary
        integrality = np.ones(n_vars)
        
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.success:
            x_vars = res.x
            y_vars = x_vars[:n_fac]
            x_assign = x_vars[n_fac:].reshape(n_cust, n_fac)
            
            # Rounding: Since it's binary, values should be 0 or 1.
            # Find assigned facility for each customer
            result = np.argmax(x_assign, axis=1)
            
            # Verify validity
            load = np.zeros(n_fac, dtype=np.int64)
            valid = True
            for i in range(n_cust):
                f = result[i]
                if f < 0 or f >= n_fac:
                    valid = False
                    break
                load[f] += demands[i]
            
            if np.any(load > capacities):
                valid = False
                
            if valid:
                # Calculate cost to ensure it's good
                cost = np.sum(open_cost[load > 0]) + np.sum(assign_cost[np.arange(n_cust), result])
                # We could compare with heuristic cost, but MILP is exact so it should be >= optimal.
                # Just return it if valid.
                return result.astype(int)
                
    except Exception:
        pass

    # Fallback to heuristic
    return _heuristic().astype(int)

# EVOLVE-BLOCK-END
