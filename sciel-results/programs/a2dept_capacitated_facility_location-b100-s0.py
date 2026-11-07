"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _try_flow(demands, capacities, open_cost, assign_cost, deadline):
    """
    Attempt to solve the Capacitated Facility Location Problem using a 
    transformation to Min-Cost Flow.
    
    This works by creating a network where:
    Source -> Facility (capacity=inf, cost=open_cost)
    Facility -> Customer (capacity=demand, cost=assign_cost)
    Customer -> Sink (capacity=demand, cost=0)
    
    However, scipy's linear_sum_assignment is for uncapacitated assignment.
    For capacitated, we can use scipy.optimize.linprog or networkx, but 
    networkx might be slow or unavailable. 
    Wait, the problem is 100 customers, 20 facilities. 
    Total demand is at most 100*35 = 3500.
    Total capacity is likely sufficient.
    
    A robust approach for small N is to use a linear programming solver 
    if available, or a specialized algorithm. 
    Since SciPy is available, we can use `scipy.optimize.linprog` with 
    the simplex method. The number of variables is N_cust * N_fac = 2000.
    Constraints: 
    1. Sum_x(i,j) = 1 for all i (100 constraints)
    2. Sum_i x(i,j) * demands[i] <= capacities[j] for all j (20 constraints)
    3. x(i,j) in {0,1}. Relaxing to [0,1] works because the constraint 
       matrix is totally unimodular for this bipartite matching with capacities?
       Actually, the constraint matrix for the assignment problem is TU.
       With capacities, it's a transportation problem, which is also TU.
       So LP relaxation gives integer solutions.
       
    Let's use linprog.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Variables x_{i,j} for i in 0..n_cust-1, j in 0..n_fac-1
    # Total vars = n_cust * n_fac
    n_vars = n_cust * n_fac
    
    # Objective: Minimize sum(open_cost[j] * y_j) + sum(assign_cost[i,j] * x_{i,j})
    # We need to handle the fixed cost y_j.
    # y_j = 1 if sum_i x_{i,j} > 0.
    # This is a mixed integer program. Linprog doesn't do integers.
    # However, if we assume we know which facilities are open, we can solve it.
    # Or, we can iterate over subsets? 2^20 is too big.
    
    # Alternative: Since N is small, we can use a heuristic that is stronger than greedy.
    # Or, we can use `scipy.optimize.milp` (available in newer scipy versions).
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
    except ImportError:
        return None

    # Variables:
    # x_{i,j} (binary) for i in N, j in F. Size: 100 * 20 = 2000
    # y_j (binary) for j in F. Size: 20
    # Total variables: 2020
    
    # Objective coefficients:
    # c_x[i,j] = assign_cost[i,j]
    # c_y[j] = open_cost[j]
    
    # Constraints:
    # 1. For each customer i: sum_j x_{i,j} = 1
    # 2. For each facility j: sum_i x_{i,j} * demands[i] <= capacities[j]
    # 3. For each facility j: x_{i,j} <= y_j for all i (if x is 1, y must be 1)
    
    # This is a standard MIP.
    
    c = np.zeros(n_vars + n_fac)
    # First n_vars are x, next n_fac are y
    for i in range(n_cust):
        for j in range(n_fac):
            c[i * n_fac + j] = assign_cost[i, j]
    for j in range(n_fac):
        c[n_vars + j] = open_cost[j]
    
    # Constraints in A_ub @ vars <= b_ub form is tricky for equality.
    # Let's use LinearConstraint.
    
    # A for x variables (n_cust * n_fac) and y variables (n_fac)
    # Total rows:
    # 1. n_cust rows for "sum x = 1"
    # 2. n_fac rows for "capacity"
    # 3. n_cust * n_fac rows for "x <= y"
    
    n_constraints = n_cust + n_fac + n_cust * n_fac
    
    A = np.zeros((n_constraints, n_vars + n_fac))
    lb = np.zeros(n_constraints)
    ub = np.zeros(n_constraints)
    
    row_idx = 0
    
    # 1. Assignment constraints: sum_j x_{i,j} = 1
    for i in range(n_cust):
        for j in range(n_fac):
            A[row_idx, i * n_fac + j] = 1.0
        lb[row_idx] = 1.0
        ub[row_idx] = 1.0
        row_idx += 1
        
    # 2. Capacity constraints: sum_i x_{i,j} * demands[i] <= capacities[j]
    for j in range(n_fac):
        for i in range(n_cust):
            A[row_idx, i * n_fac + j] = demands[i]
        lb[row_idx] = 0.0
        ub[row_idx] = capacities[j]
        row_idx += 1
        
    # 3. Linking constraints: x_{i,j} - y_j <= 0
    for i in range(n_cust):
        for j in range(n_fac):
            A[row_idx, i * n_fac + j] = 1.0
            A[row_idx, n_vars + j] = -1.0
            lb[row_idx] = -np.inf
            ub[row_idx] = 0.0
            row_idx += 1
            
    # Bounds for variables
    # x_{i,j} in [0, 1]
    # y_j in [0, 1]
    vars_bounds = Bounds(lb=np.zeros(n_vars + n_fac), ub=np.ones(n_vars + n_fac))
    
    # Integrality: x and y should be integers.
    # milp expects an array of 0 (continuous) or 1 (integer).
    integrality = np.ones(n_vars + n_fac)
    
    constraints = LinearConstraint(A, lb, ub)
    
    # Check time
    if time.time() > deadline:
        return None
        
    try:
        res = milp(c, constraints=constraints, integrality=integrality, bounds=vars_bounds)
        if res.status == 0:
            # Extract x
            x_flat = res.x[:n_vars]
            # Round to nearest integer to handle floating point errors
            x_flat = np.round(x_flat).astype(int)
            # Reshape to (n_cust, n_fac)
            x = x_flat.reshape(n_cust, n_fac)
            
            # Verify validity
            result = np.zeros(n_cust, dtype=int)
            load = np.zeros(n_fac)
            for i in range(n_cust):
                # Find assigned facility
                idx = np.argmax(x[i])
                if x[i, idx] == 1:
                    result[i] = idx
                    load[idx] += demands[i]
                else:
                    # Fallback if MIP failed to assign (should not happen with equality constraints)
                    return None
            
            for j in range(n_fac):
                if load[j] > capacities[j] + 1e-6:
                    return None
                    
            return result
        else:
            return None
    except Exception:
        return None

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Try MIP solver
    result = _try_flow(demands, capacities, open_cost, assign_cost, deadline)
    
    if result is not None:
        return result
        
    # Fallback: Greedy (Largest demand first, cheapest feasible facility)
    # This ensures we always return a valid result if possible.
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    
    # Sort customers by demand descending
    order = np.argsort(-demands)
    
    for i in order:
        # Sort facilities by cost for this customer
        fac_order = np.argsort(assign_cost[i])
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        else:
            # If no facility found, this instance is infeasible?
            # The problem statement implies a solution exists.
            # If not, we must assign to something to return an array, 
            # but it will be invalid. 
            # We'll just pick the first one to avoid crash, though score will be 0.
            result[i] = 0 
            
    return result
# EVOLVE-BLOCK-END
