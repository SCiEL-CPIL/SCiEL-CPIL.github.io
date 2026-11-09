# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        value >= 0.5 -> route the demand counter-clockwise
        value < 0.5  -> route the demand clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one entry per demand.
    """
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)

    # Precompute for each demand:
    #   cw_mask[d, e] = 1 if edge e is used when routing demand d clockwise
    #   ccw_mask[d, e] = 1 if edge e is used when routing demand d counter-clockwise
    cw_mask = np.zeros((D, n), dtype=np.float64)
    ccw_mask = np.zeros((D, n), dtype=np.float64)
    weights = np.zeros(D, dtype=np.float64)

    for d, (s, t, w) in enumerate(demands):
        weights[d] = w
        if s == t:
            continue
        e = s
        while e != t:
            cw_mask[d, e] = 1.0
            e = (e + 1) % n
        # Counter-clockwise uses all other edges
        for e in range(n):
            if cw_mask[d, e] == 0:
                ccw_mask[d, e] = 1.0

    # Total weight on all demands
    total_weight = weights.sum()

    # Precompute coefficients for the linear load calculation
    # L_e(x) = sum_d w_d * ( (1-x_d) * cw_mask[d,e] + x_d * ccw_mask[d,e] )
    # L_e(x) = sum_d w_d * cw_mask[d,e] + sum_d w_d * (ccw_mask[d,e] - cw_mask[d,e]) * x_d
    # L_e(x) = base_load[e] + sum_d coeff[d,e] * x_d
    # where coeff[d,e] = w_d * (ccw_mask[d,e] - cw_mask[d,e])
    
    base_load = (weights[:, np.newaxis] * cw_mask).sum(axis=0)
    coeff = weights[:, np.newaxis] * (ccw_mask - cw_mask)

    def compute_loads(x):
        """
        x: array of shape (D,) with values in [0, 1]
        Returns load vector of shape (n,)
        """
        return base_load + x @ coeff

    def smooth_objective(x):
        """
        Log-Sum-Exp approximation of the maximum load.
        """
        loads = compute_loads(x)
        # Use log-sum-exp for numerical stability
        alpha = 50.0
        max_load = loads.max()
        # f(x) = sum(exp(alpha * loads))
        # To avoid overflow, shift by max
        exp_terms = np.exp(alpha * (loads - max_load))
        return np.log(exp_terms.sum()) + alpha * max_load

    def smooth_gradient(x):
        """
        Gradient of the smooth objective.
        """
        loads = compute_loads(x)
        alpha = 50.0
        max_load = loads.max()
        exp_terms = np.exp(alpha * (loads - max_load))
        sum_exp = exp_terms.sum()
        
        # d/dx_d sum_e exp(alpha * L_e) 
        # = sum_e exp(alpha * L_e) * alpha * dL_e/dx_d
        # dL_e/dx_d = coeff[d, e]
        
        # grad_d = alpha * sum_e exp(alpha * L_e) * coeff[d, e]
        # We can compute this as alpha * (exp_terms @ coeff.T)
        grad = alpha * (exp_terms @ coeff.T)
        return grad

    start_time = time.time()
    deadline = 600  # safety margin

    # Initial guess: 0.5 for all
    x0 = np.full(D, 0.5)
    
    # Bounds for x
    bounds = [(0, 1) for _ in range(D)]

    best_solution = np.zeros(D, dtype=np.int8)
    best_max_load = float('inf')

    # Run optimization
    try:
        res = minimize(
            smooth_objective,
            x0,
            method='L-BFGS-B',
            jac=smooth_gradient,
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if time.time() - start_time > deadline:
             # Fallback to simple heuristic if timeout
             pass

        x_opt = res.x
        # Round to binary
        binary_solution = (x_opt >= 0.5).astype(np.int8)
        
        # Calculate actual max load for this binary solution
        flips = binary_solution.astype(bool)
        load = base_load + coeff[flips].sum(axis=0)
        current_max_load = load.max()
        
        if current_max_load < best_max_load:
            best_max_load = current_max_load
            best_solution = binary_solution

    except Exception as e:
        # If optimization fails, fall back to a simple heuristic
        # Try all CW and all CCW
        for sol in [np.zeros(D, dtype=np.int8), np.ones(D, dtype=np.int8)]:
            flips = sol.astype(bool)
            load = base_load + coeff[flips].sum(axis=0)
            current_max_load = load.max()
            if current_max_load < best_max_load:
                best_max_load = current_max_load
                best_solution = sol

    # Polish: Local Search
    # Try flipping each bit and see if it improves the max load
    current_solution = best_solution.copy()
    current_max_load = best_max_load
    
    improved = True
    while improved and (time.time() - start_time < deadline):
        improved = False
        for d in range(D):
            # Flip demand d
            new_solution = current_solution.copy()
            new_solution[d] = 1 - new_solution[d]
            
            flips = new_solution.astype(bool)
            load = base_load + coeff[flips].sum(axis=0)
            new_max_load = load.max()
            
            if new_max_load < current_max_load:
                current_solution = new_solution
                current_max_load = new_max_load
                improved = True

    best_solution = current_solution

    return best_solution.astype(float)


# EVOLVE-BLOCK-END
