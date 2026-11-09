# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        0 -> route the demand clockwise
        1 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    n = 10
    D = 15
    demands = [
        (0, 5, 3),
        (1, 7, 2),
        (2, 9, 4),
        (3, 6, 1),
        (0, 4, 5),
        (2, 8, 2),
        (5, 9, 3),
        (1, 4, 2),
        (6, 0, 4),
        (3, 8, 1),
        (7, 2, 3),
        (4, 9, 2),
        (8, 1, 5),
        (0, 6, 1),
        (5, 2, 2),
    ]

    # Build CW_use matrix: D x n
    CW_use = np.zeros((D, n), dtype=int)
    for d, (s, t, w) in enumerate(demands):
        if s == t:
            CW_use[d, :] = 1
        elif s < t:
            CW_use[d, s:t] = 1
        else:
            CW_use[d, s:] = 1
            CW_use[d, :t] = 1

    weights = np.array([w for _, _, w in demands], dtype=float)
    base_load = CW_use.T @ weights  # shape (n,)

    # Coefficient matrix A: n x D, where A[e, d] = w_d * (1 - 2 * CW_use[d][e])
    # This represents the change in load on edge e if we flip d from CW to CCW?
    # Let's verify.
    # Load_e(x) = B_e + sum_d A[e,d] x_d
    # If x_d=0 (CW), term is 0. Load contribution is part of B_e.
    # If x_d=1 (CCW), term is A[e,d].
    # We want Load_e(1) = w_d if CCW uses edge e, else 0.
    # We want Load_e(0) = w_d if CW uses edge e, else 0.
    # B_e = sum w_d * CW_use[d,e]
    # If CW_use[d,e]=1, B_e has w_d.
    #   If x_d=0, Load = w_d. Correct.
    #   If x_d=1, Load = w_d + A[e,d]. We want 0. So A[e,d] must be -w_d.
    #   Formula: w_d * (1 - 2*1) = -w_d. Correct.
    # If CW_use[d,e]=0, B_e has 0.
    #   If x_d=0, Load = 0. Correct.
    #   If x_d=1, Load = 0 + A[e,d]. We want w_d (since CCW uses it). So A[e,d] must be w_d.
    #   Formula: w_d * (1 - 2*0) = w_d. Correct.
    
    A = np.zeros((n, D), dtype=float)
    for d in range(D):
        A[:, d] = weights[d] * (1 - 2 * CW_use[d, :])

    def solve_lp(bound_dict=None):
        """
        Solve the LP relaxation.
        bound_dict: dict of {var_index: (lb, ub)} to fix variables.
        """
        # Variables: x_0 ... x_{D-1}, L
        # Total D+1 variables
        
        c = np.zeros(D + 1)
        c[D] = 1.0  # Minimize L

        # Constraints: L_e(x) <= L  =>  -L - sum A[e,d] x_d <= -B_e
        A_ub = np.zeros((n, D + 1))
        b_ub = np.zeros(n)
        
        for e in range(n):
            A_ub[e, :D] = -A[e, :]
            A_ub[e, D] = -1.0
            b_ub[e] = -base_load[e]

        bounds = []
        for d in range(D):
            if bound_dict and d in bound_dict:
                lb, ub = bound_dict[d]
            else:
                lb, ub = 0, 1
            bounds.append((lb, ub))
        
        # L >= 0
        bounds.append((0, None))

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        
        if not res.success:
            return None, None, None
            
        x = res.x
        L_val = x[D]
        x_vars = x[:D]
        
        # Check if integer
        is_int = np.allclose(x_vars, np.round(x_vars))
        
        return L_val, x_vars, is_int

    # Initial LP solve
    L_lb, x_lp, is_int = solve_lp()
    
    if L_lb is None:
        # Fallback
        return np.zeros(D)

    if is_int:
        return x_lp.astype(float)

    # Branch and Bound
    best_L = float('inf')
    best_x = None
    
    # Stack for B&B: (lower_bound, bound_dict)
    stack = [(L_lb, {})]
    
    # We can also get an initial upper bound by rounding the LP solution
    x_init = np.round(x_lp).astype(float)
    # Calculate max load for x_init
    loads_init = base_load + A @ x_init
    L_init = np.max(loads_init)
    best_L = L_init
    best_x = x_init
    
    # Sort stack by lower bound to find good solutions early
    # Since we only have one node initially, just start.
    
    while stack:
        # Get node with lowest lower bound
        # For small trees, simple list sort or pop(0) is fine, but let's use a heap or just sort
        stack.sort(key=lambda item: item[0])
        node_lb, bounds_dict = stack.pop(0)
        
        # Prune if node lower bound is worse than current best
        if node_lb >= best_L:
            continue
            
        # Solve LP for this node
        L_node, x_node, is_int_node = solve_lp(bounds_dict)
        
        if L_node is None:
            continue
            
        if L_node >= best_L:
            continue
            
        if is_int_node:
            # Found a better integer solution
            best_L = L_node
            best_x = x_node
            continue
        else:
            # Branch on a fractional variable
            # Choose variable with largest deviation from 0.5 or just first fractional
            frac_vars = np.where(np.abs(x_node - np.round(x_node)) > 1e-6)[0]
            if len(frac_vars) == 0:
                # Should not happen if is_int_node is False, but safety
                continue
                
            # Heuristic: branch on variable closest to 0.5? Or just first?
            # Let's pick the one with value closest to 0.5 to split space evenly
            # Or just the first one.
            d_branch = frac_vars[0]
            
            # Branch 1: x_d <= floor(x_d)
            # Since x_d in [0,1], floor is 0 if x_d < 1, 1 if x_d=1.
            # If x_d is fractional, floor(x_d) is 0.
            # So x_d = 0.
            bounds_1 = bounds_dict.copy()
            bounds_1[d_branch] = (0, 0)
            
            # Branch 2: x_d >= ceil(x_d)
            # Ceil is 1 if x_d > 0.
            # So x_d = 1.
            bounds_2 = bounds_dict.copy()
            bounds_2[d_branch] = (1, 1)
            
            # Push to stack
            # We don't know the lower bounds of children yet, so push with parent's lb?
            # Or we can solve them immediately to get lb?
            # To keep stack sorted by lb, we should solve children or estimate.
            # For simplicity and efficiency in this small case, we can just push them
            # with the current node's lb (which is a valid lower bound for children)
            # and let the loop handle sorting.
            stack.append((node_lb, bounds_1))
            stack.append((node_lb, bounds_2))
            
            # Optimization: Solve children immediately to get better lower bounds for pruning?
            # This might be faster for small D.
            # Let's try solving immediately and adding only if promising.
            
    if best_x is None:
        return np.zeros(D)
        
    return best_x

# EVOLVE-BLOCK-END
