# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog, milp, LinearConstraint, Bounds


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem: 10-node ring, 15 demands,
    each routed CW or CCW to minimize max edge load.
    Uses a custom Branch and Bound solver with LP relaxation bounds.
    """
    start_time = time.time()
    DEADLINE = 600.0  # seconds

    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]
    D = len(demands)

    # Precompute edge usage matrices
    A_cw = np.zeros((D, n), dtype=float)
    A_ccw = np.zeros((D, n), dtype=float)
    weights = np.zeros(D, dtype=float)

    for d, (s, t, w) in enumerate(demands):
        weights[d] = w
        e = s
        while e != t:
            A_cw[d, e % n] = 1
            e = (e + 1) % n
        A_ccw[d] = 1.0 - A_cw[d]

    # W_cw[d,e] = weight * 1 if demand d uses edge e when routed CW
    W_cw = A_cw * weights[:, None]
    W_ccw = A_ccw * weights[:, None]
    base_loads = W_cw.sum(axis=0)
    diff = W_ccw - W_cw  # diff[d,e] = W_ccw[d,e] - W_cw[d,e]

    def compute_loads(routing: np.ndarray) -> np.ndarray:
        return base_loads + diff.T @ routing

    def max_load(routing: np.ndarray) -> float:
        return float(np.max(compute_loads(routing)))

    best_routing = None
    best_val = float('inf')

    # Phase 1: Branch and Bound
    # Variables: x[0..D-1] in [0,1], x[D] = t (max load)
    # Minimize t
    # Subject to: base_loads[e] + sum_d diff[d,e] * x[d] <= t  for all e
    #             0 <= x[d] <= 1, x[d] integer for d < D
    
    # LP Relaxation Solver
    def solve_lp(fixed_vars: dict) -> tuple:
        """
        Solve LP relaxation with some variables fixed.
        Returns (objective_value, solution_vector) or (inf, None).
        """
        # Number of continuous variables: D (x) + 1 (t)
        # But some x are fixed.
        
        # We will build the LP for the remaining free variables.
        # Let's just use linprog with bounds for all D variables, and fix some by setting lb=ub.
        
        # Objective: minimize t -> c = [0, ..., 0, 1] (length D+1)
        c = np.zeros(D + 1)
        c[D] = 1.0
        
        # Constraints:
        # For each edge e: sum_d diff[d,e] * x[d] - t <= -base_loads[e]
        # A_ub shape: (n, D+1)
        A_ub = np.zeros((n, D + 1))
        b_ub = np.zeros(n)
        for e in range(n):
            A_ub[e, :D] = diff[:, e]
            A_ub[e, D] = -1.0
            b_ub[e] = -base_loads[e]
            
        # Bounds
        lb = np.zeros(D + 1)
        ub = np.ones(D + 1)
        ub[D] = np.inf
        
        # Apply fixed variables
        for d, val in fixed_vars.items():
            lb[d] = val
            ub[d] = val
            
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=list(zip(lb, ub)), method='highs')
            if res.status == 0:
                return res.fun, res.x
            else:
                return float('inf'), None
        except Exception:
            return float('inf'), None

    # Initial upper bound using a simple heuristic
    # Try all CW and all CCW
    r_all_cw = np.zeros(D)
    r_all_ccw = np.ones(D)
    val_cw = max_load(r_all_cw)
    val_ccw = max_load(r_all_ccw)
    
    # Try a greedy approach? Or just use the better of the two.
    if val_cw < val_ccw:
        best_routing = r_all_cw
        best_val = val_cw
    else:
        best_routing = r_all_ccw
        best_val = val_ccw
        
    # Also try rounding the LP solution of the root node
    root_val, root_sol = solve_lp({})
    if root_sol is not None:
        # Round to nearest integer
        x_rounded = (root_sol[:D] > 0.5).astype(float)
        val_rounded = max_load(x_rounded)
        if val_rounded < best_val:
            best_val = val_rounded
            best_routing = x_rounded

    # Branch and Bound
    # Stack of (fixed_vars, lower_bound)
    # We branch on the variable with the most fractional value in the LP solution
    # Or just branch in order 0..D-1 for simplicity, since D is small.
    
    # For better performance, we can branch on the most fractional variable.
    
    def branch_and_bound(fixed_vars: dict, current_lb: float, current_sol: np.ndarray):
        nonlocal best_val, best_routing
        
        if time.time() - start_time > DEADLINE - 10:
            return
            
        # Check if all variables are fixed
        if len(fixed_vars) == D:
            # Check if integer
            # Since we fixed them to 0 or 1, they are integer.
            # Compute actual max load
            x = np.array([fixed_vars[d] for d in range(D)])
            val = max_load(x)
            if val < best_val - 1e-9:
                best_val = val
                best_routing = x
            return

        # Find a variable to branch on
        # Prefer variables that are fractional in current_sol
        branch_var = None
        min_frac = 0.5
        for d in range(D):
            if d not in fixed_vars:
                val = current_sol[d]
                frac = min(val, 1.0 - val)
                if frac > min_frac:
                    min_frac = frac
                    branch_var = d
                    
        if branch_var is None:
            # All remaining variables are 0 or 1 in the LP solution?
            # If so, we can fix the next unfixed variable arbitrarily?
            # Or just pick the first unfixed one.
            for d in range(D):
                if d not in fixed_vars:
                    branch_var = d
                    break
        
        if branch_var is None:
            return

        # Branch 0
        fixed_vars_0 = dict(fixed_vars)
        fixed_vars_0[branch_var] = 0.0
        lb_0, sol_0 = solve_lp(fixed_vars_0)
        if lb_0 < best_val - 1e-9:
            branch_and_bound(fixed_vars_0, lb_0, sol_0)
            
        # Branch 1
        fixed_vars_1 = dict(fixed_vars)
        fixed_vars_1[branch_var] = 1.0
        lb_1, sol_1 = solve_lp(fixed_vars_1)
        if lb_1 < best_val - 1e-9:
            branch_and_bound(fixed_vars_1, lb_1, sol_1)

    # Start B&B from root
    # We already have root_val and root_sol
    if root_sol is not None and root_val < best_val:
         # Only branch if the LP lower bound is better than current best
         # Actually, we should always branch if the LP solution is not integer
         # and the lower bound is promising.
         branch_and_bound({}, root_val, root_sol)
    else:
         # If root LP was infeasible or worse, we rely on the heuristic best.
         # But usually root LP is feasible.
         pass

    # Phase 2: Local Search Polish
    # Even if B&B found an optimum, a quick local search can confirm or find a better one if B&B was cut short.
    def local_search(routing: np.ndarray) -> np.ndarray:
        """Perform 1-flip and 2-flip local search."""
        routing = routing.copy()
        cur = max_load(routing)
        improved = True
        while improved:
            improved = False
            # 1-flip
            for d in range(D):
                routing[d] = 1.0 - routing[d]
                new = max_load(routing)
                if new < cur - 1e-12:
                    cur = new
                    improved = True
                else:
                    routing[d] = 1.0 - routing[d]
            # 2-flip
            for d1 in range(D):
                for d2 in range(d1 + 1, D):
                    routing[d1] = 1.0 - routing[d1]
                    routing[d2] = 1.0 - routing[d2]
                    new = max_load(routing)
                    if new < cur - 1e-12:
                        cur = new
                        improved = True
                    else:
                        routing[d1] = 1.0 - routing[d1]
                        routing[d2] = 1.0 - routing[d2]
        return routing

    if best_routing is not None:
        best_routing = local_search(best_routing)
        best_val = max_load(best_routing)

    return best_routing

# EVOLVE-BLOCK-END
