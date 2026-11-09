# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem: 10-node ring, 15 demands,
    each routed CW or CCW to minimize max edge load.
    Uses MILP formulation with a local search polish pass.
    """
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]
    D = len(demands)

    # Precompute edge usage matrices
    # A_cw[d, e] is 1 if demand d uses edge e when routed CW
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

    # We define x_i = 1 if demand i is routed CCW, 0 if CW.
    # Load on edge e = sum_i (A_cw[i,e]*w_i * (1-x_i) + A_ccw[i,e]*w_i * x_i)
    # Load_e = C_e + sum_i (w_i * (A_ccw[i,e] - A_cw[i,e])) * x_i
    
    # Let's compute the base load (all CW) and the delta for switching to CCW
    base_loads = (A_cw * weights[:, None]).sum(axis=0) # Shape (n,)
    delta = weights[:, None] * (A_ccw - A_cw) # Shape (D, n)
    
    # Load_e = base_loads[e] + sum_i delta[i, e] * x_i
    
    # Objective: Minimize t
    # Variables: [x_0, ..., x_14, t]
    # Total variables: D + 1
    
    c = np.zeros(D + 1)
    c[D] = 1.0 # Minimize t
    
    # Constraints:
    # For each edge e:
    # base_loads[e] + sum_i delta[i, e] * x_i - t <= 0
    # sum_i delta[i, e] * x_i - t <= -base_loads[e]
    
    # Matrix A for LinearConstraint
    # Rows correspond to edges (n rows)
    # Columns correspond to variables: x_0...x_D-1, t
    A_mat = np.zeros((n, D + 1))
    # Fill x columns with delta^T
    A_mat[:, :D] = delta.T
    # Fill t column with -1
    A_mat[:, D] = -1.0
    
    # Lower bounds: -inf
    # Upper bounds: -base_loads
    lb = -np.inf * np.ones(n)
    ub = -base_loads
    
    constraints = LinearConstraint(A_mat, lb, ub)
    
    # Variable bounds
    # x_i in [0, 1]
    # t in [0, sum(weights)] (safe upper bound)
    total_weight = sum(w for _, _, w in demands)
    var_bounds = Bounds(
        lb=[0.0] * D + [0.0],
        ub=[1.0] * D + [total_weight]
    )
    
    # Integrality: x_i are integers (binary), t is continuous
    integrality = np.array([1] * D + [0])
    
    # Deadline guard: 600s wall-clock limit, use 580s for safety
    start_time = time.time()
    deadline = start_time + 580.0
    
    # Solve MILP
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=var_bounds)
    
    if res.status == 0:
        routing = res.x[:D]
        # Ensure it's 0 or 1
        routing = np.where(routing > 0.5, 1.0, 0.0)
    else:
        # Fallback to brute force if MILP fails (should not happen)
        best_max_load = float('inf')
        best_routing = np.zeros(D, dtype=float)
        
        for mask in range(1 << D):
            if time.time() > deadline - 10.0:
                break
            routing = np.array([(mask >> d) & 1 for d in range(D)], dtype=float)
            loads = base_loads + delta.T @ routing
            max_load = np.max(loads)
            if max_load < best_max_load:
                best_max_load = max_load
                best_routing = routing.copy()
        
        return best_routing

    # Local search polish: try flipping single bits and pairs of bits
    # to see if we can improve the max load
    current_loads = base_loads + delta.T @ routing
    current_max = np.max(current_loads)
    
    # Single-bit flips
    improved = True
    while improved:
        improved = False
        for i in range(D):
            if time.time() > deadline - 5.0:
                break
            new_routing = routing.copy()
            new_routing[i] = 1.0 - new_routing[i]
            new_loads = base_loads + delta.T @ new_routing
            new_max = np.max(new_loads)
            if new_max < current_max - 1e-12:
                routing = new_routing
                current_loads = new_loads
                current_max = new_max
                improved = True
        if time.time() > deadline - 5.0:
            break
    
    # Pair-bit flips
    for i in range(D):
        if time.time() > deadline - 5.0:
            break
        for j in range(i + 1, D):
            if time.time() > deadline - 5.0:
                break
            new_routing = routing.copy()
            new_routing[i] = 1.0 - new_routing[i]
            new_routing[j] = 1.0 - new_routing[j]
            new_loads = base_loads + delta.T @ new_routing
            new_max = np.max(new_loads)
            if new_max < current_max - 1e-12:
                routing = new_routing
                current_loads = new_loads
                current_max = new_max
                improved = True
    
    return routing

# EVOLVE-BLOCK-END
