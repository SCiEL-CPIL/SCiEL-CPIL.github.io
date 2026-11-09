import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def ring_routing() -> np.ndarray:
    n_nodes = 10
    n_edges = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # Precompute edge usage matrices
    A = np.zeros((n_edges, D), dtype=int) # CW usage
    B = np.zeros((n_edges, D), dtype=int) # CCW usage
    
    for i, (s, t, w) in enumerate(demands):
        cw_edges = set()
        curr = s
        while curr != t:
            cw_edges.add(curr)
            curr = (curr + 1) % n_nodes
        
        for e in range(n_edges):
            if e in cw_edges:
                A[e, i] = 1
            else:
                B[e, i] = 1
                
    weights = np.array([d[2] for d in demands])
    
    # Helper to compute max load for a given assignment
    def get_max_load(x):
        # x is binary array. 0=CW, 1=CCW
        # Load_e = sum_i w_i * (x_i * B[e,i] + (1-x_i) * A[e,i])
        # Load_e = sum_i w_i * A[e,i] + sum_i w_i * x_i * (B[e,i] - A[e,i])
        loads = np.sum(weights * A, axis=1) + x @ (weights * (B - A)).T
        return np.max(loads)

    # 1. Try MIP
    best_x = None
    best_score = np.inf
    start_time = time.time()
    timeout = 500 # Leave margin for 630s limit
    
    try:
        c = np.zeros(D + 1)
        c[D] = 1.0
        integrality = np.array([1] * D + [0])
        lb = np.zeros(D + 1)
        ub = np.ones(D + 1)
        ub[D] = np.inf
        bounds = Bounds(lb, ub)
        
        constraints = []
        for e in range(n_edges):
            diff = B[e, :] - A[e, :]
            coeff_x = weights * diff
            row = np.concatenate([coeff_x, [-1.0]])
            rhs = -np.sum(weights * A[e, :])
            constraints.append(LinearConstraint(row.reshape(1, -1), -np.inf, rhs))
            
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        if res.success and res.x is not None:
            x_mip = np.round(res.x[:D]).astype(int)
            score_mip = get_max_load(x_mip)
            best_x = x_mip
            best_score = score_mip
    except Exception:
        pass

    # 2. Polish with Local Search / Brute Force if time permits or if MIP failed
    # Since 2^15 is small, we can do a full brute force if MIP didn't finish or to ensure optimality.
    # But let's stick to "optimization family" by doing a local search from the MIP result or random starts.
    # Actually, for 15 vars, full enumeration is 32k iterations, which is < 1ms in Python if optimized, or ~1s in pure Python.
    # Let's do a randomized local search to be safe and fast.
    
    if best_x is None:
        # Fallback: Random init
        best_x = np.random.randint(0, 2, D)
        best_score = get_max_load(best_x)
        
    # Local Search: Flip bits to improve
    improved = True
    while improved and (time.time() - start_time) < timeout:
        improved = False
        for i in range(D):
            new_x = best_x.copy()
            new_x[i] = 1 - new_x[i]
            new_score = get_max_load(new_x)
            if new_score < best_score:
                best_x = new_x
                best_score = new_score
                improved = True

    # Convert to required format: >= 0.5 is CCW (1), < 0.5 is CW (0)
    return best_x.astype(float)
