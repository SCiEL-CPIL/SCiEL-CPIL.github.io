# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.
    """
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # Precompute edge usage for each demand in each direction
    # x_i = 0 -> clockwise, x_i = 1 -> counter-clockwise
    # Clockwise from s to t uses edges s, s+1, ..., t-1 (mod n)
    # Counter-clockwise uses all other edges
    
    # A[i, e] = 1 if demand i uses edge e when routed clockwise
    A = np.zeros((D, n))
    for i, (s, t, w) in enumerate(demands):
        if s == t:
            continue
        # Clockwise edges: s, s+1, ..., t-1 (mod n)
        e = s
        while e != t:
            A[i, e] = 1
            e = (e + 1) % n
    
    # B[i, e] = 1 if demand i uses edge e when routed counter-clockwise
    # This is just 1 - A[i, e] for the ring (since every demand uses exactly one path)
    B = 1.0 - A
    
    # Weight vector
    w = np.array([d[2] for d in demands], dtype=float)
    
    # Edge load when all routed clockwise: L_e = sum_i w_i * A[i,e]
    # Edge load when all routed counter-clockwise: L_e = sum_i w_i * B[i,e]
    # If demand i is routed with variable x_i (0 or 1), the load on edge e is:
    # L_e = sum_i w_i * (A[i,e] * (1 - x_i) + B[i,e] * x_i)
    #     = sum_i w_i * A[i,e] + sum_i w_i * (B[i,e] - A[i,e]) * x_i
    #     = L0_e + sum_i w_i * (B[i,e] - A[i,e]) * x_i
    
    L0 = A.T @ w  # L0[e] = sum_i w_i * A[i,e]
    # C[e, i] = w_i * (B[i,e] - A[i,e])
    C = np.zeros((n, D))
    for e in range(n):
        for i in range(D):
            C[e, i] = w[i] * (B[i, e] - A[i, e])
    
    # So L(x) = L0 + C @ x, where x is (D,)
    
    def max_load(x):
        loads = L0 + C @ x
        return np.max(loads)
    
    def objective(x):
        loads = L0 + C @ x
        # Smooth approximation using log-sum-exp
        alpha = 10.0
        # For numerical stability
        m = np.max(loads)
        if m > 1e10:
            return m
        return m + (1.0 / alpha) * np.log(np.sum(np.exp(alpha * (loads - m))))
    
    def gradient(x):
        loads = L0 + C @ x
        m = np.max(loads)
        alpha = 10.0
        if m > 1e10:
            return C.T @ np.ones(n)
        exp_term = np.exp(alpha * (loads - m))
        sum_exp = np.sum(exp_term)
        # d/dx_j of (m + (1/alpha) log(sum exp)) 
        # The max part is non-smooth, but for the smooth part:
        # d/dx_j [(1/alpha) log(sum_i exp(alpha*(L_i - m)))] = (1/alpha) * sum_i alpha * exp_term_i / sum_exp * dL_i/dx_j
        # = sum_i (exp_term_i / sum_exp) * dL_i/dx_j
        # = C.T @ (exp_term / sum_exp)
        # Note: the m term's gradient is handled implicitly; for the smooth surrogate,
        # we use the softmax weights
        weights = exp_term / sum_exp
        return C.T @ weights
    
    # Start from all clockwise (x=0) and all counter-clockwise (x=1), plus random starts
    best_x = None
    best_score = np.inf
    
    start_times = [np.zeros(D), np.ones(D)]
    # Add some random starts
    np.random.seed(42)
    for _ in range(10):
        start_times.append(np.random.rand(D))
    
    deadline = time.time() + 580  # safety margin
    
    for x0 in start_times:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                objective,
                x0,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            x_opt = res.x
        except Exception:
            continue
        
        # Round to binary
        x_bin = (x_opt >= 0.5).astype(float)
        
        # Evaluate the true max load for the rounded solution
        loads = L0 + C @ x_bin
        ml = np.max(loads)
        
        if ml < best_score:
            best_score = ml
            best_x = x_bin.copy()
        
        # Also try the opposite rounding
        x_bin_alt = (x_opt < 0.5).astype(float)
        loads_alt = L0 + C @ x_bin_alt
        ml_alt = np.max(loads_alt)
        if ml_alt < best_score:
            best_score = ml_alt
            best_x = x_bin_alt.copy()
    
    # Local search: try flipping each single demand
    if best_x is None:
        best_x = np.zeros(D)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(D):
            trial = best_x.copy()
            trial[i] = 1.0 - trial[i]
            loads = L0 + C @ trial
            ml = np.max(loads)
            if ml < best_score - 1e-12:
                best_score = ml
                best_x = trial
                improved = True
    
    # Try pairwise flips
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(D):
                for j in range(i + 1, D):
                    trial = best_x.copy()
                    trial[i] = 1.0 - trial[i]
                    trial[j] = 1.0 - trial[j]
                    loads = L0 + C @ trial
                    ml = np.max(loads)
                    if ml < best_score - 1e-12:
                        best_score = ml
                        best_x = trial
                        improved = True
                        break
                if improved:
                    break
    
    return best_x

# EVOLVE-BLOCK-END
