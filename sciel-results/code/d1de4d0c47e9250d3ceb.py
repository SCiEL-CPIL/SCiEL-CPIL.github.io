import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 5]

def _compute_error(U_flat, V_flat, W_flat, R):
    n, m, p = DIMENSIONS
    U = U_flat.reshape(n, m, R)
    V = V_flat.reshape(m, p, R)
    W = W_flat.reshape(p, n, R)
    
    T_computed = np.einsum('ijr,jkr,kir->ijk', U, V, W)
    T_target = np.ones((n, m, p))
    error = np.sum((T_computed - T_target) ** 2)
    return error

def _try_rank(R, time_budget=10.0):
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Total variables: n*m*R + m*p*R + p*n*R
    # For R=40: 12*40 + 20*40 + 15*40 = 480 + 800 + 600 = 1880 variables.
    # This is manageable for L-BFGS-B.
    
    max_restarts = 5
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
            
        # Initialize
        # Random initialization
        U_init = np.random.randn(n * m, R)
        V_init = np.random.randn(m * p, R)
        W_init = np.random.randn(p * n, R)
        
        # Scale them?
        # The product should be 1.
        # If we scale U by a, V by b, W by c, product scales by abc.
        # Let's normalize.
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def objective(x):
            U_flat = x[:n*m*R]
            V_flat = x[n*m*R : n*m*R + m*p*R]
            W_flat = x[n*m*R + m*p*R :]
            return _compute_error(U_flat, V_flat, W_flat, R)
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-15})
        except Exception:
            continue
            
        if res.success or res.fun < 1e-10:
            # Extract solution
            U_sol = res.x[:n*m*R].reshape(n*m, R)
            V_sol = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
            W_sol = res.x[n*m*R + m*p*R :].reshape(p*n, R)
            
            # Try to snap to half-integers
            # Round to nearest 0.5
            U_snap = np.round(U_sol * 2) / 2
            V_snap = np.round(V_sol * 2) / 2
            W_snap = np.round(W_sol * 2) / 2
            
            if _verify_decomposition(U_snap.tolist(), V_snap.tolist(), W_snap.tolist()):
                return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
                
            # Maybe integers?
            U_int = np.round(U_sol)
            V_int = np.round(V_sol)
            W_int = np.round(W_sol)
            if _verify_decomposition(U_int.tolist(), V_int.tolist(), W_int.tolist()):
                return U_int.tolist(), V_int.tolist(), W_int.tolist()
                
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find better rank
    # Start from a reasonable guess
    # Lower bound is 20. Schoolbook is 60.
    # Let's try 30, 35, 40, 45, 50.
    
    # Time budget management
    total_start = time.time()
    deadline = 140.0 # Safety margin
    
    best_result = None
    
    # List of ranks to try
    ranks_to_try = [30, 35, 40, 45, 50, 55]
    
    # We can also try to verify if a known decomposition exists
    # For now, run the search
    
    for R in ranks_to_try:
        if time.time() - total_start > deadline:
            break
            
        # Allocate remaining time
        remaining = deadline - (time.time() - total_start)
        time_budget = min(10.0, remaining * 0.8)
        
        if time_budget < 1.0:
            continue
            
        result = _try_rank(R, time_budget)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
