"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 10]

def _build_expected_tensor(n=2, m=2, p=10):
    """Build the expected multiplication tensor."""
    expected = np.zeros((n*m, m*p, p*n))
    for i_A in range(n*m):
        i = i_A // m
        j_A = i_A % m
        for j_B in range(m*p):
            j_B_row = j_B // p
            k_B = j_B % p
            for k_C in range(p*n):
                k_C_col = k_C // n
                i_C = k_C % n
                if j_A == j_B_row and k_B == k_C_col and i == i_C:
                    expected[i_A, j_B, k_C] = 1.0
    return expected

def _build_strassen_rank35():
    """Build the rank-35 Strassen block-diagonal decomposition."""
    n, m, p = DIMENSIONS
    rank = 35
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    strassen_A = [
        [1, 0, 0, 1],  # P1
        [0, 0, 1, 1],  # P2
        [1, 0, 0, 0],  # P3
        [0, 0, 0, 1],  # P4
        [1, 1, 0, 0],  # P5
        [-1, 0, 1, 0], # P6
        [0, 1, 0, -1]  # P7
    ]
    
    strassen_B = [
        [1, 0, 0, 1],  # P1
        [1, 0, 0, 0],  # P2
        [0, 1, 0, -1], # P3
        [-1, 0, 1, 0], # P4
        [0, 0, 0, 1],  # P5
        [1, 1, 0, 0],  # P6
        [0, 0, 1, 1]   # P7
    ]
    
    w_contrib = [
        [0] * 4 for _ in range(7)
    ]
    w_contrib[0][0] = 1;  w_contrib[0][3] = 1   # P1: C00, C11
    w_contrib[1][2] = 1;  w_contrib[1][3] = -1  # P2: C10, C11
    w_contrib[2][1] = 1;  w_contrib[2][3] = 1   # P3: C01, C11
    w_contrib[3][0] = 1;  w_contrib[3][2] = 1   # P4: C00, C10
    w_contrib[4][0] = -1; w_contrib[4][1] = 1   # P5: C00, C01
    w_contrib[5][3] = 1                              # P6: C11
    w_contrib[6][0] = 1                              # P7: C00
    
    for b in range(5):
        for r in range(7):
            col = b * 7 + r
            U[0, col] = strassen_A[r][0]
            U[1, col] = strassen_A[r][1]
            U[2, col] = strassen_A[r][2]
            U[3, col] = strassen_A[r][3]
            
            k0 = 2 * b
            k1 = 2 * b + 1
            V[0 * p + k0, col] = strassen_B[r][0]
            V[0 * p + k1, col] = strassen_B[r][1]
            V[1 * p + k0, col] = strassen_B[r][2]
            V[1 * p + k1, col] = strassen_B[r][3]
            
            W[(2 * b) * 2 + 0, col] = w_contrib[r][0]
            W[(2 * b + 1) * 2 + 0, col] = w_contrib[r][1]
            W[(2 * b) * 2 + 1, col] = w_contrib[r][2]
            W[(2 * b + 1) * 2 + 1, col] = w_contrib[r][3]
            
    return U, V, W

def _try_rank(target_rank, expected, n=2, m=2, p=10, seed=42, time_limit=10):
    """Attempt to find a rank-target_rank decomposition."""
    start_time = time.time()
    U35, V35, W35 = _build_strassen_rank35()
    rng = np.random.RandomState(seed)
    
    best_err = float('inf')
    best_result = None
    
    dim_U = n*m * target_rank
    dim_V = m*p * target_rank
    dim_W = p*n * target_rank
    
    def make_objective_and_gradient(x0):
        def objective(x):
            U = x[:dim_U].reshape(n*m, target_rank)
            V = x[dim_U:dim_U+dim_V].reshape(m*p, target_rank)
            W = x[dim_U+dim_V:].reshape(p*n, target_rank)
            T = np.einsum('ir,jr,kr->ijk', U, V, W)
            residual = T - expected
            # Stronger regularization to break scaling symmetry
            reg = 1e-4 * np.sum((np.linalg.norm(V, axis=0) - 1)**2)
            return float(np.sum(residual**2) + reg)
        
        def gradient(x):
            U = x[:dim_U].reshape(n*m, target_rank)
            V = x[dim_U:dim_U+dim_V].reshape(m*p, target_rank)
            W = x[dim_U+dim_V:].reshape(p*n, target_rank)
            T = np.einsum('ir,jr,kr->ijk', U, V, W)
            residual = T - expected
            
            grad_U = np.einsum('jr,kr,ijk->ir', V, W, residual)
            grad_V = np.einsum('ir,kr,ijk->jr', U, W, residual)
            grad_W = np.einsum('ir,jr,ijk->kr', U, V, residual)
            
            # Gradient of regularization
            norms = np.linalg.norm(V, axis=0)
            reg_grad_V = 2e-4 * (norms - 1) * (V / (norms + 1e-12))
            
            return np.concatenate([grad_U.flatten(), (grad_V + reg_grad_V).flatten(), grad_W.flatten()])
        
        return objective, gradient
    
    # Multiple initialization strategies
    init_strategies = []
    
    # Strategy 1: Strassen subset + small noise
    for _ in range(3):
        idx = rng.choice(35, size=target_rank, replace=False)
        U0 = U35[:, idx].copy() + rng.randn(n*m, target_rank) * 0.05
        V0 = V35[:, idx].copy() + rng.randn(m*p, target_rank) * 0.05
        W0 = W35[:, idx].copy() + rng.randn(p*n, target_rank) * 0.05
        init_strategies.append((U0, V0, W0))
    
    # Strategy 2: Random initialization with structure
    for _ in range(2):
        U0 = rng.randn(n*m, target_rank) * 0.5
        V0 = rng.randn(m*p, target_rank) * 0.5
        W0 = rng.randn(p*n, target_rank) * 0.5
        init_strategies.append((U0, V0, W0))
    
    # Strategy 3: Partial Strassen with some random
    for _ in range(2):
        idx = rng.choice(35, size=min(target_rank, 20), replace=False)
        U0 = U35[:, idx].copy()
        V0 = V35[:, idx].copy()
        W0 = W35[:, idx].copy()
        
        if target_rank > len(idx):
            U0 = np.hstack([U0, rng.randn(n*m, target_rank - len(idx)) * 0.3])
            V0 = np.hstack([V0, rng.randn(m*p, target_rank - len(idx)) * 0.3])
            W0 = np.hstack([W0, rng.randn(p*n, target_rank - len(idx)) * 0.3])
        init_strategies.append((U0, V0, W0))
    
    for attempt, (U0, V0, W0) in enumerate(init_strategies):
        if time.time() - start_time > time_limit:
            break
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        objective, gradient = make_objective_and_gradient(x0)
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', jac=gradient,
                          options={'maxiter': 1000, 'gtol': 1e-12, 'maxls': 50})
            if res.fun < best_err:
                best_err = res.fun
                U = res.x[:dim_U].reshape(n*m, target_rank)
                V = res.x[dim_U:dim_U+dim_V].reshape(m*p, target_rank)
                W = res.x[dim_U+dim_V:].reshape(p*n, target_rank)
                best_result = (U, V, W)
        except Exception:
            pass

    # Final polish pass if we got close - multiple stages
    if best_result is not None and best_err < 1e-4:
        U, V, W = best_result
        x_polish = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        objective, gradient = make_objective_and_gradient(x_polish)
        
        # Stage 1: Standard polish
        try:
            res_polish = minimize(objective, x_polish, method='L-BFGS-B', jac=gradient,
                                 options={'maxiter': 3000, 'gtol': 1e-14, 'maxls': 100})
            if res_polish.fun < best_err:
                best_err = res_polish.fun
                U = res_polish.x[:dim_U].reshape(n*m, target_rank)
                V = res_polish.x[dim_U:dim_U+dim_V].reshape(m*p, target_rank)
                W = res_polish.x[dim_U+dim_V:].reshape(p*n, target_rank)
                best_result = (U, V, W)
        except Exception:
            pass
        
        # Stage 2: If still not perfect, try with reduced regularization
        if best_err > 1e-8:
            def objective_noreg(x):
                U = x[:dim_U].reshape(n*m, target_rank)
                V = x[dim_U:dim_U+dim_V].reshape(m*p, target_rank)
                W = x[dim_U+dim_V:].reshape(p*n, target_rank)
                T = np.einsum('ir,jr,kr->ijk', U, V, W)
                residual = T - expected
                return float(np.sum(residual**2))
            
            def gradient_noreg(x):
                U = x[:dim_U].reshape(n*m, target_rank)
                V = x[dim_U:dim_U+dim_V].reshape(m*p, target_rank)
                W = x[dim_U+dim_V:].reshape(p*n, target_rank)
                T = np.einsum('ir,jr,kr->ijk', U, V, W)
                residual = T - expected
                
                grad_U = np.einsum('jr,kr,ijk->ir', V, W, residual)
                grad_V = np.einsum('ir,kr,ijk->jr', U, W, residual)
                grad_W = np.einsum('ir,jr,ijk->kr', U, V, residual)
                
                return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            try:
                res_noreg = minimize(objective_noreg, x_polish, method='L-BFGS-B', jac=gradient_noreg,
                                   options={'maxiter': 5000, 'gtol': 1e-15, 'maxls': 150})
                if res_noreg.fun < best_err:
                    best_err = res_noreg.fun
                    U = res_noreg.x[:dim_U].reshape(n*m, target_rank)
                    V = res_noreg.x[dim_U:dim_U+dim_V].reshape(m*p, target_rank)
                    W = res_noreg.x[dim_U+dim_V:].reshape(p*n, target_rank)
                    best_result = (U, V, W)
            except Exception:
                pass

    if best_result is not None and best_err < 1e-6:
        U, V, W = best_result
        # Try rounding to half-integers or integers
        for factor in [2, 1]:
            U_r = np.round(U * factor) / factor
            V_r = np.round(V * factor) / factor
            W_r = np.round(W * factor) / factor
            T_r = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
            diff_r = np.max(np.abs(T_r - expected))
            if diff_r < 1e-6:
                return U_r, V_r, W_r
        # If no clean rounding, return the continuous solution
        return U, V, W
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS
    expected = _build_expected_tensor(n, m, p)
    
    U35, V35, W35 = _build_strassen_rank35()
    
    # Verify rank 35
    T35 = np.einsum('ir,jr,kr->ijk', U35, V35, W35)
    err35 = np.max(np.abs(T35 - expected))
    if err35 > 1e-10:
        return {"U": U35.tolist(), "V": V35.tolist(), "W": W35.tolist()}
    
    # Try lower ranks with more time budget
    for target_rank in [34, 33, 32, 31]:
        remaining = deadline - time.time()
        if remaining < 25:
            break
        time_limit = min(20, remaining - 5)
        result = _try_rank(target_rank, expected, n, m, p, seed=target_rank, time_limit=time_limit)
        if result is not None:
            U, V, W = result
            T = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.max(np.abs(T - expected))
            if err < 1e-6:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    return {"U": U35.tolist(), "V": V35.tolist(), "W": W35.tolist()}
