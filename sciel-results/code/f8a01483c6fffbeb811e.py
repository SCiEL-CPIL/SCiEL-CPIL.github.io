import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Target tensor T[a, b, c] where:
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # T[a, b, c] = 1 if indices match, 0 otherwise
    
    N = n * m
    M = m * p
    P = p * n
    
    T = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s kill
    
    best_rank = N * M * P  # Fallback
    best_U = None
    best_V = None
    best_W = None
    
    # Helper to check if solution is valid with half-integer rounding
    def check_solution(U, V, W, tol=1e-9):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        T_calc = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        return np.allclose(T_calc, T, atol=tol), U_r, V_r, W_r
    
    # Pre-compute matricizations for ALS
    T_c = T.reshape(N * M, P)
    T_a_vec = T.reshape(N, M * P)
    T_b_vec = T.transpose(1, 0, 2).reshape(M, N * P)
    
    def als_iterate(U, V, W, R, max_iter=200, tol=1e-10):
        """Run ALS iterations. Returns (U, V, W, converged)."""
        prev_err = np.inf
        for it in range(max_iter):
            if time.time() - start_time > timeout:
                return U, V, W, False
            
            # Update W
            E_W = np.einsum('ar,br->ab r', U, V).reshape(N * M, R)
            try:
                W_T, _, _, _ = np.linalg.lstsq(E_W, T_c.T, rcond=None)
                W = W_T.T
            except:
                return U, V, W, False
            
            # Update U
            E_U = np.einsum('br,cr->bc r', V, W).reshape(M * P, R)
            try:
                U_T, _, _, _ = np.linalg.lstsq(E_U.T, T_a_vec.T, rcond=None)
                U = U_T.T
            except:
                return U, V, W, False
            
            # Update V
            E_V = np.einsum('ar,cr->ac r', U, W).reshape(N * P, R)
            try:
                V_T, _, _, _ = np.linalg.lstsq(E_V.T, T_b_vec.T, rcond=None)
                V = V_T.T
            except:
                return U, V, W, False
            
            # Check convergence
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.linalg.norm(T_calc - T)
            
            if abs(prev_err - error) < tol:
                return U, V, W, True
            prev_err = error
        
        return U, V, W, False
    
    # Search ranks from low to high
    # Known results: 3x3x3 = 23, 3x3x4 likely around 24-28
    for R in range(23, 37):
        if time.time() - start_time > timeout:
            break
        
        # Multiple random initializations
        for attempt in range(15):
            if time.time() - start_time > timeout:
                break
            
            # Different initialization strategies
            if attempt == 0:
                # Random normal
                np.random.seed(42 + R * 100 + attempt)
                U = np.random.randn(N, R)
                V = np.random.randn(M, R)
                W = np.random.randn(P, R)
            elif attempt == 1:
                # Sparse-ish init
                np.random.seed(43 + R * 100 + attempt)
                U = np.random.randn(N, R)
                V = np.random.randn(M, R)
                W = np.random.randn(P, R)
                # Add small structured component
                U[:, 0] += 0.5
                V[:, 0] += 0.5
                W[:, 0] += 0.5
            elif attempt == 2:
                # Scale variation
                np.random.seed(44 + R * 100 + attempt)
                U = np.random.randn(N, R) * 0.5
                V = np.random.randn(M, R) * 2.0
                W = np.random.randn(P, R) * 1.0
            else:
                # More random seeds
                np.random.seed(100 + R * 50 + attempt * 7)
                U = np.random.randn(N, R)
                V = np.random.randn(M, R)
                W = np.random.randn(P, R)
            
            U, V, W, converged = als_iterate(U, V, W, R, max_iter=150)
            
            if converged or time.time() - start_time > timeout:
                # Check if solution is valid with rounding
                valid, U_r, V_r, W_r = check_solution(U, V, W)
                if valid:
                    best_rank = R
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    break
        
        if best_U is not None:
            break
    
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
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
