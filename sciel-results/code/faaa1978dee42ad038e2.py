import numpy as np
import time
from scipy.optimize import least_squares

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
    timeout = 115.0 # Safety margin for 120s limit
    
    best_rank = N * M * P # Fallback (Schoolbook)
    best_U = None
    best_V = None
    best_W = None
    
    # Pre-compute matricizations for fast ALS
    T_c = T.reshape(N * M, P)
    T_a = T.reshape(N, M * P)
    T_b = T.transpose(1, 0, 2).reshape(M, N * P)
    
    def check_solution(U, V, W, tol=1e-9):
        # Verify if U, V, W contain only half-integers
        # And if they reconstruct T exactly (within tol)
        # The problem states "integer/half-integer entries"
        # We check if 2*U, 2*V, 2*W are close to integers
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-6):
            return False, None, None, None
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-6):
            return False, None, None, None
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-6):
            return False, None, None, None
            
        # Snap to exact half-integers for verification
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        T_calc = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(T_calc, T, atol=tol):
            return True, U_r, V_r, W_r
        else:
            return False, None, None, None

    def polish(U, V, W):
        # Try to refine U, V, W to be exactly half-integers while matching T
        # Use a more efficient approach: fix two factors, solve for the third
        # via least squares, then check if the solution is close to half-integers
        
        for trial in range(3):
            if time.time() - start_time > timeout:
                break
            
            # Try fixing different pairs
            if trial == 0:
                # Fix V, W, solve for U
                E_U = np.einsum('br,cr->bc r', V, W).reshape(M * P, R)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(E_U.T, T.reshape(N, M*P).T, rcond=None)
                    U_new = U_new.T
                    valid, U_r, V_r, W_r = check_solution(U_new, V, W)
                    if valid:
                        return True, U_r, V_r, W_r
                except:
                    pass
            elif trial == 1:
                # Fix U, W, solve for V
                E_V = np.einsum('ar,cr->ac r', U, W).reshape(N * P, R)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(E_V.T, T.transpose(1,0,2).reshape(M, N*P).T, rcond=None)
                    V_new = V_new.T
                    valid, U_r, V_r, W_r = check_solution(U, V_new, W)
                    if valid:
                        return True, U_r, V_r, W_r
                except:
                    pass
            else:
                # Fix U, V, solve for W
                E_W = np.einsum('ar,br->ab r', U, V).reshape(N * M, R)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(E_W, T.reshape(N*M, P).T, rcond=None)
                    W_new = W_new.T
                    valid, U_r, V_r, W_r = check_solution(U, V, W_new)
                    if valid:
                        return True, U_r, V_r, W_r
                except:
                    pass
        
        # Also try a direct snap: round each factor to nearest half-integer and check
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        valid, U_r, V_r, W_r = check_solution(U_snap, V_snap, W_snap)
        if valid:
            return True, U_r, V_r, W_r
        
        return False, None, None, None

    # Search ranks
    # 3x3x3 min rank is 23. 3x3x4 is likely 24, 25, 26...
    # We try 23 to 35.
    for R in range(23, 36):
        if time.time() - start_time > timeout:
            break
            
        # Try multiple seeds
        # The number of seeds depends on time remaining
        seeds_tried = 0
        max_seeds = 30 # Adjust based on time
        
        for seed in range(max_seeds):
            if time.time() - start_time > timeout:
                break
                
            np.random.seed(42 + R * 100 + seed)
            U = np.random.randn(N, R)
            V = np.random.randn(M, R)
            W = np.random.randn(P, R)
            
            # ALS Loop
            converged = False
            for it in range(80):
                if time.time() - start_time > timeout:
                    break
                    
                # Update W
                E_W = np.einsum('ar,br->ab r', U, V).reshape(N * M, R)
                try:
                    W_T, _, _, _ = np.linalg.lstsq(E_W, T_c.T, rcond=None)
                    W = W_T.T
                except:
                    break
                
                # Update U
                T_a_vec = T.reshape(N, M * P)
                E_U = np.einsum('br,cr->bc r', V, W).reshape(M * P, R)
                try:
                    U_T, _, _, _ = np.linalg.lstsq(E_U.T, T_a_vec.T, rcond=None)
                    U = U_T.T
                except:
                    break

                # Update V
                T_b_vec = T.transpose(1, 0, 2).reshape(M, N * P)
                E_V = np.einsum('ar,cr->ac r', U, W).reshape(N * P, R)
                try:
                    V_T, _, _, _ = np.linalg.lstsq(E_V.T, T_b_vec.T, rcond=None)
                    V = V_T.T
                except:
                    break

                # Check if current solution is close to half-integers (even if not fully converged)
                if it % 5 == 0:  # Check every 5 iterations
                    valid, U_r, V_r, W_r = check_solution(U, V, W)
                    if valid:
                        best_rank = R
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        converged = True
                        break
                
                # Check convergence
                T_calc = np.einsum('ar,br,cr->abc', U, V, W)
                error = np.linalg.norm(T_calc - T)
                
                if error < 1e-7:
                    # Try to polish and snap
                    valid, U_r, V_r, W_r = check_solution(U, V, W)
                    if valid:
                        best_rank = R
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        converged = True
                        break
                    
                    # If not valid, try polishing
                    valid, U_r, V_r, W_r = polish(U, V, W)
                    if valid:
                        best_rank = R
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        converged = True
                        break
                        
            if converged:
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
