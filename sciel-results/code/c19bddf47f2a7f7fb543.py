import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    N = n * m
    M = m * p
    P = p * n
    
    # Target tensor
    T = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    # Matricizations
    T_c = T.reshape(N * M, P)
    T_a_vec = T.reshape(N, M * P)
    T_b_vec = T.transpose(1, 0, 2).reshape(M, N * P)

    def check_solution(U, V, W, tol=1e-9):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        T_calc = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(T_calc, T, atol=tol):
            return True, U_r, V_r, W_r
        return False, None, None, None

    def als_iterate(U, V, W, R, max_iter=100, tol=1e-10):
        prev_err = np.inf
        for it in range(max_iter):
            # Update W
            E_W = np.einsum('ar,br->ab r', U, V).reshape(N * M, R)
            W_T, _, _, _ = np.linalg.lstsq(E_W, T_c.T, rcond=None)
            W = W_T.T
            
            # Update U
            E_U = np.einsum('br,cr->bc r', V, W).reshape(M * P, R)
            U_T, _, _, _ = np.linalg.lstsq(E_U.T, T_a_vec.T, rcond=None)
            U = U_T.T
            
            # Update V
            E_V = np.einsum('ar,cr->ac r', U, W).reshape(N * P, R)
            V_T, _, _, _ = np.linalg.lstsq(E_V.T, T_b_vec.T, rcond=None)
            V = V_T.T
            
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.linalg.norm(T_calc - T)
            
            if abs(prev_err - error) < tol:
                return U, V, W, error, True
            prev_err = error
        return U, V, W, error, False

    start_time = time.time()
    timeout = 140.0
    
    # Try ranks 25 to 30
    for R in range(25, 31):
        if time.time() - start_time > timeout:
            break
            
        found = False
        # More attempts for lower ranks
        max_attempts = 50 if R == 25 else 20
        
        for attempt in range(max_attempts):
            if time.time() - start_time > timeout:
                break
                
            np.random.seed(1234 + R * 1000 + attempt)
            U = np.random.randn(N, R)
            V = np.random.randn(M, R)
            W = np.random.randn(P, R)
            
            U, V, W, err, converged = als_iterate(U, V, W, R, max_iter=80)
            
            if err < 1e-4:
                valid, U_r, V_r, W_r = check_solution(U, V, W)
                if valid:
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                
                # Try to snap even if not perfectly converged, if error is small
                # Sometimes ALS gets close but not exact
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                valid, _, _, _ = check_solution(U_snap, V_snap, W_snap)
                if valid:
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}

    # Fallback to schoolbook if no better rank found
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
