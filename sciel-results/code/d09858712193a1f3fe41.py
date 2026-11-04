import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    N = n * m
    M = m * p
    P = p * n
    
    # Target tensor T[a, b, c]
    T = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    start_time = time.time()
    timeout = 110.0 # Safety margin for 120s limit
    
    best_rank = N * M * P # Fallback (Schoolbook)
    best_U = None
    best_V = None
    best_W = None
    
    # Pre-compute matricizations for fast ALS
    # T_c: shape (N*M, P) -> T[a,b,c] indexed by (a,b) rows, c cols
    T_c = T.reshape(N * M, P)
    # T_a: shape (N, M*P) -> T[a,b,c] indexed by a rows, (b,c) cols
    T_a = T.reshape(N, M * P)
    # T_b: shape (M, N*P) -> T[a,b,c] indexed by b rows, (a,c) cols
    T_b = T.transpose(1, 0, 2).reshape(M, N * P)

    def check_solution(U, V, W, tol=1e-9):
        # Verify if U, V, W contain only half-integers
        # And if they reconstruct T exactly (within tol)
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

    def als_decompose(R, seed):
        np.random.seed(seed)
        U = np.random.randn(N, R)
        V = np.random.randn(M, R)
        W = np.random.randn(P, R)
        
        # Normalize initial factors
        for r in range(R):
            norm_u = np.linalg.norm(U[:, r])
            norm_v = np.linalg.norm(V[:, r])
            norm_w = np.linalg.norm(W[:, r])
            if norm_u > 0 and norm_v > 0 and norm_w > 0:
                U[:, r] /= norm_u
                V[:, r] /= norm_v
                W[:, r] /= norm_w

        for it in range(50):
            if time.time() - start_time > timeout:
                return None, None, None, False

            # Update W
            # For each c, solve A_c W_c = y_c
            # A_c has shape (N*M, R), y_c has shape (N*M,)
            # A_c[(a*m+b), r] = U[a, r] * V[b, r]
            # We can construct the big matrix A_all of shape (P, N*M, R)
            # Or loop over c. P is small (12).
            for c in range(P):
                A_c = np.einsum('ar,br->ab r', U, V).reshape(N * M, R)
                y_c = T_c[:, c]
                W_c, _, _, _ = np.linalg.lstsq(A_c, y_c, rcond=None)
                W[c, :] = W_c
            
            # Update U
            # For each a, solve A_a U_a = y_a
            # A_a has shape (M*P, R), y_a has shape (M*P,)
            # A_a[(b*p+c), r] = V[b, r] * W[c, r]
            for a in range(N):
                A_a = np.einsum('br,cr->bc r', V, W).reshape(M * P, R)
                y_a = T_a[a, :]
                U_a, _, _, _ = np.linalg.lstsq(A_a, y_a, rcond=None)
                U[a, :] = U_a

            # Update V
            # For each b, solve A_b V_b = y_b
            # A_b has shape (N*P, R), y_b has shape (N*P,)
            # A_b[(a*n+c), r] = U[a, r] * W[c, r]
            for b in range(M):
                A_b = np.einsum('ar,cr->ac r', U, W).reshape(N * P, R)
                y_b = T_b[b, :]
                V_b, _, _, _ = np.linalg.lstsq(A_b, y_b, rcond=None)
                V[b, :] = V_b

            # Check convergence
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.linalg.norm(T_calc - T)
            
            if error < 1e-8:
                valid, U_r, V_r, W_r = check_solution(U, V, W)
                if valid:
                    return U_r, V_r, W_r, True

        return None, None, None, False

    # Search ranks
    # 3x3x3 min rank is 23. 3x3x4 is likely 24-30.
    # We try 23 to 35.
    for R in range(23, 36):
        if time.time() - start_time > timeout:
            break
            
        # Try multiple seeds
        max_seeds = 20
        
        for seed in range(max_seeds):
            if time.time() - start_time > timeout:
                break
                
            U_r, V_r, W_r, found = als_decompose(R, 42 + R * 100 + seed)
            
            if found:
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
