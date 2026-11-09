import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p))
    
    schoolbook_rank = n * m * p
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    start_time = time.time()
    deadline = 120.0

    def is_valid(U, V, W, T):
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        
        try:
            T_approx = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        except Exception:
            return False, None, None, None
            
        if np.allclose(T_approx, T, atol=1e-9):
            return True, U, V, W
        return False, None, None, None

    def round_to_half(x):
        return np.round(x * 2) / 2.0

    def solve_als(r, max_iter=50, tol=1e-10):
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Normalize initial
        for c in range(r):
            U[:, c] /= np.linalg.norm(U[:, c]) + 1e-12
            V[:, c] /= np.linalg.norm(V[:, c]) + 1e-12
            W[:, c] /= np.linalg.norm(W[:, c]) + 1e-12

        for it in range(max_iter):
            if time.time() > deadline - 5:
                return None, None, None
                
            # Update U
            # A[i, j, k, c] = V[j, k, c] * W[k, i, c]
            # V: (m, p, r)
            # W: (p, n, r) -> W_trans: (n, p, r) where W_trans[i, k, c] = W[k, i, c]
            V_reshaped = V.reshape(m, p, r)
            W_reshaped = W.reshape(p, n, r)
            W_trans = W_reshaped.transpose(1, 0, 2) # (n, p, r)
            
            # A: (n, m, p, r)
            # V_reshaped: (m, p, r) -> (1, m, p, r)
            # W_trans: (n, p, r) -> (n, 1, p, r)
            A = V_reshaped[None, :, :, :] * W_trans[:, None, :, :]
            
            # Solve for U
            # M_U: (n, m, r, r)
            M_U = np.einsum('ijkc,ijkc->ijcc', A, A)
            B_U = np.einsum('ijkc,ijk->ijc', A, T)
            
            # Solve batch of linear systems
            M_U_flat = M_U.reshape(-1, r, r)
            B_U_flat = B_U.reshape(-1, r, 1)
            try:
                U_flat = np.linalg.solve(M_U_flat, B_U_flat)
                U = U_flat.reshape(n * m, r)
            except np.linalg.LinAlgError:
                # Fallback to lstsq if singular
                U_flat, _, _, _ = np.linalg.lstsq(M_U_flat, B_U_flat, rcond=None)
                U = U_flat.reshape(n * m, r)

            # Update V
            # B[j, k, i, c] = U[i, j, c] * W[k, i, c]
            U_reshaped = U.reshape(n, m, r)
            # W_reshaped: (p, n, r)
            
            # U: (n, m, r) -> (m, n, r)
            U_perm = U_reshaped.transpose(1, 0, 2)
            # W: (p, n, r)
            
            # B: (m, p, n, r)
            # U_perm: (m, n, r) -> (m, 1, n, r)
            # W_reshaped: (p, n, r) -> (1, p, n, r)
            B_mat = U_perm[:, None, :, :] * W_reshaped[None, :, :, :]
            
            # Solve for V
            # T_trans: (m, p, n)
            T_trans = T.transpose(1, 2, 0)
            
            M_V = np.einsum('jkic,jkic->jkcc', B_mat, B_mat)
            B_V = np.einsum('jkic,jki->jkc', B_mat, T_trans)
            
            M_V_flat = M_V.reshape(-1, r, r)
            B_V_flat = B_V.reshape(-1, r, 1)
            try:
                V_flat = np.linalg.solve(M_V_flat, B_V_flat)
                V = V_flat.reshape(m * p, r)
            except np.linalg.LinAlgError:
                V_flat, _, _, _ = np.linalg.lstsq(M_V_flat, B_V_flat, rcond=None)
                V = V_flat.reshape(m * p, r)

            # Update W
            # C[k, i, j, c] = U[i, j, c] * V[j, k, c]
            U_reshaped = U.reshape(n, m, r)
            V_reshaped = V.reshape(m, p, r)
            
            # V_perm: (p, m, r)
            V_perm = V_reshaped.transpose(1, 0, 2)
            
            # C: (p, n, m, r)
            # U: (n, m, r) -> (1, n, m, r)
            # V_perm: (p, m, r) -> (p, 1, m, r)
            C_mat = U_reshaped[None, :, :, :] * V_perm[:, None, :, :]
            
            # Solve for W
            # T_trans2: (p, n, m)
            T_trans2 = T.transpose(2, 0, 1)
            
            M_W = np.einsum('kinj,kinj->kinc', C_mat, C_mat)
            B_W = np.einsum('kinj,kij->kic', C_mat, T_trans2)
            
            M_W_flat = M_W.reshape(-1, r, r)
            B_W_flat = B_W.reshape(-1, r, 1)
            try:
                W_flat = np.linalg.solve(M_W_flat, B_W_flat)
                W = W_flat.reshape(p * n, r)
            except np.linalg.LinAlgError:
                W_flat, _, _, _ = np.linalg.lstsq(M_W_flat, B_W_flat, rcond=None)
                W = W_flat.reshape(p * n, r)

            # Check convergence
            U_reshaped = U.reshape(n, m, r)
            V_reshaped = V.reshape(m, p, r)
            W_reshaped = W.reshape(p, n, r)
            T_approx = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
            err = np.linalg.norm(T_approx - T)
            if err < tol:
                return U, V, W
                
        return U, V, W

    # Try ranks from 104 down to 80
    for r in range(schoolbook_rank - 1, 79, -1):
        if time.time() > deadline - 10:
            break
            
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
                
            U, V, W = solve_als(r, max_iter=100)
            if U is None:
                continue
                
            # Round to half-integers
            U_r = round_to_half(U)
            V_r = round_to_half(V)
            W_r = round_to_half(W)
            
            # Verify rounded version
            valid, U_v, V_v, W_v = is_valid(U_r, V_r, W_r, T)
            if valid:
                best_rank = r
                best_U = U_v
                best_V = V_v
                best_W = W_v
                break
        if best_rank < schoolbook_rank:
            break

    if best_rank < schoolbook_rank:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Return schoolbook
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
