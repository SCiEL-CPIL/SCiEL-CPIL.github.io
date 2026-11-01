import numpy as np
import time

def construct():
    n, m, p = 2, 3, 3
    U_size = n * m  # 6
    V_size = m * p  # 9
    W_size = p * n  # 6
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    T = np.zeros((U_size, V_size, W_size))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    start_time = time.time()
    deadline = start_time + 110.0

    def verify(U, V, W):
        rank = U.shape[1]
        T_calc = np.zeros((U_size, V_size, W_size))
        for c in range(rank):
            T_calc += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        if not np.allclose(T_calc, T, atol=1e-9):
            return False
        return True

    def snap_and_verify(U, V, W):
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        if verify(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap
        return None

    def als_decompose(rank, max_iter=100, tol=1e-10):
        # Initialize randomly
        U = np.random.randn(U_size, rank)
        V = np.random.randn(V_size, rank)
        W = np.random.randn(W_size, rank)
        
        prev_cost = np.inf
        for it in range(max_iter):
            # Update U
            # For each i, solve U[i] @ A = T[i]
            # A is (V_size*W_size) x rank, where A[:, c] = V[:, c] outer W[:, c]
            A = np.einsum('jc,kc->jkc', V, W).reshape(V_size * W_size, rank)
            # T is (U_size, V_size, W_size) -> reshape to (U_size, V_size*W_size)
            T_flat = T.reshape(U_size, V_size * W_size)
            
            # Solve U_new = T_flat @ A.T @ inv(A @ A.T) ? 
            # Actually, we want U_new such that U_new @ A.T = T_flat
            # So U_new = T_flat @ pinv(A.T)
            # Or use lstsq
            U_new = np.linalg.lstsq(A.T, T_flat.T, rcond=None)[0].T
            
            # Update V
            # For each j, solve V[j] @ B = T[:, j, :]
            # B is (U_size*W_size) x rank, where B[:, c] = U[:, c] outer W[:, c]
            B = np.einsum('ic,kc->ikc', U_new, W).reshape(U_size * W_size, rank)
            # T[:, j, :] is (U_size, W_size) -> reshape to (U_size*W_size, 1) for each j?
            # Better: T is (U_size, V_size, W_size). We need to solve for each j.
            # V[j] @ B = T[:, j, :].reshape(-1, 1)
            V_new = np.zeros((V_size, rank))
            for j in range(V_size):
                t_j = T[:, j, :].reshape(-1, 1)
                V_new[j] = np.linalg.lstsq(B.T, t_j, rcond=None)[0].flatten()
                
            # Update W
            # For each k, solve W[k] @ C = T[:, :, k]
            # C is (U_size*V_size) x rank, where C[:, c] = U[:, c] outer V[:, c]
            C = np.einsum('ic,jc->ijc', U_new, V_new).reshape(U_size * V_size, rank)
            W_new = np.zeros((W_size, rank))
            for k in range(W_size):
                t_k = T[:, :, k].reshape(-1, 1)
                W_new[k] = np.linalg.lstsq(C.T, t_k, rcond=None)[0].flatten()
                
            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            cost = 0
            for c in range(rank):
                cost += np.sum((np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c]) - T)**2)
            
            if abs(prev_cost - cost) < tol:
                break
            prev_cost = cost
                
        return U, V, W

    # Try ranks from 14 upwards
    for rank in [14, 15, 16, 17]:
        if time.time() > deadline:
            break
            
        max_restarts = 50 if rank == 14 else 20
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Random seed for reproducibility in search? No, just random.
            np.random.seed(restart * 1000 + rank)
            
            U, V, W = als_decompose(rank)
            
            result = snap_and_verify(U, V, W)
            if result:
                U_final, V_final, W_final = result
                return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

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
