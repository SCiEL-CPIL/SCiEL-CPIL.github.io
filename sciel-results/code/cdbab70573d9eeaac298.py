import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target tensor T (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Flatten T for matrix multiplication
    # T_flat: (nm, mp*pn)
    T_flat = T.reshape(nm, mp * pn)
    
    start_time = time.time()
    deadline = start_time + 100.0
    
    for rank in range(12, 25):
        if time.time() > deadline - 5.0:
            break
            
        num_restarts = 50
        for restart in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
                
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            max_iter = 100
            prev_err = np.inf
            
            for it in range(max_iter):
                if time.time() > deadline - 5.0:
                    break
                    
                # Update U
                # A_U = T_flat @ (V kron W)
                VW = np.kron(V, W) # (mp*pn, rank)
                A_U = T_flat @ VW # (nm, rank)
                
                # S_U[c, c'] = (V_c^T V_c') (W_c^T W_c')
                # V^T V is (rank, rank)
                # W^T W is (rank, rank)
                # We need element-wise product of V^T V and W^T W?
                # S'[c', c] = M_V[c', c] M_W[c', c]
                MV = V.T @ V
                MW = W.T @ W
                S_U = MV * MW # Element-wise product. S_U[c', c] = MV[c', c] MW[c', c]
                
                # Solve U S_U = A_U
                # U = A_U @ inv(S_U)
                try:
                    U = A_U @ np.linalg.inv(S_U)
                except np.linalg.LinAlgError:
                    break
                
                # Update V
                # A_V = (T x_1 U) x_3 W ?
                # T is (nm, mp, pn)
                # T x_1 U: (mp, pn, rank) ?
                # Let's use matricization.
                # T_1: (mp, nm*pn)
                # A_V[j, c] = sum_{i,k} T[i, j, k] U[i, c] W[k, c]
                # T_1 = T.transpose(1, 0, 2).reshape(mp, nm*pn)
                T_1 = T.transpose(1, 0, 2).reshape(mp, nm * pn)
                UW = np.kron(U, W) # (nm*pn, rank)
                A_V = T_1 @ UW
                
                MV2 = V.T @ V
                MU = U.T @ U
                MW2 = W.T @ W
                S_V = MU * MW2 # S_V[c', c] = MU[c', c] MW2[c', c]
                
                try:
                    V = A_V @ np.linalg.inv(S_V)
                except np.linalg.LinAlgError:
                    break
                    
                # Update W
                # T_2: (pn, nm*mp)
                T_2 = T.transpose(2, 0, 1).reshape(pn, nm * mp)
                UV = np.kron(U, V) # (nm*mp, rank)
                A_W = T_2 @ UV
                
                MV3 = V.T @ V
                MU3 = U.T @ U
                S_W = MU3 * MV3 # S_W[c', c] = MU3[c', c] MV3[c', c]
                
                try:
                    W = A_W @ np.linalg.inv(S_W)
                except np.linalg.LinAlgError:
                    break
                
                # Calculate Error
                T_sum = np.zeros_like(T)
                for c in range(rank):
                    T_sum += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                err = np.linalg.norm(T - T_sum)
                
                if abs(prev_err - err) < 1e-12:
                    break
                prev_err = err
                
                if err < 1e-9:
                    break
            
            # Check half-integer
            U_half = np.round(2 * U) / 2.0
            V_half = np.round(2 * V) / 2.0
            W_half = np.round(2 * W) / 2.0
            
            T_sum_half = np.zeros_like(T)
            for c in range(rank):
                T_sum_half += np.einsum('i,j,k->ijk', U_half[:, c], V_half[:, c], W_half[:, c])
            
            if np.allclose(T, T_sum_half, atol=1e-9):
                return {
                    "U": U_half.tolist(),
                    "V": V_half.tolist(),
                    "W": W_half.tolist()
                }
                
    # Fallback
    rank = nm * mp * pn
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
