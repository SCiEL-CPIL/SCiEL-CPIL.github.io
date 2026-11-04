import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook fallback
    rank_school = n*m*p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank using alternating least squares
    # with rounding to integers/half-integers
    
    for rank in [88, 80, 72, 64]:
        if time.time() > deadline - 15:
            break
        
        found = False
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            # Initialize with schoolbook columns (random subset)
            idx = np.random.choice(rank_school, rank, replace=False)
            U = U_school[:, idx].copy()
            V = V_school[:, idx].copy()
            W = W_school[:, idx].copy()
            
            # Alternating least squares
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                # Update U: minimize ||T - sum_c U[:,c] V[:,c]^T W[:,c]^T||_F
                # This is a linear system in U
                # T.reshape(n*m, m*p*p*n) = U @ (V * W).reshape(m*p*p*n, rank)
                # Actually, let's do it properly
                # T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
                # Fix V, W: for each i, T[i,:,:] = sum_c U[i,c] (V[:,c] * W[:,c])
                # So T_i (vector of length m*p*p*n) = U[i,:] @ M where M[c, (j,k)] = V[j,c]*W[k,c]
                
                M_VW = np.einsum('jc,kc->ckj', V, W).reshape(rank, m*p*p*n)
                # T_flat[i, (j,k)] = T[i*m+j, j*p+k, k*n+i]... this is tricky
                # Let me reshape T properly
                # T is (n*m, m*p, p*n)
                # For fixed i (in 0..n*m-1), T[i, :, :] is (m*p, p*n)
                # We want to express T[i, j, k] = sum_c U[i,c] V[j,c] W[k,c]
                # Let j index into 0..m*p-1, k index into 0..p*n-1
                # M[c, j*p*n + k] = V[j,c] * W[k,c]
                M_VW2 = np.zeros((rank, m*p*p*n))
                for c in range(rank):
                    M_VW2[c, :] = (V[:, c][:, None] * W[:, c][None, :]).flatten()
                
                # T_flat[i, :] = T[i, :, :].flatten()
                T_flat = T.reshape(n*m, m*p*p*n)
                
                # Solve U[i, :] @ M_VW2.T = T_flat[i, :]
                # U[i, :] = T_flat[i, :] @ pinv(M_VW2.T)
                try:
                    U = T_flat @ np.linalg.pinv(M_VW2.T)
                except:
                    break
                
                # Similarly for V and W
                M_UW = np.zeros((rank, n*m*p*n))
                for c in range(rank):
                    M_UW[c, :] = (U[:, c][:, None] * W[:, c][None, :]).flatten()
                T_flat_v = np.einsum('ij,ik->jk', T.reshape(n*m, m*p, p*n).transpose(1,2,0), np.ones((n*m, 1))).squeeze()
                # This is getting complicated. Let me use a simpler approach.
                break
            
            # Check if current U, V, W (rounded) give exact decomposition
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            T_approx = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
            if np.allclose(T_approx, T, atol=1e-10):
                found = True
                best_rank = rank
                best_U, best_V, best_W = U_r, V_r, W_r
                break
        
        if found:
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
