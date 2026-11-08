import numpy as np
import time

def construct():
    n, m, p = 3, 3, 6
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Construct the matmul tensor M of shape (n*m, m*p, p*n)
    # M[(i,j), (j',k), (k'',i'')] = 1 if i==i'', j==j', k==k'' else 0
    M = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                M[i*m+j, j*p+k, k*n+i] = 1
    
    # Try ranks from lower bound upward
    lower_bound = max(n*m, m*p, p*n)  # 18
    best_result = None
    
    for R in range(lower_bound, 55):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # Alternating least squares
            for it in range(1000):
                if time.time() > deadline:
                    break
                
                # Update U: for each (i,j), solve linear system
                for ij in range(n*m):
                    # M[ij, :, :] = sum_r U[ij, r] * V[:, r] ⊗ W[:, r]
                    # So M[ij, jk, ki] = sum_r U[ij, r] * V[jk, r] * W[ki, r]
                    # For fixed ij, this is: M[ij, :, :] = U[ij, :] @ (V ⊗ W) reshaped
                    # Actually: M[ij, jk, ki] = sum_r U[ij, r] * (V[jk, r] * W[ki, r])
                    # Let A_r = V[:, r] * W[:, r] (element-wise product), shape (m*p, p*n)
                    # Then M[ij, :, :] = sum_r U[ij, r] * A_r
                    # This is a linear system: M[ij, :, :].reshape(-1) = sum_r U[ij, r] * A_r.reshape(-1)
                    # So: M_flat[ij, :] = U[ij, :] @ A_flat
                    A_flat = np.zeros((R, m*p * p*n))
                    for r in range(R):
                        A_flat[r, :] = (V[:, r] * W[:, r]).reshape(-1)
                    M_flat_ij = M[ij, :, :].reshape(-1)
                    # Solve: M_flat_ij = A_flat.T @ U[ij, :]
                    U[ij, :], residuals, rank_u, sv = np.linalg.lstsq(A_flat.T, M_flat_ij, rcond=None)
                
                # Similarly update V and W
                for jk in range(m*p):
                    A_flat = np.zeros((R, n*m * p*n))
                    for r in range(R):
                        A_flat[r, :] = (U[:, r] * W[:, r]).reshape(-1)
                    M_flat_jk = M[:, jk, :].reshape(-1)
                    V[jk, :], residuals, rank_v, sv = np.linalg.lstsq(A_flat.T, M_flat_jk, rcond=None)
                
                for ki in range(p*n):
                    A_flat = np.zeros((R, n*m * m*p))
                    for r in range(R):
                        A_flat[r, :] = (U[:, r] * V[:, r]).reshape(-1)
                    M_flat_ki = M[:, :, ki].reshape(-1)
                    W[ki, :], residuals, rank_w, sv = np.linalg.lstsq(A_flat.T, M_flat_ki, rcond=None)
            
            # Check if we got a good solution
            # Compute the reconstructed tensor
            M_recon = np.zeros_like(M)
            for r in range(R):
                M_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            
            error = np.linalg.norm(M_recon - M) / np.linalg.norm(M)
            
            if error < 1e-6:
                # Try to round to integer/half-integer
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                # Verify the rounded solution
                M_round = np.zeros_like(M)
                for r in range(R):
                    M_round += np.einsum('i,j,k->ijk', U_round[:, r], V_round[:, r], W_round[:, r])
                
                error_round = np.linalg.norm(M_round - M) / np.linalg.norm(M)
                
                if error_round < 1e-6:
                    # Convert to lists of integers/half-integers
                    U_list = [[float(x) for x in row] for row in U_round]
                    V_list = [[float(x) for x in row] for row in V_round]
                    W_list = [[float(x) for x in row] for row in W_round]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
