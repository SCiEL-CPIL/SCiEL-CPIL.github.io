"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    """
    Construct a rank-minimal exact decomposition of the (2,4,4) matrix multiplication tensor.
    
    The tensor T[i,j,k] = 1 for all i,j,k (in the sense that C[i,k] = sum_j A[i,j]*B[j,k]).
    We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    We use an alternating least squares (ALS) approach to search for a lower-rank decomposition,
    then attempt to round to half-integers. If that fails, we return the schoolbook (rank 32).
    """
    n, m, p = DIMENSIONS
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Target tensor: T[i,j,k] = 1 for all i,j,k
    T = np.ones((n, m, p))
    
    # Try ranks from 24 up to 31, attempting to find a valid half-integer solution
    for R in range(24, 32):
        start_time = time.time()
        rng = np.random.RandomState(42)
        
        for trial in range(50):
            if time.time() - start_time > 90:
                break
            
            # Initialize with random values
            U = rng.randn(n * m, R) * 0.5
            V = rng.randn(m * p, R) * 0.5
            W = rng.randn(p * n, R) * 0.5
            
            for it in range(300):
                if time.time() - start_time > 95:
                    break
                
                # Solve for U given V, W
                # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
                # For fixed V, W, this is linear in U.
                # Reshape: for each (i,j), we have a linear system in U[i*m+j, :]
                # sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = T[i,j,k] for all k
                # So for fixed (i,j), the coefficient vector for U[i*m+j, :] is:
                #   coeff[c] = sum_k V[j*p+k, c] * W[k*n+i, c] * (something)
                # Actually, let's set up the full linear system.
                
                # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # Group by (i,j): for fixed i,j, we have p equations (one per k) in R unknowns (U[i*m+j, 0..R-1])
                # But this is underdetermined if R > p. Instead, let's solve the full system.
                
                # Build the matrix M such that M @ U_flat = T_flat
                # U_flat is of length n*m*R
                # T_flat is of length n*m*p
                # M has shape (n*m*p, n*m*R)
                
                M = np.zeros((n * m * p, n * m * R))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row = i * m * p + j * p + k
                            for c in range(R):
                                col = (i * m + j) * R + c
                                M[row, col] = V[j * p + k, c] * W[k * n + i, c]
                
                T_flat = T.reshape(-1)
                U_new, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
                U = U_new.reshape(n * m, R)
                
                # Solve for V given U, W
                N = np.zeros((m * p * n, m * p * R))
                for j in range(m):
                    for k in range(p):
                        for i in range(n):
                            row = j * p * n + k * n + i
                            for c in range(R):
                                col = (j * p + k) * R + c
                                N[row, col] = U[i * m + j, c] * W[k * n + i, c]
                
                T2 = T.transpose(1, 2, 0).reshape(-1)  # T[j,k,i]
                V_new, _, _, _ = np.linalg.lstsq(N, T2, rcond=None)
                V = V_new.reshape(m * p, R)
                
                # Solve for W given U, V
                P = np.zeros((p * n * m, p * n * R))
                for k in range(p):
                    for i in range(n):
                        for j in range(m):
                            row = k * n * m + i * m + j
                            for c in range(R):
                                col = (k * n + i) * R + c
                                P[row, col] = U[i * m + j, c] * V[j * p + k, c]
                
                T3 = T.transpose(2, 0, 1).reshape(-1)  # T[k,i,j]
                W_new, _, _, _ = np.linalg.lstsq(P, T3, rcond=None)
                W = W_new.reshape(p * n, R)
            
            # Check if solution is valid
            T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
            if np.max(np.abs(T_recon - T)) < 1e-6:
                # Try rounding to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                T_r = np.einsum('ac,bc,dc->abd', U_r, V_r, W_r)
                if np.max(np.abs(T_r - T)) < 1e-6:
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    return schoolbook()
# EVOLVE-BLOCK-END
