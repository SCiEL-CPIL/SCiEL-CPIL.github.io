"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_fallback = [[0]*rank_school for _ in range(n*m)]
    V_fallback = [[0]*rank_school for _ in range(m*p)]
    W_fallback = [[0]*rank_school for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i*m+j][col] = 1
                V_fallback[j*p+k][col] = 1
                W_fallback[k*n+i][col] = 1
                col += 1
    
    # Try ALS for lower rank
    for target_rank in [98, 96, 94, 92, 90]:
        if time.time() > deadline:
            break
        result = als_decomposition(n, m, p, target_rank, deadline)
        if result is not None:
            return result
    
    return {"U": U_fallback, "V": V_fallback, "W": W_fallback}

def als_decomposition(n, m, p, target_rank, deadline):
    np.random.seed(42)
    
    for seed in range(50):
        if time.time() > deadline:
            break
        np.random.seed(seed)
        
        U = np.random.randn(n*m, target_rank) * 0.1
        V = np.random.randn(m*p, target_rank) * 0.1
        W = np.random.randn(p*n, target_rank) * 0.1
        
        for it in range(200):
            if time.time() > deadline:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    idx = i*m + j
                    # For each c, U[idx, c] should satisfy:
                    # sum_c U[idx,c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
                    # Let A_k,c = V[j*p+k, c] * W[k*n+i, c]
                    # Then sum_c U[idx,c] * A_k,c = 1 for all k
                    # This is a linear system: A @ U[idx,:] = ones
                    A = np.zeros((p, target_rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve A @ x = ones(p)
                    try:
                        U[idx, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    idx = j*p + k
                    # sum_c U[i*m+j, c] * V[idx, c] * W[k*n+i, c] = 1 for all i
                    # Let B_i,c = U[i*m+j, c] * W[k*n+i, c]
                    # Then sum_c V[idx,c] * B_i,c = 1 for all i
                    B = np.zeros((n, target_rank))
                    for i in range(n):
                        B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[idx, :] = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    idx = k*n + i
                    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[idx, c] = 1 for all j
                    # Let C_j,c = U[i*m+j, c] * V[j*p+k, c]
                    # Then sum_c W[idx,c] * C_j,c = 1 for all j
                    C = np.zeros((m, target_rank))
                    for j in range(m):
                        C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[idx, :] = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check error
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a, b, c = i*m+j, j*p+k, k*n+i
                        val = np.dot(U[a,:], V[b,:] * W[c,:])
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-8:
                # Try rounding to half-integers
                Ur = np.round(U*2)/2
                Vr = np.round(V*2)/2
                Wr = np.round(W*2)/2
                max_err2 = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a, b, c = i*m+j, j*p+k, k*n+i
                            val = np.dot(Ur[a,:], Vr[b,:] * Wr[c,:])
                            max_err2 = max(max_err2, abs(val - 1.0))
                if max_err2 < 1e-9:
                    return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
        
        # Perturb for next attempt
        U += 0.5 * np.random.randn(n*m, target_rank)
        V += 0.5 * np.random.randn(m*p, target_rank)
        W += 0.5 * np.random.randn(p*n, target_rank)
    
    return None
# EVOLVE-BLOCK-END
