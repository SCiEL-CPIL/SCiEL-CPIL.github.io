"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36
    
    # Schoolbook solution
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank decompositions
    for rank in range(35, 29, -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        for seed in range(12):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(seed * 1000 + rank * 100)
            
            # Initialize
            if rank >= 32:
                U = U_sb[:, :rank].copy()
                V = V_sb[:, :rank].copy()
                W = W_sb[:, :rank].copy()
                U += np.random.randn(n*m, rank) * 0.1
                V += np.random.randn(m*p, rank) * 0.1
                W += np.random.randn(p*n, rank) * 0.1
            else:
                U = np.random.randn(n*m, rank) * 0.5
                V = np.random.randn(m*p, rank) * 0.5
                W = np.random.randn(p*n, rank) * 0.5
            
            # Alternating Least Squares
            for it in range(80):
                if time.time() > deadline - 20:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A_mat = np.zeros((p, rank))
                        for k in range(p):
                            A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(A_mat.T, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        B_mat = np.zeros((n, rank))
                        for i in range(n):
                            B_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(B_mat.T, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        C_mat = np.zeros((m, rank))
                        for j in range(m):
                            C_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(C_mat.T, np.ones(m), rcond=None)[0]
                        except:
                            pass
            
            # Round to half-integers and verify
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U_r[i*m+j, r] * V_r[j*p+k, r] * W_r[k*n+i, r]
                        if abs(val - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
                if not valid:
                    break
            
            if valid:
                best_rank = rank
                best_U, best_V, best_W = U_r, V_r, W_r
                found = True
                break
        
        if found:
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
