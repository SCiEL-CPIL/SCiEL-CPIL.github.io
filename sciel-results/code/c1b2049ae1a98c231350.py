"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _validate(U, V, W, n, m, p, tol=1e-10):
    """Check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    t0 = time.time()
    DEADLINE = 145.0
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # First, try the block Strassen approach (rank 34) as a baseline
    def strassen_terms():
        terms = []
        U1 = [[1,0],[0,1]]; V1 = [[1,0],[0,1]]; W1 = [[1,1],[1,1]]
        terms.append((U1, V1, W1))
        U2 = [[0,0],[1,1]]; V2 = [[1,0],[0,0]]; W2 = [[0,0],[1,0]]
        terms.append((U2, V2, W2))
        U3 = [[1,0],[0,0]]; V3 = [[0,0],[1,-1]]; W3 = [[0,1],[0,0]]
        terms.append((U3, V3, W3))
        U4 = [[0,0],[0,1]]; V4 = [[-1,1],[0,0]]; W4 = [[1,-1],[0,0]]
        terms.append((U4, V4, W4))
        U5 = [[1,1],[0,0]]; V5 = [[0,0],[0,1]]; W5 = [[1,0],[0,0]]
        terms.append((U5, V5, W5))
        U6 = [[-1,0],[1,0]]; V6 = [[1,0],[1,0]]; W6 = [[0,0],[0,1]]
        terms.append((U6, V6, W6))
        U7 = [[0,1],[0,-1]]; V7 = [[0,1],[0,1]]; W7 = [[1,0],[1,0]]
        terms.append((U7, V7, W7))
        return terms
    
    st = strassen_terms()
    
    # Build rank-34 decomposition
    total_rank = 34
    U34 = [[0]*total_rank for _ in range(n*m)]
    V34 = [[0]*total_rank for _ in range(m*p)]
    W34 = [[0]*total_rank for _ in range(p*n)]
    col = 0
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        for (Ur, Vr, Wr) in st:
            for ii in range(2):
                for jj in range(2):
                    if Ur[ii][jj] != 0:
                        U34[(i_off+ii)*m + (j_off+jj)][col] += Ur[ii][jj]
            for jj in range(2):
                for kk in range(2):
                    if Vr[jj][kk] != 0:
                        V34[(j_off+jj)*p + (k_off+kk)][col] += Vr[jj][kk]
            for kk in range(2):
                for ii in range(2):
                    if Wr[kk][ii] != 0:
                        W34[(k_off+kk)*n + (i_off+ii)][col] += Wr[kk][ii]
            col += 1
    
    def add_school(i_idx, j_idx, k_idx):
        nonlocal col
        for i in i_idx:
            for j in j_idx:
                for k in k_idx:
                    U34[i*m + j][col] = 1
                    V34[j*p + k][col] = 1
                    W34[k*n + i][col] = 1
                    col += 1
    
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_school([0,1], [2], [0,1])
    add_school([0,1], [2], [2,3])
    add_school([2], [0,1], [0,1])
    add_school([2], [0,1], [2,3])
    add_school([2], [2], [0,1])
    add_school([2], [2], [2,3])
    
    best_result = {"U": U34, "V": V34, "W": W34}
    best_rank = total_rank
    
    # Now try to find a lower-rank decomposition using optimization
    # The tensor equation: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    # This is a system of n*m*p = 36 equations in 3*r variables (U, V, W each have r columns)
    
    # Try ranks from 33 down to 20
    for target_rank in range(33, 19, -1):
        if time.time() - t0 > DEADLINE - 30:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() - t0 > DEADLINE - 5:
                break
            
            # Random initialization
            U_r = np.random.randn(n*m, target_rank) * 0.5
            V_r = np.random.randn(m*p, target_rank) * 0.5
            W_r = np.random.randn(p*n, target_rank) * 0.5
            
            # Use gradient descent to minimize sum of squared errors
            # Error for (i,j,k): (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
            
            lr = 0.01
            for step in range(2000):
                if time.time() - t0 > DEADLINE - 5:
                    break
                
                # Compute gradients
                U_grad = np.zeros_like(U_r)
                V_grad = np.zeros_like(V_r)
                W_grad = np.zeros_like(W_r)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Compute current value
                            s = 0.0
                            for r in range(target_rank):
                                s += U_r[i*m+j, r] * V_r[j*p+k, r] * W_r[k*n+i, r]
                            err = s - 1.0
                            
                            for r in range(target_rank):
                                U_grad[i*m+j, r] += 2 * err * V_r[j*p+k, r] * W_r[k*n+i, r]
                                V_grad[j*p+k, r] += 2 * err * U_r[i*m+j, r] * W_r[k*n+i, r]
                                W_grad[k*n+i, r] += 2 * err * U_r[i*m+j, r] * V_r[j*p+k, r]
                
                # Update
                U_r -= lr * U_grad
                V_r -= lr * V_grad
                W_r -= lr * W_grad
                
                # Check convergence
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(target_rank):
                                s += U_r[i*m+j, r] * V_r[j*p+k, r] * W_r[k*n+i, r]
                            max_err = max(max_err, abs(s - 1.0))
                
                if max_err < 1e-8:
                    # Found a valid decomposition!
                    U_list = U_r.tolist()
                    V_list = V_r.tolist()
                    W_list = W_r.tolist()
                    if _validate(U_list, V_list, W_list, n, m, p):
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = target_rank
                    break
            
            if best_rank == target_rank:
                break
        
        if best_rank < 34:
            break
    
    return best_result
# EVOLVE-BLOCK-END
