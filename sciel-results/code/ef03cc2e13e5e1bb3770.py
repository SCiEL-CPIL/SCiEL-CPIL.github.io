"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    # U: (n*m) x rank = 6 x rank
    # V: (m*p) x rank = 18 x rank  
    # W: (p*n) x rank = 12 x rank
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # where T[i*m+j, j*p+k, k*n+i] = 1
    
    schoolbook_rank = n * m * p  # 36
    
    # Build the tensor T
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Compute lower bound from matricization ranks
    T0 = T.reshape(n*m, m*p * p*n)
    T1 = T.reshape(m*p, n*m * p*n)
    T2 = T.reshape(p*n, n*m * m*p)
    
    r0 = np.linalg.matrix_rank(T0, tol=1e-10)
    r1 = np.linalg.matrix_rank(T1, tol=1e-10)
    r2 = np.linalg.matrix_rank(T2, tol=1e-10)
    
    lower_bound = max(r0, r1, r2)
    
    # Try to find a decomposition using ALS (Alternating Least Squares)
    # We'll try ranks from lower_bound up to schoolbook_rank - 1
    # If we find one, great; otherwise return schoolbook
    
    best_U = None
    best_V = None
    best_W = None
    best_rank = schoolbook_rank
    
    # Try ranks in increasing order
    for target_rank in range(lower_bound, schoolbook_rank):
        if time.time() > deadline - 5:
            break
            
        # Multiple random restarts
        found = False
        for attempt in range(30):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Normalize columns to avoid scaling issues
            for r in range(target_rank):
                nu = np.linalg.norm(U[:, r])
                nv = np.linalg.norm(V[:, r])
                nw = np.linalg.norm(W[:, r])
                if nu > 1e-10 and nv > 1e-10 and nw > 1e-10:
                    U[:, r] /= nu
                    V[:, r] /= nv
                    W[:, r] /= nw
            
            # ALS iterations
            converged = False
            for it in range(100):
                if time.time() > deadline - 3:
                    break
                    
                # Solve for U: fix V, W
                # U[:,r] = argmin ||vec(T) - sum_s U_s * (V_s ⊗ W_s)||^2
                # Build Khatri-Rao product of V and W
                VkrW = np.zeros((m*p * p*n, target_rank))
                for r in range(target_rank):
                    VkrW[:, r] = np.kron(V[:, r], W[:, r])
                
                try:
                    U_new, _, _, _ = np.linalg.lstsq(VkrW, T.reshape(-1), rcond=None)
                    U_new = U_new.reshape(n*m, target_rank)
                except:
                    break
                
                # Solve for V: fix U, W
                WkrU = np.zeros((p*n * n*m, target_rank))
                for r in range(target_rank):
                    WkrU[:, r] = np.kron(W[:, r], U[:, r])
                
                try:
                    V_new, _, _, _ = np.linalg.lstsq(WkrU, T.reshape(-1), rcond=None)
                    V_new = V_new.reshape(m*p, target_rank)
                except:
                    break
                
                # Solve for W: fix U, V
                UkrV = np.zeros((n*m * m*p, target_rank))
                for r in range(target_rank):
                    UkrV[:, r] = np.kron(U[:, r], V[:, r])
                
                try:
                    W_new, _, _, _ = np.linalg.lstsq(UkrV, T.reshape(-1), rcond=None)
                    W_new = W_new.reshape(p*n, target_rank)
                except:
                    break
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence by computing error
                # Only check every few iterations to save time
                if it % 10 == 9 or it == 99:
                    error = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i * m + j
                                b = j * p + k
                                c = k * n + i
                                val = np.dot(U[a, :], np.dot(V[b, :], W[c, :]))
                                err = abs(val - 1.0)
                                if err > error:
                                    error = err
                    
                    if error < 1e-8:
                        converged = True
                        break
            
            if converged:
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    # If no better decomposition found, use schoolbook
    if best_U is None:
        rank = schoolbook_rank
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
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
