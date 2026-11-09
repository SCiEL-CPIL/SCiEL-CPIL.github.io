import numpy as np
import time
import sys

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Precompute the target tensor unfoldings
    # T1: (n*m, m*p * p*n)
    # T2: (m*p, n*m * p*n)
    # T3: (p*n, n*m * m*p)
    
    T1 = np.zeros((n*m, m*p * p*n))
    T2 = np.zeros((m*p, n*m * p*n))
    T3 = np.zeros((p*n, n*m * m*p))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                
                T1[a, b * (p*n) + c] = 1.0
                T2[b, a * (p*n) + c] = 1.0
                T3[c, a * (m*p) + b] = 1.0
                
    T1_T = T1.T
    T2_T = T2.T
    T3_T = T3.T

    def khatri_rao(A, B):
        # A: (dim1, R), B: (dim2, R)
        # Returns: (R, dim1*dim2)
        dim1, R = A.shape
        dim2, _ = B.shape
        # Vectorized Kronecker product of columns
        # A.T is (R, dim1), B.T is (R, dim2)
        # We want sum_r A[:,r] kron B[:,r]
        # This is equivalent to (A.T[:, :, None] * B.T[:, None, :]).reshape(R, -1).T
        # But let's use a more memory efficient way if possible, or just standard
        prod = A.T[:, :, np.newaxis] * B.T[:, np.newaxis, :]
        return prod.reshape(R, -1).T

    def check_exact(U, V, W):
        MWV = khatri_rao(W, V)
        T1_recon = U @ MWV.T
        if not np.allclose(T1_recon, T1, atol=1e-9):
            return False
        MUW = khatri_rao(U, W)
        T2_recon = V @ MUW.T
        if not np.allclose(T2_recon, T2, atol=1e-9):
            return False
        MVU = khatri_rao(V, U)
        T3_recon = W @ MVU.T
        if not np.allclose(T3_recon, T3, atol=1e-9):
            return False
        return True

    def snap_and_verify(U, V, W):
        # Snap to half-integers
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        if check_exact(U_s, V_s, W_s):
            return U_s, V_s, W_s
        return None

    def solve_als(U, V, W, max_iter=20):
        prev_err = np.inf
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            # Update U
            MWV = khatri_rao(W, V)
            # T1 = U @ MWV.T  => T1.T = MWV @ U.T
            # Solve for U.T: MWV @ U.T = T1.T
            U = np.linalg.lstsq(MWV, T1_T, rcond=None)[0].T
            
            # Update V
            MUW = khatri_rao(U, W)
            # T2 = V @ MUW.T  => T2.T = MUW @ V.T
            V = np.linalg.lstsq(MUW, T2_T, rcond=None)[0].T
            
            # Update W
            MVU = khatri_rao(V, U)
            # T3 = W @ MVU.T  => T3.T = MVU @ W.T
            W = np.linalg.lstsq(MVU, T3_T, rcond=None)[0].T
            
            # Calculate error
            MWV = khatri_rao(W, V)
            T1_recon = U @ MWV.T
            err = np.linalg.norm(T1_recon - T1)
            
            if err < 1e-7:
                res = snap_and_verify(U, V, W)
                if res:
                    return res, True
            
            if err > prev_err * 1.5:
                break
            prev_err = err
            
        return (U, V, W), False

    min_rank = 20
    max_rank = 59 # We only care if we find < 60
    
    found_solution = None
    
    # Try ranks from low to high
    for R in range(min_rank, max_rank + 1):
        if time.time() > deadline:
            break
            
        # Number of restarts decreases as rank increases to save time
        # Higher ranks are easier, so fewer restarts needed?
        # Actually, lower ranks are harder, so we might want more restarts there?
        # But time is tight. Let's do a fixed number of restarts for lower ranks.
        restarts = 50 if R < 40 else 20
        
        for r in range(restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            sol, success = solve_als(U, V, W)
            if success:
                found_solution = sol
                break
        if found_solution:
            break

    if found_solution:
        U, V, W = found_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to Schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
