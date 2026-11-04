import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
                
    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Try to find rank 41
    target_rank = 41
    
    # Random search
    for attempt in range(1000):
        if time.time() > deadline:
            break
            
        # Random U, V with small integers
        U = np.random.randint(-2, 3, size=(n * m, target_rank))
        V = np.random.randint(-2, 3, size=(m * p, target_rank))
        
        # Solve for W
        # For each k, i:
        # sum_r W[k,i,r] * (sum_j U[i,j,r] V[j,k,r]) = 1?
        # No, the equation is:
        # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
        # Let's fix k, i.
        # We need sum_r W[k,i,r] * (U[i, :, r] dot V[:, k, r]) = 1?
        # No, U[i,j,r] and V[j,k,r] depend on j.
        # So for fixed k, i, we have m equations (j=0..m-1) and target_rank unknowns (W[k,i,0..r-1]).
        # Let A[k,i,j,r] = U[i,j,r] * V[j,k,r].
        # Then sum_r A[k,i,j,r] W[k,i,r] = 1.
        
        W = np.zeros((p * n, target_rank))
        valid = True
        for k in range(p):
            for i in range(n):
                # Construct system for W[k*n+i, :]
                # Equations: for j in 0..m-1:
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                
                # Matrix M of size m x target_rank
                M = np.zeros((m, target_rank))
                for j in range(m):
                    for r in range(target_rank):
                        M[j, r] = U[i * m + j, r] * V[j * p + k, r]
                
                # Solve M @ W_vec = ones(m)
                # Use least squares
                try:
                    W_vec, residuals, rank, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W[k * n + i, :] = W_vec
                except:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
            
        # Check if W is close to half-integers
        # Multiply by 2 and check if close to integers
        W_check = W * 2
        W_rounded = np.round(W_check)
        if np.allclose(W_check, W_rounded, atol=1e-9):
            # Verify exact decomposition
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = (W_rounded / 2).tolist()
            
            # Verify
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(target_rank):
                            s += U_list[i * m + j][r] * V_list[j * p + k][r] * W_list[k * n + i][r]
                        if abs(s - 1.0) > 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                best_rank = target_rank
                best_U = U_list
                best_V = V_list
                best_W = W_list
                break # Found rank 41, stop searching for lower? Or try 40?
                
    # If found rank 41, we could try 40, but 41 is already an improvement.
    # Given time, we stop at first success.
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
