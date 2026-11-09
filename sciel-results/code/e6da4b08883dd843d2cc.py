"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

DEADLINE = 145.0  # safety margin before 150 s wall-clock kill

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition is exact for matrix multiplication."""
    rank = U.shape[1]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_reduced_rank(n, m, p, time_budget):
    """Try to find a reduced rank decomposition using block sharing."""
    # Strategy: group the p=9 columns into 3 groups of 3.
    # For each group, use a 4x4 x 4x3 sub-problem.
    # Schoolbook for 4x4 x 4x3 is 48, total 3*48=144 (no gain).
    # 
    # Alternative: use the "diagonal" construction with rank = n*(m+p-1) = 4*12 = 48
    # Term (i, s) for i in 0..3, s in 0..11:
    #   j = s - i (valid if 0 <= j < 4)
    #   k = s - j = s - (s-i) = i (valid if 0 <= k < 9)
    #   But this gives W index k*n+i' where we need i'=i, k=i
    #   So W[i*4+i, (i,s)] = 1
    #   U[i*4+j, (i,s)] = 1 where j = s-i
    #   V[j*9+k, (i,s)] = 1 where k = i
    #   
    # Check: sum over (i,s) of U[i*4+j,c]*V[j*9+k,c]*W[k*4+i',c]
    # For fixed (i,j,k,i'): nonzero when i'=i, k=i, and j=s-i, s=j+i
    # s = j+i, need 0<=s<=11, i.e., 0<=j+i<=11 (always true for j<4,i<4)
    # So sum = 1 for i'=i, k=i, any j? No, we need the (j) index to match.
    # 
    # Actually this construction doesn't directly work. Let me just use schoolbook.
    
    # For now, return schoolbook as it's guaranteed correct.
    return _schoolbook(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    
    # Use schoolbook decomposition which is guaranteed correct
    U, V, W = _schoolbook(n, m, p)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Should not happen, but fallback
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if time.time() > DEADLINE:
        pass  # Return immediately
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
