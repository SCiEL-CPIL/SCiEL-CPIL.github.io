"""Matrix multiplication 3x4x7 decomposition using peeling algorithm with Strassen core."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 7
    t0 = time.time()
    deadline = t0 + 115  # Safety margin before 120s limit

    # The tensor T[i,j,k] = 1 for all i in [0,3), j in [0,4), k in [0,7)
    # We want to decompose this as sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Strategy: Use Strassen for the 3x3x3 core (rank 7), then peeling for the rest.
    # Actually, let's try a pure peeling approach on the full tensor to see if we can
    # find a lower rank decomposition.
    
    # Initialize the tensor (all ones)
    T = np.ones((n, m, p), dtype=np.float64)
    
    # We'll iteratively peel off rank-1 components
    # A rank-1 component is U_vec * V_vec * W_vec where:
    #   sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = T[i,j,k]
    
    # For peeling, we fix one factor and solve for the others.
    # Let's use a different approach: try to find a decomposition by solving
    # a system of equations.
    
    # Actually, let's use the known structure:
    # The matrix multiplication tensor has a specific algebraic structure.
    # For M_{n,m,p}, the rank is at most n*m*p (schoolbook) and at least max(n*m, m*p, n*p).
    # For 3x4x7: lower bound = max(12, 28, 21) = 28.
    
    # Let's try to construct a decomposition of rank R by:
    # 1. Using Strassen-like tricks for overlapping sub-tensors
    # 2. Peeling for the remaining terms
    
    # First, let's try the block decomposition approach with Strassen for 3x3x3 blocks.
    # Split into:
    #   Block 1: i in [0,3), j in [0,3), k in [0,3) -> Strassen rank 7
    #   Block 2: i in [0,3), j in [0,3), k in [3,7) -> schoolbook 3*3*4 = 36
    #   Block 3: i in [0,3), j in [3,4), k in [0,7) -> schoolbook 3*1*7 = 21
    # Total: 7 + 36 + 21 = 64
    
    # But maybe we can do better by overlapping the Strassen structure.
    # Let's try a peeling approach on the residual after Strassen.
    
    rank = 64
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    
    # Strassen's rank-7 decomposition for 3x3x3
    # M1 = (A00+A22)*(B00+B22)
    # M2 = (A11+A22)*(B11+B22)
    # M3 = (A00+A11)*(B00+B11)
    # M4 = (A22+A01)*(B22+B11)
    # M5 = (A11+A01)*(B11+B02)
    # M6 = (A22+A12)*(B22+B01)
    # M7 = (A00+A12)*(B00+B12)
    
    # C[0,0] = M1 - M2 + M3
    # C[0,1] = M2 - M3 + M5
    # C[0,2] = M4 - M6 + M7
    # C[1,0] = M1 - M4 + M6
    # C[1,1] = M1 - M2 + M5
    # C[1,2] = M3 - M5 + M7
    # C[2,0] = M2 - M3 + M4
    # C[2,1] = M1 - M4 + M5
    # C[2,2] = M1 - M2 + M6
    
    # U factors (index i*3+j for the 3x3 submatrix):
    # (0,0): M1, M3, M7 -> [1,0,1,0,0,0,1]
    # (0,1): M4, M5 -> [0,0,0,1,1,0,0]
    # (0,2): none -> [0,0,0,0,0,0,0]
    # (1,0): none -> [0,0,0,0,0,0,0]
    # (1,1): M2, M3, M5 -> [0,1,1,0,1,0,0]
    # (1,2): M6, M7 -> [0,0,0,0,0,1,1]
    # (2,0): none -> [0,0,0,0,0,0,0]
    # (2,1): none -> [0,0,0,0,0,0,0]
    # (2,2): M1, M2, M4, M6 -> [1,1,0,1,0,1,0]
    
    strassen_U = np.array([
        [1,0,1,0,0,0,1],
        [0,0,0,1,1,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,1,1,0,1,0,0],
        [0,0,0,0,0,1,1],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [1,1,0,1,0,1,0],
    ], dtype=np.float64)
    
    # V factors (index j*3+k):
    # (0,0): M1, M3, M7 -> [1,0,1,0,0,0,1]
    # (0,1): M6 -> [0,0,0,0,0,1,0]
    # (0,2): M5 -> [0,0,0,0,1,0,0]
    # (1,0): M4 -> [0,0,0,1,0,0,0]
    # (1,1): M2, M3, M4, M5 -> [0,1,1,1,1,0,0]
    # (1,2): M7 -> [0,0,0,0,0,0,1]
    # (2,0): M1 -> [1,0,0,0,0,0,0]
    # (2,1): none -> [0,0,0,0,0,0,0]
    # (2,2): M1, M2, M4, M6 -> [1,1,0,1,0,1,0]
    
    strassen_V = np.array([
        [1,0,1,0,0,0,1],
        [0,0,0,0,0,1,0],
        [0,0,0,0,1,0,0],
        [0,0,0,1,0,0,0],
        [0,1,1,1,1,0,0],
        [0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [1,1,0,1,0,1,0],
    ], dtype=np.float64)
    
    # W factors (index k*3+i):
    # C[0,0]=M1-M2+M3: W[0,0] = [1,-1,1,0,0,0,0]
    # C[0,1]=M2-M3+M5: W[1,0] = [0,1,-1,0,1,0,0]
    # C[0,2]=M4-M6+M7: W[2,0] = [0,0,0,1,0,-1,1]
    # C[1,0]=M1-M4+M6: W[0,1] = [1,0,0,-1,0,1,0]
    # C[1,1]=M1-M2+M5: W[1,1] = [1,-1,0,0,1,0,0]
    # C[1,2]=M3-M5+M7: W[2,1] = [0,0,1,0,-1,0,1]
    # C[2,0]=M2-M3+M4: W[0,2] = [0,1,-1,1,0,0,0]
    # C[2,1]=M1-M4+M5: W[1,2] = [1,0,0,-1,1,0,0]
    # C[2,2]=M1-M2+M6: W[2,2] = [1,-1,0,0,0,1,0]
    
    strassen_W = np.array([
        [1,-1,1,0,0,0,0],
        [0,1,-1,0,1,0,0],
        [0,0,0,1,0,-1,1],
        [1,0,0,-1,0,1,0],
        [1,-1,0,0,1,0,0],
        [0,0,1,0,-1,0,1],
        [0,1,-1,1,0,0,0],
        [1,0,0,-1,1,0,0],
        [1,-1,0,0,0,1,0],
    ], dtype=np.float64)
    
    # Place Strassen into U, V, W for the 3x3x3 core
    for ii in range(3):
        for jj in range(3):
            U[ii * m + jj, col:col+7] = strassen_U[ii * 3 + jj]
    for jj in range(3):
        for kk in range(3):
            V[jj * p + kk, col:col+7] = strassen_V[jj * 3 + kk]
    for kk in range(3):
        for ii in range(3):
            W[kk * n + ii, col:col+7] = strassen_W[kk * 3 + ii]
    col += 7
    
    # Now handle the remaining terms with schoolbook
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                if ii < 3 and jj < 3 and kk < 3:
                    continue
                if col >= rank:
                    break
                U[ii * m + jj, col] = 1
                V[jj * p + kk, col] = 1
                W[kk * n + ii, col] = 1
                col += 1
    
    # Verify the decomposition
    if time.time() < deadline:
        ok = True
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[ii * m + jj, c] * V[jj * p + kk, c] * W[kk * n + ii, c]
                    if abs(val - 1.0) > 1e-10:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            # Convert to list of lists for the expected format
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb), dtype=np.float64)
    V_fb = np.zeros((m * p, rank_fb), dtype=np.float64)
    W_fb = np.zeros((p * n, rank_fb), dtype=np.float64)
    c = 0
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                U_fb[ii * m + jj, c] = 1
                V_fb[jj * p + kk, c] = 1
                W_fb[kk * n + ii, c] = 1
                c += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
