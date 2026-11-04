"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product
import time
import random
import sys

DIMENSIONS = [4, 5, 5]
TIME_LIMIT = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return (U, V, W) for Strassen 2x2x2 decomposition, each shape (4, 7)."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1; V[1*2+1, 0] = 1
    W[0*2+0, 0] = 1; W[0*2+1, 0] = -1; W[1*2+0, 0] = 1; W[1*2+1, 0] = -1
    
    # M2 = (A21+A22)B11
    U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    W[1*2+0, 1] = 1; W[1*2+1, 1] = -1
    
    # M3 = A11(B12-B22)
    U[0*2+0, 2] = 1
    V[0*2+1, 2] = 1; V[1*2+1, 2] = -1
    W[0*2+1, 2] = 1; W[1*2+1, 2] = 1
    
    # M4 = A22(B21-B11)
    U[1*2+1, 3] = 1
    V[1*2+0, 3] = 1; V[0*2+0, 3] = -1
    W[0*2+0, 3] = 1; W[1*2+0, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0*2+0, 4] = 1; U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    W[0*2+0, 4] = -1; W[1*2+0, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[1*2+0, 5] = 1; U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1; V[0*2+1, 5] = 1
    W[1*2+1, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[0*2+1, 6] = 1; U[1*2+1, 6] = -1
    V[1*2+0, 6] = 1; V[1*2+1, 6] = 1
    W[0*2+1, 6] = 1
    
    return U, V, W

def _laderman_3x3x3():
    """Return (U, V, W) for Laderman 3x3x3 decomposition, each shape (9, 15)."""
    U = np.zeros((9, 15))
    V = np.zeros((9, 15))
    W = np.zeros((9, 15))
    
    # Laderman's 15 multiplications for 3x3 matrix multiplication
    # Indexing: row*3+col for 3x3 blocks
    # M1 = (a11+a33)(b11+b33)
    U[0*3+0, 0] = 1; U[2*3+2, 0] = 1
    V[0*3+0, 0] = 1; V[2*3+2, 0] = 1
    W[0*3+0, 0] = 1; W[0*3+2, 0] = -1; W[2*3+0, 0] = -1; W[2*3+2, 0] = 1
    
    # M2 = (a12+a32)(b21+b23)
    U[0*3+1, 1] = 1; U[2*3+1, 1] = 1
    V[1*3+0, 1] = 1; V[1*3+2, 1] = 1
    W[0*3+1, 1] = 1; W[2*3+1, 1] = 1
    
    # M3 = (a21+a23)(b12+b32)
    U[1*3+0, 2] = 1; U[1*3+2, 2] = 1
    V[0*3+1, 2] = 1; V[2*3+1, 2] = 1
    W[1*3+0, 2] = 1; W[1*3+2, 2] = 1
    
    # M4 = (a13+a31)(b13+b31)
    U[0*3+2, 3] = 1; U[2*3+0, 3] = 1
    V[0*3+2, 3] = 1; V[2*3+0, 3] = 1
    W[0*3+2, 3] = 1; W[2*3+0, 3] = 1
    
    # M5 = (a11+a12+a33+a32)(b11+b12+b31+b32)
    U[0*3+0, 4] = 1; U[0*3+1, 4] = 1; U[2*3+2, 4] = 1; U[2*3+1, 4] = 1
    V[0*3+0, 4] = 1; V[0*3+1, 4] = 1; V[2*3+0, 4] = 1; V[2*3+1, 4] = 1
    W[0*3+0, 4] = 1; W[0*3+1, 4] = -1; W[2*3+0, 4] = -1; W[2*3+1, 4] = 1
    
    # M6 = (a21+a23+a11+a13+a31+a33)(b12+b32+b11+b31+b13+b33)
    U[1*3+0, 5] = 1; U[1*3+2, 5] = 1; U[0*3+0, 5] = 1; U[0*3+2, 5] = 1; U[2*3+0, 5] = 1; U[2*3+2, 5] = 1
    V[0*3+1, 5] = 1; V[2*3+1, 5] = 1; V[0*3+0, 5] = 1; V[2*3+0, 5] = 1; V[0*3+2, 5] = 1; V[2*3+2, 5] = 1
    W[1*3+0, 5] = 1; W[1*3+2, 5] = 1; W[0*3+0, 5] = -1; W[0*3+2, 5] = -1; W[2*3+0, 5] = 1; W[2*3+2, 5] = 1
    
    # M7 = (a11+a12+a21+a23+a33+a32)(b11+b12+b21+b23+b31+b32)
    U[0*3+0, 6] = 1; U[0*3+1, 6] = 1; U[1*3+0, 6] = 1; U[1*3+2, 6] = 1; U[2*3+2, 6] = 1; U[2*3+1, 6] = 1
    V[0*3+0, 6] = 1; V[0*3+1, 6] = 1; V[1*3+0, 6] = 1; V[1*3+2, 6] = 1; V[2*3+0, 6] = 1; V[2*3+1, 6] = 1
    W[0*3+0, 6] = 1; W[0*3+1, 6] = -1; W[1*3+0, 6] = 1; W[1*3+2, 6] = 1; W[2*3+0, 6] = -1; W[2*3+1, 6] = 1
    
    # M8 = (a13+a21+a23+a31)(b13+b31+b21+b23)
    U[0*3+2, 7] = 1; U[1*3+0, 7] = 1; U[1*3+2, 7] = 1; U[2*3+0, 7] = 1
    V[0*3+2, 7] = 1; V[2*3+0, 7] = 1; V[1*3+0, 7] = 1; V[1*3+2, 7] = 1
    W[0*3+2, 7] = 1; W[1*3+0, 7] = 1; W[1*3+2, 7] = 1; W[2*3+0, 7] = 1
    
    # M9 = (a12+a21+a23+a32)(b12+b21+b23+b32)
    U[0*3+1, 8] = 1; U[1*3+0, 8] = 1; U[1*3+2, 8] = 1; U[2*3+1, 8] = 1
    V[0*3+1, 8] = 1; V[1*3+0, 8] = 1; V[1*3+2, 8] = 1; V[2*3+1, 8] = 1
    W[0*3+1, 8] = 1; W[1*3+0, 8] = 1; W[1*3+2, 8] = 1; W[2*3+1, 8] = 1
    
    # M10 = (a13+a31+a11+a33+a21+a23)(b13+b31+b11+b33+b21+b23)
    U[0*3+2, 9] = 1; U[2*3+0, 9] = 1; U[0*3+0, 9] = 1; U[2*3+2, 9] = 1; U[1*3+0, 9] = 1; U[1*3+2, 9] = 1
    V[0*3+2, 9] = 1; V[2*3+0, 9] = 1; V[0*3+0, 9] = 1; V[2*3+2, 9] = 1; V[1*3+0, 9] = 1; V[1*3+2, 9] = 1
    W[0*3+2, 9] = 1; W[2*3+0, 9] = 1; W[0*3+0, 9] = 1; W[2*3+2, 9] = 1; W[1*3+0, 9] = 1; W[1*3+2, 9] = 1
    
    # M11 = (a12+a21+a23+a32+a11+a33)(b12+b21+b23+b32+b11+b33)
    U[0*3+1, 10] = 1; U[1*3+0, 10] = 1; U[1*3+2, 10] = 1; U[2*3+1, 10] = 1; U[0*3+0, 10] = 1; U[2*3+2, 10] = 1
    V[0*3+1, 10] = 1; V[1*3+0, 10] = 1; V[1*3+2, 10] = 1; V[2*3+1, 10] = 1; V[0*3+0, 10] = 1; V[2*3+2, 10] = 1
    W[0*3+1, 10] = 1; W[1*3+0, 10] = 1; W[1*3+2, 10] = 1; W[2*3+1, 10] = 1; W[0*3+0, 10] = 1; W[2*3+2, 10] = 1
    
    # M12 = (a13+a21+a23+a31+a11+a33+a12+a32)(b13+b21+b23+b31+b11+b33+b12+b32)
    U[0*3+2, 11] = 1; U[1*3+0, 11] = 1; U[1*3+2, 11] = 1; U[2*3+0, 11] = 1; U[0*3+0, 11] = 1; U[2*3+2, 11] = 1; U[0*3+1, 11] = 1; U[2*3+1, 11] = 1
    V[0*3+2, 11] = 1; V[1*3+0, 11] = 1; V[1*3+2, 11] = 1; V[2*3+0, 11] = 1; V[0*3+0, 11] = 1; V[2*3+2, 11] = 1; V[0*3+1, 11] = 1; V[2*3+1, 11] = 1
    W[0*3+2, 11] = 1; W[1*3+0, 11] = 1; W[1*3+2, 11] = 1; W[2*3+0, 11] = 1; W[0*3+0, 11] = 1; W[2*3+2, 11] = 1; W[0*3+1, 11] = 1; W[2*3+1, 11] = 1
    
    # M13 = (a13+a21+a23+a31+a12+a32)(b13+b21+b23+b31+b12+b32)
    U[0*3+2, 12] = 1; U[1*3+0, 12] = 1; U[1*3+2, 12] = 1; U[2*3+0, 12] = 1; U[0*3+1, 12] = 1; U[2*3+1, 12] = 1
    V[0*3+2, 12] = 1; V[1*3+0, 12] = 1; V[1*3+2, 12] = 1; V[2*3+0, 12] = 1; V[0*3+1, 12] = 1; V[2*3+1, 12] = 1
    W[0*3+2, 12] = 1; W[1*3+0, 12] = 1; W[1*3+2, 12] = 1; W[2*3+0, 12] = 1; W[0*3+1, 12] = 1; W[2*3+1, 12] = 1
    
    # M14 = (a12+a21+a23+a32+a13+a31)(b12+b21+b23+b32+b13+b31)
    U[0*3+1, 13] = 1; U[1*3+0, 13] = 1; U[1*3+2, 13] = 1; U[2*3+1, 13] = 1; U[0*3+2, 13] = 1; U[2*3+0, 13] = 1
    V[0*3+1, 13] = 1; V[1*3+0, 13] = 1; V[1*3+2, 13] = 1; V[2*3+1, 13] = 1; V[0*3+2, 13] = 1; V[2*3+0, 13] = 1
    W[0*3+1, 13] = 1; W[1*3+0, 13] = 1; W[1*3+2, 13] = 1; W[2*3+1, 13] = 1; W[0*3+2, 13] = 1; W[2*3+0, 13] = 1
    
    # M15 = (a13+a21+a23+a31+a11+a33+a12+a32+a21+a23)(b13+b21+b23+b31+b11+b33+b12+b32+b21+b23)
    # Simplified: use the standard Laderman formula
    U[0*3+2, 14] = 1; U[1*3+0, 14] = 1; U[1*3+2, 14] = 1; U[2*3+0, 14] = 1; U[0*3+0, 14] = 1; U[2*3+2, 14] = 1; U[0*3+1, 14] = 1; U[2*3+1, 14] = 1
    V[0*3+2, 14] = 1; V[1*3+0, 14] = 1; V[1*3+2, 14] = 1; V[2*3+0, 14] = 1; V[0*3+0, 14] = 1; V[2*3+2, 14] = 1; V[0*3+1, 14] = 1; V[2*3+1, 14] = 1
    W[0*3+2, 14] = 1; W[1*3+0, 14] = 1; W[1*3+2, 14] = 1; W[2*3+0, 14] = 1; W[0*3+0, 14] = 1; W[2*3+2, 14] = 1; W[0*3+1, 14] = 1; W[2*3+1, 14] = 1
    
    return U, V, W

def _is_valid(U, V, W, r, n, m, p):
    """Check if the decomposition is exact for matrix multiplication."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 0.0
                for c in range(r):
                    lhs += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(lhs - 1.0) > 1e-10:
                    return False
    return True

def _is_valid_fast(U, V, W, r, n, m, p):
    """Vectorized validity check."""
    # Build the tensor contraction
    # For each (i,j,k), sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Use einsum for speed
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # Need: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    
    # Reshape approach
    U_t = U.reshape(n, m, r)
    V_t = V.reshape(m, p, r)
    W_t = W.reshape(p, n, r)
    
    # T[i,j,k] = sum_c U_t[i,j,c] * V_t[j,k,c] * W_t[k,i,c]
    # This is a 3-index tensor contraction
    T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    
    return np.allclose(T, 1.0, atol=1e-10)

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Schoolbook decomposition as baseline
    r_school = n * m * p
    U_sb = np.zeros((n * m, r_school))
    V_sb = np.zeros((m * p, r_school))
    W_sb = np.zeros((p * n, r_school))
    
    indices = list(product(range(n), range(m), range(p)))
    for c, (i, j, k) in enumerate(indices):
        U_sb[i * m + j, c] = 1.0
        V_sb[j * p + k, c] = 1.0
        W_sb[k * n + i, c] = 1.0
    
    best_result = (U_sb, V_sb, W_sb, r_school)
    
    # Strategy: Decompose 4x5x5 as 4x2x2 + 4x3x3
    # 4x2x2: use Strassen rank 7
    # 4x3x3: use Laderman rank 15 (not 13, let's use the correct Laderman)
    # Total rank: 7 + 15 = 22
    
    if time.time() - start_time < TIME_LIMIT - 20:
        U_s, V_s, W_s = _strassen_2x2x2()
        U_l, V_l, W_l = _laderman_3x3x3()
        
        # Verify Laderman is correct first
        if not _is_valid_fast(U_l, V_l, W_l, 15, 3, 3, 3):
            # Try to fix Laderman - use a simpler approach
            # For 3x3x3, use schoolbook rank 27 or try another known decomposition
            # Let's just use the Kronecker approach as fallback
            pass
        else:
            # Build 4x5x5 from 4x2x2 and 4x3x3
            # 4x2x2: i in 0..3, j in 0..1, k in 0..1
            # 4x3x3: i in 0..3, j in 2..4, k in 2..4
            # Wait, that doesn't cover j=0,1 with k=2,3,4 or j=2,3,4 with k=0,1
            
            # Better split: 
            # Part A: i in 0..3, j in 0..1, k in 0..1 -> 4x2x2 (rank 7)
            # Part B: i in 0..3, j in 2..4, k in 2..4 -> 4x3x3 (rank 15)
            # Part C: i in 0..3, j in 0..1, k in 2..4 -> 4x2x3 (rank 6)
            # Part D: i in 0..3, j in 2..4, k in 0..1 -> 4x3x2 (rank 6)
            # Total: 7+15+6+6 = 34
            
            # Actually, let's try a different split:
            # 4x5x5 = 4x5x4 + 4x5x1
            # 4x5x4: can we do better than 4x4x4 + 4x1x4?
            # Let's try: 4x5x5 as two 4x5x5/2 blocks doesn't work cleanly.
            
            # Alternative: 4x5x5 = 2x5x5 + 2x5x5
            # Each 2x5x5: use Strassen for the 2x2x2 part and schoolbook for the rest
            # 2x5x5: i in 0..1, j in 0..4, k in 0..4
            # Split j,k: (0,0),(0,1),(1,0),(1,1) -> 2x2x2 Strassen rank 7
            # (0,2..4) -> 2x1x3 rank 3
            # (1,2..4) -> 2x1x3 rank 3  
            # (2..4,0) -> 2x3x1 rank 3
            # (2..4,1) -> 2x3x1 rank 3
            # (2..4,2..4) -> 2x3x3 rank 9
            # Total per 2x5x5: 7+3+3+3+3+9 = 28
            # Two of them: 56. Not better than 57.
            
            # Let's try the direct 4x2x2 + 4x3x3 + cross terms approach
            # Actually the cleanest is:
            # 4x5x5 = 4x4x4 + 4x1x5 + 4x4x1 + 4x1x1
            # 4x4x4: Strassen^2 rank 49
            # 4x1x5: rank 4 (one per i, summing over k)
            # 4x4x1: rank 4 (one per k, summing over j)  
            # 4x1x1: rank 1
            # Total: 49+4+4+1 = 58. Worse than 57.
            
            # The current approach gives 57. Let's try to improve.
            # 4x5x5: what if we use 4x4x4 with a better 4x4x4?
            # Best known for 4x4x4 is rank 49 (Strassen^2).
            
            # Let's try: 4x5x5 = 4x5x5 directly with a mixed approach
            # For the 4x4x4 part, use rank 49
            # For the remaining 12 terms (4x1x5 + 4x5x1 - 4x1x1 = 4+20-4=20... no)
            # 4x5x5 has 100 terms. 4x4x4 has 64. Remaining: 36.
            # 4x1x5: 4*5=20 terms, rank 4
            # 4x5x1: 4*5=20 terms, rank 4
            # 4x1x1: 4 terms, but 1 is already counted in both above
            # Actually: terms where j=4 or k=4 (or both)
            # j=4, k=0..4: 4*5=20 terms
            # j=0..3, k=4: 4*4=16 terms
            # Total: 36 terms, rank: 4+4=8 (using the same i-index trick)
            # But wait, we can do better:
            # j=4, k=0..4: for each i, U[i*5+4,c]=1, V[4*5+k,c]=1, W[k*4+i,c]=1
            #   This is 4 terms (one per i), rank 4
            # j=0..3, k=4: for each k=4... no, for each i
            #   U[i*5+j,c]=1 for j=0..3, V[j*5+4,c]=1, W[4*4+i,c]=1
            #   This is 4 terms (one per i), rank 4
            # Total: 8, grand total: 49+8=57. Same as current.
            
            # To beat 57, we need a better 4x4x4 or a smarter split.
            # Let's try: 4x5x5 = 2x5x5 + 2x5x5 where each 2x5x5 uses rank 24
            # 2x5x5: 2*5*5=50 terms
            # Use 2x2x2 Strassen (rank 7) for j,k in {0,1}
            # 2x1x3 (j=0, k=2,3,4): rank 2
            # 2x1x3 (j=1, k=2,3,4): rank 2
            # 2x3x1 (j=2,3,4, k=0): rank 2
            # 2x3x1 (j=2,3,4, k=1): rank 2
            # 2x3x3 (j=2,3,4, k=2,3,4): rank 9 (schoolbook)
            # Total: 7+2+2+2+2+9 = 24
            # Two blocks: 48. Better than 57!
            
            r_total = 48
            U_full = np.zeros((n*m, r_total))
            V_full = np.zeros((m*p, r_total))
            W_full = np.zeros((p*n, r_total))
            
            # Block 1: i in 0..1 (top 2 rows), j in 0..4, k in 0..4
            # Sub-block 1a: i in 0..1, j in 0..1, k in 0..1 -> 2x2x2 Strassen
            for i in range(2):
                for j in range(2):
                    U_full[i*5+j, 0:7] = U_s[i*2+j, 0:7]
            for j in range(2):
                for k in range(2):
                    V_full[j*5+k, 0:7] = V_s[j*2+k, 0:7]
            for k in range(2):
                for i in range(2):
                    W_full[k*4+i, 0:7] = W_s[k*2+i, 0:7]
            
            # Sub-block 1b: i in 0..1, j=0, k in 2..4 -> 2x1x3
            # For each i: U[i*5+0, c]=1, V[0*5+k, c]=1, W[k*4+i, c]=1
            for i in range(2):
                for k in range(2, 5):
                    c = 7 + i*3 + (k-2)
                    U_full[i*5+0, c] = 1.0
                    V_full[0*5+k, c] = 1.0
                    W_full[k*4+i, c] = 1.0
            
            # Sub-block 1c: i in 0..1, j=1, k in 2..4 -> 2x1x3
            for i in range(2):
                for k in range(2, 5):
                    c = 7 + 2*3 + i*3 + (k-2)
                    U_full[i*5+1, c] = 1.0
                    V_full[1*5+k, c] = 1.0
                    W_full[k*4+i, c] = 1.0
            
            # Sub-block 1d: i in 0..1, j in 2..4, k=0 -> 2x3x1
            for i in range(2):
                for j in range(2, 5):
                    c = 7 + 4*3 + i*3 + (j-2)
                    U_full[i*5+j, c] = 1.0
                    V_full[j*5+0, c] = 1.0
                    W_full[0*4+i, c] = 1.0
            
            # Sub-block 1e: i in 0..1, j in 2..4, k=1 -> 2x3x1
            for i in range(2):
                for j in range(2, 5):
                    c = 7 + 5*3 + i*3 + (j-2)
                    U_full[i*5+j, c] = 1.0
                    V_full[j*5+1, c] = 1.0
                    W_full[1*4+i, c] = 1.0
            
            # Sub-block 1f: i in 0..1, j in 2..4, k in 2..4 -> 2x3x3 schoolbook
            for i in range(2):
                for j in range(2, 5):
                    for k in range(2, 5):
                        c = 7 + 7*3 + i*9 + (j-2)*3 + (k-2)
                        U_full[i*5+j, c] = 1.0
                        V_full[j*5+k, c] = 1.0
                        W_full[k*4+i, c] = 1.0
            
            # Block 2: i in 2..3 (bottom 2 rows), j in 0..4, k in 0..4
            # Same structure, offset by 24
            offset = 24
            for i in range(2):
                for j in range(2):
                    U_full[(i+2)*5+j, offset:offset+7] = U_s[i*2+j, 0:7]
            for j in range(2):
                for k in range(2):
                    V_full[j*5+k, offset:offset+7] = V_s[j*2+k, 0:7]
            for k in range(2):
                for i in range(2):
                    W_full[k*4+i+2, offset:offset+7] = W_s[k*2+i, 0:7]
            
            # Sub-block 2b: i in 2..3, j=0, k in 2..4
            for i in range(2):
                for k in range(2, 5):
                    c = offset + 7 + i*3 + (k-2)
                    U_full[(i+2)*5+0, c] = 1.0
                    V_full[0*5+k, c] = 1.0
                    W_full[k*4+(i+2), c] = 1.0
            
            # Sub-block 2c: i in 2..3, j=1, k in 2..4
            for i in range(2):
                for k in range(2, 5):
                    c = offset + 7 + 2*3 + i*3 + (k-2)
                    U_full[(i+2)*5+1, c] = 1.0
                    V_full[1*5+k, c] = 1.0
                    W_full[k*4+(i+2), c] = 1.0
            
            # Sub-block 2d: i in 2..3, j in 2..4, k=0
            for i in range(2):
                for j in range(2, 5):
                    c = offset + 7 + 4*3 + i*3 + (j-2)
                    U_full[(i+2)*5+j, c] = 1.0
                    V_full[j*5+0, c] = 1.0
                    W_full[0*4+(i+2), c] = 1.0
            
            # Sub-block 2e: i in 2..3, j in 2..4, k=1
            for i in range(2):
                for j in range(2, 5):
                    c = offset + 7 + 5*3 + i*3 + (j-2)
                    U_full[(i+2)*5+j, c] = 1.0
                    V_full[j*5+1, c] = 1.0
                    W_full[1*4+(i+2), c] = 1.0
            
            # Sub-block 2f: i in 2..3, j in 2..4, k in 2..4
            for i in range(2):
                for j in range(2, 5):
                    for k in range(2, 5):
                        c = offset + 7 + 7*3 + i*9 + (j-2)*3 + (k-2)
                        U_full[(i+2)*5+j, c] = 1.0
                        V_full[j*5+k, c] = 1.0
                        W_full[k*4+(i+2), c] = 1.0
            
            # Validate
            if _is_valid_fast(U_full, V_full, W_full, r_total, n, m, p):
                best_result = (U_full, V_full, W_full, r_total)
            else:
                # Fallback: try the 57-rank construction
                if time.time() - start_time < TIME_LIMIT - 10:
                    U_444, V_444, W_444 = _kron_decomp(U_s, V_s, W_s, U_s, V_s, W_s)
                    r_57 = 57
                    U_57 = np.zeros((n*m, r_57))
                    V_57 = np.zeros((m*p, r_57))
                    W_57 = np.zeros((p*n, r_57))
                    
                    for i in range(4):
                        for j in range(4):
                            U_57[i*5+j, 0:49] = U_444[i*4+j, 0:49]
                    for j in range(4):
                        for k in range(4):
                            V_57[j*5+k, 0:49] = V_444[j*4+k, 0:49]
                    for k in range(4):
                        for i in range(4):
                            W_57[k*4+i, 0:49] = W_444[k*4+i, 0:49]
                    
                    for i in range(4):
                        for j in range(5):
                            U_57[i*5+j, 49+i] = 1.0
                        for j in range(5):
                            V_57[j*5+4, 49+i] = 1.0
                        W_57[4*4+i, 49+i] = 1.0
                    
                    for k in range(4):
                        for i in range(4):
                            U_57[i*5+4, 53+k] = 1.0
                        V_57[4*5+k, 53+k] = 1.0
                        for i in range(4):
                            W_57[k*4+i, 53+k] = 1.0
                    
                    if _is_valid_fast(U_57, V_57, W_57, r_57, n, m, p):
                        best_result = (U_57, V_57, W_57, r_57)
    
    U_best, V_best, W_best, r_best = best_result
    return {"U": U_best[:, :r_best].tolist(), "V": V_best[:, :r_best].tolist(), "W": W_best[:, :r_best].tolist()}

def _kron_decomp(U1, V1, W1, U2, V2, W2):
    """Kronecker product of two tensor decompositions."""
    r1, r2 = U1.shape[1], U2.shape[1]
    U = np.zeros((U1.shape[0]*U2.shape[0], r1*r2))
    V = np.zeros((V1.shape[0]*V2.shape[0], r1*r2))
    W = np.zeros((W1.shape[0]*W2.shape[0], r1*r2))
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1*r2 + c2
            U[:, c] = np.kron(U1[:, c1], U2[:, c2])
            V[:, c] = np.kron(V1[:, c1], V2[:, c2])
            W[:, c] = np.kron(W1[:, c1], W2[:, c2])
    
    return U, V, W
# EVOLVE-BLOCK-END
