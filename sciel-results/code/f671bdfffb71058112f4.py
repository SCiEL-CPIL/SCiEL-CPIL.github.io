"""Matrix multiplication 3x4x8 rank-minimal decomposition search."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def _verify_safe(U, V, W, n=3, m=4, p=8, tol=1e-6):
    """
    Safely verify the decomposition using vectorized operations.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    Target: T[i,j,k] = 1 for all i,j,k.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    if U_arr.shape[1] != V_arr.shape[1] or U_arr.shape[1] != W_arr.shape[1]:
        return False
    rank = U_arr.shape[1]
    
    # Reshape for vectorized computation
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T_recon = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    if not np.allclose(T_recon, 1.0, atol=tol):
        return False
    return True

def _construct_rank_56():
    """
    Construct a rank-56 decomposition using rank-6 algorithms for 2x2x4 sub-problems.
    
    Strategy:
    1. Handle i=0 (1x4x8) with rank 32 (Schoolbook).
    2. Handle i=1,2 (2x4x8) with rank 24.
       - Split B into 4 blocks of 4x2 (columns 0-1, 2-3, 4-5, 6-7)
       - For each block, compute C_block(2x2) = A(2x4) * B_block(4x2)
       - This is a 2x4x2 problem, equivalent to 2x2x4 by symmetry
       - Use rank-6 algorithm for each: 4 * 6 = 24
       Grand total: 32 + 24 = 56
    """
    n, m, p = 3, 4, 8
    rank = 56
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: i=0, schoolbook 1x4x8 (32 terms)
    for j in range(4):
        for k in range(8):
            U[0 * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + 0, col] = 1.0
            col += 1
    
    # Part 2: i=1,2 using rank-6 algorithm for 2x4x2 sub-problems
    # For each k_block in 0..3, B_block is rows 0..3, cols k_block*2..k_block*2+1
    # C_block(2x2) = A(2x4) * B_block(4x2)
    
    # Rank-6 algorithm for 2x2x4: C(2x2) = A(2x2) * B(2x4)
    # Products:
    # P1 = (A00+A11) * (B00+B01)
    # P2 = A10 * B00
    # P3 = A01 * B11
    # P4 = (A00+A11+A01) * B00
    # P5 = (A10+A00) * B01
    # P6 = A11 * (B00+B01)
    
    # For our case: A is 2x4, B_block is 4x2
    # C_ij = sum_l A_il * B_lj
    # = sum_{l=0}^{1} A_il * B_lj + sum_{l=2}^{3} A_il * B_lj
    # Let A_L = A[:,0:2], A_R = A[:,2:4]
    # Let B_top = B_block[0:2,:], B_bot = B_block[2:4,:]
    # C = A_L * B_top + A_R * B_bot
    
    # This is a sum of two 2x2x2 problems. But we can do better with rank-6.
    
    # Actually, let me use the direct rank-6 for 2x4x2:
    # C(2x2) = A(2x4) * B(4x2)
    # 6 products:
    # P1 = (A00+A12) * (B00+B20)  # where A00=A[0,0], A12=A[1,2], B00=B[0,0], B20=B[2,0]
    # This is getting complex. Let me use a verified rank-6 algorithm.
    
    # Standard rank-6 for 2x2x4 (C(2x2) = A(2x2) * B(2x4)):
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e  
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # where A = [[a,b],[c,d]], B = [[e,f],[g,h]] (B is 2x2, but we need 2x4)
    
    # For 2x2x4: B is 2x4 = [[e,f,g,h],[i,j,k,l]]
    # C00 = a*e + b*i
    # C01 = a*f + b*j
    # C02 = a*g + b*k
    # C03 = a*h + b*l
    # C10 = c*e + d*i
    # C11 = c*f + d*j
    # C12 = c*g + d*k
    # C13 = c*h + d*l
    
    # Rank-6 algorithm (from Baur-Strassen or similar):
    # P1 = (a+d) * (e+f)
    # P2 = c * e
    # P3 = b * (f-e)
    # P4 = (a+c) * (e+f)  -- not quite
    # Let me look up the actual rank-6 algorithm.
    
    # Actually, I recall now: R(2x2x4) = 14, not 6. The rank-6 claim was wrong.
    # Let me fall back to the rank-88 solution which is verified.
    
    return None

def _construct_rank_88():
    """
    Construct a rank-88 decomposition.
    Split A (3x4) into A0 (1x4) and A12 (2x4).
    A0 x B (4x8): Schoolbook, rank 32.
    A12 x B (4x8): Split A12 into two 2x2 blocks (left/right).
                   Split B into four 2x2 blocks (top/bottom, left/right).
                   8 sub-problems of 2x2x2, each Strassen (rank 7).
                   Total rank 56.
    Grand total: 32 + 56 = 88.
    """
    n, m, p = 3, 4, 8
    rank = 88
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: i=0, schoolbook 1x4x8
    for j in range(4):
        for k in range(8):
            U[0 * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + 0, col] = 1.0
            col += 1
            
    # Part 2: i=1,2 using Strassen on 2x2 blocks
    strassen_U = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]).T  # shape (4, 7)
    
    strassen_V = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]).T  # shape (4, 7)
    
    strassen_W = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]).T  # shape (4, 7)
    
    for k_block in range(4):
        k_start = k_block * 2
        
        # Sub-problem 1: A_L (cols 0-1) * B_TL (rows 0-1, cols k_start..k_start+1)
        for t in range(7):
            U[4, col] += strassen_U[0, t]
            U[5, col] += strassen_U[1, t]
            U[8, col] += strassen_U[2, t]
            U[9, col] += strassen_U[3, t]
            
            V[k_start, col] += strassen_V[0, t]
            V[k_start+1, col] += strassen_V[1, t]
            V[p+k_start, col] += strassen_V[2, t]
            V[p+k_start+1, col] += strassen_V[3, t]
            
            W[k_start*n + 1, col] += strassen_W[0, t]
            W[(k_start+1)*n + 1, col] += strassen_W[1, t]
            W[k_start*n + 2, col] += strassen_W[2, t]
            W[(k_start+1)*n + 2, col] += strassen_W[3, t]
            
            col += 1
            
        # Sub-problem 2: A_R (cols 2-3) * B_BL (rows 2-3, cols k_start..k_start+1)
        for t in range(7):
            U[6, col] += strassen_U[0, t]
            U[7, col] += strassen_U[1, t]
            U[10, col] += strassen_U[2, t]
            U[11, col] += strassen_U[3, t]
            
            V[2*p+k_start, col] += strassen_V[0, t]
            V[2*p+k_start+1, col] += strassen_V[1, t]
            V[3*p+k_start, col] += strassen_V[2, t]
            V[3*p+k_start+1, col] += strassen_V[3, t]
            
            W[k_start*n + 1, col] += strassen_W[0, t]
            W[(k_start+1)*n + 1, col] += strassen_W[1, t]
            W[k_start*n + 2, col] += strassen_W[2, t]
            W[(k_start+1)*n + 2, col] += strassen_W[3, t]
            
            col += 1

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Fallback: Schoolbook
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_schoolbook
    
    # Try Rank 88
    if time.time() < deadline:
        try:
            U_88, V_88, W_88 = _construct_rank_88()
            if _verify_safe(U_88, V_88, W_88):
                if U_88.shape[1] < best_rank:
                    best_result = {"U": U_88.tolist(), "V": V_88.tolist(), "W": W_88.tolist()}
                    best_rank = U_88.shape[1]
        except Exception:
            pass

    return best_result
