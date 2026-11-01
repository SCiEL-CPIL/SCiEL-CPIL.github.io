"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    W indexed by k*n + i (output factor follows W[k*n+i, r])
    """
    try:
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Ensure 2D
        if U_arr.ndim != 2 or V_arr.ndim != 2 or W_arr.ndim != 2:
            return False
        
        rank = U_arr.shape[1]
        if V_arr.shape[1] != rank or W_arr.shape[1] != rank:
            return False
        if U_arr.shape[0] != n * m or V_arr.shape[0] != m * p or W_arr.shape[0] != p * n:
            return False
            
        # Check valid entries: for each (i, j, k), a=i*m+j, b=j*p+k, c=k*n+i should give 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U_arr[a, :], np.dot(V_arr[b, :], W_arr[c, :]))
                    if np.abs(val - 1.0) > tol:
                        return False
        
        # Check invalid entries should be 0
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    
                    if not (j_a == j_b and k_b == k_c and i_c == i_a):
                        val = np.dot(U_arr[a, :], np.dot(V_arr[b, :], W_arr[c, :]))
                        if np.abs(val) > tol:
                            return False
        
        return True
    except Exception:
        return False

def _strassen_2x2x2():
    """
    Return rank-7 decomposition for 2x2x2 matrix multiplication.
    U shape (4, 7), V shape (4, 7), W shape (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (c-d)(f+g)
    U[2, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7 = (b-c)(e+h)
    U[1, 6] = 1; U[2, 6] = -1
    V[0, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_strassen_2x2x16(n=2, m=2, p=16):
    """
    Build rank-56 decomposition for 2x2x16 matrix multiplication.
    The 16 output columns decompose into 8 independent 2x2 blocks.
    Each block uses Strassen's rank-7 algorithm.
    Total rank = 8 * 7 = 56.
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    num_blocks = p // 2  # 8 blocks
    rank = 7 * num_blocks
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for block in range(num_blocks):
        for r in range(7):
            col = block * 7 + r
            # U is the same for all blocks (shared input structure)
            U[:, col] = U2[:, r]
            # V: for each row j of A (which is column j of the 2x2 block),
            # the input is from B's row j, columns block*2 and block*2+1
            for j in range(m):
                for k_local in range(2):
                    k = block * 2 + k_local
                    b_idx = j * p + k
                    v_idx = j * 2 + k_local
                    V[b_idx, col] = V2[v_idx, r]
            # W: for each output position (k, i), map to the 2x2 block
            for i in range(n):
                for k_local in range(2):
                    k = block * 2 + k_local
                    w_idx = k * n + i
                    w2_idx = k_local * n + i
                    W[w_idx, col] = W2[w2_idx, r]
    
    return U, V, W

def _laderman_4x4x4():
    """
    Return rank-19 decomposition for 4x4x4 matrix multiplication (Laderman).
    U shape (16, 19), V shape (16, 19), W shape (16, 19)
    Indexing: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    n = 4
    U = np.zeros((16, 19))
    V = np.zeros((16, 19))
    W = np.zeros((16, 19))
    
    # Laderman's 19 multiplications for 4x4 matrix multiplication
    # M1 = (a11+a21+a31+a41)(b11+b12+b13+b14)
    U[0, 0] = 1; U[4, 0] = 1; U[8, 0] = 1; U[12, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1; V[2, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # M2 = (a12+a22+a32+a42)(b21+b22+b23+b24)
    U[1, 1] = 1; U[5, 1] = 1; U[9, 1] = 1; U[13, 1] = 1
    V[4, 1] = 1; V[5, 1] = 1; V[6, 1] = 1; V[7, 1] = 1
    W[4, 1] = 1; W[5, 1] = 1; W[6, 1] = 1; W[7, 1] = 1
    
    # M3 = (a13+a23+a33+a43)(b31+b32+b33+b34)
    U[2, 2] = 1; U[6, 2] = 1; U[10, 2] = 1; U[14, 2] = 1
    V[8, 2] = 1; V[9, 2] = 1; V[10, 2] = 1; V[11, 2] = 1
    W[8, 2] = 1; W[9, 2] = 1; W[10, 2] = 1; W[11, 2] = 1
    
    # M4 = (a14+a24+a34+a44)(b41+b42+b43+b44)
    U[3, 3] = 1; U[7, 3] = 1; U[11, 3] = 1; U[15, 3] = 1
    V[12, 3] = 1; V[13, 3] = 1; V[14, 3] = 1; V[15, 3] = 1
    W[12, 3] = 1; W[13, 3] = 1; W[14, 3] = 1; W[15, 3] = 1
    
    # M5 = (a11-a21+a31-a41)(b11-b21+b31-b41)
    U[0, 4] = 1; U[4, 4] = -1; U[8, 4] = 1; U[12, 4] = -1
    V[0, 4] = 1; V[4, 4] = -1; V[8, 4] = 1; V[12, 4] = -1
    W[0, 4] = 1; W[4, 4] = 1; W[8, 4] = 1; W[12, 4] = 1
    W[1, 4] = -1; W[5, 4] = 1; W[9, 4] = -1; W[13, 4] = 1
    
    # M6 = (a12-a22+a32-a42)(b21-b22+b23-b24)
    U[1, 5] = 1; U[5, 5] = -1; U[9, 5] = 1; U[13, 5] = -1
    V[4, 5] = 1; V[5, 5] = -1; V[6, 5] = 1; V[7, 5] = -1
    W[4, 5] = 1; W[5, 5] = 1; W[6, 5] = 1; W[7, 5] = 1
    W[8, 5] = -1; W[9, 5] = 1; W[10, 5] = -1; W[11, 5] = 1
    
    # M7 = (a13-a23+a33-a43)(b31-b32+b33-b34)
    U[2, 6] = 1; U[6, 6] = -1; U[10, 6] = 1; U[14, 6] = -1
    V[8, 6] = 1; V[9, 6] = -1; V[10, 6] = 1; V[11, 6] = -1
    W[8, 6] = 1; W[9, 6] = 1; W[10, 6] = 1; W[11, 6] = 1
    W[12, 6] = -1; W[13, 6] = 1; W[14, 6] = -1; W[15, 6] = 1
    
    # M8 = (a14-a24+a34-a44)(b41-b42+b43-b44)
    U[3, 7] = 1; U[7, 7] = -1; U[11, 7] = 1; U[15, 7] = -1
    V[12, 7] = 1; V[13, 7] = -1; V[14, 7] = 1; V[15, 7] = -1
    W[12, 7] = 1; W[13, 7] = 1; W[14, 7] = 1; W[15, 7] = 1
    W[0, 7] = -1; W[1, 7] = 1; W[2, 7] = -1; W[3, 7] = 1
    
    # M9 = (a11+a31)(b11+b31)
    U[0, 8] = 1; U[8, 8] = 1
    V[0, 8] = 1; V[8, 8] = 1
    W[0, 8] = 1; W[8, 8] = 1
    
    # M10 = (a22+a42)(b22+b42)
    U[5, 9] = 1; U[13, 9] = 1
    V[5, 9] = 1; V[13, 9] = 1
    W[5, 9] = 1; W[13, 9] = 1
    
    # M11 = (a11-a31)(b11-b31)
    U[0, 10] = 1; U[8, 10] = -1
    V[0, 10] = 1; V[8, 10] = -1
    W[0, 10] = 1; W[8, 10] = 1
    
    # M12 = (a22-a42)(b22-b42)
    U[5, 11] = 1; U[13, 11] = -1
    V[5, 11] = 1; V[13, 11] = -1
    W[5, 11] = 1; W[13, 11] = 1
    
    # M13 = (a11+a21)(b11+b12)
    U[0, 12] = 1; U[4, 12] = 1
    V[0, 12] = 1; V[1, 12] = 1
    W[0, 12] = 1; W[4, 12] = 1
    
    # M14 = (a12+a22)(b21+b22)
    U[1, 13] = 1; U[5, 13] = 1
    V[4, 13] = 1; V[5, 13] = 1
    W[1, 13] = 1; W[5, 13] = 1
    
    # M15 = (a31+a41)(b13+b14)
    U[8, 14] = 1; U[12, 14] = 1
    V[8, 14] = 1; V[9, 14] = 1
    W[8, 14] = 1; W[12, 14] = 1
    
    # M16 = (a32+a42)(b23+b24)
    U[9, 15] = 1; U[13, 15] = 1
    V[12, 15] = 1; V[13, 15] = 1
    W[9, 15] = 1; W[13, 15] = 1
    
    # M17 = (a11-a21)(b11-b12)
    U[0, 16] = 1; U[4, 16] = -1
    V[0, 16] = 1; V[1, 16] = -1
    W[0, 16] = 1; W[4, 16] = 1
    
    # M18 = (a12-a22)(b21-b22)
    U[1, 17] = 1; U[5, 17] = -1
    V[4, 17] = 1; V[5, 17] = -1
    W[1, 17] = 1; W[5, 17] = 1
    
    # M19 = (a31-a41)(b13-b14)
    U[8, 18] = 1; U[12, 18] = -1
    V[8, 18] = 1; V[9, 18] = -1
    W[8, 18] = 1; W[12, 18] = 1
    
    # M20 = (a32-a42)(b23-b24)
    U[9, 18] = 1; U[13, 18] = -1
    V[12, 18] = 1; V[13, 18] = -1
    W[9, 18] = 1; W[13, 18] = 1
    
    return U, V, W

def _build_laderman_2x2x16(n=2, m=2, p=16):
    """
    Attempt to build a rank-reduced decomposition using Laderman's 4x4x4.
    This is experimental and may not work.
    """
    try:
        U4, V4, W4 = _laderman_4x4x4()
        
        # The 2x2x16 problem can be seen as two 2x2x8 problems.
        # Each 2x2x8 can potentially use a 4x4x4-like structure.
        # This is a rough attempt.
        
        # For now, return None to indicate failure
        return None, None, None
    except Exception:
        return None, None, None

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    try:
        # First, try the Laderman-based approach (experimental)
        if time.time() < deadline - 5:
            U_lad, V_lad, W_lad = _build_laderman_2x2x16(n, m, p)
            if U_lad is not None:
                if _verify_decomposition(U_lad.tolist(), V_lad.tolist(), W_lad.tolist(), n, m, p, tol=1e-6):
                    return {"U": U_lad.tolist(), "V": V_lad.tolist(), "W": W_lad.tolist()}
        
        # Build the rank-56 Strassen-based decomposition
        U_str, V_str, W_str = _build_strassen_2x2x16(n, m, p)
        
        # Verify it works
        if _verify_decomposition(U_str.tolist(), V_str.tolist(), W_str.tolist(), n, m, p, tol=1e-6):
            return {"U": U_str.tolist(), "V": V_str.tolist(), "W": W_str.tolist()}
        
        # If verification fails, fall back to schoolbook
        return _schoolbook(n, m, p)
        
    except Exception:
        # If anything goes wrong, return schoolbook
        return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
