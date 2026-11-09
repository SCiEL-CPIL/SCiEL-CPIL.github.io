"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START

def _schoolbook():
    n, m, p = DIMENSIONS
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
    return U, V, W


def _round_to_half_int(arr, tol=1e-6):
    """Round each entry to the nearest half-integer; return rounded array and success flag."""
    rounded = np.round(arr * 2) / 2
    diff = np.abs(arr - rounded)
    return rounded, np.all(diff <= tol)


def _verify_fast(U, V, W, tol=1e-8):
    """Vectorized verification using einsum."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    return np.allclose(T, np.ones((n, m, p)), atol=tol)


def _strassen_2x2x2():
    """
    Return Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
    For C = A * B where A, B are 2x2 matrices:
    A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    
    Strassen products:
    M1 = (a00 + a11) * (b00 + b11)
    M2 = (a10 + a11) * b00
    M3 = a00 * (b01 - b11)
    M4 = a11 * (b10 - b00)
    M5 = (a00 + a01) * b11
    M6 = (a10 - a00) * (b00 + b01)
    M7 = (a01 - a11) * (b10 + b11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    Returns U, V, W arrays of shape (4, 7), (4, 7), (4, 7) for the 2x2x2 case.
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    R = 7
    U = np.zeros((4, R))
    V = np.zeros((4, R))
    W = np.zeros((4, R))
    
    # M1 = (a00 + a11) * (b00 + b11)
    # U: a00 -> U[0,0]=1, a11 -> U[3,0]=1
    # V: b00 -> V[0,0]=1, b11 -> V[3,0]=1
    # W: contributes to C00 (k=0,i=0 -> W[0,0]), C11 (k=1,i=1 -> W[3,0])
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C00 += M1, C11 += M1
    
    # M2 = (a10 + a11) * b00
    # U: a10 -> U[2,1]=1, a11 -> U[3,1]=1
    # V: b00 -> V[0,1]=1
    # W: contributes to C10 (k=0,i=1 -> W[2,1])
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C10 += M2
    
    # M3 = a00 * (b01 - b11)
    # U: a00 -> U[0,2]=1
    # V: b01 -> V[1,2]=1, b11 -> V[3,2]=-1
    # W: contributes to C01 (k=1,i=0 -> W[2,2]), C11 (k=1,i=1 -> W[3,2])
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1  # C01 += M3, C11 += M3
    
    # M4 = a11 * (b10 - b00)
    # U: a11 -> U[3,3]=1
    # V: b10 -> V[2,3]=1, b00 -> V[0,3]=-1
    # W: contributes to C00 (k=0,i=0 -> W[0,3]), C10 (k=0,i=1 -> W[2,3])
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1  # C00 += M4, C10 += M4
    
    # M5 = (a00 + a01) * b11
    # U: a00 -> U[0,4]=1, a01 -> U[1,4]=1
    # V: b11 -> V[3,4]=1
    # W: contributes to C00 (k=0,i=0 -> W[0,4]), C01 (k=1,i=0 -> W[2,4])
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1  # C00 -= M5, C01 += M5
    
    # M6 = (a10 - a00) * (b00 + b01)
    # U: a10 -> U[2,5]=1, a00 -> U[0,5]=-1
    # V: b00 -> V[0,5]=1, b01 -> V[1,5]=1
    # W: contributes to C11 (k=1,i=1 -> W[3,5])
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C11 += M6
    
    # M7 = (a01 - a11) * (b10 + b11)
    # U: a01 -> U[1,6]=1, a11 -> U[3,6]=-1
    # V: b10 -> V[2,6]=1, b11 -> V[3,6]=1
    # W: contributes to C00 (k=0,i=0 -> W[0,6]), C11 (k=1,i=1 -> W[3,6])
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1  # C00 += M7, C11 += M7
    
    return U, V, W


def _build_strassen_block(deadline=None):
    """
    Build a rank-33 decomposition for 2x3x6 using Strassen on 2x2 sub-blocks.
    
    Structure: A is 2x3, B is 3x6, C is 2x6.
    Split B into three 3x2 blocks: B = [B0 | B1 | B2]
    Split A into A = [A0 | A1] where A0 is 2x2, A1 is 2x1.
    
    For each block j in {0,1,2}:
      C_j = A * B_j = A0 * B0_j + A1 * B1_j
      where B0_j is the top 2x2 of B_j, B1_j is the bottom 1x2 of B_j.
    
    A0 * B0_j: 2x2 * 2x2 -> use Strassen (rank 7)
    A1 * B1_j: 2x1 * 1x2 -> rank 4 (outer product)
    
    Total per block: 7 + 4 = 11
    Total: 3 * 11 = 33
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    if deadline is not None and time.time() > deadline:
        return None
    
    R = 33
    U = np.zeros((n * m, R))  # (6, 33)
    V = np.zeros((m * p, R))  # (18, 33)
    W = np.zeros((p * n, R))  # (12, 33)
    
    # Get Strassen 2x2x2 decomposition
    U_s, V_s, W_s = _strassen_2x2x2()
    
    col = 0
    for block in range(3):
        # --- Strassen part: A0(2x2) * B0_block(2x2) ---
        # A0 has elements: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices 0,1,2,3
        # B0_block has elements: B[0,block*2], B[0,block*2+1], B[1,block*2], B[1,block*2+1]
        #   -> V indices: 0, 1, 6+block*2, 7+block*2  (since V[j*p+k] with j=0,1 and k=block*2, block*2+1)
        
        for r in range(7):
            # U: copy Strassen U pattern (indices 0-3 correspond to A[0,0],A[0,1],A[1,0],A[1,1])
            U[0, col] = U_s[0, r]  # A[0,0]
            U[1, col] = U_s[1, r]  # A[0,1]
            U[2, col] = U_s[2, r]  # A[1,0]
            U[3, col] = U_s[3, r]  # A[1,1]
            
            # V: map Strassen V indices (0-3 for B[0,0],B[0,1],B[1,0],B[1,1])
            # to actual V indices for this block
            # Strassen V index 0 -> B[0,0] of block -> V[0*6 + block*2] = V[block*2]
            # Strassen V index 1 -> B[0,1] of block -> V[0*6 + block*2+1] = V[block*2+1]
            # Strassen V index 2 -> B[1,0] of block -> V[1*6 + block*2] = V[6+block*2]
            # Strassen V index 3 -> B[1,1] of block -> V[1*6 + block*2+1] = V[7+block*2]
            v_map = [block*2, block*2+1, 6+block*2, 7+block*2]
            for sv_idx in range(4):
                V[v_map[sv_idx], col] = V_s[sv_idx, r]
            
            # W: Strassen W indices (0-3 for C[0,0],C[0,1],C[1,0],C[1,1])
            # C[0,0] of block -> W[0*2+0] = W[0], but this is the (0, block*2) element of C
            # Actually C is 2x6, so C[i, block*2 + k]
            # W index in our convention: W[k*n + i] where k is column, i is row of C
            # For C[0, block*2]: k=block*2, i=0 -> W[block*2*2 + 0] = W[2*block*2] = W[4*block]
            # For C[0, block*2+1]: k=block*2+1, i=0 -> W[(block*2+1)*2] = W[4*block+2]
            # For C[1, block*2]: k=block*2, i=1 -> W[4*block+1]
            # For C[1, block*2+1]: k=block*2+1, i=1 -> W[4*block+3]
            w_map = [4*block, 4*block+2, 4*block+1, 4*block+3]
            for sw_idx in range(4):
                W[w_map[sw_idx], col] = W_s[sw_idx, r]
            
            col += 1
        
        # --- Outer product part: A1(2x1) * B1_block(1x2) ---
        # A1 has elements: A[0,2], A[1,2] -> U indices 4, 5
        # B1_block has elements: B[2, block*2], B[2, block*2+1] -> V indices 12+block*2, 13+block*2
        # C contribution: C[0, block*2] += A[0,2]*B[2,block*2], C[0, block*2+1] += A[0,2]*B[2,block*2+1]
        #                  C[1, block*2] += A[1,2]*B[2,block*2], C[1, block*2+1] += A[1,2]*B[2,block*2+1]
        
        # Term 1: A[0,2] * B[2, block*2] -> C[0, block*2]
        U[4, col] = 1
        V[12 + block*2, col] = 1
        W[4*block, col] = 1  # C[0, block*2]
        col += 1
        
        # Term 2: A[0,2] * B[2, block*2+1] -> C[0, block*2+1]
        U[4, col] = 1
        V[13 + block*2, col] = 1
        W[4*block + 2, col] = 1  # C[0, block*2+1]
        col += 1
        
        # Term 3: A[1,2] * B[2, block*2] -> C[1, block*2]
        U[5, col] = 1
        V[12 + block*2, col] = 1
        W[4*block + 1, col] = 1  # C[1, block*2]
        col += 1
        
        # Term 4: A[1,2] * B[2, block*2+1] -> C[1, block*2+1]
        U[5, col] = 1
        V[13 + block*2, col] = 1
        W[4*block + 3, col] = 1  # C[1, block*2+1]
        col += 1
    
    # Verify
    if _verify_fast(U, V, W):
        return U, V, W
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Start with schoolbook
    U_s, V_s, W_s = _schoolbook()
    
    # Try Strassen block-structured approach (rank 33)
    result = _build_strassen_block(deadline)
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return schoolbook if no better found
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

# EVOLVE-BLOCK-END
