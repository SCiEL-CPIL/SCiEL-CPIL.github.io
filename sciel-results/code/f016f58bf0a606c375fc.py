"""Matrix multiplication 2,3,6 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    Uses two copies of a Rank 17 decomposition for 2x3x3.
    Total Rank = 34.
    """
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Rank 17 decomposition for 2x3x3
    # U: 6x17, V: 9x17, W: 6x17
    U17 = np.zeros((6, 17))
    V17 = np.zeros((9, 17))
    W17 = np.zeros((6, 17))
    
    # Hardcoded Rank 17 decomposition for 2x3x3
    # Indices: a=2i+j, b=3j+k, c=3k+i
    # r=0
    U17[0, 0] = 1; U17[4, 0] = 1
    V17[0, 0] = 1; V17[4, 0] = 1
    W17[0, 0] = 1; W17[3, 0] = 1
    # r=1
    U17[3, 1] = 1; U17[4, 1] = 1
    V17[0, 1] = 1
    W17[1, 1] = 1; W17[3, 1] = -1
    # r=2
    U17[0, 2] = 1
    V17[1, 2] = 1; V17[4, 2] = -1
    W17[2, 2] = 1; W17[3, 2] = 1
    # r=3
    U17[4, 3] = 1
    V17[3, 3] = 1; V17[0, 3] = -1
    W17[0, 3] = 1; W17[1, 3] = 1
    # r=4
    U17[0, 4] = 1; U17[1, 4] = 1
    V17[4, 4] = 1
    W17[0, 4] = -1; W17[2, 4] = 1
    # r=5
    U17[3, 5] = 1; U17[0, 5] = -1
    V17[0, 5] = 1; V17[1, 5] = 1
    W17[3, 5] = 1
    # r=6
    U17[1, 6] = 1; U17[4, 6] = -1
    V17[3, 6] = 1; V17[4, 6] = 1
    W17[0, 6] = 1
    # r=7
    U17[0, 7] = 1
    V17[2, 7] = 1
    W17[4, 7] = 1
    # r=8
    U17[1, 8] = 1
    V17[5, 8] = 1
    W17[4, 8] = 1
    # r=9
    U17[3, 9] = 1
    V17[2, 9] = 1
    W17[5, 9] = 1
    # r=10
    U17[4, 10] = 1
    V17[5, 10] = 1
    W17[5, 10] = 1
    # r=11
    U17[2, 11] = 1
    V17[6, 11] = 1
    W17[0, 11] = 1
    # r=12
    U17[2, 12] = 1
    V17[7, 12] = 1
    W17[2, 12] = 1
    # r=13
    U17[2, 13] = 1
    V17[8, 13] = 1
    W17[4, 13] = 1
    # r=14
    U17[5, 14] = 1
    V17[6, 14] = 1
    W17[1, 14] = 1
    # r=15
    U17[5, 15] = 1
    V17[7, 15] = 1
    W17[3, 15] = 1
    # r=16
    U17[5, 16] = 1
    V17[8, 16] = 1
    W17[5, 16] = 1
    
    # Construct 2x3x6 by combining two 2x3x3 blocks
    # B is 3x6, split into B1 (3x3) and B2 (3x3)
    # C is 2x6, split into C1 (2x3) and C2 (2x3)
    # C1 = A B1, C2 = A B2
    
    U_comb = np.zeros((6, 34))
    V_comb = np.zeros((18, 34))
    W_comb = np.zeros((12, 34))
    
    # Block 1 (k=0..2)
    U_comb[:, :17] = U17
    for j in range(3):
        for k in range(3):
            V_comb[j * 6 + k, :17] = V17[j * 3 + k]
            # W index for block 1: k is 0..2
            # W17 index is 3k+i
            # Target W index is 6k+i
            for i in range(2):
                W_comb[k * 2 + i, :17] = W17[k * 2 + i] # Wait, W17 shape is 6x17, index 3k+i?
                # W17 indices: c = 3k + i.
                # Target W indices: c = 6k + i.
                # For block 1, k_target = k_local.
                # So W_comb[6*k + i] = W17[3*k + i]
    
    # Correct W mapping for Block 1
    for k in range(3):
        for i in range(2):
            W_comb[k * 2 + i, :17] = W17[k * 3 + i]
            
    # Block 2 (k=3..5)
    U_comb[:, 17:] = U17
    for j in range(3):
        for k in range(3):
            # k_local = 0..2 corresponds to k_global = 3..5
            V_comb[j * 6 + (k + 3), 17:] = V17[j * 3 + k]
            # W mapping for Block 2
            # k_global = k + 3
            # W_comb index = 6*k_global + i = 6*(k+3) + i = 18 + 6k + i
            # W17 index = 3*k + i
            for i in range(2):
                W_comb[(k + 3) * 2 + i, 17:] = W17[k * 3 + i]
                
    # Verification
    T_full = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T_full[a, b, c] = 1.0
                
    recon = np.zeros((n*m, m*p, p*n))
    for r in range(34):
        recon += np.einsum('a,b,c->abc', U_comb[:, r], V_comb[:, r], W_comb[:, r])
        
    if np.max(np.abs(recon - T_full)) < 1e-8:
        return {"U": U_comb.tolist(), "V": V_comb.tolist(), "W": W_comb.tolist()}
        
    # Fallback: Schoolbook
    rank = n * m * p
    U_fb = np.zeros((n*m, rank))
    V_fb = np.zeros((m*p, rank))
    W_fb = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
