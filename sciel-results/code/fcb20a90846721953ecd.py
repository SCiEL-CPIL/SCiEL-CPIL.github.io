"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2_2():
    """Return U, V, W for 2x2x2 Strassen decomposition, rank 7."""
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # c=0: P1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # c=1: P2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # c=2: P3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # c=3: P4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[1, 3] = 1; W[0, 3] = 1
    
    # c=4: P5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[2, 4] = 1; W[0, 4] = -1
    
    # c=5: P6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # c=6: P7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _decompose_4x4_4():
    """Return U, V, W for 4x4x4 decomposition using block Strassen, rank 49."""
    n = m = p = 4
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U2, V2, W2 = _strassen_2x2_2()
    
    # Block coefficients for each Strassen term
    # A_blocks[c_block] = list of (block_row, block_col, coeff)
    # B_blocks[c_block] = list of (block_row, block_col, coeff)
    # C_blocks[c_block] = list of (block_row, block_col, coeff)
    
    A_blocks = [
        [(0,0,1), (1,1,1)],      # P1
        [(1,0,1), (1,1,1)],      # P2
        [(0,0,1)],                # P3
        [(1,1,1)],                # P4
        [(0,0,1), (0,1,1)],      # P5
        [(1,0,1), (0,0,-1)],     # P6
        [(0,1,1), (1,1,-1)],     # P7
    ]
    B_blocks = [
        [(0,0,1), (1,1,1)],      # P1
        [(0,0,1)],                # P2
        [(0,1,1), (1,1,-1)],     # P3
        [(1,0,1), (0,0,-1)],     # P4
        [(1,1,1)],                # P5
        [(0,0,1), (0,1,1)],      # P6
        [(1,0,1), (1,1,1)],      # P7
    ]
    C_blocks = [
        [(0,0,1), (1,1,1)],      # P1
        [(1,0,1), (1,1,-1)],     # P2
        [(0,1,1), (1,1,1)],      # P3
        [(1,0,1), (0,0,1)],      # P4
        [(0,1,1), (0,0,-1)],     # P5
        [(1,1,1)],                # P6
        [(0,0,1)],                # P7
    ]
    
    for cb in range(7):
        for cs in range(7):
            c = cb * 7 + cs
            for (ar, ac, a_coeff) in A_blocks[cb]:
                for (br, bc, b_coeff) in B_blocks[cb]:
                    for (cr, cc, c_coeff) in C_blocks[cb]:
                        # Map block indices to element indices
                        # A element: (2*ar + i', 2*ac + j') for i',j' in {0,1}
                        # U index: (2*ar + i')*4 + (2*ac + j')
                        # Similarly for V and W
                        for i_prime in range(2):
                            for j_prime in range(2):
                                u_idx = (2*ar + i_prime)*4 + (2*ac + j_prime)
                                U[u_idx, c] += a_coeff * U2[i_prime*2 + j_prime, cs]
                        for j_prime in range(2):
                            for k_prime in range(2):
                                v_idx = (2*br + j_prime)*4 + (2*bc + k_prime)
                                V[v_idx, c] += b_coeff * V2[j_prime*2 + k_prime, cs]
                        for k_prime in range(2):
                            for i_prime in range(2):
                                w_idx = (2*cr + k_prime)*4 + (2*cc + i_prime)
                                W[w_idx, c] += c_coeff * W2[k_prime*2 + i_prime, cs]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    rank = 114
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Get 4x4x4 decomposition
    U4, V4, W4 = _decompose_4x4_4()
    
    # First 4 columns: use 4x4x4 decomposition
    U[:, 0:49] = U4
    V[:, 0:49] = V4[:, 0:16]  # V4 has shape (16, 49), we need V for 4x4 part
    # Wait, V4 has shape (m*p, rank) = (16, 49) for 4x4x4
    # For the first 4 columns of B (4x9), we use B[:, 0:4]
    # V[j*p+k, c] where k is in [0,4) for the first block
    # So V[:, 0:49] should be V4[:, 0:49] but V4 is for p=4
    # V4[j*4+k, c] for k in [0,4)
    # But in the 4x4x9 case, V[j*9+k, c] for k in [0,9)
    # So I need to map V4[j*4+k, c] to V[j*9+k, c] for k in [0,4)
    
    # Let me redo this more carefully
    
    # For the 4x4x9 case:
    # U[i*4+j, c] for i in [0,4), j in [0,4)
    # V[j*9+k, c] for j in [0,4), k in [0,9)
    # W[k*4+i, c] for k in [0,9), i in [0,4)
    
    # For the first 4 columns (k in [0,4)):
    # U[i*4+j, c] = U4[i*4+j, c] for c in [0,49)
    # V[j*9+k, c] = V4[j*4+k, c] for c in [0,49), k in [0,4)
    # W[k*4+i, c] = W4[k*4+i, c] for c in [0,49), k in [0,4)
    
    # For the next 4 columns (k in [4,8)):
    # U[i*4+j, c] = U4[i*4+j, c-49] for c in [49,98)
    # V[j*9+k, c] = V4[j*4+(k-4), c-49] for c in [49,98), k in [4,8)
    # W[k*4+i, c] = W4[(k-4)*4+i, c-49] for c in [49,98), k in [4,8)
    
    # For the last column (k=8):
    # Schoolbook: rank 16
    # U[i*4+j, c] = 1 if i*4+j == c-98
    # V[j*9+8, c] = 1 if j*9+8 == c-98
    # W[8*4+i, c] = 1 if 8*4+i == c-98
    
    # First block (k in [0,4))
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*9+k, 0:49] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W4[k*4+i, 0:49]
    
    # Second block (k in [4,8))
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4, 8):
            V[j*9+k, 49:98] = V4[j*4+(k-4), 0:49]
    for k in range(4, 8):
        for i in range(4):
            W[k*4+i, 49:98] = W4[(k-4)*4+i, 0:49]
    
    # Third block (k=8): schoolbook
    for idx in range(16):
        c = 98 + idx
        i = idx // 4
        j = idx % 4
        U[i*4+j, c] = 1
        V[j*9+8, c] = 1
        W[8*4+i, c] = 1
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
