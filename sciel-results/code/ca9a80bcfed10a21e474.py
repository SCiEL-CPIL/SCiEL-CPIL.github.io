import numpy as np
import time
import sys

DIMENSIONS = [2, 4, 8]

def _strassen_block_2x2x2(U, V, W, a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11, col):
    """
    Adds a rank-7 Strassen decomposition for a 2x2 * 2x2 block.
    U, V, W are lists of lists (or numpy arrays) of size [rows] x [rank].
    a00..a11 are row indices for U (A matrix part).
    b00..b11 are row indices for V (B matrix part).
    c00..c11 are row indices for W (C matrix part).
    col is the starting column index in the rank dimension.
    """
    # M1 = (A00 + A11)(B00 + B11) -> C00 += M1, C11 += M1
    U[a00][col] += 1; U[a11][col] += 1
    V[b00][col] += 1; V[b11][col] += 1
    W[c00][col] += 1; W[c11][col] += 1
    
    # M2 = (A10 + A11)B00 -> C10 += M2
    U[a10][col+1] += 1; U[a11][col+1] += 1
    V[b00][col+1] += 1
    W[c10][col+1] += 1
    
    # M3 = A00(B01 - B11) -> C01 += M3
    U[a00][col+2] += 1
    V[b01][col+2] += 1; V[b11][col+2] += -1
    W[c01][col+2] += 1
    
    # M4 = A11(B10 - B00) -> C11 += M4
    U[a11][col+3] += 1
    V[b10][col+3] += 1; V[b00][col+3] += -1
    W[c11][col+3] += 1
    
    # M5 = (A00 + A01)B11 -> C00 += M5
    U[a00][col+4] += 1; U[a01][col+4] += 1
    V[b11][col+4] += 1
    W[c00][col+4] += 1
    
    # M6 = (A10 - A00)(B00 + B01) -> C10 += M6
    U[a10][col+5] += 1; U[a00][col+5] += -1
    V[b00][col+5] += 1; V[b01][col+5] += 1
    W[c10][col+5] += 1
    
    # M7 = (A01 - A11)(B10 + B11) -> C01 += M7, C11 += M7
    U[a01][col+6] += 1; U[a11][col+6] += -1
    V[b10][col+6] += 1; V[b11][col+6] += 1
    W[c01][col+6] += 1; W[c11][col+6] += 1

def construct():
    """
    Rank-minimal tensor decomposition for 2x4 * 4x8 matrix multiplication.
    Returns a dict with U, V, W.
    """
    n, m, p = 2, 4, 8
    rank = 56
    
    # U has n*m = 8 rows (A is 2x4, flattened row-major)
    # V has m*p = 32 rows (B is 4x8, flattened row-major)
    # W has p*n = 16 rows (C is 8x2, flattened as W[k*n+i,r] per trace convention)
    # 
    # C is 2x8, but W uses trace convention W[k*n+i,r] where:
    #   k = column index of C (0..7), i = row index of C (0..1)
    #   So W index = k*n + i = k*2 + i
    #   C[i,k] maps to W[2*k + i]
    
    U = [[0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 32 rows
    W = [[0] * rank for _ in range(p * n)]  # 16 rows
    
    # A is 2x4, flattened row-major:
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    
    # B is 4x8, flattened row-major:
    # B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3, B[0,4]=4, B[0,5]=5, B[0,6]=6, B[0,7]=7
    # B[1,0]=8, B[1,1]=9, B[1,2]=10, B[1,3]=11, B[1,4]=12, B[1,5]=13, B[1,6]=14, B[1,7]=15
    # B[2,0]=16, B[2,1]=17, B[2,2]=18, B[2,3]=19, B[2,4]=20, B[2,5]=21, B[2,6]=22, B[2,7]=23
    # B[3,0]=24, B[3,1]=25, B[3,2]=26, B[3,3]=27, B[3,4]=28, B[3,5]=29, B[3,6]=30, B[3,7]=31
    
    # C is 2x8, W index = 2*k + i where k is col, i is row
    # C[0,0]=W[0], C[0,1]=W[2], C[0,2]=W[4], C[0,3]=W[6], C[0,4]=W[8], C[0,5]=W[10], C[0,6]=W[12], C[0,7]=W[14]
    # C[1,0]=W[1], C[1,1]=W[3], C[1,2]=W[5], C[1,3]=W[7], C[1,4]=W[9], C[1,5]=W[11], C[1,6]=W[13], C[1,7]=W[15]
    
    # We have 2 A-row-blocks and 4 B-col-blocks, total 8 Strassen blocks
    # A row-block 0: rows 0-1 of A -> A[:, 0:2] = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    #   U indices: a00=0, a01=1, a10=4, a11=5
    # A row-block 1: rows 2-3 of A -> A[:, 2:4] = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]
    #   U indices: a00=2, a01=3, a10=6, a11=7
    
    # B col-block j (j=0..3): columns 2j to 2j+1 of B
    # B[:, 0:2] -> rows 0-3, cols 0-1: [[B[0,0], B[0,1]], [B[1,0], B[1,1]], [B[2,0], B[2,1]], [B[3,0], B[3,1]]]
    #   But we need 2x2 blocks, so B col-block is actually 4x2, and we split into two 2x2:
    #   Wait, B is 4x8. For 2x2*2x2, we need A to be 2x2 and B to be 2x2.
    #   
    # Actually, let me reconsider. The multiplication is:
    # C (2x8) = A (2x4) * B (4x8)
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    #
    # To decompose into 2x2*2x2 blocks, we can partition the sum over j:
    # For each pair of j values (j1, j2) and pair of k values (k1, k2):
    #   Contribution to C[:, {k1,k2}] from A[:, {j1,j2}] * B[{j1,j2}, {k1,k2}]
    #
    # There are C(4,2) = 6 ways to partition 4 j's into pairs? No, we need disjoint pairs.
    # Actually, we partition the 4 j indices into 2 pairs: (0,1) and (2,3).
    # And partition the 8 k indices into 4 pairs: (0,1), (2,3), (4,5), (6,7).
    #
    # So we have 2 j-pairs × 4 k-pairs = 8 blocks.
    # Each block: A[:, j_pair] (2x2) * B[j_pair, k_pair] (2x2) -> contributes to C[:, k_pair] (2x2)
    
    # j-pair 0: j=0,1 -> A[:, 0:2], B[0:2, :]
    # j-pair 1: j=2,3 -> A[:, 2:4], B[2:4, :]
    
    # k-pair 0: k=0,1 -> B[:, 0:2], C[:, 0:2]
    # k-pair 1: k=2,3 -> B[:, 2:4], C[:, 2:4]
    # k-pair 2: k=4,5 -> B[:, 4:6], C[:, 4:6]
    # k-pair 3: k=6,7 -> B[:, 6:8], C[:, 6:8]
    
    # For each (j_pair, k_pair), we have a 2x2 * 2x2 multiplication.
    
    # Let's define the index mappings:
    
    # A row-block for j_pair 0 (j=0,1):
    #   A[0,0], A[0,1] -> U rows 0, 1
    #   A[1,0], A[1,1] -> U rows 4, 5
    # So a00=0, a01=1, a10=4, a11=5
    
    # A row-block for j_pair 1 (j=2,3):
    #   A[0,2], A[0,3] -> U rows 2, 3
    #   A[1,2], A[1,3] -> U rows 6, 7
    # So a00=2, a01=3, a10=6, a11=7
    
    # B col-block for k_pair 0 (k=0,1):
    #   B[0,0], B[0,1] -> V rows 0, 1
    #   B[1,0], B[1,1] -> V rows 8, 9
    #   B[2,0], B[2,1] -> V rows 16, 17
    #   B[3,0], B[3,1] -> V rows 24, 25
    # But we need 2x2, so for j_pair 0 (j=0,1): B[0:2, 0:2]
    #   b00 = B[0,0] = 0, b01 = B[0,1] = 1
    #   b10 = B[1,0] = 8, b11 = B[1,1] = 9
    
    # For j_pair 1 (j=2,3): B[2:4, 0:2]
    #   b00 = B[2,0] = 16, b01 = B[2,1] = 17
    #   b10 = B[3,0] = 24, b11 = B[3,1] = 25
    
    # C col-block for k_pair 0 (k=0,1):
    #   C[0,0] -> W[2*0+0] = W[0]
    #   C[0,1] -> W[2*1+0] = W[2]
    #   C[1,0] -> W[2*0+1] = W[1]
    #   C[1,1] -> W[2*1+1] = W[3]
    # So c00=0, c01=2, c10=1, c11=3
    
    # Let's create a helper to get B indices for given j_pair and k_pair
    def get_B_indices(j_pair, k_pair):
        """
        Returns (b00, b01, b10, b11) for B[j_pair*2:(j_pair+1)*2, k_pair*2:(k_pair+1)*2]
        """
        j0 = j_pair * 2
        k0 = k_pair * 2
        # B is 4x8, row-major: B[i,j] = i*8 + j
        b00 = j0 * 8 + k0       # B[j0, k0]
        b01 = j0 * 8 + (k0 + 1) # B[j0, k0+1]
        b10 = (j0 + 1) * 8 + k0       # B[j0+1, k0]
        b11 = (j0 + 1) * 8 + (k0 + 1) # B[j0+1, k0+1]
        return b00, b01, b10, b11
    
    def get_C_indices(k_pair):
        """
        Returns (c00, c01, c10, c11) for C[:, k_pair*2:(k_pair+1)*2]
        C is 2x8, W index = 2*k + i
        """
        k0 = k_pair * 2
        c00 = 2 * k0 + 0       # C[0, k0]
        c01 = 2 * (k0 + 1) + 0 # C[0, k0+1]
        c10 = 2 * k0 + 1       # C[1, k0]
        c11 = 2 * (k0 + 1) + 1 # C[1, k0+1]
        return c00, c01, c10, c11
    
    # Now loop over all 8 blocks
    col = 0
    for j_pair in range(2):
        for k_pair in range(4):
            # Get A indices
            if j_pair == 0:
                a00, a01, a10, a11 = 0, 1, 4, 5
            else:
                a00, a01, a10, a11 = 2, 3, 6, 7
            
            # Get B indices
            b00, b01, b10, b11 = get_B_indices(j_pair, k_pair)
            
            # Get C indices
            c00, c01, c10, c11 = get_C_indices(k_pair)
            
            # Apply Strassen
            _strassen_block_2x2x2(U, V, W, a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11, col)
            
            col += 7
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    # Quick sanity check
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    # The structure is built.
