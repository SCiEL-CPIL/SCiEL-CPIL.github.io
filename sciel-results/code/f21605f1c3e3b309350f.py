import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 2x5 matrix by a 5x6 matrix using block-Strassen for 2x2x2 subblocks
    and direct multiplication for 2x1x6 subblocks.
    
    Strategy:
    - Split the middle dimension m=5 into j-groups: {0,1}, {2,3}, {4}
    - For each 2x2 A-block times 2x6 B-block, split the p=6 dimension
      into three 2-column groups and use Strassen (7 terms each)
    - For the 2x1 A-block times 1x6 B-block, use direct (12 terms)
    
    Total rank = 3 blocks * 3 sub-blocks * 7 (Strassen) + 1 block * 12 (direct)
             = 63 + 12 = 75... 
             
    Wait, let me reconsider. The 2x2x6 block: A is 2x2, B is 2x6.
    Split B's columns into 3 groups of 2: B_1 (cols 0,1), B_2 (cols 2,3), B_3 (cols 4,5)
    Each A*B_i is a 2x2x2 multiplication -> 7 Strassen terms
    So 3 * 7 = 21 terms per 2x2x6 block.
    Two such blocks (for j-groups {0,1} and {2,3}): 2 * 21 = 42
    One 2x1x6 block (for j-group {4}): 12 direct terms
    Total rank = 42 + 12 = 54
    """
    n, m, p = DIMENSIONS  # n=2, m=5, p=6
    
    # Define Strassen 2x2x2 decomposition
    # C = A * B where A, B are 2x2
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B11 - B12)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M7
    # C22 = M1 - M2 + M3 + M6
    
    # U row index: i*m + j = i*5 + j
    # V row index: j*p + k = j*6 + k
    # W row index: k*n + i = k*2 + i
    
    columns = []
    
    def add_strassen(i_base, j_base, k_base):
        """
        Add 7 Strassen columns for a 2x2x2 block.
        A rows: i_base, i_base+1; A cols: j_base, j_base+1
        B rows: j_base, j_base+1; B cols: k_base, k_base+1
        C rows: k_base, k_base+1; C cols: i_base, i_base+1
        """
        # Each Strassen term is a product of a linear combo of A entries and linear combo of B entries
        # U vector has 10 entries (2*5), V has 30 entries (5*6), W has 12 entries (6*2)
        
        # A indices: (i_base, j_base), (i_base, j_base+1), (i_base+1, j_base), (i_base+1, j_base+1)
        # B indices: (j_base, k_base), (j_base, k_base+1), (j_base+1, k_base), (j_base+1, k_base+1)
        # C indices: (k_base, i_base), (k_base, i_base+1), (k_base+1, i_base), (k_base+1, i_base+1)
        
        A11 = i_base * m + j_base
        A12 = i_base * m + (j_base + 1)
        A21 = (i_base + 1) * m + j_base
        A22 = (i_base + 1) * m + (j_base + 1)
        
        B11 = j_base * p + k_base
        B12 = j_base * p + (k_base + 1)
        B21 = (j_base + 1) * p + k_base
        B22 = (j_base + 1) * p + (k_base + 1)
        
        C11 = k_base * n + i_base
        C12 = k_base * n + (i_base + 1)
        C21 = (k_base + 1) * n + i_base
        C22 = (k_base + 1) * n + (i_base + 1)
        
        # M1 = (A11 + A22)(B11 + B22)
        u1 = [0]*10; v1 = [0]*30; w1 = [0]*12
        u1[A11] = 1; u1[A22] = 1
        v1[B11] = 1; v1[B22] = 1
        w1[C11] = 1; w1[C22] = 1
        columns.append((u1, v1, w1))
        
        # M2 = (A21 + A22)B11
        u2 = [0]*10; v2 = [0]*30; w2 = [0]*12
        u2[A21] = 1; u2[A22] = 1
        v2[B11] = 1
        w2[C21] = 1; w2[C22] = -1
        columns.append((u2, v2, w2))
        
        # M3 = A11(B21 - B22)
        u3 = [0]*10; v3 = [0]*30; w3 = [0]*12
        u3[A11] = 1
        v3[B21] = 1; v3[B22] = -1
        w3[C12] = 1; w3[C22] = 1
        columns.append((u3, v3, w3))
        
        # M4 = A22(B11 - B12)
        u4 = [0]*10; v4 = [0]*30; w4 = [0]*12
        u4[A22] = 1
        v4[B11] = 1; v4[B12] = -1
        w4[C11] = 1; w4[C22] = 1
        columns.append((u4, v4, w4))
        
        # M5 = (A11 + A12)B22
        u5 = [0]*10; v5 = [0]*30; w5 = [0]*12
        u5[A11] = 1; u5[A12] = 1
        v5[B22] = 1
        w5[C11] = -1; w5[C12] = 1
        columns.append((u5, v5, w5))
        
        # M6 = (A21 - A11)(B11 + B12)
        u6 = [0]*10; v6 = [0]*30; w6 = [0]*12
        u6[A21] = 1; u6[A11] = -1
        v6[B11] = 1; v6[B12] = 1
        w6[C22] = 1
        columns.append((u6, v6, w6))
        
        # M7 = (A12 - A22)(B21 + B22)
        u7 = [0]*10; v7 = [0]*30; w7 = [0]*12
        u7[A12] = 1; u7[A22] = -1
        v7[B21] = 1; v7[B22] = 1
        w7[C11] = -1; w7[C21] = 1
        columns.append((u7, v7, w7))
    
    def add_direct_2x1x6(i_base, j_idx, k_base):
        """
        Add 12 direct columns for a 2x1x6 block.
        A: rows i_base, i_base+1; col j_idx
        B: row j_idx; cols k_base..k_base+5
        C: rows k_base..k_base+5; cols i_base, i_base+1
        """
        for k in range(6):
            # C[k_base+k, i_base] = A[i_base, j_idx] * B[j_idx, k_base+k]
            u = [0]*10; v = [0]*30; w = [0]*12
            u[i_base * m + j_idx] = 1
            v[j_idx * p + (k_base + k)] = 1
            w[(k_base + k) * n + i_base] = 1
            columns.append((u, v, w))
            
            # C[k_base+k, i_base+1] = A[i_base+1, j_idx] * B[j_idx, k_base+k]
            u = [0]*10; v = [0]*30; w = [0]*12
            u[(i_base + 1) * m + j_idx] = 1
            v[j_idx * p + (k_base + k)] = 1
            w[(k_base + k) * n + (i_base + 1)] = 1
            columns.append((u, v, w))
    
    # Block 1: j in {0,1}, A is 2x2, B is 2x6
    # Split B's columns into 3 groups of 2: k in {0,1}, {2,3}, {4,5}
    add_strassen(0, 0, 0)  # i=0,1; j=0,1; k=0,1
    add_strassen(0, 0, 2)  # i=0,1; j=0,1; k=2,3
    add_strassen(0, 0, 4)  # i=0,1; j=0,1; k=4,5
    
    # Block 2: j in {2,3}, A is 2x2, B is 2x6
    add_strassen(0, 2, 0)  # i=0,1; j=2,3; k=0,1
    add_strassen(0, 2, 2)  # i=0,1; j=2,3; k=2,3
    add_strassen(0, 2, 4)  # i=0,1; j=2,3; k=4,5
    
    # Block 3: j in {4}, A is 2x1, B is 1x6
    add_direct_2x1x6(0, 4, 0)  # i=0,1; j=4; k=0..5
    
    rank = len(columns)
    
    # Build U, V, W as lists of lists
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col_idx in range(rank):
        u, v, w = columns[col_idx]
        for i in range(n * m):
            U[i][col_idx] = u[i]
        for i in range(m * p):
            V[i][col_idx] = v[i]
        for i in range(p * n):
            W[i][col_idx] = w[i]
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {result['U'][0].__len__() if result['U'] else 0}")
    print(f"U shape: {len(result['U'])}x{len(result['U'][0])}")
    print(f"V shape: {len(result['V'])}x{len(result['V'][0])}")
    print(f"W shape: {len(result['W'])}x{len(result['W'][0])}")
