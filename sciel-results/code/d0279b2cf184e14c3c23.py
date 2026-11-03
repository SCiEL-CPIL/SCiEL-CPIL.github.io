import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def _strassen_2x2_coeffs():
    """
    Returns U, V, W for 2x2 Strassen decomposition.
    Shapes: U (4, 7), V (4, 7), W (4, 7)
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11 + A22)(B11 + B22)
    # A11 is (0,0), A22 is (1,1)
    # U indices: i*2+j
    # (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # C11 += M1, C22 += M1
    # C11 is (0,0) -> W[0*2+0]=W[0]
    # C22 is (1,1) -> W[1*2+1]=W[3]
    W[0, 0] = 1; W[3, 0] = 1

    # M2 = (A21 + A22)B11
    # A21 is (1,0)->2, A22 is (1,1)->3
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C21 += M2, C22 -= M2
    # C21 is (1,0) -> W[0*2+1]=W[2]
    # C22 is (1,1) -> W[3]
    W[2, 1] = 1; W[3, 1] = -1

    # M3 = A11(B12 - B22)
    # A11 is (0,0)->0
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1 # B12 is (0,1)->1, B22 is (1,1)->3
    # C12 += M3, C22 += M3
    # C12 is (0,1) -> W[1*2+0]=W[1]
    # C22 is (1,1) -> W[3]
    W[1, 2] = 1; W[3, 2] = 1

    # M4 = A22(B21 - B11)
    # A22 is (1,1)->3
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1 # B21 is (1,0)->2, B11 is (0,0)->0
    # C11 += M4, C21 += M4
    # C11 is (0,0) -> W[0]
    # C21 is (1,0) -> W[2]
    W[0, 3] = 1; W[2, 3] = 1

    # M5 = (A11 + A12)B22
    # A11 is (0,0)->0, A12 is (0,1)->1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C11 -= M5, C12 += M5
    # C11 is (0,0) -> W[0]
    # C12 is (0,1) -> W[1]
    W[0, 4] = -1; W[1, 4] = 1

    # M6 = (A21 - A11)(B11 + B12)
    # A21 is (1,0)->2, A11 is (0,0)->0
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C22 += M6
    # C22 is (1,1) -> W[3]
    W[3, 5] = 1

    # M7 = (A12 - A22)(B21 + B22)
    # A12 is (0,1)->1, A22 is (1,1)->3
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C11 += M7
    # C11 is (0,0) -> W[0]
    W[0, 6] = 1

    return U, V, W

def _construct_4x4_strassen():
    """
    Constructs U, V, W for 4x4 matrix multiplication using recursive Strassen.
    Returns U (16, 49), V (16, 49), W (16, 49)
    Indices for 4x4: i*4+j
    """
    U_2, V_2, W_2 = _strassen_2x2_coeffs()
    
    U_4 = np.zeros((16, 49))
    V_4 = np.zeros((16, 49))
    W_4 = np.zeros((16, 49))
    
    # Outer Strassen steps
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x2)
    # C blocks: C00, C01, C10, C11 (each 2x2)
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We map outer index k (0..6) to inner terms.
    # For each k, we have linear combinations for X_k (from A) and Y_k (from B).
    # X_k and Y_k are 2x2 matrices.
    # We compute X_k Y_k using U_2, V_2, W_2.
    
    # Define the linear maps for X_k and Y_k
    # A entries are 16. We need to extract the 2x2 blocks.
    # Block (bi, bj) has entries A[bi*4 + i, bj*4 + j] for i,j in 0,1.
    
    # For each outer term k:
    # x_coeffs: 4x16 matrix (2x2 X_k from 16 A entries)
    # y_coeffs: 4x16 matrix (2x2 Y_k from 16 B entries)
    # c_coeffs: 4x16 matrix (2x2 Z_k contribution to 16 C entries)
    
    # Actually, it's easier to just iterate.
    
    # Helper to get block indices
    def get_block_indices(bi, bj):
        # Returns indices for 2x2 block at (bi, bj) in 4x4 matrix
        # i in 0..1, j in 0..1
        # Global index = (bi*2 + i)*4 + (bj*2 + j)
        idxs = []
        for i in range(2):
            for j in range(2):
                idxs.append((bi*2 + i)*4 + (bj*2 + j))
        return idxs

    # Outer terms definition:
    # k, x_blocks (list of (bi, bj, coeff)), y_blocks (list of (bi, bj, coeff)), c_blocks (list of (ci, cj, coeff))
    outer_terms = [
        # M1
        { 'x': [(0,0,1), (1,1,1)], 'y': [(0,0,1), (1,1,1)], 'c': [(0,0,1), (1,1,1)] },
        # M2
        { 'x': [(1,0,1), (1,1,1)], 'y': [(0,0,1)], 'c': [(1,0,1), (1,1,-1)] },
        # M3
        { 'x': [(0,0,1)], 'y': [(0,1,1), (1,1,-1)], 'c': [(0,1,1), (1,1,1)] },
        # M4
        { 'x': [(1,1,1)], 'y': [(1,0,1), (0,0,-1)], 'c': [(0,0,1), (1,0,1)] },
        # M5
        { 'x': [(0,0,1), (0,1,1)], 'y': [(1,1,1)], 'c': [(0,0,-1), (0,1,1)] },
        # M6
        { 'x': [(1,0,1), (0,0,-1)], 'y': [(0,0,1), (0,1,1)], 'c': [(1,1,1)] },
        # M7
        { 'x': [(0,1,1), (1,1,-1)], 'y': [(1,0,1), (1,1,1)], 'c': [(0,0,1)] },
    ]
    
    rank = 0
    for k in range(7):
        term = outer_terms[k]
        # Construct x_coeffs (4x16) and y_coeffs (4x16)
        # x_coeffs[i*2+j, global_idx]
        x_coeffs = np.zeros((4, 16))
        for bi, bj, coeff in term['x']:
            idxs = get_block_indices(bi, bj)
            for local_idx, global_idx in enumerate(idxs):
                x_coeffs[local_idx, global_idx] = coeff
                
        y_coeffs = np.zeros((4, 16))
        for bi, bj, coeff in term['y']:
            idxs = get_block_indices(bi, bj)
            for local_idx, global_idx in enumerate(idxs):
                y_coeffs[local_idx, global_idx] = coeff

        # c_coeffs (4x16)
        c_coeffs = np.zeros((4, 16))
        for ci, cj, coeff in term['c']:
            idxs = get_block_indices(ci, cj)
            for local_idx, global_idx in enumerate(idxs):
                c_coeffs[local_idx, global_idx] = coeff

        # Now combine with inner 2x2 Strassen
        # U_4[:, rank:rank+7] = x_coeffs @ U_2
        # V_4[:, rank:rank+7] = y_coeffs @ V_2
        # W_4[:, rank:rank+7] = c_coeffs @ W_2
        
        U_4[:, rank:rank+7] = x_coeffs @ U_2
        V_4[:, rank:rank+7] = y_coeffs @ V_2
        W_4[:, rank:rank+7] = c_coeffs @ W_2
        
        rank += 7
        
    return U_4, V_4, W_4

def construct():
    n, m, p = DIMENSIONS
    # Total rank estimate: 49 + 61 = 110
    rank = 110
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # 1. 4x4 block (indices 0..3)
    U_4, V_4, W_4 = _construct_4x4_strassen()
    # Map 4x4 indices to 5x5 indices
    # 4x4 index: i*4+j -> 5x5 index: i*5+j
    # Since i,j < 4, i*4+j != i*5+j generally.
    # We need to permute the rows/cols of U_4, V_4, W_4.
    
    # U_4 rows are i*4+j. We want i*5+j.
    # Create a mapping
    def map_4_to_5_U(i4, j4):
        # i4, j4 are 0..3
        return i4 * 5 + j4
        
    def map_4_to_5_V(j4, k4):
        return j4 * 5 + k4
        
    def map_4_to_5_W(k4, i4):
        return k4 * 5 + i4

    # Fill U, V, W for the first 49 columns
    for c in range(49):
        for r4 in range(16):
            val = U_4[r4, c]
            if val != 0:
                # Decode r4
                i4 = r4 // 4
                j4 = r4 % 4
                r5 = map_4_to_5_U(i4, j4)
                U[r5, c] = val
                
        for c4 in range(16):
            val = V_4[c4, c]
            if val != 0:
                j4 = c4 // 4
                k4 = c4 % 4
                c5 = map_4_to_5_V(j4, k4)
                V[c5, c] = val
                
        for r4 in range(16):
            val = W_4[r4, c]
            if val != 0:
                k4 = r4 // 4
                i4 = r4 % 4
                r5 = map_4_to_5_W(k4, i4)
                W[r5, c] = val

    # 2. Interactions with index 4
    current_rank = 49
    
    # i in 0..3, j in 0..3, k=4
    # C[i,4] += A[i,j] * B[j,4]
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, current_rank] = 1
            V[j * 5 + 4, current_rank] = 1
            W[4 * 5 + i, current_rank] = 1
            current_rank += 1
            
    # i in 0..3, j=4, k in 0..3
    # C[i,k] += A[i,4] * B[4,k]
    for i in range(4):
        for k in range(4):
            U[i * 5 + 4, current_rank] = 1
            V[4 * 5 + k, current_rank] = 1
            W[k * 5 + i, current_rank] = 1
            current_rank += 1
            
    # i=4, j in 0..3, k in 0..3
    # C[4,k] += A[4,j] * B[j,k]
    for j in range(4):
        for k in range(4):
            U[4 * 5 + j, current_rank] = 1
            V[j * 5 + k, current_rank] = 1
            W[k * 5 + 4, current_rank] = 1
            current_rank += 1
            
    # i=4, j=4, k in 0..3
    # C[4,k] += A[4,4] * B[4,k]
    for k in range(4):
        U[4 * 5 + 4, current_rank] = 1
        V[4 * 5 + k, current_rank] = 1
        W[k * 5 + 4, current_rank] = 1
        current_rank += 1
        
    # i in 0..3, j=4, k=4
    # C[i,4] += A[i,4] * B[4,4]
    for i in range(4):
        U[i * 5 + 4, current_rank] = 1
        V[4 * 5 + 4, current_rank] = 1
        W[4 * 5 + i, current_rank] = 1
        current_rank += 1
        
    # i=4, j in 0..3, k=4
    # C[4,4] += A[4,j] * B[j,4]
    for j in range(4):
        U[4 * 5 + j, current_rank] = 1
        V[j * 5 + 4, current_rank] = 1
        W[4 * 5 + 4, current_rank] = 1
        current_rank += 1
        
    # i=4, j=4, k=4
    # C[4,4] += A[4,4] * B[4,4]
    U[4 * 5 + 4, current_rank] = 1
    V[4 * 5 + 4, current_rank] = 1
    W[4 * 5 + 4, current_rank] = 1
    current_rank += 1
    
    # current_rank should be 110
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", res["U"].shape[1])
