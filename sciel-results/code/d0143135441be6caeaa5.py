"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """
    Returns U, V, W for 2x2 matrix multiplication using Strassen's algorithm.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: 00, 01, 10, 11
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10 + A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = A00(B01 - B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4 = A11(B10 - B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = -1
    
    # M5 = (A00 + A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = -1; W[2][5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[3][6] = 1
    
    return U, V, W

def _strassen_4x4():
    """
    Returns U, V, W for 4x4 matrix multiplication using Strassen's algorithm.
    Composes 7 instances of 2x2 Strassen.
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    # 2x2 Strassen factors
    U2, V2, W2 = _strassen_2x2()
    
    # The 7 products M1..M7 in 4x4 Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # We need to map the 2x2 blocks of the 4x4 matrix to the 2x2 Strassen inputs.
    # A4x4 blocks: A11 (0-3), A12 (4-7), A21 (8-11), A22 (12-15)
    # B4x4 blocks: B11 (0-3), B12 (4-7), B21 (8-11), B22 (12-15)
    # C4x4 blocks: C11 (0-3), C12 (4-7), C21 (8-11), C22 (12-15)
    
    # For each of the 7 high-level Strassen terms, we invoke a 2x2 Strassen decomposition.
    # The inputs to the 2x2 Strassen are linear combinations of the 4x4 blocks.
    
    # Term 1: M1 = (A11 + A22)(B11 + B22)
    # Input A blocks: A11, A22. Input B blocks: B11, B22.
    # Output C blocks: C11, C12, C21, C22? No, M1 is a 2x2 matrix.
    # The result of M1 contributes to C11, C12, C21, C22?
    # In 4x4 Strassen, the 7 products are 2x2 matrices.
    # C11 = M1 - M4 + M6 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M7
    
    # So for each high-level term k (1..7), we have a 2x2 matrix product.
    # We decompose this 2x2 product using _strassen_2x2().
    # The U, V, W of the 2x2 product need to be mapped to the 4x4 indices.
    
    # Let's define the linear combinations for A and B for each term.
    # A_combinations: list of 7 lists of (block_index, coeff)
    # B_combinations: list of 7 lists of (block_index, coeff)
    # C_combinations: list of 7 lists of (block_index, coeff)
    
    # Note: The 2x2 Strassen decomposition produces a 2x2 result.
    # The "indices" of the 2x2 result correspond to the 4 2x2 blocks of the 4x4 result?
    # No, the 2x2 Strassen computes a single 2x2 matrix product.
    # The result is a 2x2 matrix.
    # In the context of 4x4 Strassen, each M_k is a 2x2 matrix.
    # The entries of M_k contribute to the entries of C.
    # C is 4x4. M_k is 2x2.
    # How do they map?
    # Actually, the 4x4 Strassen algorithm treats the 4x4 matrices as 2x2 blocks of 2x2 matrices.
    # So M1 is a 2x2 matrix.
    # C11 (2x2) = M1(2x2) - M4(2x2) + M6(2x2) - M7(2x2)
    # C12 (2x2) = M3(2x2) + M5(2x2)
    # C21 (2x2) = M2(2x2) + M4(2x2)
    # C22 (2x2) = M1(2x2) - M3(2x2) + M7(2x2)
    
    # So, for each high-level term k, we have a 2x2 matrix multiplication.
    # We use the 2x2 Strassen factors for this multiplication.
    # The U, V, W factors of the 2x2 multiplication operate on the 2x2 blocks.
    
    # Let's iterate through the 7 high-level terms.
    # For each term, we need to specify:
    # 1. Which 2x2 blocks of A4x4 are involved and with what coefficients.
    # 2. Which 2x2 blocks of B4x4 are involved and with what coefficients.
    # 3. Which 2x2 blocks of C4x4 receive the result and with what coefficients.
    
    # Term 1: M1 = (A11 + A22)(B11 + B22)
    # A_blocks: {0: 1, 3: 1} (A11 is block 0, A22 is block 3)
    # B_blocks: {0: 1, 3: 1}
    # C_blocks: {0: 1, 3: 1} (Contributes to C11 and C22? No, M1 is 2x2.
    # Wait, C11 is a 2x2 block. M1 is a 2x2 matrix.
    # So M1 contributes to C11 with coeff 1.
    # Does M1 contribute to C22? Yes, C22 = M1 - M3 + M7.
    # So M1 contributes to C11 (block 0) and C22 (block 3).
    
    # We need to be careful. The 2x2 Strassen decomposition computes X*Y = Z.
    # Here X is a 2x2 matrix formed by linear combo of A blocks.
    # Y is a 2x2 matrix formed by linear combo of B blocks.
    # Z is a 2x2 matrix.
    # Z is added to C blocks.
    
    # For Term 1:
    # X = A11 + A22. (Blocks 0 and 3 of A)
    # Y = B11 + B22. (Blocks 0 and 3 of B)
    # Z = M1.
    # C11 += M1. (Block 0 of C)
    # C22 += M1. (Block 3 of C)
    
    # So for Term 1, we use _strassen_2x2() factors.
    # U_2x2 is 4x7. V_2x2 is 4x7. W_2x2 is 4x7.
    # We need to map the 4 indices of U_2x2 to the 16 indices of U_4x4.
    # U_2x2[i, c] corresponds to the i-th element of the 2x2 input matrix X.
    # The elements of X are linear combinations of the elements of the A blocks.
    # A block is 2x2.
    # X = A11 + A22.
    # X[0,0] = A11[0,0] + A22[0,0]
    # X[0,1] = A11[0,1] + A22[0,1]
    # X[1,0] = A11[1,0] + A22[1,0]
    # X[1,1] = A11[1,1] + A22[1,1]
    
    # So U_4x4[0, col] += U_2x2[0, c] * 1 (from A11[0,0])
    # U_4x4[4, col] += U_2x2[0, c] * 1 (from A22[0,0])
    # U_4x4[1, col] += U_2x2[1, c] * 1 (from A11[0,1])
    # U_4x4[5, col] += U_2x2[1, c] * 1 (from A22[0,1])
    # ... and so on.
    
    # This is a standard block expansion.
    
    # Let's define the expansion logic.
    # A_block_map: dict from 2x2 index (0..3) to 4x4 index (0..15)
    # For A11 (block 0): 0->0, 1->1, 2->2, 3->3
    # For A22 (block 3): 0->12, 1->13, 2->14, 3->15
    
    # We will define the 7 terms explicitly.
    
    terms = [
        # (A_blocks, B_blocks, C_blocks)
        # Each block is (block_id, coeff)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1}), # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1}), # M3
        ({3: 1}, {2: 1, 0: -1}, {2: 1, 0: -1}), # M4
        ({0: 1, 1: 1}, {3: 1}, {1: 1}), # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {0: -1, 2: 1}), # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: -1, 3: 1}), # M7
    ]
    
    # Helper to expand block coefficients to full 16x1 matrix
    def expand_block_coeffs(block_coeffs):
        # block_coeffs: dict {block_id: coeff}
        # Returns 16x1 vector
        vec = [0.0] * 16
        for bid, c in block_coeffs.items():
            # Block bid is 2x2.
            # Indices:
            # 0: 0,1,2,3
            # 1: 4,5,6,7
            # 2: 8,9,10,11
            # 3: 12,13,14,15
            base = bid * 4
            for i in range(4):
                vec[base + i] += c
        return vec
    
    for t_idx, (a_blocks, b_blocks, c_blocks) in enumerate(terms):
        # Get 2x2 Strassen factors
        U2, V2, W2 = _strassen_2x2()
        
        # Expand A and B block coefficients to 16-length vectors
        # Note: The 2x2 Strassen U2 is 4x7.
        # U2[i, c] is the coefficient of the i-th element of the 2x2 input X.
        # The i-th element of X is a sum of elements from the A blocks.
        # We need to construct the full U_4x4 contribution for this term.
        
        # Let's build the full U, V, W for this term (16x7)
        U_term = [[0]*7 for _ in range(16)]
        V_term = [[0]*7 for _ in range(16)]
        W_term = [[0]*7 for _ in range(16)]
        
        # Map 2x2 indices to 4x4 indices for each block involved
        # For each 2x2 index i (0..3), it corresponds to a specific element in the 2x2 matrix.
        # The 2x2 matrix X is formed by A blocks.
        # X[i] = sum_{bid} coeff_{bid} * A_block[bid][i]
        
        # We can iterate over the 2x2 indices (0..3)
        for i2 in range(4):
            # Determine which 4x4 indices correspond to X[i2]
            # X[i2] is the i2-th element of the 2x2 matrix.
            # The 2x2 matrix elements are:
            # 0: (0,0)
            # 1: (0,1)
            # 2: (1,0)
            # 3: (1,1)
            
            # For each block involved in A:
            for bid, c in a_blocks.items():
                base = bid * 4
                idx4 = base + i2
                # U2[i2, c_strassen] is the coefficient
                for c_s in range(7):
                    U_term[idx4][c_s] += c * U2[i2][c_s]
        
        # Similarly for V
        for i2 in range(4):
            for bid, c in b_blocks.items():
                base = bid * 4
                idx4 = base + i2
                for c_s in range(7):
                    V_term[idx4][c_s] += c * V2[i2][c_s]
        
        # Similarly for W
        for i2 in range(4):
            for bid, c in c_blocks.items():
                base = bid * 4
                idx4 = base + i2
                for c_s in range(7):
                    W_term[idx4][c_s] += c * W2[i2][c_s]
        
        # Add to global U, V, W
        col_offset = t_idx * 7
        for r in range(16):
            for c_s in range(7):
                U[r][col_offset + c_s] += U_term[r][c_s]
                V[r][col_offset + c_s] += V_term[r][c_s]
                W[r][col_offset + c_s] += W_term[r][c_s]
                
    return U, V, W

def _schoolbook_rect(n, m, p):
    """
    Returns U, V, W for n x m by m x p matrix multiplication (schoolbook).
    U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _construct_rank_105():
    """
    Constructs a rank 105 decomposition for 4x5 x 5x6 matrix multiplication.
    """
    n, m, p = 4, 5, 6
    
    # Part 1: A1 (4x4) * B1a (4x4) -> C1 (4x4)
    # A1 is A[:, 0:4], B1a is B[0:4, 0:4]
    U1, V1, W1 = _strassen_4x4()
    rank1 = 49
    
    # Part 2: A1 (4x4) * B1b (4x2) -> C2 (4x2)
    # B1b is B[0:4, 4:6]
    U2, V2, W2 = _schoolbook_rect(4, 4, 2)
    rank2 = 32
    
    # Part 3: A2 (4x1) * B2 (1x6) -> C3 (4x6)
    # A2 is A[:, 4], B2 is B[4, 0:6]
    U3, V3, W3 = _schoolbook_rect(4, 1, 6)
    rank3 = 24
    
    total_rank = rank1 + rank2 + rank3 # 105
    
    U = [[0]*total_rank for _ in range(n*m)] # 20 x 105
    V = [[0]*total_rank for _ in range(m*p)] # 30 x 105
    W = [[0]*total_rank for _ in range(p*n)] # 24 x 105
    
    # Fill Part 1
    # A1 indices: i in 0..3, j in 0..3 -> U1 rows 0..15
    # B1a indices: j in 0..3, k in 0..3 -> V1 rows 0..15
    # C1 indices: k in 0..3, i in 0..3 -> W1 rows 0..15
    
    for r in range(16):
        for c in range(rank1):
            U[r][c] = U1[r][c]
            V[r][c] = V1[r][c]
            W[r][c] = W1[r][c]
    
    # Fill Part 2
    # A1 indices: i in 0..3, j in 0..3 -> U2 rows 0..15
    # B1b indices: j in 0..3, k in 4..5 -> V2 rows 0..15 (mapped to B rows j*p + k)
    # C2 indices: k in 4..5, i in 0..3 -> W2 rows 0..15 (mapped to C rows k*n + i)
    
    offset2 = rank1
    for r in range(16):
        for c in range(rank2):
            U[r][offset2 + c] = U2[r][c]
            
            # V mapping: V2 row r corresponds to j, k_local (0..1)
            # j = r // 2, k_local = r % 2
            # B row index = j * p + (4 + k_local)
            j = r // 2
            k_local = r % 2
            b_row = j * p + 4 + k_local
            V[b_row][offset2 + c] = V2[r][c]
            
            # W mapping: W2 row r corresponds to k_local, i
            # k_local = r // 4? No, W2 is (p*n, rank) = (2*4, 32).
            # W2 row r: k_local = r // 4, i = r % 4?
            # _schoolbook_rect(4, 4, 2) -> W shape (2*4, 32).
            # W2[r, c] where r = k_local * n + i?
            # In _schoolbook_rect, W is indexed by k*n + i.
            # Here p=2, n=4. So k_local in 0..1, i in 0..3.
            # r = k_local * 4 + i.
            k_local = r // 4
            i = r % 4
            c_row = (4 + k_local) * n + i
            W[c_row][offset2 + c] = W2[r][c]
    
    # Fill Part 3
    # A2 indices: i in 0..3, j=4 -> U3 rows 0..3 (mapped to A rows i*m + 4)
    # B2 indices: j=4, k in 0..5 -> V3 rows 0..5 (mapped to B rows 4*p + k)
    # C3 indices: k in 0..5, i in 0..3 -> W3 rows 0..23 (mapped to C rows k*n + i)
    
    offset3 = rank1 + rank2
    for r in range(4):
        for c in range(rank3):
            # U mapping: U3 row r corresponds to i=r, j=0 (local)
            # A row index = r * m + 4
            a_row = r * m + 4
            U[a_row][offset3 + c] = U3[r][c]
            
            # V mapping: V3 row r?
            # V3 is (m*p, rank) = (1*6, 24).
            # V3[r, c] where r = j_local * p + k?
            # Here m=1, p=6. j_local=0. r = k.
            b_row = 4 * p + r
            V[b_row][offset3 + c] = V3[r][c]
            
            # W mapping: W3 row r?
            # W3 is (p*n, rank) = (6*4, 24).
            # W3[r, c] where r = k * n + i.
            c_row = r
            W[c_row][offset3 + c] = W3[r][c]
            
            # Note: The loop for W3 should iterate over all 24 rows of W3
            # But U3 and V3 only have 4 and 6 rows respectively.
            # We need to iterate c over rank3, and for each c, update U, V, W.
            # My loop structure above is wrong for W3.
            
    # Let's redo the filling logic more carefully.
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    # Part 1
    for c in range(rank1):
        for r in range(16):
            U[r][c] = U1[r][c]
            V[r][c] = V1[r][c]
            W[r][c] = W1[r][c]
            
    # Part 2
    for c in range(rank2):
        for r in range(16):
            U[r][offset2 + c] = U2[r][c]
            j = r // 2
            k_local = r % 2
            b_row = j * p + 4 + k_local
            V[b_row][offset2 + c] = V2[r][c]
            
            k_local_w = r // 4
            i = r % 4
            c_row = (4 + k_local_w) * n + i
            W[c_row][offset2 + c] = W2[r][c]
            
    # Part 3
    for c in range(rank3):
        # U3 has 4 rows
        for r in range(4):
            a_row = r * m + 4
            U[a_row][offset3 + c] = U3[r][c]
            
        # V3 has 6 rows
        for r in range(6):
            b_row = 4 * p + r
            V[b_row][offset3 + c] = V3[r][c]
            
        # W3 has 24 rows
        for r in range(24):
            c_row = r
            W[c_row][offset3 + c] = W3[r][c]
            
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Check if the decomposition is exact. Returns True if valid."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    result = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        uc = Ua[:, c].reshape(n, m)
        vc = Va[:, c].reshape(m, p)
        wc = Wa[:, c].reshape(p, n)
        
        # Efficient contraction
        # result[i,j,k] += uc[i,j] * vc[j,k] * wc[k,i]
        # We can do: temp[j,k] = vc[j,k] * wc[k,i] for each i?
        # Or: result += np.einsum('ij,jk,ki->ijk', uc, vc, wc)
        result += np.einsum('ij,jk,ki->ijk', uc, vc, wc)
        
    return np.allclose(result, 1.0, atol=1e-9)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try the rank 105 decomposition
    best = _construct_rank_105()
    
    if _verify(best["U"], best["V"], best["W"]):
        return best
    
    # Fallback to schoolbook
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
