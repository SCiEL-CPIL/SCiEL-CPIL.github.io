"""Matrix multiplication 4,4,9 - Strassen-based rank-114 decomposition."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36
    
    def build_strassen_2x2x2():
        """
        Return U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen).
        U: (4, 7) - indices for A: a00=0, a01=1, a10=2, a11=3
        V: (4, 7) - indices for B: b00=0, b01=1, b10=2, b11=3
        W: (4, 7) - indices for C: c[i,k] -> W[k*2+i]: c00=0, c01=1, c10=2, c11=3
        
        Strassen formulas:
        M1 = (a00+a11)*(b00+b11)
        M2 = (a10+a11)*b00
        M3 = a00*(b01-b11)
        M4 = a11*(b10-b00)
        M5 = (a00+a01)*b11
        M6 = (a10-a00)*(b00+b01)
        M7 = (a01-a11)*(b10+b11)
        
        c00 = M1 + M4 - M5 + M7
        c01 = M3 + M5
        c10 = M2 + M4
        c11 = M1 - M2 + M3 + M6
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1: (a00+a11)*(b00+b11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1  # c00
        W[3, 0] = 1  # c11
        
        # M2: (a10+a11)*b00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1  # c10
        
        # M3: a00*(b01-b11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1  # c01
        
        # M4: a11*(b10-b00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1  # c10
        W[0, 3] = 1  # c00
        
        # M5: (a00+a01)*b11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1  # c00
        W[1, 4] = 1   # c01
        
        # M6: (a10-a00)*(b00+b01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1  # c11
        
        # M7: (a01-a11)*(b10+b11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1  # c00
        
        return U, V, W
    
    def build_strassen_4x4x4():
        """
        Return U, V, W for 4x4x4 matrix multiplication with rank 49.
        Uses recursive Strassen: split into 2x2 blocks, apply Strassen at top level,
        then each 2x2 product uses Strassen again.
        
        U: (16, 49) - indices for A: A[i,j] -> index i*4+j
        V: (16, 49) - indices for B: B[j,k] -> index j*4+k
        W: (16, 49) - indices for C: C[i,k] -> W[k*4+i]
        """
        U2, V2, W2 = build_strassen_2x2x2()
        
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        # Top-level Strassen for 4x4:
        # A = [[A00, A01], [A10, A11]] where each is 2x2
        # B = [[B00, B01], [B10, B11]] where each is 2x2
        # 
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For each top-level term, we need to compute a 2x2 matrix product
        # where the operands are sums/differences of 2x2 blocks.
        # Each such product decomposes into 7 rank-1 terms using Strassen.
        
        # Define helper to add a Strassen term for a 2x2 product
        # where left operand is sum of blocks with given coefficients
        # and right operand is sum of blocks with given coefficients.
        
        def add_2x2_product_term(term_idx, left_blocks, right_blocks, output_blocks):
            """
            Add 7 Strassen terms for a 2x2 matrix product.
            left_blocks: list of (block_idx, coeff) for left operand
            right_blocks: list of (block_idx, coeff) for right operand
            output_blocks: list of (block_idx, coeff) for output contribution
            
            block_idx: 0=A00, 1=A01, 2=A10, 3=A11 for A
                      0=B00, 1=B01, 2=B10, 3=B11 for B
                      0=C00, 1=C01, 2=C10, 3=C11 for C
            """
            # For the 2x2 product, we use the Strassen decomposition.
            # The left 2x2 matrix has elements that are linear combinations of A elements.
            # The right 2x2 matrix has elements that are linear combinations of B elements.
            
            # For each of the 7 Strassen terms, we add contributions to U, V, W.
            # Strassen term t: U_t * V_t * W_t
            # where U_t involves left matrix elements, V_t involves right matrix elements,
            # and W_t involves output matrix elements.
            
            # Left matrix elements (2x2): L[i,j] = sum over (block, coeff) of coeff * A_block[i,j]
            # Right matrix elements (2x2): R[i,j] = sum over (block, coeff) of coeff * B_block[i,j]
            # Output: sum over (block, coeff) of coeff * C_block[i,j]
            
            for s in range(7):
                global_term = term_idx * 7 + s
                
                # U contribution: for each left block with coeff c, add c * U2[:, s] to appropriate rows
                for (blk, coeff) in left_blocks:
                    row_off = (blk // 2) * 2  # 0 for blk 0,1; 2 for blk 2,3
                    col_off = (blk % 2) * 2
                    for li in range(2):
                        for lj in range(2):
                            U[(row_off + li) * 4 + (col_off + lj), global_term] += coeff * U2[li * 2 + lj, s]
                
                # V contribution: for each right block with coeff c, add c * V2[:, s] to appropriate rows
                for (blk, coeff) in right_blocks:
                    row_off = (blk // 2) * 2
                    col_off = (blk % 2) * 2
                    for lj in range(2):
                        for lk in range(2):
                            V[(row_off + lj) * 4 + (col_off + lk), global_term] += coeff * V2[lj * 2 + lk, s]
                
                # W contribution: for each output block with coeff c, add c * W2[:, s] to appropriate rows
                # W index for C[i,k] is k*4+i
                for (blk, coeff) in output_blocks:
                    row_off = (blk // 2) * 2  # row offset in C
                    col_off = (blk % 2) * 2   # col offset in C
                    for li in range(2):
                        for lk in range(2):
                            W[(col_off + lk) * 4 + (row_off + li), global_term] += coeff * W2[lk * 2 + li, s]
        
        # M1 = (A00+A11)*(B00+B11)
        # Left: A00 (coeff 1) + A11 (coeff 1)
        # Right: B00 (coeff 1) + B11 (coeff 1)
        # Output: C00 (coeff 1) + C11 (coeff 1)
        add_2x2_product_term(0, [(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])
        
        # M2 = (A10+A11)*B00
        # Left: A10 (coeff 1) + A11 (coeff 1)
        # Right: B00 (coeff 1)
        # Output: C10 (coeff 1)
        add_2x2_product_term(1, [(2, 1), (3, 1)], [(0, 1)], [(2, 1)])
        
        # M3 = A00*(B01-B11)
        # Left: A00 (coeff 1)
        # Right: B01 (coeff 1) + B11 (coeff -1)
        # Output: C01 (coeff 1)
        add_2x2_product_term(2, [(0, 1)], [(1, 1), (3, -1)], [(1, 1)])
        
        # M4 = A11*(B10-B00)
        # Left: A11 (coeff 1)
        # Right: B10 (coeff 1) + B00 (coeff -1)
        # Output: C10 (coeff 1) + C00 (coeff 1)
        add_2x2_product_term(3, [(3, 1)], [(2, 1), (0, -1)], [(2, 1), (0, 1)])
        
        # M5 = (A00+A01)*B11
        # Left: A00 (coeff 1) + A01 (coeff 1)
        # Right: B11 (coeff 1)
        # Output: C00 (coeff -1) + C01 (coeff 1)
        add_2x2_product_term(4, [(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)])
        
        # M6 = (A10-A00)*(B00+B01)
        # Left: A10 (coeff 1) + A00 (coeff -1)
        # Right: B00 (coeff 1) + B01 (coeff 1)
        # Output: C11 (coeff 1)
        add_2x2_product_term(5, [(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])
        
        # M7 = (A01-A11)*(B10+B11)
        # Left: A01 (coeff 1) + A11 (coeff -1)
        # Right: B10 (coeff 1) + B11 (coeff 1)
        # Output: C00 (coeff 1)
        add_2x2_product_term(6, [(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
        
        return U, V, W
    
    def build_4x4x1_schoolbook():
        """
        Return U, V, W for 4x4x1 matrix multiplication with rank 16 (schoolbook).
        U: (16, 16) - indices for A: A[i,j] -> index i*4+j
        V: (4, 16)  - indices for B: B[j,0] -> index j*1+0 = j (since p=1)
        W: (4, 16)  - indices for C: C[i,0] -> W[0*4+i] = W[i]
        """
        U = np.zeros((16, 16))
        V = np.zeros((4, 16))
        W = np.zeros((4, 16))
        
        for i in range(4):
            for j in range(4):
                col = i * 4 + j
                U[i * 4 + j, col] = 1
                V[j, col] = 1
                W[i, col] = 1
        
        return U, V, W
    
    # Build the 4x4x4 Strassen decomposition
    U_444, V_444, W_444 = build_strassen_4x4x4()
    
    # Verify the 4x4x4 decomposition
    C_check = np.zeros((4, 4))
    for i in range(4):
        for k in range(4):
            for j in range(4):
                C_check[i, k] += np.dot(U_444[i * 4 + j, :], V_444[j * 4 + k, :]) * W_444[k * 4 + i, :]
    
    valid_444 = np.allclose(C_check, np.ones((4, 4)), atol=1e-8)
    
    # Build the 4x4x1 schoolbook decomposition
    U_441, V_441, W_441 = build_4x4x1_schoolbook()
    
    # Verify the 4x4x1 decomposition
    C_check_441 = np.zeros((4, 1))
    for i in range(4):
        for k in range(1):
            for j in range(4):
                C_check_441[i, k] += np.dot(U_441[i * 4 + j, :], V_441[j * 1 + k, :]) * W_441[k * 4 + i, :]
    
    valid_441 = np.allclose(C_check_441, np.ones((4, 1)), atol=1e-8)
    
    if valid_444 and valid_441:
        # Combine: 4x4x9 = two 4x4x4 blocks + one 4x4x1 block
        # Total rank: 49 + 49 + 16 = 114
        total_rank = 49 + 49 + 16
        U_total = np.zeros((16, total_rank))
        V_total = np.zeros((36, total_rank))
        W_total = np.zeros((36, total_rank))
        
        # First 4x4x4 block: columns 0-3 of B
        U_total[:, 0:49] = U_444
        V_total[0:16, 0:49] = V_444
        W_total[0:16, 0:49] = W_444  # W[k*4+i] for k=0..3, i=0..3 -> indices 0..15
        
        # Second 4x4x4 block: columns 4-7 of B
        U_total[:, 49:98] = U_444
        V_total[16:32, 49:98] = V_444
        W_total[16:32, 49:98] = W_444  # W[k*4+i] for k=4..7 -> indices 16..31
        
        # Third 4x4x1 block: column 8 of B (schoolbook, rank 16)
        for i in range(4):
            for j in range(4):
                col = 98 + i * 4 + j
                U_total[i * 4 + j, col] = 1
                V_total[j * 9 + 8, col] = 1  # B[j, 8] -> index j*9+8
                W_total[8 * 4 + i, col] = 1  # C[i, 8] -> W[8*4+i]
        
        # Verify the full 4x4x9 decomposition
        C_full = np.zeros((4, 9))
        for i in range(4):
            for k in range(9):
                for j in range(4):
                    C_full[i, k] += np.dot(U_total[i * 4 + j, :], V_total[j * 9 + k, :]) * W_total[k * 4 + i, :]
        
        valid_full = np.allclose(C_full, np.ones((4, 9)), atol=1e-8)
        
        if valid_full:
            return {
                "U": [[float(x) for x in row] for row in U_total],
                "V": [[float(x) for x in row] for row in V_total],
                "W": [[float(x) for x in row] for row in W_total]
            }
    
    # Fallback: schoolbook decomposition with rank 144
    rank_schoolbook = 144
    U_fb = np.zeros((16, rank_schoolbook))
    V_fb = np.zeros((36, rank_schoolbook))
    W_fb = np.zeros((36, rank_schoolbook))
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(9):
                U_fb[i * 4 + j, col] = 1
                V_fb[j * 9 + k, col] = 1
                W_fb[k * 4 + i, col] = 1
                col += 1
    
    return {
        "U": [[float(x) for x in row] for row in U_fb],
        "V": [[float(x) for x in row] for row in V_fb],
        "W": [[float(x) for x in row] for row in W_fb]
    }
# EVOLVE-BLOCK-END
