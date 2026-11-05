"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def _build_strassen_6x6():
    """
    Build a rank-7^2 = 49 decomposition for 6x6 matrix multiplication
    using a 2x2 block Strassen decomposition where each block is 3x3,
    and each 3x3 block multiplication is done with schoolbook (rank 27).
    
    Wait, that gives 7 * 27 = 189, which is worse than schoolbook 216.
    
    Actually, let's think more carefully.
    
    For 6x6, we can partition into 2x2 blocks of 3x3.
    Strassen for 2x2 block matrices uses 7 multiplications of 3x3 matrices.
    Each 3x3 multiplication has rank 27 (schoolbook).
    Total rank = 7 * 27 = 189.
    
    Schoolbook for 6x6 is 6*6*6 = 216.
    So 189 < 216. This is an improvement!
    
    Let me construct this carefully.
    
    A = [[A11, A12], [A21, A22]] where each Ai is 3x3
    B = [[B11, B12], [B21, B22]] where each Bj is 3x3
    
    Strassen:
    M1 = (A11 + A22) * (B11 + B22)
    M2 = (A21 + A22) * B11
    M3 = A11 * (B12 - B22)
    M4 = A22 * (B21 - B11)
    M5 = (A11 + A12) * B22
    M6 = (A21 - A11) * (B11 + B12)
    M7 = (A12 - A22) * (B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Each Mi is a 3x3 matrix product.
    
    Now, the tensor decomposition:
    For the overall 6x6 multiplication tensor T[i,j,k] where:
    - i indexes rows of A: 0..5
    - j indexes cols of A / rows of B: 0..5  
    - k indexes cols of B: 0..5
    
    Wait, let me use the standard convention. The task says:
    W[k*n+i, r] where n=6, p=6, m=6.
    
    So A is n x m = 6x6, B is m x p = 6x6, C is p x n = 6x6.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The output factor W has shape (p*n, rank) = (36, rank).
    W[k*n + i, r] corresponds to C[i,k].
    
    For the block structure:
    A is divided into 2x2 blocks of 3x3:
    A11 = A[0:3, 0:3], A12 = A[0:3, 3:6]
    A21 = A[3:6, 0:3], A22 = A[3:6, 3:6]
    
    B is divided into 2x2 blocks of 3x3:
    B11 = B[0:3, 0:3], B12 = B[0:3, 3:6]
    B21 = B[3:6, 0:3], B22 = B[3:6, 3:6]
    
    C is divided into 2x2 blocks of 3x3:
    C11 = C[0:3, 0:3], C12 = C[0:3, 3:6]
    C21 = C[3:6, 0:3], C22 = C[3:6, 3:6]
    
    For each Strassen multiplication Mi, we need to:
    1. Form the input matrices (linear combinations of blocks)
    2. Multiply them (3x3 schoolbook, rank 27)
    3. Combine outputs (linear combinations of result blocks)
    
    The U matrix has shape (n*m, rank) = (36, rank).
    U[i*m + j, r] gives the coefficient for A[i,j] in the r-th term.
    
    The V matrix has shape (m*p, rank) = (36, rank).
    V[j*p + k, r] gives the coefficient for B[j,k] in the r-th term.
    
    The W matrix has shape (p*n, rank) = (36, rank).
    W[k*n + i, r] gives the coefficient for the r-th term in C[i,k].
    
    Let me define the block indices:
    For A: block (bi, bj) where bi in {0,1}, bj in {0,1}
      Row range: [bi*3, (bi+1)*3)
      Col range: [bj*3, (bj+1)*3)
      So A[i,j] is in block (i//3, j//3)
    
    For B: block (bj, bk) where bj in {0,1}, bk in {0,1}
      Row range: [bj*3, (bj+1)*3)
      Col range: [bk*3, (bk+1)*3)
      So B[j,k] is in block (j//3, k//3)
    
    For C: block (bi, bk) where bi in {0,1}, bk in {0,1}
      Row range: [bi*3, (bi+1)*3)
      Col range: [bk*3, (bk+1)*3)
      So C[i,k] is in block (i//3, k//3)
    
    For each Strassen term Mi, the input A-side is a linear combination of A-blocks,
    and the input B-side is a linear combination of B-blocks.
    
    Let's define the Strassen terms:
    M1: A_side = A11 + A22, B_side = B11 + B22
    M2: A_side = A21 + A22, B_side = B11
    M3: A_side = A11, B_side = B12 - B22
    M4: A_side = A22, B_side = B21 - B11
    M5: A_side = A11 + A12, B_side = B22
    M6: A_side = A21 - A11, B_side = B11 + B12
    M7: A_side = A12 - A22, B_side = B21 + B22
    
    For each Mi, the A-side is a sum of coefficients times A-blocks.
    Each A-block is 3x3, so for each (i,j) in the block, U[i*6+j, r] gets the coefficient.
    
    Similarly for B-side and V.
    
    For the output, C-blocks are linear combinations of Mi results.
    Each Mi is a 3x3 matrix. For each (i,k) in the C-block, W[k*6+i, r] gets the coefficient.
    
    Let me code this up.
    """
    n = m = p = 6
    block_size = 3
    
    # Define Strassen terms: (A_block_coeffs, B_block_coeffs)
    # A_block_coeffs: dict from (bi, bj) to coefficient
    # B_block_coeffs: dict from (bj, bk) to coefficient
    strassen_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),  # M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),              # M2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),             # M3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),             # M4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),              # M5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),   # M6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),   # M7
    ]
    
    # Output combination: C_block_coeffs for each C-block (ci, ck)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    c_combinations = {
        (0,0): {0: 1, 3: 1, 4: -1, 6: 1},   # M1, M4, M5, M7
        (0,1): {2: 1, 4: 1},                 # M3, M5
        (1,0): {1: 1, 3: 1},                 # M2, M4
        (1,1): {0: 1, 1: -1, 2: 1, 5: 1},    # M1, M2, M3, M6
    }
    
    # Each Strassen term uses schoolbook 3x3 multiplication, which has 27 terms.
    # Total rank = 7 * 27 = 189
    
    rank = 7 * 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for mi, (a_coeffs, b_coeffs) in enumerate(strassen_terms):
        # For each schoolbook term in the 3x3 multiplication
        for ai in range(block_size):
            for aj in range(block_size):
                for bk in range(block_size):
                    r = mi * 27 + ai * 9 + aj * 3 + bk
                    
                    # U: for each A-block (bi, bj) with coefficient c
                    for (bi, bj), c in a_coeffs.items():
                        if c != 0:
                            # A[i,j] where i in [bi*3, (bi+1)*3), j in [bj*3, (bj+1)*3)
                            # The local indices are ai, aj within the block
                            # But wait: the schoolbook term (ai, aj, bk) means:
                            #   result[ai, bk] += A_local[ai, aj] * B_local[aj, bk]
                            # So A_local[ai, aj] corresponds to global A[bi*3+ai, bj*3+aj]
                            i_global = bi * block_size + ai
                            j_global = bj * block_size + aj
                            U[i_global * m + j_global][r] = c
                    
                    # V: for each B-block (bj, bk_block) with coefficient c
                    for (bj, bk_block), c in b_coeffs.items():
                        if c != 0:
                            # B[j,k] where j in [bj*3, (bj+1)*3), k in [bk_block*3, (bk_block+1)*3)
                            # The local index for B is (aj, bk) within the block
                            # Wait, I need to be careful. In the 3x3 multiplication:
                            #   (A_local * B_local)[ai, bk] = sum_aj A_local[ai, aj] * B_local[aj, bk]
                            # So B_local[aj, bk] corresponds to global B[bj*3+aj, bk_block*3+bk]
                            j_global = bj * block_size + aj
                            k_global = bk_block * block_size + bk
                            V[j_global * p + k_global][r] = c
                    
                    # W: for each C-block (ci, ck) that includes this Mi
                    for (ci, ck), comb in c_combinations.items():
                        if mi in comb:
                            c_val = comb[mi]
                            if c_val != 0:
                                # C[i,k] where i in [ci*3, (ci+1)*3), k in [ck*3, (ck+1)*3)
                                # The local result index is (ai, bk)
                                i_global = ci * block_size + ai
                                k_global = ck * block_size + bk
                                W[k_global * n + i_global][r] = c_val
    
    return {"U": U, "V": V, "W": W}


def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    # Try Strassen-based decomposition first
    try:
        result = _build_strassen_6x6()
        # Verify quickly
        import numpy as np
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)
        n, m, p = DIMENSIONS
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 if i==i', j==j', k==k' in the tensor sense
        # Actually, the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # should equal delta_{i,i'} delta_{j,j'} delta_{k,k'} ... no.
        # The matrix multiplication tensor is T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k]
        # So T[i,j,k] = 1 for all valid (i,j,k) in the sense that it's the structure tensor.
        # Actually T[i,j,k] is the coefficient such that C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k]
        # No, that's not right either. Let me think again.
        
        # The standard formulation: C = A * B
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T has T[i,j,k] = 1 (it's the identity in the middle index)
        # The decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] (with appropriate indexing)
        
        # Given the interface:
        # U has shape (n*m, rank), indexed as U[i*m + j, r]
        # V has shape (m*p, rank), indexed as V[j*p + k, r]
        # W has shape (p*n, rank), indexed as W[k*n + i, r]
        
        # The condition is: for all i, j, k:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 (if this represents the multiplication tensor)
        
        # Wait, actually the matrix multiplication tensor T[i,j,k] is defined such that
        # (A*B)[i,k] = sum_j A[i,j] B[j,k]
        # So the tensor T[i,j,k] = 1 for the "correct" combination, but really it's:
        # T[i,j,k] is the coefficient of A[i,j]*B[j,k] in C[i,k], which is 1 if the indices match.
        
        # In the rank decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This should equal 1 for all i in [0,n), j in [0,m), k in [0,p).
        
        # Let's verify
        errors = 0
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    err = abs(val - 1.0)
                    if err > 1e-9:
                        errors += 1
                        max_err = max(max_err, err)
        
        if errors == 0:
            elapsed = time.time() - start_time
            if elapsed < deadline:
                return result
        else:
            # Verification failed, fall back to schoolbook
            pass
    except Exception:
        pass
    
    # Fall back to schoolbook
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
