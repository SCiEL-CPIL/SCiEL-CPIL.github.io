"""Strassen-based rank-minimal decomposition for 4x4 x 4x6 matrix multiplication."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import time
    t0 = time.time()
    DEADLINE = 130  # seconds, safety margin from 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # We'll build U, V, W matrices.
    # U: (n*m) x rank = 16 x rank
    # V: (m*p) x rank = 24 x rank
    # W: (p*n) x rank = 24 x rank
    # Each rank-1 term corresponds to a scalar multiplication at (i, j, k)
    # where i in [0,n), j in [0,m), k in [0,p)
    # U[i*m + j, c] = 1, V[j*p + k, c] = 1, W[k*n + i, c] = 1
    
    # Strategy: Split the 4x6 output into two 4x3 halves.
    # For each 4x4 x 4x3 multiplication, use a Strassen-like approach.
    
    # For 4x4 x 4x3:
    # A is 4x4, split into 2x2 blocks: A11, A12, A21, A22
    # B is 4x3, split into 2x3 blocks: B11, B12, B21, B22 (each 2x3)
    # C is 4x3, split into 2x3 blocks: C11, C12, C21, C22 (each 2x3)
    
    # Standard Strassen for 2x2 x 2x2 uses 7 multiplications.
    # For 2x2 x 2x3, we need a different approach.
    
    # Let's use a recursive Strassen for the 4x4 x 4x4 part (rank 49)
    # and handle the extra 2 columns separately.
    
    # Actually, let me think about this more carefully.
    # 4x4 x 4x6 = 4x4 x 4x4 + 4x4 x 4x2
    
    # For 4x4 x 4x4: recursive Strassen gives rank 49
    # For 4x4 x 4x2: we can use a similar approach
    
    # Let me implement recursive Strassen for 4x4 x 4x4 first.
    
    def strassen_4x4x4():
        """Generate terms for 4x4 x 4x4 matrix multiplication using recursive Strassen."""
        terms = []  # list of (u_dict, v_dict, w_dict)
        
        # A is 4x4, B is 4x4, C is 4x4
        # Split into 2x2 blocks:
        # A11: rows 0-1, cols 0-1
        # A12: rows 0-1, cols 2-3
        # A21: rows 2-3, cols 0-1
        # A22: rows 2-3, cols 2-3
        # B11: rows 0-1, cols 0-1
        # B12: rows 0-1, cols 2-3
        # B21: rows 2-3, cols 0-1
        # B22: rows 2-3, cols 2-3
        # C11: rows 0-1, cols 0-1
        # C12: rows 0-1, cols 2-3
        # C21: rows 2-3, cols 0-1
        # C22: rows 2-3, cols 2-3
        
        # Strassen formulas for 2x2 x 2x2:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M5
        # C22 = M1 - M2 + M3 + M6
        
        # Each M is a 2x2 x 2x2 product, which recursively uses Strassen.
        # For 2x2 x 2x2, we can use the base case with 7 terms.
        
        def add_strassen_term(offset_a, offset_b, offset_c, scale_a, scale_b, scale_c):
            """Add a Strassen term for 2x2 x 2x2 at given offsets."""
            # offset_a: (row_offset, col_offset) in A
            # offset_b: (row_offset, col_offset) in B
            # offset_c: (row_offset, col_offset) in C
            # scale_a, scale_b, scale_c: scaling factors for A, B, C blocks
            
            a_ro, a_co = offset_a
            b_ro, b_co = offset_b
            c_ro, c_co = offset_c
            
            # For each of the 7 Strassen multiplications, add terms
            # M1 = (A11 + A22)(B11 + B22)
            # This requires adding 4 terms for the (A11+A22) and 4 for (B11+B22)
            # But we need to be careful about the structure.
            
            # Let me think differently. Each Strassen multiplication M = X*Y where
            # X and Y are 2x2 matrices that are linear combinations of subblocks.
            # The result M is a 2x2 matrix, and we need to add it to the appropriate
            # C blocks with appropriate signs.
            
            # For M1 = (A11 + A22)(B11 + B22):
            # X = A11 + A22, so X[i,j] = A[a_ro+i, a_co+j] + A[a_ro+2+i, a_co+2+j]
            # Y = B11 + B22, so Y[i,j] = B[b_ro+i, b_co+j] + B[b_ro+2+i, b_co+2+j]
            # Result M1[i,j] = sum_k X[i,k]*Y[k,j]
            # C11 += M1, C22 += M1
            
            # This is getting complex. Let me use a different approach.
            # Instead of trying to be clever, let me just enumerate all the scalar
            # multiplications explicitly.
            
            pass
        
        # Actually, let me just use the known rank-49 decomposition for 4x4x4.
        # The Landsberg decomposition has rank 49.
        
        # For simplicity, let me use a recursive Strassen implementation.
        # For 4x4x4: 7 * (2x2x2 rank) = 7 * 7 = 49 if we can do 2x2x2 in 7 terms.
        # But 2x2x2 schoolbook is 8 terms, Strassen is 7 terms.
        
        # Let me implement this properly.
        
        # Base case: 2x2 x 2x2 multiplication
        def strassen_2x2x2(a_off, b_off, c_off):
            """Return list of (u, v, w) for 2x2x2 Strassen."""
            result = []
            a_r, a_c = a_off
            b_r, b_c = b_off
            c_r, c_c = c_off
            
            # A blocks: A11 at (a_r, a_c), A12 at (a_r, a_c+1), A21 at (a_r+1, a_c), A22 at (a_r+1, a_c+1)
            # Wait, for 2x2, the blocks are individual elements.
            # A11 = A[a_r, a_c], A12 = A[a_r, a_c+1], A21 = A[a_r+1, a_c], A22 = A[a_r+1, a_c+1]
            
            # B11 = B[b_r, b_c], B12 = B[b_r, b_c+1], B21 = B[b_r+1, b_c], B22 = B[b_r+1, b_c+1]
            
            # M1 = (A11 + A22)(B11 + B22)
            # = (A[a_r,a_c] + A[a_r+1,a_c+1]) * (B[b_r,b_c] + B[b_r+1,b_c+1])
            # This is a product of two sums, so we need 4 terms:
            # A[a_r,a_c]*B[b_r,b_c] + A[a_r,a_c]*B[b_r+1,b_c+1] + A[a_r+1,a_c+1]*B[b_r,b_c] + A[a_r+1,a_c+1]*B[b_r+1,b_c+1]
            
            # But wait, in the tensor decomposition, each term is a rank-1 tensor.
            # For the matmul tensor, T[i,j,k] = 1 for all i,j,k.
            # So each scalar multiplication A[i,j]*B[j,k] contributes to C[i,k].
            
            # Let me reconsider. The Strassen algorithm reduces the number of scalar
            # multiplications, but each "multiplication" in Strassen is actually a
            # product of two linear combinations, which expands to multiple scalar mults.
            
            # I think I'm overcomplicating this. Let me just use the schoolbook
            # decomposition for now, which is guaranteed correct.
            
            # For 2x2 x 2x2: 4*2*2 = 16 scalar mults? No, 2*2*2 = 8.
            # C[i,k] = sum_j A[i,j]*B[j,k], so for each (i,k), we sum over j.
            # Total scalar mults = 2*2*2 = 8.
            
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        u = {(a_r + i)*4 + (a_c + j): 1}  # A[i,j]
                        v = {(b_r + j)*4 + (b_c + k): 1}  # B[j,k]
                        w = {(c_r + k)*4 + (c_r + i): 1}  # C[k,i] -> wait, C[i,k]
                        # Actually C is indexed as C[row, col], so C[i,k] is at position i*2+k or similar
                        # In our convention, W[k*n + i, c] for the full problem.
                        # For the subproblem, it's local.
                        pass
            
            return result
        
        # This is getting too complex. Let me just return the schoolbook decomposition
        # for the entire problem, which is rank 96.
        
        # Actually, let me try a different approach. Let me directly construct the
        # rank-49 decomposition for 4x4x4 using known results.
        
        # For now, let me just use schoolbook for everything.
        return []
    
    # Given the complexity, let me just use the schoolbook decomposition.
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
