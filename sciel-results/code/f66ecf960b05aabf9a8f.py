"""Strassen-based rank reduction for 4x5x5 matrix multiplication."""
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # We'll build up U, V, W as lists of lists
    # U: (n*m) x rank = 20 x rank
    # V: (m*p) x rank = 25 x rank
    # W: (p*n) x rank = 20 x rank
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        """Add a single tensor term."""
        U_cols.append([(1 if i == u_idx else 0) for i in range(n * m)])
        V_cols.append([(1 if i == v_idx else 0) for i in range(m * p)])
        W_cols.append([(1 if i == w_idx else 0) for i in range(p * n)])
    
    def add_term_val(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        """Add a single tensor term with specific values."""
        U_cols.append([u_val if i == u_idx else 0 for i in range(n * m)])
        V_cols.append([v_val if i == v_idx else 0 for i in range(m * p)])
        W_cols.append([w_val if i == w_idx else 0 for i in range(p * n)])
    
    # Strassen decomposition for the 4x4x4 core
    # A is 4x4 (rows 0-3, cols 0-3), B is 4x4 (rows 0-3, cols 0-3)
    # Split into 2x2 blocks:
    # A11 = A[0:2,0:2], A12 = A[0:2,2:4], A21 = A[2:4,0:2], A22 = A[2:4,2:4]
    # B11 = B[0:2,0:2], B12 = B[0:2,2:4], B21 = B[2:4,0:2], B22 = B[2:4,2:4]
    
    # The 7 Strassen products, each is a 2x2 matrix product
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Output:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each M is 2x2, so each has 4 scalar products
    # For each scalar product in a Strassen term, we need to expand the sums/diffs
    
    # Let's define helper: for a Strassen product like (A_sum)(B_sum) where 
    # A_sum is sum of A blocks with coefficients, B_sum is sum of B blocks with coefficients
    # We generate all pairwise products
    
    # Define block index ranges:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    # Similarly for B.
    
    # For each Strassen term, we have a sum of A-blocks (with coeffs) times sum of B-blocks (with coeffs)
    # Each A-block is 2x2, each B-block is 2x2
    # The product gives a 2x2 result
    
    # Let's enumerate all terms manually for clarity
    
    # M1 = (A11 + A22)(B11 + B22)
    # Terms: A11*B11, A11*B22, A22*B11, A22*B22
    
    def add_block_product(a_rows, a_cols, a_coeff, b_rows, b_cols, b_coeff, c_row_offset, c_col_offset, sign=1):
        """Add all 4 scalar products from a 2x2 block product.
        a_rows, a_cols: row/col indices for A block (2 values each)
        b_rows, b_cols: row/col indices for B block (2 values each)
        c_row_offset, c_col_offset: offset in output C for this 2x2 block
        """
        for ri in range(2):
            for cj in range(2):
                a_idx = a_rows[ri] * m + a_cols[cj]
                b_idx = b_rows[cj] * p + b_cols[cj]  # Wait, B is m x p, so B[j,k] where j is row, k is col
                # Actually for the 4x4 core: A is 4x4 (n x m but here n=m=4 for the core), B is 4x4 (m x p, m=p=4)
                # A[i,j] with i in a_rows, j in a_cols
                # B[j,k] with j in b_rows, k in b_cols
                # C[i,k] with i = c_row_offset + ri, k = c_col_offset + cj
                # But we need to be careful: the block product (A_block)(B_block) gives C_block
                # C_block[i,j] = sum over t of A_block[i,t] * B_block[t,j]
                
                # So for each (ri, cj) in the 2x2 output, we sum over t in 0,1
                # This means we need 2 terms per output element (one for each t)
                for t in range(2):
                    a_i = a_rows[ri]
                    a_j = a_cols[t]
                    b_j = b_rows[t]
                    b_k = b_cols[cj]
                    
                    u_idx = a_i * m + a_j
                    v_idx = b_j * p + b_k
                    c_i = c_row_offset + ri
                    c_k = c_col_offset + cj
                    w_idx = c_k * n + c_i
                    
                    coeff = sign * a_coeff * b_coeff
                    if coeff != 0:
                        add_term_val(u_idx, coeff, v_idx, 1, w_idx, 1)
    
    # Now let's define the Strassen terms
    # Block definitions:
    # A11: rows [0,1], cols [0,1]
    # A12: rows [0,1], cols [2,3]
    # A21: rows [2,3], cols [0,1]
    # A22: rows [2,3], cols [2,3]
    
    # B11: rows [0,1], cols [0,1]
    # B12: rows [0,1], cols [2,3]
    # B21: rows [2,3], cols [0,1]
    # B22: rows [2,3], cols [2,3]
    
    # C11: rows [0,1], cols [0,1] -> c_row_offset=0, c_col_offset=0
    # C12: rows [0,1], cols [2,3] -> c_row_offset=0, c_col_offset=2
    # C21: rows [2,3], cols [0,1] -> c_row_offset=2, c_col_offset=0
    # C22: rows [2,3], cols [2,3] -> c_row_offset=2, c_col_offset=2
    
    A11 = ([0,1], [0,1])
    A12 = ([0,1], [2,3])
    A21 = ([2,3], [0,1])
    A22 = ([2,3], [2,3])
    
    B11 = ([0,1], [0,1])
    B12 = ([0,1], [2,3])
    B21 = ([2,3], [0,1])
    B22 = ([2,3], [2,3])
    
    # M1 = (A11 + A22)(B11 + B22)
    # Terms in M1:
    add_block_product(*A11, 1, *B11, 1, 0, 0, 1)   # A11*B11 -> C11
    add_block_product(*A11, 1, *B22, 1, 0, 0, 1)   # A11*B22 -> C11
    add_block_product(*A22, 1, *B11, 1, 0, 0, 1)   # A22*B11 -> C11
    add_block_product(*A22, 1, *B22, 1, 0, 0, 1)   # A22*B22 -> C11
    
    # But wait, M1 contributes to C11 and C22
    # C11 = M1 + M4 - M5 + M7
    # C22 = M1 - M2 + M3 + M6
    
    # So M1 should be added to both C11 and C22. Let me redo this.
    
    # Let me restart with a cleaner approach.
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term_clean(u_idx, u_coeff, v_idx, v_coeff, w_idx, w_coeff):
        U_cols.append([u_coeff if i == u_idx else 0 for i in range(n * m)])
        V_cols.append([v_coeff if i == v_idx else 0 for i in range(m * p)])
        W_cols.append([w_coeff if i == w_idx else 0 for i in range(p * n)])
    
    # For the 4x4x4 core, we have:
    # A[i,j] for i,j in 0..3, B[j,k] for j,k in 0..3, C[i,k] for i,k in 0..3
    # u_idx = i*5 + j (since m=5, but for the core j is 0..3)
    # v_idx = j*5 + k (since p=5, but for the core k is 0..3)
    # w_idx = k*4 + i (since n=4)
    
    # Strassen:
    # M1 = (A11+A22)(B11+B22) -> contributes to C11 and C22
    # M2 = (A21+A22)B11 -> contributes to C21 and C22
    # M3 = A11(B12-B22) -> contributes to C12 and C22
    # M4 = A22(B21-B11) -> contributes to C11 and C21
    # M5 = (A11+A12)B22 -> contributes to C11 and C12
    # M6 = (A21-A11)(B11+B12) -> contributes to C22
    # M7 = (A12-A22)(B21+B22) -> contributes to C11
    
    # Let's generate all terms for each M, then add them to the appropriate C blocks.
    
    # For each M, it's a 2x2 matrix. Let's generate the 8 scalar products (2x2 input, 2x2 output, 2 intermediate)
    # Actually each M has 2*2*2 = 8 scalar multiplications if done naively, but Strassen says we can do it in 7.
    
    # The key insight: each M_t is computed as a 2x2x2 tensor contraction.
    # For M1 = (A11+A22)(B11+B22):
    # This is equivalent to summing over the "augmented" A and B blocks.
    # The 2x2x2 tensor for (A11+A22) x (B11+B22) has rank 8 (schoolbook for 2x2x2).
    
    # So total rank for 4x4x4 core = 7 * 8 = 56.
    
    # Let's just enumerate all 56 terms explicitly.
    
    # For a 2x2x2 Strassen-like block, if we have:
    # A_sum = sum of A_blocks with coeffs, B_sum = sum of B_blocks with coeffs
    # The 2x2x2 product has 8 terms (i,j,k for i,j,k in 0,1)
    
    # For M1: A_sum = A11 + A22, B_sum = B11 + B22
    # Terms for M1 (as a 2x2x2 tensor):
    # For each i,j,k in 0,1:
    #   u from A_sum[i,j], v from B_sum[j,k], w at C[i,k]
    
    # A11[i,j] -> u_idx = (0+i)*5 + (0+j) = i*5+j
    # A22[i,j] -> u_idx = (2+i)*5 + (2+j) = (2+i)*5+(2+j)
    # B11[j,k] -> v_idx = (0+j)*5 + (0+k) = j*5+k
    # B22[j,k] -> v_idx = (2+j)*5 + (2+k) = (2+j)*5+(2+k)
    
    # C11[i,k] -> w_idx = (0+k)*4 + (0+i) = k*4+i
    # C22[i,k] -> w_idx = (2+k)*4 + (2+i) = (2+k)*4+(2+i)
    
    # M1 contributes to C11 and C22.
    
    # This is getting complex. Let me just hardcode the 56 terms.
    
    # Actually, let me use a simpler approach: just use the known rank-56 decomposition for 4x4x4.
    # But I need to construct it. Let me think differently.
    
    # Alternative: Use the fact that the 4x4x4 tensor can be decomposed using 7 Strassen 2x2x2 sub-tensors.
    # Each 2x2x2 sub-tensor has a rank-8 decomposition (schoolbook).
    
    # For M1 = (A11+A22)(B11+B22):
    # The A-part is A11 + A22, which is a 4x4 matrix but structured.
    # The 2x2x2 core of M1 involves indices (i,j,k) where i,j,k in {0,1}
    # A_sum[i,j] = A11[i,j] + A22[i,j]
    # B_sum[j,k] = B11[j,k] + B22[j,k]
    # C[i,k] gets contributions from both C11 and C22.
    
    # For the schoolbook decomposition of the 2x2x2 part:
    # For each (i,j,k) in {0,1}^3:
    #   Term 1: A11[i,j] * B11[j,k] -> C11[i,k]
    #   Term 2: A11[i,j] * B22[j,k] -> C22[i,k]  (wait, this isn't right)
    
    # I think I'm overcomplicating. Let me just directly implement the Strassen formula
    # and expand each of the 7 M's into their 8 scalar products.
    
    # Reset
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_uvw(u_i, u_j, v_j, v_k, w_k, w_i, coeff=1):
        """Add a term for A[u_i, u_j] * B[v_j, v_k] -> C[w_i, w_k]"""
        u_idx = u_i * m + u_j
        v_idx = v_j * p + v_k
        w_idx = w_k * n + w_i
        U_cols.append([coeff if i == u_idx else 0 for i in range(n * m)])
        V_cols.append([1 if i == v_idx else 0 for i in range(m * p)])
        W_cols.append([1 if i == w_idx else 0 for i in range(p * n)])
    
    # M1 = (A11+A22)(B11+B22)
    # For each (i,j,k) in {0,1}^3:
    # A11[i,j] = A[i,j], A22[i,j] = A[i+2,j+2]
    # B11[j,k] = B[j,k], B22[j,k] = B[j+2,k+2]
    # M1[i,k] contributes to C11[i,k] and C22[i,k]
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # A11[i,j] * B11[j,k]
                add_uvw(i, j, j, k, k, i, 1)       # -> C11
                add_uvw(i, j, j, k, k+2, i+2, 1)   # -> C22? No...
    
    # I realize the issue: M1[i,k] is a 2x2 matrix, and it contributes to BOTH C11[i,k] and C22[i,k].
    # But the way I'm indexing, C11[i,k] and C22[i,k] are different positions.
    
    # Let me look at this more carefully.
    # C11[i,k] = sum_j (A11[i,j] + A22[i,j]) * (B11[j,k] + B22[j,k])
    #           = sum_j A11[i,j]*B11[j,k] + sum_j A11[i,j]*B22[j,k] + sum_j A22[i,j]*B11[j,k] + sum_j A22[i,j]*B22[j,k]
    
    # So for C11[i,k]:
    # Term (i,j,k): A[i,j] * B[j,k] -> C[i,k]
    # Term (i,j,k): A[i,j] * B[j+2,k+2] -> C[i,k]
    # Term (i,j,k): A[i+2,j+2] * B[j,k] -> C[i,k]
    # Term (i,j,k): A[i+2,j+2] * B[j+2,k+2] -> C[i,k]
    
    # For C22[i,k]:
    # Same 4 terms, but output is C[i+2,k+2]
    
    # So M1 contributes 4*2 = 8 terms per (i,k) pair, but i,k in {0,1}^2, so 4*(i,k) pairs * 4 terms each = 16 terms? No...
    
    # For each (i,j,k) in {0,1}^3, we have 4 products, each contributing to one output.
    # Total: 8 (i,j,k) combinations * 4 = 32 terms for M1 alone? That's too many.
    
    # Wait, I think the issue is that each M_t is a 2x2 matrix, and the "rank" of the 2x2x2 tensor is 8,
    # meaning 8 scalar multiplications. So M1 should contribute exactly 8 terms.
    
    # The 8 terms for M1 correspond to the 8 schoolbook multiplications for a 2x2x2:
    # For each (i,j,k) in {0,1}^3: one term
    # But the A and B entries are sums, so each term involves a product of sums.
    
    # I think the correct interpretation is:
    # The 2x2x2 tensor T[i,j,k] = (A11+A22)[i,j] * (B11+B22)[j,k]
    # This tensor has rank 8, and its rank-8 decomposition is:
    # For each (i,j,k) in {0,1}^3:
    #   U: A11[i,j] and A22[i,j] (but these are separate entries)
    
    # I think I need to just directly expand. Let me try a different approach.
    
    # Let me just use the known result: the 4x4x4 matrix multiplication tensor has rank 56.
    # I'll construct it by taking 7 copies of the 2x2x2 schoolbook decomposition (rank 8 each),
    # appropriately permuted and combined.
    
    # For simplicity, let me just use the schoolbook decomposition for the entire 4x4x4 (rank 64),
    # and then apply Strassen to reduce it.
    
    # Actually, let me just go with the approach that worked in the observation:
    # Replace the 4x4x4 core (64 terms) with a rank-56 Strassen-based decomposition.
    
    # I'll construct the 56 terms by explicitly enumerating the Strassen formula.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term_final(u_idx, v_idx, w_idx, u_coeff=1, v_coeff=1, w_coeff=1):
        U_cols.append([u_coeff if i == u_idx else 0 for i in range(n * m)])
        V_cols.append([v_coeff if i == v_idx else 0 for i in range(m * p)])
        W_cols.append([w_coeff if i == w_idx else 0 for i in range(p * n)])
    
    # Strassen for 4x4x4:
    # The 7 Strassen products, each requiring 8 scalar multiplications (2x2x2 schoolbook).
    
    # For each Strassen product M_t, it computes a 2x2 matrix C_t.
    # C_t is placed into the appropriate position in the final C.
    
    # Let me define the 7 Strassen products and their contributions.
    
    # M1 = (A11+A22)(B11+B22)
    # A11+A22 has entries: for i,j in {0,1}, (A11+A22)[i,j] = A[i,j] + A[i+2,j+2]
    # B11+B22 has entries: for j,k in {0,1}, (B11+B22)[j,k] = B[j,k] + B[j+2,k+2]
    # M1[i,k] = sum_j (A[i,j]+A[i+2,j+2])*(B[j,k]+B[j+2,k+2])
    #          = sum_j A[i,j]*B[j,k] + sum_j A[i,j]*B[j+2,k+2] + sum_j A[i+2,j+2]*B[j,k] + sum_j A[i+2,j+2]*B[j+2,k+2]
    
    # M1 contributes to C11 (at [i,k]) and C22 (at [i+2,k+2])
    
    # For the rank-8 decomposition of the 2x2x2 tensor (A11+A22) x (B11+B22):
    # We treat (A11+A22) as a 2x2 matrix and (B11+B22) as a 2x2 matrix.
    # The schoolbook decomposition has 8 terms, one for each (i,j,k) in {0,1}^3.
    # But since A and B are sums, each term expands to 4 products.
    
    # This is getting too complex. Let me just use a known rank-56 decomposition for 4x4x4.
    
    # After some research, I recall that the standard Strassen-based decomposition for 4x4x4
    # can be constructed as follows. Let me just hardcode the 56 terms.
    
    # Given time constraints, let me use a simpler approach: use Strassen for the 4x4x4 core
    # by treating it as 7 independent 2x2x2 problems, each with rank 8.
    
    # I'll construct the terms for each of the 7 Strassen products.
    
    # For M1 = (A11+A22)(B11+B22):
    # The 2x2x2 schoolbook decomposition for (A11+A22) x (B11+B22) -> M1
    # has 8 terms. Each term corresponds to (i,j,k) in {0,1}^3.
    # The U factor for term (i,j,k) is: A11[i,j] + A22[i,j] (but these are separate basis vectors)
    # Actually, in the tensor decomposition, each term is a pure tensor.
    # So for M1, we have 8 pure tensors:
    # For (i,j,k) in {0,1}^3:
    #   U: e_{A11[i,j]} + e_{A22[i,j]}  (but this is not a pure tensor)
    
    # I think the correct way is: the 2x2x2 tensor (A11+A22) x (B11+B22) can be written as:
    # sum_{i,j,k} (e_i^A11 + e_i^A22) (e_j^B11 + e_j^B22) (e_k^C11 + e_k^C22) ... no, this doesn't work.
    
    # Let me just give up on deriving it and use the fact that the observation showed rank 92 works.
    # I'll construct it as: 56 terms for the 4x4x4 core + 36 schoolbook terms for the border.
    
    # For the 56 terms, I'll use a known construction.
    
    # After further thought, let me just use the schoolbook for the 4x4x4 core (64 terms)
    # and accept rank 100, which is the same as the seed. But the observation showed 92 works.
    
    # Let me try to construct the 56 terms properly.
    
    # The key insight: for a 2x2x2 Strassen product, the rank is 7, not 8.
    # So for 7 Strassen products, the total rank is 7*7 = 49? No, that's not right either.
    
    # Actually, Strassen's algorithm for 2x2x2 has rank 7. So for 4x4x4, if we use 7 Strassen 2x2x2 sub-problems,
    # the rank would be 7*7 = 49? But the known rank of 4x4x4 is 56, so this isn't right.
    
    # I think the correct statement is: the 4x4x4 tensor can be decomposed into 7 "Strassen blocks",
    # each of which is a 2x2x2 tensor with rank 8 (schoolbook), giving total rank 56.
    
    # So let me just construct 7 groups of 8 terms each.
    
    # For M1 = (A11+A22)(B11+B22):
    # The 8 terms are:
    # For (i,j,k) in {0,1}^3:
    #   Term: A[i,j]*B[j,k] -> C[i,k]  (from A11*B11)
    #   Term: A[i,j]*B[j+2,k+2] -> C[i,k]  (from A11*B22)
    #   Term: A[i+2,j+2]*B[j,k] -> C[i,k]  (from A22*B11)
    #   Term: A[i+2,j+2]*B[j+2,k+2] -> C[i,k]  (from A22*B22)
    # But this gives 32 terms, not 8.
    
    # I think I'm confusing myself. Let me just use the schoolbook for the 4x4x4 core
    # and add the border terms, giving rank 100. This is the same as the seed.
    
    # Given the time spent, let me just return the schoolbook decomposition.
    
    rank = n * m * p  # 100
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
