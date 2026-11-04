"""
Program to find a rank-minimal exact tensor decomposition for 5x5x5 matrix multiplication.

Strategy: Use the 4x4 + 1x1 split.
- The 4x4x4 core is decomposed using Strassen's algorithm recursively (rank 49).
- The cross terms involving the 5th row/col are handled with schoolbook multiplication.
- The 1x1x1 corner is a single multiplication.

This yields a total rank of 49 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 110, which is less than 125.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 5x5x5 matrix multiplication.
    
    Uses block partitioning: indices 0-3 form the 4x4 core, index 4 is the boundary.
    The 4x4 core uses Strassen (rank 49).
    Cross terms use schoolbook.
    Total rank: 110.
    """
    n, m, p = 5, 5, 5
    
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    # We'll build the decomposition.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The tensor is sum_r U[:,r] * V[:,r] * W[:,r]
    # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
    
    # Pre-allocate for max expected rank (110)
    # We will fill it column by column.
    
    rank_target = 110
    U = np.zeros((n * m, rank_target), dtype=float)
    V = np.zeros((m * p, rank_target), dtype=float)
    W = np.zeros((p * n, rank_target), dtype=float)
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices):
        """
        Add a rank-1 term.
        u_indices: list of (index, coeff) for U
        v_indices: list of (index, coeff) for V
        w_indices: list of (index, coeff) for W
        """
        nonlocal col
        if col >= rank_target:
            return
        for idx, val in u_indices:
            U[idx, col] += val
        for idx, val in v_indices:
            V[idx, col] += val
        for idx, val in w_indices:
            W[idx, col] += val
        col += 1
    
    def add_schoolbook_block(i1, i2, j1, j2, k1, k2):
        """
        Add schoolbook terms for A[i1:i2, j1:j2] * B[j1:j2, k1:k2] -> C[i1:i2, k1:k2]
        """
        for i in range(i1, i2):
            for j in range(j1, j2):
                for k in range(k1, k2):
                    add_term([(i*m+j, 1.0)], [(j*p+k, 1.0)], [(k*n+i, 1.0)])
    
    def add_strassen_2x2(i1, j1, k1):
        """
        Add 7 Strassen terms for 2x2 block multiplication.
        A is at rows i1, i1+1 and cols j1, j1+1.
        B is at rows j1, j1+1 and cols k1, k1+1.
        C is at rows i1, i1+1 and cols k1, k1+1.
        """
        i2 = i1 + 1
        j2 = j1 + 1
        k2 = k1 + 1
        
        # M1 = (A00+A11)(B00+B11)
        # u: (i1,j1)=1, (i2,j2)=1
        # v: (j1,k1)=1, (j2,k2)=1
        # w: (k1,i1)=1, (k2,i2)=1
        add_term(
            [(i1*m+j1, 1.0), (i2*m+j2, 1.0)],
            [(j1*p+k1, 1.0), (j2*p+k2, 1.0)],
            [(k1*n+i1, 1.0), (k2*n+i2, 1.0)]
        )
        
        # M2 = (A10+A11)B00
        # u: (i2,j1)=1, (i2,j2)=1
        # v: (j1,k1)=1
        # w: (k1,i2)=1
        add_term(
            [(i2*m+j1, 1.0), (i2*m+j2, 1.0)],
            [(j1*p+k1, 1.0)],
            [(k1*n+i2, 1.0)]
        )
        
        # M3 = A00(B01-B11)
        # u: (i1,j1)=1
        # v: (j1,k2)=1, (j2,k2)=-1
        # w: (k2,i1)=1
        add_term(
            [(i1*m+j1, 1.0)],
            [(j1*p+k2, 1.0), (j2*p+k2, -1.0)],
            [(k2*n+i1, 1.0)]
        )
        
        # M4 = A11(B10-B00)
        # u: (i2,j2)=1
        # v: (j2,k1)=1, (j1,k1)=-1
        # w: (k1,i2)=1, (k1,i1)=-1
        add_term(
            [(i2*m+j2, 1.0)],
            [(j2*p+k1, 1.0), (j1*p+k1, -1.0)],
            [(k1*n+i2, 1.0), (k1*n+i1, -1.0)]
        )
        
        # M5 = (A00+A01)B11
        # u: (i1,j1)=1, (i1,j2)=1
        # v: (j2,k2)=1
        # w: (k2,i1)=1, (k1,i1)=-1
        add_term(
            [(i1*m+j1, 1.0), (i1*m+j2, 1.0)],
            [(j2*p+k2, 1.0)],
            [(k2*n+i1, 1.0), (k1*n+i1, -1.0)]
        )
        
        # M6 = (A10-A00)(B00+B01)
        # u: (i2,j1)=1, (i1,j1)=-1
        # v: (j1,k1)=1, (j1,k2)=1
        # w: (k2,i2)=1
        add_term(
            [(i2*m+j1, 1.0), (i1*m+j1, -1.0)],
            [(j1*p+k1, 1.0), (j1*p+k2, 1.0)],
            [(k2*n+i2, 1.0)]
        )
        
        # M7 = (A01-A11)(B10+B11)
        # u: (i1,j2)=1, (i2,j2)=-1
        # v: (j2,k1)=1, (j2,k2)=1
        # w: (k1,i1)=1, (k2,i2)=-1
        add_term(
            [(i1*m+j2, 1.0), (i2*m+j2, -1.0)],
            [(j2*p+k1, 1.0), (j2*p+k2, 1.0)],
            [(k1*n+i1, 1.0), (k2*n+i2, -1.0)]
        )
    
    # --- Part 1: 4x4x4 Core (indices 0-3) ---
    # We apply Strassen at the 4x4 level by treating it as 2x2 blocks of 2x2 matrices.
    # Top-level Strassen on 2x2 blocks:
    # Blocks:
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # B00: rows 0-1, cols 0-1
    # B01: rows 0-1, cols 2-3
    # B10: rows 2-3, cols 0-1
    # B11: rows 2-3, cols 2-3
    
    # C00: rows 0-1, cols 0-1
    # C01: rows 0-1, cols 2-3
    # C10: rows 2-3, cols 0-1
    # C11: rows 2-3, cols 2-3
    
    # Strassen M1 = (A00+A11)(B00+B11)
    # This is a product of two 2x2 matrices (conceptually).
    # But in the tensor decomposition, we need to express this as 7 rank-1 terms.
    # The "A" part for M1 is the sum of A00 and A11.
    # The "B" part for M1 is the sum of B00 and B11.
    # The "C" part for M1 contributes to C00 and C11.
    
    # How do we add Strassen terms for this?
    # We can't just call add_strassen_2x2 because that assumes contiguous indices.
    # Instead, we manually construct the 7 terms for this specific "sum-of-blocks" multiplication.
    
    # Let's define a helper to add Strassen terms where A and B are sums of blocks.
    # Actually, it's easier to just expand the 7 Strassen formulas for the specific indices.
    
    # Let's define the indices for the 4 sub-blocks of A, B, C in the 4x4 core.
    # A_blocks:
    # A00: i in {0,1}, j in {0,1}
    # A01: i in {0,1}, j in {2,3}
    # A10: i in {2,3}, j in {0,1}
    # A11: i in {2,3}, j in {2,3}
    
    # B_blocks:
    # B00: j in {0,1}, k in {0,1}
    # B01: j in {0,1}, k in {2,3}
    # B10: j in {2,3}, k in {0,1}
    # B11: j in {2,3}, k in {2,3}
    
    # C_blocks:
    # C00: i in {0,1}, k in {0,1}
    # C01: i in {0,1}, k in {2,3}
    # C10: i in {2,3}, k in {0,1}
    # C11: i in {2,3}, k in {2,3}
    
    # We will implement the 7 top-level Strassen multiplications.
    # Each is a 2x2 * 2x2 product.
    # We can reuse the add_strassen_2x2 logic if we map the logical 2x2 indices to physical 4x4 indices.
    
    # Let's create a more flexible Strassen function that takes index lists.
    
    def add_strassen_2x2_flex(i_list, j_list, k_list):
        """
        Add 7 Strassen terms for a 2x2 multiplication where:
        A is formed by rows i_list[0], i_list[1] and cols j_list[0], j_list[1]
        B is formed by rows j_list[0], j_list[1] and cols k_list[0], k_list[1]
        C is formed by rows i_list[0], i_list[1] and cols k_list[0], k_list[1]
        
        Note: This assumes the "2x2" structure is preserved in the indices, 
        i.e., A[0,0] is at (i_list[0], j_list[0]), etc.
        """
        i1, i2 = i_list
        j1, j2 = j_list
        k1, k2 = k_list
        
        # M1 = (A00+A11)(B00+B11)
        add_term(
            [(i1*m+j1, 1.0), (i2*m+j2, 1.0)],
            [(j1*p+k1, 1.0), (j2*p+k2, 1.0)],
            [(k1*n+i1, 1.0), (k2*n+i2, 1.0)]
        )
        
        # M2 = (A10+A11)B00
        add_term(
            [(i2*m+j1, 1.0), (i2*m+j2, 1.0)],
            [(j1*p+k1, 1.0)],
            [(k1*n+i2, 1.0)]
        )
        
        # M3 = A00(B01-B11)
        add_term(
            [(i1*m+j1, 1.0)],
            [(j1*p+k2, 1.0), (j2*p+k2, -1.0)],
            [(k2*n+i1, 1.0)]
        )
        
        # M4 = A11(B10-B00)
        add_term(
            [(i2*m+j2, 1.0)],
            [(j2*p+k1, 1.0), (j1*p+k1, -1.0)],
            [(k1*n+i2, 1.0), (k1*n+i1, -1.0)]
        )
        
        # M5 = (A00+A01)B11
        add_term(
            [(i1*m+j1, 1.0), (i1*m+j2, 1.0)],
            [(j2*p+k2, 1.0)],
            [(k2*n+i1, 1.0), (k1*n+i1, -1.0)]
        )
        
        # M6 = (A10-A00)(B00+B01)
        add_term(
            [(i2*m+j1, 1.0), (i1*m+j1, -1.0)],
            [(j1*p+k1, 1.0), (j1*p+k2, 1.0)],
            [(k2*n+i2, 1.0)]
        )
        
        # M7 = (A01-A11)(B10+B11)
        add_term(
            [(i1*m+j2, 1.0), (i2*m+j2, -1.0)],
            [(j2*p+k1, 1.0), (j2*p+k2, 1.0)],
            [(k1*n+i1, 1.0), (k2*n+i2, -1.0)]
        )
    
    # Now, the 4x4 Strassen multiplications:
    
    # M1_top = (A00+A11)(B00+B11)
    # A part: A00 (rows 0,1; cols 0,1) + A11 (rows 2,3; cols 2,3)
    # This is NOT a simple 2x2 block in the 4x4 matrix. It's a sum of two disjoint 2x2 blocks.
    # The "A" matrix for this multiplication has non-zero entries only at:
    # (0,0), (0,1), (1,0), (1,1) from A00
    # (2,2), (2,3), (3,2), (3,3) from A11
    # This is a 4x4 matrix, not 2x2.
    # So we cannot use add_strassen_2x2_flex directly.
    
    # We need to decompose (A00+A11)(B00+B11) into rank-1 terms.
    # Since A00 and A11 are disjoint, and B00 and B11 are disjoint:
    # (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # Each of these is a 2x2 * 2x2 product.
    # We can compute each with Strassen (7 terms).
    # Total 28 terms for M1_top.
    
    # This is the key: we don't save any rank by doing Strassen at the top level if we have to expand the sums.
    # The rank of 4x4x4 is 49, which is 7*7.
    # If we expand M1_top into 4 sub-products, each of rank 7, that's 28.
    # There are 7 top-level M's.
    # M1_top: 4 sub-products * 7 = 28
    # M2_top = (A10+A11)B00 = A10B00 + A11B00. Two sub-products * 7 = 14.
    # M3_top = A00(B01-B11) = A00B01 - A00B11. Two sub-products * 7 = 14.
    # M4_top = A11(B10-B00) = A11B10 - A11B00. Two sub-products * 7 = 14.
    # M5_top = (A00+A01)B11 = A00B11 + A01B11. Two sub-products * 7 = 14.
    # M6_top = (A10-A00)(B00+B01) = (A10B00 + A10B01 - A00B00 - A00B01). Four sub-products * 7 = 28.
    # M7_top = (A01-A11)(B10+B11) = (A01B10 + A01B11 - A11B10 - A11B11). Four sub-products * 7 = 28.
    
    # Total: 28 + 14 + 14 + 14 + 14 + 28 + 28 = 140 terms.
    # This is worse than schoolbook (64) and worse than the direct 49 rank decomposition!
    
    # Why? Because the 49 rank decomposition of 4x4x4 does NOT come from expanding the top-level Strassen sums.
    # It comes from a different structure.
    
    # Actually, the rank of 4x4x4 is 49 because 4x4x4 is isomorphic to a sub-tensor of a larger problem?
    # No, the rank of n x n x n matmul is known to be at most 7^{log2(n)}.
    # For n=4, 7^2 = 49.
    # The decomposition is recursive.
    # T_4 = sum of 7 terms, each of which is a "scaled and permuted" T_2.
    # But T_2 is not a tensor in the same space.
    
    # The correct recursive decomposition:
    # T_4 = sum_{r=1}^7 u_r ⊗ v_r ⊗ w_r
    # where u_r, v_r, w_r are vectors in R^{16}.
    # The entries of u_r are 0, 1, -1.
    # Specifically, for each of the 7 Strassen multiplications, we have 7 sub-multiplications.
    # So there are 49 terms total.
    # Each term corresponds to a specific (i,j,k) triple in the 4x4 space, with coefficients.
    
    # Let's generate the 49 terms for 4x4x4 explicitly.
    # We can do this by iterating over the 7 top-level M's and the 7 bottom-level M's.
    
    # Define the 7 top-level patterns (coefficients for A, B, C blocks)
    # and the 7 bottom-level patterns (coefficients for 2x2 elements).
    
    # This is complex to code manually. Let's use a simpler approach:
    # Just use schoolbook for the 4x4 core? Rank 64.
    # Total rank: 64 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 125. No improvement.
    
    # We MUST use the 49 rank decomposition for 4x4.
    
    # Let's hardcode the 49 terms for 4x4x4.
    # Or better, generate them programmatically.
    
    # The 7 Strassen multiplications for 2x2 are:
    # 1: (A00+A11)(B00+B11) -> C00, C11
    # 2: (A10+A11)B00 -> C10
    # 3: A00(B01-B11) -> C01
    # 4: A11(B10-B00) -> C10, C00
    # 5: (A00+A01)B11 -> C01, C00
    # 6: (A10-A00)(B00+B01) -> C11
    # 7: (A01-A11)(B10+B11) -> C00, C11
    
    # For 4x4, we apply this to the 2x2 blocks.
    # Let's define the block indices:
    # A_blocks: 00, 01, 10, 11
    # B_blocks: 00, 01, 10, 11
    # C_blocks: 00, 01, 10, 11
    
    # Top-level Strassen M1: (A00+A11)(B00+B11)
    # This expands to 4 block products: A00B00, A00B11, A11B00, A11B11.
    # Each block product is a 2x2*2x2 multiplication.
    # We decompose each into 7 rank-1 terms.
    # So M1 contributes 28 terms.
    
    # Wait, I already established this gives 140 terms.
    # Is the rank of 4x4x4 really 49?
    # Yes. But the decomposition is not "7 top-level M's, each expanded to 4 block products".
    # The decomposition is a single set of 49 terms.
    
    # The terms are:
    # For each of the 7 top-level Strassen multiplications (t=1..7),
    # and for each of the 7 bottom-level Strassen multiplications (b=1..7),
    # there is one rank-1 term.
    # The term is determined by the combination of the top-level pattern and the bottom-level pattern.
    
    # Let's define the top-level patterns as matrices U_t, V_t, W_t of size 2x2 (for blocks).
    # And the bottom-level patterns as matrices u_b, v_b, w_b of size 2x2 (for elements).
    
    # Then the term for (t,b) has:
    # U component: U_t[i_block, j_block] * u_b[i_elem, j_elem]
    # V component: V_t[j_block, k_block] * v_b[j_elem, k_elem]
    # W component: W_t[k_block, i_block] * w_b[k_elem, i_elem]
    
    # This seems correct.
    
    # Let's implement this.
    
    # Top-level Strassen patterns (2x2 matrices for blocks):
    # M1: U = [[1,0],[0,1]], V = [[1,0],[0,1]], W = [[1,0],[0,1]] (diagonal)
    # Wait, M1 = (A00+A11)(B00+B11).
    # U_M1 = [[1,0],[0,1]]? No.
    # A00 is at (0,0), A11 is at (1,1).
    # So U_M1 has 1 at (0,0) and 1 at (1,1).
    # V_M1 has 1 at (0,0) and 1 at (1,1).
    # W_M1 has 1 at (0,0) and 1 at (1,1).
    
    # Let's list the 7 patterns:
    # 1: U=[[1,0],[0,1]], V=[[1,0],[0,1]], W=[[1,0],[0,1]]
    # 2: U=[[0,0],[1,1]], V=[[1,0],[0,0]], W=[[0,0],[1,0]]
    # 3: U=[[1,0],[0,0]], V=[[0,1],[0,-1]], W=[[0,1],[0,0]]
    # 4: U=[[0,0],[0,1]], V=[[0,-1],[1,0]], W=[[-1,0],[1,0]]  (Note: W has 2 non-zeros)
    # 5: U=[[1,1],[0,0]], V=[[0,0],[0,1]], W=[[-1,1],[0,0]]  (Note: W has 2 non-zeros)
    # 6: U=[[-1,0],[1,0]], V=[[1,1],[0,0]], W=[[0,0],[0,1]]
    # 7: U=[[0,1],[0,-1]], V=[[0,1],[0,1]], W=[[1,0],[0,-1]]
    
    # Let's verify M4: A11(B10-B00)
    # U_M4: 1 at (1,1). So U=[[0,0],[0,1]].
    # V_M4: B10 is at (1,0), B00 is at (0,0). So V=[[0,-1],[1,0]].
    # W_M4: Contributes to C10 (1,0) and C00 (0,0).
    # C10 gets +1, C00 gets -1.
    # So W has 1 at (1,0) and -1 at (0,0). W=[[-1,0],[1,0]].
    # Correct.
    
    # Now, the bottom-level patterns (2x2 matrices for elements):
    # These are the same as the top-level patterns, but applied to 2x2 elements.
    # Let's call them u_b, v_b, w_b for b=1..7.
    # They are the same matrices as above.
    
    # So, for each top-level pattern t and bottom-level pattern b:
    # We create a rank-1 term.
    # The U vector for this term has non-zero entries at:
    # (i_block * 2 + i_elem, j_block * 2 + j_elem)
    # with value U_t[i_block, j_block] * u_b[i_elem, j_elem]
    
    # Similarly for V and W.
    
    # Let's implement this.
    
    def add_4x4_strassen():
        nonlocal col
        # Define the 7 patterns
        patterns = [
            # t=1
            ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),
            # t=2
            ([[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[1,0]]),
            # t=3
            ([[1,0],[0,0]], [[0,1],[0,-1]], [[0,1],[0,0]]),
            # t=4
            ([[0,0],[0,1]], [[0,-1],[1,0]], [[-1,0],[1,0]]),
            # t=5
            ([[1,1],[0,0]], [[0,0],[0,1]], [[-1,1],[0,0]]),
            # t=6
            ([[ -1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]),
            # t=7
            ([[0,1],[0,-1]], [[0,1],[0,1]], [[1,0],[0,-1]])
        ]
        
        # The bottom-level patterns are the same
        bottom_patterns = patterns
        
        for t in range(7):
            U_t, V_t, W_t = patterns[t]
            for b in range(7):
                u_b, v_b, w_b = bottom_patterns[b]
                
                u_indices = []
                v_indices = []
                w_indices = []
                
                for ib in range(2):
                    for jb in range(2):
                        coeff_t_u = U_t[ib][jb]
                        if coeff_t_u == 0:
                            continue
                        for ie in range(2):
                            for je in range(2):
                                coeff_b_u = u_b[ie][je]
                                if coeff_b_u == 0:
                                    continue
                                i = ib * 2 + ie
                                j = jb * 2 + je
                                val = coeff_t_u * coeff_b_u
                                u_indices.append((i*m + j, val))
                
                for jb in range(2):
                    for kb in range(2):
                        coeff_t_v = V_t[jb][kb]
                        if coeff_t_v == 0:
                            continue
                        for je in range(2):
                            for ke in range(2):
                                coeff_b_v = v_b[je][ke]
                                if coeff_b_v == 0:
                                    continue
                                j = jb * 2 + je
                                k = kb * 2 + ke
                                val = coeff_t_v * coeff_b_v
                                v_indices.append((j*p + k, val))
                
                for kb in range(2):
                    for ib in range(2):
                        coeff_t_w = W_t[kb][ib]
                        if coeff_t_w == 0:
                            continue
                        for ke in range(2):
                            for ie in range(2):
                                coeff_b_w = w_b[ke][ie]
                                if coeff_b_w == 0:
                                    continue
                                k = kb * 2 + ke
                                i = ib * 2 + ie
                                val = coeff_t_w * coeff_b_w
                                w_indices.append((k*n + i, val))
                
                add_term(u_indices, v_indices, w_indices)
    
    # Call the 4x4 Strassen decomposition
    add_4x4_strassen()
    
    # --- Part 2: Cross Terms ---
    # Indices 0-3 are the 4x4 core, index 4 is the boundary.
    
    # a: A[0:4, 4] (4x1)
    # b: A[4, 0:4] (1x4)
    # c: A[4, 4] (1x1)
    
    # d: B[0:4, 4] (4x1)
    # e: B[4, 0:4] (1x4)
    # f: B[4, 4] (1x1)
    
    # C = A*B
    # C[0:4, 0:4] = A[0:4, 0:4]B[0:4, 0:4] + A[0:4, 4]B[4, 0:4]
    # C[0:4, 4] = A[0:4, 0:4]B[0:4, 4] + A[0:4, 4]B[4, 4]
    # C[4, 0:4] = A[4, 0:4]B[0:4, 0:4] + A[4, 4]B[4, 0:4]
    # C[4, 4] = A[4, 0:4]B[0:4, 4] + A[4, 4]B[4, 4]
    
    # We already did A[0:4, 0:4]B[0:4, 0:4] (49 terms).
    
    # Now add the cross terms:
    
    # 1. A[0:4, 4]B[4, 0:4] -> C[0:4, 0:4]
    # a * e
    # a is 4x1, e is 1x4.
    # This is an outer product. Rank 4*1*4 = 16.
    add_schoolbook_block(0, 4, 4, 5, 0, 4)
    
    # 2. A[0:4, 0:4]B[0:4, 4] -> C[0:4, 4]
    # A_core * d
    # A_core is 4x4, d is 4x1.
    # Rank 4*4*1 = 16.
    add_schoolbook_block(0, 4, 0, 4, 4, 5)
    
    # 3. A[0:4, 4]B[4, 4] -> C[0:4, 4]
    # a * f
    # a is 4x1, f is 1x1.
    # Rank 4*1*1 = 4.
    add_schoolbook_block(0, 4, 4, 5, 4, 5)
    
    # 4. A[4, 0:4]B[0:4, 0:4] -> C[4, 0:4]
    # b * A_core
    # b is 1x4, A_core is 4x4.
    # Rank 1*4*4 = 16.
    add_schoolbook_block(4, 5, 0, 4, 0, 4)
    
    # 5. A[4, 4]B[4, 0:4] -> C[4, 0:4]
    # c * e
    # c is 1x1, e is 1x4.
    # Rank 1*1*4 = 4.
    add_schoolbook_block(4, 5, 4, 5, 0, 4)
    
    # 6. A[4, 0:4]B[0:4, 4] -> C[4, 4]
    # b * d
    # b is 1x4, d is 4x1.
    # Rank 1*4*1 = 4.
    add_schoolbook_block(4, 5, 0, 4, 4, 5)
    
    # 7. A[4, 4]B[4, 4] -> C[4, 4]
    # c * f
    # c is 1x1, f is 1x1.
    # Rank 1.
    add_schoolbook_block(4, 5, 4, 5, 4, 5)
    
    # Total rank: 49 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 110.
    
    # Check if we are within the deadline
    if time.time() > deadline:
        return {"U": U[:, :col].tolist(), "V": V[:, :col].tolist(), "W": W[:, :col].tolist()}
    
    return {"U": U[:, :col].tolist(), "V": V[:, :col].tolist(), "W": W[:, :col].tolist()}
# EVOLVE-BLOCK-END
