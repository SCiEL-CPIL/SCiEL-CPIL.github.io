import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """Construct a rank-minimal (or lower-rank) tensor decomposition for 4x4 * 4x7 matrix multiplication."""
    n, m, p = 4, 4, 7
    deadline = time.time() + 110  # Leave 10s safety margin
    
    def verify_full(U, V, W, n, m, p):
        """Verify that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U.shape[1]):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        """Schoolbook decomposition: rank = n*m*p = 112."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def build_strassen_4x4x4():
        """
        Build a rank-49 decomposition for 4x4 * 4x4 using two levels of Strassen.
        Level 1: 4x4 = 2x2 blocks, 7 Strassen products of 2x2*2x2
        Level 2: Each 2x2*2x2 uses 7 scalar products
        Total: 7*7 = 49
        """
        # Strassen 2x2x2 coefficients
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C00 = M1 - M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 - M6
        
        # For 4x4x4, we split into 2x2 blocks:
        # A = [[A00, A01], [A10, A11]] where each Ai* is 2x2
        # B = [[B00, B01], [B10, B11]]
        # C = [[C00, C01], [C10, C11]]
        
        # Strassen at block level:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 - M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 - M6
        
        # Each Mi is a product of two 2x2 matrices (which may be sums of blocks)
        # We compute each Mi using the 2x2x2 Strassen (7 scalar products)
        
        # Block index mapping for 2x2 blocks within 4x4:
        # Block (br, bc) has local indices (0,1,2,3) mapping to global:
        # local 0 -> (br*2+0)*4 + (bc*2+0) = br*8 + bc*2
        # local 1 -> (br*2+0)*4 + (bc*2+1) = br*8 + bc*2 + 1
        # local 2 -> (br*2+1)*4 + (bc*2+0) = br*8 + bc*2 + 4
        # local 3 -> (br*2+1)*4 + (bc*2+1) = br*8 + bc*2 + 5
        
        def get_block_indices(br, bc):
            """Return the 4 global indices for block (br, bc) in a 4x4 matrix."""
            base_r = br * 2
            base_c = bc * 2
            return [
                (base_r + 0) * 4 + (base_c + 0),  # 0
                (base_r + 0) * 4 + (base_c + 1),  # 1
                (base_r + 1) * 4 + (base_c + 0),  # 2
                (base_r + 1) * 4 + (base_c + 1),  # 3
            ]
        
        # For each of the 7 Strassen products at block level, we need to:
        # 1. Identify which A-blocks and B-blocks are involved (with coefficients)
        # 2. For each pair of A-block and B-block, compute the 2x2*2x2 product using Strassen
        
        # A_coeff[r] = [coeff for A00, A01, A10, A11]
        A_coeff = [
            [1, 0, 0, 1],   # M1: A00 + A11
            [0, 0, 1, 1],   # M2: A10 + A11
            [1, 0, 0, 0],   # M3: A00
            [0, 0, 0, 1],   # M4: A11
            [1, 1, 0, 0],   # M5: A00 + A01
            [-1, 0, 1, 0],  # M6: -A00 + A10
            [0, 1, 0, -1],  # M7: A01 - A11
        ]
        
        # B_coeff[r] = [coeff for B00, B01, B10, B11]
        B_coeff = [
            [1, 0, 0, 1],   # M1: B00 + B11
            [1, 0, 0, 0],   # M2: B00
            [0, 1, 0, -1],  # M3: B01 - B11
            [0, 0, 1, -1],  # M4: B10 - B00
            [0, 0, 0, 1],   # M5: B11
            [1, 1, 0, 0],   # M6: B00 + B01
            [0, 0, 1, 1],   # M7: B10 + B11
        ]
        
        # C_out_coeff[c_block][r] = coefficient of Mr in C_block
        # C00 = M1 - M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 - M6
        C_out_coeff = [
            [1, 0, 0, -1, -1, 0, 1],   # C00: M1 - M4 - M5 + M7
            [0, 0, 1, 0, 1, 0, 0],     # C01: M3 + M5
            [0, 1, 0, 1, 0, 0, 0],     # C10: M2 + M4
            [1, -1, 1, 0, 0, -1, 0],   # C11: M1 - M2 + M3 - M6
        ]
        
        # Now build the decomposition.
        # For each Strassen product r (0..6), we have:
        #   A_sum = sum_ab A_coeff[r][ab] * A_ab
        #   B_sum = sum_bb B_coeff[r][bb] * B_bb
        #   Mi = A_sum * B_sum
        # 
        # Mi is a 2x2 * 2x2 product, computed via Strassen (7 scalar products)
        # 
        # For each pair (ab, bb) with nonzero coefficients, we compute A_ab * B_bb
        # using the 2x2x2 Strassen, scaled by A_coeff[r][ab] * B_coeff[r][bb]
        
        total_rank = 49
        U = np.zeros((16, total_rank))
        V = np.zeros((16, total_rank))
        W = np.zeros((16, total_rank))
        
        # 2x2x2 Strassen coefficients
        # For 2x2*2x2: A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # M1 = (a+d)(e+h): U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]
        # M2 = (c+d)e:     U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,-1]
        # M3 = a(f-h):     U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,1]
        # M4 = d(g-e):     U=[0,0,0,1], V=[0,0,1,-1], W=[0,-1,1,0]
        # M5 = (a+b)h:     U=[1,1,0,0], V=[0,0,0,1], W=[-1,1,0,0]
        # M6 = (c-a)(e+f): U=[0,-1,1,0], V=[1,1,0,0], W=[0,0,0,-1]
        # M7 = (b-d)(g+h): U=[0,1,0,-1], V=[0,0,1,1], W=[1,0,0,0]
        
        # 2x2x2 Strassen output coefficients:
        # C00 = M1 - M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 - M6
        S2_C_out = [
            [1, 0, 0, -1, -1, 0, 1],   # C00
            [0, 0, 1, 0, 1, 0, 0],     # C01
            [0, 1, 0, 1, 0, 0, 0],     # C10
            [1, -1, 1, 0, 0, -1, 0],   # C11
        ]
        
        # 2x2x2 Strassen input coefficients:
        # For each Mr (0..6), the A and B inputs:
        # M1: A=[1,0,0,1], B=[1,0,0,1]
        # M2: A=[0,0,1,1], B=[1,0,0,0]
        # M3: A=[1,0,0,0], B=[0,1,0,-1]
        # M4: A=[0,0,0,1], B=[0,0,1,-1]
        # M5: A=[1,1,0,0], B=[0,0,0,1]
        # M6: A=[0,-1,1,0], B=[1,1,0,0]
        # M7: A=[0,1,0,-1], B=[0,0,1,1]
        S2_A_coeff = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, -1, 1, 0],
            [0, 1, 0, -1],
        ]
        S2_B_coeff = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [0, 0, 1, -1],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ]
        
        # For each block-level Strassen product r (0..6):
        col = 0
        for r in range(7):
            a_coeff = A_coeff[r]
            b_coeff = B_coeff[r]
            
            # For each pair of A-block (ab) and B-block (bb) with nonzero coefficients:
            for ab in range(4):
                if a_coeff[ab] == 0:
                    continue
                ab_r, ab_c = divmod(ab, 2)
                a_block_indices = get_block_indices(ab_r, ab_c)
                a_scale = a_coeff[ab]
                
                for bb in range(4):
                    if b_coeff[bb] == 0:
                        continue
                    bb_r, bb_c = divmod(bb, 2)
                    b_block_indices = get_block_indices(bb_r, bb_c)
                    b_scale = b_coeff[bb]
                    
                    # The product A_ab * B_bb contributes to output block (ab_r, bb_c)
                    # We compute this 2x2*2x2 product using Strassen (7 scalar products)
                    
                    for s in range(7):
                        # s-th scalar product in the Strassen decomposition of A_ab * B_bb
                        s2_a = S2_A_coeff[s]  # [a00, a01, a10, a11]
                        s2_b = S2_B_coeff[s]  # [b00, b01, b10, b11]
                        
                        # The scalar product involves:
                        # sum_{ai} s2_a[ai] * A_ab[ai] * sum_{bi} s2_b[bi] * B_bb[bi]
                        # = (sum_{ai} s2_a[ai] * A_ab[ai]) * (sum_{bi} s2_b[bi] * B_bb[bi])
                        
                        # In the tensor decomposition, this becomes:
                        # U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col]
                        # where the indices are mapped through the block structure
                        
                        # For the A part: A_ab has elements at positions a_block_indices
                        # a_block_indices = [idx0, idx1, idx2, idx3] corresponding to local (0,1,2,3)
                        # which are (0,0), (0,1), (1,0), (1,1) within the block
                        
                        # For the B part: B_bb has elements at positions b_block_indices
                        
                        # The product contributes to C at block (ab_r, bb_c)
                        # which has indices get_block_indices(ab_r, bb_c)
                        
                        c_block_indices = get_block_indices(ab_r, bb_c)
                        
                        # Now, for each element of the A-block and B-block:
                        # The scalar product s involves:
                        # (sum_{ai} s2_a[ai] * A_ab[ai]) * (sum_{bi} s2_b[bi] * B_bb[bi])
                        
                        # This is a single scalar product, so in the tensor decomposition:
                        # U and V have non-zero entries for the elements involved,
                        # and W has non-zero entries for the output elements.
                        
                        # But wait - this isn't quite right. The Strassen decomposition
                        # gives us 7 scalar products, each of which is a product of two
                        # linear combinations. In the tensor decomposition, each scalar
                        # product becomes one column in U, V, W.
                        
                        # For scalar product s:
                        # U[i*m+j, col] = sum of contributions from A elements
                        # V[j*p+k, col] = sum of contributions from B elements
                        # W[k*n+i, col] = output coefficient
                        
                        # Actually, let me think about this differently.
                        # 
                        # The 2x2x2 Strassen decomposition says:
                        # C[ci] = sum_s S2_C_out[ci][s] * (sum_ai S2_A_coeff[s][ai] * A[ai]) * (sum_bi S2_B_coeff[s][bi] * B[bi])
                        #
                        # So for each s, we have a term:
                        # S2_C_out[ci][s] * (sum_ai S2_A_coeff[s][ai] * A[ai]) * (sum_bi S2_B_coeff[s][bi] * B[bi])
                        #
                        # This is a rank-1 tensor: (sum_ai S2_A_coeff[s][ai] * A[ai]) ⊗ (sum_bi S2_B_coeff[s][bi] * B[bi]) ⊗ (S2_C_out[ci][s] * C[ci])
                        #
                        # But this isn't quite right either, because the sums are over different indices.
                        
                        # Let me use a cleaner formulation.
                        # 
                        # The matrix multiplication tensor for 2x2*2x2 is:
                        # T[a,b,c] = 1 if c = a*b (in the appropriate sense)
                        # 
                        # The Strassen decomposition gives:
                        # T = sum_s u_s ⊗ v_s ⊗ w_s
                        # where u_s, v_s, w_s are vectors in R^4 (for the 4 elements of 2x2)
                        
                        # For our 4x4*4x4 problem, we're using Strassen recursively.
                        # Each "scalar product" at the top level becomes a 2x2*2x2 product,
                        # which is decomposed into 7 scalar products.
                        
                        # Let me just build it directly by tracking all the indices.
                        
                        # For the s-th product in the Strassen decomposition of A_ab * B_bb:
                        # The A-part is: sum_{ai} S2_A_coeff[s][ai] * A_ab[ai]
                        # where A_ab[ai] is the element at position a_block_indices[ai]
                        
                        # The B-part is: sum_{bi} S2_B_coeff[s][bi] * B_bb[bi]
                        # where B_bb[bi] is the element at position b_block_indices[bi]
                        
                        # The output goes to C[ab_r, bb_c][ci] with coefficient S2_C_out[ci][s]
                        
                        # In the tensor decomposition, for this particular scalar product:
                        # U[i*m+j, col] = 1 if (i,j) corresponds to an A element in A_ab
                        # V[j*p+k, col] = 1 if (j,k) corresponds to a B element in B_bb  
                        # W[k*n+i, col] = 1 if (k,i) corresponds to a C element in C[ab_r,bb_c]
                        
                        # But we need to account for the linear combinations.
                        
                        # Let me just accumulate the contributions directly.
                        
                        # For each ai (0..3) with S2_A_coeff[s][ai] != 0:
                        #   a_idx = a_block_indices[ai]
                        #   a_i, a_j = divmod(a_idx, 4)
                        #   U[a_i*4 + a_j, col] += S2_A_coeff[s][ai] * a_scale
                        # 
                        # For each bi (0..3) with S2_B_coeff[s][bi] != 0:
                        #   b_idx = b_block_indices[bi]
                        #   b_j, b_k = divmod(b_idx, 4)
                        #   V[b_j*4 + b_k, col] += S2_B_coeff[s][bi] * b_scale
                        # 
                        # For each ci (0..3) with S2_C_out[ci][s] != 0:
                        #   c_idx = c_block_indices[ci]
                        #   c_i, c_j = divmod(c_idx, 4)
                        #   W[c_j*4 + c_i, col] += S2_C_out[ci][s]
                        
                        # Wait, this isn't right. The issue is that the Strassen decomposition
                        # gives us products of linear combinations, not individual elements.
                        
                        # Let me reconsider. The correct way to build the tensor decomposition
                        # for Strassen is:
                        
                        # For each of the 7 Strassen products at the top level, and for each
                        # of the 7 scalar products within it, we create one column in U, V, W.
                        
                        # The key insight is that the Strassen decomposition is a bilinear
                        # algorithm, so each "product" is a scalar multiplication of two
                        # linear combinations. In the tensor decomposition, this becomes:
                        # U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col]
                        
                        # For the s-th scalar product in the r-th Strassen product:
                        # U[i*m+j, col] = coefficient of A[i,j] in the A-linear-combination
                        # V[j*p+k, col] = coefficient of B[j,k] in the B-linear-combination
                        # W[k*n+i, col] = coefficient of C[i,k] in the C-output
                        # 
                        # But we also need to account for the block structure.
                        
                        # Let me just compute it explicitly.
                        
                        # For the r-th Strassen product (at block level):
                        # A_sum = sum_ab A_coeff[r][ab] * A_ab
                        # B_sum = sum_bb B_coeff[r][bb] * B_bb
                        # 
                        # For the s-th scalar product (within the 2x2*2x2 Strassen):
                        # scalar_prod = (sum_ai S2_A_coeff[s][ai] * A_sum[ai]) * (sum_bi S2_B_coeff[s][bi] * B_sum[bi])
                        # 
                        # where A_sum[ai] is the ai-th element of A_sum (a 2x2 matrix)
                        # and B_sum[bi] is the bi-th element of B_sum (a 2x2 matrix)
                        
                        # A_sum[ai] = sum_ab A_coeff[r][ab] * A_ab[ai]
                        # where A_ab[ai] is the ai-th element of block ab
                        
                        # So scalar_prod = sum_ab sum_ai sum_bb sum_bi 
                        #   A_coeff[r][ab] * S2_A_coeff[s][ai] * A_ab[ai] * 
                        #   B_coeff[r][bb] * S2_B_coeff[s][bi] * B_bb[bi]
                        
                        # This is a sum of terms, each of which is:
                        # (A_coeff[r][ab] * S2_A_coeff[s][ai] * A_ab[ai]) * 
                        # (B_coeff[r][bb] * S2_B_coeff[s][bi] * B_bb[bi])
                        
                        # Each such term is a scalar product of an A element and a B element,
                        # contributing to specific C elements.
                        
                        # In the tensor decomposition, each such term becomes one column.
                        # But that would give us 4*4*7*7 = 784 columns, which is way too many.
                        
                        # The key is that the Strassen decomposition is bilinear, so we can
                        # factor it differently. Let me look at this from a different angle.
                        
                        # Actually, I think the issue is that I'm overcomplicating this.
                        # The standard way to implement Strassen recursively in a tensor
                        # decomposition is:
                        
                        # 1. At the top level, we have 7 products of 2x2*2x2 matrices.
                        # 2. Each 2x2*2x2 product is decomposed into 7 scalar products.
                        # 3. Total: 49 scalar products.
                        
                        # For each of the 49 scalar products, we need to determine:
                        # - Which A elements are involved (with what coefficients)
                        # - Which B elements are involved (with what coefficients)
                        # - Which C elements are involved (with what coefficients)
                        
                        # Let me just compute this directly.
                        
                        # For the r-th Strassen product (0..6) and s-th scalar product (0..6):
                        # col = r * 7 + s
                        
                        # The A-part: sum_ab A_coeff[r][ab] * A_ab
                        # The B-part: sum_bb B_coeff[r][bb] * B_bb
                        # 
                        # Within each 2x2*2x2 product, the s-th scalar product involves:
                        # A_elements: S2_A_coeff[s][ai] for ai in 0..3
                        # B_elements: S2_B_coeff[s][bi] for bi in 0..3
                        # C_elements: S2_C_out[ci][s] for ci in 0..3
                        
                        # So for this scalar product:
                        # U[i*4+j, col] = sum over all (ab, ai) such that A_ab[ai] = A[i,j]
                        #   A_coeff[r][ab] * S2_A_coeff[s][ai]
                        
                        # V[j*4+k, col] = sum over all (bb, bi) such that B_bb[bi] = B[j,k]
                        #   B_coeff[r][bb] * S2_B_coeff[s][bi]
                        
                        # W[k*4+i, col] = sum over all (ci) such that C[ab_r,bb_c][ci] = C[i,k]
                        #   S2_C_out[ci][s]
                        
                        # But we need to be careful about the block indices.
                        
                        # Let me just compute it numerically.
                        
                        # For each (i, j) in 0..3 x 0..3:
                        #   U[i*4+j, col] = 0
                        #   for ab in 0..3:
                        #     if A_coeff[r][ab] != 0:
                        #       ab_r, ab_c = divmod(ab, 2)
                        #       a_block = get_block_indices(ab_r, ab_c)
                        #       for ai in 0..3:
                        #         if S2_A_coeff[s][ai] != 0:
                        #           a_idx = a_block[ai]
                        #           a_i, a_j = divmod(a_idx, 4)
                        #           if a_i == i and a_j == j:
                        #             U[i*4+j, col] += A_coeff[r][ab] * S2_A_coeff[s][ai]
                        
                        # Similarly for V and W.
                        
                        # This is O(4*4*4*4*4*4) = O(4096) per column, which is fine.
                        
                        # Let me implement this.
                        
                        for i in range(4):
                            for j in range(4):
                                u_val = 0.0
                                for ab in range(4):
                                    if A_coeff[r][ab] == 0:
                                        continue
                                    ab_r, ab_c = divmod(ab, 2)
                                    a_block = get_block_indices(ab_r, ab_c)
                                    for ai in range(4):
                                        if S2_A_coeff[s][ai] == 0:
                                            continue
                                        a_idx = a_block[ai]
                                        a_i, a_j = divmod(a_idx, 4)
                                        if a_i == i and a_j == j:
                                            u_val += A_coeff[r][ab] * S2_A_coeff[s][ai]
                                U[i * 4 + j, col] = u_val
                        
                        for j in range(4):
                            for k in range(4):
                                v_val = 0.0
                                for bb in range(4):
                                    if B_coeff[r][bb] == 0:
                                        continue
                                    bb_r, bb_c = divmod(bb, 2)
                                    b_block = get_block_indices(bb_r, bb_c)
                                    for bi in range(4):
                                        if S2_B_coeff[s][bi] == 0:
                                            continue
                                        b_idx = b_block[bi]
                                        b_j, b_k = divmod(b_idx, 4)
                                        if b_j == j and b_k == k:
                                            v_val += B_coeff[r][bb] * S2_B_coeff[s][bi]
                                V[j * 4 + k, col] = v_val
                        
                        for i in range(4):
                            for k in range(4):
                                w_val = 0.0
                                # The output block is (ab_r, bb_c) but we need to sum over all
                                # (ab, bb) pairs that contribute to this (i, k)
                                for ab in range(4):
                                    if A_coeff[r][ab] == 0:
                                        continue
                                    ab_r, ab_c = divmod(ab, 2)
                                    for bb in range(4):
                                        if B_coeff[r][bb] == 0:
                                            continue
                                        bb_r, bb_c = divmod(bb, 2)
                                        # The product A_ab * B_bb contributes to C[ab_r, bb_c]
                                        if ab_r != i // 2 or bb_c != k // 2:
                                            continue
                                        c_block = get_block_indices(ab_r, bb_c)
                                        # Find which ci corresponds to (i, k)
                                        for ci in range(4):
                                            c_idx = c_block[ci]
                                            c_i, c_k = divmod(c_idx, 4)
                                            if c_i == i and c_k == k:
                                                w_val += S2_C_out[ci][s]
                                W[k * 4 + i, col] = w_val
                        
                        col += 1
        
        return U, V, W
    
    def build_4x4x4x3():
        """Build a decomposition for 4x4 * 4x3. Use schoolbook: rank = 4*4*3 = 48."""
        rank = 4 * 4 * 3
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((12, rank))
        col = 0
        for i in range(4):
            for k in range(4):
                for j in range(3):
                    U[i * 4 + k, col] = 1
                    V[k * 3 + j, col] = 1
                    W[j * 4 + i, col] = 1
                    col += 1
        return U, V, W
    
    # Try to build the Strassen-based decomposition
    try:
        U_44, V_44, W_44 = build_strassen_4x4x4()
        
        # Verify the 4x4x4 part
        def verify_4x4x4(U, V, W):
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        val = 0.0
                        for c in range(U.shape[1]):
                            val += U[i * 4 + j, c] * V[j * 4 + k, c] * W[k * 4 + i, c]
                        if abs(val - 1.0) > 1e-9:
                            return False
            return True
        
        if not verify_4x4x4(U_44, V_44, W_44):
            # Strassen decomposition is incorrect, fall back to schoolbook
            U, V, W = schoolbook()
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Now extend to 4x4 * 4x7
        # C = A * B where A is 4x4, B is 4x7
        # Split B = [B4 | B3] where B4 is 4x4 and B3 is 4x3
        # C = A*B4 + A*B3
        
        # For A*B4: use the Strassen decomposition (rank 49)
        # For A*B3: use schoolbook (rank 48)
        
        rank_44 = 49
        rank_43 = 48
        total_rank = rank_44 + rank_43
        
        U_full = np.zeros((16, total_rank))
        V_full = np.zeros((28, total_rank))  # 4*7
        W_full = np.zeros((28, total_rank))  # 7*4
        
        # Copy the 4x4x4 part
        for c in range(rank_44):
            U_full[:, c] = U_44[:, c]
            V_full[:16, c] = V_44[:, c]  # First 16 rows of V (for B4)
            W_full[:16, c] = W_44[:, c]  # First 16 rows of W (for C4)
        
        # Add the 4x4 x 4x3 part
        col = rank_44
        for i in range(4):
            for k in range(4):
                for j in range(4, 7):  # Columns 4, 5, 6 of B
                    U_full[i * 4 + k, col] = 1
                    V_full[k * 7 + j, col] = 1
                    W_full[j * 4 + i, col] = 1
                    col += 1
        
        # Verify the full decomposition
        if verify_full(U_full, V_full, W_full, n, m, p):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
        
        # If verification fails, fall back to schoolbook
        U, V, W = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    except Exception:
        U, V, W = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
