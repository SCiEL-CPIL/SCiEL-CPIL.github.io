import time
import sys
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 4, 4, 4
    
    # Strassen 4x4 decomposition: rank 49
    # A (4x4) is split into 2x2 blocks: A11, A12, A21, A22
    # B (4x4) is split into 2x2 blocks: B11, B12, B21, B22
    # C (4x4) is split into 2x2 blocks: C11, C12, C21, C22
    #
    # Strassen formulas:
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
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each M is a 2x2 matrix product, decomposed via Strassen 2x2 (rank 7).
    # Total rank = 7 * 7 = 49.
    
    rank = 49
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)
    
    # Block indices in the 4x4 matrices:
    # Block (br, bc) where br, bc in {0,1} corresponds to rows br*2..br*2+1, cols bc*2..bc*2+1
    # A block (br, bc): flat index (br*2 + i)*4 + (bc*2 + j) for i,j in {0,1}
    # B block (br, bc): flat index (br*2 + i)*4 + (bc*2 + j) for i,j in {0,1}
    # C block (br, bc): flat index (br*2 + i)*4 + (bc*2 + j) for i,j in {0,1}
    
    # Strassen 2x2 terms:
    # For 2x2 matrices X and Y, computing X*Y:
    # s1 = (x00 + x11)(y00 + y11)
    # s2 = (x10 + x11)(y00)
    # s3 = (x00)(y01 - y11)
    # s4 = (x11)(y10 - y00)
    # s5 = (x00 + x01)(y11)
    # s6 = (x10 - x00)(y00 + y01)
    # s7 = (x01 - x11)(y10 + y11)
    #
    # Result:
    # z00 = s1 + s4 - s5 + s7
    # z01 = s3 + s5
    # z10 = s2 + s4
    # z11 = s1 - s2 + s3 + s6
    
    # Each s_t has:
    # X coefficients: dict {(i,j): val}
    # Y coefficients: dict {(i,j): val}
    # Z coefficients: dict {(i,j): val}
    
    strassen_2x2_terms = [
        # s1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # s2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # s3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # s4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # s5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # s6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # s7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
    ]
    
    # For each of the 7 Strassen products M1..M7, we need to decompose
    # the product of two 2x2 matrices (which may be sums/differences of blocks)
    # using the 7-term Strassen 2x2 algorithm.
    #
    # The key insight: since the Strassen 2x2 algorithm is bilinear, if we have
    # (A_block1 * c1 + A_block2 * c2) * (B_block1 * d1 + B_block2 * d2),
    # we can apply the Strassen 2x2 terms to the combined matrices.
    # The X coefficients for each term are linear combinations of the entries
    # of the combined X matrix, which is a linear combination of the block entries.
    
    def add_strassen_product(x_blocks, y_blocks, z_block, z_coeff, base_col):
        """
        Add the Strassen 2x2 decomposition for the product of two 2x2 matrices.
        
        x_blocks: list of (block_idx, coeff) where block_idx is (br, bc)
        y_blocks: list of (block_idx, coeff)
        z_block: (br, bc) - which 2x2 block of C this contributes to
        z_coeff: multiplier for the z contribution
        base_col: starting column index in U, V, W
        """
        # For each of the 7 Strassen terms, add the contributions
        for t_idx, (x_coeff, y_coeff, z_coeff_dict) in enumerate(strassen_2x2_terms):
            col = base_col + t_idx
            
            # X is a linear combination of blocks in x_blocks
            # The x_coeff dict gives coefficients for entries (i,j) in the 2x2 matrix
            # We need to map these to the actual A matrix entries
            for (xi, xj), xv in x_coeff.items():
                # For each block in x_blocks, add xv * block_coeff to the appropriate entry
                for (br, bc), bc_coeff in x_blocks:
                    # Entry (xi, xj) in the 2x2 block (br, bc) of A
                    # corresponds to A entry (br*2 + xi, bc*2 + xj)
                    a_flat = (br * 2 + xi) * 4 + (bc * 2 + xj)
                    U[a_flat, col] += xv * bc_coeff
            
            # Y is a linear combination of blocks in y_blocks
            for (yi, yj), yv in y_coeff.items():
                for (br, bc), bc_coeff in y_blocks:
                    # Entry (yi, yj) in the 2x2 block (br, bc) of B
                    # corresponds to B entry (br*2 + yi, bc*2 + yj)
                    b_flat = (br * 2 + yi) * 4 + (bc * 2 + yj)
                    V[b_flat, col] += yv * bc_coeff
            
            # Z contribution
            for (zi, zj), zv in z_coeff_dict.items():
                # Entry (zi, zj) in the 2x2 block z_block of C
                # corresponds to C entry (z_block[0]*2 + zi, z_block[1]*2 + zj)
                cr, cc = z_block
                c_flat = (cr * 2 + zi) * 4 + (cc * 2 + zj)
                W[c_flat, col] += z_coeff * zv
    
    # Now define the 7 Strassen products for 4x4:
    # Block indices: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
    #                B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1)
    #                C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1)
    
    # M1 = (A11 + A22)(B11 + B22)
    # X = A11 + A22, Y = B11 + B22
    # Result contributes to C11 with coeff +1, C22 with coeff +1
    add_strassen_product(
        x_blocks=[(0,0), (1,1)],  # A11 + A22
        y_blocks=[(0,0), (1,1)],  # B11 + B22
        z_block=(0,0), z_coeff=1,  # C11 += M1
        base_col=0
    )
    # But wait, M1 contributes to both C11 and C22.
    # The Strassen 2x2 decomposition gives us 7 terms that compute X*Y as a 2x2 matrix.
    # The result X*Y is a 2x2 matrix whose entries are:
    # (X*Y)00, (X*Y)01, (X*Y)10, (X*Y)11
    # And C11 = M1 + M4 - M5 + M7 means C11 gets the entire 2x2 matrix M1.
    # C22 = M1 - M2 + M3 + M6 means C22 also gets the entire 2x2 matrix M1.
    # So the same 7 terms contribute to both C11 and C22.
    
    # I need to redo this. The Strassen 2x2 decomposition computes X*Y as a 2x2 matrix.
    # The 7 terms are scalar multiplications that, when combined, give the 4 entries of X*Y.
    # For M1, the result is a 2x2 matrix that contributes to C11 (as +M1) and C22 (as +M1).
    # So each of the 7 terms contributes to both C11 and C22.
    
    # Let me restructure: I'll create a function that adds the Strassen 2x2 decomposition
    # for a product, and allows the result to contribute to multiple C blocks with different coefficients.
    
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)
    
    def add_strassen_product_v2(x_blocks, y_blocks, z_contributions, base_col):
        """
        x_blocks: list of (block_idx, coeff)
        y_blocks: list of (block_idx, coeff)
        z_contributions: list of (z_block, z_coeff) - the 2x2 result contributes to these C blocks
        base_col: starting column
        """
        for t_idx, (x_coeff, y_coeff, z_coeff_dict) in enumerate(strassen_2x2_terms):
            col = base_col + t_idx
            
            # X contribution
            for (xi, xj), xv in x_coeff.items():
                for (br, bc), bc_coeff in x_blocks:
                    a_flat = (br * 2 + xi) * 4 + (bc * 2 + xj)
                    U[a_flat, col] += xv * bc_coeff
            
            # Y contribution
            for (yi, yj), yv in y_coeff.items():
                for (br, bc), bc_coeff in y_blocks:
                    b_flat = (br * 2 + yi) * 4 + (bc * 2 + yj)
                    V[b_flat, col] += yv * bc_coeff
            
            # Z contribution to multiple blocks
            for (zi, zj), zv in z_coeff_dict.items():
                for (z_block, z_coeff) in z_contributions:
                    cr, cc = z_block
                    c_flat = (cr * 2 + zi) * 4 + (cc * 2 + zj)
                    W[c_flat, col] += z_coeff * zv
    
    # M1 = (A11 + A22)(B11 + B22)
    # Contributes to C11 with +1 and C22 with +1
    add_strassen_product_v2(
        x_blocks=[(0,0,1), (1,1,1)],  # Wait, I need (block_idx, coeff) format
        y_blocks=[(0,0,1), (1,1,1)],
        z_contributions=[((0,0), 1), ((1,1), 1)],
        base_col=0
    )
    
    # Hmm, I have a format issue. Let me fix the block representation.
    # x_blocks should be list of ((br, bc), coeff)
    
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)
    
    def add_strassen_product_v3(x_blocks, y_blocks, z_contributions, base_col):
        """
        x_blocks: list of ((br, bc), coeff)
        y_blocks: list of ((br, bc), coeff)
        z_contributions: list of ((br, bc), coeff)
        base_col: starting column
        """
        for t_idx, (x_coeff, y_coeff, z_coeff_dict) in enumerate(strassen_2x2_terms):
            col = base_col + t_idx
            
            # X contribution
            for (xi, xj), xv in x_coeff.items():
                for ((br, bc), bc_coeff) in x_blocks:
                    a_flat = (br * 2 + xi) * 4 + (bc * 2 + xj)
                    U[a_flat, col] += xv * bc_coeff
            
            # Y contribution
            for (yi, yj), yv in y_coeff.items():
                for ((br, bc), bc_coeff) in y_blocks:
                    b_flat = (br * 2 + yi) * 4 + (bc * 2 + yj)
                    V[b_flat, col] += yv * bc_coeff
            
            # Z contribution to multiple blocks
            for (zi, zj), zv in z_coeff_dict.items():
                for ((zbr, zbc), z_coeff) in z_contributions:
                    c_flat = (zbr * 2 + zi) * 4 + (zbc * 2 + zj)
                    W[c_flat, col] += z_coeff * zv
    
    # M1 = (A11 + A22)(B11 + B22)
    add_strassen_product_v3(
        x_blocks=[((0,0), 1), ((1,1), 1)],
        y_blocks=[((0,0), 1), ((1,1), 1)],
        z_contributions=[((0,0), 1), ((1,1), 1)],
        base_col=0
    )
    
    # M2 = (A21 + A22)B11
    add_strassen_product_v3(
        x_blocks=[((1,0), 1), ((1,1), 1)],
        y_blocks=[((0,0), 1)],
        z_contributions=[((1,0), 1), ((1,1), -1)],
        base_col=7
    )
    
    # M3 = A11(B12 - B22)
    add_strassen_product_v3(
        x_blocks=[((0,0), 1)],
        y_blocks=[((0,1), 1), ((1,1), -1)],
        z_contributions=[((0,1), 1), ((1,1), 1)],
        base_col=14
    )
    
    # M4 = A22(B21 - B11)
    add_strassen_product_v3(
        x_blocks=[((1,1), 1)],
        y_blocks=[((1,0), 1), ((0,0), -1)],
        z_contributions=[((0,0), 1), ((1,0), 1)],
        base_col=21
    )
    
    # M5 = (A11 + A12)B22
    add_strassen_product_v3(
        x_blocks=[((0,0), 1), ((0,1), 1)],
        y_blocks=[((1,1), 1)],
        z_contributions=[((0,0), -1), ((0,1), 1)],
        base_col=28
    )
    
    # M6 = (A21 - A11)(B11 + B12)
    add_strassen_product_v3(
        x_blocks=[((1,0), 1), ((0,0), -1)],
        y_blocks=[((0,0), 1), ((0,1), 1)],
        z_contributions=[((1,1), 1)],
        base_col=35
    )
    
    # M7 = (A12 - A22)(B21 + B22)
    add_strassen_product_v3(
        x_blocks=[((0,1), 1), ((1,1), -1)],
        y_blocks=[((1,0), 1), ((1,1), 1)],
        z_contributions=[((0,0), 1), ((1,1), 1)],
        base_col=42
    )
    
    # Now verify the tensor identity
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # should equal 1 if c = k*4+i where a = i*4+j, b = j*4+k, else 0
    
    # Verify with random test
    A_test = np.random.randn(16)
    B_test = np.random.randn(16)
    
    A_mat = A_test.reshape(4, 4)
    B_mat = B_test.reshape(4, 4)
    C_expected = A_mat @ B_mat
    
    # Compute C using the decomposition
    # C[c_idx] = sum_r W[c_idx, r] * (sum_a U[a, r] * A[a]) * (sum_b V[b, r] * B[b])
    # Wait, the tensor identity is:
    # C = sum_r (U[:,r]^T A) * (V[:,r]^T B) * W[:,r]
    # No, that's not right either.
    # The tensor decomposition is:
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # And matrix multiplication is:
    # C[k,i] = sum_j A[i,j] B[j,k]
    # So T[a,b,c] = 1 if a=i*4+j, b=j*4+k, c=k*4+i, else 0.
    # The decomposition gives:
    # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
    # This means:
    # C = sum_r (sum_a U[a,r] A[a]) * (sum_b V[b,r] B[b]) * W[:,r]
    # No wait. Let me think more carefully.
    # 
    # The matrix multiplication tensor is:
    # C[k,i] = sum_j A[i,j] B[j,k]
    # In flat indices: C[k*4+i] = sum_j A[i*4+j] B[j*4+k]
    # The tensor T[a,b,c] = 1 if there exists i,j,k such that a=i*4+j, b=j*4+k, c=k*4+i
    # 
    # A rank-R decomposition is: T[a,b,c] = sum_{r=1}^R U[a,r] V[b,r] W[c,r]
    # This means:
    # sum_j A[i*4+j] B[j*4+k] = sum_r U[i*4+j, r] V[j*4+k, r] W[k*4+i, r]
    # for all i, j, k.
    # 
    # Equivalently, for any A and B:
    # C = sum_r (U[:,r]^T A) * (V[:,r]^T B) * W[:,r]
    # where the products are scalar products.
    # Let u_r = U[:,r], v_r = V[:,r], w_r = W[:,r]
    # C = sum_r (u_r^T A) * (v_r^T B) * w_r
    # where A and B are flattened to 16-vectors.
    
    A_flat = A_test
    B_flat = B_test
    
    C_computed = np.zeros(16)
    for r in range(rank):
        u_r = U[:, r]
        v_r = V[:, r]
        w_r = W[:, r]
        scalar_u = np.dot(u_r, A_flat)
        scalar_v = np.dot(v_r, B_flat)
        C_computed += scalar_u * scalar_v * w_r
    
    C_computed_mat = C_computed.reshape(4, 4)
    
    # Check if C_computed matches C_expected
    error = np.max(np.abs(C_computed_mat - C_expected))
    
    if error > 1e-9:
        # Verification failed. Return schoolbook as fallback.
        rank_school = 64
        U_s = np.zeros((16, rank_school), dtype=float)
        V_s = np.zeros((16, rank_school), dtype=float)
        W_s = np.zeros((16, rank_school), dtype=float)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_s[i*4+j, col] = 1
                    V_s[j*4+k, col] = 1
                    W_s[k*4+i, col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    # Success! Return the Strassen 49 decomposition.
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
