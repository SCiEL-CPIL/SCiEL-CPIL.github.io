"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose 4x5x5 into (4x4x4) + (4x1x5)
    # Rank 4x4x4 using Strassen is 49
    # Rank 4x1x5 using Schoolbook is 4
    # Total Rank = 53
    
    rank = 53
    U = [[0] * rank for _ in range(n * m)] # 20 x 53
    V = [[0] * rank for _ in range(m * p)] # 25 x 53
    W = [[0] * rank for _ in range(p * n)] # 20 x 53
    
    col = 0
    
    # Part 1: 4x4x4 multiplication using Strassen
    # A is 4x4 (indices 0-3 in i, 0-3 in j)
    # B is 4x4 (indices 0-3 in j, 0-3 in k)
    # C is 4x4 (indices 0-3 in i, 0-3 in k)
    
    # Strassen for 2x2x2
    def strassen_2x2x2(offset_i, offset_j, offset_k, col_start):
        # This function adds 7 factors for a 2x2x2 block
        # A is 2x2, B is 2x2, C is 2x2
        # Indices in U: (offset_i + i) * m + (offset_j + j)
        # Indices in V: (offset_j + j) * p + (offset_k + k)
        # Indices in W: (offset_k + k) * n + (offset_i + i)
        
        # Strassen steps:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to map these to tensor factors.
        # Each factor is a triplet (u, v, w) with coefficients.
        # For M1: (A11+A22)(B11+B22)
        # This expands to 4 terms: A11B11, A11B22, A22B11, A22B22
        # But Strassen combines them.
        # The tensor factor for M1 contributes to C11 and C22.
        # C11 gets +1 * M1
        # C22 gets +1 * M1
        
        # We will add 7 columns.
        
        # Helper to add a factor
        def add_factor(u_idx, v_idx, w_idx, coeff_u, coeff_v, coeff_w):
            nonlocal col
            # U is indexed by [i*m + j, c]
            # V is indexed by [j*p + k, c]
            # W is indexed by [k*n + i, c]
            
            # u_idx is (i, j) relative to block
            # v_idx is (j, k) relative to block
            # w_idx is (k, i) relative to block
            
            i_u, j_u = u_idx
            j_v, k_v = v_idx
            k_w, i_w = w_idx
            
            # Global indices
            gi_u = (offset_i + i_u) * m + (offset_j + j_u)
            gi_v = (offset_j + j_v) * p + (offset_k + k_v)
            gi_w = (offset_k + k_w) * n + (offset_i + i_w)
            
            U[gi_u][col] += coeff_u
            V[gi_v][col] += coeff_v
            W[gi_w][col] += coeff_w
            
        # M1 = (A11+A22)(B11+B22)
        # Terms: A11B11, A11B22, A22B11, A22B22
        # We can represent this as 4 separate factors or combine?
        # Tensor decomposition requires linear terms.
        # (A11+A22) is a sum. We can't multiply sums directly in one factor.
        # We must expand.
        # So M1 contributes 4 factors? No, Strassen reduces the number of multiplications.
        # The "multiplication" M1 is one scalar multiplication in the algorithm, but in the tensor decomposition,
        # it corresponds to the product of the linear forms.
        # The tensor factor for M1 is:
        # u = A11 + A22
        # v = B11 + B22
        # w = C11 + C22 (with appropriate signs)
        
        # Let's trace M1:
        # u: A11 (1,1) coeff 1, A22 (2,2) coeff 1
        # v: B11 (1,1) coeff 1, B22 (2,2) coeff 1
        # w: C11 (1,1) coeff 1, C22 (2,2) coeff 1
        
        # But wait, the tensor factor is a single column.
        # U[col] has non-zeros at (1,1) and (2,2).
        # V[col] has non-zeros at (1,1) and (2,2).
        # W[col] has non-zeros at (1,1) and (2,2).
        # This is ONE factor.
        
        # So for M1:
        add_factor((0,0), (0,0), (0,0), 1, 1, 1) # A11 B11 -> C11
        # Wait, this is not how it works.
        # If U has A11 and A22, and V has B11 and B22, the product is (A11+A22)(B11+B22).
        # This term appears in C11 and C22.
        # So W must have C11 and C22.
        # But the factor is a single column.
        # U[col][A11] = 1, U[col][A22] = 1
        # V[col][B11] = 1, V[col][B22] = 1
        # W[col][C11] = 1, W[col][C22] = 1
        # This single factor generates:
        # 1*1*1 * A11 B11 C11
        # 1*1*1 * A11 B11 C22 (WRONG)
        # 1*1*1 * A22 B22 C11 (WRONG)
        # 1*1*1 * A22 B22 C22
        
        # The tensor decomposition must be exact.
        # Strassen's algorithm works because the errors cancel out in the sum.
        # But each factor must be a valid rank-1 tensor.
        # The rank-1 tensor for M1 is NOT (A11+A22)(B11+B22)(C11+C22).
        # It is a combination.
        
        # Actually, Strassen's decomposition is:
        # C = M1 * (E11 + E22) - M2 * (E21) + M3 * (E12) + ...
        # No, that's the reconstruction.
        # The factors are:
        # Factor 1: u = A11+A22, v = B11+B22, w = C11+C22 ?
        # If we use this factor, it produces cross terms.
        # The cross terms are cancelled by other factors?
        # No, in tensor decomposition, the sum of rank-1 tensors must equal the multiplication tensor.
        # The multiplication tensor is sparse.
        # Strassen's method works by embedding the problem in a larger space or using cancellations.
        # For $2 \times 2 \times 2$, the 7 factors are:
        # 1. u=A11+A22, v=B11+B22, w=C11+C22
        # 2. u=A21+A22, v=B11, w=C21
        # 3. u=A11, v=B12-B22, w=C12
        # 4. u=A22, v=B21-B11, w=C21
        # 5. u=A11+A12, v=B22, w=C12
        # 6. u=A21-A11, v=B11+B12, w=C22
        # 7. u=A12-A22, v=B21+B22, w=C22
        
        # Let's verify Factor 1:
        # (A11+A22)(B11+B22)(C11+C22)
        # Terms: A11B11C11, A11B11C22, A11B22C11, A11B22C22, ...
        # The multiplication tensor only has A11B11C11, A11B22C12, etc.
        # So Factor 1 alone is not part of the decomposition.
        # The decomposition is a sum of 7 such tensors that cancel the unwanted terms.
        
        # So I can simply define the 7 factors for the 2x2x2 block.
        
        # Factor 1
        add_factor((0,0), (0,0), (0,0), 1, 1, 1)
        add_factor((1,1), (0,0), (0,0), 1, 1, 1) # A22 part of u, B11 part of v? No.
        
        # This manual expansion is error-prone.
        # I will use the known 7 factors for 2x2x2.
        
        # Let's define the 7 factors explicitly for a 2x2x2 block at (oi, oj, ok).
        # Indices:
        # A: (0,0), (0,1), (1,0), (1,1)
        # B: (0,0), (0,1), (1,0), (1,1)
        # C: (0,0), (0,1), (1,0), (1,1)
        
        # F1: u={00:1, 11:1}, v={00:1, 11:1}, w={00:1, 11:1}
        # F2: u={10:1, 11:1}, v={00:1}, w={10:1}
        # F3: u={00:1}, v={01:1, 11:-1}, w={01:1}
        # F4: u={11:1}, v={10:1, 00:-1}, w={10:1}
        # F5: u={00:1, 01:1}, v={11:1}, w={01:1}
        # F6: u={10:1, 00:-1}, v={00:1, 01:1}, w={11:1}
        # F7: u={01:1, 11:-1}, v={10:1, 11:1}, w={11:1}
        
        # Note: w indices are (k, i).
        # C11 is (k=0, i=0). C12 is (k=1, i=0). C21 is (k=0, i=1). C22 is (k=1, i=1).
        
        # F1:
        # u: (0,0), (1,1)
        # v: (0,0), (1,1)
        # w: (0,0), (1,1)
        
        # I will implement this by adding to the matrices.
        
        # F1
        U[(offset_i+0)*m + (offset_j+0)][col] += 1
        U[(offset_i+1)*m + (offset_j+1)][col] += 1
        V[(offset_j+0)*p + (offset_k+0)][col] += 1
        V[(offset_j+1)*p + (offset_k+1)][col] += 1
        W[(offset_k+0)*n + (offset_i+0)][col] += 1
        W[(offset_k+1)*n + (offset_i+1)][col] += 1
        col += 1
        
        # F2
        U[(offset_i+1)*m + (offset_j+0)][col] += 1
        U[(offset_i+1)*m + (offset_j+1)][col] += 1
        V[(offset_j+0)*p + (offset_k+0)][col] += 1
        W[(offset_k+0)*n + (offset_i+1)][col] += 1
        col += 1
        
        # F3
        U[(offset_i+0)*m + (offset_j+0)][col] += 1
        V[(offset_j+0)*p + (offset_k+1)][col] += 1
        V[(offset_j+1)*p + (offset_k+1)][col] += -1
        W[(offset_k+1)*n + (offset_i+0)][col] += 1
        col += 1
        
        # F4
        U[(offset_i+1)*m + (offset_j+1)][col] += 1
        V[(offset_j+1)*p + (offset_k+0)][col] += 1
        V[(offset_j+0)*p + (offset_k+0)][col] += -1
        W[(offset_k+0)*n + (offset_i+1)][col] += 1
        col += 1
        
        # F5
        U[(offset_i+0)*m + (offset_j+0)][col] += 1
        U[(offset_i+0)*m + (offset_j+1)][col] += 1
        V[(offset_j+1)*p + (offset_k+1)][col] += 1
        W[(offset_k+1)*n + (offset_i+0)][col] += 1
        col += 1
        
        # F6
        U[(offset_i+1)*m + (offset_j+0)][col] += 1
        U[(offset_i+0)*m + (offset_j+0)][col] += -1
        V[(offset_j+0)*p + (offset_k+0)][col] += 1
        V[(offset_j+0)*p + (offset_k+1)][col] += 1
        W[(offset_k+1)*n + (offset_i+1)][col] += 1
        col += 1
        
        # F7
        U[(offset_i+0)*m + (offset_j+1)][col] += 1
        U[(offset_i+1)*m + (offset_j+1)][col] += -1
        V[(offset_j+1)*p + (offset_k+0)][col] += 1
        V[(offset_j+1)*p + (offset_k+1)][col] += 1
        W[(offset_k+1)*n + (offset_i+1)][col] += 1
        col += 1

    # Apply Strassen to the 4x4x4 block
    # Split 4x4 into four 2x2 blocks
    # Top-Left: (0,0)
    strassen_2x2x2(0, 0, 0, col)
    # Top-Right: (0,2)
    strassen_2x2x2(0, 2, 2, col) # Wait, j and k offsets
    # B is 4x4. j is 0-3. k is 0-3.
    # A is 4x4. i is 0-3. j is 0-3.
    
    # Block 1: i=0,1; j=0,1; k=0,1
    # Block 2: i=0,1; j=0,1; k=2,3
    # Block 3: i=2,3; j=0,1; k=0,1
    # Block 4: i=2,3; j=0,1; k=2,3
    # But A and B are coupled.
    # A is 4x4. B is 4x4.
    # We split A into 4 2x2 blocks. B into 4 2x2 blocks.
    # A11 (0,0), A12 (0,2), A21 (2,0), A22 (2,2)
    # B11 (0,0), B12 (0,2), B21 (2,0), B22 (2,2)
    
    # C11 = A11B11 + A12B21
    # C12 = A11B12 + A12B22
    # C21 = A21B11 + A22B21
    # C22 = A21B12 + A22B22
    
    # We need to compute 8 multiplications of 2x2 matrices.
    # 1. A11 B11
    # 2. A12 B21
    # 3. A11 B12
    # 4. A12 B22
    # 5. A21 B11
    # 6. A22 B21
    # 7. A21 B12
    # 8. A22 B22
    
    # Each uses 7 factors. Total 56?
    # No, Strassen for 4x4 uses 7 multiplications of 2x2 matrices.
    # The 7 multiplications are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So we need 7 calls to strassen_2x2x2?
    # No, the inputs to the 2x2 multiplications are sums/differences of blocks.
    # This means the U and V matrices will have entries that are sums of block entries.
    # My strassen_2x2x2 function assumes the input blocks are contiguous in memory.
    # But here, the "block" A11+A22 is not contiguous.
    # So I cannot just call strassen_2x2x2 with offsets.
    
    # I must manually construct the 7 factors for the 4x4x4 block.
    # This is complex.
    
    # Alternative: Use Schoolbook for 4x4x4 (Rank 64) + Outer Product (Rank 4) = 68.
    # Schoolbook is easier to implement correctly.
    # 68 is better than 100.
    
    # Let's revert to Schoolbook for the 4x4x4 part.
    
    # Clear U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Part 1: 4x4x4 Schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Part 2: 4x1x5 Schoolbook (Outer Product)
    # A is 4x1 (j=4), B is 1x5 (j=4, k=0..4)
    # C is 4x5 (i=0..3, k=0..4)
    for i in range(4):
        for k in range(5):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
