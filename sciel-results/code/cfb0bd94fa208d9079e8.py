"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    
    Strategy: Split B (4x8) into B1 (4x4) and B2 (4x4).
    Compute C1 = A*B1 and C2 = A*B2, each using the rank-49 Strassen 
    decomposition for 4x4*4x4. Total rank = 49 + 49 = 98.
    
    The rank-49 4x4*4x4 uses Strassen at the top level (7 products of 2x2*2x2),
    where each 2x2*2x2 uses Strassen (rank 7). Total: 7*7 = 49.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # Build rank-7 Strassen for 2x2*2x2
    # Indexing: A[row,col] where row in {0,1}, col in {0,1}
    # 2x2*2x2 Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 4x4*4x4, split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]]  (each 2x2)
    # B = [[B00, B01], [B10, B11]]  (each 2x2)
    # C = [[C00, C01], [C10, C11]]  (each 2x2)
    
    # Strassen at top level (4x4*4x4 -> 7 products of 2x2*2x2):
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2*2x2 product, done with rank-7 Strassen.
    # So total rank = 7 * 7 = 49 for 4x4*4x4.
    
    # For 4x4*4x8: split B into B_left (cols 0-3) and B_right (cols 4-7).
    # C_left = A * B_left (4x4 result)
    # C_right = A * B_right (4x4 result)
    # Output C is 4x8: C[:, 0:4] = C_left, C[:, 4:8] = C_right
    
    # Tensor convention: W[k*n+i, r] for output index (k, i) where k is row of output (0..p-1), i is col of output (0..n-1)
    # Wait, let me re-read: "The output factor follows the official trace convention W[k*n+i,r]"
    # For matrix multiplication C = A*B where A is n*m, B is m*p, C is n*p.
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # The tensor has indices (i, k, j) for A[i,k], B[k,j], C[i,j]
    # U has shape (n*m, rank): indexed by (i, k) -> i*m + k
    # V has shape (m*p, rank): indexed by (k, j) -> k*p + j  
    # W has shape (p*n, rank): indexed by (j, i) -> j*n + i (the "trace convention" W[k*n+i,r] means output index is j*n+i where j is the column of C and i is the row of C)
    
    # Wait, re-reading: W[k*n+i, r]. So the output index in W is k*n+i.
    # In standard matrix mult C = A*B, C[i,j] = sum_k A[i,k] B[k,j].
    # The output C has dimensions n x p (rows = n, cols = p).
    # W index k*n+i: if we interpret this as the output being indexed by (k, i) where the first index ranges over p (0..p-1) and second over n (0..n-1)...
    # Actually W has shape (p*n, rank). So the index ranges from 0 to p*n-1.
    # W[k*n+i] where k in [0,p), i in [0,n). So k is the column index of C (0..p-1), i is the row index of C (0..n-1).
    # So W[k*n+i, r] corresponds to C[i,k] (row i, column k).
    
    # Let me verify with schoolbook:
    # U[i*m+j, c] = 1 for column c corresponding to (i,j,k)
    # V[j*p+k, c] = 1
    # W[k*n+i, c] = 1
    # So for a given term: U picks A[i,j], V picks B[j,k], W picks C[i,k] (stored at position k*n+i).
    # This confirms: W index = k*n + i, where i is row of C, k is column of C.
    
    # Now let me build the decomposition.
    # I'll use a helper to build the rank-7 Strassen for 2x2*2x2.
    # Each "term" in the decomposition contributes to U, V, W.
    
    # Let me define the Strassen 2x2*2x2 decomposition.
    # For 2x2*2x2: A is 2x2, B is 2x2, C is 2x2.
    # A indices: (0,0), (0,1), (1,0), (1,1) -> U index = row*2 + col (since m=2 for 2x2)
    # B indices: (0,0), (0,1), (1,0), (1,1) -> V index = row*2 + col (since p=2 for 2x2)
    # C indices: (0,0), (0,1), (1,0), (1,1) -> W index = col*2 + row (trace convention: k*n+i where n=2)
    
    # Strassen 2x2*2x2 (7 terms):
    # Each term has coefficients for A entries, B entries, and contributes to C entries.
    
    # Term 1: M1 = (A00+A11)*(B00+B11)
    # A part: A00: +1, A11: +1
    # B part: B00: +1, B11: +1
    # C part: contributes to C00: +1, C11: +1
    # (M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11)
    # C00 = M1 + M4 - M5 + M7, C11 = M1 - M2 + M3 + M6
    # So M1 contributes +1 to C00 and +1 to C11.
    
    # Let me define each Strassen term as:
    # (a_coeffs, b_coeffs, c_coeffs)
    # where a_coeffs is dict {(row,col): coeff}, etc.
    
    def strassen_2x2_terms():
        """Return list of 7 terms, each is (a_dict, b_dict, c_dict) for 2x2*2x2."""
        terms = []
        # M1 = (A00+A11)*(B00+B11)
        terms.append((
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}  # C00 += M1, C11 += M1
        ))
        # M2 = (A10+A11)*B00
        terms.append((
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1, (1,1): -1}  # C10 += M2, C11 -= M2
        ))
        # M3 = A00*(B01-B11)
        terms.append((
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1}  # C01 += M3, C11 += M3
        ))
        # M4 = A11*(B10-B00)
        terms.append((
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1}  # C00 += M4, C10 += M4
        ))
        # M5 = (A00+A01)*B11
        terms.append((
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1}  # C00 -= M5, C01 += M5
        ))
        # M6 = (A10-A00)*(B00+B01)
        terms.append((
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}  # C11 += M6
        ))
        # M7 = (A01-A11)*(B10+B11)
        terms.append((
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}  # C00 += M7
        ))
        return terms
    
    strassen_terms = strassen_2x2_terms()
    
    # Now build 4x4*4x4 using Strassen at top level.
    # A is 4x4, split into 2x2 blocks: A00, A01, A10, A11
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # Similarly for B and C.
    
    # Top-level Strassen for 4x4*4x4:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2*2x2 product. To compute Mi with rank 7, we use the 
    # 2x2 Strassen decomposition. But the "A" and "B" for each Mi are linear 
    # combinations of the 2x2 blocks of the original A and B.
    
    # This gets complex. Let me think of it differently.
    
    # Alternative approach: directly enumerate all 49 terms.
    # For each of the 7 top-level Strassen products, and for each of the 7 
    # inner Strassen terms, we get one rank-1 term in the full decomposition.
    
    # Let me define the top-level Strassen structure more carefully.
    
    # Top-level: 7 products, each is (A_linear_comb) * (B_linear_comb) where
    # A_linear_comb is a 2x2 matrix (linear combo of A's 2x2 blocks) and
    # B_linear_comb is a 2x2 matrix (linear combo of B's 2x2 blocks).
    
    # For each such product, we apply the 7-term Strassen for 2x2*2x2.
    
    # Let me define the top-level Strassen coefficients.
    # Each top-level product t (t=0..6) has:
    #   A_coeffs: dict from (block_row, block_col) to coeff, where block in {0,1}^2
    #   B_coeffs: dict from (block_row, block_col) to coeff
    #   C_coeffs: dict from (block_row, block_col) to coeff (for the 2x2 result block)
    
    # Top-level Strassen:
    # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)*B00 -> C10 += M2, C11 -= M2
    # M3 = A00*(B01-B11) -> C01 += M3, C11 += M3
    # M4 = A11*(B10-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)*B11 -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)*(B00+B01) -> C11 += M6
    # M7 = (A01-A11)*(B10+B11) -> C00 += M7
    
    top_level = [
        # (A_block_coeffs, B_block_coeffs, C_block_coeffs)
        # A_block_coeffs: {(br, bc): coeff} for 2x2 blocks of A
        # B_block_coeffs: {(br, bc): coeff} for 2x2 blocks of B
        # C_block_coeffs: {(br, bc): coeff} for 2x2 blocks of C
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),  # M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),           # M2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),             # M3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),             # M4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),             # M5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),             # M6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),             # M7
    ]
    
    # For each top-level product and each inner Strassen term, we get a rank-1 term.
    # The A part: for each block (br, bc) with coeff ca, and for each inner A entry (ir, ic) with coeff ci,
    #   the full A index is (br*2+ir, bc*2+ic) in the 4x4 matrix.
    #   U index = row*4 + col = (br*2+ir)*4 + (bc*2+ic)
    #   coefficient = ca * ci
    
    # The B part: for each block (br, bc) with coeff cb, and for each inner B entry (ir, ic) with coeff ci,
    #   the full B index is (br*2+ir, bc*2+ic) in the 4x4 matrix.
    #   V index = row*4 + col = (br*2+ir)*4 + (bc*2+ic)
    #   coefficient = cb * ci
    
    # The C part: for each block (br, bc) with coeff cc, and for each inner C entry (ir, ic) with coeff ci,
    #   the full C index is (br*2+ir, bc*2+ic) in the 4x4 matrix.
    #   W index = col*4 + row = (bc*2+ic)*4 + (br*2+ir)  [trace convention: k*n+i where k=col, i=row]
    #   coefficient = cc * ci
    
    # Now for 4x4*4x8:
    # A is 4x4, B is 4x8, C is 4x8.
    # Split B into B_left (4x4, cols 0-3) and B_right (4x4, cols 4-7).
    # C_left = A * B_left (4x4), C_right = A * B_right (4x4).
    # C[:, 0:4] = C_left, C[:, 4:8] = C_right.
    
    # For B_left: V index = row*4 + col where col in 0..3, so V index = row*4 + col (same as 4x4 case)
    # For B_right: V index = row*4 + (col+4) where col in 0..3, so V index = row*4 + col + 4
    
    # For C_left: W index = col*4 + row where col in 0..3 (col of C_left, which is col of C)
    # For C_right: W index = col*4 + row where col in 4..7 (col of C), so W index = (col_left+4)*4 + row = col_left*4 + 16 + row
    
    # Wait, W has shape (p*n, rank) = (8*4, rank) = (32, rank).
    # W index = k*n + i = k*4 + i where k is column of C (0..7), i is row of C (0..3).
    # For C_left: column k in 0..3, so W index = k*4 + i.
    # For C_right: column k in 4..7, so W index = k*4 + i = (k_left+4)*4 + i = k_left*4 + 16 + i.
    
    # U has shape (n*m, rank) = (4*4, rank) = (16, rank). U index = i*m + j = i*4 + j.
    # V has shape (m*p, rank) = (4*8, rank) = (32, rank). V index = j*p + k = j*8 + k.
    # Wait, that's different! V index = j*p + k where j is row of B (0..3), k is col of B (0..7).
    # So V index = j*8 + k.
    
    # Let me redo. For the 4x4*4x4 subproblem (left half):
    # A is 4x4: U index = i*4 + j (i=row 0..3, j=col 0..3)
    # B_left is 4x4: V index = j*8 + k (j=row 0..3, k=col 0..3) [since p=8]
    # C_left is 4x4: W index = k*4 + i (k=col 0..3, i=row 0..3)
    
    # For the right half:
    # A is 4x4: U index = i*4 + j (same)
    # B_right is 4x4: V index = j*8 + (k+4) (j=row 0..3, k=col 0..3 within right half)
    # C_right is 4x4: W index = (k+4)*4 + i = k*4 + 16 + i (k=col 0..3 within right half, i=row 0..3)
    
    # Now let me build all 98 terms.
    rank = 98
    U = [[0] * rank for _ in range(16)]  # n*m = 4*4
    V = [[0] * rank for _ in range(32)]  # m*p = 4*8
    W = [[0] * rank for _ in range(32)]  # p*n = 8*4
    
    col = 0
    for half in range(2):  # 0 = left, 1 = right
        for t in range(7):  # top-level Strassen product
            a_blocks, b_blocks, c_blocks = top_level[t]
            for s in range(7):  # inner Strassen term
                a_inner, b_inner, c_inner = strassen_terms[s]
                # A part
                for (abr, abc), ca in a_blocks.items():
                    for (air, aic), ci in a_inner.items():
                        a_row = abr * 2 + air
                        a_col = abc * 2 + aic
                        u_idx = a_row * 4 + a_col
                        U[u_idx][col] += ca * ci
                # B part
                for (bbr, bbc), cb in b_blocks.items():
                    for (bir, bic), ci in b_inner.items():
                        b_row = bbr * 2 + bir
                        b_col = bbc * 2 + bic
                        v_idx = b_row * 8 + (b_col + half * 4)
                        V[v_idx][col] += cb * ci
                # C part
                for (cbr, cbc), cc in c_blocks.items():
                    for (cir, cic), ci in c_inner.items():
                        c_row = cbr * 2 + cir
                        c_col = cbc * 2 + cic
                        w_idx = (c_col + half * 4) * 4 + c_row
                        W[w_idx][col] += cc * ci
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
