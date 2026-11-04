"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Block decomposition: partition 5x5 into 3x3 and 2x2 blocks.
    # Use Laderman (rank 17) for 3x3, Strassen (rank 7) for 2x2,
    # and schoolbook for the 4 cross-block products.
    n, m, p = DIMENSIONS
    
    # U has shape (n*m, rank) = (25, rank)
    # V has shape (m*p, rank) = (25, rank)
    # W has shape (p*n, rank) = (25, rank)
    
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add one rank-1 term to the decomposition."""
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
    
    # Block indices:
    # A[i,j] where i in 0..4, j in 0..4
    # A11: i in 0..2, j in 0..2 (3x3)
    # A12: i in 0..2, j in 3..4 (3x2)
    # A21: i in 3..4, j in 0..2 (2x3)
    # A22: i in 3..4, j in 3..4 (2x2)
    # Similarly for B and C.
    
    # Laderman's rank-17 decomposition for 3x3 matrix multiplication.
    # C11 = A11 * B11 where A11, B11, C11 are 3x3.
    # The 17 terms for Laderman:
    laderman_terms = [
        # Each term: (A coefficients, B coefficients, C coefficients)
        # A is 3x3, B is 3x3, C is 3x3
        # Represented as flat lists of length 9 in row-major order.
        
        # M1 = A11 * B11
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        
        # M2 = A12 * B21
        ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        
        # M3 = A11 - A12
        ([1,-1,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        
        # M4 = B11 + B12
        ([0,0,0, 0,0,0, 0,0,0], [1,1,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        
        # M5 = A11 - A21
        ([1,0,0, 0,0,-1, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        
        # M6 = B21 + B11
        ([0,0,0, 0,0,0, 0,0,0], [1,0,0, 1,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        
        # M7 = A22 - A21
        ([0,0,0, 0,0,-1, 0,0,1], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        
        # M8 = B12 - B11
        ([0,0,0, 0,0,0, 0,0,0], [1,-1,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        
        # M9 = A22 - A12
        ([0,0,-1, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        
        # M10 = B12 - B22
        ([0,0,0, 0,0,0, 0,0,0], [0,1,-1, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        
        # M11 = A21 - A22
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,-1, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        
        # M12 = B22 - B12
        ([0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,-1], [0,0,0, 0,0,1, 0,0,0]),
        
        # M13 = A11 + A12
        ([1,1,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        
        # M14 = B11 - B12
        ([0,0,0, 0,0,0, 0,0,0], [1,-1,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        
        # M15 = A22 + A12
        ([0,0,1, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
        
        # M16 = B22 + B12
        ([0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,1], [0,0,1, 0,0,0, 0,0,0]),
        
        # M17 = A11 - A21 + A22
        ([1,0,0, 0,0,-1, 0,0,1], [1,0,0, 0,0,0, 0,0,0], [1,0,1, 0,0,0, 0,0,0]),
    ]
    
    # This is getting complex. Let me use a known explicit Laderman decomposition.
    # Actually, let me just use the standard Laderman rank-17 decomposition properly.
    
    # For simplicity and correctness, let me use a different approach:
    # Use the known minimal rank decompositions directly.
    
    # Actually, the simplest correct approach: use Strassen for the whole thing by
    # padding to 8x8? No, that would be rank 63 which is worse than 84.
    
    # Let me just implement the block approach carefully.
    
    # Reset
    U = []
    V = []
    W = []
    
    def vec25(row, col):
        """Create a 25-element vector with 1 at position row*m+col, 0 elsewhere."""
        v = [0] * 25
        v[row * 5 + col] = 1
        return v
    
    def add_rank1(iA, jA, iB, jB, iC, jC, a=1, b=1, c=1):
        """Add a rank-1 term: a * A[iA,jA] * b * B[iB,jB] -> c * C[iC,jC]
        where indices are in the 5x5 grid."""
        u = [0] * 25
        v = [0] * 25
        w = [0] * 25
        u[iA * 5 + jA] = a
        v[iB * 5 + jB] = b
        w[iC * 5 + jC] = c
        U.append(u)
        V.append(v)
        W.append(w)
    
    # Strassen for 2x2 blocks
    # For block (r1,r2) x rows, (c1,c2) x cols
    def strassen_2x2(row_start, col_start):
        """Add Strassen rank-7 terms for 2x2 multiplication at given block position."""
        r1, r2 = row_start, row_start + 1
        c1, c2 = col_start, col_start + 1
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # Each M is a product of two 2x2 matrices, giving a 2x2 result.
        # We need to express each M as rank-1 terms in the full 5x5 space.
        
        # M1 = (A[r1,r1]+A[r2,r2]) * (B[r1,c1]+B[r2,c2])
        # This is (A11+A22)(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
        # But in Strassen, M1 is a single matrix product of two 2x2 matrices.
        # We need to decompose M1 (a 2x2 matrix) into rank-1 terms for the bilinear form.
        
        # Actually, each M in Strassen is itself a 2x2 matrix multiplication, which
        # can be done in rank 4 (schoolbook) or rank 7 (Strassen again, but that's recursive).
        # For a 2x2 product, the minimum rank is 7 (Strassen), but we're already using Strassen.
        
        # Let me use schoolbook for the 2x2 products within Strassen:
        # M1 contributes 4 rank-1 terms
        # M2 contributes 4
        # etc.
        # Total: 7 * 4 = 28 terms per 2x2 Strassen block.
        
        # That's not good. Let me think again.
        
        # Actually, the Strassen decomposition for 2x2x2 has rank 7, meaning 7 scalar multiplications.
        # Each scalar multiplication gives a rank-1 term in the tensor.
        
        # The 7 terms for Strassen 2x2:
        # Let A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]], C = AB
        # m1 = (a11+a22)(b11+b22)
        # m2 = (a21+a22)b11
        # m3 = a11(b21-b22)
        # m4 = a22(b12-b11)
        # m5 = (a11+a12)b22
        # m6 = (a21-a11)(b11+b12)
        # m7 = (a12-a22)(b21+b22)
        # c11 = m1+m4-m5+m7
        # c12 = m3+m5
        # c21 = m2+m4
        # c22 = m1-m2+m3+m6
        
        # Each mi is a product of linear forms in A and B, giving a contribution to C.
        # For the bilinear form tensor, each mi corresponds to a rank-1 term where:
        # - The A part is a linear combination of A entries
        # - The B part is a linear combination of B entries
        # - The C part is a linear combination of C entries
        
        # For m1 = (a11+a22)(b11+b22):
        # A-part: [1,0,0,1] (for a11, a12, a21, a22)
        # B-part: [1,0,0,1]
        # C-part: c11 += 1, c12 += 0, c21 += 0, c22 += 1
        # So C-part: [1,0,0,1]
        
        # Let me encode the 7 Strassen terms for a 2x2 block.
        
        strassen_coeffs = [
            # (A coeffs, B coeffs, C coeffs) each of length 4 for 2x2
            # m1
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            # m2
            ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
            # m3
            ([1,0,0,0], [0,0,1,-1], [0,1,0,0]),
            # m4
            ([0,0,0,1], [0,1,-1,0], [1,0,0,0]),
            # m5
            ([1,1,0,0], [0,0,0,1], [0,1,0,0]),
            # m6
            ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
            # m7
            ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
        ]
        
        for a_coeffs, b_coeffs, c_coeffs in strassen_coeffs:
            u = [0] * 25
            v = [0] * 25
            w = [0] * 25
            # Map 2x2 local indices to global 5x5 indices
            local_to_global = [(r1,c1), (r1,c2), (r2,c1), (r2,c2)]
            for li, val in enumerate(a_coeffs):
                if val != 0:
                    gr, gc = local_to_global[li]
                    u[gr * 5 + gc] = val
            for li, val in enumerate(b_coeffs):
                if val != 0:
                    gr, gc = local_to_global[li]
                    v[gr * 5 + gc] = val
            for li, val in enumerate(c_coeffs):
                if val != 0:
                    gr, gc = local_to_global[li]
                    w[gr * 5 + gc] = val
            U.append(u)
            V.append(v)
            W.append(w)
    
    # Laderman for 3x3 blocks - this is more complex.
    # For now, let me just use schoolbook for everything and see if I can do better.
    
    # Actually, let me just use schoolbook for all terms.
    # The schoolbook rank is 125. Let me at least try to do the block approach
    # with schoolbook for cross terms and Strassen for the 2x2 diagonal blocks.
    
    # Reset
    U = []
    V = []
    W = []
    
    # 1. A11 * B11 -> C11 (3x3 * 3x3 = 3x3) - use schoolbook, rank 27
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u = [0] * 25
                v = [0] * 25
                w = [0] * 25
                u[i * 5 + j] = 1
                v[(j + 3) * 5 + (k + 3)] = 1  # B11 is at rows 0-2, cols 0-2... 
                # Wait, B11 is B[0:3, 0:3], so rows 0-2, cols 0-2.
                # A11 is A[0:3, 0:3], rows 0-2, cols 0-2.
                # C11 is C[0:3, 0:3], rows 0-2, cols 0-2.
                # So for A[i,j] * B[j,k] -> C[i,k]:
                # A is at (i, j), B is at (j, k), C is at (i, k)
                # But B's row index is j (0-2) and col index is k (0-2).
                u = [0] * 25
                v = [0] * 25
                w = [0] * 25
                u[i * 5 + j] = 1
                v[j * 5 + k] = 1
                w[i * 5 + k] = 1
                U.append(u)
                V.append(v)
                W.append(w)
    
    # 2. A12 * B12 -> C12 (3x2 * 2x2 = 3x2) - schoolbook, rank 12
    for i in range(3):
        for j in range(2):
            for k in range(2):
                u = [0] * 25
                v = [0] * 25
                w = [0] * 25
                # A12: rows 0-2, cols 3-4
                # B12: rows 3-4, cols 3-4
                # C12: rows 0-2, cols 3-4
                # A[i, j+3] * B[j+3, k+3] -> C[i, k+3]
                u[i * 5 + (j + 3)] = 1
                v[(j + 3) * 5 + (k + 3)] = 1
                w[i * 5 + (k + 3)] = 1
                U.append(u)
                V.append(v)
                W.append(w)
    
    # 3. A21 * B21 -> C21 (2x3 * 3x3 = 2x3) - schoolbook, rank 18
    for i in range(2):
        for j in range(3):
            for k in range(3):
                u = [0] * 25
                v = [0] * 25
                w = [0] * 25
                # A21: rows 3-4, cols 0-2
                # B21: rows 0-2, cols 3-4
                # C21: rows 3-4, cols 0-2
                # A[i+3, j] * B[j, k+3] -> C[i+3, k]
                u[(i + 3) * 5 + j] = 1
                v[j * 5 + (k + 3)] = 1
                w[(i + 3) * 5 + k] = 1
                U.append(u)
                V.append(v)
                W.append(w)
    
    # 4. A22 * B22 -> C22 (2x2 * 2x2 = 2x2) - use Strassen, rank 7
    strassen_2x2(3, 3)
    
    # 5. Cross terms for C11: A11*B11 (done), A12*B21
    # C11[i,k] = sum_j A11[i,j]*B11[j,k] + sum_j A12[i,j]*B21[j,k]
    # A12*B21: A12 is 3x2 (rows 0-2, cols 3-4), B21 is 2x3 (rows 3-4, cols 0-2)
    # Product is 3x3: C11[i,k] += sum_{j=0}^{1} A12[i,j]*B21[j,k]
    # A12[i, j] is at (i, j+3), B21[j, k] is at (j+3, k)
    for i in range(3):
        for j in range(2):
            for k in range(3):
                u = [0] * 25
                v = [0] * 25
                w = [0] * 25
                u[i * 5 + (j + 3)] = 1
                v[(j + 3) * 5 + k] = 1
                w[i * 5 + k] = 1
                U.append(u)
                V.append(v)
                W.append(w)
    
    # 6. Cross terms for C12: A11*B12 (done), A12*B22
    # C12[i,k] = sum_j A11[i,j]*B12[j,k] + sum_j A12[i,j]*B22[j,k]
    # A12*B22: A12 is 3x2, B22 is 2x2
    # A12[i, j] at (i, j+3), B22[j, k] at (j+3, k+3)
    for i in range(3):
        for j in range(2):
            for k in range(2):
                u = [0] * 25
                v = [0] * 25
                w = [0] * 25
                u[i * 5 + (j + 3)] = 1
                v[(j + 3) * 5 + (k + 3)] = 1
                w[i * 5 + (k + 3)] = 1
                U.append(u)
                V.append(v)
                W.append(w)
    
    # 7. Cross terms for C21: A21*B11, A22*B21
    # C21[i,k] = sum_j A21[i,j]*B11[j,k] + sum_j A22[i,j]*B21[j,k]
    # A21*B11: A21 is 2x3 (rows 3-4, cols 0-2), B11 is 3x3 (rows 0-2, cols 0-2)
    # A21[i, j] at (i+3, j), B11[j, k] at (j, k)
    for i in range(2):
        for j in range(3):
            for k in range(3):
                u = [0] * 25
                v = [0] * 25
                w = [0] * 25
                u[(i + 3) * 5 + j] = 1
                v[j * 5 + k] = 1
                w[(i + 3) * 5 + k] = 1
                U.append(u)
                V.append(v)
                W.append(w)
    
    # A22*B21: A22 is 2x2 (rows 3-4, cols 3-4), B21 is 2x3 (rows 3-4, cols 0-2)
    # A22[i, j] at (i+3, j+3), B21[j, k] at (j+3, k)
    for i in range(2):
        for j in range(2):
            for k in range(3):
                u = [0] * 25
                v = [0] * 25
                w = [0] * 25
                u[(i + 3) * 5 + (j + 3)] = 1
                v[(j + 3) * 5 + k] = 1
                w[(i + 3) * 5 + k] = 1
                U.append(u)
                V.append(v)
                W.append(w)
    
    # 8. Cross terms for C22: A21*B12, A22*B22 (Strassen handles A22*B22)
    # C22[i,k] = sum_j A21[i,j]*B12[j,k] + sum_j A22[i,j]*B22[j,k]
    # A21*B12: A21 is 2x3 (rows 3-4, cols 0-2), B12 is 2x2 (rows 3-4, cols 3-4)
    # A21[i, j] at (i+3, j), B12[j, k] at (j+3, k+3)
    for i in range(2):
        for j in range(3):
            for k in range(2):
                u = [0] * 25
                v = [0] * 25
                w = [0] * 25
                u[(i + 3) * 5 + j] = 1
                v[(j + 3) * 5 + (k + 3)] = 1
                w[(i + 3) * 5 + (k + 3)] = 1
                U.append(u)
                V.append(v)
                W.append(w)
    
    # Count total rank
    rank = len(U)
    # Verify shapes
    assert len(U) == rank
    assert len(V) == rank
    assert len(W) == rank
    for r in range(rank):
        assert len(U[r]) == 25
        assert len(V[r]) == 25
        assert len(W[r]) == 25
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
