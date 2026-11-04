import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank-49 decomposition via Strassen on 2x2 blocks.
    # This is a structural change from the rank-64 schoolbook decomposition.
    
    n, m, p = DIMENSIONS
    # We are multiplying 4x4 by 4x4.
    # Split into 2x2 blocks of 2x2 submatrices.
    # Let A = [[A11, A12], [A21, A22]] and B = [[B11, B12], [B21, B22]]
    # where each block is 2x2.
    # C = A*B = [[C11, C12], [C21, C22]]
    
    # Strassen formulas for 2x2 block multiplication:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mt is a 2x2 matrix product of 2x2 linear combinations.
    # Since the submatrices are 2x2, each Mt requires a 2x2 matrix multiplication.
    # A 2x2 matrix multiplication has rank 7 (Strassen).
    # So total rank = 7 (outer Strassen terms) * 7 (inner Strassen rank) = 49.
    
    # We need to construct U, V, W matrices of shapes:
    # U: (n*m, rank) = (16, 49)
    # V: (m*p, rank) = (16, 49)
    # W: (p*n, rank) = (16, 49)
    
    # The index convention for U is row = i*m + j (i in 0..n-1, j in 0..m-1)
    # For A (n x m), A[i, j] is at row i*m + j.
    # Wait, the problem says:
    # U shape (n*m, rank). U[i*m+j, c] corresponds to coefficient of A[i, j].
    # V shape (m*p, rank). V[j*p+k, c] corresponds to coefficient of B[j, k].
    # W shape (p*n, rank). W[k*n+i, c] corresponds to coefficient of C[i, k].
    # Note: C = A * B. C[i, k] = sum_j A[i, j] B[j, k].
    # The trace convention is W[k*n+i, r]. This means the row index in W is k*n + i.
    # This corresponds to C[i, k] because the row index is derived from k (col of C) and i (row of C).
    # Let's verify: W[k*n+i, r] is the coefficient for the term involving C[i, k].
    
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Define the 7 outer Strassen terms.
    # Each term t corresponds to a product X_t * Y_t.
    # X_t is a linear combination of A blocks.
    # Y_t is a linear combination of B blocks.
    # The result M_t is a 2x2 matrix.
    
    # We need to map the 2x2 blocks to the flat indices of A, B, C.
    # A is 4x4. Blocks:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    #
    # Flat index for A[i, j] is i*4 + j.
    
    # Let's define the linear forms for X_t and Y_t in terms of the flat indices.
    # And the reconstruction for C blocks.
    
    # Inner Strassen for 2x2 matrices.
    # We need to decompose the product of two 2x2 matrices.
    # Let X = [[x00, x01], [x10, x11]] and Y = [[y00, y01], [y10, y11]].
    # Z = X * Y = [[z00, z01], [z10, z11]].
    # Strassen rank-7 decomposition for 2x2:
    # N1 = (x00 + x11) * (y00 + y11)
    # N2 = (x10 + x11) * y00
    # N3 = x00 * (y01 - y11)
    # N4 = x11 * (y10 - y00)
    # N5 = (x00 + x01) * y11
    # N6 = (x10 - x00) * (y00 + y01)
    # N7 = (x01 - x11) * (y10 + y11)
    #
    # z00 = N1 + N4 - N5 + N7
    # z01 = N3 + N5
    # z10 = N2 + N4
    # z11 = N1 - N2 + N3 + N6
    
    # For each outer term t (0..6), we have an inner rank-7 decomposition (0..6).
    # Total columns = 7 * 7 = 49.
    
    # Outer terms:
    # t=0: M1 = (A11 + A22) * (B11 + B22)
    # t=1: M2 = (A21 + A22) * B11
    # t=2: M3 = A11 * (B12 - B22)
    # t=3: M4 = A22 * (B21 - B11)
    # t=4: M5 = (A11 + A12) * B22
    # t=5: M6 = (A21 - A11) * (B11 + B12)
    # t=6: M7 = (A12 - A22) * (B21 + B22)
    
    # For each term t, X_t is a 2x2 matrix, Y_t is a 2x2 matrix.
    # We need to express X_t and Y_t in terms of the entries of A and B.
    
    # Let's define the coefficients for X_t and Y_t.
    # X_t entries: x00, x01, x10, x11.
    # Y_t entries: y00, y01, y10, y11.
    
    # We will build U, V, W by iterating over t and inner term l.
    # For a fixed t and l:
    # Column c = t * 7 + l.
    # The contribution to C is:
    # C += (Reconstruction of M_t) * (Inner Strassen l contribution to X_t * Y_t)
    
    # Let's formalize.
    # C[i, k] = sum_{t=0..6} sum_{l=0..6} coeff_W(t, l, i, k) * (X_t * Y_t)_{p, q} * ...
    # Wait, M_t is a 2x2 matrix. The entries of M_t are z00, z01, z10, z11.
    # The reconstruction of C from M_t is linear.
    # C11 = M1_00 + M4_00 - M5_00 + M7_00 ? No, M_t are matrices.
    # C11 (2x2 block) = M1 (2x2) + M4 (2x2) - M5 (2x2) + M7 (2x2).
    # So C11[a, b] = M1[a, b] + M4[a, b] - M5[a, b] + M7[a, b].
    
    # So for each C entry, we sum over t and inner l.
    # C[i, k] = sum_{t} sum_{l} W_coeff[t, l, i, k] * (X_t * Y_t)_{a, b}
    # where (a, b) is the position within the 2x2 block corresponding to (i, k).
    
    # Let's define the mapping from (i, k) to block (r, c) and position (a, b).
    # i is row 0..3, k is col 0..3.
    # r = i // 2, c = k // 2.
    # a = i % 2, b = k % 2.
    
    # Now, (X_t * Y_t)_{a, b} = sum_{l=0..6} U2[a, l] V2[b, l] ... wait.
    # The Strassen decomposition says:
    # (X * Y)_{a, b} = sum_{l=0..6} u2[a, l] * v2[b, l] * w2[?]
    # Actually, the tensor decomposition is:
    # T_{i, j, k} = sum_l U_{i, l} V_{j, l} W_{k, l}
    # Here, for the inner 2x2 multiplication:
    # (X * Y)_{a, b} = sum_{l=0..6} U2[a, l] V2[b, l] W2_inner[a, b, l]
    # Wait, the standard tensor form is C_{i, j} = sum_k A_{i, k} B_{k, j}.
    # Here A=X, B=Y, C=Z.
    # Z_{a, b} = sum_{c=0..1} X_{a, c} Y_{c, b}.
    # So the tensor is T_{a, c, b}.
    # U2[a, l], V2[c, l], W2[b, l]? No, W index is for the output C.
    # In the problem statement: W[k*n+i, r].
    # For the inner product, the "output" is Z_{a, b}.
    # So the index in W would be b*2 + a? Or a*2 + b?
    # The problem says W[k*n+i, r] for C[i, k].
    # So for inner, W[b*2+a, l].
    
    # Let's define U2, V2, W2 for the inner 2x2 Strassen.
    # U2 shape (2, 7), V2 shape (2, 7), W2 shape (2, 7)? No, V2 is for B.
    # B is 2x2. B_{c, b}. So V2 index is c*2 + b? No, V shape is (m*p, rank).
    # For inner, m=2, p=2. So V2 shape (4, 7).
    # V2[c*2+b, l].
    # U2 shape (2*2, 7)? No, U shape is (n*m, rank).
    # For inner, n=2, m=2. So U2 shape (4, 7).
    # U2[a*2+c, l].
    # W2 shape (2*2, 7) = (4, 7).
    # W2[b*2+a, l].
    
    # Let's construct U2, V2, W2 for 2x2 Strassen.
    # Strassen terms:
    # N1: (x00+x11)(y00+y11)
    # N2: (x10+x11)y00
    # N3: x00(y01-y11)
    # N4: x11(y10-y00)
    # N5: (x00+x01)y11
    # N6: (x10-x00)(y00+y01)
    # N7: (x01-x11)(y10+y11)
    
    # Z00 = N1 + N4 - N5 + N7
    # Z01 = N3 + N5
    # Z10 = N2 + N4
    # Z11 = N1 - N2 + N3 + N6
    
    # U2[a*2+c, l]:
    # a=0, c=0 (x00):
    #   l0: 1 (from N1)
    #   l2: 1 (from N3)
    #   l4: 1 (from N5)
    #   l5: -1 (from N6)
    # a=0, c=1 (x01):
    #   l4: 1 (from N5)
    #   l6: 1 (from N7)
    # a=1, c=0 (x10):
    #   l1: 1 (from N2)
    #   l5: 1 (from N6)
    # a=1, c=1 (x11):
    #   l0: 1 (from N1)
    #   l1: 1 (from N2)
    #   l3: 1 (from N4)
    #   l6: -1 (from N7)
    
    # V2[c*2+b, l]:
    # c=0, b=0 (y00):
    #   l0: 1 (from N1)
    #   l1: 1 (from N2)
    #   l3: -1 (from N4)
    #   l5: 1 (from N6)
    # c=0, b=1 (y01):
    #   l2: 1 (from N3)
    #   l5: 1 (from N6)
    # c=1, b=0 (y10):
    #   l3: 1 (from N4)
    #   l6: 1 (from N7)
    # c=1, b=1 (y11):
    #   l0: 1 (from N1)
    #   l2: -1 (from N3)
    #   l4: 1 (from N5)
    #   l6: 1 (from N7)
    
    # W2[b*2+a, l]:
    # b=0, a=0 (Z00):
    #   l0: 1 (N1)
    #   l3: 1 (N4)
    #   l4: -1 (N5)
    #   l6: 1 (N7)
    # b=1, a=0 (Z01):
    #   l2: 1 (N3)
    #   l4: 1 (N5)
    # b=0, a=1 (Z10):
    #   l1: 1 (N2)
    #   l3: 1 (N4)
    # b=1, a=1 (Z11):
    #   l0: 1 (N1)
    #   l1: -1 (N2)
    #   l2: 1 (N3)
    #   l5: 1 (N6)
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Fill U2
    U2[0, 0] = 1; U2[0, 2] = 1; U2[0, 4] = 1; U2[0, 5] = -1
    U2[1, 4] = 1; U2[1, 6] = 1
    U2[2, 1] = 1; U2[2, 5] = 1
    U2[3, 0] = 1; U2[3, 1] = 1; U2[3, 3] = 1; U2[3, 6] = -1
    
    # Fill V2
    V2[0, 0] = 1; V2[0, 1] = 1; V2[0, 3] = -1; V2[0, 5] = 1
    V2[1, 2] = 1; V2[1, 5] = 1
    V2[2, 3] = 1; V2[2, 6] = 1
    V2[3, 0] = 1; V2[3, 2] = -1; V2[3, 4] = 1; V2[3, 6] = 1
    
    # Fill W2
    W2[0, 0] = 1; W2[0, 3] = 1; W2[0, 4] = -1; W2[0, 6] = 1
    W2[1, 2] = 1; W2[1, 4] = 1
    W2[2, 1] = 1; W2[2, 3] = 1
    W2[3, 0] = 1; W2[3, 1] = -1; W2[3, 2] = 1; W2[3, 5] = 1
    
    # Now define outer terms.
    # For each t, we need to map the entries of X_t and Y_t to A and B.
    # X_t is a 2x2 matrix. Its entries are linear combinations of A blocks.
    # Y_t is a 2x2 matrix. Its entries are linear combinations of B blocks.
    
    # Let's define the coefficients for X_t and Y_t.
    # X_t entries: x00, x01, x10, x11.
    # Y_t entries: y00, y01, y10, y11.
    
    # We will create a mapping for each t.
    # For a given t, and for each entry of X_t (e.g., x00),
    # we need to know which entries of A it depends on, and with what coefficients.
    # Similarly for Y_t and B.
    
    # Let's use a dictionary or a list of tuples.
    # For X_t: list of (coeff, a_block, a_row, a_col) ?
    # Actually, it's easier to just iterate over the 16 entries of A and B.
    
    # For each t, let's define the linear forms for X_t and Y_t.
    # X_t[a, c] = sum_{i,j} coeff_X[t, a, c, i, j] A[i, j]
    # Y_t[c, b] = sum_{j,k} coeff_Y[t, c, b, j, k] B[j, k]
    
    # This is getting complicated to hardcode. Let's just build it programmatically.
    
    # Define blocks:
    # A11: [[A00, A01], [A20, A21]] -> indices 0, 1, 8, 9
    # A12: [[A02, A03], [A22, A23]] -> indices 2, 3, 10, 11
    # A21: [[A20, A21], [A30, A31]] -> indices 8, 9, 16, 17
    # A22: [[A22, A23], [A32, A33]] -> indices 10, 11, 18, 19
    
    # Wait, flat index i*4+j.
    # A11: i in 0..1, j in 0..1. Indices: 0, 1, 4, 5.
    # A12: i in 0..1, j in 2..3. Indices: 2, 3, 6, 7.
    # A21: i in 2..3, j in 0..1. Indices: 8, 9, 12, 13.
    # A22: i in 2..3, j in 2..3. Indices: 10, 11, 14, 15.
    
    # Let's define the linear forms for X_t and Y_t.
    # t=0: X0 = A11 + A22. Y0 = B11 + B22.
    # t=1: X1 = A21 + A22. Y1 = B11.
    # t=2: X2 = A11. Y2 = B12 - B22.
    # t=3: X3 = A22. Y3 = B21 - B11.
    # t=4: X4 = A11 + A12. Y4 = B22.
    # t=5: X5 = A21 - A11. Y5 = B11 + B12.
    # t=6: X6 = A12 - A22. Y6 = B21 + B22.
    
    # We can represent X_t as a 4x4 matrix of coefficients for A.
    # X_t[a, c] is a 2x2 block.
    # Actually, X_t is a 2x2 matrix of 2x2 blocks? No, X_t is a 2x2 matrix.
    # The entries of X_t are 2x2 matrices (blocks of A).
    # So X_t[a, c] is a 2x2 matrix.
    
    # Let's define coeff_X[t, a, c, i, j].
    # coeff_Y[t, c, b, j, k].
    
    coeff_X = np.zeros((7, 2, 2, 4, 4))
    coeff_Y = np.zeros((7, 2, 2, 4, 4))
    
    # Helper to add a block to coeff_X or coeff_Y
    def add_block(coeff, t, a, c, block_indices, sign):
        # block_indices is a list of 4 flat indices for the 2x2 block.
        # Order: 00, 01, 10, 11
        coeff[t, a, c, block_indices[0]] += sign
        coeff[t, a, c, block_indices[1]] += sign
        coeff[t, a, c, block_indices[2]] += sign
        coeff[t, a, c, block_indices[3]] += sign
    
    A11 = [0, 1, 4, 5]
    A12 = [2, 3, 6, 7]
    A21 = [8, 9, 12, 13]
    A22 = [10, 11, 14, 15]
    
    B11 = [0, 1, 4, 5]
    B12 = [2, 3, 6, 7]
    B21 = [8, 9, 12, 13]
    B22 = [10, 11, 14, 15]
    
    # t=0
    add_block(coeff_X, 0, 0, 0, A11, 1)
    add_block(coeff_X, 0, 0, 1, A11, 1)
    add_block(coeff_X, 0, 1, 0, A22, 1)
    add_block(coeff_X, 0, 1, 1, A22, 1)
    add_block(coeff_Y, 0, 0, 0, B11, 1)
    add_block(coeff_Y, 0, 0, 1, B11, 1)
    add_block(coeff_Y, 0, 1, 0, B22, 1)
    add_block(coeff_Y, 0, 1, 1, B22, 1)
    
    # t=1
    add_block(coeff_X, 1, 0, 0, A21, 1)
    add_block(coeff_X, 1, 0, 1, A21, 1)
    add_block(coeff_X, 1, 1, 0, A22, 1)
    add_block(coeff_X, 1, 1, 1, A22, 1)
    add_block(coeff_Y, 1, 0, 0, B11, 1)
    add_block(coeff_Y, 1, 0, 1, B11, 1)
    add_block(coeff_Y, 1, 1, 0, B11, 1)
    add_block(coeff_Y, 1, 1, 1, B11, 1)
    
    # t=2
    add_block(coeff_X, 2, 0, 0, A11, 1)
    add_block(coeff_X, 2, 0, 1, A11, 1)
    add_block(coeff_X, 2, 1, 0, A11, 1)
    add_block(coeff_X, 2, 1, 1, A11, 1)
    add_block(coeff_Y, 2, 0, 0, B12, 1)
    add_block(coeff_Y, 2, 0, 1, B12, 1)
    add_block(coeff_Y, 2, 1, 0, B22, -1)
    add_block(coeff_Y, 2, 1, 1, B22, -1)
    
    # t=3
    add_block(coeff_X, 3, 0, 0, A22, 1)
    add_block(coeff_X, 3, 0, 1, A22, 1)
    add_block(coeff_X, 3, 1, 0, A22, 1)
    add_block(coeff_X, 3, 1, 1, A22, 1)
    add_block(coeff_Y, 3, 0, 0, B21, 1)
    add_block(coeff_Y, 3, 0, 1, B21, 1)
    add_block(coeff_Y, 3, 1, 0, B11, -1)
    add_block(coeff_Y, 3, 1, 1, B11, -1)
    
    # t=4
    add_block(coeff_X, 4, 0, 0, A11, 1)
    add_block(coeff_X, 4, 0, 1, A11, 1)
    add_block(coeff_X, 4, 1, 0, A12, 1)
    add_block(coeff_X, 4, 1, 1, A12, 1)
    add_block(coeff_Y, 4, 0, 0, B22, 1)
    add_block(coeff_Y, 4, 0, 1, B22, 1)
    add_block(coeff_Y, 4, 1, 0, B22, 1)
    add_block(coeff_Y, 4, 1, 1, B22, 1)
    
    # t=5
    add_block(coeff_X, 5, 0, 0, A21, 1)
    add_block(coeff_X, 5, 0, 1, A21, 1)
    add_block(coeff_X, 5, 1, 0, A11, -1)
    add_block(coeff_X, 5, 1, 1, A11, -1)
    add_block(coeff_Y, 5, 0, 0, B11, 1)
    add_block(coeff_Y, 5, 0, 1, B11, 1)
    add_block(coeff_Y, 5, 1, 0, B12, 1)
    add_block(coeff_Y, 5, 1, 1, B12, 1)
    
    # t=6
    add_block(coeff_X, 6, 0, 0, A12, 1)
    add_block(coeff_X, 6, 0, 1, A12, 1)
    add_block(coeff_X, 6, 1, 0, A22, -1)
    add_block(coeff_X, 6, 1, 1, A22, -1)
    add_block(coeff_Y, 6, 0, 0, B21, 1)
    add_block(coeff_Y, 6, 0, 1, B21, 1)
    add_block(coeff_Y, 6, 1, 0, B22, 1)
    add_block(coeff_Y, 6, 1, 1, B22, 1)
    
    # Now build U, V, W.
    # For each t in 0..6, l in 0..6:
    # c = t*7 + l.
    # U[i*4+j, c] = sum_{a,c0} coeff_X[t, a, c0, i, j] * U2[a*2+c0, l]
    # V[j*4+k, c] = sum_{c0,b} coeff_Y[t, c0, b, j, k] * V2[c0*2+b, l]
    # W[k*4+i, c] = sum_{a,b} (Reconstruction coeff for C[i,k] from M_t[a,b]) * W2[b*2+a, l]
    
    # Reconstruction of C from M_t:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # M_t is a 2x2 matrix. M_t[a, b].
    # C[r, c] block is a 2x2 matrix.
    # C[r, c][a, b] = sum_t coeff_W_outer[t, r, c, a, b] * M_t[a, b]
    
    coeff_W_outer = np.zeros((7, 2, 2, 2, 2))
    # t=0 (M1)
    coeff_W_outer[0, 0, 0] = 1
    coeff_W_outer[0, 1, 1] = 1
    # t=1 (M2)
    coeff_W_outer[1, 1, 0] = 1
    coeff_W_outer[1, 1, 1] = -1
    # t=2 (M3)
    coeff_W_outer[2, 0, 1] = 1
    coeff_W_outer[2, 1, 1] = 1
    # t=3 (M4)
    coeff_W_outer[3, 0, 0] = 1
    coeff_W_outer[3, 1, 0] = 1
    # t=4 (M5)
    coeff_W_outer[4, 0, 0] = -1
    coeff_W_outer[4, 0, 1] = 1
    # t=5 (M6)
    coeff_W_outer[5, 1, 1] = 1
    # t=6 (M7)
    coeff_W_outer[6, 0, 0] = 1
    
    # Note: coeff_W_outer[t, r, c] is a 2x2 matrix?
    # My indexing: coeff_W_outer[t, r, c] was a scalar in the above code?
    # No, I did coeff_W_outer[t, r, c] = val. This sets the whole 2x2 block?
    # No, numpy scalar assignment to a 4D array slice?
    # coeff_W_outer shape is (7, 2, 2, 2, 2).
    # coeff_W_outer[t, r, c] is a 2x2 slice.
    # So coeff_W_outer[0, 0, 0] = 1 sets all 4 elements to 1. Correct.
    
    # Now fill U, V, W.
    for t in range(7):
        for l in range(7):
            c = t * 7 + l
            
            # U
            for a in range(2):
                for c0 in range(2):
                    u_val = U2[a*2+c0, l]
                    if u_val != 0:
                        # Add u_val * coeff_X[t, a, c0, :, :] to U[:, c]
                        U[:, c] += u_val * coeff_X[t, a, c0, :, :].flatten()
            
            # V
            for c0 in range(2):
                for b in range(2):
                    v_val = V2[c0*2+b, l]
                    if v_val != 0:
                        V[:, c] += v_val * coeff_Y[t, c0, b, :, :].flatten()
            
            # W
            for r in range(2):
                for c_block in range(2):
                    # C[r, c_block] is a 2x2 block.
                    # i = r*2 + a, k = c_block*2 + b
                    w_block = coeff_W_outer[t, r, c_block]
                    if np.any(w_block != 0):
                        for a in range(2):
                            for b in range(2):
                                w_val = W2[b*2+a, l]
                                if w_val != 0:
                                    coeff = w_block[a, b]
                                    if coeff != 0:
                                        i = r*2 + a
                                        k = c_block*2 + b
                                        W_idx = k*4 + i
                                        W[W_idx, c] += coeff * w_val
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
