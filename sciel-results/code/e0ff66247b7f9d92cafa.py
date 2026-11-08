"""Matrix multiplication 2x4x7 with rank reduction using Strassen-style blocking."""
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 7
    DEADLINE = time.time() + 100  # Safety margin before 120s limit

    # Split columns into blocks of size 2, 2, 2, 1
    block_sizes = [2, 2, 2, 1]
    col_offsets = []
    offset = 0
    for bs in block_sizes:
        col_offsets.append(offset)
        offset += bs
    # col_offsets = [0, 2, 4, 6]

    rank = 50  # 3*7 + 1*4 = 21 + 4 = 25? No.
    # Let me re-count.
    # 3 blocks of 2x2x2: 7 * 3 = 21
    # 1 block of 2x2x1: 4 * 1 = 4
    # Total 25?
    # Wait, the previous response said 50. Let me check the math.
    # Strassen 2x2x2 is rank 7.
    # 3 blocks -> 21.
    # 1 block of 2x2x1.
    # For 2x2x1, we can just do schoolbook: 2*2*1 = 4.
    # Total 25.
    # Why did the previous response say 50?
    # "Rank 50 is guaranteed."
    # Maybe I can do better.
    # Let's stick to the 25 plan first. If it works, great.
    # Is 25 valid?
    # The blocks are disjoint in terms of B's columns.
    # The blocks are disjoint in terms of A's columns.
    # So we can decompose each block independently and sum the ranks.
    # Yes, rank is additive for block diagonal structures.
    # So rank = 7 + 7 + 7 + 4 = 25.

    # Let's verify the 2x2x1 block.
    # A is 2x2, B is 2x1, C is 2x1.
    # C_0 = A_00 B_0 + A_01 B_1
    # C_1 = A_10 B_0 + A_11 B_1
    # This is just 4 multiplications.
    # Can we do it in fewer?
    # Tensor rank of 2x2x1 is 4. (It's just the matrix multiplication tensor of that shape).
    # So yes, 4 is minimal for that block.

    # So total rank 25.
    # This is a huge improvement over 56.
    # Let's implement rank 25.

    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    col = 0

    # Strassen 2x2x2 coefficients
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Indices for a 2x2x2 block starting at col_offset
    # A indices: (0,0), (0,1), (1,0), (1,1) relative to block
    # Global A row: i * m + j
    # Global B col: k * 7 + l (wait, B is 4x7. Rows 0..3, Cols 0..6)
    # My block split was on columns of A and B.
    # A is 2x4. B is 4x7.
    # Block 1: A cols 0,1. B rows 0,1. (Since A col j multiplies B row j)
    # So B block is rows 0,1.
    # C block is cols corresponding to B rows? No.
    # C = A B.
    # C_i, k = sum_j A_i, j B_j, k
    # If we partition j (A cols, B rows), then for a fixed j-block, we compute contributions to all k?
    # No, B_j, k depends on j.
    # If we fix j in {0,1}, we use B rows 0,1.
    # Then C_i, k gets contributions from B_0,k and B_1,k.
    # So the B block is the submatrix B[0:2, :].
    # The A block is A[:, 0:2].
    # The result is A_block @ B_block.
    # A_block is 2x2. B_block is 2x7.
    # Result is 2x7.
    # So this is a 2x2x7 multiplication?
    # No, I split B into blocks of rows.
    # B is 4x7.
    # Block 1: B rows 0,1. Shape 2x7.
    # A block: A cols 0,1. Shape 2x2.
    # Product: 2x7.
    # This is not a square block multiplication.
    # Strassen applies to square blocks.
    # So I need to split B's columns too?
    # Or split the problem differently.

    # Alternative: Split B into two 2x7 blocks? No, that's just splitting the sum.
    # To use Strassen, I need 2x2 A, 2x2 B, 2x2 C.
    # So I need to split the 7 columns of C (and B) into blocks of 2?
    # 7 = 2 + 2 + 2 + 1.
    # So I have 3 blocks of (2x2 A) * (2x2 B) -> (2x2 C).
    # And 1 block of (2x2 A) * (2x1 B) -> (2x1 C).
    #
    # Let's trace the indices carefully.
    # A is 2x4. Indices (i, j) with i in 0..1, j in 0..3.
    # B is 4x7. Indices (j, k) with j in 0..3, k in 0..6.
    # C is 2x7. Indices (i, k) with i in 0..1, k in 0..6.
    #
    # We partition the summation index j into blocks.
    # Block 0: j in {0, 1}.
    #   A_sub = A[:, 0:2] (2x2)
    #   B_sub = B[0:2, :] (2x7) -> This is not square.
    #   We need to partition k as well to make B_sub square?
    #   If we partition k, say k in {0, 1}.
    #   Then B_sub = B[0:2, 0:2] (2x2).
    #   C_sub = C[:, 0:2] (2x2).
    #   This gives a 2x2x2 block.
    #
    # So we partition both j and k.
    # j partition: 0,1 | 2,3
    # k partition: 0,1 | 2,3 | 4,5 | 6
    #
    # This creates 2 * 4 = 8 blocks.
    # Block (j_block, k_block).
    # For j_block 0 (j=0,1):
    #   A_sub is A[:, 0:2].
    #   For k_block 0 (k=0,1): B_sub is B[0:2, 0:2]. C_sub is C[:, 0:2]. Rank 7.
    #   For k_block 1 (k=2,3): B_sub is B[0:2, 2:4]. C_sub is C[:, 2:4]. Rank 7.
    #   For k_block 2 (k=4,5): B_sub is B[0:2, 4:6]. C_sub is C[:, 4:6]. Rank 7.
    #   For k_block 3 (k=6): B_sub is B[0:2, 6:7]. C_sub is C[:, 6:7]. Rank 4.
    # For j_block 1 (j=2,3):
    #   A_sub is A[:, 2:4].
    #   For k_block 0: Rank 7.
    #   For k_block 1: Rank 7.
    #   For k_block 2: Rank 7.
    #   For k_block 3: Rank 4.
    #
    # Total Rank = 4 * (7+7+7+4) = 4 * 25 = 100.
    # This is worse than 56!
    #
    # Why? Because I'm duplicating the A_sub for each k_block.
    # The A_sub is the same for all k_blocks in a given j_block.
    # But Strassen's algorithm computes the product of the whole A_sub and whole B_block.
    # If B_block is 2x7, can we use a fast algorithm for 2x2 * 2x7?
    #
    # 2x2 * 2x7.
    # B is 2x7.
    # We can split B into three 2x2 blocks and one 2x1 block.
    # B = [B1, B2, B3, B4].
    # A_sub * B = A_sub*B1 + A_sub*B2 + A_sub*B3 + A_sub*B4.
    # Each A_sub*Bk is a 2x2 * 2x2 (or 2x1) product.
    #
    # For the 2x2 * 2x2 parts:
    # We can use Strassen. Rank 7 each.
    # For the 2x2 * 2x1 part:
    # Rank 4.
    #
    # So for one j_block (A is 2x2), we have 3 Strassen blocks and 1 schoolbook block.
    # Rank = 3*7 + 4 = 25.
    #
    # We have 2 j_blocks.
    # Total Rank = 2 * 25 = 50.
    #
    # This matches the previous response's 50!
    # My previous 25 calculation was wrong because I forgot that A is shared across k-blocks but the multiplication is independent for each k-block?
    # No, A_sub * B_full is the sum of A_sub * B_k.
    # Since A_sub is the same, can we reuse the Strassen intermediates?
    # Strassen computes M1...M7 which are linear combinations of A_sub.
    # Then C = M1...M7 combined with B.
    # The combination with B is linear in B.
    # So if B = B1 + B2 + ...
    # C = f(M, B1) + f(M, B2) + ...
    # The M's are the same for all B_k.
    # So we only compute M's once (rank 7).
    # Then for each B_k, we do the linear combination.
    # But the linear combination involves multiplications? No, the M's are already products.
    # Wait. Strassen's algorithm:
    # M1 = (A00+A11)(B00+B11)
    # ...
    # C00 = M1 + M4 - M5 + M7
    # The C's are sums of M's. No more multiplications.
    # So if we have multiple B's, say B1, B2.
    # C1 = f(M, B1)
    # C2 = f(M, B2)
    # The M's depend on B.
    # M1 = (A00+A11)(B1_00+B1_11) for B1.
    # M1' = (A00+A11)(B2_00+B2_11) for B2.
    # These are different M's.
    # So we can't reuse the M's.
    # We have to compute 7 multiplications for each 2x2 block of B.
    #
    # So yes, for each 2x2 block of B, we need 7 multiplications.
    # And for each 2x1 block of B, we need 4 multiplications.
    #
    # So for one j_block (2 rows of B):
    # B has 7 columns.
    # Split into 2,2,2,1.
    # 3 blocks of 2x2 -> 3 * 7 = 21 mults.
    # 1 block of 2x1 -> 1 * 4 = 4 mults.
    # Total 25 mults.
    #
    # We have 2 j_blocks (rows 0,1 and 2,3).
    # Total 50 mults.
    #
    # So rank 50 is correct.
    #
    # Can we do better?
    # Maybe a different split?
    # 4 = 2 + 2.
    # 7 = 3 + 4? No, Strassen needs powers of 2 or similar structures.
    #
    # What if we use a 4x4 Strassen variant?
    # A is 2x4. Not square.
    #
    # What if we treat it as (2x2) * (2x7) and (2x2) * (2x7).
    # Is there a fast algorithm for 2x2 * 2x7 with rank < 25?
    # 2x2 * 2x7 tensor.
    # This is equivalent to multiplying a 2x2 matrix by 7 column vectors.
    # Each column is a 2x1 vector.
    # So it's 7 instances of 2x2 * 2x1.
    # But they share the same A.
    #
    # Let A = [[a, b], [c, d]].
    # Let B columns be v1...v7. vi = [x_i, y_i]^T.
    # C columns are w_i = A vi.
    # w_i = [a x_i + b y_i, c x_i + d y_i]^T.
    #
    # We want to compute these 7 vectors.
    # This is a bilinear map A x V -> W.
    # Rank of this map?
    # We can precompute some combinations of A.
    # E.g., a+b, a-b, etc.
    # But y_i and x_i vary.
    #
    # If we compute a, b, c, d (4 mults? No, A is input).
    # The multiplication is the bilinear part.
    #
    # Consider the 2x2 * 2x2 case. Rank 7.
    # 2x2 * 2x1 case. Rank 4.
    #
    # Is it possible to do 2x2 * 2x7 in rank 20?
    # 20 / 7 = 2.85.
    # If we could batch 3 columns with 18 mults (3 * 6)?
    # Rank of 2x2 * 2x3?
    #
    # Let's stick with 50. It's a solid improvement from 56.
    #
    # Implementation details:
    # We iterate over j_blocks (0, 1).
    # For each j_block, we have A_sub (2x2).
    # We iterate over k_blocks (0, 1, 2, 3).
    # If k_block size is 2:
    #   We have B_sub (2x2).
    #   We perform Strassen.
    #   We update U, V, W.
    # If k_block size is 1:
    #   We have B_sub (2x1).
    #   We perform schoolbook.
    #   We update U, V, W.
    #
    # Indices:
    # j_block 0: j in {0, 1}. A rows i in {0, 1}.
    #   A global index: i * 4 + j.
    #   B global index: j * 7 + k.
    #   C global index: k * 2 + i. (Trace convention W[k*n+i, r])
    #
    # j_block 1: j in {2, 3}.
    #   A global index: i * 4 + j.
    #   B global index: j * 7 + k.
    #   C global index: k * 2 + i.
    #
    # k_blocks:
    # 0: k in {0, 1}. Size 2.
    # 1: k in {2, 3}. Size 2.
    # 2: k in {4, 5}. Size 2.
    # 3: k in {6}. Size 1.
    #
    # For a Strassen block (j_block, k_block):
    #   A00 = A[i0, j0] (global i0*m + j0)
    #   A01 = A[i0, j1]
    #   A10 = A[i1, j0]
    #   A11 = A[i1, j1]
    #
    #   B00 = B[j0, k0] (global j0*p + k0)
    #   B01 = B[j0, k1]
    #   B10 = B[j1, k0]
    #   B11 = B[j1, k1]
    #
    #   C00 = C[i0, k0] (global k0*n + i0)
    #   C01 = C[i0, k1]
    #   C10 = C[i1, k0]
    #   C11 = C[i1, k1]
    #
    #   M1 = (A00 + A11) (B00 + B11)
    #   M2 = (A10 + A11) B00
    #   M3 = A00 (B01 - B11)
    #   M4 = A11 (B10 - B00)
    #   M5 = (A00 + A01) B11
    #   M6 = (A10 - A00) (B00 + B01)
    #   M7 = (A01 - A11) (B10 + B11)
    #
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    #
    # We need to map these to U, V, W.
    # U has shape (n*m, rank) = (8, rank).
    # V has shape (m*p, rank) = (28, rank).
    # W has shape (p*n, rank) = (14, rank).
    #
    # Each term Mx corresponds to a product u * v = w.
    # u is a linear combo of A's.
    # v is a linear combo of B's.
    # w is added to a linear combo of C's.
    #
    # For M1:
    # u = A00 + A11.
    #   U[A00_idx][col] += 1
    #   U[A11_idx][col] += 1
    # v = B00 + B11.
    #   V[B00_idx][col] += 1
    #   V[B11_idx][col] += 1
    # w = C00 + C01? No.
    # M1 contributes to C00 (+1) and C11 (+1).
    #   W[C00_idx][col] += 1
    #   W[C11_idx][col] += 1
    #
    # For M2:
    # u = A10 + A11.
    #   U[A10_idx][col] += 1
    #   U[A11_idx][col] += 1
    # v = B00.
    #   V[B00_idx][col] += 1
    # w = C10 (+1) and C11 (-1).
    #   W[C10_idx][col] += 1
    #   W[C11_idx][col] -= 1
    #
    # For M3:
    # u = A00.
    #   U[A00_idx][col] += 1
    # v = B01 - B11.
    #   V[B01_idx][col] += 1
    #   V[B11_idx][col] -= 1
    # w = C01 (+1) and C11 (+1).
    #   W[C01_idx][col] += 1
    #   W[C11_idx][col] += 1
    #
    # For M4:
    # u = A11.
    #   U[A11_idx][col] += 1
    # v = B10 - B00.
    #   V[B10_idx][col] += 1
    #   V[B00_idx][col] -= 1
    # w = C00 (+1) and C10 (+1).
    #   W[C00_idx][col] += 1
    #   W[C10_idx][col] += 1
    #
    # For M5:
    # u = A00 + A01.
    #   U[A00_idx][col] += 1
    #   U[A01_idx][col] += 1
    # v = B11.
    #   V[B11_idx][col] += 1
    # w = C00 (-1) and C01 (+1).
    #   W[C00_idx][col] -= 1
    #   W[C01_idx][col] += 1
    #
    # For M6:
    # u = A10 - A00.
    #   U[A10_idx][col] += 1
    #   U[A00_idx][col] -= 1
    # v = B00 + B01.
    #   V[B00_idx][col] += 1
    #   V[B01_idx][col] += 1
    # w = C11 (+1).
    #   W[C11_idx][col] += 1
    #
    # For M7:
    # u = A01 - A11.
    #   U[A01_idx][col] += 1
    #   U[A11_idx][col] -= 1
    # v = B10 + B11.
    #   V[B10_idx][col] += 1
    #   V[B11_idx][col] += 1
    # w = C00 (+1).
    #   W[C00_idx][col] += 1
    #
    # For the 2x2x1 block (j_block, k=6):
    # A is 2x2. B is 2x1. C is 2x1.
    # B0 = B[j0, 6], B1 = B[j1, 6].
    # C0 = C[i0, 6], C1 = C[i1, 6].
    #
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    #
    # 4 terms.
    # T1: A00 * B0 -> C0 (+1)
    # T2: A01 * B1 -> C0 (+1)
    # T3: A10 * B0 -> C1 (+1)
    # T4: A11 * B1 -> C1 (+1)

    # Helper to add a Strassen term
    def add_strassen_term(j_block, k_block, col):
        i0, i1 = 0, 1
        j0 = 2 * j_block
        j1 = j0 + 1
        k0 = 2 * k_block
        k1 = k0 + 1

        # Indices
        a00 = i0 * m + j0
        a01 = i0 * m + j1
        a10 = i1 * m + j0
        a11 = i1 * m + j1

        b00 = j0 * p + k0
        b01 = j0 * p + k1
        b10 = j1 * p + k0
        b11 = j1 * p + k1

        c00 = k0 * n + i0
        c01 = k0 * n + i1 # Wait. C index is k*n + i.
        # C00 is C[i0, k0]. Index k0*n + i0.
        # C01 is C[i0, k1]. Index k1*n + i0.
        # C10 is C[i1, k0]. Index k0*n + i1.
        # C11 is C[i1, k1]. Index k1*n + i1.
        
        c00_idx = k0 * n + i0
        c01_idx = k1 * n + i0
        c10_idx = k0 * n + i1
        c11_idx = k1 * n + i1

        # M1
        U[a00][col] += 1; U[a11][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00_idx][col] += 1; W[c11_idx][col] += 1

        # M2
        U[a10][col] += 1; U[a11][col] += 1
        V[b00][col] += 1
        W[c10_idx][col] += 1; W[c11_idx][col] -= 1

        # M3
        U[a00][col] += 1
        V[b01][col] += 1; V[b11][col] -= 1
        W[c01_idx][col] += 1; W[c11_idx][col] += 1

        # M4
        U[a11][col] += 1
        V[b10][col] += 1; V[b00][col] -= 1
        W[c00_idx][col] += 1; W[c10_idx][col] += 1

        # M5
        U[a00][col] += 1; U[a01][col] += 1
        V[b11][col] += 1
        W[c00_idx][col] -= 1; W[c01_idx][col] += 1

        # M6
        U[a10][col] += 1; U[a00][col] -= 1
        V[b00][col] += 1; V[b01][col] += 1
        W[c11_idx][col] += 1

        # M7
        U[a01][col] += 1; U[a11][col] -= 1
        V[b10][col] += 1; V[b11][col] += 1
        W[c00_idx][col] += 1

    # Helper to add a schoolbook 2x2x1 term
    def add_schoolbook_term(j_block, k, col):
        i0, i1 = 0, 1
        j0 = 2 * j_block
        j1 = j0 + 1

        a00 = i0 * m + j0
        a01 = i0 * m + j1
        a10 = i1 * m + j0
        a11 = i1 * m + j1

        b0 = j0 * p + k
        b1 = j1 * p + k

        c0 = k * n + i0
        c1 = k * n + i1

        # A00 * B0 -> C0
        U[a00][col] += 1
        V[b0][col] += 1
        W[c0][col] += 1

        # A01 * B1 -> C0
        U[a01][col] += 1
        V[b1][col] += 1
        W[c0][col] += 1

        # A10 * B0 -> C1
        U[a10][col] += 1
        V[b0][col] += 1
        W[c1][col] += 1

        # A11 * B1 -> C1
        U[a11][col] += 1
        V[b1][col] += 1
        W[c1][col] += 1

    col = 0
    for jb in range(2):
        for kb in range(3):
            add_strassen_term(jb, kb, col)
            col += 7
        add_schoolbook_term(jb, 6, col)
        col += 4

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
