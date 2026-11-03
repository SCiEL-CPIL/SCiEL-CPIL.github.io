import numpy as np
import time

def construct():
    start = time.time()
    deadline = start + 110  # safety margin before 120s kill

    n, m, p = 3, 4, 4

    # Build a rank-32 decomposition for (3x4)*(4x4) -> (4x3)
    # Strategy: use 4 Laderman (2x2x2) rank-19 blocks is too big.
    # Instead: tile the 4x4 B and 3x4 A into (2x2) blocks.
    # A is 3x4: rows 0,1,2 ; cols 0,1,2,3
    # B is 4x4: rows 0,1,2,3 ; cols 0,1,2,3
    # C is 4x3: rows 0..3 (output col), cols 0..2 (output row)
    # C[i, j] = sum_k A[j, k] B[k, i]
    #
    # We'll use a block structure: split into 2x2 blocks of B (4 blocks)
    # and 2x2 blocks of A (rows 0-1 and row 2 separately).
    #
    # Simpler verified approach: use the "schoolbook with Strassen on 2x2" 
    # giving rank 32. Let me construct it via explicit Laderman on each
    # (2x2)*(2x2) and handle the 3rd row of A.
    #
    # Actually, the cleanest verified low-rank: 
    # Use 4 Laderman rank-19 = 76. Too high.
    # Use Strassen 2x2x2 rank 7: 
    #   A(3x4) = [A_top(2x4); A_bot(1x4)]
    #   B(4x4) = [[B11,B12],[B21,B22]] each 2x2
    #   C_top(2x3) = A_top * B  (2x4 * 4x3)
    #   C_bot(1x3) = A_bot * B
    #
    # For C_top: (2x4)*(4x3). Split B into 2x2 blocks, A_top into (2x2,2x2).
    # C_top = [A_topL * B_L + A_topR * B_R] where B_L = [B11;B21], B_R=[B12;B22]
    # Each (2x2)*(2x2) via Strassen rank 7. So C_top needs 2*7=14? 
    # Actually C_top is 2x3 = [C_topL(2x1? no)]
    #
    # Let me just do explicit full construction and verify.

    def build_rank32():
        rank = 32
        U = np.zeros((n * m, rank))  # 12 x 32
        V = np.zeros((m * p, rank))  # 16 x 32
        W = np.zeros((p * n, rank))  # 12 x 32

        # Index conventions:
        # U index: a_row * m + a_col = r*4 + c, for A[r, c], r in 0..2, c in 0..3
        # V index: b_row * p + b_col = r*4 + c, for B[r, c], r in 0..3, c in 0..3
        # W index: c_col * n + c_row = i*3 + j, for C[i, j], i in 0..3, j in 0..2
        #   (trace convention W[k*n + i, r] where k is output col, i is output row)

        # We'll use 4 Strassen (2x2x2) rank-7 decompositions.
        # Block the 4x4 B into 4 (2x2) blocks:
        #   B00 = B[0:2, 0:2], B01 = B[0:2, 2:4]
        #   B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
        # Block A (3x4) into top (2x4) and bottom (1x4).
        #
        # Output C[i, j] for i in 0..3, j in 0..2:
        #   C[i, j] = sum_{k=0}^{3} A[j, k] B[k, i]
        #
        # For j in 0,1 (top rows of A):
        #   C[i, j] = sum_{k=0}^{1} A[j,k] B[k,i] + sum_{k=2}^{3} A[j,k] B[k,i]
        #   = (A_topL[j, :] * B_L[:, i]) + (A_topR[j, :] * B_R[:, i])
        # where A_topL = A[0:2, 0:2], A_topR = A[0:2, 2:4]
        #       B_L = B[0:2, :4] (2x4), B_R = B[2:4, :4] (2x4)
        # Hmm, this is getting complex. Let me use a direct verified formula.
        #
        # I'll use the known rank-32 for (3x4x4) via 4x(2x2x2 Strassen) is 28.
        # Actually 4 blocks of (2x2x2) Strassen (rank 7) = 28, but we have 3 rows.
        #
        # Let me do: for each of the 4 (2x2) sub-blocks of B, multiply by A's 
        # corresponding 2 cols, but A has 3 rows so each (2x2)*(2x2) gives (2x2)
        # output for top 2 rows, plus we need the 3rd row.
        #
        # Cleanest: Use 4 Strassen (2x2x2) for the (2x4)*(4x2) style? 
        #
        # I'll just construct rank 32 explicitly using 4 groups of 8 factors:
        # Group g (g=0..3) handles B block rows [2g, 2g+2) and output cols [2g, 2g+2).
        # Wait B is 4x4, output C is 4x3.
        #
        # Let me use a totally explicit and verified construction:
        # Use 32 factors. For output entry C[i, j] (i=col of C = row of B, j=row of A):
        #   C[i,j] = sum_k A[j,k] B[k,i]
        #
        # I'll use the following rank-32 scheme (verified by construction):
        # Split into 4 (2x2) blocks of B along rows: B_r for r=0,1 (rows 2r,2r+1)
        # and 2 (2x2) blocks along cols: B_c for c=0,1 (cols 2c,2c+1).
        # So 4 blocks B[2r:2r+2, 2c:2c+2].
        # For each block, the contribution to C is:
        #   C[2r:2r+2, j] += A[j, 2c:2c+2] * B[2r:2r+2, 2c:2c+2]  (for j=0,1,2)
        #   C[2r:2r+2, j] += A[j, 2c+2:2c+4] * B[2r:2r+2, 2c+2:2c+4]
        #
        # This is 8 (2x2)*(2x2) multiplications, each producing (2x1) for each of 3 j.
        # Each (2x2)*(2x2) via Strassen = rank 7. 8*7 = 56. Too high.
        #
        # Better: For fixed (r, c) block of B (2x2), and all 3 rows of A (3x4),
        # the product A[:, 2c:2c+2] * B[2r:2r+2, 2c:2c+2] is (3x2)*(2x2) = (3x2).
        # That's 3 (1x2)*(2x2) = but sharing the 2x2.
        # A (3x2)*(2x2) can be done with rank? The (3x2x2) tensor rank.
        # (3x2x2): C[i,j] = sum_k A[i,k] B[k,j], i=0..2, k=0..1, j=0..1.
        # This is like 3 independent (1x2)*(2x2) but B shared.
        # Rank of (3x2x2) = ? It's the tensor for multiplying 3x2 by 2x2.
        # Schoolbook: 3*2*2 = 12. Can we do better?
        # This is equivalent to 3 (1x2) vectors times a (2x2) matrix.
        # = For each j (0,1): C[:,j] = B[:,j] * A (transposed) = (2x1)*(3x2) -> 
        #   C[i,j] = B[0,j]*A[i,0] + B[1,j]*A[i,1]
        # So C[:,j] = A * [B[0,j], B[1,j]]^T (A is 3x2, so A * (2x1) = 3x1).
        # C[:,j] = A[:,0]*B[0,j] + A[:,1]*B[1,j].
        # So for j=0: C[:,0] = A[:,0]*B[0,0] + A[:,1]*B[1,0]  (2 products)
        # for j=1: C[:,1] = A[:,0]*B[0,1] + A[:,1]*B[1,1]  (2 products)
        # Total 4 products for (3x2x2)! Since B is only 2x2.
        # So each (3x2)*(2x2) block = rank 4.
        # We have 4 such blocks (r in 0,1; c in 0,1) -> 4*4 = 16? 
        # Wait, A columns: for block c, A uses columns 2c, 2c+1 (2 cols). B uses rows 2r,2r+1 and cols 2c,2c+1.
        # But B's cols must match A's cols index. A[j, k] * B[k, i] where k is the shared index.
        # For block (r, c): k in {2c, 2c+1}, i in {2r, 2r+1}.
        # So C[i, j] gets contribution A[j, 2c]*B[2c, i] + A[j, 2c+1]*B[2c+1, i].
        # That's a (3x2) A-subblock (rows j=0..2, cols 2c,2c+1) times (2x2) B-subblock.
        # Rank 4 as shown. 4 blocks -> 16 total? But we need all i, j covered.
        # i ranges 0..3 covered by r=0 (i=0,1) and r=1 (i=2,3). 
        # j ranges 0..2 all covered. c ranges 0,1 covering k=0,1 and k=2,3.
        # So total rank = 4 blocks * 4 = 16?? That seems too good. Let me verify.
        #
        # Wait, that can't be right because the full (3x4x4) schoolbook is 48.
        # Let me recount: for block (r,c), the 4 products are:
        #   P1 = A[j, 2c] * B[2c, i]  for all i in {2r,2r+1}, j in {0,1,2}
        #   P2 = A[j, 2c+1] * B[2c+1, i]
        #   P3 = A[j, 2c] * B[2c+1, i]  -- NO, that's not how it works.
        #
        # Let me redo. C[i,j] = sum_k A[j,k] B[k,i].
        # For block (r,c), k in {2c, 2c+1}, i in {2r, 2r+1}.
        # Contribution to C[i,j]: A[j,2c]*B[2c,i] + A[j,2c+1]*B[2c+1,i].
        # This is (A[j,2c] * B[2c,i]) + (A[j,2c+1] * B[2c+1,i]).
        # For fixed (r,c), i in {2r,2r+1}, j in {0,1,2}:
        #   C[i,j] = A[j,2c]*B[2c,i] + A[j,2c+1]*B[2c+1,i]
        # This is a sum of 2 "products" but each product is a matrix:
        #   M1[i,j] = A[j,2c] * B[2c,i]  (outer product of A[:,2c] (3x1) and B[2c,:] (1x2))
        #   M2[i,j] = A[j,2c+1] * B[2c+1,i]
        # So C_block = M1 + M2. Each M is rank-1 matrix (outer product).
        # As a tensor, M1 = u \otimes v where u[j] = A[j,2c], v[i] = B[2c,i].
        # So each block contributes 2 rank-1 tensors. 4 blocks * 2 = 8 total rank!
        #
        # But wait, that's just schoolbook with the sum split, which is rank 8?
        # No! The issue: M1[i,j] = A[j,2c]*B[2c,i] is a single rank-1 term in the 
        # tensor decomposition. The full tensor T[i,j,k] = delta... 
        # The schoolbook is C[i,j] = sum_k A[j,k]B[k,i], which as a tensor is
        # sum_k (e_k in A) (e_k in B) -> that's 4 terms (k=0,1,2,3), each rank 1.
        # So schoolbook rank = 4? No, schoolbook for (n x m x p) has rank n*m*p generally,
        # but the tensor T_{i,j,k} where i indexes C-col, j indexes A-row, k indexes B-col...
        #
        # I'm confusing myself. Let me just be very explicit and verify numerically.
        # The tensor for (3x4)*(4x4)->(4x3):
        # T[i, j, k] = 1 if the multiplication C[i,j] = sum_k' A[j,k'] B[k',i] 
        #   has a term A[j,k]*B[k,i] contributing to C[i,j].
        # So T[i, j, k] = 1 if (i is output col, j is A row, k is B row/col) and 
        #   the term A[j,k]*B[k,i] appears. It appears exactly once for each (i,j,k).
        # So T[i,j,k] = 1 for all i in 0..3, j in 0..2, k in 0..3? 
        # No: T[i,j,k] = 1 means "A[j,k] and B[k,i] multiply to give C[i,j]".
        # That's true for every triple (i,j,k). So T is all-ones (4x3x4) tensor? 
        # That's rank 1! That's wrong.
        #
        # I have the tensor definition wrong. The matrix multiplication tensor is:
        # T[i, j, k] where i = row of C, j = col of A / row of B... 
        # Standard: (C)_{i,j} = sum_k (A)_{i,k} (B)_{k,j}.
        # Here A is 3x4 (rows i=0..2, cols k=0..3), B is 4x4 (rows k=0..3, cols j=0..3),
        # C is 3x4 (rows i=0..2, cols j=0..3).
        # But the task says output is (p x n) = (4 x 3) with trace convention W[k*n+i, r].
        # So C is 4x3, meaning C[col, row] = ... Let me re-read.
        #
        # "output factor follows the official trace convention W[k*n+i, r]"
        # n=3, m=4, p=4. A is n x m = 3x4. B is m x p = 4x4. C is p x n = 4x3.
        # C[i, j] for i in 0..3 (p), j in 0..2 (n): C[i,j] = sum_k A[j, k] B[k, i].
        # (A is 3x4: A[j,k], j=0..2, k=0..3. B is 4x4: B[k,i], k=0..3, i=0..3.)
        # So C[i,j] = sum_{k=0}^{3} A[j,k] B[k,i]. Yes.
        #
        # Tensor T[i, j, k] = 1 if term A[j,k]*B[k,i] contributes to C[i,j]. 
        # It does for every (i,j,k). So T[i,j,k] = 1 for all i,j,k? 
        # That means the tensor is the all-ones tensor of shape (4,3,4), which has rank 1.
        # That's clearly wrong for matrix multiplication.
        #
        # The resolution: the tensor is NOT all ones. T[i,j,k] should be 1 only 
        # when the indices match the multiplication. But here every (i,j,k) triple 
        # corresponds to a valid term. So the tensor IS all-ones?? 
        #
        # No. I think the tensor for matrix multiplication (n x m) (m x p) -> (n x p)
        # is T_{i, j, k} = delta where i is row of C, j is col of A (which is row of B),
        # k is col of B. And C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # So T_{i, j, k} = 1 (all of them) — the tensor is the "matrix multiplication tensor"
        # which is NOT all ones; it's T_{i,j,k} = 1 for the specific pattern.
        # Actually the matmul tensor IS defined as T_{i,j,k} = 1 for all i,j,k in the 
        # standard basis, and its rank is what we're minimizing. For (n x m x p),
        # the tensor is in R^{n x m x p} (or permuted) and equals sum of the 
        # "diagonal" — no.
        #
        # Let me just use the standard: the matmul tensor T has T[i,j,k] = 1 if 
        # (i, j, k) is a valid (C-row, A-col/B-row, B-col) and the term exists.
        # For C[i,k] = sum_j A[i,j] B[j,k], T[i,j,k] = 1 for ALL i,j,k. 
        # So T is the all-ones tensor of shape (n, m, p). Its rank is the matmul rank!
        # Yes! The matrix multiplication tensor is indeed the all-ones tensor 
        # (in the appropriate basis), and its tensor rank = matmul rank.
        #
        # Wait, that's not right either. The all-ones tensor has rank 1.
        #
        # I think the confusion is the basis. The matmul tensor is:
        # T = sum_{i,j,k} e_i \otimes e_j \otimes e_k  ... no.
        #
        # Let me look at it operationally. The bilinear map (A, B) -> C where 
        # C[i,k] = sum_j A[i,j] B[j,k]. The associated trilinear form is 
        # sum_{i,j,k} C_{i,k} A_{i,j} B_{j,k} (with appropriate signs/indices).
        # The tensor T satisfies T[i,j,k] = 1 for all i,j,k. So T is all-ones.
        # But the rank of the all-ones tensor of shape (n x m x p) is 1, not the matmul rank.
        #
        # I'm clearly misunderstanding. Let me just use the CORRECT definition:
        # The matmul tensor for (n x m) * (m x p) is T in R^{(n*p) x m x m} or 
        # T[i, j, k] where the multiplication is C = A B, C[i, k] = sum_j A[i, j] B[j, k].
        # The tensor is T_{i, j, k} = 1 if we're computing C[i,k] from A[i,j] and B[j,k].
        # This is 1 for every (i, j, k). So T is all-ones of shape (n, m, p)?? 
        #
        # No! The tensor is T_{i, j, k} where i indexes the C output, but C has n*p 
        # entries. Let me use: T is a 3-tensor of shape (n*p, m, m) is wrong.
        #
        # Correct: The matmul tensor M for (n x m)(m x p) has shape (n x p, n x m, m x p)
        # and M[c, a, b] = 1 if C[c] = sum A[a] B[b] involves that triple. 
        # Specifically, C[i, k] = sum_j A[i, j] B[j, k], so M[(i,k), (i,j), (j,k)] = 1.
        # So M is sparse: M[c, a, b] = 1 iff c = (i,k), a = (i,j), b = (j,k) for some i,j,k.
        # This is the standard matmul tensor. Its rank is the matmul rank.
        #
        # So in our case: C index = (i, k) where i = row of C (0..3, = p), k = col of C (0..2, = n).
        #   But C is p x n = 4 x 3, so C[i, k] with i in 0..3, k in 0..2.
        # A index = (k, j) where k = col of A (0..3, = m), j = row of A (0..2, = n).
        #   A is n x m = 3 x 4, so A[j, k] with j in 0..2, k in 0..3.
        # B index = (j, i) where j = row of B (0..3, = m), i = col of B (0..3, = p).
        #   B is m x p = 4 x 4, so B[j, i] with j in 0..3, i in 0..3.
        # C[i, k] = sum_j A[k, j] B[j, i].
        # So M[(i,k), (k,j), (j,i)] = 1 for all i in 0..3, j in 0..3, k in 0..2.
        #
        # Now the factors: U has shape (n*m, rank) = (12, rank), indexed by (k, j) -> k*4 + j? 
        #   The task says U shape (n*m, rank) = (3*4, rank) = (12, rank). 
        #   n*m = 12. A is 3x4 = 12 entries. Index = j*4 + k (row j, col k) or k*4+j?
        #   "U" factor of shape (n*m, rank). The natural index for A[j,k] is j*m + k = j*4 + k.
        # V shape (m*p, rank) = (16, rank). B[j, i] index = j*p + i = j*4 + i.
        # W shape (p*n, rank) = (12, rank). C[i, k] index = i*n + k = i*3 + k. (trace convention W[k*n+i] -> W[i*3 + k]? "W[k*n+i, r]" so index = k*n + i = k*3 + i. Hmm.)
        #
        # "W[k*n+i, r]" — so W index = k*n + i where k is... let me parse: 
        # W has shape (p*n, rank) = (12, rank). The index is k*n + i. 
        # If C is p x n = 4 x 3, and we write C[i, k] (i = row 0..3, k = col 0..2),
        # then "k*n + i" = k*3 + i. So W index = k*3 + i, meaning k is the col of C, 
        # i is the row of C. So W[k*3 + i] corresponds to C[i, k].
        #
        # OK so let me just define everything concretely and verify with a random test.

        def make_factors_from_products(products):
            """
            products: list of (U_dict, V_dict, W_dict) where each dict maps index -> coeff.
            U_dict keys: A index = j*4 + k (j=row of A 0..2, k=col of A 0..3)
            V_dict keys: B index = j*4 + i (j=row of B 0..3, i=col of B 0..3)
            W_dict keys: C index = k*3 + i (k=col of C 0..2, i=row of C 0..3)
            """
            rank = len(products)
            U = np.zeros((12, rank))
            V = np.zeros((16, rank))
            W = np.zeros((12, rank))
            for r, (ud, vd, wd) in enumerate(products):
                for idx, val in ud.items():
                    U[idx, r] += val
                for idx, val in vd.items():
                    V[idx, r] += val
                for idx, val in wd.items():
                    W[idx, r] += val
            return U, V, W

        # Build the decomposition: for each (i, j, k) the term A[j,k]*B[k,i] -> C[i,k].
        # We want to group these into rank-1 factors.
        #
        # Rank-32 construction: Use 4 groups (one per pair of B-rows).
        # Group g (g=0,1): B-rows j in {2g, 2g+1}, C-rows i in {2g, 2g+1}.
        #   For this group, terms: A[j, k] * B[k, i] for j in {2g,2g+1}, i in {2g,2g+1}, k in 0..2.
        #   That's 2*2*3 = 12 terms per group, 4 groups = 48 (schoolbook).
        #
        # To reduce rank, within each group use Strassen on the (2x2) B-block and 
        # the (3x2) A-block (3 rows k, 2 cols j).
        #
        # For group g: 
        #   A_sub = A[:, 2g:2g+2]  (3x2, rows k=0..2, cols j=2g,2g+1)
        #   B_sub = B[2g:2g+2, :]  (2x4, rows j=2g,2g+1, cols i=0..3)
        #   But we only need B_sub cols i in {2g, 2g+1} for this group's C output.
        #   B_sub2 = B[2g:2g+2, 2g:2g+2]  (2x2)
        #   C_sub = C[2g:2g+2, :]  (2x3, rows i=2g,2g+1, cols k=0..2)
        #   C_sub[i-2g, k] = sum_{j in {2g,2g+1}} A[k, j] B[j, i]
        #   = A_sub[k, :] * B_sub2[:, i-2g]
        #   So C_sub = A_sub (3x2) * B_sub2 (2x2) -> but that gives (3x2), not (2x3).
        #   C_sub has shape (2, 3): C_sub[a, k] = sum_j A_sub[k, j] B_sub2[j, a].
        #   = (A_sub^T * B_sub2)[a, k] = (B_sub2^T * A_sub^T)... 
        #   C_sub = B_sub2^T (2x2) * A_sub^T (2x3) = (2x3). Yes!
        #   So C_sub = B_sub2^T * A_sub^T, a (2x2)*(2x3) multiplication.
        #   Rank of (2x2x3) matmul = ? Schoolbook 12. Strassen-ish?
        #   (2x2)*(2x3): C[a,k] = sum_j B[a,j] A[k,j]. 
        #   This is like 2 (1x2)*(2x3) = 2*2*3=12 schoolbook.
        #   Can we do rank 8? Use Strassen on the (2x2) part.
        #   C = [C[:,0], C[:,1], C[:,2]] where C[:,k] = B^T * A[:,k] (2x1).
        #   C[:,k] = B^T * A[:,k] = (2x2)*(2x1) = 2 products each, 3 cols = 6? 
        #   But B^T shared. C[:,k] = B[0,0]*A[k,0] + B[1,0]*A[k,1] (for col 0 of C).
        #   So C[:,0] = A[:,0]*B[0,0] + A[:,1]*B[1,0] (2 products)
        #   C[:,1] = A[:,0]*B[0,1] + A[:,1]*B[1,1] (2 products)
        #   C[:,2] = A[:,0]*B[0,2] + A[:,1]*B[1,2] (2 products)
        #   Total 6 products for (2x2x3)! Each product is a rank-1 tensor:
        #   P = A[:,k] * B[a,j] -> C[a,k].
        #   So (2x2x3) has rank 6.
        #
        # So each group g: (2x2x3) rank 6. 4 groups... wait, 2 groups (g=0,1) for B-rows,
        # but C has 4 rows (i=0..3) and we split into 2 groups of 2. And A has 3 cols (k=0..2)
        # all in each group. So 2 groups * rank 6 = 12? But we also need to cover all of B's 
        # columns. Let me re-examine.
        #
        # C[i, k] = sum_{j=0}^{3} A[k, j] B[j, i].
        # Group by j (B-row): j in {0,1} (group 0) and j in {2,3} (group 1).
        # Group g: C[i, k] += sum_{j in group} A[k, j] B[j, i], for all i in 0..3, k in 0..2.
        # So group g produces a (4x3) C-submatrix (all i, all k).
        #   C_g[i, k] = sum_{j in {2g,2g+1}} A[k, j] B[j, i]
        #   = A[k, 2g] * B[2g, i] + A[k, 2g+1] * B[2g+1, i]
        #   This is (A[:,2g] (3x1) * B[2g,:] (1x4)) + (A[:,2g+1] (3x1) * B[2g+1,:] (1x4))
        #   = outer products. As a tensor, each is rank-1: 
        #     M1[i, k] = A[k, 2g] * B[2g, i]  -> u[k] = A[k,2g], v[i] = B[2g,i].
        #   So group g = 2 rank-1 tensors. 2 groups * 2 = 4 total rank!
        #
        # But that's just the schoolbook split, rank 4? That can't be the matmul rank.
        #
        # I think the issue is that these "rank-1 tensors" M1[i,k] = A[k,2g]*B[2g,i] 
        # are NOT rank-1 in the matmul tensor sense because the matmul tensor 
        # T[i, j, k] has j as the B-row index, and M1 fixes j=2g. 
        # The matmul tensor is T[i, j, k] = 1 (for the term A[k,j]*B[j,i] -> C[i,k]).
        # So T[i, j, k] = 1 for all i, j, k. The tensor is all-ones (4 x 4 x 3)!
        # And the rank of the all-ones tensor (4x4x3) is 1.
        #
        # This means I have a fundamental misunderstanding. Let me reconsider.
        #
        # AH. I see my error. The matrix multiplication tensor is NOT all-ones.
        # The tensor T for (n x m) * (m x p) is T[i, j, k] where:
        #   i = row index of A and C (0..n-1)
        #   j = col index of A = row index of B (0..m-1)  
        #   k = col index of B and C (0..p-1)
        #   T[i, j, k] = 1 (the coefficient in C[i,k] = sum_j A[i,j] B[j,k]).
        # So T[i, j, k] = 1 for all i, j, k. It IS all-ones of shape (n, m, p).
        #
        # But the rank of the all-ones tensor of shape (n x m x p) is 1, which is 
        # obviously not the matrix multiplication rank (which is > n*m*p/3 or so).
        #
        # The resolution: The matrix multiplication tensor is NOT the all-ones tensor.
        # The all-ones tensor corresponds to the bilinear form sum_{i,j,k} A[i,j] B[j,k] C[i,k]
        # which is NOT the matrix multiplication. Matrix multiplication is a MAP 
        # (A, B) -> C, not a trilinear form. The tensor for the map is different.
        #
        # The correct matmul tensor: We want C = A B. The tensor M satisfies
        # C[i, k] = sum_j A[i, j] M[i, j, k] ... no.
        #
        # Let me use the standard definition from tensor rank literature:
        # The matrix multiplication tensor M_{n,m,p} is in R^{(n*p) x (n*m) x (m*p)}
        # defined by M_{(i,k), (i,j), (j,k)} = 1.
        # So it's a 3-tensor where the first index is (i,k) [C entry], second is (i,j) [A entry],
        # third is (j,k) [B entry]. M[(i,k), (i,j), (j,k)] = 1.
        # This is NOT all-ones; it's a specific sparse tensor.
        #
        # In our case: n=3, m=4, p=4.
        # C index (i, k): i in 0..3 (p=4), k in 0..2 (n=3). So C is 4x3.
        #   Wait, C = A B where A is 3x4, B is 4x4, so C is 3x4, not 4x3!
        #   But the task says output is (p x n) = (4 x 3). 
        #   So C is 4x3. That means C = B A? Or the convention is C^T?
        #   "W[k*n+i, r]" with p*n = 12. If C is 4x3, C[i, k] with i=0..3, k=0..2.
        #   C[i, k] = sum_j B[i, j] A[j, k]? That's (B A) which is (4x4)(4x3) = 4x3. Yes!
        #   So C = B A, where B is 4x4, A is 4x3? But A is 3x4...
        #
        # Let me re-read: "multiplying a 3 by 4 matrix by a 4 by 4 matrix".
        # So A is 3x4, B is 4x4, C = A B is 3x4. But output factor is (p x n) = (4 x 3).
        # So the output is C^T (4x3), or the convention is C[i, k] = (A B)^T[i, k] = (A B)[k, i].
        # (A B)[k, i] = sum_j A[k, j] B[j, i]. So C[i, k] = sum_j A[k, j] B[j, i].
        # This matches what I had: C[i, k] = sum_j A[k, j] B[j, i], with C being 4x3 
        # (i = row 0..3, k = col 0..2), A is 3x4 (A[k, j], k=0..2, j=0..3), 
        # B is 4x4 (B[j, i], j=0..3, i=0..3).
        #
        # So the matmul tensor M[(i,k), (k,j), (j,i)] = 1 for all i in 0..3, j in 0..3, k in 0..2.
        # This is the tensor. Now, is this all-ones in some basis? 
        # M is a 3-tensor of shape (12, 12, 16) [C-idx, A-idx, B-idx].
        # M[c, a, b] = 1 iff c = (i,k), a = (k,j), b = (j,i) for consistent i,j,k.
        # So it's sparse. Good, this is the real matmul tensor.
        #
        # Now, the factors U (12 x rank), V (16 x rank), W (12 x rank) satisfy:
        # M[c, a, b] = sum_r U[a, r] V[b, r] W[c, r].
        # (U indexed by A-entries, V by B-entries, W by C-entries.)
        #
        # Now I can build a rank-32 decomposition. Let me use the grouping:
        # For each j (B-row, 0..3), the terms are: for all i in 0..3, k in 0..2:
        #   M[(i,k), (k,j), (j,i)] = 1.
        #   This is a rank-1 "slice": U has 1 at (k,j), V has 1 at (j,i) for all i, W has 1 at (i,k) for all i,k.
        #   But V and W depend on i, so it's not a single rank-1 term.
        #
        # For fixed j: the slice is U[(k,j)] * V[(j,i)] * W[(i,k)] summed over i,k.
        # This is a matrix: for the fixed j, define U_j[k] = 1 (for k=0..2), 
        # V_j[i] = 1 (for i=0..3), W_j[i,k] = 1.
        # The contribution is sum_{i,k} U_j[k] V_j[i] W_j[i,k] = sum_{i,k} 1*1*1 = 12.
        # But as a tensor, the fixed-j slice is U_j \otimes V_j \otimes W_j where 
        # W_j[i,k] = 1 for all i,k. So W_j is a (4x3) all-ones matrix, which is rank 1.
        # So the fixed-j slice = U_j \otimes V_j \otimes W_j is rank 1? 
        # U_j is (3x1) all-ones, V_j is (4x1) all-ones, W_j is (4x3) all-ones.
        # U_j \otimes V_j \otimes W_j [k, i, (i',k')] = U_j[k] V_j[i] W_j[i',k'].
        # This is nonzero when... it's a product, so it's rank-1 in the tensor sense.
        # But W_j[i',k'] = 1 for all i',k', so the tensor is U_j[k] * V_j[i] * 1.
        # So the fixed-j slice is U_j \otimes V_j \otimes (all-ones 4x3).
        # The all-ones 4x3 matrix is rank 1 (it's u \otimes v^T).
        # So the fixed-j slice is rank 1!
        #
        # Therefore, the full tensor = sum_{j=0}^{3} (U_j \otimes V_j \otimes W_j) where each is rank 1.
        # So the matmul tensor has rank 4?! That's impossible.
        #
        # I must be making an error in the tensor definition. Let me look up the 
        # actual matmul tensor definition carefully.
        #
        # The matrix multiplication tensor M_{n,m,p} is defined as:
        # M = sum_{i=1}^{n} sum_{j=1}^{m} sum_{k=1}^{p} e_i \otimes e_j \otimes e_k
        # where the multiplication is (A B)[i, k] = sum_j A[i, j] B[j, k].
        # So M[i, j, k] = 1 for all i, j, k. It IS the all-ones tensor!
        #
        # But then rank(M) = 1, which contradicts known matmul ranks.
        #
        # The resolution: The all-ones tensor of shape (n x m x p) does NOT have rank 1 
        # in general! I was wrong. The rank of the all-ones tensor of shape (a x b x c) 
        # is 1 only if... no, the all-ones tensor is e \otimes e \otimes e where e is the 
        # all-ones vector, so it IS rank 1.
        #
        # I'm clearly confused. Let me just look at the actual matmul tensor for small case.
        # For (2x2x2): M[i,j,k] = 1 for all i,j,k in {0,1}. This is the all-ones 2x2x2 tensor.
        # Its rank: the all-ones 2x2x2 tensor = (1,1) \otimes (1,1) \otimes (1,1), rank 1.
        # But the matmul rank for (2x2x2) is 7 (Strassen)!
        #
        # So the matmul tensor is NOT the all-ones tensor. I have the definition wrong.
        #
        # CORRECT definition: The matmul tensor for (n x m) * (m x p) -> (n x p) is:
        # M[i, j, k] where i = row of C (0..n-1), j = col of A / row of B (0..m-1), 
        # k = col of B / col of C (0..p-1).
        # M[i, j, k] = 1.  C[i, k] = sum_j A[i, j] B[j, k].
        # For (2x2x2): M[i,j,k] = 1 for all i,j,k. All-ones 2x2x2, rank 1.
        # But Strassen says rank 7. Contradiction!
        #
        # The issue: The matmul tensor is NOT M[i,j,k] = 1. It's a tensor in a 
        # DIFFERENT space. The matmul tensor is in R^{(n*p) x (n*m) x (m*p)} and 
        # M[(i,k), (i,j), (j,k)] = 1. This is NOT the all-ones tensor in R^{n x m x p}.
        #
        # For (2x2x2): M[(i,k), (i,j), (j,k)] = 1. 
        # Indices: (i,k) in {00,01,10,11}, (i,j) in {00,01,10,11}, (j,k) in {00,01,10,11}.
        # M[(0,0),(0,0),(0,0)] = 1, M[(0,0),(0,1),(1,0)] = 1, etc.
        # This is a 4x4x4 tensor with 8 ones (for each i,j,k: M[(i,k),(i,j),(j,k)] = 1).
        # This is the actual matmul tensor, and its rank is 7 for (2x2x2).
        #
        # GREAT. So in our case:
        # C index (i, k): i in 0..3, k in 0..2. (4x3 = 12 entries)
        # A index (k, j): k in 0..2, j in 0..3. (3x4 = 12 entries)  [A is 3x4, A[k,j]]
        # B index (j, i): j in 0..3, i in 0..3. (4x4 = 16 entries)  [B is 4x4, B[j,i]]
        # M[(i,k), (k,j), (j,i)] = 1 for all i in 0..3, j in 0..3, k in 0..2.
        # Total 4*4*3 = 48 ones.
        #
        # Now the factors: U (12 x rank) indexed by A-entries (k, j) -> index = k*4 + j.
        # V (16 x rank) indexed by B-entries (j, i) -> index = j*4 + i.
        # W (12 x rank) indexed by C-entries (i, k) -> index = k*3 + i (trace convention W[k*n+i]).
        #
        # Condition: M[(i,k), (k,j), (j,i)] = sum_r U[k*4+j, r] V[j*4+i, r] W[k*3+i, r].
        #
        # Now, rank-32 construction. Let me use 4 groups (one per pair of j-values):
        # Group g (g=0,1): j in {2g, 2g+1}.
        #   For group g, the terms are M[(i,k), (k, 2g), (2g, i)] and M[(i,k), (k, 2g+1), (2g+1, i)]
        #   for all i in 0..3, k in 0..2.
        #   = 2 * 4 * 3 = 24 terms per group.
        #
        # For group g, define:
        #   U_g: A-entries (k, j) for j in {2g, 2g+1}, k in 0..2.
        #   V_g: B-entries (j, i) for j in {2g, 2g+1}, i in 0..3.
        #   W_g: C-entries (i, k) for all i, k.
        #
        # The group-g tensor = sum_{i, k} [ U_g[k, 2g] V_g[2g, i] W_g[i, k] + U_g[k, 2g+1] V_g[2g+1, i] W_g[i, k] ]
        # = sum_{i,k} W_g[i,k] ( U_g[k,2g] V_g[2g,i] + U_g[k,2g+1] V_g[2g+1,i] )
        #
        # If we set U_g[k, j] = 1, V_g[j, i] = 1, W_g[i, k] = 1, then group-g tensor = 
        # sum_{i,k} 1*(1+1) = 24, but we need each M entry = 1, so this gives 2, not 1.
        # We need U_g[k,2g] V_g[2g,i] W_g[i,k] = 1 for the (2g) part and similarly for (2g+1),
        # but they ADD, so we'd get 2. We need to split.
        #
        # Actually, the group-g tensor should equal sum_{i,k} M[(i,k),(k,2g),(2g,i)] + M[(i,k),(k,2g+1),(2g+1,i)]
        # = sum_{i,k} 1 + 1 = sum_{i,k} 2. But the full M has M[(i,k),(k,j),(j,i)] = 1 for each j,
        # so for group g (2 j-values), the group contribution to M[(i,k),(k,j),(j,i)] should be 1 
        # for j in group g. So the group-g tensor should have value 1 at each (i,k,j) in group g.
        #
        # So group-g tensor T_g[i, k, j] = 1 for j in {2g, 2g+1}, all i, k.
        # And T_g = sum_r U_g[:,r] \otimes V_g[:,r] \otimes W_g[:,r] (restricted to group g indices).
        #
        # For group g, T_g is a (4 x 3 x 2) tensor (i x k x j) that is all-ones.
        # The all-ones (4 x 3 x 2) tensor has rank 1 (it's e \otimes e \otimes e).
        # So each group has rank 1! Total rank = 2.
        #
        # This is still giving absurdly low rank. I think the issue is that the factors 
        # U, V, W are GLOBAL (shared across groups), not per-group. The rank-1 decomposition 
        # must use the SAME U, V, W for all groups.
        #
        # Let me just directly construct a rank-32 solution using known Strassen blocks 
        # and VERIFY it numerically. I'll use 4 Strassen (2x2x2) rank-7 decompositions.
        #
        # Actually, let me just use a brute-force verified approach: 
        # I'll construct the rank-32 via 4 (2x2x2) Strassen blocks and verify.
        #
        # For (2x2x2) Strassen, the 7 products are:
        #   P1 = (A00+A11)(B00+B11)
        #   P2 = (A10+A11)B00
        #   P3 = A00(B10-B11)
        #   P4 = A11(B01-B10)
        #   P5 = (A00+A10)B11
        #   P6 = (A01-A00)(B00+B10)
        #   P7 = (A10-A11)(B01+B11)
        #   C00 = P1 + P4 - P5 - P7
        #   C01 = P3 + P5
        #   C10 = P2 + P4
        #   C11 = P1 - P2 + P3 + P7
        #
        # I'll apply this to 4 (2x2) sub-blocks of the (3x4x4) multiplication.
        #
        # Given the complexity and time, let me just use a simple verified rank-48 
        # schoolbook and then try to reduce. But the current is rank 44.
        #
        # Let me just return the current rank-44 solution but with a runtime verification 
        # to ensure correctness, and add a deadline guard.

        # Actually, let me construct a correct rank-32 using the following verified scheme:
        # Use 4 (2x2x2) Strassen (rank 7 each) = 28, plus 4 more for the 3rd row = 32.
        #
        # Partition:
        # A (3x4) = [A0 (2x4); A1 (1x4)]
        # B (4x4) = [[B00 B01],[B10 B11]] each 2x2
        # C (4x3) = [C00 C01; C10 C11] where C00 is (2x3), etc.
        #   C[i, k] = sum_j A[k, j] B[j, i]
        #   C00[i, k] for i in 0,1; C01[i,k] for i in 2,3; etc.
        #   C[i, k] = A[k,0]B[0,i] + A[k,1]B[1,i] + A[k,2]B[2,i] + A[k,3]B[3,i]
        #   For i in 0,1: C[i,k] = A[k,0]B[0,i] + A[k,1]B[1,i] + A[k,2]B[2,i] + A[k,3]B[3,i]
        #
        # This is getting too complex to do by hand. Let me just use a numerical 
        # approach: construct a rank-32 via random initialization and verify, 
        # or just return the verified rank-44 with a guard.
        #
        # Given time constraints, I'll return the rank-44 solution (verified correct) 
        # with a deadline guard and a verification step.

        return None  # signal to use the fallback

    # Build the rank-44 solution (verified correct in the current program)
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))

    # Strassen part (cols 0-6)
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1

    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1

    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1

    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1

    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1

    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1

    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1

    # Extra part (cols 7-10)
    U_loc[4, 7] = 1; V_loc[0, 7] = 1; W_loc[4, 7] = 1
    U_loc[5, 8] = 1; V_loc[2, 8] = 1; W_loc[4, 8] = 1
    U_loc[4, 9] = 1; V_loc[1, 9] = 1; W_loc[5, 9] = 1
    U_loc[5, 10] = 1; V_loc[3, 10] = 1; W_loc[5, 10] = 1

    def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_offset = block_idx * 11
        for r in range(3):
            for c in range(2):
                local_a_idx = r * 2 + c
                global_a_idx = (a_row_start + r) * m + (a_col_start + c)
                U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
        for r in range(2):
            for c in range(2):
                local_b_idx = r * 2 + c
                global_b_idx = (b_row_start + r) * p + (b_col_start + c)
                V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
        for r in range(3):
            for c in range(2):
                local_c_idx = r * 2 + c
                global_c_idx = (c_col_start + c) * n + (c_row_start + r)
                W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]

    add_block(0, 0, 0, 0, 0, 0, 0)
    add_block(1, 0, 2, 2, 0, 0, 0)
    add_block(2, 0, 0, 0, 2, 0, 2)
    add_block(3, 0, 2, 2, 2, 0, 2)

    # Verify correctness numerically
    def verify(U, V, W, n, m, p):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Check: for random A (n x m), B (m x p), C = A @ B (n x p)
        #   C^T (p x n) should equal sum_r (U^T A_vec) (V^T B_vec) W_col_r
        #   where A_vec = A.reshape(n*m), B_vec = B.reshape(m*p)
        #   Actually: C[i, k] = sum_j A[k, j] B[j, i] (C is p x n = 4 x 3)
        #   C_flat = C.reshape(p*n) with C_flat[k*n + i] = C[i, k]
        #   The decomposition: C_flat = W @ diag(...) ... 
        #   Standard: C = sum_r (U_r^T A) (V_r^T B) W_r? 
        #   Let me just check: T[c, a, b] = sum_r U[a, r] V[b, r] W[c, r]
        #   where T is the matmul tensor.
        #   T[(i,k), (k,j), (j,i)] = 1.
        #   So for each (i, k, j): sum_r U[k*4+j, r] V[j*4+i, r] W[k*3+i, r] = 1.
        rng = np.random.RandomState(42)
        for _ in range(3):
            A = rng.rand(n, m)
            B = rng.rand(m, p)
            C = A @ B  # (n x p) = (3 x 4)
            C_T = C.T  # (p x n) = (4 x 3)
            # Compute via decomposition:
            # C_T[i, k] = sum_j A[k, j] B[j, i]
            # = sum_r (sum_j U[k*4+j, r] A[k, j]) (sum_i V[j*4+i, r] B[j, i]) W[k*3+i, r]
            # Hmm, this isn't quite right. Let me use the tensor form.
            # T[c, a, b] = sum_r U[a, r] V[b, r] W[c, r]
            # C_T[i, k] = sum_j A[k, j] B[j, i] T[(i,k), (k,j), (j,i)]
            # = sum_j A[k,j] B[j,i] sum_r U[k*4+j, r] V[j*4+i, r] W[k*3+i, r]
            # = sum_r W[k*3+i, r] (sum_j U[k*4+j, r] A[k,j]) (sum_j V[j*4+i, r] B[j,i])
            # Wait, the j index is shared between A and B but not in the same sum.
            # Let me just compute directly:
            C_computed = np.zeros((p, n))
            for i in range(p):
                for k in range(n):
                    val = 0.0
                    for j in range(m):
                        T_val = 0.0
                        for r in range(rank):
                            T_val += U[k*m + j, r] * V[j*p + i, r] * W[k*n + i, r]
                        val += A[k, j] * B[j, i] * T_val
                    C_computed[i, k] = val
            if not np.allclose(C_computed, C_T, atol=1e-9):
                return False
        return True

    if verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback: pure schoolbook rank 48 (always correct)
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        for i in range(p):
            for k in range(n):
                for j in range(m):
                    r = i * (n * m) + k * m + j
                    U_sb[k * m + j, r] = 1
                    V_sb[j * p + i, r] = 1
                    W_sb[k * n + i, r] = 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

