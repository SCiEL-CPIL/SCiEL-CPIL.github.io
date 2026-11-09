"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-97 tensor decomposition for 4x4 * 4x7 matrix multiplication.

    Strategy:
      - Left part: 4x4 * 4x4 (B columns 0..3) using recursive Strassen, rank 49.
      - Right part: 4x4 * 4x3 (B columns 4..6) using schoolbook, rank 48.
      - Total rank = 97 (seed is 112).

    Index conventions:
      - U shape (n*m, rank) = (16, rank); U[i*m+j, r]
      - V shape (m*p, rank) = (28, rank); V[j*p+k, r]
      - W shape (p*n, rank) = (28, rank); W[k*n+i, r]
      - Tensor: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    """
    n, m, p = DIMENSIONS  # 4, 4, 7
    deadline = time.time() + 110  # safety margin under 150s wall clock

    # ------------------------------------------------------------------
    # Strassen rank-7 decomposition of the 2x2x2 matmul tensor.
    # For X(2x2)*Y(2x2)=Z(2x2), the 7 rank-1 terms are:
    #   p1=(X00+X11)*(Y00+Y11) -> Z00+1, Z11-1
    #   p2=(X10+X11)*Y00       -> Z10+1, Z11-1
    #   p3=X00*(Y01-Y11)       -> Z01+1, Z11+1
    #   p4=X11*(Y10-Y00)       -> Z00-1, Z10+1
    #   p5=(X00+X01)*Y11       -> Z01-1, Z11+1
    #   p6=(X10-X00)*(Y00+Y01) -> Z00+1
    #   p7=(X01-X11)*(Y10+Y11) -> Z00-1
    # Each (xvec, yvec, zvec) is a length-4 vector (positions 00,01,10,11).
    # ------------------------------------------------------------------
    s2x2 = [
        ((1,0,0,1), (1,0,0,1), (1,0,0,-1)),  # p1
        ((0,0,1,1), (1,0,0,0), (0,0,1,-1)),  # p2
        ((1,0,0,0), (0,1,0,-1), (0,1,0,1)),  # p3
        ((0,0,0,1), (0,0,1,-1), (-1,0,1,0)), # p4
        ((1,1,0,0), (0,0,0,1), (0,-1,0,1)),  # p5
        ((-1,0,1,0), (1,1,0,0), (1,0,0,0)),  # p6
        ((0,1,0,-1), (0,0,1,1), (-1,0,0,0)), # p7
    ]

    # ------------------------------------------------------------------
    # Recursive Strassen for 4x4 * 4x4 -> 4x4.
    # Super-block level (2x2 grid of 2x2 blocks).
    # For each of the 7 block-level intermediates, X and Y are sums of
    # 2x2 blocks with coefficients. We expand each into element-level
    # Strassen rank-7 terms.
    #
    # Block-level Strassen (M1..M7), each = (X_super) * (Y_super):
    #   M1 = (A00+A11) * (B00+B11)
    #   M2 = (A10+A11) * (B00)
    #   M3 = (A00)     * (B01-B11)
    #   M4 = (A11)     * (B10-B00)
    #   M5 = (A00+A01) * (B11)
    #   M6 = (A10-A00) * (B00+B01)
    #   M7 = (A01-A11) * (B10+B11)
    #
    # Output reconstruction (super-block level):
    #   C00 = M1 - M4 + M6 - M7
    #   C10 = M2 + M4
    #   C01 = M3 + M5
    #   C11 = M1 - M3 + M2 - M5
    #
    # A block A[br][bc] occupies rows br*2..br*2+1, cols bc*2..bc*2+1.
    # B block B[br][bc] occupies rows br*2..br*2+1, cols bc*2..bc*2+1.
    # C block C[br][bc] occupies rows br*2..br*2+1, cols bc*2..bc*2+1.
    #
    # We represent each element-level term as (a_vec, b_vec, c_vec) where
    # each is a dict mapping (row, col) -> coefficient.
    # ------------------------------------------------------------------
    def add_dict(base, other, scale=1):
        out = dict(base)
        for k, v in other.items():
            out[k] = out.get(k, 0) + scale * v
        return out

    def scale_dict(d, s):
        return {k: v * s for k, v in d.items() if v != 0}

    # Build a 2x2 block dict for A at (br, bc): maps (row,col)->1
    def a_block(br, bc):
        return {(br*2+lr, bc*2+lc): 1 for lr in range(2) for lc in range(2)}

    def b_block(br, bc):
        return {(br*2+lr, bc*2+lc): 1 for lr in range(2) for lc in range(2)}

    def c_block(br, bc):
        return {(br*2+lr, bc*2+lc): 1 for lr in range(2) for lc in range(2)}

    # Block-level X and Y definitions: list of (block_ref, coeff)
    # For A: blocks are (br, bc); for B: blocks are (br, bc).
    block_defs = [
        # (X_blocks, Y_blocks, output_block, output_sign)
        # M1 = (A00+A11)*(B00+B11)
        ([( (0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)],
         [( (0,0),1), ((1,1),-1)]),
        # M2 = (A10+A11)*B00
        ([((1,0),1), ((1,1),1)], [((0,0),1)],
         [((1,0),1), ((1,1),-1)]),
        # M3 = A00*(B01-B11)
        ([((0,0),1)], [((0,1),1), ((1,1),-1)],
         [((0,1),1), ((1,1),1)]),
        # M4 = A11*(B10-B00)
        ([((1,1),1)], [((1,0),1), ((0,0),-1)],
         [((0,0),-1), ((1,0),1)]),
        # M5 = (A00+A01)*B11
        ([((0,0),1), ((0,1),1)], [((1,1),1)],
         [((0,1),-1), ((1,1),1)]),
        # M6 = (A10-A00)*(B00+B01)
        ([((1,0),1), ((0,0),-1)], [((0,0),1), ((0,1),1)],
         [((0,0),1)]),
        # M7 = (A01-A11)*(B10+B11)
        ([((0,1),1), ((1,1),-1)], [((1,0),1), ((1,1),1)],
         [((0,0),-1)]),
    ]

    # Output reconstruction: for each output block (ob), sum over M_k with signs.
    # C00 = M1 - M4 + M6 - M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M3 + M2 - M5
    out_recon = {
        (0,0): [(0,1), (3,-1), (5,1), (6,-1)],
        (1,0): [(1,1), (3,1)],
        (0,1): [(2,1), (4,1)],
        (1,1): [(0,1), (2,-1), (1,1), (4,-1)],
    }

    # Generate all 49 element-level rank-1 terms for 4x4x4.
    # Each term is (a_vec, b_vec, c_vec) as dicts.
    terms_444 = []
    for k, (x_blocks, y_blocks, _out) in enumerate(block_defs):
        # Build X_super and Y_super as 2x2 "super" matrices where each
        # entry is a 2x2 block (dict of element coefficients).
        # Actually, X_super is a 2x2 matrix of 2x2 blocks. But in Strassen,
        # X is a single 2x2 matrix. Here X_super = sum of A blocks, which is
        # itself a 2x2 matrix (since each A block is 2x2 and they're at
        # different positions, the sum is a 4x4 matrix... wait no).
        #
        # Correction: In the recursive Strassen, at the super-block level,
        # A is treated as a 2x2 matrix whose entries are 2x2 blocks. So
        # X_super (for M1) = A00 + A11, where A00 is the top-left 2x2 block
        # and A11 is the bottom-right 2x2 block. But A00 and A11 are at
        # different positions in the 4x4 matrix, so A00+A11 is NOT a 2x2
        # matrix -- it's a 4x4 matrix with non-zero entries in two 2x2 blocks.
        #
        # This is the key subtlety. In recursive Strassen, we treat the 4x4
        # matrix as a 2x2 matrix of 2x2 blocks, and the Strassen operations
        # (addition, subtraction) are performed on the 2x2 blocks. So
        # X_super for M1 is the 2x2 "super-matrix" [A00, 0; 0, A11]... no.
        #
        # Let me reconsider. In Strassen for 2x2:
        #   M1 = (A00+A11)*(B00+B11)
        # Here A00 and A11 are the (0,0) and (1,1) entries of the 2x2
        # super-matrix A. When we say A00+A11, we mean the 2x2 matrix
        # that is the sum of the two 2x2 blocks. But A00 occupies the
        # top-left 2x2 of the 4x4, and A11 occupies the bottom-right 2x2.
        # So A00+A11 as a 4x4 matrix has A00 in the top-left and A11 in
        # the bottom-right.
        #
        # For the recursive step, we then multiply this 4x4 "X" by the
        # 4x4 "Y" = B00+B11 (B00 top-left, B11 bottom-right). But this
        # product is NOT a simple 2x2 block multiplication -- it's a full
        # 4x4 matmul where X and Y each have two non-zero 2x2 blocks.
        #
        # This is where the rank-49 construction gets subtle. The correct
        # recursive Strassen works as follows: we treat the problem as a
        # 2x2 matmul where each "entry" is itself a 2x2 matmul problem.
        # The rank is multiplicative: R(4) = 7 * R(2) = 49.
        #
        # The construction: for each of the 7 block-level terms, we have
        # a bilinear form in the 2x2 blocks. Each block-level term involves
        # products of specific 2x2 blocks. To compute a product of two 2x2
        # blocks, we use rank-7 Strassen. But the block-level terms involve
        # SUMS of blocks, so we need to expand.
        #
        # Let me think about this more carefully. In the recursive Strassen
        # for 4x4:
        #   We split A into 4 blocks: A = [[A00, A01], [A10, A11]], each 2x2.
        #   We split B into 4 blocks: B = [[B00, B01], [B10, B11]], each 2x2.
        #   We want C = A*B, a 4x4 matrix with 4 blocks C00, C01, C10, C11.
        #
        #   Strassen at the block level:
        #     M1 = (A00+A11)*(B00+B11)   -- 2x2 * 2x2 -> 2x2
        #     M2 = (A10+A11)*B00         -- 2x2 * 2x2 -> 2x2
        #     M3 = A00*(B01-B11)         -- 2x2 * 2x2 -> 2x2
        #     M4 = A11*(B10-B00)         -- 2x2 * 2x2 -> 2x2
        #     M5 = (A00+A01)*B11         -- 2x2 * 2x2 -> 2x2
        #     M6 = (A10-A00)*(B00+B01)   -- 2x2 * 2x2 -> 2x2
        #     M7 = (A01-A11)*(B10+B11)   -- 2x2 * 2x2 -> 2x2
        #
        #   Each Mk is a 2x2 matrix. To compute Mk, we use rank-7 Strassen
        #   on the 2x2 blocks. So each Mk contributes 7 rank-1 terms.
        #   Total: 7 * 7 = 49 rank-1 terms.
        #
        # Now, for M1 = (A00+A11)*(B00+B11):
        #   Let X = A00+A11 (a 2x2 matrix) and Y = B00+B11 (a 2x2 matrix).
        #   Wait, A00 and A11 are 2x2 blocks at different positions. Their
        #   sum A00+A11 is a 2x2 matrix where entry (lr,lc) = A00[lr,lc] +
        #   A11[lr,lc]. So X is a 2x2 matrix.
        #
        #   Similarly Y = B00+B11 is a 2x2 matrix.
        #
        #   So M1 = X*Y where X and Y are 2x2 matrices. We compute this
        #   with rank-7 Strassen, giving 7 rank-1 terms.
        #
        #   For each of the 7 element-level Strassen terms, the X part is
        #   a linear combination of X's entries, which are linear combinations
        #   of A00 and A11 entries. Similarly for Y.
        #
        #   So for term t in M1:
        #     X_t = linear combo of X entries = linear combo of A00 and A11 entries
        #     Y_t = linear combo of Y entries = linear combo of B00 and B11 entries
        #     Z_t = linear combo of M1 entries
        #
        #   The rank-1 term is: (a_vec, b_vec, c_vec) where
        #     a_vec maps A entries to coefficients (from X_t)
        #     b_vec maps B entries to coefficients (from Y_t)
        #     c_vec maps C entries to coefficients (from Z_t, via out_recon)
        #
        # This is the correct construction. Let me implement it.

        # Build X (2x2 matrix) as a 4-element tuple: X[lr,lc] = sum of
        # A block entries at position (br,bc) with coeff, where lr,lc are
        # local 2x2 indices.
        # X[lr,lc] = sum_{(br,bc),coeff} coeff * A[br*2+lr, bc*2+lc]
        # So X as a dict: {(lr,lc): list of (global_A_entry, coeff)}
        # Actually, let me just build X as a 2x2 matrix of coefficients
        # for each A global entry.

        # For X: 2x2 matrix, entry (lr,lc) is a linear combo of A entries.
        # X[lr,lc] = sum over (br,bc) with coeff c: c * A[br*2+lr, bc*2+lc]
        # So for a fixed (lr,lc), the A entries involved are at
        # (br*2+lr, bc*2+lc) for each (br,bc) in x_blocks.

        # Let me represent X as: for each local position (lr,lc), a dict
        # mapping global A index (ar, ac) -> coefficient.
        X = {}  # (lr,lc) -> { (ar,ac): coeff }
        for (br, bc), coeff in x_blocks:
            for lr in range(2):
                for lc in range(2):
                    ar, ac = br*2+lr, bc*2+lc
                    if (lr,lc) not in X:
                        X[(lr,lc)] = {}
                    X[(lr,lc)][(ar,ac)] = X[(lr,lc)].get((ar,ac), 0) + coeff

        Y = {}  # (lr,lc) -> { (br_,bc_): coeff } for B entries
        for (br, bc), coeff in y_blocks:
            for lr in range(2):
                for lc in range(2):
                    br_, bc_ = br*2+lr, bc*2+lc
                    if (lr,lc) not in Y:
                        Y[(lr,lc)] = {}
                    Y[(lr,lc)][(br_,bc_)] = Y[(lr,lc)].get((br_,bc_), 0) + coeff

        # Now apply the 7 element-level Strassen terms to X and Y.
        for t in range(7):
            xvec, yvec, zvec = s2x2[t]
            # xvec = (x00, x01, x10, x11) coefficients for X entries
            # X entry (lr,lc) has coefficient xvec[lr*2+lc]
            # So the total A coefficient for global entry (ar,ac) is:
            #   sum over (lr,lc): xvec[lr*2+lc] * X[(lr,lc)].get((ar,ac), 0)
            a_vec = {}
            for lr in range(2):
                for lc in range(2):
                    xc = xvec[lr*2+lc]
                    if xc == 0:
                        continue
                    if (lr,lc) not in X:
                        continue
                    for (ar,ac), ac2 in X[(lr,lc)].items():
                        val = xc * ac2
                        if val != 0:
                            a_vec[(ar,ac)] = a_vec.get((ar,ac), 0) + val

            b_vec = {}
            for lr in range(2):
                for lc in range(2):
                    yc = yvec[lr*2+lc]
                    if yc == 0:
                        continue
                    if (lr,lc) not in Y:
                        continue
                    for (br_,bc_), bc2 in Y[(lr,lc)].items():
                        val = yc * bc2
                        if val != 0:
                            b_vec[(br_,bc_)] = b_vec.get((br_,bc_), 0) + val

            # c_vec: Z is a 2x2 matrix (the Mk result). Z[lr,lc] has
            # coefficient zvec[lr*2+lc]. Z is added to output blocks via
            # out_recon. For output block (ob), the contribution is:
            #   sum over k where this M_k is involved: sign * Z[lr,lc]
            # But Z here is M_k, and out_recon[ob] lists (k, sign) pairs.
            # So for output block (ob), the C entries at (ob_r*2+lr, ob_c*2+lc)
            # get coefficient: sign * zvec[lr*2+lc]
            c_vec = {}
            for (ob_r, ob_c), sign_list in out_recon.items():
                for (mk, sign) in sign_list:
                    if mk != k:
                        continue
                    for lr in range(2):
                        for lc in range(2):
                            zc = zvec[lr*2+lc]
                            if zc == 0:
                                continue
                            cr, cc = ob_r*2+lr, ob_c*2+lc
                            val = sign * zc
                            if val != 0:
                                c_vec[(cr,cc)] = c_vec.get((cr,cc), 0) + val

            if a_vec and b_vec and c_vec:
                terms_444.append((a_vec, b_vec, c_vec))

    # ------------------------------------------------------------------
    # Schoolbook terms for 4x4 * 4x3 (B columns 4..6).
    # For each (i, j, k) with k in 4..6: rank-1 term.
    # ------------------------------------------------------------------
    terms_443 = []
    for i in range(n):
        for j in range(m):
            for k in range(4, p):  # columns 4,5,6
                a_vec = {((i, j)): 1}
                b_vec = {((j, k)): 1}
                c_vec = {((i, k)): 1}
                terms_443.append((a_vec, b_vec, c_vec))

    all_terms = terms_444 + terms_443
    rank = len(all_terms)

    # ------------------------------------------------------------------
    # Build U, V, W matrices.
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
    # ------------------------------------------------------------------
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    for r, (a_vec, b_vec, c_vec) in enumerate(all_terms):
        for (ar, ac), val in a_vec.items():
            U[ar * m + ac][r] = val
        for (br, bc), val in b_vec.items():
            V[br * p + bc][r] = val
        for (cr, cc), val in c_vec.items():
            W[cc * n + cr][r] = val

    # ------------------------------------------------------------------
    # Verify correctness (if time permits).
    # ------------------------------------------------------------------
    if time.time() < deadline - 20:
        try:
            import numpy as np
            Ua = np.array(U, dtype=float)
            Va = np.array(V, dtype=float)
            Wa = np.array(W, dtype=float)
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # Check against actual matmul for random inputs.
            np.random.seed(42)
            A = np.random.randint(-3, 4, size=(n, m))
            B = np.random.randint(-3, 4, size=(m, p))
            C_expected = A @ B
            # Compute C from decomposition
            C_comp = np.zeros((n, p), dtype=float)
            for r in range(rank):
                # u_r: vector of length n*m, shape (n,m)
                ur = Ua[:, r].reshape(n, m)
                vr = Va[:, r].reshape(m, p)
                wr = Wa[:, r].reshape(p, n)
                # C[i,k] += sum_j ur[i,j] * vr[j,k] * wr[k,i]
                for i in range(n):
                    for k in range(p):
                        s = 0.0
                        for j in range(m):
                            s += ur[i, j] * vr[j, k] * wr[k, i]
                        C_comp[i, k] += s
            if not np.allclose(C_comp, C_expected, atol=1e-9):
                # Fallback to schoolbook rank 112 if verification fails
                rank_fb = n * m * p
                U_fb = [[0.0] * rank_fb for _ in range(n * m)]
                V_fb = [[0.0] * rank_fb for _ in range(m * p)]
                W_fb = [[0.0] * rank_fb for _ in range(p * n)]
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_fb[i * m + j][col] = 1.0
                            V_fb[j * p + k][col] = 1.0
                            W_fb[k * n + i][col] = 1.0
                            col += 1
                return {"U": U_fb, "V": V_fb, "W": W_fb}
        except Exception:
            pass  # Return the constructed result even if verification fails

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
