import numpy as np
import time
import random

def _verify(U, V, W, seed=42, n_trials=5):
    """Verify decomposition correctness with multiple random integer matrices."""
    rng = np.random.default_rng(seed)
    for t in range(n_trials):
        A = rng.integers(-3, 4, size=(4, 4)).astype(float)
        B = rng.integers(-3, 4, size=(4, 6)).astype(float)
        A_flat = A.flatten()
        B_flat = B.flatten()
        C = np.zeros((24,))
        for r in range(U.shape[1]):
            ua = np.dot(U[:, r], A_flat)
            vb = np.dot(V[:, r], B_flat)
            C += (ua * vb) * W[:, r]
        C = C.reshape(6, 4).T
        if not np.allclose(C, A @ B, atol=1e-8):
            return False
    return True


def _build_strassen_block(n1, m1, p1):
    """Build rank-7 Strassen decomposition for n1 x m1 by m1 x p1 using 2x2x2 blocks.
    Returns U (n1*m1 x 7), V (m1*p1 x 7), W (p1*n1 x 7) for the block-level tensor.
    This is only valid when n1, m1, p1 are even (we tile into 2x2 sub-blocks).
    """
    R = 7
    U = np.zeros((n1 * m1, R))
    V = np.zeros((m1 * p1, R))
    W = np.zeros((p1 * n1, R))

    # Strassen 7 products for 2x2x2
    # M1 = (A11+A22)(B12+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    # We tile the n1 x m1 matrix into 2x2 blocks of size (n1/2) x (m1/2), etc.
    # For simplicity, we only handle the case where the whole matrix IS a 2x2 block
    # of unit sub-blocks (n1=m1=p1=2). For larger, we use recursive tiling.
    # Here we assume n1=m1=p1=2 for the base case.
    if n1 != 2 or m1 != 2 or p1 != 2:
        return None

    # A is 2x2, indices: A11=0, A12=1, A21=2, A22=3 (row-major flatten)
    # B is 2x2, indices: B11=0, B12=1, B21=2, B22=3
    # C is 2x2, indices: C11=0, C12=1, C21=2, C22=3 (but W uses trace convention)

    # U: maps A_flat to the 7 Strassen inputs
    # M1 = (A11+A22)(B12+B22): A-part = A11 + A22
    U[0, 0] = 1; U[3, 0] = 1
    # M2 = (A21+A22)B11: A-part = A21 + A22
    U[2, 1] = 1; U[3, 1] = 1
    # M3 = A11(B21-B22): A-part = A11
    U[0, 2] = 1
    # M4 = A22(B11-B21): A-part = A22
    U[3, 3] = 1
    # M5 = (A11+A12)B22: A-part = A11 + A12
    U[0, 4] = 1; U[1, 4] = 1
    # M6 = (A21-A11)(B11+B12): A-part = A21 - A11
    U[2, 5] = 1; U[0, 5] = -1
    # M7 = (A12-A22)(B21+B22): A-part = A12 - A22
    U[1, 6] = 1; U[3, 6] = -1

    # V: maps B_flat to the 7 Strassen inputs
    # M1: B-part = B12 + B22
    V[1, 0] = 1; V[3, 0] = 1
    # M2: B-part = B11
    V[0, 1] = 1
    # M3: B-part = B21 - B22
    V[2, 2] = 1; V[3, 2] = -1
    # M4: B-part = B11 - B21
    V[0, 3] = 1; V[2, 3] = -1
    # M5: B-part = B22
    V[3, 4] = 1
    # M6: B-part = B11 + B12
    V[0, 5] = 1; V[1, 5] = 1
    # M7: B-part = B21 + B22
    V[2, 6] = 1; V[3, 6] = 1

    # W: maps the 7 products to C (trace convention: W[k*n+i, r] for C[i,j] = sum_k A[i,k]B[k,j])
    # For 2x2x2, C is 2x2. W has shape (p1*n1, R) = (4, 7).
    # Trace convention: W[k*n+i, r] where n=2, so index = k*2 + i.
    # C11 (i=0,j=0): M1+M4-M5+M7 -> W[0*2+0, r] = W[0, r]
    # C12 (i=0,j=1): M3+M5 -> W[1*2+0, r] = W[2, r]
    # C21 (i=1,j=0): M2+M4 -> W[0*2+1, r] = W[1, r]
    # C22 (i=1,j=1): M1-M2+M3+M6 -> W[1*2+1, r] = W[3, r]

    # M1: C11 += 1, C22 += 1
    W[0, 0] = 1; W[3, 0] = 1
    # M2: C21 += 1, C22 -= 1
    W[1, 1] = 1; W[3, 1] = -1
    # M3: C12 += 1, C22 += 1
    W[2, 2] = 1; W[3, 2] = 1
    # M4: C11 += 1, C21 += 1
    W[0, 3] = 1; W[1, 3] = 1
    # M5: C11 -= 1, C12 += 1
    W[0, 4] = -1; W[2, 4] = 1
    # M6: C22 += 1
    W[3, 5] = 1
    # M7: C11 += 1
    W[0, 6] = 1

    return U, V, W


def _build_rank84():
    """Build rank-84 decomposition: 7 Strassen blocks, each 2x2x2 with schoolbook 12.
    Actually, for 4x4x6, we tile A into 2x2 blocks of 2x2, B into 2x2 blocks of 2x3.
    This is complex. Let's use a simpler approach: schoolbook rank 96, then apply
    Strassen at the block level.
    """
    # For 4x4 x 4x6, we can view A as 2x2 blocks of 2x2, B as 2x2 blocks of 2x3.
    # The product C (4x6) is 2x2 blocks of 2x3.
    # Strassen reduces 8 block multiplications to 7.
    # Each block multiplication is (2x2) * (2x3) -> rank 12 (schoolbook).
    # Total rank = 7 * 12 = 84.

    n, m, p = 4, 4, 6
    R = 84
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))

    # Block structure:
    # A blocks: A11 (rows 0-1, cols 0-1), A12 (rows 0-1, cols 2-3), A21 (rows 2-3, cols 0-1), A22 (rows 2-3, cols 2-3)
    # B blocks: B11 (rows 0-1, cols 0-2), B12 (rows 0-1, cols 3-5), B21 (rows 2-3, cols 0-2), B22 (rows 2-3, cols 3-5)
    # C blocks: C11 (rows 0-1, cols 0-2), C12 (rows 0-1, cols 3-5), C21 (rows 2-3, cols 0-2), C22 (rows 2-3, cols 3-5)

    # Strassen 7 products:
    # M1 = (A11+A22)(B12+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)

    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    def add_block_product(U, V, W, col_offset, a_signs, b_signs, c_targets):
        """Add a block product (2x2)*(2x3) with given signs to U, V, W.
        a_signs: dict from (row_block, col_block) to sign for A
        b_signs: dict from (row_block, col_block) to sign for B
        c_targets: list of (c_row_block, c_col_block, sign) for C
        """
        for (ar, ac), asgn in a_signs.items():
            for (br, bc), bsgn in b_signs.items():
                for c in range(2):  # 2x2 A block has 4 elements
                    pass

        # Schoolbook for (2x2)*(2x3): 4*3=12 products
        # A element (i,j) where i in 0,1 (within block), j in 0,1
        # B element (j,k) where j in 0,1, k in 0,1,2
        # C element (i,k)
        col = col_offset
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    # A global index: (ar*2+i) * 4 + (ac*2+j)
                    # B global index: (br*2+j) * 6 + (bc*3+k)
                    # C global index: (c_br*2+k) * 4 + (c_ar*2+i)
                    a_sign = a_signs.get((0, 0), 0)  # placeholder
                    # This is getting complex. Let's use a different approach.
                    col += 1
        return col

    # Let's just build it directly with explicit indices
    col = 0
    for s in range(7):
        # Define A-part and B-part for each Strassen product
        if s == 0:  # M1 = (A11+A22)(B12+B22)
            a_blocks = [(0,0,1), (1,1,1)]
            b_blocks = [(0,1,1), (1,1,1)]
        elif s == 1:  # M2 = (A21+A22)B11
            a_blocks = [(1,0,1), (1,1,1)]
            b_blocks = [(0,0,1)]
        elif s == 2:  # M3 = A11(B21-B22)
            a_blocks = [(0,0,1)]
            b_blocks = [(1,0,1), (1,1,-1)]
        elif s == 3:  # M4 = A22(B11-B21)
            a_blocks = [(1,1,1)]
            b_blocks = [(0,0,1), (1,0,-1)]
        elif s == 4:  # M5 = (A11+A12)B22
            a_blocks = [(0,0,1), (0,1,1)]
            b_blocks = [(1,1,1)]
        elif s == 5:  # M6 = (A21-A11)(B11+B12)
            a_blocks = [(1,0,1), (0,0,-1)]
            b_blocks = [(0,0,1), (0,1,1)]
        else:  # M7 = (A12-A22)(B21+B22)
            a_blocks = [(0,1,1), (1,1,-1)]
            b_blocks = [(1,0,1), (1,1,1)]

        # For each combination of A-block and B-block, add schoolbook products
        for (ar, ac, asgn) in a_blocks:
            for (br, bc, bsgn) in b_blocks:
                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            a_idx = (ar * 2 + i) * 4 + (ac * 2 + j)
                            b_idx = (br * 2 + j) * 6 + (bc * 3 + k)
                            U[a_idx, col] = asgn
                            V[b_idx, col] = bsgn
                            col += 1

        # Now add W contributions for this Strassen product
        # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
        c_contribs = {}
        if s == 0:  # M1
            c_contribs = {(0,0,1), (1,1,1)}
        elif s == 1:  # M2
            c_contribs = {(1,0,1), (1,1,-1)}
        elif s == 2:  # M3
            c_contribs = {(0,1,1), (1,1,1)}
        elif s == 3:  # M4
            c_contribs = {(0,0,1), (1,0,1)}
        elif s == 4:  # M5
            c_contribs = {(0,0,-1), (0,1,1)}
        elif s == 5:  # M6
            c_contribs = {(1,1,1)}
        else:  # M7
            c_contribs = {(0,0,1)}

        # For each C contribution, we need to map back to the schoolbook columns
        # This is tricky because the W contributions depend on which (i,j,k) combinations
        # contributed to this Strassen product.
        # For now, let's just set W to 0 and rely on verification to catch errors.
        # Actually, we need to properly set W.
        # Each schoolbook column in this Strassen product corresponds to a specific (i,j,k).
        # The C contribution for that (i,j,k) is at C[(c_ar*2+i), (c_bc*3+k)] for each (c_ar, c_bc, csgn) in c_contribs.
        # But the mapping from (i,j,k) to C is not one-to-one in general.
        # For schoolbook, C[i,k] = sum_j A[i,j] B[j,k], so each (i,j,k) contributes to C[i,k].
        # In our block structure, i is within the A-block row, k is within the B-block col.
        # So for a given (ar,ac,asgn) and (br,bc,bsgn), the (i,j,k) contributes to C[(c_ar*2+i), (c_bc*3+k)]
        # where c_ar and c_bc depend on the Strassen product.

        # This is getting very complex. Let's use a simpler verification-based approach.
        pass

    # This approach is too error-prone. Let's use the known rank-77 from the original
    # and focus on the search algorithm improvement.
    return None


def _build_rank77():
    """Build the rank-77 Strassen-based decomposition (from original, verified working)."""
    n, m, p = 4, 4, 6
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))

    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1}
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1}
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]

    col = 0
    for s in range(7):
        for si in range(7):
            for a in range(4):
                if alpha[s].get(a, 0) == 0:
                    continue
                if alpha[si].get(a, 0) == 0:
                    continue
                x_row, x_col = a // 2, a % 2
                for r in range(4):
                    if alpha[s].get(r, 0) == 0:
                        continue
                    ar, br = r // 2, r % 2
                    u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                    U[u_idx, col] += alpha[si][a] * alpha[s][r]

            for b in range(4):
                if alpha[s].get(b, 0) == 0:
                    continue
                if alpha[si].get(b, 0) == 0:
                    continue
                y_row, y_inner_col = b // 2, b % 2
                for c in range(4):
                    if beta[s].get(c, 0) == 0:
                        continue
                    ac, bc = c // 2, c % 2
                    v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                    V[v_idx, col] += alpha[si][b] * beta[s][c]

            for a2, b2, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b2 * 3 + zb) * 4 + (a2 * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            col += 1

        for u in range(2):
            for v in range(2):
                for r in range(4):
                    if alpha[s].get(r, 0) == 0:
                        continue
                    ar, br = r // 2, r % 2
                    u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                    U[u_idx, col] += alpha[s][r]

                for c in range(4):
                    if beta[s].get(c, 0) == 0:
                        continue
                    ac, bc = c // 2, c % 2
                    v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                    V[v_idx, col] += beta[s][c]

                for a2, b2, gamma in recon[s]:
                    w_idx = (b2 * 3 + 2) * 4 + (a2 * 2 + u)
                    W[w_idx, col] += gamma
                col += 1

    return U, V, W


def _build_schoolbook():
    """Fallback schoolbook decomposition, rank 96."""
    R = 96
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    idx = 0
    for i in range(4):
        for k in range(4):
            for j in range(6):
                U[i * 4 + k, idx] = 1
                V[k * 6 + j, idx] = 1
                W[j * 4 + i, idx] = 1
                idx += 1
    return U, V, W


def _nullspace_elimination(U, V, W, max_time, rng):
    """Try to reduce rank via null-space projection: for each column, check if it can be
    expressed as a linear combination of the others (in the tensor sense), and if so, eliminate it.
    This is a different approach from random invertible transforms.
    """
    R = U.shape[1]
    deadline = time.time() + max_time

    best_rank = R
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Try eliminating columns one at a time
    for attempt in range(200):
        if time.time() > deadline:
            break

        # Pick a random column to try eliminating
        c = rng.randint(0, R - 1)

        # Check if column c can be expressed as a combination of others
        # In the tensor decomposition, column c contributes (U[:,c] ⊗ V[:,c] ⊗ W[:,c])
        # to the tensor. We need to check if this can be written as a sum of other columns'
        # tensor products. This is a non-linear condition, so we use a heuristic:
        # check if U[:,c], V[:,c], W[:,c] are in the span of the others.

        # Check if U[:,c] is in the span of U[:, others]
        others = [i for i in range(R) if i != c]
        U_others = U[:, others]
        V_others = V[:, others]
        W_others = W[:, others]

        # Compute projection of U[:,c] onto span of U_others
        try:
            # Use least squares to find coefficients
            coef_U, _, rank_U, _ = np.linalg.lstsq(U_others, U[:, c], rcond=None)
            resid_U = U[:, c] - U_others @ coef_U

            coef_V, _, rank_V, _ = np.linalg.lstsq(V_others, V[:, c], rcond=None)
            resid_V = V[:, c] - V_others @ coef_V

            coef_W, _, rank_W, _ = np.linalg.lstsq(W_others, W[:, c], rcond=None)
            resid_W = W[:, c] - W_others @ coef_W
        except np.linalg.LinAlgError:
            continue

        # If all residuals are small, the column might be eliminable
        if (np.linalg.norm(resid_U) < 1e-10 and
            np.linalg.norm(resid_V) < 1e-10 and
            np.linalg.norm(resid_W) < 1e-10):
            # Try eliminating column c
            new_U = U[:, others]
            new_V = V[:, others]
            new_W = W[:, others]

            if _verify(new_U, new_V, new_W, seed=attempt, n_trials=3):
                U, V, W = new_U, new_V, new_W
                R = U.shape[1]
                best_rank = R
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    return best_U, best_V, best_W, best_rank


def _coordinate_descent(U, V, W, max_time, rng):
    """Try to reduce rank by adjusting entries to create zeros, then eliminating zero columns.
    This is a coordinate descent approach on the tensor entries.
    """
    R = U.shape[1]
    deadline = time.time() + max_time

    best_rank = R
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    for attempt in range(50):
        if time.time() > deadline:
            break

        # Try to zero out one entry in a random column
        c = rng.randint(0, R - 1)
        mat = rng.choice([U, V, W])
        i = rng.randint(0, mat.shape[0] - 1)

        # Save original
        orig_val = mat[i, c]
        if abs(orig_val) < 1e-10:
            continue

        # Try setting to 0 and see if we can compensate
        mat[i, c] = 0.0

        # Check if the decomposition is still valid
        if _verify(U, V, W, seed=attempt, n_trials=2):
            # Valid! Keep it and check for zero columns
            zero_cols = []
            for cc in range(R):
                if (np.all(np.abs(U[:, cc]) < 1e-10) or
                    np.all(np.abs(V[:, cc]) < 1e-10) or
                    np.all(np.abs(W[:, cc]) < 1e-10)):
                    zero_cols.append(cc)

            if zero_cols:
                keep = [cc for cc in range(R) if cc not in zero_cols]
                new_U = U[:, keep]
                new_V = V[:, keep]
                new_W = W[:, keep]
                if _verify(new_U, new_V, new_W, seed=attempt, n_trials=3):
                    U, V, W = new_U, new_V, new_W
                    R = U.shape[1]
                    best_rank = R
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        else:
            # Revert
            mat[i, c] = orig_val

    return best_U, best_V, best_W, best_rank


def _to_list(mat):
    """Convert numpy array to list with integer/half-integer rounding."""
    result = []
    for row in mat:
        new_row = []
        for x in row:
            rx = round(x)
            if abs(x - rx) < 1e-9:
                new_row.append(int(rx))
            else:
                hx = round(x * 2) / 2.0
                if abs(x - hx) < 1e-9:
                    new_row.append(hx)
                else:
                    new_row.append(float(x))
        result.append(new_row)
    return result


def construct():
    _t0 = time.time()
    deadline = _t0 + 140.0  # 10s safety margin before 150s kill

    # Build baseline rank-77 decomposition
    U, V, W = _build_rank77()

    # Verify baseline
    if not _verify(U, V, W):
        U, V, W = _build_schoolbook()
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

    # Try null-space elimination (new approach)
    remaining = deadline - time.time() - 5.0
    if remaining > 10.0:
        rng = random.Random(12345)
        U, V, W, rank = _nullspace_elimination(U, V, W, remaining * 0.4, rng)
        if not _verify(U, V, W):
            U, V, W = _build_rank77()

    # Try coordinate descent (new approach)
    remaining = deadline - time.time() - 5.0
    if remaining > 10.0:
        rng = random.Random(67890)
        U, V, W, rank = _coordinate_descent(U, V, W, remaining * 0.4, rng)
        if not _verify(U, V, W):
            U, V, W = _build_rank77()

    # Final verification
    if not _verify(U, V, W):
        U, V, W = _build_schoolbook()

    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
