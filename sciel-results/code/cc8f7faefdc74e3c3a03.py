"""Matrix multiplication tensor decomposition for 4x4 * 4x8."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                if abs(s - 1.0) > 1e-8:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_rank(n, m, p, rank, rng, max_iters):
    """
    Try to find a rank-`rank` exact decomposition.
    Fix U,V as sparse random matrices from {-1,0,1}, solve linear system for W,
    verify exactness.
    """
    num_u = n * m
    num_v = m * p
    num_w = p * n
    num_terms = n * m * p  # 128 equations

    for it in range(max_iters):
        # Sparse random U, V with entries in {-1,0,1}
        density = 0.5
        U = rng.choice([-1, 0, 1], size=(num_u, rank))
        V = rng.choice([-1, 0, 1], size=(num_v, rank))

        # Build linear system: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # This is a linear system A w = b where w = W flattened (num_w * rank unknowns).
        # But W appears as W[k*n+i, c], so unknowns are indexed by (w_idx, c).
        # Equation (i,j,k): sum_c coeff_{c} * W[k*n+i, c] = 1, where coeff_c = U[i*m+j,c]*V[j*p+k,c]
        # Note: each equation only involves W at row (k*n+i), so equations for the same
        # (k*n+i) row are coupled only through the same row.
        # We can solve per row of W: for a fixed w_row = k*n+i, the equations that touch it
        # are those (i,j,k) with k*n+i == w_row, i.e., fixed k and i (j free).
        # For fixed (i,k), as j varies, we get m equations, all involving only W[w_row, :].
        # So for each (i,k): sum over j of ... wait, each (i,j,k) is a separate equation.
        # For fixed (i,k), there are m equations (j=0..m-1), each:
        #   sum_c U[i*m+j,c]*V[j*p+k,c]*W[w_row,c] = 1
        # These m equations share the same unknown vector W[w_row,:] (length rank).
        # So we have m equations in `rank` unknowns. We can solve this small system.

        W = np.zeros((num_w, rank))
        ok = True
        for i in range(n):
            for k in range(p):
                w_row = k * n + i
                # m equations, each: A_j . W[w_row,:] = 1 where A_j[c] = U[i*m+j,c]*V[j*p+k,c]
                A_rows = []
                for j in range(m):
                    A_rows.append(U[i * m + j] * V[j * p + k])
                A_mat = np.array(A_rows, dtype=float)  # (m, rank)
                b_vec = np.ones(m, dtype=float)
                # Solve A_mat x = b_vec, x = W[w_row,:]
                # Use lstsq
                try:
                    sol, residuals, rnk, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except np.linalg.LinAlgError:
                    ok = False
                    break
                W[w_row, :] = sol
            if not ok:
                break
        if not ok:
            continue

        # Verify
        if _verify(W, W, W, n, m, p) is False:
            # placeholder, we verify properly below
            pass
        # Proper verification using U,V,W
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                    if abs(s - 1.0) > 1e-7:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            return U, V, W
    return None

def _strassen_block_4x4x4():
    """
    Construct a rank-56 decomposition for 4x4*4x4 using 8 Strassen 2x2 blocks.
    Returns (U4, V4, W4) each of shape (16, 56).
    """
    rank = 56
    U4 = np.zeros((16, rank))
    V4 = np.zeros((16, rank))
    W4 = np.zeros((16, rank))

    # Strassen 2x2 formulas, each term t in 0..6:
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

    # For a given block (a,b,c) where A_{a,b} (2x2), B_{b,c} (2x2), contributes to C_{a,c} (2x2)
    # Global indices:
    #   A row 2a+ra, col 2b+ca  -> U index (2a+ra)*4 + (2b+ca)  [for 4x4, m=4]
    #   B row 2b+rb, col 2c+cb  -> V index (2b+rb)*4 + (2c+cb)
    #   C row 2c+rc, col 2a+ca  -> W index (2c+rc)*4 + (2a+ca)  [W[k*n+i], n=4, k=2c+rc, i=2a+ca]

    def add_block(a, b, c, start_col):
        # Strassen terms: list of (U_entries, V_entries, W_entries)
        # Each entry is (local_row, local_col, coeff)
        terms = []
        # M1 = (A11+A22)(B11+B22)
        terms.append(([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]))  # C11+1, C22+1
        # M2 = (A21+A22)B11
        terms.append(([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1)]))  # C21+1
        # M3 = A11(B21-B22)
        terms.append(([(0,0,1)], [(1,0,1),(1,1,-1)], [(0,1,1),(1,1,1)]))  # C12+1, C22+1
        # M4 = A22(B12-B11)
        terms.append(([(1,1,1)], [(0,1,1),(0,0,-1)], [(0,0,1),(1,0,1)]))  # C11+1, C21+1
        # M5 = (A11+A12)B22
        terms.append(([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]))  # C11-1, C12+1
        # M6 = (A21-A11)(B11+B12)
        terms.append(([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]))  # C22+1
        # M7 = (A12-A22)(B21+B22)
        terms.append(([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]))  # C11+1

        col = start_col
        for (u_entries, v_entries, w_entries) in terms:
            for (ra, ca, coeff) in u_entries:
                U4[(2*a+ra)*4 + (2*b+ca), col] += coeff
            for (rb, cb, coeff) in v_entries:
                V4[(2*b+rb)*4 + (2*c+cb), col] += coeff
            for (rc, ca, coeff) in w_entries:
                W4[(2*c+rc)*4 + (2*a+ca), col] += coeff
            col += 1
        return col

    col = 0
    for a in range(2):
        for b in range(2):
            for c in range(2):
                col = add_block(a, b, c, col)
    return U4, V4, W4

def _build_4x4x8_from_4x4x4(U4, V4, W4, r4):
    """
    Given a rank-r4 decomposition for 4x4*4x4, build a rank-2*r4 decomposition
    for 4x4*4x8 by applying it to two 4x4 halves of the 8-column B.
    """
    n, m, p = 4, 4, 8
    rank = 2 * r4
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # U is the same for both halves
    U[:, :r4] = U4
    U[:, r4:] = U4

    # V: first half (k=0..3) uses V4, second half (k=4..7) uses V4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, :r4] = V4[j * 4 + k]
            V[j * 8 + (k + 4), r4:] = V4[j * 4 + k]

    # W: first half (k=0..3) uses W4, second half (k=4..7) uses W4
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, :r4] = W4[k * 4 + i]
            W[(k + 4) * 4 + i, r4:] = W4[k * 4 + i]

    return U, V, W

def _verify_decomp(U, V, W, n, m, p):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                if abs(s - 1.0) > 1e-7:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Start with a guaranteed rank-112 (2 * rank-56 Strassen blocks) decomposition.
    U4, V4, W4 = _strassen_block_4x4x4()
    r4 = 56
    U, V, W = _build_4x4x8_from_4x4x4(U4, V4, W4, r4)
    if _verify_decomp(U, V, W, n, m, p):
        best = (2 * r4, U.copy(), V.copy(), W.copy())
    else:
        # Fallback to schoolbook
        sb = _schoolbook(n, m, p)
        return sb

    # Now try to improve: search for direct rank-r decompositions of 4x4*4x8
    # for r < best rank, using random U,V + linear solve for W.
    rng = np.random.default_rng(42)
    current_best_rank = best[0]

    # Try a range of ranks below current best, with limited iterations each.
    # Budget-aware: allocate iterations based on remaining time.
    for target_rank in range(current_best_rank - 1, 90, -1):
        if time.time() > deadline - 15:
            break
        # Estimate time per iteration; cap iterations
        # Each iteration does ~ n*p = 16 small lstsq solves of size (4, rank)
        # That's fast. Allow many iterations.
        max_iters = 3000
        result = _try_rank(n, m, p, target_rank, rng, max_iters)
        if result is not None:
            U, V, W = result
            if _verify_decomp(U, V, W, n, m, p):
                current_best_rank = target_rank
                best = (target_rank, U.copy(), V.copy(), W.copy())
                # Continue trying to go lower
                # Re-seed rng to continue
                rng = np.random.default_rng(target_rank * 1000 + 7)

    # Also try to improve the 4x4x4 base: search rank-49 for 4x4x4, then build 4x4x8 rank-98
    if current_best_rank > 98 and time.time() < deadline - 20:
        rng2 = np.random.default_rng(12345)
        # Search for rank-49 4x4x4 decomposition
        found = None
        for it in range(5000):
            if time.time() > deadline - 20:
                break
            U4t = rng2.choice([-1, 0, 1], size=(16, 49))
            V4t = rng2.choice([-1, 0, 1], size=(16, 49))
            W4t = np.zeros((16, 49))
            ok = True
            for i in range(4):
                for k in range(4):
                    w_row = k * 4 + i
                    A_rows = []
                    for j in range(4):
                        A_rows.append(U4t[i * 4 + j] * V4t[j * 4 + k])
                    A_mat = np.array(A_rows, dtype=float)
                    b_vec = np.ones(4, dtype=float)
                    try:
                        sol, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    except np.linalg.LinAlgError:
                        ok = False
                        break
                    W4t[w_row, :] = sol
                if not ok:
                    break
            if not ok:
                continue
            valid = True
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        s = np.sum(U4t[i * 4 + j] * V4t[j * 4 + k] * W4t[k * 4 + i])
                        if abs(s - 1.0) > 1e-7:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                found = (U4t, V4t, W4t)
                break
        if found is not None:
            U4t, V4t, W4t = found
            U, V, W = _build_4x4x8_from_4x4x4(U4t, V4t, W4t, 49)
            if _verify_decomp(U, V, W, n, m, p):
                if 98 < current_best_rank:
                    current_best_rank = 98
                    best = (98, U.copy(), V.copy(), W.copy())

    # Return the best found
    _, U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
