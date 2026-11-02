import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2_factors():
    """
    Returns U, V, W for 2x2 matrix multiplication using Strassen's algorithm (rank 7).
    Indexing: U[i*2+j, r] for A[i,j], V[j*2+k, r] for B[j,k], W[k*2+i, r] for C[i,k]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    terms = [
        # P1 = (A00+A11)(B00+B11) -> C00 += P1
        ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1)]),
        # P2 = (A10+A11)B00 -> C00 += P2, C10 += P2
        ([(2,1), (3,1)], [(0,1)], [(0,1), (2,1)]),
        # P3 = A00(B01-B11) -> C01 += P3, C11 += P3
        ([(0,1)], [(1,1), (3,-1)], [(1,1), (3,1)]),
        # P4 = A11(B10-B00) -> C00 += P4, C10 += P4
        ([(3,1)], [(2,1), (0,-1)], [(0,1), (2,1)]),
        # P5 = (A00+A01)B11 -> C00 -= P5, C01 += P5
        ([(0,1), (1,1)], [(3,1)], [(0,-1), (1,1)]),
        # P6 = (A10-A00)(B00+B01) -> C11 += P6
        ([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]),
        # P7 = (A01-A11)(B10+B11) -> C00 += P7
        ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1)]),
    ]

    for r, (a_coeffs, b_coeffs, c_coeffs) in enumerate(terms):
        for idx, coeff in a_coeffs:
            U[idx][r] = coeff
        for idx, coeff in b_coeffs:
            V[idx][r] = coeff
        for idx, coeff in c_coeffs:
            W[idx][r] = coeff

    return U, V, W

def _strassen_recursive_factors(n):
    """
    Build Strassen factors for n x n (n power of 2) matrix multiplication.
    Returns U, V, W with shapes (n*n, rank), (n*n, rank), (n*n, rank).
    U[i*n+j], V[j*n+k], W[k*n+i].
    """
    if n == 1:
        U = [[1]]
        V = [[1]]
        W = [[1]]
        return U, V, W
    if n == 2:
        return _strassen_2x2_factors()

    half = n // 2
    U_half, V_half, W_half = _strassen_recursive_factors(half)
    rank = len(U_half[0])

    # Build full-size factors.
    # A is n x n, split into 4 blocks of half x half:
    #   A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A*B = [[C00, C01], [C10, C11]]
    #
    # For each Strassen term r (acting on the half-size blocks), we need to
    # map block-level (i_block, j_block) to full (i, j).
    # Block (bi, bj) of A corresponds to rows bi*half..(bi+1)*half-1, cols bj*half..(bj+1)*half-1.
    # Full index i*n + j where i = bi*half + ii, j = bj*half + jj.
    #
    # U_half[bi*half + jj_local, r] contributes to A block (bi, bj)? 
    # In the half-size problem, A_half is half x half with index bi*half + jj.
    # But here A_half's "row index" bi*half + jj means block-row bi, and within-block col jj.
    # Wait: in the half-size matmul, A_half[i,j] with i in [0,half), j in [0,half).
    # The block (bi, bj) of the full A maps to A_half[bi, bj] (a single scalar in the half problem).
    # So U_half[bi*half + bj, r] is the coefficient for A_half[bi, bj] = full A block (bi, bj).
    #
    # So for full U, for each (bi, bj) in [0,2)^2 and each local (ii, jj) in [0,half)^2:
    #   full row = (bi*half + ii) * n + (bj*half + jj)
    #   U[full_row, r] += U_half[bi*half + bj, r]
    #
    # Similarly V: V_half[bj*half + bk, r] for B block (bj, bk).
    #   full row = (bj*half + jj) * n + (bk*half + kk)
    #   V[full_row, r] += V_half[bj*half + bk, r]
    #
    # W: W_half[bk*half + bi, r] for C block (bi, bk) [since W[k*n+i], C[i,k] block (bi,bk)].
    #   full row = (bk*half + kk) * n + (bi*half + ii)
    #   W[full_row, r] += W_half[bk*half + bi, r]

    U = [[0]*rank for _ in range(n*n)]
    V = [[0]*rank for _ in range(n*n)]
    W = [[0]*rank for _ in range(n*n)]

    for bi in range(2):
        for bj in range(2):
            coeff_a = U_half[bi*half + bj]
            for ii in range(half):
                for jj in range(half):
                    full_row = (bi*half + ii) * n + (bj*half + jj)
                    for r in range(rank):
                        U[full_row][r] += coeff_a[r]

    for bj in range(2):
        for bk in range(2):
            coeff_b = V_half[bj*half + bk]
            for jj in range(half):
                for kk in range(half):
                    full_row = (bj*half + jj) * n + (bk*half + kk)
                    for r in range(rank):
                        V[full_row][r] += coeff_b[r]

    for bk in range(2):
        for bi in range(2):
            coeff_c = W_half[bk*half + bi]
            for kk in range(half):
                for ii in range(half):
                    full_row = (bk*half + kk) * n + (bi*half + ii)
                    for r in range(rank):
                        W[full_row][r] += coeff_c[r]

    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] == 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Compute X[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # Vectorize: X[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(len(U[0])):
                    s += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                if abs(s - 1.0) > 1e-9:
                    ok = False
                    return ok
    return ok

def _project_strassen_8x8_to_5x5():
    """
    Build 8x8 Strassen factors, then drop terms (columns) that do not
    contribute to the 5x5 output. A term c contributes to the 5x5 output
    only if there exist i,j,k in [0,5) with U[i*5+j,c] != 0 AND V[j*5+k,c] != 0
    AND W[k*5+i,c] != 0 (in the 5x5 sub-indexing).
    """
    n8 = 8
    U8, V8, W8 = _strassen_recursive_factors(n8)
    rank8 = len(U8[0])

    n = 5
    # We keep only columns c where the term is "active" within the 5x5 region.
    # A term c is active if there exist (i,j,k) in [0,5)^3 such that
    # U8[i*8+j, c] != 0, V8[j*8+k, c] != 0, W8[k*8+i, c] != 0.
    # But for correctness of the 5x5 decomposition, we need:
    #   sum_c U5[i*5+j,c] V5[j*5+k,c] W5[k*5+i,c] = 1 for all i,j,k in [0,5).
    # If we simply take the 5x5 sub-matrices of U8, V8, W8 (rows/cols restricted to
    # the 5x5 index set) and keep ALL 343 columns, the sum equals 1 (since the full
    # 8x8 decomposition is exact and the 5x5 is a sub-tensor). But we can drop columns
    # that are identically zero on the 5x5 index set.
    #
    # Actually, dropping a column that is non-zero somewhere in the 5x5 set would break
    # correctness. So we can only drop columns that are ZERO on the entire 5x5 index set
    # (i.e., U8[i*8+j,c]=0 for all i,j in [0,5), or V8[j*8+k,c]=0 for all j,k in [0,5),
    # or W8[k*8+i,c]=0 for all k,i in [0,5)).
    #
    # But a column might have U8 non-zero on 5x5 rows, V8 non-zero, W8 non-zero, yet the
    # sum over such columns still reconstructs 1. We CANNOT drop them.
    # So the only safe drop is columns that are entirely zero on the 5x5 region.
    #
    # Let's check how many columns are entirely zero on the 5x5 region.
    keep = []
    for c in range(rank8):
        # Check if U8 column c has any non-zero in rows corresponding to 5x5 A indices.
        # A index (i,j) -> row i*8+j, for i,j in [0,5).
        u_active = any(U8[i*8 + j, c] != 0 for i in range(5) for j in range(5))
        v_active = any(V8[j*8 + k, c] != 0 for j in range(5) for k in range(5))
        w_active = any(W8[k*8 + i, c] != 0 for k in range(5) for i in range(5))
        if u_active and v_active and w_active:
            keep.append(c)

    # Build projected factors.
    U = [[0]*len(keep) for _ in range(25)]
    V = [[0]*len(keep) for _ in range(25)]
    W = [[0]*len(keep) for _ in range(25)]
    for new_c, c in enumerate(keep):
        for i in range(5):
            for j in range(5):
                U[i*5 + j][new_c] = U8[i*8 + j, c]
        for j in range(5):
            for k in range(5):
                V[j*5 + k][new_c] = V8[j*8 + k, c]
        for k in range(5):
            for i in range(5):
                W[k*5 + i][new_c] = W8[k*8 + i, c]
    return U, V, W

def _try_5x5_decomposition():
    """
    Try to build a low-rank 5x5 decomposition.
    Strategy: pad 5x5 into 8x8, use Strassen (rank 343), project to 5x5,
    then drop columns that are zero on the 5x5 region.
    """
    U, V, W = _project_strassen_8x8_to_5x5()
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # safety margin before 120s limit

    # Best fallback: schoolbook rank 125.
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1

    best = (rank_fb, U_fb, V_fb, W_fb)

    # Try the Strassen-projection approach.
    try:
        U, V, W = _try_5x5_decomposition()
        rank = len(U[0])
        # Verify correctness.
        if _verify(U, V, W, n, m, p):
            if rank < best[0]:
                best = (rank, U, V, W)
    except Exception:
        pass

    if time.time() > deadline:
        rank, U, V, W = best
        return {"U": U, "V": V, "W": W}

    rank, U, V, W = best
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
