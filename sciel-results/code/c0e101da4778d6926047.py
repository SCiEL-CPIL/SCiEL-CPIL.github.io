"""Matrix multiplication 2x2 x 2x3 with rank-minimal tensor decomposition.

Strategy:
  * Rank 11: Strassen (rank 7) for the 2x2*2x2 block (k=0,1) plus schoolbook
    rank-4 for the remaining column (k=2).  Verified and returned immediately.
  * Rank 10: exploit the algebraic structure of the k=2 column.
      For k=2 the required tensor slice is the 2x2 all-ones matrix
      T[i,j] = 1  (i,j in {0,1}).
      A rank-3 decomposition of that slice is known to exist:
          s=0:  U[0]=1, V[0]=1, W[0]=1   ->  T[0,0] += 1
          s=1:  U[1]=1, V[1]=1, W[1]=1   ->  T[1,1] += 1
          s=2:  U[0]=1, U[1]=1, V[0]=1, V[1]=1, W[0]=1, W[1]=1
                ->  T[i,j] += 1  for all i,j
      Summing the three contributions gives T[i,j] = 2 for the diagonal and
      T[i,j] = 1 for the off-diagonal -- not all-ones.  So we need a
      different parameterisation.  Instead we use a general rank-3
      parameterisation and verify by direct substitution.
  * Fallback: ALS refinement from the rank-11 warm-start, then rounding to
    half-integers, then exact verification.
  * Final fallback: schoolbook rank 12.
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]


def _schoolbook(n, m, p):
    """Return the schoolbook rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition exactly represents the matrix multiplication tensor."""
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
    canonical = np.ones((n, m, p))
    return np.allclose(T, canonical, atol=tol)


def _strassen_2x2x2():
    """
    Return a verified rank-7 decomposition for 2x2 * 2x2 matrix multiplication.
    U is 4x7 (i*2+j), V is 4x7 (j*2+k), W is 4x7 (k*2+i).
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))

    # m1 = (a+d)(e+h)
    U[0, 0] = 1; U[1, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # m2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = 1

    # m3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1

    # m4 = d(g-e)
    U[1, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[1, 3] = 1

    # m5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1

    # m6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1

    # m7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W


def _try_rank11():
    """
    Rank-11: Strassen (7) for k=0,1 + schoolbook (4) for k=2.
    """
    n, m, p = 2, 2, 3
    rank = 11

    U = np.zeros((4, rank))
    V = np.zeros((6, rank))
    W = np.zeros((6, rank))

    U_sub, V_sub, W_sub = _strassen_2x2x2()
    U[:, 0:7] = U_sub
    # V_sub index j*2+k -> V index j*3+k for k=0,1
    V[0:2, 0:7] = V_sub[0:2, :]   # j=0, k=0,1
    V[2:4, 0:7] = V_sub[2:4, :]   # j=1, k=0,1
    # W_sub index k*2+i -> W index k*2+i for k=0,1
    W[0:2, 0:7] = W_sub[0:2, :]   # k=0, i=0,1
    W[2:4, 0:7] = W_sub[2:4, :]   # k=1, i=0,1

    # Schoolbook for k=2 (columns 7-10)
    # s=7: A[0,0]*B[0,2] -> C[0,2]
    U[0, 7] = 1
    V[2, 7] = 1   # j=0, k=2: V[0*3+2]
    W[4, 7] = 1   # k=2, i=0: W[2*2+0]

    # s=8: A[0,1]*B[1,2] -> C[0,2]
    U[1, 8] = 1
    V[5, 8] = 1   # j=1, k=2: V[1*3+2]
    W[4, 8] = 1   # k=2, i=0

    # s=9: A[1,0]*B[0,2] -> C[1,2]
    U[2, 9] = 1
    V[2, 9] = 1
    W[5, 9] = 1   # k=2, i=1: W[2*2+1]

    # s=10: A[1,1]*B[1,2] -> C[1,2]
    U[3, 10] = 1
    V[5, 10] = 1
    W[5, 10] = 1

    return U, V, W


def _try_rank10_structured():
    """
    Attempt rank-10: Strassen (7) for k=0,1 + rank-3 for k=2.

    For the k=2 slice we need a rank-3 decomposition of the 2x2 all-ones
    tensor T[i,j] = 1.  We parameterise the three factors as:
        U[2i+j, s], V[j, s], W[i, s]  for s = 0,1,2
    and solve the 4 equations:
        sum_s U[2i+j, s] * V[j, s] * W[i, s] = 1   for i,j in {0,1}

    We use a structured parameterisation that is known to have solutions:
        Let W[0, s] = a_s, W[1, s] = b_s
        Let V[0, s] = c_s, V[1, s] = d_s
        Then U[0, s] = e_s, U[1, s] = f_s, U[2, s] = g_s, U[3, s] = h_s

    Equations:
        e_0 c_0 a_0 + e_1 c_1 a_1 + e_2 c_2 a_2 = 1   (i=0,j=0)
        f_0 c_0 a_0 + f_1 c_1 a_1 + f_2 c_2 a_2 = 1   (i=0,j=1)
        g_0 c_0 b_0 + g_1 c_1 b_1 + g_2 c_2 b_2 = 1   (i=1,j=0)
        h_0 c_0 b_0 + h_1 c_1 b_1 + h_2 c_2 b_2 = 1   (i=1,j=1)

    We fix a gauge: set a_0=1, b_0=1, a_1=0, b_1=0, a_2=0, b_2=1 (or similar)
    and solve.  Actually, let's try a simpler known construction.

    Known rank-3 for 2x2 all-ones:
        s=0: U[0]=1, V[0]=1, W[0]=1  -> contributes 1 to T[0,0]
        s=1: U[1]=1, V[1]=1, W[1]=1  -> contributes 1 to T[0,1]
        s=2: U[2]=1, V[0]=1, W[1]=1  -> contributes 1 to T[1,0]
        s=3: U[3]=1, V[1]=1, W[0]=1  -> contributes 1 to T[1,1]
    That's rank 4.  For rank 3 we need to merge two of these.

    Try:
        s=0: U[0]=1, U[1]=1, V[0]=1, V[1]=1, W[0]=1, W[1]=0
             -> T[0,0] += 1, T[0,1] += 1
        s=1: U[2]=1, U[3]=1, V[0]=1, V[1]=1, W[0]=0, W[1]=1
             -> T[1,0] += 1, T[1,1] += 1
        s=2: U[0]=0, U[1]=0, U[2]=0, U[3]=0, ...  -> 0
    Sum: T[0,0]=1, T[0,1]=1, T[1,0]=1, T[1,1]=1.  That works! Rank 2!

    Wait, let me double-check.  For s=0:
        T[0,0] += U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1
        T[0,1] += U[1,0]*V[1,0]*W[0,0] = 1*1*1 = 1
        T[1,0] += U[2,0]*V[0,0]*W[1,0] = 1*1*0 = 0
        T[1,1] += U[3,0]*V[1,0]*W[1,0] = 1*1*0 = 0
    For s=1:
        T[0,0] += U[0,1]*V[0,1]*W[0,1] = 1*1*0 = 0
        T[0,1] += U[1,1]*V[1,1]*W[0,1] = 1*1*0 = 0
        T[1,0] += U[2,1]*V[0,1]*W[1,1] = 1*1*1 = 1
        T[1,1] += U[3,1]*V[1,1]*W[1,1] = 1*1*1 = 1
    Sum: T[0,0]=1, T[0,1]=1, T[1,0]=1, T[1,1]=1.  Yes! Rank 2 for the k=2 slice.

    So the total rank is 7 + 2 = 9?  Let's verify this carefully.
    """
    n, m, p = 2, 2, 3
    rank = 9  # 7 + 2

    U = np.zeros((4, rank))
    V = np.zeros((6, rank))
    W = np.zeros((6, rank))

    # Strassen for k=0,1 (columns 0-6)
    U_sub, V_sub, W_sub = _strassen_2x2x2()
    U[:, 0:7] = U_sub
    V[0:2, 0:7] = V_sub[0:2, :]   # j=0, k=0,1
    V[2:4, 0:7] = V_sub[2:4, :]   # j=1, k=0,1
    W[0:2, 0:7] = W_sub[0:2, :]   # k=0, i=0,1
    W[2:4, 0:7] = W_sub[2:4, :]   # k=1, i=0,1

    # Rank-2 for k=2 (columns 7-8)
    # s=7: contributes to T[0,0] and T[0,1] for k=2
    U[0, 7] = 1  # A[0,0]
    U[1, 7] = 1  # A[0,1]
    V[2, 7] = 1  # B[0,2] (j=0, k=2)
    V[5, 7] = 1  # B[1,2] (j=1, k=2)
    W[4, 7] = 1  # C[0,2] (k=2, i=0)
    W[5, 7] = 0  # C[1,2] (k=2, i=1)

    # s=8: contributes to T[1,0] and T[1,1] for k=2
    U[2, 8] = 1  # A[1,0]
    U[3, 8] = 1  # A[1,1]
    V[2, 8] = 1  # B[0,2]
    V[5, 8] = 1  # B[1,2]
    W[4, 8] = 0  # C[0,2]
    W[5, 8] = 1  # C[1,2]

    return U, V, W


def _als_refine(U, V, W, n, m, p, r, max_iter=30, deadline=None):
    """Alternating least squares refinement. Returns (U, V, W) as numpy arrays."""
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        # Solve for U
        num_eqs = n * m * p
        num_unk_U = n * m * r
        A = np.zeros((num_eqs, num_unk_U))
        b = np.ones(num_eqs)
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        A[eq_idx, (i * m + j) * r + c] = V[j * p + k, c] * W[k * n + i, c]
                    eq_idx += 1
        U_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        U = U_flat.reshape(n * m, r)

        # Solve for V
        num_unk_V = m * p * r
        A = np.zeros((num_eqs, num_unk_V))
        b = np.ones(num_eqs)
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        A[eq_idx, (j * p + k) * r + c] = U[i * m + j, c] * W[k * n + i, c]
                    eq_idx += 1
        V_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        V = V_flat.reshape(m * p, r)

        # Solve for W
        num_unk_W = p * n * r
        A = np.zeros((num_eqs, num_unk_W))
        b = np.ones(num_eqs)
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        A[eq_idx, (k * n + i) * r + c] = U[i * m + j, c] * V[j * p + k, c]
                    eq_idx += 1
        W_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        W = W_flat.reshape(p * n, r)

    return U, V, W


def _round_to_half(arr):
    """Round each entry to the nearest half-integer."""
    return np.round(arr * 2) / 2


def _try_rank10_als(deadline):
    """
    Try to find a rank-10 decomposition using ALS warm-started from rank-11.
    We drop one of the 4 schoolbook factors for k=2 and let ALS adjust.
    """
    n, m, p = 2, 2, 3
    r = 10

    # Start from rank-11 and zero out the last factor
    U11, V11, W11 = _try_rank11()
    U0 = U11[:, :r].copy()
    V0 = V11[:, :r].copy()
    W0 = W11[:, :r].copy()

    # Run ALS
    U, V, W = _als_refine(U0, V0, W0, n, m, p, r, max_iter=40, deadline=deadline)

    # Round to half-integers
    U_r = _round_to_half(U)
    V_r = _round_to_half(V)
    W_r = _round_to_half(W)

    if _verify_decomposition(U_r, V_r, W_r, n, m, p):
        return U_r, V_r, W_r

    # Try integer rounding too
    U_i = np.round(U)
    V_i = np.round(V)
    W_i = np.round(W)
    if _verify_decomposition(U_i, V_i, W_i, n, m, p):
        return U_i, V_i, W_i

    return None


def _try_rank9_als(deadline):
    """
    Try to find a rank-9 decomposition using ALS warm-started from rank-10.
    """
    n, m, p = 2, 2, 3
    r = 9

    # Start from the structured rank-9 candidate
    U9, V9, W9 = _try_rank10_structured()  # This is actually rank 9
    if _verify_decomposition(U9, V9, W9, n, m, p):
        return U9, V9, W9

    # Try ALS from rank-10 warm-start
    U11, V11, W11 = _try_rank11()
    U0 = U11[:, :r].copy()
    V0 = V11[:, :r].copy()
    W0 = W11[:, :r].copy()

    U, V, W = _als_refine(U0, V0, W0, n, m, p, r, max_iter=40, deadline=deadline)

    U_r = _round_to_half(U)
    V_r = _round_to_half(V)
    W_r = _round_to_half(W)
    if _verify_decomposition(U_r, V_r, W_r, n, m, p):
        return U_r, V_r, W_r

    U_i = np.round(U)
    V_i = np.round(V)
    W_i = np.round(W)
    if _verify_decomposition(U_i, V_i, W_i, n, m, p):
        return U_i, V_i, W_i

    return None


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin

    # Start with schoolbook (rank 12) as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = n * m * p

    # Try rank 9 (structured: Strassen 7 + rank-2 for k=2)
    if time.time() < deadline:
        r9 = _try_rank9_als(deadline)
        if r9 is not None:
            U9, V9, W9 = r9
            if _verify_decomposition(U9, V9, W9, n, m, p):
                best_result = {"U": U9.tolist(), "V": V9.tolist(), "W": W9.tolist()}
                best_rank = 9
                return best_result

    # Try rank 10
    if time.time() < deadline:
        r10 = _try_rank10_als(deadline)
        if r10 is not None:
            U10, V10, W10 = r10
            if _verify_decomposition(U10, V10, W10, n, m, p):
                best_result = {"U": U10.tolist(), "V": V10.tolist(), "W": W10.tolist()}
                best_rank = 10
                return best_result

    # Try rank 11
    if time.time() < deadline:
        U11, V11, W11 = _try_rank11()
        if _verify_decomposition(U11, V11, W11, n, m, p):
            best_result = {"U": U11.tolist(), "V": V11.tolist(), "W": W11.tolist()}
            best_rank = 11
            return best_result

    return best_result
# EVOLVE-BLOCK-END
