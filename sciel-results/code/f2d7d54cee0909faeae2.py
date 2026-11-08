"""Rank-revealing decomposition for the 3x5 * 5x5 matrix multiplication tensor."""
import time
from fractions import Fraction

import numpy as np

DIMENSIONS = [3, 5, 5]

# Safety margin: the evaluator kills at 150s; leave headroom.
DEADLINE = 125.0


def _schoolbook():
    """Fallback exact decomposition of rank n*m*p (always valid)."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _is_valid(U, V, W, n, m, p, tol=1e-6):
    """Check sum_c U[:,c] V[:,c] == W[:,c] (W indexed by k*n+i)."""
    prod = U @ V.T  # (n*m, m*p)
    for i in range(n):
        for k in range(p):
            for j in range(m):
                lhs = prod[i * m + j, j * p + k]
                rhs = W[k * n + i]
                if abs(lhs - rhs) > tol:
                    return False
    return True


def _rref_rational(A, deadline):
    """Gaussian elimination over the rationals.

    Returns (row_indices, pivot_cols) for a row basis: for each row r in
    row_indices, pivot_cols[r] is the column that pivots that row, and the
    pivot value in A[r, pivot_cols[r]] is 1 (normalized).
    """
    import sympy as sp

    A = sp.Matrix(A)
    nrows, ncols = A.rows, A.cols
    pivot_cols = []
    r = 0
    for c in range(ncols):
        if time.time() > deadline:
            break
        # find a nonzero pivot at or below row r in column c
        pivot = None
        for i in range(r, nrows):
            if A[i, c] != 0:
                pivot = i
                break
        if pivot is None:
            continue
        if pivot != r:
            A = A.row_swap(pivot, r)
        piv_val = A[r, c]
        A = A.row_scale(r, 1 / piv_val)
        for i in range(nrows):
            if i != r and A[i, c] != 0:
                factor = A[i, c]
                A = A.row_add(i, r, -factor)
        pivot_cols.append(c)
        r += 1
        if r == nrows:
            break
    return A, pivot_cols, r


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + DEADLINE

    # Build the linear system: unknowns U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    # Equation (i, j, k): sum_c U[i*m+j,c] V[j*p+k,c] == W[k*n+i,c].
    #
    # Solve over the rationals for the solution space in the W variables,
    # with U and V fixed to the schoolbook basis (each (i,j,k) term has
    # U[i*m+j,c]=V[j*p+k,c]=1 for a single c and 0 otherwise). This is a
    # linear system in the W variables whose solution space is spanned by
    # the rank-1 terms, giving a rank-minimal decomposition.

    # We instead do a direct rank-revealing approach: treat each output entry
    # C[i,k] = sum_j A[i,j] B[j,k] as a linear form in the scalar products
    # P_{i,j,k} = A[i,j]*B[j,k]. The multiplication tensor T has entries
    # T[(i,k),(j)] = 1 if ... Actually the standard tensor:
    #   T_{(i,j),(j,k)} = delta  -> C[i,k] = sum_j A[i,j] B[j,k].
    #
    # Rank decomposition: find the minimal R such that there exist U (n*m x R),
    # V (m*p x R), W (p*n x R) with sum_r U[a,r] V[b,r] = W[(k*n+i), r]
    # for a = i*m+j, b = j*p+k. This is exactly the bilinear structure.
    #
    # We solve it by expressing W entries as linear combinations of the
    # products A[i,j] B[j,k]. The space of such products has dimension at most
    # n*m*p. We build the linear map and find its rank.

    # Construct the linear system directly in the W variables.
    # For each (i, j, k):  sum_c U[i*m+j,c] V[j*p+k,c] - W[k*n+i,c] = 0.
    # Fix U and V to be the standard basis vectors (schoolbook) so that the
    # only freedom is in W, and the solution space for W is spanned by the
    # standard basis. That gives rank = n*m*p, which is not minimal.

    # Better: do a true rank-revealing elimination on the tensor.
    # Flatten the multiplication tensor as a matrix M of size (n*p) x (m*m)
    # where M[(i,k), (j, j')] = 1 if j == j' (i.e., the identity in the
    # middle index). The rank of this matrix is the rank of the tensor when
    # sliced this way, which equals m (the shared dimension). But that's the
    # "slice rank", not the tensor rank.

    # The correct approach: the tensor T in V1*V2*V3 where V1=R^(n*m),
    # V2=R^(m*p), V3=R^(p*n). T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # for some (i,j,k), else 0. We want the minimal R with T = sum_r u_r v_r w_r.

    # Build the "matricization" M1: shape (n*m) x (m*p * p*n).
    # M1[a, (b, c)] = T[a, b, c].
    # The rank of M1 is an upper bound on the tensor rank. We compute a
    # basis of the column space of M1; each basis column gives a rank-1 term.

    import sympy as sp

    # Build M1 as a rational matrix.
    rows = n * m
    cols = m * p * p * n
    M1 = sp.zeros(rows, cols)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                M1[a, b * (p * n) + c] = 1

    # Compute RREF to find the column space dimension and a basis.
    M1_rref, pivots = M1.rref()
    rank = len(pivots)

    # For each pivot column, extract the corresponding rank-1 term.
    # The RREF tells us which columns are pivot columns. For each pivot
    # column pc, the corresponding original column of M1 gives the (u, v*w)
    # decomposition. We need to recover u (length n*m), and the pair (v, w).

    # Strategy: for each pivot column pc, the column of M1 is a vector in
    # R^(n*m). We set u_r = that column. Then we need v_r (length m*p) and
    # w_r (length p*n) such that outer(v_r, w_r) = M1[:, pc] reshaped as
    # (m*p, p*n). Since M1[:, pc] is the column, reshape to (m*p, p*n) and
    # factor as rank-1.

    U_cols = []
    V_cols = []
    W_cols = []

    for pc in pivots:
        col = M1[:, pc]  # length n*m
        # Reshape to (m*p, p*n)
        mat = col.reshape(m * p, p * n)  # sympy Matrix reshape
        # Factor mat = v * w^T (rank-1). Since mat should be rank-1.
        # Find a nonzero entry to set v and w.
        v = sp.zeros(m * p, 1)
        w = sp.zeros(p * n, 1)
        found = False
        for bi in range(m * p):
            if time.time() > deadline:
                break
            for bk in range(p * n):
                val = mat[bi, bk]
                if val != 0:
                    v[bi] = 1
                    for bk2 in range(p * n):
                        w[bk2] = mat[bi, bk2]
                    found = True
                    break
            if found:
                break
        if not found:
            # Should not happen for a pivot column.
            continue
        U_cols.append(col)
        V_cols.append(v)
        W_cols.append(w)

    R = len(U_cols)
    if R == 0:
        U, V, W = _schoolbook()
        return {"U": U, "V": V, "W": W}

    U_mat = sp.Matrix(U_cols).T  # (n*m, R)
    V_mat = sp.Matrix(V_cols).T  # (m*p, R)
    W_mat = sp.Matrix(W_cols).T  # (p*n, R)

    # Verify exactness.
    try:
        U_np = np.array(U_mat.tolist(), dtype=float)
        V_np = np.array(V_mat.tolist(), dtype=float)
        W_np = np.array(W_mat.tolist(), dtype=float)
        if _is_valid(U_np, V_np, W_np, n, m, p):
            return {"U": U_np, "V": V_np, "W": W_np}
    except Exception:
        pass

    # Fallback.
    U, V, W = _schoolbook()
    return {"U": U, "V": V, "W": W}
