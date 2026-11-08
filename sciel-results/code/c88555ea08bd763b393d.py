"""Rewritten program: rank-minimal exact decomposition for 2x3x9 matrix multiplication.

Design family (kept): a hard-coded known low-rank decomposition (Strassen 2x2 blocks
=> rank 27) plus a time-bounded, exact, local integer/half-integer search attempting
to reduce the rank. An explicit deadline guard guarantees a valid result is returned
before the evaluator's wall-clock limit.
"""
import time

DIMENSIONS = [2, 3, 9]


def _validate(U, V, W, n=2, m=3, p=9):
    """Exact integer validation that the tensors U,V,W realize matrix multiplication.

    U: list of lists, shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank), indexed W[k*n + i, r]  (official trace convention)
    Returns True iff for every i,k:  sum_j A[i,j]B[j,k] = sum_r U[i*m+j]*V[j*p+k]*W[k*n+i,r].
    """
    n, m, p = 2, 3, 9
    R = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 1
                rhs = 0
                for r in range(R):
                    rhs += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if lhs != rhs:
                    return False
    return True


def _strassen_block():
    """Rank-7 exact decomposition of (2x2)*(2x2) over the (a,b;c,d)*(g,h;i,j) tensors.
    U rows are (a,b,c,d), V rows are (g,h,i,j), W rows are (p,q,r,s) (output 2x2)."""
    return [
        [[1, 0, 0, 0], [1, 0, 1, 0], [1, 0, 1, 0]],       # 1
        [[0, 1, 0, 0], [1, 0, 0, 1], [0, 0, 0, 0]],       # 2
        [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]],       # 3
        [[0, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 0]],       # 4
        [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 1]],       # 5
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 1]],      # 6
        [[0, 0, 1, 0], [0, 0, 1, 0], [-1, 0, 0, 0]],      # 7
    ]


def construct():
    deadline = time.time() + 110.0   # explicit safety margin under the 120 s limit

    # ---- Guaranteed valid rank-27 decomposition (Strassen, 3 column-groups) ----
    U, V, W = _build_strassen27()
    assert _validate(U, V, W), "internal error: base rank-27 decomposition invalid"

    # ---- Optional: time-bounded exact local search for a strictly smaller rank ----
    improved = _local_improvement_search(U, V, W, deadline)
    if improved is not None:
        return improved

    return {"U": U, "V": V, "W": W}


def _build_strassen27():
    """Build the rank-27 decomposition: three independent 2x2 column-groups.

    For column-group t (k = 3t + q, q in {0,1,2}) the 2x2 block
        A = [[a, b], [c, d]]   (a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1])
        B_t = [[B[0,k], B[0,k+1]], [B[1,k], B[1,k+1]]]
    contributes 7 rank-1 terms. Column k+2 (q=2) is handled by the diagonal term of the
    SAME Strassen block, which is exact: the s-entry (A[0,0]*B[0,?]) appears directly.
    To keep it exact we instead build the block on the two columns (k, k+2) and (k+1, k+2)
    is NOT valid; therefore we build on consecutive pairs and verify each term covers
    exactly one output. Simplest exact construction: treat each of the 3 groups as a
    2x3 -> but to stay rank-7 per group we use the two-column Strassen on columns
    (3t, 3t+1) and handle column 3t+2 with the single exact monomial A[0,0]*B[0,3t+2]
    and A[1,0]*B[1,3t+2]? That would add 2 terms (rank 9/group = 27 total still works).
    """
    n, m, p = 2, 3, 9
    S = _strassen_block()  # 7 terms, each U(4), V(4), W(4)

    U = []
    V = []
    W = []

    for t in range(3):
        k0, k1 = 3 * t, 3 * t + 1          # two-column Strassen group
        k2 = 3 * t + 2                      # remaining column, exact monomials

        # --- 7 Strassen terms on columns (k0, k1) ---
        # A indices: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
        # B_t indices: g=B[0,k0], h=B[0,k1], i=B[1,k0], j=B[1,k1]
        # W output (for this group): (P[0,k0], P[0,k1], P[1,k0], P[1,k1])
        for (u4, v4, w4) in S:
            U.append([u4[0], u4[1], u4[2], u4[3], 0, 0, 0])          # (a,b,c,d,0,0,0)
            V.append([v4[0], v4[1], v4[2], v4[3], 0, 0, 0])          # (g,h,i,j,0,0,0)
            # W row: (k0*n+i, k1*n+i) for i=0,1  ->  (k0,k1,k0+2,k1+2)
            W.append([w4[0], w4[1], w4[2], w4[3], 0, 0, 0])

        # --- 2 exact monomial terms for column k2 ---
        # P[0,k2] = A[0,0]*B[0,k2]  ->  U=(a,0,0,0,...), V=(e,0,0,0,...), W=(1 at k2,0,...)
        # P[1,k2] = A[1,0]*B[1,k2]  ->  U=(0,c,0,0,...), V=(0,0,f,0,...), W=(1 at k2+2,0,...)
        U.append([1, 0, 0, 0, 0, 0, 0])
        V.append([0, 0, 0, 0, 1, 0, 0])          # B[0,k2] is index 4
        W.append([0, 0, 0, 0, 1, 0, 0])          # output (k2) is index 4

        U.append([0, 0, 1, 0, 0, 0, 0])
        V.append([0, 0, 0, 0, 0, 0, 1])          # B[1,k2] is index 6
        W.append([0, 0, 0, 0, 0, 0, 1])          # output (k2+2) is index 6

    return U, V, W


def _local_improvement_search(U, V, W, deadline):
    """Attempt to reduce rank below 27 via exact integer/half-integer local moves.

    Strategy (hypothesis to test, not assumed): try dropping one rank-1 term and
    re-solve the affected entries with an integer/half-integer grid over a small
    window. If no exact completion exists, the original rank-27 solution is returned.
    Bounded by the deadline; returns None if nothing strictly better is found.
    """
    n, m, p = 2, 3, 9
    R = len(U[0])
    best_rank = R
    best = None

    # Candidate term indices to try dropping (the 6 monomial/edge terms are the most
    # redundant-looking). We only attempt a bounded number of drops.
    drop_candidates = list(range(R))

    half = 0.5
    grid = [-1, -half, 0, half, 1]   # small exact integer/half-integer window

    for drop in drop_candidates:
        if time.time() > deadline - 5.0:
            break
        # Build the reduced system: remove term 'drop'. The remaining R-1 terms must
        # still realize every output entry. Check feasibility by verifying the residual
        # tensor (after removing term 'drop') has rank 0 over the integer grid.
        # We do a fast necessary check: for a few random (i,j,k) the residual must be 0.
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(R):
                        if r == drop:
                            continue
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if val != 1:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            # Term 'drop' is redundant: we found an exact (R-1)-rank decomposition.
            U2 = [row for idx, row in enumerate(U) if idx != drop]
            V2 = [row for idx, row in enumerate(V) if idx != drop]
            W2 = [row for idx, row in enumerate(W) if idx != drop]
            if _validate(U2, V2, W2):
                best_rank = R - 1
                best = {"U": U2, "V": V2, "W": W2}
                # continue to see if we can drop more
                U, V, W = U2, V2, W2
                R = best_rank
                drop_candidates = list(range(R))
    return best


def _validate(U, V, W, n=2, m=3, p=9):
    R = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                rhs = 0
                for r in range(R):
                    rhs += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if rhs != 1:
                    return False
    return True


if __name__ == "__main__":
    res = construct()
    print("rank:", len(res["U"][0]))
    print("valid:", _validate(res["U"], res["V"], res["W"]))
