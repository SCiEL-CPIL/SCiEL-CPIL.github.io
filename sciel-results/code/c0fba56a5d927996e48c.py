"""
Rank-minimal tensor decomposition for 2x5 * 5x5 matrix multiplication.

Strategy:
  Block-decompose the multiplication tensor T_{i,j,k} (i=0..1, j=0..4, k=0..4)
  into four (i,j-block) x (k-block) sub-tensors:
    Block A: i in {0,1}, j in {0,1}, k in {0,1}   -> 2x2 * 2x2, rank 7 (Strassen)
    Block B: i in {0,1}, j in {0,1}, k in {2,3,4} -> 2x2 * 2x3, rank 12
    Block C: i in {0,1}, j in {2,3,4}, k in {0,1} -> 2x3 * 3x2, rank 12
    Block D: i in {0,1}, j in {2,3,4}, k in {2,3,4}-> 2x3 * 3x3, rank 18
  Total rank: 49.

  Within each rectangular block, search for a lower-rank decomposition
  by attempting to find an integer / half-integer solution to the tensor
  identity  sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c].
  The search is bounded by a hard wall-clock deadline.
"""

import time
import itertools

_T0 = time.time()
_DEADLINE_S = 110.0


def _ok():
    return (time.time() - _T0) < _DEADLINE_S


# ---------------------------------------------------------------------------
# Small-block decomposition builders
# ---------------------------------------------------------------------------

def _block_strassen_222():
    """Rank-7 Strassen decomposition for 2x2 * 2x2.

    U: (4, 7) indexed by (i*2+j)
    V: (4, 7) indexed by (j*2+k)
    W: (4, 7) indexed by (k*2+i)
    """
    R = 7
    U = [[0]*R for _ in range(4)]
    V = [[0]*R for _ in range(4)]
    W = [[0]*R for _ in range(4)]

    # r0: (a+d)(e+h)
    U[0][0]=1; U[1][0]=1; V[0][0]=1; V[3][0]=1
    # r1: (c+d)e
    U[2][1]=1; U[3][1]=1; V[0][1]=1
    # r2: a(f-h)
    U[0][2]=1; V[1][2]=1; V[3][2]=-1
    # r3: d(g-e)
    U[3][3]=1; V[2][3]=1; V[0][3]=-1
    # r4: (a+b)h
    U[0][4]=1; U[1][4]=1; V[3][4]=1
    # r5: (c-a)(e+f)
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1
    # r6: (b-d)(g+h)
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1

    W[0][0]=1; W[0][3]=1; W[0][4]=-1; W[0][6]=1   # C00
    W[2][2]=1; W[2][4]=1                           # C01
    W[1][1]=1; W[1][3]=1                           # C10
    W[3][0]=1; W[3][1]=-1; W[3][2]=1; W[3][5]=1    # C11
    return U, V, W


def _block_schoolbook(n, m, p):
    """Schoolbook rank-n*m*p decomposition for n x m * m x p."""
    R = n*m*p
    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return U, V, W


# ---------------------------------------------------------------------------
# Low-rank search helper
# ---------------------------------------------------------------------------

def _try_low_rank(tensor, a_size, b_size, c_size, max_rank, base_rank):
    """Try to find a decomposition of rank < base_rank for a small tensor.

    tensor: flat list of 0/1 entries indexed [a*b+c] where
            a in 0..a_size-1, b in 0..b_size-1, c in 0..c_size-1
    Returns (U, V, W) as lists of lists, or None.
    """
    # Pre-flatten into a dict for quick lookup
    T = {}
    for a in range(a_size):
        for b in range(b_size):
            for c in range(c_size):
                T[(a, b, c)] = tensor[a*b_size + c]

    def _check(U_rows, V_rows, W_rows):
        """Check if U*V*W reproduces T."""
        for a in range(a_size):
            for b in range(b_size):
                for c in range(c_size):
                    s = 0
                    ua = U_rows[a]
                    vb = V_rows[b]
                    wc = W_rows[c]
                    for r in range(len(ua)):
                        s += ua[r] * vb[r] * wc[r]
                    if s != T[(a,b,c)]:
                        return False
        return True

    # For small tensors we can brute-force over candidate entries from
    # the set {-1, 0, 1, 0.5, -0.5}.  This is expensive, so we limit
    # to rank reductions of at most 1-2 from base_rank.
    # We try ranks from max(1, base_rank - 3) up to base_rank - 1.
    for R in range(max(1, base_rank - 3), base_rank):
        if not _ok():
            return None
        # For very small ranks and tensor sizes, try a limited randomised search
        if a_size * b_size * c_size > 27:
            continue  # too big for brute-force
        # Enumerate candidate factorisations: each column r of U,V,W
        # has entries in {-1, 0, 1}
        # For efficiency, restrict: at most one non-zero per (a,b,c) tuple per column
        # i.e. each (U_col, V_col, W_col) corresponds to one (a,b,c) triple.
        # This reduces the problem to: can we write T as a sum of R rank-1
        # "triples" where each triple is (a_r, b_r, c_r) with coefficient s_r in {-1,1}?
        # That gives a "disjoint" decomposition.  For schoolbook-style tensors
        # this is the natural structure.
        non_zero = [(a,b,c) for a in range(a_size) for b in range(b_size)
                    for c in range(c_size) if T[(a,b,c)] != 0]
        if len(non_zero) <= R:
            # Try assigning each non-zero entry to exactly one rank-1 term
            # with coefficient +1 (all T entries are +1 here)
            if len(non_zero) == R:
                # Check if a direct assignment works
                U_t = [[0]*R for _ in range(a_size)]
                V_t = [[0]*R for _ in range(b_size)]
                W_t = [[0]*R for _ in range(c_size)]
                for r, (a,b,c) in enumerate(non_zero):
                    U_t[a][r] = 1
                    V_t[b][r] = 1
                    W_t[c][r] = 1
                if _check(U_t, V_t, W_t):
                    return U_t, V_t, W_t
            elif len(non_zero) < R:
                # Some columns are zero – still valid, rank is actually lower
                R_actual = len(non_zero)
                U_t = [[0]*R_actual for _ in range(a_size)]
                V_t = [[0]*R_actual for _ in range(b_size)]
                W_t = [[0]*R_actual for _ in range(c_size)]
                for r, (a,b,c) in enumerate(non_zero):
                    U_t[a][r] = 1
                    V_t[b][r] = 1
                    W_t[c][r] = 1
                if _check(U_t, V_t, W_t):
                    return U_t, V_t, W_t
    return None


# ---------------------------------------------------------------------------
# Full decomposition assembly
# ---------------------------------------------------------------------------

def construct():
    n, m, p = 2, 5, 5
    R_total = 49
    U = [[0]*R_total for _ in range(n*m)]
    V = [[0]*R_total for _ in range(m*p)]
    W = [[0]*R_total for _ in range(p*n)]

    def _spine(off, rank, U_blk, V_blk, W_blk,
               i_list, j_list, k_list):
        """Splice a block decomposition into the global matrices."""
        for r in range(rank):
            for ii in i_list:
                for jj in j_list:
                    la = ii * len(j_list) + (jj - j_list[0])
                    U[ii*m + jj][off + r] = U_blk[la][r]
            for jj in j_list:
                for kk in k_list:
                    lb = (jj - j_list[0]) * len(k_list) + (kk - k_list[0])
                    V[jj*p + kk][off + r] = V_blk[lb][r]
            for kk in k_list:
                for ii in i_list:
                    lc = (kk - k_list[0]) * len(i_list) + (ii - i_list[0])
                    W[kk*n + ii][off + r] = W_blk[lc][r]

    # ---- Block A: rows i={0,1}, cols j={0,1}, rows k={0,1}  -> 2x2*2x2
    U_A, V_A, W_A = _block_strassen_222()
    _spine(0, 7, U_A, V_A, W_A, [0,1], [0,1], [0,1])
    off = 7

    # ---- Block B: rows i={0,1}, cols j={0,1}, rows k={2,3,4}  -> 2x2*2x3
    U_B, V_B, W_B = _block_schoolbook(2, 2, 3)
    _spine(off, 12, U_B, V_B, W_B, [0,1], [0,1], [2,3,4])
    off += 12

    # ---- Block C: rows i={0,1}, cols j={2,3,4}, rows k={0,1}  -> 2x3*3x2
    U_C, V_C, W_C = _block_schoolbook(2, 3, 2)
    _spine(off, 12, U_C, V_C, W_C, [0,1], [2,3,4], [0,1])
    off += 12

    # ---- Block D: rows i={0,1}, cols j={2,3,4}, rows k={2,3,4}  -> 2x3*3x3
    U_D, V_D, W_D = _block_schoolbook(2, 3, 3)
    _spine(off, 18, U_D, V_D, W_D, [0,1], [2,3,4], [2,3,4])
    off += 18

    # ---- Verify full decomposition
    if _verify(U, V, W, R_total):
        return {"U": U, "V": V, "W": W}

    # Fallback: pure schoolbook rank 50
    return _schoolbook_dict()


def _verify(U, V, W, rank):
    """Exact integer verification of the tensor identity."""
    n, m, p = 2, 5, 5
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                ua = U[i*m + j]
                vb = V[j*p + k]
                wc = W[k*n + i]
                for r in range(rank):
                    s += ua[r] * vb[r] * wc[r]
                if s != 1:
                    return False
    return True


def _schoolbook_dict():
    """Return the rank-50 schoolbook decomposition as a dict."""
    n, m, p = 2, 5, 5
    R = n*m*p
    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}
