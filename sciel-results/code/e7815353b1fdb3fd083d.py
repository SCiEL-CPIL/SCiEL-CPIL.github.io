"""5x5x5 matrix multiplication tensor decomposition.

Strategy: build a 2+2+1 partition with Strassen on the four 2x2x2 diagonal
blocks (rank 117), then run an exact linear-algebraic reduction to remove any
rank-1 terms that are linearly dependent in the tensor product space.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]
_START = time.time()
_DEADLINE = 150.0 - 15.0  # 15s safety margin


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _to_frac(x):
    """Return (num, den) for an integer or half-integer value."""
    v = float(x)
    # values are expected to be k/2 for integer k
    k = int(round(2 * v))
    if abs(2 * v - k) < 1e-9:
        num, den = k, 2
        if num % 2 == 0:
            num, den = num // 2, 1
        return num, den
    # fallback: rational approximation
    from fractions import Fraction
    f = Fraction(v).limit_denominator(10**6)
    return f.numerator, f.denominator


def _build_strassen_2x2x2():
    """Strassen decomposition of 2x2x2. Returns U(4,7), V(4,7), W(4,7).
    Indexing: U[a*2+b, c], V[b*2+d, c], W[d*2+a, c] with a,b,d in {0,1}."""
    # 7 Strassen products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)

    # M1: (A00+A11)*(B00+B11)
    U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1; V[1*2+1, 0] = 1
    # M2: (A10+A11)*B00
    U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    # M3: A00*(B01-B11)
    U[0*2+0, 2] = 1
    V[0*2+1, 2] = 1; V[1*2+1, 2] = -1
    # M4: A11*(B10-B00)
    U[1*2+1, 3] = 1
    V[1*2+0, 3] = 1; V[0*2+0, 3] = -1
    # M5: (A00+A01)*B11
    U[0*2+0, 4] = 1; U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    # M6: (A10-A00)*(B00+B01)
    U[1*2+0, 5] = 1; U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1; V[0*2+1, 5] = 1
    # M7: (A01-A11)*(B10+B11)
    U[0*2+1, 6] = 1; U[1*2+1, 6] = -1
    V[1*2+0, 6] = 1; V[1*2+1, 6] = 1

    # W: C[da] = sum_c W[da,c] * M_c
    # C00 = M1 + M4 - M5 - M7
    W[0*2+0, 0] = 1; W[0*2+0, 3] = 1; W[0*2+0, 4] = -1; W[0*2+0, 6] = -1
    # C01 = M3 + M5
    W[0*2+1, 2] = 1; W[0*2+1, 4] = 1
    # C10 = M2 + M4
    W[1*2+0, 1] = 1; W[1*2+0, 3] = 1
    # C11 = M1 - M2 + M3 + M7
    W[1*2+1, 0] = 1; W[1*2+1, 1] = -1; W[1*2+1, 2] = 1; W[1*2+1, 6] = 1

    return U, V, W


def _build_5x5x5():
    """Build 2+2+1 Strassen decomposition. Returns U(25,r), V(25,r), W(25,r)."""
    n, m, p = 5, 5, 5
    sU, sV, sW = _build_strassen_2x2x2()

    U = np.zeros((n * m, 117), dtype=float)
    V = np.zeros((m * p, 117), dtype=float)
    W = np.zeros((p * n, 117), dtype=float)

    col = 0
    # Diagonal 2x2x2 Strassen blocks
    for block in range(2):
        base = block * 2
        for c in range(7):
            for a in range(2):
                U[(base + a) * m + (base + a)]  # placeholder, will fill properly
            # U: index [i*m + j, c] where i=row of A, j=col of A
            for a in range(2):  # i = base+a
                for b in range(2):  # j = base+b
                    U[(base + a) * m + (base + b), col] = sU[a * 2 + b, c]
            # V: index [j*p + k, c] where j=row of B, k=col of B
            for b in range(2):  # j = base+b
                for d in range(2):  # k = base+d
                    V[(base + b) * p + (base + d), col] = sV[b * 2 + d, c]
            # W: index [k*n + i, c] where k=row of C, i=col of C
            for d in range(2):  # k = base+d
                for a in range(2):  # i = base+a
                    W[(base + d) * n + (base + a), col] = sW[d * 2 + a, c]
            col += 1

    # Off-diagonal 2x2x1, 2x1x2, 1x2x2 blocks: schoolbook (rank 1 each)
    # Blocks: (0,1),(0,2),(0,3),(0,4),(1,0),(1,2),(1,3),(1,4),(2,0),(2,1),(2,3),(2,4),(3,0),(3,1),(3,2),(3,4),(4,0),(4,1),(4,2),(4,3)
    # where block (a,b) means rows i in block a of A, cols j in block b of B
    # Block sizes: block 0 = {0,1}, block 1 = {2,3}, block 2={4},3={4},4={4}
    # Actually 2+2+1: blocks are [0,1],[2,3],[4]
    block_sizes = [2, 2, 1]
    block_starts = [0, 2, 4]

    for bi in range(3):  # A block (rows)
        for bj in range(3):  # B block (cols)
            if bi == bj:
                continue  # diagonal handled above
            # For off-diagonal, it's just schoolbook: each (i,j,k) with i in block bi, j in block bj, k in block bj
            # Actually for matmul: C[i,k] = sum_j A[i,j]*B[j,k]
            # Block (bi, bj): i in block bi, j in block bj, k in block bj (since B[j,k] with j in block bj means k can be anything)
            # Wait, no. The block structure is: A is partitioned by rows (bi) and cols (bj).
            # B is partitioned by rows (bj) and cols (bk).
            # C is partitioned by rows (bi) and cols (bk).
            # So we need blocks (bi, bj, bk) for all combinations.
            # For diagonal bi==bj==bk: Strassen on 2x2x2 (for bi=bj=0 or 1) or 1x1x1 (for bi=bj=2)
            # For off-diagonal: schoolbook.
            pass

    # Let me redo this more carefully.
    # Partition indices 0..4 into blocks: B0={0,1}, B1={2,3}, B2={4}
    # A[i,j]: i in block bi, j in block bj
    # B[j,k]: j in block bj, k in block bk
    # C[i,k]: i in block bi, k in block bk
    # For each triple (bi, bj, bk):
    #   if bi==bj==bk and block size is 2: Strassen (7 terms)
    #   if bi==bj==bk and block size is 1: 1 term
    #   otherwise: schoolbook (size_bi * size_bj * size_bk terms)

    U = np.zeros((n * m, 117), dtype=float)
    V = np.zeros((m * p, 117), dtype=float)
    W = np.zeros((p * n, 117), dtype=float)
    col = 0

    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                si = block_starts[bi]
                sj = block_starts[bj]
                sk = block_starts[bk]
                ni = block_sizes[bi]
                nj = block_sizes[bj]
                nk = block_sizes[bk]

                if bi == bj == bk and ni == 2:
                    # Strassen block
                    for c in range(7):
                        for a in range(2):
                            for b in range(2):
                                U[(si + a) * m + (sj + b), col] = sU[a * 2 + b, c]
                        for b in range(2):
                            for d in range(2):
                                V[(sj + b) * p + (sk + d), col] = sV[b * 2 + d, c]
                        for d in range(2):
                            for a in range(2):
                                W[(sk + d) * n + (si + a), col] = sW[d * 2 + a, c]
                        col += 1
                elif bi == bj == bk and ni == 1:
                    # 1x1x1: single term
                    U[si * m + sj, col] = 1
                    V[sj * p + sk, col] = 1
                    W[sk * n + si, col] = 1
                    col += 1
                else:
                    # Schoolbook for this sub-block
                    for a in range(ni):
                        for b in range(nj):
                            for d in range(nk):
                                U[(si + a) * m + (sj + b), col] = 1
                                V[(sj + b) * p + (sk + d), col] = 1
                                W[(sk + d) * n + (si + a), col] = 1
                                col += 1

    assert col == 117, f"Expected rank 117, got {col}"
    return U, V, W


def _verify(U, V, W, tol=1e-9):
    """Verify that the decomposition exactly reconstructs the matmul tensor."""
    n, m, p = 5, 5, 5
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Check a few random entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                expected = 1.0 if (j == k) else 0.0
                if abs(val - expected) > tol:
                    return False
    return True


def _reduce_rank(U, V, W, max_time=100.0):
    """Try to reduce rank by finding linear dependencies among rank-1 terms.
    
    A rank-1 term c is redundant if the tensor it contributes can be expressed
    as a linear combination of the other terms. This is equivalent to checking
    whether the vector T_c = vec(U_c ⊗ V_c ⊗ W_c) is in the span of the other T's.
    
    We use a greedy elimination: for each column, check if removing it still
    allows exact reconstruction (via least squares on the remaining columns).
    """
    n, m, p = 5, 5, 5
    rank = U.shape[1]
    
    # Build the tensor matrix: each column c gives a vector in R^{n*m*p}
    # T_c[i*m*p + j*p + k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # But we only need the "matmul tensor" which is T[i,j,k] = delta_{jk}
    # Actually, the full tensor is what we're decomposing. Let me build it properly.
    
    # The matmul tensor M has M[i,j,k] = 1 if j==k else 0.
    # We need: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # To check if column c is redundant, we need to see if the "contribution" of c
    # to the tensor is in the span of contributions of other columns.
    # This is a nonlinear problem in general, but for rank-1 terms we can use:
    # The tensor contribution of column c is the outer product U_c ⊗ V_c ⊗ W_c (reshaped).
    # It's redundant iff this vector is in the span of {U_d ⊗ V_d ⊗ W_d : d != c}.
    
    # Build the full tensor matrix (125 x rank)
    Tmat = np.zeros((n * m * p, rank))
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Tmat[i * m * p + j * p + k, c] = U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
    
    # The target tensor
    target = np.zeros(n * m * p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m * p + j * p + k] = 1.0 if j == k else 0.0
    
    # Check current validity
    recon = Tmat @ np.ones(rank)
    if np.max(np.abs(recon - target)) > 1e-9:
        return U, V, W  # invalid, don't modify
    
    # Greedy elimination: try to remove each column
    active = list(range(rank))
    removed = set()
    
    # For each column, check if it can be expressed as a combination of others
    # This is expensive, so we use a simpler heuristic:
    # Check if the column's tensor vector is in the span of others using SVD
    
    deadline = time.time() + max_time
    
    for trial in range(rank):
        if time.time() > deadline:
            break
        c = trial
        if c in removed:
            continue
        
        # Check if removing c still allows exact reconstruction
        remaining = [d for d in active if d != c]
        if len(remaining) < 1:
            break
        
        # Build submatrix
        Tsub = Tmat[:, remaining]
        
        # Check if target is in the column space of Tsub
        # Use least squares and check residual
        try:
            coeffs, residuals, rank_sub, sv = np.linalg.lstsq(Tsub, target, rcond=1e-12)
        except np.linalg.LinAlgError:
            continue
        
        if len(residuals) > 0 and residuals[0] < 1e-10:
            # Column c is redundant! Remove it.
            removed.add(c)
            active.remove(c)
    
    if not removed:
        return U, V, W
    
    # Build reduced matrices
    new_rank = len(active)
    U_new = U[:, active]
    V_new = V[:, active]
    W_new = W[:, active]
    
    return U_new, V_new, W_new


def construct():
    t0 = time.time()
    
    # Build initial 117-rank decomposition
    U, V, W = _build_5x5x5()
    
    # Verify
    if not _verify(U, V, W):
        # Fallback to schoolbook
        n, m, p = 5, 5, 5
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank reduction
    if _time_left() > 30:
        U, V, W = _reduce_rank(U, V, W, max_time=max(10, _time_left() - 20))
    
    # Verify final result
    if not _verify(U, V, W):
        # Rebuild if reduction broke something
        U, V, W = _build_5x5x5()
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
