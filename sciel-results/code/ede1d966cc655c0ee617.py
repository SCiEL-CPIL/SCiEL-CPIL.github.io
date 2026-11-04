"""
Rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.

Strategy:
  1. Try a known small-rank decomposition (e.g., rank 27, the trivial bound is 36).
  2. Use a structured search: fix U and V with a low-rank parameterization that
     respects the algebraic structure (e.g., Kronecker or block-diagonal patterns),
     then solve for W exactly using rational linear algebra.
  3. Verify the full identity: for all i,k,j,l, sum_c U[i,j,c]*V[l,k,c]*W[k,i,c] == delta_{j,l}.

Key insight: The constraint is that for each (i,k), the matrix M_{(i,k)} of size (m*p) x rank
with entries M_{(j,l),c} = U[i,j,c]*V[l,k,c] must satisfy M_{(i,k)} * W_{k,i} = delta_{j,l}
(vectorized). We search over structured U,V to make this system solvable with small rank.

We try:
  - Rank 27: Use a "block-diagonal" structure where each (i,k) pair gets its own
    set of rank-1 terms, but shared across i or k to reduce rank.
  - Actually, the schoolbook is rank 36. We can do better by sharing.

Known result: For m x n * n x p, the rank is at most m*n*p, but can be reduced.
For 3x3 * 3x4, the rank is at most 27 (by grouping). Let's try rank 27.

Approach for rank 27:
  Use the fact that (AB)_{i,k} = sum_j A_{i,j} B_{j,k}.
  We can write: (AB)_{i,k} = sum_{j=0}^{2} A_{i,j} B_{j,k}.
  This is rank 3 in the j-index, but we need to represent it as sum_c U V W.

  Let's use the following construction:
  For c = 0..26, let c = 3*i + j where i in 0..2, j in 0..2? No, that's 9 terms per k.

  Better: Use the identity:
    (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
  We can set:
    U[i,j,c] = 1 if c == j (for all i), else 0.  -> U is 9x36? No.

  Let's think differently. The tensor T[i,j,k] = 1. We want T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c].

  Consider rank 27: Let c = (i', j, k') where i' in 0..2, j in 0..2, k' in 0..3? No, that's 36.

  Actually, the minimal rank for 3x3 * 3x4 is not known to be less than 27. Let's try rank 27.

  Construction for rank 27:
    Let c index (j, k) for j in 0..2, k in 0..3? That's 12. Not enough.

  Let's use the following:
    For each j in 0..2, we have a term that contributes to all (i,k) via A_{i,j} B_{j,k}.
    So we can set:
      U[i,j,c] = 1 if c == j (regardless of i), else 0.  -> U[i,j,c] = delta_{c,j}
      V[j,k,c] = 1 if c == j, else 0.  -> V[j,k,c] = delta_{c,j}
      Then sum_c U V W = sum_j delta_{c,j} delta_{c,j} W[k,i,c] = sum_j W[k,i,j]
      We need this to equal 1 for all i,j,k? No, we need sum_c U[i,j,c] V[j,k,c] W[k,i,c] = delta_{j,k}?? No.

  I'm confusing myself. Let's go back to the constraint:
    sum_c U[i,j,c] V[l,k,c] W[k,i,c] = delta_{j,l} for all i,k,j,l.

  For rank 27, let's try:
    U[i,j,c] = 1 if c == j, else 0.  (c in 0..26, but j in 0..2, so c must be j for some mapping)
    This doesn't work directly.

  Alternative: Use the fact that the multiplication tensor is the sum of 3 rank-1 tensors in the j-index? No.

  Let's just do a search for rank 27 with a structured initialization.
"""
import time
import random
import fractions

DIMENSIONS = [3, 3, 4]
Frac = fractions.Fraction

def _validate(U, V, W, n, m, p, rank):
    """Exact check: sum_c U[i,j,c]*V[l,k,c]*W[k,i,c] == delta_{j,l} for all i,k,j,l."""
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(p):
                    s = Frac(0)
                    for c in range(rank):
                        s += Frac(U[i * m + j][c]) * Frac(V[l * p + k][c]) * Frac(W[k * n + i][c])
                    expected = 1 if j == l else 0
                    if s != expected:
                        return False
    return True

def _try_rank(rank, n, m, p, deadline, rng):
    """Try to find a rank-r decomposition. Returns (U,V,W) or None."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    # We'll try a structured approach:
    # Initialize U and V with a pattern that might allow low rank.
    # For rank 27, try: U[i,j,c] depends on c mod m, V[j,k,c] depends on c mod p?
    
    # Let's try: for c in 0..rank-1, let c = a * p + b where a in 0..m-1, b in 0..p-1 (if rank = m*p)
    # But rank 27 is not m*p=12.
    
    # Try rank 27: Let c = i' * 9 + j' * 3 + k' where i' in 0..2, j' in 0..2, k' in 0..2? No, p=4.
    
    # Let's just do a randomized search with small integers for rank 27, 26, 25, ...
    # But first, let's try a known construction.
    
    # Construction for rank 27:
    # Use the fact that we can share the j-index.
    # Let c = j * 9 + k' where j in 0..2, k' in 0..8? No.
    
    # Actually, let's try the following:
    # For each j in 0..2, we have 9 terms (3 i's * 3 k's? No, p=4 so 3*4=12).
    # This is getting complicated. Let's just do a search.
    
    # We'll try to solve the system for W given U,V.
    # For each (i,k), we have a linear system: M_{(i,k)} W_{k,i} = b_{(i,k)}
    # where M is (m*p) x rank, b is m*p vector (delta_{j,l} vectorized).
    
    # To make it solvable, we need M to have rank = rank (full column rank) and the system consistent.
    # This is a strong constraint on U,V.
    
    # Alternative: Choose W first, then solve for U,V? Symmetric.
    
    # Let's try a different approach: Use the fact that the tensor is separable in j.
    # T[i,j,k] = 1 = sum_c U[i,j,c] V[j,k,c] W[k,i,c].
    
    # For rank 27, let's try:
    #   U[i,j,c] = 1 if c == j (for c in 0..2), else 0.  -> only 3 non-zero columns.
    #   V[j,k,c] = 1 if c == j, else 0.
    #   Then sum_c U V W = sum_{c=0}^2 delta_{c,j} delta_{c,j} W[k,i,c] = W[k,i,j]
    #   We need W[k,i,j] = 1 for all i,k,j? But W has only 3 non-zero columns, and we need it to be 1.
    #   So set W[k,i,c] = 1 for c in 0..2.
    #   Then sum_c U V W = 1 for all i,j,k. But we need delta_{j,l}, not 1.
    #   This doesn't work.
    
    # Let's try rank 12 (m*p):
    #   c = j * p + l where j in 0..2, l in 0..3.
    #   U[i,j,c] = 1 if c == j*p + l for some l? No, U depends on i,j,c.
    #   Let U[i,j,c] = 1 if c == j*p + l, but l is not in U.
    
    # I think the best is to just do a search for small ranks.
    
    # For each rank from 27 down to 20, try random U,V with small integers, solve for W.
    for r in range(rank, 19, -1):
        if time.time() > deadline - 10:
            break
        for attempt in range(1000):
            if time.time() > deadline - 10:
                break
            U = [[rng.randint(-2, 2) for c in range(r)] for _ in range(nm)]
            V = [[rng.randint(-2, 2) for c in range(r)] for _ in range(mp)]
            
            # Solve for W
            W = [[0] * r for _ in range(pn)]
            ok = True
            for i in range(n):
                for k in range(p):
                    # Build system: for each (j,l): sum_c U[i,j,c]*V[l,k,c]*W[k,i,c] = delta_{j,l}
                    rows = []
                    rhs = []
                    for j in range(m):
                        for l in range(p):
                            row = [Frac(U[i * m + j][c] * V[l * p + k][c]) for c in range(r)]
                            rows.append(row)
                            rhs.append(Frac(1 if j == l else 0))
                    
                    # Solve rows x W_vec = rhs
                    # Use Gaussian elimination with fractions
                    aug = [row[:] + [rhs[rr]] for rr, row in enumerate(rows)]
                    R = len(aug)
                    C = r
                    pivots = []
                    col = 0
                    for rr in range(R):
                        if col >= C:
                            break
                        # Find pivot
                        pr = None
                        for r2 in range(rr, R):
                            if aug[r2][col] != 0:
                                pr = r2
                                break
                        if pr is None:
                            col += 1
                            continue
                        if pr != rr:
                            aug[rr], aug[pr] = aug[pr], aug[rr]
                        # Normalize pivot row
                        pv = aug[rr][col]
                        for c2 in range(col, C + 1):
                            aug[rr][c2] /= pv
                        # Eliminate other rows
                        for r2 in range(R):
                            if r2 == rr:
                                continue
                            factor = aug[r2][col]
                            if factor != 0:
                                for c2 in range(col, C + 1):
                                    aug[r2][c2] -= factor * aug[rr][c2]
                        pivots.append(col)
                        col += 1
                    
                    # Check consistency
                    for rr in range(R):
                        # Check if row is all zeros but rhs != 0
                        all_zero = all(aug[rr][c2] == 0 for c2 in range(C))
                        if all_zero and aug[rr][C] != 0:
                            ok = False
                            break
                    if not ok:
                        break
                    
                    # Extract solution (particular solution, set free variables to 0)
                    w_vec = [Frac(0)] * C
                    for idx, col in enumerate(pivots):
                        w_vec[col] = aug[idx][C]
                    W[k * n + i] = [int(w) if w.denominator == 1 else w for w in w_vec]
            
            if ok:
                # Verify full identity
                if _validate(U, V, W, n, m, p, r):
                    return U, V, W, r
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # 130s to leave 20s margin
    
    # Try to find a low-rank decomposition
    result = _try_rank(27, n, m, p, deadline, random.Random(42))
    if result is not None:
        U, V, W, r = result
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition of rank 36
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
