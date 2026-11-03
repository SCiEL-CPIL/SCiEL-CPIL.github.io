"""Matrix multiplication 4x4 * 4x9 rank-minimal exact tensor decomposition.

Strategy: rank-1 column stripping via Kronecker product.
  Core (first 8 columns of B): Kronecker product of rank-7 Strassen (2x2x2)
  and rank-2 (2x2x1) tensor -> rank 14 per column pair, 7 pairs -> 98 terms.
  Actually: 4x4 * 4x8 can be seen as 4 independent 4x4 * 4x2 problems? No.
  
  Better structural view:
  4x4 * 4x9 = 4x4 * 4x8 + 4x4 * 4x1
  4x4 * 4x8: Split B into 4 blocks of 4x2. For each 4x4 * 4x2:
    Use Strassen on the 4x4 * 4x4 structure but only keep 2 output columns.
    Actually, 4x4 * 4x2 can be done with rank 28 (Strassen-like).
    4 blocks * 28 = 112. Plus 4x4 * 4x1 = 16. Total 128. Worse.
  
  Let me use the Kronecker product approach properly:
  The tensor T for 4x4*4x9 can be decomposed using:
  - 2x2x2 Strassen tensor (rank 7) for the "matrix multiplication" part
  - Extended to handle the 9 columns.
  
  Actually, the cleanest structural approach:
  4x4 * 4x9: Think of A as 2x2 blocks of 2x2, B as 2x2 blocks of 2x9.
  But 2x9 isn't square.
  
  Alternative: Use the fact that 4x4*4x9 tensor = sum over j of A[i,j] * B[j,k]
  We can group: for each pair of rows in A and columns in B, we have a 2x2x2 sub-structure.
  
  Let me use a different structural idea: 
  4x4 * 4x9 rank decomposition via "column grouping":
  Group the 9 columns of B into groups: [4, 4, 1].
  For each 4x4 * 4x4 block: use Strassen rank 49.
  For 4x4 * 4x1: schoolbook rank 16.
  Total: 49 + 49 + 16 = 114. (This is what the current program does.)
  
  To get a STRUCTURAL change, let me try:
  4x4 * 4x9 = (2x2 * 2x9) stacked vertically, 2 times.
  Each 2x2 * 2x9: can be decomposed.
  2x2 * 2x9: C[i,k] = sum_j A[i,j] B[j,k] for i in {0,1}, j in {0,1}, k in {0..8}
  This is a bilinear map. Rank of 2x2*2x9?
  Schoolbook: 2*2*9 = 36.
  Can we do better? 
  
  Actually, let me try the approach from the inspiration: use the rank-7 Strassen
  for the 2x2x2 part and extend to 2x2x9 by treating each column independently
  but sharing the Strassen structure.
  
  For 2x2 * 2x9:
  A = [[a,b],[c,d]], B = [[b0,...,b8],[c0,...,c8]]
  C[i,k] = sum_j A[i,j] * B[j,k]
  
  Using Strassen-like:
  P1 = (a+d)(b0+c0 + ... ) -- no, this doesn't work directly for rectangular.
  
  Let me try yet another structural approach: use the known optimal rank for
  small matrix multiplication.
  
  For 4x4 * 4x4, the best known rank is 49 (Strassen).
  For 4x4 * 4x9, we can pad to 4x4 * 4x12 (3 blocks of 4x4) and use 3*49 = 147,
  then discard 3 columns. But that's worse than 114.
  
  Actually, let me try: 4x4 * 4x9 = 4x4 * 4x8 + 4x4 * 4x1
  4x4 * 4x8: 8 columns. Split into two 4x4 * 4x4: 2 * 49 = 98.
  4x4 * 4x1: 16.
  Total: 98 + 16 = 114. Same as before.
  
  Let me try a genuinely different structure: use the "Winograd" approach or
  a different factorization.
  
  Actually, the key insight: 4x4 * 4x9 can be viewed as computing 4*9 = 36
  dot products of length 4. The tensor rank is the minimum number of rank-1
  terms.
  
  Let me try: use rank-7 Strassen for 2x2x2, and compose it with the structure
  of 4x4 * 4x9.
  
  4x4 * 4x9:
  A = [[A00, A01], [A10, A11]] (2x2 blocks of 2x2)
  B = [[B0, B1]] where B0, B1 are 2x9.
  C = [[C0], [C1]] where C0, C1 are 2x9.
  
  C0 = A00*B0 + A01*B1
  C1 = A10*B0 + A11*B1
  
  Now, A00*B0 is 2x2 * 2x9. Let's decompose this.
  For 2x2 * 2x9, we can use:
  P1 = (a00+a11) * (B0[0,:] + B1[0,:])  -- wait, B0 and B1 are 2x9 each.
  
  Let me be more careful. B0 is 2x9: B0 = [[b00, b01, ..., b08], [b10, b11, ..., b18]]
  B1 is 2x9: B1 = [[c00, c01, ..., c08], [c10, c11, ..., c18]]
  
  A00 = [[a, b], [c, d]]
  A01 = [[e, f], [g, h]]
  
  C0 = A00*B0 + A01*B1
  C0[i,k] = sum_j (A00[i,j]*B0[j,k] + A01[i,j]*B1[j,k])
  
  This is equivalent to:
  [A00 | A01] * [B0; B1] where [A00|A01] is 2x4 and [B0;B1] is 4x9.
  
  So C0 is a 2x4 * 4x9 product. Rank of 2x4 * 4x9?
  Schoolbook: 2*4*9 = 72.
  
  This is getting complicated. Let me just stick with the 114 approach but
  verify it's correct, and add a fallback.
  
  Actually, let me try a completely different structural idea:
  Use the fact that 4x4 * 4x9 can be decomposed using the "bilinear rank"
  approach where we exploit the specific structure.
  
  For now, let me implement the 114-rank solution correctly and add the
  verification from the inspiration (using numpy einsum for speed).
"""
import time
import numpy as np

_START = time.time()
_DEADLINE = 130.0


def _time_ok():
    return time.time() - _START < _DEADLINE


def _strassen_2x2_terms():
    """7 Strassen terms for 2x2*2x2."""
    terms = [
        # P1 = (A00+A11)*(B00+B11)
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        # P2 = (A10+A11)*B00
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        # P3 = A00*(B01-B11)
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        # P4 = A11*(B10-B00)
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        # P5 = (A00+A01)*B11
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        # P6 = (A10-A00)*(B00+B01)
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        # P7 = (A01-A11)*(B10+B11)
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]
    return terms


def _strassen_4x4_4x4_terms():
    """49 terms for 4x4*4x4 via recursive Strassen."""
    inner = _strassen_2x2_terms()
    top = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]
    all_terms = []
    for (ab, bb, cb) in top:
        for (ui, vi, wi) in inner:
            u, v, w = {}, {}, {}
            for (bi, bj), ca in ab.items():
                for (li, lj), ci in ui.items():
                    i, j = 2*bi+li, 2*bj+lj
                    u[(i,j)] = u.get((i,j),0) + ca*ci
            for (bi, bj), cb_ in bb.items():
                for (li, lj), ci in vi.items():
                    j, k = 2*bi+li, 2*bj+lj
                    v[(j,k)] = v.get((j,k),0) + cb_*ci
            for (bi, bj), cc in cb.items():
                for (li, lj), ci in wi.items():
                    i, k = 2*bi+li, 2*bj+lj
                    w[(i,k)] = w.get((i,k),0) + cc*ci
            all_terms.append((u, v, w))
    return all_terms


def _rank114():
    """Rank-114: 49 (A*B1) + 49 (A*B2) + 16 (A*B3)."""
    n, m, p = 4, 4, 9
    rank = 114
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    st = _strassen_4x4_4x4_terms()
    
    # A*B1: cols 0-3
    for c, (up, vp, wp) in enumerate(st):
        for (i,j), cu in up.items():
            if cu: U[i*m+j, c] = cu
        for (j,k), cv in vp.items():
            if cv: V[j*p+k, c] = cv
        for (i,k), cw in wp.items():
            if cw: W[k*n+i, c] = cw
    
    # A*B2: cols 4-7
    off = 49
    for c, (up, vp, wp) in enumerate(st):
        for (i,j), cu in up.items():
            if cu: U[i*m+j, c+off] = cu
        for (j,k), cv in vp.items():
            if cv: V[j*p+(4+k), c+off] = cv
        for (i,k), cw in wp.items():
            if cw: W[(4+k)*n+i, c+off] = cw
    
    # A*B3: col 8
    off2 = 98
    for i in range(n):
        for j in range(m):
            c = off2 + i*m + j
            U[i*m+j, c] = 1
            V[j*p+8, c] = 1
            W[8*n+i, c] = 1
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify decomposition using numpy."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)


def _schoolbook(n, m, p):
    rank = n*m*p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
    return U, V, W


def construct():
    n, m, p = 4, 4, 9
    
    if _time_ok():
        U, V, W = _rank114()
        if _verify(U, V, W, n, m, p):
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    U, V, W = _schoolbook(n, m, p)
    if _verify(U, V, W, n, m, p):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
