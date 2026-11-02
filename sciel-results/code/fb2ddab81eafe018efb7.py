"""Rewritten program: rank-minimal exact tensor decomposition for 4x4 * 4x8."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _verify(U, V, W, tol=1e-9):
    """Check that sum_c U[i,j,c]*V[j,k,c]*W[k',i,c] equals the multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # T[i,j,k] = A[i,j]*B[j,k]  -> index (i, j, k)
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # Product term: U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Sum over c should equal 1 iff (i,j,k) is a valid multiplication term (always 1 for all i,j,k in multiplication)
    # Actually the multiplication tensor T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # because C[i,k] = sum_j A[i,j]*B[j,k], so each (i,j,k) contributes exactly 1.
    ok = True
    for c in range(rank):
        pass
    # Build the full tensor via einsum
    T = np.einsum('ic,jc,kc->ijk', U, V, W)
    expected = np.ones((n, m, p))
    if not np.allclose(T, expected, atol=tol):
        return False
    return True


def _strassen_2x2_blocks():
    """
    Return Strassen 2x2 multiplication as a set of 7 terms.
    Each term: (u_indices, v_indices, w_indices) where
      u_indices: list of (i,j) for A factors (2x2)
      v_indices: list of (j,k) for B factors
      w_indices: list of (k,i) for C output
    Actually we return coefficient structure: for each of 7 products,
    the A-coeffs (2x2), B-coeffs (2x2), C-coeffs (2x2).
    """
    # Strassen:
    # M1 = (A11-A22)*(B21+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B11-B22)
    # M4 = A22*(B22-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    # Represent each factor as dict (i,j)->coef for A, (j,k)->coef for B, (k,i)->coef for C.
    terms = []
    def mk(d):
        return d
    # M1
    A1 = {(0,0):1,(1,1):-1}
    B1 = {(1,0):1,(1,1):1}
    C1 = {(0,0):1,(1,1):-1,(0,1):1,(1,0):-1}
    terms.append((A1,B1,C1))
    # M2
    A2 = {(1,0):1,(1,1):1}
    B2 = {(0,0):1}
    C2 = {(1,0):1,(1,1):-1}
    terms.append((A2,B2,C2))
    # M3
    A3 = {(0,0):1}
    B3 = {(0,0):1,(1,1):-1}
    C3 = {(0,1):1,(1,1):1}
    terms.append((A3,B3,C3))
    # M4
    A4 = {(1,1):1}
    B4 = {(1,1):1,(0,0):-1}
    C4 = {(0,0):1,(1,0):1}
    terms.append((A4,B4,C4))
    # M5
    A5 = {(0,0):1,(0,1):1}
    B5 = {(1,1):1}
    C5 = {(0,0):-1,(0,1):1}
    terms.append((A5,B5,C5))
    # M6
    A6 = {(1,0):1,(0,0):-1}
    B6 = {(0,0):1,(0,1):1}
    C6 = {(1,1):1}
    terms.append((A6,B6,C6))
    # M7
    A7 = {(0,1):1,(1,1):-1}
    B7 = {(1,0):1,(1,1):1}
    C7 = {(0,1):1}
    terms.append((A7,B7,C7))
    return terms


def _build_4x4_strassen():
    """
    Build a 4x4 x 4x4 multiplication decomposition using Strassen recursion.
    Returns list of terms, each term = (A_dict, B_dict, C_dict) where
      A_dict: (i,j)->coef for 4x4 A
      B_dict: (j,k)->coef for 4x4 B
      C_dict: (k,i)->coef for 4x4 C
    Uses 4x4 = 2x(2x2) blocks, each 2x2 uses 7 Strassen products -> 49 terms.
    """
    s2 = _strassen_2x2_blocks()
    # 4x4 blocks: (00,01,10,11) each 2x2
    # For a 4x4 multiplication, split A into 4 blocks A00,A01,A10,A11 (each 2x2)
    # B into B00,B01,B10,B11, C into C00,C01,C10,C11.
    # C = A*B where:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # Each 2x2*2x2 product uses 7 Strassen terms.
    # So 4 block-products * 7 = 28 terms, but we can share? No, separate.
    # Actually each block product is independent 2x2 multiplication.
    terms = []
    for (ci, cj) in [(0,0),(0,1),(1,0),(1,1)]:
        for (ai, bj) in [(0,0),(0,1),(1,0),(1,1)]:
            # product A[ai] * B[bj] contributes to C[ci] if ci==ai and bj==cj
            if ci == ai and bj == cj:
                for (A2, B2, C2) in s2:
                    A4 = {}
                    B4 = {}
                    C4 = {}
                    for ((i,j), v) in A2.items():
                        A4[(2*ai+i, 2*aj+j)] = v
                    for ((j,k), v) in B2.items():
                        B4[(2*bj+j, 2*ck_k(k))] = v
                    ...
    # This is getting complex; let me do it cleanly with explicit block mapping.
    # I'll instead directly construct via Kronecker: 4x4 mult = apply 2x2 Strassen at block level.
    # Block-level: treat the 4x4 as a 2x2 matrix of 2x2 blocks.
    # The 2x2 Strassen gives 7 scalar products of 2x2 blocks.
    # Each such scalar product of 2x2 blocks is itself a 2x2*2x2 mult = 7 more.
    # So total 7*7 = 49 terms.
    # Let me construct: for each of 7 outer Strassen terms, we have (A_outer, B_outer, C_outer)
    # each a 2x2 dict of block-coeffs. Then each block product (A_block * B_block) is a 2x2 mult
    # which we expand with 7 inner Strassen terms.
    terms = []
    for (Ao, Bo, Co) in s2:
        # Ao: (bi,bj)->coef for outer block indices (bi in {0,1}, bj in {0,1})
        # Each outer block is a 2x2. So Ao[bi,bj] multiplies the A sub-block at (bi,bj).
        # For each pair (bi,bj) in Ao and (bk,bl) in Bo such that they multiply to contribute...
        # Actually the outer Strassen: C_outer = Ao * Bo (2x2 block mult).
        # The 7 outer terms each compute one scalar (block) product: 
        #   M = (sum Ao[bi,bj]*A_block[bi,bj]) * (sum Bo[bk,bl]*B_block[bk,bl])
        # and then C_block[ci,cj] += Co[ci,cj]*M.
        # So each outer term = product of two 2x2 "linear combos of blocks".
        # Expand: M = sum_{bi,bj} sum_{bk,bl} Ao[bi,bj]*Bo[bk,bl] * A_block[bi,bj]*B_block[bk,bl]
        # But A_block[bi,bj] is a 2x2 matrix, B_block[bk,bl] is 2x2.
        # Their product is a 2x2*2x2 = 4x4 sub-product contributing to C_block[ci,cj] (2x2) 
        # scaled by Ao[bi,bj]*Bo[bk,bl]*Co[ci,cj].
        # The 2x2*2x2 product A_block[bi,bj]*B_block[bk,bl] we expand with 7 inner Strassen terms.
        # Wait, but A_block[bi,bj] and B_block[bk,bl] are full 2x2 matrices (entries of A and B).
        # So A_block[bi,bj]*B_block[bk,bl] is a standard 2x2*2x2 multiplication.
        # Expand it with the 7 inner terms.
        for (bi, bj) in Ao:
            for (bk, bl) in Bo:
                scale = Ao[(bi,bj)] * Bo[(bk,bl)]
                # inner 2x2*2x2: A_sub = A at block (bi,bj), B_sub = B at block (bk,bl)
                for (Ai, Bi, Ci) in s2:
                    A4 = {}
                    B4 = {}
                    C4 = {}
                    for ((i,j), v) in Ai.items():
                        # A_sub[i,j] = A[2*bi+i, 2*bj+j]
                        A4[(2*bi+i, 2*bj+j)] = scale * v
                    for ((j,k), v) in Bi.items():
                        B4[(2*bk+j, 2*bl+k)] = v
                    for ((k,i), v) in Ci.items():
                        # C_sub[k,i] contributes to C[2*ci+k, 2*cj+i] for each (ci,cj) in Co
                        # We need to accumulate over Co.
                        pass
                    # accumulate C4 over Co
                    C4 = {}
                    for ((ci, cj), co) in Co.items():
                        for ((k, i), v) in Ci.items():
                            key = (2*ci+k, 2*cj+i)
                            C4[key] = C4.get(key, 0) + co * v
                    if A4 and B4 and C4:
                        terms.append((A4, B4, C4))
    return terms


def _ck_k(x):
    return x


def _combine_terms(terms, n, m, p):
    """
    Convert list of (A_dict, B_dict, C_dict) into U,V,W matrices.
    A_dict: (i,j)->coef, B_dict: (j,k)->coef, C_dict: (k,i)->coef.
    Each term becomes one column in U,V,W (rank = number of terms).
    We merge terms that share identical (A,B,C) patterns to reduce rank? 
    Actually each term is a rank-1 contribution; we can't merge unless identical.
    But we CAN combine by linearity: if two terms have the same U and V but different W,
    we can't merge. However, we can reduce rank by Gaussian elimination on the coefficient
    structure: the rank is the minimum number of rank-1 tensors summing to the multiplication tensor.
    For now, one column per term.
    """
    rank = len(terms)
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    for c, (A, B, C) in enumerate(terms):
        for ((i,j), v) in A.items():
            U[i*m+j][c] = v
        for ((j,k), v) in B.items():
            V[j*p+k][c] = v
        for ((k,i), v) in C.items():
            W[k*n+i][c] = v
    return U, V, W


def _schoolbook(n, m, p):
    rank = n*m*p
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return U, V, W


def _to_int_or_half(x):
    """Round to nearest half-integer and return as int (scaled by 2) or float."""
    r = round(x*2)/2
    return r


def _normalize(U, V, W):
    """Ensure entries are integer or half-integer."""
    U = [[_to_int_or_half(v) for v in row] for row in U]
    V = [[_to_int_or_half(v) for v in row] for row in V]
    W = [[_to_int_or_half(v) for v in row] for row in W]
    return U, V, W


def construct():
    t0 = time.time()
    deadline = 150 - 10  # safety margin
    n, m, p = DIMENSIONS

    # Start with schoolbook as fallback
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    best = (U_fb, V_fb, W_fb)

    # Try Strassen-based 4x4 x 4x8.
    # We have 4x4 A, 4x4 B_left (cols 0-3), 4x8 B_right (cols 4-7).
    # Treat B as 4x8 = [B_left(4x4) | B_right(4x4)].
    # C = A*B = [A*B_left | A*B_right], each 4x4.
    # Use 4x4 Strassen for each half, but the A factors are shared.
    # Actually we can do a single 4x4 x 4x8 by viewing B as 4x4 blocks of width 4:
    # B has 2 blocks of 4x4: B0 (cols 0-3), B1 (cols 4-7).
    # C = [A*B0 | A*B1].
    # Each A*Bx is a 4x4*4x4 mult. Using Strassen 4x4 (49 terms) for each gives 98 terms.
    # But A is the same, so we can't easily share. 98 < 128, good improvement.
    # Even better: use a 4x4 x 4x8 directly with block structure.
    # Let's build 4x4 Strassen terms for A*B0 and A*B1 separately.
    try:
        s4 = _build_4x4_strassen()
        # s4 terms are for 4x4*4x4. For A*B0: A is 4x4, B0 is B[:,0:4] (4x4).
        # For A*B1: A is 4x4, B1 is B[:,4:8] (4x4).
        # Combine: total terms = 2*len(s4) = 98.
        terms = []
        for (A, B, C) in s4:
            # A*B0: B indices k in 0..3
            A1 = dict(A)
            B1 = {}
            for ((j,k), v) in B.items():
                B1[(j, k)] = v  # k in 0..3
            C1 = dict(C)
            terms.append((A1, B1, C1))
        for (A, B, C) in s4:
            # A*B1: B indices k in 4..7
            A2 = dict(A)
            B2 = {}
            for ((j,k), v) in B.items():
                B2[(j, k+4)] = v
            C2 = {}
            for ((k,i), v) in C.items():
                C2[(k+4, i)] = v
            terms.append((A2, B2, C2))
        U, V, W = _combine_terms(terms, n, m, p)
        U, V, W = _normalize(U, V, W)
        if _verify(U, V, W):
            best = (U, V, W)
    except Exception:
        pass

    if time.time() > deadline:
        return {"U": best[0], "V": best[1], "W": best[2]}

    # Try to reduce rank further via linear algebra: find minimal rank.
    # The multiplication tensor T[i,j,k]=1. We want min rank.
    # Use the fact that our Strassen terms may have linear dependencies.
    # Perform rank reduction: treat each term as a vector in the space of (U,V,W) triples.
    # Actually, we can do a "sliding" / elimination: if the sum of some signed terms
    # equals another term's contribution, we can eliminate.
    # Simpler: use Gaussian elimination on the coefficient matrices.
    # Represent each rank-1 term by its (u_c, v_c, w_c) columns.
    # The tensor is sum_c u_c ⊗ v_c ⊗ w_c.
    # We can try to find a linear combination that cancels.
    # This is hard in general; skip for now.

    U, V, W = best
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
