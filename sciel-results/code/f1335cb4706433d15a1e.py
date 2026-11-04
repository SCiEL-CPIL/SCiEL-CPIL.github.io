"""Rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _mul_tensor_support():
    """Return the support S = {(i,j,k)} and coefficient 1 for the 2x3*3x6 product tensor.
    A is 2x3, B is 3x6. C[i,k] = sum_j A[i,j] B[j,k].
    The tensor M[i,j,k] has coefficient 1 for every triple (i,j,k) with i in [0,2),
    j in [0,3), k in [0,6) because A[i,j] and B[j,k] are both nonzero free variables,
    so every (i,j,k) appears with coefficient 1.
    """
    n, m, p = DIMENSIONS
    S = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                S.append((i, j, k))
    return S


def _build_strassen_blocks():
    """Construct a block decomposition using 2x2 Strassen (rank 7) for the
    2x2 x 2x2 core and handle the extra column/rows. Returns (U,V,W) lists.

    Layout:
      A (2x3) = [Aa Ab Ac]  where Aa,Ab,Ac are 2x1 columns.
      B (3x6) = rows b0,b1,b2 each 1x6.
      C = A B, C[i,k] = Aa[i] b0[k] + Ab[i] b1[k] + Ac[i] b2[k].

    We group columns: C1 = Aa b0 + Ab b1 (2x6, uses 2 columns of A),
    C2 = Ac b2 (2x6, uses 1 column of A).

    For C1: Aa,Ab are 2x2 (stack as 2x2 matrix M1 with cols Aa,Ab), and
    the corresponding B part is [b0; b1] which is 2x6. So C1 = M1 * N1 where
    M1 is 2x2, N1 is 2x6.
    C2 = Ac * b2, Ac is 2x1, b2 is 1x6.

    We use Strassen (rank 7) for the 2x2 * 2x2 core of N1's first two columns,
    and direct (rank 4) for the remaining 4 columns of N1, plus rank 2 for C2.
    This is a valid exact decomposition; rank = 7 + 4 + 2 = 13? Let's be careful.
    Actually N1 is 2x6, we can multiply 2x2 * 2x6 by treating N1 as two 2x3 blocks,
    each 2x2*2x3 needs rank 7? No: 2x2 times 2x3: C1 = M1*N1, M1 2x2, N1 2x3.
    The tensor for 2x2*2x3 has size 2*2*3 = 12, schoolbook rank 12, but we can do
    better. 2x2*2x3: think of N1 = [N1a N1b] two 2x2? No, 2x3 = 2 columns of 2x1 plus...
    Simpler: 2x2 * 2x3 = (2x2)*(2x2) for first 2 cols (rank 7) + (2x2)*(2x1) for 3rd col.
    2x2*2x1: C = M1 * v (v 2x1). C[i] = M1[i,0] v[0] + M1[i,1] v[1]. That's 2 outputs,
    each is a bilinear form. Rank of 2x2*2x1 tensor = ? It's 4 entries, schoolbook 4,
    can we do less? C[0]=M00 v0 + M01 v1, C[1]=M10 v0 + M11 v1. This is 2 independent
    bilinear forms; rank at least 2 (need at least 2 monomials since C[0] and C[1] differ).
    Actually rank = 4? Let's just use schoolbook for the small pieces to be safe and
    rely on Strassen for the big 2x2*2x2.

    To keep it simple and correct, I'll use Strassen only where it clearly helps and
    schoolbook elsewhere, then verify.
    """
    n, m, p = DIMENSIONS
    # We'll build factor matrices as dicts keyed by (row, col) -> value, accumulate.
    U = {}
    V = {}
    W = {}
    col = 0

    def add_u(r, v):
        U[(r, col)] = U.get((r, col), 0) + v
    def add_v(r, v):
        V[(r, col)] = V.get((r, col), 0) + v
    def add_w(r, v):
        W[(r, col)] = W.get((r, col), 0) + v

    # ---- Part 1: C1 = M1 * N1, M1 = A[:, 0:2] (2x2), N1 = B[0:2, :] (2x6) ----
    # Split N1 into N1a = N1[:, 0:2] (2x2) and N1b = N1[:, 2:6] (2x4).
    # C1a = M1 * N1a : 2x2 * 2x2 -> Strassen rank 7.
    # C1b = M1 * N1b : 2x2 * 2x4 -> do as two 2x2*2x2? N1b is 2x4 = two 2x2 blocks.
    #   Each 2x2*2x2 is rank 7, so C1b rank 14. Total so far 21. Plus C2.
    # That's worse than schoolbook 36? 21 + C2. C2 = Ac * b2, 2x1 * 1x6:
    #   C2[i,k] = Ac[i] b2[k]. That's rank 2 (two monomials: Ac[0]b2[k] and Ac[1]b2[k]).
    # Total = 21 + 2 = 23. Better than 36.
    # But we can do better on C1b: 2x2 * 2x4. Treat as M1 * [N1b1 N1b2], each 2x2*2x2.
    # Alternatively use the fact that M1 is shared: we can compute the 4 columns of C1b
    # with rank 7 each? No, each column-pair is independent 2x2*2x2.
    #
    # Let's instead do the whole C1 = M1 * N1 (2x2 * 2x6) more cleverly.
    # 2x2 * 2x6 tensor: we can view N1 as 6 columns. The product C1[i,k] =
    #   M1[i,0] N1[0,k] + M1[i,1] N1[1,k]. For fixed k this is a 2-vector = M1 * N1[:,k].
    # The full tensor has rank? We can use Strassen on pairs of columns.
    # Pair columns (0,1): 2x2*2x2 rank 7. Pair (2,3): rank 7. Pair (4,5): rank 7. Total 21.
    # Same as before.
    #
    # Can we share the Strassen intermediates across column pairs? The M1 is the same,
    # but N1 differs per pair, so the 7 Strassen products differ. No sharing.
    #
    # Alternative: use a rank-3 scheme for 2x2? No, 2x2 min rank is 7.
    #
    # Let me reconsider: maybe treat A differently. A is 2x3, B is 3x6.
    # C = A B. Think of it as: for each of the 3 "middle" indices j, we have
    # A[:,j] (2x1) and B[j,:] (1x6), contributing outer product A[:,j] B[j,:].
    # C = sum_j (A[:,j]) (B[j,:]). That's 3 terms, each a 2x6 matrix = rank-? 
    # Each term A[:,j] B[j,:] is a 2x6 matrix of rank 1, but as a *bilinear* form in
    # the entries of A and B, the term for fixed j is: C[i,k] += A[i,j] B[j,k].
    # That's a single monomial per (i,k) for fixed j -> 2*6 = 12 monomials per j,
    # 3 j's -> 36. That's schoolbook.
    #
    # The Strassen win comes from combining the A-rows. Let me just implement the
    # column-pair Strassen (rank 21 + 2 = 23) and verify. Then search for less.

    # Strassen for 2x2 * 2x2:
    # Given P (2x2), Q (2x2): R = P Q.
    # M1 = P00 Q11 - P01 Q10
    # M2 = (P00+P01) Q10
    # M3 = P01 (Q01 - Q11)
    # M4 = P10 (Q00+Q11)
    # M5 = (P10+P11) Q01
    # M6 = (P00 - P10) (Q10+Q11)
    # M7 = (P01+P11) (Q00 - P00)  -> (P01+P11)(Q00-Q11)
    # R00 = M1 + M4 - M5 + M7
    # R01 = M3 + M5
    # R10 = M2 + M4
    # R11 = M1 - M2 + M3 + M6

    def strassen_2x2(P, Q):
        """P, Q: dicts mapping (r,c)->value for 2x2. Returns list of 7 (U-entries, V-entries, W-entries)
        as (rowU, rowV, rowW, value) tuples to accumulate, using a fresh column."""
        global col
        # P[r][c], Q[r][c]
        P00, P01 = P[(0,0)], P[(0,1)]
        P10, P11 = P[(1,0)], P[(1,1)]
        Q00, Q01 = Q[(0,0)], Q[(0,1)]
        Q10, Q11 = Q[(1,0)], Q[(1,1)]
        # M1
        col += 1
        add_u(0, P00); add_v(1, Q11); add_w(0, 1)
        add_u(1, -P01); add_v(1, Q10); add_w(1, 1)
        # wait, M1 = P00*Q11 - P01*Q10 is a product of two *linear forms*, but here
        # P00 and Q11 are single variables (since P,Q are 2x2 with distinct entries).
        # So M1 = P00*Q11 - P01*Q10 = (P00)(Q11) + (-P01)(Q10).
        # As rank-1 tensors: (P00)(Q11)(R00+R11) and (-P01)(Q10)(R00+R11).
        # This is getting complicated to track. Let me instead directly build the
        # 7 Strassen monomials as rank-1 tensors in the (i,j,k) space.
        # For a 2x2*2x2 block occupying A-rows ia,ib and B-cols ka,kb and C-rows/cols...
        # I'll handle the mapping carefully below.
        col -= 1  # revert, I'll redo cleanly
        return []

    # This is getting too intricate to do ad hoc. Let me instead build the whole
    # decomposition at the (i,j,k) tensor level directly using Strassen on the
    # appropriate sub-tensors, then verify. I'll write it more carefully.
    return None


def _verify(U, V, W):
    """Verify that sum_c U[:,c] V[:,c] W[:,c] (as rank-1 tensors) equals the
    product tensor M[i,j,k]=1 for all i,j,k. U shape (n*m, rank), V (m*p, rank),
    W (p*n, rank). The rank-1 tensor for column c is T[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    """
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    # Build tensor M as dict (i,j,k)->sum
    M = {}
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if val != 0:
                        M[(i,j,k)] = M.get((i,j,k), 0) + val
    # Expected: every (i,j,k) has value 1.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if M.get((i,j,k), 0) != 1:
                    return False
    # No extra nonzero entries
    for key, val in M.items():
        if val != 1:
            return False
    return True


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_block_decomposition():
    """Build a block decomposition of the 2x3*3x6 product tensor using Strassen
    on 2x2 sub-blocks. We partition:
      A (2x3) columns: c0, c1, c2.
      B (3x6) rows: r0, r1, r2.
      C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k].

    Group A columns into pairs: (c0,c1) and (c2).
    Group B rows correspondingly: (r0,r1) and (r2).
    Group C columns into pairs of 2: (k0,k1),(k2,k3),(k4,k5).

    Part 1: C1 = A[:,0:2] * B[0:2,:]  (2x2 * 2x6).
      Split B[0:2,:] into 3 column-pairs, each 2x2. So 3 sub-problems of 2x2*2x2.
      Each 2x2*2x2 uses Strassen rank 7. Part 1 rank = 21.
    Part 2: C2 = A[:,2:3] * B[2:3,:]  (2x1 * 1x6).
      C2[i,k] = A[i,2] B[2,k]. This is 2*6 = 12 monomials in schoolbook, but it's
      a rank-2 tensor: C2 = A[:,2] * B[2,:] (outer product of a 2-vector and 6-vector).
      As a bilinear tensor, C2[i,k] = A[i,2] B[2,k]. This is a single rank-1 tensor?
      No: it's the outer product, which in the (i, j, k) tensor with j fixed=2 is
      T[i,2,k] = A[i,2] B[2,k]. That's rank 1 in the (i,k) space but as a 3-tensor
      it's a single monomial per (i,k): 2*6 = 12 monomials? No!
      T[i,2,k] = A[i,2] * B[2,k]. For each (i,k) this is one product of two variables.
      The tensor has support {(i,2,k) : i in 0..1, k in 0..5} with coefficient 1,
      and the value is A[i,2]B[2,k]. This is a rank-1 tensor in the 3D space?
      A rank-1 tensor is u_i v_j w_k. Here we need u_i v_2 w_k with v_2 = 1 and
      u_i = A[i,2], w_k = B[2,k]. But u_i and w_k depend on the *variables*, not
      constants. In a bilinear (actually trilinear) tensor decomposition, the
      coefficients of the rank-1 terms must be constants (scalars), and the
      variables appear in U, V, W linearly.
      So T[i,2,k] = A[i,2] B[2,k] = (A[i,2]) * (1) * (B[2,k])? But the middle
      factor must be a linear form in B's variables. B[2,k] is in W (which is
      indexed by k*n+i, i.e., W depends on B's variables? No.

    I'm confusing myself on which matrix holds which variables. Let me re-read the
    task: U shape (n*m, rank), V (m*p, rank), W (p*n, rank). The product tensor is
    M[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
    Here U is associated with A's variables (indexed by i,j), V with B's variables
    (indexed by j,k), W with C's "output" (indexed by k,i). The decomposition says
    the trilinear form sum_{i,j,k} C[i,k] A[i,j] B[j,k] = sum_c (sum_{i,j} U[i*m+j,c] A[i,j])
    (sum_{j,k} V[j*p+k,c] B[j,k]) (sum_{k,i} W[k*n+i,c] C[i,k]).
    So U,c is a linear form in A, V,c in B, W,c in C. The product of the three linear
    forms equals the trilinear form of matrix multiplication.

    So for C2[i,k] = A[i,2] B[2,k], the trilinear form is C[i,k] A[i,2] B[2,k].
    This is a single rank-1 term: U = A[i,2] (linear form in A), V = B[2,k] (linear
    form in B), W = C[i,k] (linear form in C). But U must be indexed by (i,j) over
    all j, and here it only depends on j=2. So U[i*m+2, c] = 1 (coefficient of A[i,2]),
    U[i*m+j, c] = 0 for j != 2. Similarly V[j*p+k, c] = 1 if j=2, else 0 (coefficient
    of B[2,k]). W[k*n+i, c] = 1 (coefficient of C[i,k]).
    So C2 is a SINGLE rank-1 term! Because the trilinear form C[i,k] A[i,2] B[2,k]
    factors as (sum_i A[i,2] * delta) ... let me check:
    (sum_{i,j} U[i*m+j,c] A[i,j]) = sum_i A[i,2] (since U[i*m+2,c]=1, others 0).
    (sum_{j,k} V[j*p+k,c] B[j,k]) = sum_k B[2,k] (since V[2*p+k,c]=1).
    (sum_{k,i} W[k*n+i,c] C[i,k]) = sum_{k,i} C[i,k] (since W[k*n+i,c]=1).
    Product = (sum_i A[i,2]) (sum_k B[2,k]) (sum_{k,i} C[i,k]).
    That's NOT equal to sum_{i,k} C[i,k] A[i,2] B[2,k]. The product of sums is not
    the sum of products. So C2 is NOT rank 1; it's rank 12 (one monomial per (i,k)).
    Right, because the indices i and k are "summed over" separately in each factor.

    OK so my earlier analysis: each (i,j,k) triple contributes one monomial in
    schoolbook, and the total is 36. Strassen reduces this by combining.

    This is quite involved. Given the complexity and time, let me implement a
    principled exact search using a rank-1 peeling / linear algebra approach with
    rational reconstruction, bounded by a deadline, and fall back to the best
    verified construction (schoolbook or a hand-built Strassen if I can verify one).

    Actually, let me just build the column-pair Strassen decomposition concretely
    at the (i,j,k) level and verify it. I'll be very explicit.
    """
    n, m, p = DIMENSIONS
    # We'll accumulate U,V,W as lists of length rank, each a dict row->value.
    U = []
    V = []
    W = []

    def new_col():
        U.append({})
        V.append({})
        W.append({})
        return len(U) - 1

    def add(c, Urow, Uval, Vrow, Vval, Wrow, Wval):
        U[c][Urow] = U[c].get(Urow, 0) + Uval
        V[c][Vrow] = V[c].get(Vrow, 0) + Vval
        W[c][Wrow] = W[c].get(Wrow, 0) + Wval

    # Strassen 2x2*2x2 where:
    #   A-block rows: ia, ib (global i), A-columns: ja, jb (global j)
    #   B-block rows: ja, jb (global j), B-columns: ka, kb (global k)
    #   C-block rows: ia, ib (global i), C-columns: ka, kb (global k)
    # The 7 Strassen products M1..M7 are linear forms in A and B; each contributes
    # to C entries. We encode each M as one rank-1 term: U = linear form in A,
    # V = linear form in B, W = linear form in C (the combination of C entries).
    def strassen(ia, ib, ja, jb, ka, kb):
        # A entries: A[ia,ja], A[ia,jb], A[ib,ja], A[ib,jb]
        # B entries: B[ja,ka], B[ja,kb], B[jb,ka], B[jb,kb]
        # C entries: C[ia,ka], C[ia,kb], C[ib,ka], C[ib,kb]
        P00 = (ia, ja); P01 = (ia, jb); P10 = (ib, ja); P11 = (ib, jb)
        Q00 = (ja, ka); Q01 = (ja, kb); Q10 = (jb, ka); Q11 = (jb, kb)
        R00 = (ia, ka); R01 = (ia, kb); R10 = (ib, ka); R11 = (ib, kb)

        def Urow(i, j): return i*m + j
        def Vrow(j, k): return j*p + k
        def Wrow(k, i): return k*n + i

        # M1 = P00*Q11 - P01*Q10  -> contributes to R00 and R11 (coeff +1 each)
        c = new_col()
        add(c, Urow(*P00), 1, Vrow(*Q11), 1, Wrow(*R00), 1)
        add(c, Urow(*P00), 0, Vrow(*Q11), 0, Wrow(*R11), 1)  # same U,V; add R11
        # but M1 also has -P01*Q10 term; that's a SEPARATE monomial.
        # Strassen M1 is a *product of two linear forms*: (P00*Q11 - P01*Q10) is not
        # a single product; it's the difference. Actually in Strassen, M1 = (P00)(Q11) - (P01)(Q10)
        # is computed as ONE scalar, but it's the difference of two products, so it
        # requires 2 multiplications in the trilinear form? No -- in the tensor
        # decomposition, each "multiplication" is a rank-1 term. M1 as a bilinear
        # form in (P,Q) is P00*Q11 - P01*Q10, which is a sum of 2 rank-1 bilinear
        # forms. So M1 itself has rank 2 in the (P,Q) space, but it's multiplied by
        # the C-linear form (R00+R11). So the contribution to the trilinear form is
        # (P00*Q11 - P01*Q10)*(R00+R11) = P00*Q11*R00 + P00*Q11*R11 - P01*Q10*R00 - P01*Q10*R11.
        # That's 4 monomials! So Strassen's "7 multiplications" become 7 * (rank of
        # each Mi in P,Q) monomials. Each Mi is a bilinear form of rank up to 2
        # (a difference of 2 products) or 1 (a single product) or 2 (a product of
        # sums).
        # This means Strassen 2x2*2x2 in the *trilinear* (with C) form has rank
        # equal to sum over Mi of (rank of Mi as bilinear in P,Q). Let me compute:
        # M1 = P00 Q11 - P01 Q10 : rank 2 (two monomials)
        # M2 = (P00+P01) Q10 : rank 2 (P00 Q10 + P01 Q10)
        # M3 = P01 (Q01 - Q11) : rank 2 (P01 Q01 - P01 Q11)
        # M4 = P10 (Q00+Q11) : rank 2
        # M5 = (P10+P11) Q01 : rank 2
        # M6 = (P00-P10)(Q10+Q11) : rank 4 (P00Q10+P00Q11-P10Q10-P10Q11)
        # M7 = (P01+P11)(Q00-Q11) : rank 4
        # Total rank = 2+2+2+2+2+4+4 = 18. That's worse than schoolbook 8!
        #
        # AH. This is the key subtlety: Strassen's rank 7 is for the *bilinear*
        # complexity (counting scalar multiplications), where each Mi is one
        # "multiplication" even if it's a product of two linear forms. In the
        # tensor (trilinear with C) decomposition, a product of two linear forms
        # (a,b) where a is linear in P and b is linear in Q is a SINGLE rank-1
        # term in the (P,Q) bilinear tensor! Because (sum P_i a_i)(sum Q_j b_j) is
        # a rank-1 bilinear form (outer product of the vector a and vector b).
        # I was wrong to expand it. A product of a linear form in P and a linear
        # form in Q is rank 1 as a bilinear form. A SUM of such products is a sum
        # of rank-1 forms.
        # So M1 = P00*Q11 - P01*Q10 = (P00)(Q11) + (-P01)(Q10) is a sum of 2 rank-1
        # bilinear forms -> rank 2. But wait, is (P00)(Q11) rank 1? Yes: it's the
        # outer product of the vector (1,0,0,0) in P-space and (0,0,0,1) in Q-space.
        # And M1 = (P00)(Q11) - (P01)(Q10) is a sum of 2 such outer products, so
        # rank 2 (assuming independent).
        # But actually M1 in Strassen is computed as ONE multiplication of the two
        # linear forms (P00*Q11 - P01*Q10)?? No. Strassen computes:
        #   M1 = (P00)(Q11) - (P01)(Q10)  -- this is TWO scalar multiplications
        #   (P00*Q11 and P01*Q10) and one subtraction.
        # Hmm, actually the standard Strassen is:
        #   M1 = (P00 + P11)(Q00 + Q11)
        #   M2 = (P10 + P11) Q00
        #   M3 = P00 (Q01 - Q11)
        #   M4 = P11 (Q10 - Q00)
        #   M5 = (P00 + P01) Q11
        #   M6 = (P10 - P00)(Q00 + Q10)
        #   M7 = (P01 - P11)(Q01 + Q11)
        # Each Mi is ONE product of two linear forms (or a single variable times a
        # linear form). So each Mi is a SINGLE rank-1 bilinear form! Because
        # (linear form in P) * (linear form in Q) = rank-1 bilinear form.
        # So Strassen 2x2*2x2 has bilinear rank 7, and as a trilinear (with C)
        # tensor, each Mi is multiplied by a linear form in C (a combination of
        # R entries), so each Mi contributes ONE rank-1 trilinear term.
        # Total trilinear rank = 7. 
        #
        # I made an error earlier. Let me redo with the correct Strassen formulas
        # where each Mi is a single product of two linear forms.
        return []

    # OK, the correct Strassen (each Mi = one product of two linear forms):
    # M1 = (P00+P11)(Q00+Q11)
    # M2 = (P10+P11) Q00
    # M3 = P00 (Q01-Q11)
    # M4 = P11 (Q10-Q00)
    # M5 = (P00+P01) Q11
    # M6 = (P10-P00)(Q00+Q10)
    # M7 = (P01-P11)(Q01+Q11)
    # R00 = M1 - M4 + M5 - M7
    # R01 = M3 + M5
    # R10 = M2 + M4
    # R11 = M1 - M2 + M3 + M6
    #
    # Each Mi is a product (linP)(linQ). As a trilinear term with C, Mi * (linC)
    # is one rank-1 term. So 7 terms per 2x2*2x2 block.
    #
    # Let me implement this correctly.
    U = []
    V = []
    W = []
    def new_col():
        U.append({}); V.append({}); W.append({})
        return len(U)-1
    def add(c, Ur, Uv, Vr, Vv, Wr, Wv):
        U[c][Ur] = U[c].get(Ur,0)+Uv
        V[c][Vr] = V[c].get(Vr,0)+Vv
        W[c][Wr] = W[c].get(Wr,0)+Wv
    def Urow(i,j): return i*m+j
    def Vrow(j,k): return j*p+k
    def Wrow(k,i): return k*n+i

    def strassen_block(ia, ib, ja, jb, ka, kb):
        # P00=A[ia,ja], P01=A[ia,jb], P10=A[ib,ja], P11=A[ib,jb]
        # Q00=B[ja,ka], Q01=B[ja,kb], Q10=B[jb,ka], Q11=B[jb,kb]
        # R00=C[ia,ka], R01=C[ia,kb], R10=C[ib,ka], R11=C[ib,kb]
        # M1 = (P00+P11)(Q00+Q11): linP = P00+P11, linQ = Q00+Q11
        c = new_col()
        add(c, Urow(ia,ja),1, Vrow(ja,ka),1, Wrow(ka,ia), 1)   # R00
        add(c, Urow(ia,ja),0, Vrow(ja,ka),0, Wrow(kb,ia), 1)   # R01? no, M1*R01? 
        # M1 contributes to R00 and R11 (from R00=M1-M4+M5-M7, R11=M1-M2+M3+M6).
        # So M1's C-linear form is R00 + R11.
        add(c, Urow(ia,ja),0, Vrow(ja,ka),0, Wrow(ka,ib), 1)   # R10? no.
        # R11 = C[ib,kb] -> Wrow(kb, ib).
        add(c, Urow(ia,ja),0, Vrow(ja,ka),0, Wrow(kb,ib), 1)   # R11
        # Now add the P11 part of linP and Q11 part of linQ:
        add(c, Urow(ib,jb),1, Vrow(ja,ka),0, Wrow(ka,ia), 0)   # P11*Q00 -> but M1=(P00+P11)(Q00+Q11)
        # This is getting unwieldy because (P00+P11)(Q00+Q11) = P00Q00+P00Q11+P11Q00+P11Q11,
        # 4 monomials, but as a product of two linear forms it's ONE rank-1 bilinear
        # form (outer product of (1,0,0,1) in P and (1,0,0,1) in Q). In the trilinear
        # decomposition, the term is (linP)(linQ)(linC) which is ONE rank-1 trilinear
        # term, where linP, linQ, linC are linear forms. So I should add ONE column
        # where U has the coefficients of linP (P00 and P11 both =1), V has linQ
        # (Q00 and Q11 both =1), W has linC (R00 and R11 both =1).
        # Let me redo cleanly: each Strassen M is ONE column.
        pass

    # Clean implementation: each Strassen product = one column with U = linP coeffs,
    # V = linQ coeffs, W = linC coeffs.
    U = []; V = []; W = []
    def new_col():
        U.append({}); V.append({}); W.append({}); return len(U)-1
    def add(c, Ur, Uv, Vr, Vv, Wr, Wv):
        U[c][Ur] = U[c].get(Ur,0)+Uv
        V[c][Vr] = V[c].get(Vr,0)+Vv
        W[c][Wr] = W[c].get(Wr,0)+Wv

    def strassen_block(ia, ib, ja, jb, ka, kb):
        # P00=A[ia,ja], P01=A[ia,jb], P10=A[ib,ja], P11=A[ib,jb]
        # Q00=B[ja,ka], Q01=B[ja,kb], Q10=B[jb,ka], Q11=B[jb,kb]
        # R00=C[ia,ka], R01=C[ia,kb], R10=C[ib,ka], R11=C[ib,kb]
        #
        # Standard Strassen (each Mi is ONE product of two linear forms = one rank-1 term):
        # M1 = (P00+P11)(Q00+Q11)
        # M2 = (P10+P11)Q00
        # M3 = P00(Q01-Q11)
        # M4 = P11(Q10-Q00)
        # M5 = (P00+P01)Q11
        # M6 = (P10-P00)(Q00+Q10)
        # M7 = (P01-P11)(Q01+Q11)
        #
        # R00 = M1 - M4 + M5 - M7
        # R01 = M3 + M5
        # R10 = M2 + M4
        # R11 = M1 - M2 + M3 + M6

        # M1 = (P00+P11)(Q00+Q11), contributes to R00 (+1) and R11 (+1)
        c = new_col()
        add(c, Urow(ia,ja), 1, Vrow(ja,ka), 1, Wrow(ka,ia), 1)  # P00*Q00 -> R00
        add(c, Urow(ia,ja), 1, Vrow(jb,kb), 1, Wrow(ka,ia), 0)  # P00*Q11 -> (implicit in product)
        add(c, Urow(ib,jb), 1, Vrow(ja,ka), 1, Wrow(ka,ia), 0)  # P11*Q00
        add(c, Urow(ib,jb), 1, Vrow(jb,kb), 1, Wrow(ka,ia), 0)  # P11*Q11
        # Wait - this is wrong. (P00+P11)(Q00+Q11) is ONE rank-1 bilinear form.
        # In the trilinear decomposition, this is ONE column where:
        #   U has coeff 1 at (ia,ja) and 1 at (ib,jb)  [linear form P00+P11]
        #   V has coeff 1 at (ja,ka) and 1 at (jb,kb)  [linear form Q00+Q11]
        #   W has coeff 1 at (ka,ia) and 1 at (kb,ib)  [linear form R00+R11]
        # The product of these three linear forms automatically generates all 8 monomials.
        # Let me redo this properly.
        U.pop(); V.pop(); W.pop()  # remove the bad column

        # M1 = (P00+P11)(Q00+Q11), C-part = R00+R11
        c = new_col()
        add(c, Urow(ia,ja), 1, Vrow(ja,ka), 1, Wrow(ka,ia), 1)
        add(c, Urow(ia,ja), 0, Vrow(jb,kb), 1, Wrow(ka,ia), 0)
        add(c, Urow(ib,jb), 1, Vrow(ja,ka), 0, Wrow(ka,ia), 0)
        add(c, Urow(ib,jb), 0, Vrow(jb,kb), 0, Wrow(ka,ia), 0)
        add(c, Urow(ia,ja), 0, Vrow(ja,ka), 0, Wrow(kb,ib), 1)
        add(c, Urow(ia,ja), 0, Vrow(jb,kb), 0, Wrow(kb,ib), 0)
        add(c, Urow(ib,jb), 0, Vrow(ja,ka), 0, Wrow(kb,ib), 0)
        add(c, Urow(ib,jb), 0, Vrow(jb,kb), 0, Wrow(kb,ib), 0)
        # Hmm, this is still wrong. A rank-1 trilinear term is u_i * v_j * w_k.
        # If U has entries at two positions, V at two positions, W at two positions,
        # the product generates 2*2*2=8 monomials, which is correct for
        # (P00+P11)(Q00+Q11)(R00+R11). But I need to set the coefficients correctly.
        # U[i*m+j, c] should be the coefficient of A[i,j] in the linear form.
        # For M1: linA = A[ia,ja] + A[ib,jb], so U[ia*m+ja,c]=1, U[ib*m+jb,c]=1.
        # linB = B[ja,ka] + B[jb,kb], so V[ja*p+ka,c]=1, V[jb*p+kb,c]=1.
        # linC = C[ia,ka] + C[ib,kb], so W[ka*n+ia,c]=1, W[kb*n+ib,c]=1.
        # This is ONE column with 2 nonzeros in U, 2 in V, 2 in W.
        U.pop(); V.pop(); W.pop()

        # M1 = (P00+P11)(Q00+Q11), C-part = R00+R11
        c = new_col()
        add(c, Urow(ia,ja), 1, Vrow(ja,ka), 1, Wrow(ka,ia), 1)
        add(c, Urow(ib,jb), 1, Vrow(jb,kb), 1, Wrow(kb,ib), 1)
        # Now U has {Urow(ia,ja):1, Urow(ib,jb):1}
        # V has {Vrow(ja,ka):1, Vrow(jb,kb):1}
        # W has {Wrow(ka,ia):1, Wrow(kb,ib):1}
        # Product gives: A[ia,ja]B[ja,ka]C[ia,ka] + A[ia,ja]B[jb,kb]C[ia,ka]
        #   + A[ib,jb]B[ja,ka]C[ia,ka] + A[ib,jb]B[jb,kb]C[ia,ka]
        #   + A[ia,ja]B[ja,ka]C[ib,kb] + A[ia,ja]B[jb,kb]C[ib,kb]
        #   + A[ib,jb]B[ja,ka]C[ib,kb] + A[ib,jb]B[jb,kb]C[ib,kb]
        # But we want: (A[ia,ja]+A[ib,jb])(B[ja,ka]+B[jb,kb])(C[ia,ka]+C[ib,kb])
        # = A[ia,ja]B[ja,ka]C[ia,ka] + A[ia,ja]B[ja,ka]C[ib,kb]
        #   + A[ia,ja]B[jb,kb]C[ia,ka] + A[ia,ja]B[jb,kb]C[ib,kb]
        #   + A[ib,jb]B[ja,ka]C[ia,ka] + A[ib,jb]B[ja,ka]C[ib,kb]
        #   + A[ib,jb]B[jb,kb]C[ia,ka] + A[ib,jb]B[jb,kb]C[ib,kb]
        # The product of the linear forms gives exactly these 8 terms. Correct!

        # M2 = (P10+P11)Q00, C-part = R10
        c = new_col()
        add(c, Urow(ib,ja), 1, Vrow(ja,ka), 1, Wrow(ka,ib), 1)
        add(c, Urow(ib,jb), 1, Vrow(ja,ka), 0, Wrow(ka,ib), 0)

        # M3 = P00(Q01-Q11), C-part = R01+R11
        c = new_col()
        add(c, Urow(ia,ja), 1, Vrow(ja,kb), 1, Wrow(kb,ia), 1)
        add(c, Urow(ia,ja), 0, Vrow(jb,kb), -1, Wrow(kb,ia), 0)
        add(c, Urow(ia,ja), 0, Vrow(ja,kb), 0, Wrow(kb,ib), 1)
        add(c, Urow(ia,ja), 0, Vrow(jb,kb), 0, Wrow(kb,ib), 0)
        # U: {Urow(ia,ja):1}, V: {Vrow(ja,kb):1, Vrow(jb,kb):-1}, W: {Wrow(kb,ia):1, Wrow(kb,ib):1}
        # Product: A[ia,ja]*(B[ja,kb]-B[jb,kb])*(C[ia,kb]+C[ib,kb])
        # = A[ia,ja]B[ja,kb]C[ia,kb] + A[ia,ja]B[ja,kb]C[ib,kb]
        #   - A[ia,ja]B[jb,kb]C[ia,kb] - A[ia,ja]B[jb,kb]C[ib,kb]
        # Correct for M3 = P00(Q01-Q11) with C-part R01+R11.

        # M4 = P11(Q10-Q00), C-part = -R00+R10
        c = new_col()
        add(c, Urow(ib,jb), 1, Vrow(jb,ka), 1, Wrow(ka,ia), -1)
        add(c, Urow(ib,jb), 0, Vrow(ja,ka), -1, Wrow(ka,ia), 0)
        add(c, Urow(ib,jb), 0, Vrow(jb,ka), 0, Wrow(ka,ib), 1)
        add(c, Urow(ib,jb), 0, Vrow(ja,ka), 0, Wrow(ka,ib), 0)
        # U: {Urow(ib,jb):1}, V: {Vrow(jb,ka):1, Vrow(ja,ka):-1}, W: {Wrow(ka,ia):-1, Wrow(ka,ib):1}
        # Product: A[ib,jb]*(B[jb,ka]-B[ja,ka])*(-C[ia,ka]+C[ib,ka])
        # = -A[ib,jb]B[jb,ka]C[ia,ka] + A[ib,jb]B[jb,ka]C[ib,ka]
        #   + A[ib,jb]B[ja,ka]C[ia,ka] - A[ib,jb]B[ja,ka]C[ib,ka]
        # Correct for M4 = P11(Q10-Q00) with C-part -R00+R10.

        # M5 = (P00+P01)Q11, C-part = R00+R01
        c = new_col()
        add(c, Urow(ia,ja), 1, Vrow(jb,kb), 1, Wrow(ka,ia), 1)
        add(c, Urow(ia,jb), 1, Vrow(jb,kb), 0, Wrow(ka,ia), 0)
        add(c, Urow(ia,ja), 0, Vrow(jb,kb), 0, Wrow(kb,ia), 1)
        add(c, Urow(ia,jb), 0, Vrow(jb,kb), 0, Wrow(kb,ia), 0)
        # U: {Urow(ia,ja):1, Urow(ia,jb):1}, V: {Vrow(jb,kb):1}, W: {Wrow(ka,ia):1, Wrow(kb,ia):1}
        # Product: (A[ia,ja]+A[ia,jb])*B[jb,kb]*(C[ia,ka]+C[ia,kb])
        # = A[ia,ja]B[jb,kb]C[ia,ka] + A[ia,ja]B[jb,kb]C[ia,kb]
        #   + A[ia,jb]B[jb,kb]C[ia,ka] + A[ia,jb]B[jb,kb]C[ia,kb]
        # Correct for M5 = (P00+P01)Q11 with C-part R00+R01.

        # M6 = (P10-P00)(Q00+Q10), C-part = R11
        c = new_col()
        add(c, Urow(ib,ja), 1, Vrow(ja,ka), 1, Wrow(kb,ib), 1)
        add(c, Urow(ia,ja), -1, Vrow(ja,ka), 0, Wrow(kb,ib), 0)
        add(c, Urow(ib,ja), 0, Vrow(jb,ka), 1, Wrow(kb,ib), 0)
        add(c, Urow(ia,ja), 0, Vrow(jb,ka), 0, Wrow(kb,ib), 0)
        # U: {Urow(ib,ja):1, Urow(ia,ja):-1}, V: {Vrow(ja,ka):1, Vrow(jb,ka):1}, W: {Wrow(kb,ib):1}
        # Product: (A[ib,ja]-A[ia,ja])*(B[ja,ka]+B[jb,ka])*C[ib,kb]
        # = A[ib,ja]B[ja,ka]C[ib,kb] + A[ib,ja]B[jb,ka]C[ib,kb]
        #   - A[ia,ja]B[ja,ka]C[ib,kb] - A[ia,ja]B[jb,ka]C[ib,kb]
        # Correct for M6 = (P10-P00)(Q00+Q10) with C-part R11.

        # M7 = (P01-P11)(Q01+Q11), C-part = -R00
        c = new_col()
        add(c, Urow(ia,jb), 1, Vrow(ja,kb), 1, Wrow(ka,ia), -1)
        add(c, Urow(ib,jb), -1, Vrow(ja,kb), 0, Wrow(ka,ia), 0)
        add(c, Urow(ia,jb), 0, Vrow(jb,kb), 1, Wrow(ka,ia), 0)
        add(c, Urow(ib,jb), 0, Vrow(jb,kb), 0, Wrow(ka,ia), 0)
        # U: {Urow(ia,jb):1, Urow(ib,jb):-1}, V: {Vrow(ja,kb):1, Vrow(jb,kb):1}, W: {Wrow(ka,ia):-1}
        # Product: (A[ia,jb]-A[ib,jb])*(B[ja,kb]+B[jb,kb])*(-C[ia,ka])
        # = -A[ia,jb]B[ja,kb]C[ia,ka] - A[ia,jb]B[jb,kb]C[ia,ka]
        #   + A[ib,jb]B[ja,kb]C[ia,ka] + A[ib,jb]B[jb,kb]C[ia,ka]
        # Correct for M7 = (P01-P11)(Q01+Q11) with C-part -R00.

    # Now assemble the full 2x3*3x6 decomposition.
    # A columns: 0,1,2. B rows: 0,1,2. C columns: 0..5.
    # Part 1: A[:,0:2] * B[0:2, :] : 2x2 * 2x6.
    #   B[0:2,:] has 6 columns; pair them: (0,1),(2,3),(4,5).
    #   Each pair is a 2x2*2x2 block with A-rows (0,1), A-cols (0,1), B-rows (0,1),
    #   B-cols (pair), C-rows (0,1), C-cols (pair).
    for (ka, kb) in [(0,1),(2,3),(4,5)]:
        strassen_block(ia=0, ib=1, ja=0, jb=1, ka=ka, kb=kb)
    # Part 2: A[:,2:3] * B[2:3, :] : 2x1 * 1x6.
    #   C[i,k] = A[i,2] B[2,k]. Trilinear form: C[i,k] A[i,2] B[2,k].
    #   This is NOT a single rank-1 term (as established). It's 12 monomials:
    #   for each (i,k): C[i,k] A[i,2] B[2,k]. So 2*6 = 12 columns.
    #   But we can do better: group by i. For fixed i, C[i,k] A[i,2] B[2,k] for k=0..5
    #   is (A[i,2]) * (sum_k C[i,k] B[2,k])? No, that's not a product of linear forms
    #   in the right way. Let me think: the trilinear form for fixed i is
    #   sum_k C[i,k] A[i,2] B[2,k] = A[i,2] * (sum_k C[i,k] B[2,k]).
    #   sum_k C[i,k] B[2,k] is a bilinear form in (C,B). And A[i,2] is linear in A.
    #   So for fixed i, it's (A[i,2]) * (bilinear form in C,B). The bilinear form
    #   sum_k C[i,k] B[2,k] has rank 6 (one monomial per k). So fixed i contributes
    #   rank 6. Two i's -> rank 12. No improvement.
    #   Alternatively, for fixed k: C[i,k] A[i,2] B[2,k] = B[2,k] * (sum_i C[i,k] A[i,2]),
    #   rank 2 per k, 6 k's -> 12. Same.
    #   So Part 2 = rank 12 in this naive grouping. Total = 21 + 12 = 33.
    #   That matches the "Strassen-based block decomposition rank 33" from the
    #   knowledge! Great.
    for i in range(2):
        for k in range(6):
            c = new_col()
            add(c, Urow(i,2),1, Vrow(2,k),1, Wrow(k,i),1)

    # Convert to list-of-lists.
    rank = len(U)
    Umat = [[0]*rank for _ in range(n*m)]
    Vmat = [[0]*rank for _ in range(m*p)]
    Wmat = [[0]*rank for _ in range(p*n)]
    for c in range(rank):
        for r, v in U[c].items():
            Umat[r][c] = v
        for r, v in V[c].items():
            Vmat[r][c] = v
        for r, v in W[c].items():
            Wmat[r][c] = v
    return {"U": Umat, "V": Vmat, "W": Wmat}


def construct():
    deadline = time.time() + 120.0
    best = None
    # Try Strassen block decomposition (rank 33).
    try:
        cand = _strassen_block_decomposition()
        if _verify(cand["U"], cand["V"], cand["W"]):
            best = cand
        else:
            # Debug: count mismatches
            n, m, p = DIMENSIONS
            rank = len(cand["U"][0]) if cand["U"] else 0
            mismatches = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(rank):
                            val += cand["U"][i*m+j][c] * cand["V"][j*p+k][c] * cand["W"][k*n+i][c]
                        if val != 1:
                            mismatches.append((i,j,k,val))
            if len(mismatches) <= 5:
                pass  # would print in debug
    except Exception:
        pass
    # Fallback to schoolbook (rank 36).
    school = _schoolbook()
    if best is None or len(school["U"][0]) < len(best["U"][0]):
        best = school
    return best
# EVOLVE-BLOCK-END
