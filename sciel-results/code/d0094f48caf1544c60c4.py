import time

DIMENSIONS = [3, 3, 3]
_DEADLINE_S = 120.0
_START = time.time()

def _time_left():
    return _DEADLINE_S - (time.time() - _START)

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify_exact(U, V, W):
    n, m, p = DIMENSIONS
    r = len(U[0])
    Ud = [[int(round(2 * x)) for x in row] for row in U]
    Vd = [[int(round(2 * x)) for x in row] for row in V]
    Wd = [[int(round(2 * x)) for x in row] for row in W]
    for M in (Ud, Vd, Wd):
        for row in M:
            for x in row:
                if x not in (-2, -1, 0, 1, 2):
                    return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for c in range(r):
                    s += Ud[ui][c] * Vd[vj][c] * Wd[wk][c]
                if s != 8:
                    return False
    return True

def _rank23_construct():
    """Construct rank-23 3x3x3 matmul decomposition via 2x2 Strassen cores.
    
    Layout: 3x3 matrix partitioned. We use:
      - Top-left 2x2 Strassen core (A[0:2,0:2] x B[0:2,0:2]) -> C[0:2,0:2]
      - Bottom-right 2x2 Strassen core (A[1:3,1:3] x B[1:3,1:3]) -> C[1:3,1:3]
      - The C[1,1] entry is shared; we split it 1/2 + 1/2 between the two cores.
      - Off-diagonal C[0,2] and C[2,0] via schoolbook (3 products each).
    
    Total: 7 + 7 + 3 + 3 = 20? But C[1,1] double-counting means we need to
    halve both Strassen C[1,1] contributions. Let me just build and verify.
    """
    n = m = p = 3
    U = [[0] * 23 for _ in range(9)]
    V = [[0] * 23 for _ in range(9)]
    W = [[0] * 23 for _ in range(9)]
    
    def add_term(u, v, w):
        # u, v, w are dicts mapping index -> value
        for a, val in u.items():
            U[a][cur_c] += val
        for b, val in v.items():
            V[b][cur_c] += val
        for d, val in w.items():
            W[d][cur_c] += val
    
    cur_c = 0
    # We'll build 23 terms. Let me use a cleaner approach: define each term
    # explicitly as (U_vec, V_vec, W_vec) and verify at the end.
    # 
    # I'll use the known explicit rank-23 construction. Let me define it as
    # 23 triples. (Derivation: Strassen 2x2 on overlapping blocks.)
    
    # Given the complexity and risk of error, I'll use a verified explicit table.
    # This is the standard rank-23 3x3x3 decomposition.
    terms = []
    
    # --- Top-left 2x2 Strassen (7 terms) ---
    # A indices: (0,0)=0,(0,1)=1,(1,0)=3,(1,1)=4
    # B indices: (0,0)=0,(0,1)=1,(1,0)=3,(1,1)=4
    # C indices: (0,0)=0,(0,1)=3,(1,0)=1,(1,1)=4  [W[k*n+i]: k=0,i=0 ->0; k=1,i=0->1; k=0,i=1->3; k=1,i=1->4]
    # M1=(A00+A11)(B00+B11)
    terms.append((
        {0:1, 4:1},   # A00+A11
        {0:1, 4:1},   # B00+B11
        {0:1, 3:1, 1:1, 4:1},  # C00+C01+C10+C11
    ))
    # M2=(A10+A11)B00
    terms.append((
        {3:1, 4:1},   # A10+A11
        {0:1},        # B00
        {1:1, 4:1},   # C10+C11
    ))
    # M3=A00(B10-B11)
    terms.append((
        {0:1},        # A00
        {3:1, 4:-1},  # B10-B11
        {1:1, 4:-1},  # C10-C11
    ))
    # M4=A11(B01-B00)
    terms.append((
        {4:1},        # A11
        {1:1, 0:-1},  # B01-B00
        {0:1, 3:-1},  # C00-C01  ... wait
    ))
    # Hmm, let me recheck Strassen formulas.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # So M4 contributes to C11 (top-right of 2x2 = C[0,1]) and C21 (bottom-left = C[1,0]).
    # In our 3x3 indexing, the 2x2 block C[0:2,0:2] has:
    #   C[0,0] (top-left), C[0,1] (top-right), C[1,0] (bottom-left), C[1,1] (bottom-right)
    # Strassen C11 = top-left = our C[0,0]
    # Strassen C12 = top-right = our C[0,1]
    # Strassen C21 = bottom-left = our C[1,0]
    # Strassen C22 = bottom-right = our C[1,1]
    # So:
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Let me redo the W vectors with correct C indices:
    # C[0,0] -> W index k*n+i = 0*3+0 = 0
    # C[0,1] -> W index k*n+i = 1*3+0 = 3
    # C[1,0] -> W index k*n+i = 0*3+1 = 1
    # C[1,1] -> W index k*n+i = 1*3+1 = 4
    
    # M1=(A00+A11)(B00+B11): contributes +1 to C[0,0],C[0,1],C[1,0],C[1,1]
    #   W: {0:1, 3:1, 1:1, 4:1}  (C00,C01,C10,C11)
    # M2=(A10+A11)B00: contributes +1 to C[1,0],C[1,1]
    #   W: {1:1, 4:1}
    # M3=A00(B10-B11): contributes +1 to C[0,1], -1 to C[1,1]
    #   W: {3:1, 4:-1}
    # M4=A11(B01-B00): contributes +1 to C[0,0], +1 to C[1,0]
    #   W: {0:1, 1:1}
    # M5=(A00+A01)B11: contributes -1 to C[0,0], +1 to C[0,1]
    #   W: {0:-1, 3:1}
    # M6=(A10-A00)(B00+B01): contributes +1 to C[1,1]
    #   W: {4:1}
    # M7=(A01-A11)(B10+B11): contributes +1 to C[0,0]
    #   W: {0:1}
    
    # Let me rebuild terms properly.
    terms = []
    # M1
    terms.append(({0:1, 4:1}, {0:1, 4:1}, {0:1, 3:1, 1:1, 4:1}))
    # M2
    terms.append(({3:1, 4:1}, {0:1}, {1:1, 4:1}))
    # M3
    terms.append(({0:1}, {3:1, 4:-1}, {3:1, 4:-1}))
    # M4
    terms.append(({4:1}, {1:1, 0:-1}, {0:1, 1:1}))
    # M5
    terms.append(({0:1, 1:1}, {4:1}, {0:-1, 3:1}))
    # M6
    terms.append(({3:1, 0:-1}, {0:1, 1:1}, {4:1}))
    # M7
    terms.append(({1:1, 4:-1}, {3:1, 4:1}, {0:1}))
    
    # That's 7 terms for top-left. Now bottom-right 2x2 Strassen.
    # A[1:3,1:3]: A[1,1]=4, A[1,2]=5, A[2,1]=7, A[2,2]=8
    # B[1:3,1:3]: B[1,1]=4, B[1,2]=5, B[2,1]=7, B[2,2]=8
    # C[1:3,1:3]: C[1,1]=4, C[1,2]=7, C[2,1]=5, C[2,2]=8
    # W indices: C[1,1]->4, C[1,2]->k=2,i=1->2*3+1=7, C[2,1]->k=1,i=2->1*3+2=5, C[2,2]->k=2,i=2->2*3+2=8
    # Strassen for bottom-right:
    # N1=(A11+A22)(B11+B22): +1 to C[1,1],C[1,2],C[2,1],C[2,2]
    # N2=(A21+A22)B11: +1 to C[2,1],C[2,2]
    # N3=A11(B21-B22): +1 to C[1,2], -1 to C[2,2]
    # N4=A22(B12-B11): +1 to C[1,1], +1 to C[2,1]
    # N5=(A11+A12)B22: -1 to C[1,1], +1 to C[1,2]
    # N6=(A21-A11)(B11+B12): +1 to C[2,2]
    # N7=(A12-A22)(B21+B22): +1 to C[1,1]
    
    # But C[1,1] is already computed by the top-left core (M1-M2+M3+M6). 
    # If we add the bottom-right core's C[1,1] contribution, C[1,1] would be 2.
    # We need C[1,1] = 1 total. So we must split: halve both.
    # This requires half-integer entries (1/2), which is allowed.
    
    # Alternative: don't use full Strassen for bottom-right; instead, 
    # compute C[1,1] only from top-left, and use 3 additional products for 
    # C[1,2], C[2,1], C[2,2]. That gives 7 + 3 + 3 (for C[0,2],C[2,0]) + ... 
    # Let me count: top-left 7 (covers C00,C01,C10,C11), plus C12,C21,C22 (3 schoolbook), 
    # plus C02,C20 (3+3 schoolbook) = 7+3+3+3 = 16? No, schoolbook for C[1,2] is 
    # A[1,0]B[0,2]+A[1,1]B[1,2]+A[1,2]B[2,2] = 3 products. Similarly C[2,1], C[0,2], C[2,0].
    # Total: 7 + 3*4 = 19. That's rank 19! But that's below the known lower bound of 23, 
    # which is impossible. So I'm making an error.
    
    # The error: the Strassen products M1..M7 are NOT independent schoolbook products. 
    # They are linear combinations, so they don't directly give the C entries as sums 
    # of A[i,j]B[j,k] terms. The tensor decomposition requires each "product" to be 
    # a single A[i,j]*B[j,k] bilinear term (or a linear combination that factors).
    
    # Actually, in a tensor decomposition, each term r is a product of a linear form 
    # in A, a linear form in B, and a linear form in C. The Strassen M1 = (A00+A11)(B00+B11) 
    # is a product of two linear forms, so it's ONE tensor term. Its W vector has the 
    # coefficients of C. So M1 contributes to C[0,0],C[0,1],C[1,0],C[1,1] with coefficient 1 each.
    # That's valid! Each M is one term.
    
    # So the 7 Strassen terms are 7 valid tensor terms. The issue is just the C[1,1] overlap.
    # If top-left gives C[1,1] = M1-M2+M3+M6 (coefficient 1 in the sum) and bottom-right 
    # gives C[1,1] = N1-N2+N3+N7 (coefficient 1 in the sum), then total C[1,1] = 2.
    # To fix: scale both cores by 1/2. Then C[1,1] = 1/2 + 1/2 = 1. But then C[0,0] = 1/2 
    # (only from top-left), which is wrong.
    
    # So we can't simply halve. The standard rank-23 construction must handle this differently.
    
    # Given the difficulty, let me just use a KNOWN correct rank-23 table. I'll look it up 
    # in my knowledge: the standard 3x3x3 rank-23 decomposition.
    
    # I'll use the following explicit construction (verified in code):
    # Actually, let me just return schoolbook to be safe, OR use a simpler correct lower rank.
    
    # The safest: return None to indicate "no verified lower rank found", and let 
    # construct() fall back to schoolbook. But that gives rank 27 = score 1.0.
    
    # Let me try a different, simpler approach: use the 2x2 Strassen for the WHOLE 
    # 3x3 by padding? No.
    
    # I'll just construct the top-left Strassen (7 terms) + schoolbook for the rest 
    # (C[0,2], C[1,2], C[2,0], C[2,1], C[2,2] = 5 entries * 3 = 15 products) = 22 terms.
    # But C[1,1] is covered by Strassen, so we don't need schoolbook for it.
    # Entries to cover by schoolbook: C[0,2], C[1,2], C[2,0], C[2,1], C[2,2] = 5 entries.
    # Each needs 3 products = 15 products. Total = 7 + 15 = 22.
    # But wait, do the Strassen terms correctly give C[0,0],C[0,1],C[1,0],C[1,1]? Yes.
    # And schoolbook gives the other 5. Total rank = 22.
    # But 22 < 23 (known lower bound), which is impossible. So there's still an error.
    
    # The error: the Strassen M terms, when written as tensor terms, their W vectors 
    # must sum to give the correct C. Let me verify M1..M7 actually reconstruct C[0:2,0:2].
    # C[0,0] = M1 + M4 - M5 + M7 = (A00+A11)(B00+B11) + A11(B01-B00) - (A00+A01)B11 + (A01-A11)(B10+B11)
    # Let me expand:
    # M1 = A00B00+A00B11+A11B00+A11B11
    # M4 = A11B01-A11B00
    # -M5 = -A00B11-A01B11
    # M7 = A01B10+A01B11-A11B10-A11B11
    # Sum = A00B00+A00B11+A11B00+A11B11 + A11B01-A11B00 -A00B11-A01B11 + A01B10+A01B11-A11B10-A11B11
    # = A00B00 + (A00B11-A00B11) + (A11B00-A11B00) + (A11B11-A11B11) + A11B01 - A01B11 + A01B10 + A01B11 - A11B10
    # = A00B00 + A11B01 + A01B10 - A11B10
    # But C[0,0] should be A00B00 + A01B10 + A02B20. We're missing A02B20 and have an extra -A11B10.
    # So the Strassen formula gives C[0,0] for a 2x2 matrix (no A02,B20). For 3x3, C[0,0] = A00B00+A01B10+A02B20.
    # The Strassen 2x2 only covers the top-left 2x2 submatrix, so it gives A00B00+A01B10 (missing A02B20).
    # So we ALSO need to add A02B20 to C[0,0]. That's an additional product.
    
    # Ah, this is the key insight I missed! The 2x2 Strassen core only covers the 2x2 subproblem, 
    # so it misses the "third row/col" contributions. For C[0,0], we need A02B20 additionally.
    # Similarly for other entries.
    
    # So the correct count: Strassen top-left (7 terms) covers the "2x2 part" of C[0:2,0:2], 
    # but each of C[0,0],C[0,1],C[1,0],C[1,1] needs an additional A[i,2]B[2,k] term. 
    # That's 4 additional products. Plus C[0,2],C[1,2],C[2,0],C[2,1],C[2,2] each need 3 products = 15.
    # Total = 7 + 4 + 15 = 26. Still not 23.
    
    # I'm clearly not getting this right by hand. Let me just use a verified explicit rank-23 table 
    # from the literature, or fall back to schoolbook.
    
    # Given the risk, I'll return a schoolbook-based construction but try to find a 
    # verified lower rank via a small search. Actually, let me just return schoolbook 
    # to guarantee correctness, and structure the code differently from the current program.
    
    # I'll return None to signal "no lower rank constructed", and construct() uses schoolbook.
    return None

# EVOLVE-BLOCK-START
def construct():
    best = _schoolbook()
    if _time_left() > 5.0:
        cand = _rank23_construct()
        if cand is not None and _verify_exact(cand["U"], cand["V"], cand["W"]):
            best = cand
    if _time_left() <= 5.0:
        return best
    return best
# EVOLVE-BLOCK-END
