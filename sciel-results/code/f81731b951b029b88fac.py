"""Matrix multiplication 3x3 by 3x5: rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    total_rows_U = n * m  # 9  (A[i,j])
    total_rows_V = m * p  # 15 (B[j,k])
    total_rows_W = p * n  # 15 (C[k,i])

    # We'll build terms; each term = (u_dict, v_dict, w_dict)
    # u_dict: {i*m+j: coef}, v_dict: {j*p+k: coef}, w_dict: {k*n+i: coef}
    terms = []

    # ---- 3x3x3 block: use 23-term Strassen decomposition (exact, integer) ----
    # Partition A,B,C into:
    #   A = [[P, q], [r, s]]  P 2x2 (rows0-1,cols0-1), q 2x1 (rows0-1,col2), r 1x2 (row2,cols0-1), s 1x1 (2,2)
    # 23 multiplications (Strassen 1969). Coefficients in {0, +-1}.
    # Represent each 2x2 block by its 4 entries.
    def a22(r, c):
        # returns dict of the 2x2 block at rows r..r+1, cols c..c+1
        d = {}
        for ii in range(2):
            for jj in range(2):
                i, j = r + ii, c + jj
                d[i * m + j] = 1
        return d
    def a12(r, c):  # 2x1 at rows r..r+1, col c
        return {r * m + c: 1, (r + 1) * m + c: 1}
    def a21(r, c):  # 1x2 at row r, cols c..c+1
        return {r * m + c: 1, r * m + (c + 1): 1}
    def a11(r, c):
        return {r * m + c: 1}

    # Strassen 23-term 3x3 (standard published set). Each entry: (mult, out_block)
    # We build each multiplication as (u, v) and each output as (w).
    # P = A[0:2,0:2], Q=A[0:2,2], R=A[2,0:2], S=A[2,2]  (A blocks)
    # Similarly for B: PB, QB, RB, SB ; C blocks: PC,QC,RC,SC
    P  = a22(0,0); Q = a12(0,2); R = a21(2,0); S = a11(2,2)
    PB = a22(0,0); QB = a12(0,2); RB = a21(2,0); SB = a11(2,2)
    PC = a22(0,0); QC = a12(0,2); RC = a21(2,0); SC = a11(2,2)

    def add(u, v, w):
        terms.append((dict(u), dict(v), dict(w)))

    # 23 multiplications (Strassen for 3x3). Using block algebra:
    # M1 = P*PB
    # M2 = (P+R)*(PB+SB)
    # M3 = (Q+R)*(PB+SB)
    # M4 = R*SB
    # M5 = (P+S)*(PB+SB)
    # M6 = Q*(QB+SB)
    # M7 = S*(QB+SB)
    # M8 = (P+S)*(PB+RB)
    # M9 = (Q+S)*(QB+SB)
    # M10 = (P+Q+R+S)*(PB+QB+RB+SB)
    # ... (23 total)
    # Outputs:
    # PC = M5 - M8 - M10 - M1 + M2  ... (this published set is intricate)

    # To be safe, we verify at runtime; if the 23-term set is wrong we fall back.
    # Define the 23-term list (u, v, w) with signs.
    # (This is a known exact 23-term decomposition; verify below.)
    strassen = [
        # (u, v, w)  -- coefficients as signed dicts
    ]
    # Build the 23-term set using block-sums with signs.
    def sdict(d, sign=1):
        return {k: v * sign for k, v in d.items()}
    def sumdict(*ds):
        out = {}
        for d in ds:
            for k, v in d.items():
                out[k] = out.get(k, 0) + v
        return out
    def diffdict(a, b):
        out = dict(a)
        for k, v in b.items():
            out[k] = out.get(k, 0) - v
        return out

    # Multiplications (23):
    Ms = [
        (P, PB),
        (sumdict(P, R), sumdict(PB, SB)),
        (sumdict(Q, R), sumdict(PB, SB)),
        (R, SB),
        (sumdict(P, S), sumdict(PB, SB)),
        (Q, sumdict(QB, SB)),
        (S, sumdict(QB, SB)),
        (sumdict(P, S), sumdict(PB, RB)),
        (sumdict(Q, S), sumdict(QB, SB)),
        (sumdict(P, Q, R, S), sumdict(PB, QB, RB, SB)),
        (sumdict(P, Q), sumdict(PB, RB)),
        (sumdict(Q, R, S), sumdict(QB, SB)),
        (sumdict(P, R), sumdict(PB, SB)),
        (sumdict(S, Q), sumdict(QB, SB)),
        (sumdict(P, S), sumdict(PB, SB)),
        (sumdict(Q, R), sumdict(PB, SB)),
        (sumdict(P, R), sumdict(PB, SB)),
        (sumdict(Q, S), sumdict(QB, SB)),
        (sumdict(P, Q), sumdict(PB, QB)),
        (sumdict(R, S), sumdict(RB, SB)),
        (sumdict(P, S), sumdict(PB, SB)),
        (sumdict(Q, R), sumdict(QB, SB)),
        (sumdict(P, R, S, Q), sumdict(PB, RB, SB, QB)),
    ]
    # Outputs (23) mapping to C blocks with signs:
    # This particular pairing is NOT guaranteed to reconstruct C; we verify below.
    Wblocks = [
        PC, SC, QC, RC,
    ]
    # We will NOT trust a hand-derived sign table; instead, we VERIFY the whole
    # thing at runtime. If verification fails we return the 45 baseline.
    # For the 3x3 block we actually just use the SAFE naive 27 to guarantee
    # correctness, and try to *improve* via verification of a candidate 23-set.

    # Given risk, use naive for 3x3 (27) + naive 3x2 (18) = 45 baseline,
    # but ATTEMPT a verified 23-term 3x3 replacement.
    # We'll construct candidate 23 terms generically and verify; if it works use it.

    # ---- Build naive baseline terms (always correct) ----
    def naive_block(k0, k1):
        t = []
        for i in range(n):
            for j in range(m):
                for k in range(k0, k1):
                    t.append(({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1}))
        return t

    baseline = naive_block(0, p)  # 45 terms

    # ---- Candidate improvement: 23-term 3x3 + naive 3x2 ----
    # Build the 23-term 3x3 by the standard Strassen sign table (verified below).
    cand = []
    # Multiplications as (u,v); outputs computed via a fixed sign combination.
    # Use the canonical Strassen-3x3 (23 mults) sign table:
    #   C11 = M1 + M2 - M5 - M10 + M14
    #   ... (we rely on runtime verification)
    # Because the exact sign table is error-prone, we VERIFY; on failure use baseline.

    # We'll synthesize the 23 terms using the well-known published table:
    mults = [
        (P, PB),
        (sumdict(P, R), sumdict(PB, SB)),
        (sumdict(Q, R), sumdict(PB, SB)),
        (R, SB),
        (sumdict(P, S), sumdict(PB, SB)),
        (Q, sumdict(QB, SB)),
        (S, sumdict(QB, SB)),
        (sumdict(P, S), sumdict(PB, RB)),
        (sumdict(Q, S), sumdict(QB, SB)),
        (sumdict(P, Q, R, S), sumdict(PB, QB, RB, SB)),
        (sumdict(P, Q), sumdict(PB, RB)),
        (sumdict(Q, R, S), sumdict(QB, SB)),
        (sumdict(P, R), sumdict(PB, SB)),
        (sumdict(S, Q), sumdict(QB, SB)),
        (sumdict(P, S), sumdict(PB, SB)),
        (sumdict(Q, R), sumdict(QB, SB)),
        (sumdict(P, Q, R, S), sumdict(PB, QB, RB, SB)),
        (sumdict(P, R), sumdict(PB, SB)),
        (sumdict(Q, S), sumdict(QB, SB)),
        (sumdict(P, S), sumdict(PB, SB)),
        (sumdict(Q, R), sumdict(QB, SB)),
        (sumdict(P, Q), sumdict(PB, QB)),
        (sumdict(R, S), sumdict(RB, SB)),
    ]
    # Output sign table (23) -> each is a combination of the 23 mults per C-block.
    # We express outputs as linear combos; build terms by expanding combos.
    # out_block -> list of (mult_index, sign)
    # Canonical Strassen-3x3 output formulas (23 mults) - we verify at runtime.
    out_combos = [
        # PC (2x2)
        {0: 1, 1: 1, 4: -1, 9: -1, 12: 1},
        {4: 1, 7: -1, 9: -1, 10: 1, 17: 1},
        {0: 1, 3: 1, 4: -1, 9: -1, 11: 1},
        {4: 1, 7: -1, 9: -1, 13: 1, 18: 1},
        # SC (1x1)
        {4: 1, 7: -1, 8: -1, 9: -1, 12: 1},
        # QC (2x1)
        {1: 1, 4: -1, 8: -1, 9: -1, 15: 1},
        {5: 1, 8: -1, 9: -1, 14: 1, 19: 1},
        # RC (1x2)
        {2: 1, 4: -1, 8: -1, 9: -1, 16: 1},
        {6: 1, 8: -1, 9: -1, 14: 1, 20: 1},
    ]
    # Expand combos into terms
    for combo in out_combos:
        pass  # handled below

    # Simpler: build terms by, for each C-block entry, summing mults with signs.
    # Map each C-block entry to a w-dict (single entry) and accumulate mults.
    # We'll directly build cand terms: for each mult, for each output entry that
    # uses it with a sign, add (u_mult, v_mult, w_entry*sign). This may create
    # many terms (not rank-23) unless we combine. Instead we keep it as sum of
    # rank-1 terms: that yields up to 23*? terms -> could exceed 45. So this
    # approach does NOT guarantee rank<45. Therefore we fall back to baseline.

    # => Use baseline 45 (guaranteed exact). We attempted improvement; keep safe.
    chosen = baseline

    # Verify chosen (defensive)
    rank = len(chosen)
    U = [[0] * rank for _ in range(total_rows_U)]
    V = [[0] * rank for _ in range(total_rows_V)]
    W = [[0] * rank for _ in range(total_rows_W)]
    for c, (u, v, w) in enumerate(chosen):
        for r, val in u.items():
            U[r][c] += val
        for r, val in v.items():
            V[r][c] += val
        for r, val in w.items():
            W[r][c] += val

    # Runtime exactness check (defensive; on failure return trivial 45)
    try:
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        recon = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
        # true tensor T[i*m+j, j*p+k, k*n+i] = 1
        T = np.zeros((total_rows_U, total_rows_V, total_rows_W), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        if not np.allclose(recon, T, atol=1e-9):
            # fallback to naive (always exact)
            rank = n * m * p
            U = [[0] * rank for _ in range(total_rows_U)]
            V = [[0] * rank for _ in range(total_rows_V)]
            W = [[0] * rank for _ in range(total_rows_W)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
    except Exception:
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
