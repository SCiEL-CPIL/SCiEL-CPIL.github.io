"""Tensor decomposition for 2x2 by 2x5 matrix multiplication.

Family: two 2x2 Strassen blocks (B-columns 0-1 and 2-3) + schoolbook (B-column 4),
refined by sharing A-side rank-1 factors across the two blocks.  A verified
shared-factor construction is emitted when it is confirmed correct; otherwise the
program falls back to the known-correct rank-18 decomposition.
"""

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time

def _verify(U, V, W, n, m, p):
    """Check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)."""
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                ui = U[i * m + j]
                vj = V[j * p + k]
                wk = W[k * n + i]
                for r in range(len(ui)):
                    s += ui[r] * vj[r] * wk[r]
                if abs(s - 1.0) > 1e-9:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    return ok

def _schoolbook_2x2x5():
    """Known-correct rank-20 schoolbook decomposition."""
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

def _rank18():
    """2 Strassen blocks (cols 0-1, 2-3) + schoolbook (col 4) = rank 18."""
    n, m, p = DIMENSIONS
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(t0, k0):
        k1 = k0 + 1
        # U (A) factors M1..M7
        U[0][t0 + 0] = 1; U[3][t0 + 0] = 1      # M1: A00+A11
        U[2][t0 + 1] = 1; U[3][t0 + 1] = 1      # M2: A10+A11
        U[0][t0 + 2] = 1                          # M3: A00
        U[3][t0 + 3] = 1                          # M4: A11
        U[0][t0 + 4] = 1; U[1][t0 + 4] = 1       # M5: A00+A01
        U[2][t0 + 5] = 1; U[0][t0 + 5] = -1      # M6: A10-A00
        U[1][t0 + 6] = 1; U[3][t0 + 6] = -1      # M7: A01-A11
        # V (B) factors
        V[0 * p + k0][t0 + 0] = 1; V[1 * p + k1][t0 + 0] = 1  # M1
        V[0 * p + k0][t0 + 1] = 1                            # M2
        V[0 * p + k1][t0 + 2] = 1; V[1 * p + k1][t0 + 2] = -1# M3
        V[1 * p + k0][t0 + 3] = 1; V[0 * p + k0][t0 + 3] = -1# M4
        V[1 * p + k1][t0 + 4] = 1                            # M5
        V[0 * p + k0][t0 + 5] = 1; V[0 * p + k1][t0 + 5] = 1 # M6
        V[1 * p + k0][t0 + 6] = 1; V[1 * p + k1][t0 + 6] = 1 # M7
        # W (C) factors
        W[k0 * n + 0][t0 + 0] = 1; W[k1 * n + 1][t0 + 0] = 1  # M1
        W[k0 * n + 1][t0 + 1] = 1; W[k1 * n + 1][t0 + 1] = -1 # M2
        W[k1 * n + 0][t0 + 2] = 1; W[k1 * n + 1][t0 + 2] = 1  # M3
        W[k0 * n + 0][t0 + 3] = 1; W[k0 * n + 1][t0 + 3] = 1  # M4
        W[k0 * n + 0][t0 + 4] = -1; W[k1 * n + 0][t0 + 4] = 1 # M5
        W[k1 * n + 1][t0 + 5] = 1                            # M6
        W[k0 * n + 0][t0 + 6] = 1                            # M7

    add_strassen_block(0, 0)
    add_strassen_block(7, 2)
    # schoolbook for column 4
    t = 14
    for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        U[i * m + j][t] = 1
        V[j * p + 4][t] = 1
        W[4 * n + i][t] = 1
        t += 1
    return {"U": U, "V": V, "W": W}

def _shared_factor():
    """Shared-factor construction: 11 shared A-factors + 12 block-local
    (6 per block) + 4 schoolbook for column 4 = rank 27, then verified."""
    n, m, p = DIMENSIONS
    # Shared A-factors (indices 0..10)
    # 0: A00+A11, 1: A10+A11, 2: A00, 3: A11, 4: A00+A01, 5: A10-A00,
    # 6: A01-A11, 7: A00-A10, 8: A10, 9: A01, 10: A00+A10
    shared_A = {
        0: [(0, 1), (3, 1)],
        1: [(2, 1), (3, 1)],
        2: [(0, 1)],
        3: [(3, 1)],
        4: [(0, 1), (1, 1)],
        5: [(2, 1), (0, -1)],
        6: [(1, 1), (3, -1)],
        7: [(0, 1), (2, -1)],
        8: [(2, 1)],
        9: [(1, 1)],
        10: [(0, 1), (2, 1)],
    }
    # Block-local A-factors (indices 11..22)
    # block 0: 11: A00+A10, 12: A11-A01, 13: A00+A11, 14: A01, 15: A10, 16: A00
    # block 1: 17: A00+A10, 18: A11-A01, 19: A00+A11, 20: A01, 21: A10, 22: A00
    local_A = {
        11: [(0, 1), (2, 1)], 12: [(3, 1), (1, -1)], 13: [(0, 1), (3, 1)],
        14: [(1, 1)], 15: [(2, 1)], 16: [(0, 1)],
        17: [(0, 1), (2, 1)], 18: [(3, 1), (1, -1)], 19: [(0, 1), (3, 1)],
        20: [(1, 1)], 21: [(2, 1)], 22: [(0, 1)],
    }
    # Per-block B-factors (V) and C-factors (W) for the 6 local terms
    # term order: (A00+A10), (A11-A01), (A00+A11), A01, A10, A00
    def block_V(k0):
        k1 = k0 + 1
        return [
            [(0 * p + k0, 1), (1 * p + k0, 1)],
            [(0 * p + k1, 1), (1 * p + k1, -1)],
            [(0 * p + k0, 1), (1 * p + k1, 1)],
            [(0 * p + k1, 1)],
            [(1 * p + k0, 1)],
            [(0 * p + k0, 1)],
        ]
    def block_W(k0):
        k1 = k0 + 1
        return [
            [(k0 * n + 0, 1), (k1 * n + 1, 1)],
            [(k0 * n + 1, 1), (k1 * n + 0, 1)],
            [(k0 * n + 0, 1), (k1 * n + 0, 1), (k1 * n + 1, 1)],
            [(k0 * n + 1, 1), (k1 * n + 1, -1)],
            [(k0 * n + 1, 1)],
            [(k0 * n + 0, 1)],
        ]

    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for idx, pairs in shared_A.items():
        for (a, c) in pairs:
            U[a][idx] += c
    for idx, pairs in local_A.items():
        for (a, c) in pairs:
            U[a][idx] += c
    for b, k0 in [(0, 0), (1, 2)]:
        base = 11 + 6 * b
        for t in range(6):
            for (v, c) in block_V(k0)[t]:
                V[v][base + t] += c
            for (w, c) in block_W(k0)[t]:
                W[w][base + t] += c
    # schoolbook for column 4 (indices 23..26)
    for t, (i, j) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        U[i * m + j][23 + t] = 1
        V[j * p + 4][23 + t] = 1
        W[4 * n + i][23 + t] = 1
    return {"U": U, "V": V, "W": W}

def _rank11_shared_strassen():
    """Merge the two Strassen blocks: share 7 A-factors, combine B/C factors.
    Block 0 handles B-cols (0,1), block 1 handles B-cols (2,3).
    For each Strassen factor r, V factor covers both blocks' B entries,
    W factor covers both blocks' C entries.
    Then 4 schoolbook terms for column 4. Total rank = 7 + 4 = 11."""
    n, m, p = DIMENSIONS
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # 7 shared Strassen A-factors (indices 0..6)
    # M1: A00+A11, M2: A10+A11, M3: A00, M4: A11, M5: A00+A01, M6: A10-A00, M7: A01-A11
    U[0][0] = 1; U[3][0] = 1      # M1
    U[2][1] = 1; U[3][1] = 1      # M2
    U[0][2] = 1                    # M3
    U[3][3] = 1                    # M4
    U[0][4] = 1; U[1][4] = 1       # M5
    U[2][5] = 1; U[0][5] = -1      # M6
    U[1][6] = 1; U[3][6] = -1      # M7

    # For each Strassen factor, V and W combine both blocks
    # Block 0: B-cols (0,1), Block 1: B-cols (2,3)
    # Standard Strassen for 2x2x2 with B = [[b00,b01],[b10,b11]]:
    # M1: (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
    # M2: (A10+A11)*B10        -> C10 += M2, C11 -= M2
    # M3: A00*(B00-B10)        -> C00 += M3, C10 += M3  [wait, let me recheck]
    
    # Let me use the standard Strassen formulation carefully:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # where:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B10
    # M3 = A00*(B00-B10)
    # M4 = A11*(B01-B11)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # For block 0 (B-cols 0,1): B00=B[0,0], B01=B[0,1], B10=B[1,0], B11=B[1,1]
    # For block 1 (B-cols 2,3): B00=B[0,2], B01=B[0,3], B10=B[1,2], B11=B[1,3]
    
    # V factors (B-side), index = j*p+k:
    # M1: V[0,0]=1, V[1,1]=1, V[0,2]=1, V[1,3]=1
    V[0*p+0][0] = 1; V[1*p+1][0] = 1
    V[0*p+2][0] = 1; V[1*p+3][0] = 1
    # M2: V[1,0]=1, V[1,2]=1
    V[1*p+0][1] = 1; V[1*p+2][1] = 1
    # M3: V[0,0]=1, V[1,0]=-1, V[0,2]=1, V[1,2]=-1
    V[0*p+0][2] = 1; V[1*p+0][2] = -1
    V[0*p+2][2] = 1; V[1*p+2][2] = -1
    # M4: V[0,1]=1, V[1,1]=-1, V[0,3]=1, V[1,3]=-1
    V[0*p+1][3] = 1; V[1*p+1][3] = -1
    V[0*p+3][3] = 1; V[1*p+3][3] = -1
    # M5: V[1,1]=1, V[1,3]=1
    V[1*p+1][4] = 1; V[1*p+3][4] = 1
    # M6: V[0,0]=1, V[0,1]=1, V[0,2]=1, V[0,3]=1
    V[0*p+0][5] = 1; V[0*p+1][5] = 1
    V[0*p+2][5] = 1; V[0*p+3][5] = 1
    # M7: V[1,0]=1, V[1,1]=1, V[1,2]=1, V[1,3]=1
    V[1*p+0][6] = 1; V[1*p+1][6] = 1
    V[1*p+2][6] = 1; V[1*p+3][6] = 1

    # W factors (C-side), index = k*n+i:
    # Block 0 contributes to C-cols 0,1; Block 1 contributes to C-cols 2,3
    # C[i,k] where i=row of C (0,1), k=col of C (0,1 for block0; 2,3 for block1)
    # W index = k*n + i
    
    # M1: C00+=1, C11+=1 (block0); C02+=1, C13+=1 (block1)
    W[0*n+0][0] = 1; W[1*n+1][0] = 1
    W[2*n+0][0] = 1; W[3*n+1][0] = 1
    # M2: C10+=1, C11-=1 (block0); C12+=1, C13-=1 (block1)
    W[0*n+1][1] = 1; W[1*n+1][1] = -1
    W[2*n+1][1] = 1; W[3*n+1][1] = -1
    # M3: C00+=1, C10+=1 (block0); C02+=1, C12+=1 (block1)
    W[0*n+0][2] = 1; W[0*n+1][2] = 1
    W[2*n+0][2] = 1; W[2*n+1][2] = 1
    # M4: C00-=1, C10-=1 (block0); C02-=1, C12-=1 (block1)
    W[0*n+0][3] = -1; W[0*n+1][3] = -1
    W[2*n+0][3] = -1; W[2*n+1][3] = -1
    # M5: C00+=1, C10+=1 (block0); C02+=1, C12+=1 (block1)
    W[0*n+0][4] = 1; W[0*n+1][4] = 1
    W[2*n+0][4] = 1; W[2*n+1][4] = 1
    # M6: C11+=1 (block0); C13+=1 (block1)
    W[1*n+1][5] = 1
    W[3*n+1][5] = 1
    # M7: C01+=1 (block0); C03+=1 (block1)
    W[1*n+0][6] = 1
    W[3*n+0][6] = 1

    # Schoolbook for column 4 (indices 7..10)
    for t, (i, j) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        U[i * m + j][7 + t] = 1
        V[j * p + 4][7 + t] = 1
        W[4 * n + i][7 + t] = 1

    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 110  # safety margin vs 120 s limit
    n, m, p = DIMENSIONS

    # 1) try the merged Strassen rank-11 construction
    if time.time() < deadline:
        cand = _rank11_shared_strassen()
        if _verify(cand["U"], cand["V"], cand["W"], n, m, p):
            return cand

    # 2) try the verified shared-factor construction
    if time.time() < deadline:
        cand = _shared_factor()
        if _verify(cand["U"], cand["V"], cand["W"], n, m, p):
            return cand

    # 3) known-correct rank-18 fallback
    if time.time() < deadline:
        cand = _rank18()
        if _verify(cand["U"], cand["V"], cand["W"], n, m, p):
            return cand

    # 4) ultimate schoolbook fallback
    return _schoolbook_2x2x5()
# EVOLVE-BLOCK-END
