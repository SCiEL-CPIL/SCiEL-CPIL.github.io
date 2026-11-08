"""Combined: Kronecker product of Strassen (2x2x2) and schoolbook (2x3x1)."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns (U, V, W) lists of shape (4, 7), (4, 7), (4, 7).
    Indexing: U row = i*2+j, V row = j*2+k, W row = k*2+i.
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    # 7 Strassen products
    defs = [
        # P1 = (a00+a11)(b00+b11)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # P2 = (a10+a11)(b00)
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,1)]),
        # P3 = a00(b01-b11)
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # P4 = a11(b10-b00)
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(1,0,1),(0,0,1)]),
        # P5 = (a00+a01)(b11)
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,1),(0,1,1)]),
        # P6 = (a10-a00)(b00+b01)
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # P7 = (a01-a11)(b10+b11)
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    # Standard Strassen output:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    # Re-encode W with correct signs for each output (k,i)
    W = [[0] * 7 for _ in range(4)]
    # W row = k*2+i
    # C00 (k=0,i=0): P1(+1), P4(+1), P5(-1), P7(+1)
    W[0*2+0][0] += 1; W[0*2+0][3] += 1; W[0*2+0][4] += -1; W[0*2+0][6] += 1
    # C01 (k=0,i=1): P3(+1), P5(+1)
    W[0*2+1][2] += 1; W[0*2+1][4] += 1
    # C10 (k=1,i=0): P2(+1), P4(+1)
    W[1*2+0][1] += 1; W[1*2+0][3] += 1
    # C11 (k=1,i=1): P1(+1), P2(-1), P3(+1), P6(+1)
    W[1*2+1][0] += 1; W[1*2+1][1] += -1; W[1*2+1][2] += 1; W[1*2+1][5] += 1

    for t in range(7):
        for (i, j, c) in defs[t][0]:
            U[i * 2 + j][t] += c
        for (j, k, c) in defs[t][1]:
            V[j * 2 + k][t] += c
    return U, V, W


def _schoolbook_2x3x1():
    """
    Schoolbook decomposition for (2x3) * (3x1) -> (2x1).
    Rank = 2*3*1 = 6.
    U: shape (2*3, 6) = (6, 6), indexed by i*3+j
    V: shape (3*1, 6) = (3, 6), indexed by j*1+k
    W: shape (1*2, 6) = (2, 6), indexed by k*2+i
    """
    rank = 2 * 3 * 1
    U = [[0] * rank for _ in range(2 * 3)]
    V = [[0] * rank for _ in range(3 * 1)]
    W = [[0] * rank for _ in range(1 * 2)]
    c = 0
    for i in range(2):
        for j in range(3):
            for k in range(1):
                U[i * 3 + j][c] = 1
                V[j * 1 + k][c] = 1
                W[k * 2 + i][c] = 1
                c += 1
    return U, V, W


def _kronecker_decomp(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """
    Combine two tensor decompositions via Kronecker product.
    Decomp 1: (n1 x m1) * (m1 x p1) -> (n1 x p1)
    Decomp 2: (n2 x m2) * (m2 x p2) -> (n2 x p2)
    Result: (n1*n2 x m1*m2) * (m1*m2 x p1*p2) -> (n1*n2 x p1*p2)
    
    Indexing convention:
    A row (i1,i2), col (j1,j2) -> global (i1*n2+i2)*(m1*m2) + (j1*m2+j2)
    B row (j1,j2), col (k1,k2) -> global (j1*m2+j2)*(p1*p2) + (k1*p2+k2)
    C row (k1,k2), col (i1,i2) -> global (k1*p2+k2)*(n1*n2) + (i1*n2+i2)
    """
    n, m, p = n1 * n2, m1 * m2, p1 * p2
    r1 = len(U1[0])
    r2 = len(U2[0])
    rank = r1 * r2

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for t1 in range(r1):
        for t2 in range(r2):
            col = t1 * r2 + t2
            # U: for each (i1,i2,j1,j2)
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            a_val = U1[i1 * m1 + j1][t1]
                            b_val = U2[i2 * m2 + j2][t2]
                            if a_val != 0 and b_val != 0:
                                gi = i1 * n2 + i2
                                gj = j1 * m2 + j2
                                U[gi * m + gj][col] += a_val * b_val
            # V: for each (j1,j2,k1,k2)
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            a_val = V1[j1 * p1 + k1][t1]
                            b_val = V2[j2 * p2 + k2][t2]
                            if a_val != 0 and b_val != 0:
                                gj = j1 * m2 + j2
                                gk = k1 * p2 + k2
                                V[gj * p + gk][col] += a_val * b_val
            # W: for each (k1,k2,i1,i2)
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            a_val = W1[k1 * n1 + i1][t1]
                            b_val = W2[k2 * n2 + i2][t2]
                            if a_val != 0 and b_val != 0:
                                gk = k1 * p2 + k2
                                gi = i1 * n2 + i2
                                W[gk * n + gi][col] += a_val * b_val
    return U, V, W, rank


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify exact decomposition using random integer matrices."""
    import random
    rng = random.Random(987654321)
    rank = len(U[0])
    for _ in range(5):
        A = [[rng.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        for i in range(n):
            for k in range(p):
                val = 0
                for t in range(rank):
                    au = sum(U[i * m + j][t] * A[i][j] for j in range(m))
                    bu = sum(V[j * p + k][t] * B[j][k] for j in range(m))
                    val += W[k * n + i][t] * au * bu
                if abs(val - C[i][k]) > tol:
                    return False
    # Also check the basis tensors (e_{ij} * f_{jk} * g_{ki})
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for t in range(rank):
                    s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                if abs(s - 1.0) > tol:
                    return False
    return True


def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS  # 4, 5, 6

    # Schoolbook fallback
    def schoolbook():
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

    best_result = schoolbook()
    best_rank = n * m * p

    try:
        # Kronecker: (2x2 x 2x3) for A, (2x2 x 1x3) for B
        # A: (2*2) x (2*3) = 4 x 6? No, we need 4 x 5.
        # Actually: n1*n2 = 4, m1*m2 = 5, p1*p2 = 6
        # n1=2, n2=2 -> 4
        # m1=2, m2=3 -> 6? No, we need 5.
        # Hmm, 5 is prime. Let me reconsider.
        # 4 = 2*2, 5 = 1*5 or 5*1, 6 = 2*3 or 3*2 or 1*6 or 6*1
        # We need m1*m2 = 5, so (m1,m2) = (1,5) or (5,1)
        # And p1*p2 = 6, so (p1,p2) = (2,3) or (3,2) or (1,6) or (6,1)
        # And n1*n2 = 4, so (n1,n2) = (2,2) or (1,4) or (4,1)
        
        # Option A: n1=2,n2=2, m1=1,m2=5, p1=2,p2=3
        # Decomp 1: (2x1)*(1x2)->(2x2), rank = 2*1*2 = 4 (schoolbook) or Strassen? 
        #   2x1 x 1x2 is outer product, rank 4 (schoolbook)
        # Decomp 2: (2x5)*(5x3)->(2x3), rank = 2*5*3 = 30 (schoolbook)
        # Total: 4*30 = 120. Same as schoolbook. Not helpful.
        
        # Option B: n1=1,n2=4, m1=5,m2=1, p1=6,p2=1
        # Decomp 1: (1x5)*(5x6)->(1x6), rank = 1*5*6 = 30
        # Decomp 2: (4x1)*(1x1)->(4x1), rank = 4*1*1 = 4
        # Total: 30*4 = 120. Same.
        
        # The Kronecker approach with these dimensions doesn't help directly
        # because 5 is prime and we can't factor it into smaller known decompositions.
        
        # Let me try a different structural approach: 
        # Split 4x5x6 as (4x4x4) + (4x1x4) + (4x4x2) + (4x1x2)
        # 4x4x4: Strassen rank 49
        # 4x1x4: rank 16
        # 4x4x2: rank 32 (or Strassen 2x2x2 * 2 = 28?)
        # 4x1x2: rank 8
        # Total: 49+16+32+8 = 105. Better than 120.
        
        # Let me build 4x4x4 Strassen properly.
        U1, V1, W1 = _strassen_2x2()
        
        def build_4x4x4():
            """4x4x4 via 2-level Strassen: rank 49."""
            U = [[0] * 49 for _ in range(16)]
            V = [[0] * 49 for _ in range(16)]
            W = [[0] * 49 for _ in range(16)]
            U2, V2, W2 = _strassen_2x2()
            outer = [
                ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
                ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,1)]),
                ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
                ([(1,1,1)], [(1,0,1),(0,0,-1)], [(1,0,1),(0,0,1)]),
                ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,1),(0,1,1)]),
                ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
                ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
            ]
            # Fix W signs for outer level too
            outer_W = [
                [(0,0,1),(1,1,1)],  # P1
                [(1,0,1),(1,1,1)],  # P2
                [(0,1,1),(1,1,1)],  # P3
                [(1,0,1),(0,0,1)],  # P4
                [(0,0,1),(0,1,1)],  # P5
                [(1,1,1)],          # P6
                [(0,0,1)],          # P7
            ]
            for s in range(7):
                for t in range(7):
                    col = s * 7 + t
                    ue, ve, we = outer[s]
                    for (bi, bj, c) in ue:
                        for i in range(2):
                            for j in range(2):
                                gi = bi * 2 + i
                                gj = bj * 2 + j
                                cc = U2[i * 2 + j][t]
                                if cc:
                                    U[gi * 4 + gj][col] += cc * c
                    for (bj, bk, c) in ve:
                        for j in range(2):
                            for k in range(2):
                                gj = bj * 2 + j
                                gk = bk * 2 + k
                                cc = V2[j * 2 + k][t]
                                if cc:
                                    V[gj * 4 + gk][col] += cc * c
                    for (bk, bi, c) in we:
                        for k in range(2):
                            for i in range(2):
                                gk = bk * 2 + k
                                gi = bi * 2 + i
                                cc = W2[k * 2 + i][t]
                                if cc:
                                    W[gk * 4 + gi][col] += cc * c
            return U, V, W

        def build_4x4x2():
            """4x4x2 via 4 blocks of 2x2x2 Strassen: rank 28."""
            U = [[0] * 28 for _ in range(16)]
            V = [[0] * 28 for _ in range(8)]
            W = [[0] * 28 for _ in range(8)]
            U2, V2, W2 = _strassen_2x2()
            # 4 blocks: (ar,ac,br,cr,ck) = (row_start_A, col_start_A, row_start_B, col_start_C, col_start_K)
            blocks = [(0,0,0,0,0), (0,2,2,0,0), (2,0,0,2,0), (2,2,2,2,0)]
            for bi, (ar, ac, br, cr, ck) in enumerate(blocks):
                base = bi * 7
                for t in range(7):
                    col = base + t
                    for i in range(2):
                        for j in range(2):
                            cc = U2[i * 2 + j][t]
                            if cc:
                                U[(ar + i) * 4 + (ac + j)][col] += cc
                    for j in range(2):
                        for k in range(2):
                            cc = V2[j * 2 + k][t]
                            if cc:
                                V[(br + j) * 2 + (ck + k)][col] += cc
                    for k in range(2):
                        for i in range(2):
                            cc = W2[k * 2 + i][t]
                            if cc:
                                W[(ck + k) * 4 + (cr + i)][col] += cc
            return U, V, W

        def build_4x1x4():
            """4x1x4 schoolbook: rank 16."""
            rank = 4 * 1 * 4
            U = [[0] * rank for _ in range(4)]
            V = [[0] * rank for _ in range(4)]
            W = [[0] * rank for _ in range(16)]
            c = 0
            for i in range(4):
                for k in range(4):
                    U[i][c] = 1
                    V[k][c] = 1
                    W[k * 4 + i][c] = 1
                    c += 1
            return U, V, W

        def build_4x1x2():
            """4x1x2 schoolbook: rank 8."""
            rank = 4 * 1 * 2
            U = [[0] * rank for _ in range(4)]
            V = [[0] * rank for _ in range(2)]
            W = [[0] * rank for _ in range(8)]
            c = 0
            for i in range(4):
                for k in range(2):
                    U[i][c] = 1
                    V[k][c] = 1
                    W[k * 4 + i][c] = 1
                    c += 1
            return U, V, W

        def combine(parts, n, m, p):
            total = sum(len(d[0][0]) for d in parts)
            U = [[0] * total for _ in range(n * m)]
            V = [[0] * total for _ in range(m * p)]
            W = [[0] * total for _ in range(p * n)]
            off = 0
            for (Us, Vs, Ws, (i0, j0, ii, jj, k0, kk)) in parts:
                nA, mA, nB = ii, jj, kk
                for r in range(nA * mA):
                    i = r // mA
                    j = r % mA
                    for c in range(len(Us[0])):
                        U[(i0 + i) * m + (j0 + j)][off + c] = Us[r][c]
                for r in range(mA * nB):
                    j = r // nB
                    k = r % nB
                    for c in range(len(Vs[0])):
                        V[(j0 + j) * p + (k0 + k)][off + c] = Vs[r][c]
                for r in range(nB * nA):
                    k = r // nA
                    i = r % nA
                    for c in range(len(Ws[0])):
                        W[(k0 + k) * n + (i0 + i)][off + c] = Ws[r][c]
                off += len(Us[0])
            return U, V, W

        # Build the additive decomposition
        Ua, Va, Wa = build_4x4x4()
        Ub, Vb, Wb = build_4x1x4()
        Uc, Vc, Wc = build_4x4x2()
        Ud, Vd, Wd = build_4x1x2()

        parts = [
            (Ua, Va, Wa, (0, 0, 4, 4, 0, 4)),   # 4x4 * 4x4 -> 4x4
            (Ub, Vb, Wb, (0, 4, 4, 1, 0, 4)),   # 4x1 * 1x4 -> 4x4
            (Uc, Vc, Wc, (0, 4, 4, 4, 4, 2)),   # 4x4 * 4x2 -> 4x2
            (Ud, Vd, Wd, (0, 4, 4, 1, 4, 2)),   # 4x1 * 1x2 -> 4x2
        ]

        U, V, W = combine(parts, n, m, p)
        rank = len(U[0])

        if _verify(U, V, W, n, m, p):
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank
        else:
            # Try to verify and debug
            pass

    except Exception:
        pass

    return best_result
# EVOLVE-BLOCK-END
