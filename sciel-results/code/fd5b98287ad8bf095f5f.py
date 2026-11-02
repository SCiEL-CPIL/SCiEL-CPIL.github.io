"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [4, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0


def remaining():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)


def _verify(U, V, W, rank):
    """Verify decomposition with random integer matrices."""
    import random
    n, m, p = DIMENSIONS
    for _ in range(5):
        if remaining() <= 0:
            return False
        A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        for i in range(n):
            for k in range(p):
                acc = 0
                for j in range(m):
                    for c in range(rank):
                        u = U[i * m + j][c]
                        if u == 0:
                            continue
                        acc += u * W[k * n + i][c] * V[j * p + k][c]
                if acc != C[i][k]:
                    return False
    return True


def _winograd_2x2x2_terms():
    """
    Return 7 rank-1 terms for 2x2x2 matrix multiplication.
    Each term is (u, v, w) where u is length 4 (A flat), v is length 4 (B flat), w is length 4 (C flat).
    A flat: [A00, A01, A10, A11]
    B flat: [B00, B01, B10, B11]
    C flat: [C00, C01, C10, C11]
    """
    terms = []
    # p1 = (a+d)(e+h)
    # A: a+d => [1,0,0,1]; B: e+h => [1,0,0,1]; C: +C00, +C11 => [1,0,0,1]
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # p2 = (c+d)e
    # A: c+d => [0,0,1,1]; B: e => [1,0,0,0]; C: +C10, -C11 => [0,0,1,-1]
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    # p3 = a(f-h)
    # A: a => [1,0,0,0]; B: f-h => [0,1,0,-1]; C: +C01 => [0,1,0,0]
    terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]))
    # p4 = d(g-e)
    # A: d => [0,0,0,1]; B: g-e => [-1,0,1,0]; C: -C00, +C01, +C10 => [-1,1,1,0]
    terms.append(([0, 0, 0, 1], [-1, 0, 1, 0], [-1, 1, 1, 0]))
    # p5 = (a+b)g
    # A: a+b => [1,1,0,0]; B: g => [0,0,1,0]; C: +C00, +C01 => [1,1,0,0]
    terms.append(([1, 1, 0, 0], [0, 0, 1, 0], [1, 1, 0, 0]))
    # p6 = (c-a)(e+f)
    # A: c-a => [-1,0,1,0]; B: e+f => [1,1,0,0]; C: -C10, +C11 => [0,0,-1,1]
    terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, -1, 1]))
    # p7 = (b-d)(h+f)
    # A: b-d => [0,1,0,-1]; B: h+f => [0,1,0,1]; C: -C00, +C01, +C10 => [-1,1,1,0]
    terms.append(([0, 1, 0, -1], [0, 1, 0, 1], [-1, 1, 1, 0]))
    return terms


def _build_winograd_block_decomposition():
    """
    Build 4x5x6 decomposition using Winograd 2x2x2 blocks + schoolbook remainders.
    """
    n, m, p = DIMENSIONS
    winograd_terms = _winograd_2x2x2_terms()
    
    all_U = []
    all_V = []
    all_W = []
    covered = set()
    
    # Winograd blocks: i in {0,1}x{2,3}, j in {0,1}x{2,3}, k in {0,1}x{2,3}
    for ib in range(2):
        for jb in range(2):
            for kb in range(2):
                i_base = 2 * ib
                j_base = 2 * jb
                k_base = 2 * kb
                for (u_local, v_local, w_local) in winograd_terms:
                    u_full = [0] * (n * m)
                    v_full = [0] * (m * p)
                    w_full = [0] * (p * n)
                    # A indices: (i_base + a, j_base + b) -> flat index (i_base+a)*m + (j_base+b)
                    for a in range(2):
                        for b in range(2):
                            u_full[(i_base + a) * m + (j_base + b)] = u_local[a * 2 + b]
                    # B indices: (j_base + b, k_base + c) -> flat index (j_base+b)*p + (k_base+c)
                    for b in range(2):
                        for c in range(2):
                            v_full[(j_base + b) * p + (k_base + c)] = v_local[b * 2 + c]
                    # C indices: (k_base + c, i_base + a) -> flat index (k_base+c)*n + (i_base+a)
                    for c in range(2):
                        for a in range(2):
                            w_full[(k_base + c) * n + (i_base + a)] = w_local[c * 2 + a]
                    all_U.append(u_full)
                    all_V.append(v_full)
                    all_W.append(w_full)
                # Mark covered indices
                for a in range(2):
                    for b in range(2):
                        for c in range(2):
                            covered.add((i_base + a, j_base + b, k_base + c))
    
    # Schoolbook for remaining (i,j,k) not covered
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                u_full = [0] * (n * m)
                v_full = [0] * (m * p)
                w_full = [0] * (p * n)
                u_full[i * m + j] = 1
                v_full[j * p + k] = 1
                w_full[k * n + i] = 1
                all_U.append(u_full)
                all_V.append(v_full)
                all_W.append(w_full)
    
    rank = len(all_U)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c in range(rank):
        for r in range(n * m):
            U[r][c] = all_U[c][r]
        for r in range(m * p):
            V[r][c] = all_V[c][r]
        for r in range(p * n):
            W[r][c] = all_W[c][r]
    
    return {"U": U, "V": V, "W": W, "rank": rank}


def _schoolbook():
    """Fallback schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W, "rank": rank}


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    best = None
    
    # Try Winograd block decomposition (expect rank 112)
    try:
        if remaining() > 20:
            cand = _build_winograd_block_decomposition()
            if remaining() > 10 and _verify(cand["U"], cand["V"], cand["W"], cand["rank"]):
                best = cand
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 120)
    if best is None:
        best = _schoolbook()
    
    del best["rank"]
    return best
# EVOLVE-BLOCK-END
