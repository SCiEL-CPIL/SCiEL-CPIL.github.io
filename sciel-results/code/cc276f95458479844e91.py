import time
import sys

DIMENSIONS = [2, 3, 8]

def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        from fractions import Fraction
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        acc += S_U[r][i * 2 + a] * S_V[r][a * 2 + k] * S_W[r][k * 2 + i]
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Try rank-40 solution: 3 blocks of 2x2x3 (rank 8 each) + 2x1x1x8 (rank 16)
    # 2x2x3 decomposition with rank 8
    # C[i,k] = sum_a A[i,a] * B[a,k] for i in {0,1}, a in {0,1}, k in {0,1,2}
    # Use 8 terms
    
    # Let me construct a rank-8 decomposition for 2x2x3
    # Terms: (u, v, w) where u is 2x2, v is 2x3, w is 3x2
    # We need sum_r u_r[i,a] * v_r[a,k] * w_r[k,i] = delta_{i,a,k}
    
    # Try: 4 terms for the "diagonal" part + 4 terms for Strassen-like corrections
    # Actually, let me just use schoolbook for 2x2x3 (12 terms) and see if total is better
    # 3 blocks * 12 + 16 = 52, which is worse than 44
    
    # So 2x2x3 with rank 8 would give 3*8 + 16 = 40, which is better
    # Let me find a rank-8 decomposition for 2x2x3
    
    # I'll use a known result: 2x2x3 has rank 8
    # Construct it manually
    
    # For 2x2x3: A is 2x2, B is 2x3, C is 2x3
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    # 8-term decomposition:
    # M1: u=(1,0;0,0), v=(1,0,0;0,0,0), w=(1,0;0,0) -> C[0,0] += A[0,0]*B[0,0]
    # M2: u=(0,0;0,1), v=(0,0,0;1,0,0), w=(0,1;0,0) -> C[1,0] += A[1,1]*B[1,0]
    # M3: u=(0,1;0,0), v=(0,0,0;0,1,0), w=(0,0;0,1) -> C[0,1] += A[0,1]*B[1,1]
    # M4: u=(0,0;1,0), v=(0,0,0;0,0,1), w=(0,0;1,0) -> C[1,2] += A[1,0]*B[1,2]
    # M5: u=(1,1;0,0), v=(1,0,0;0,1,0), w=(1,0;0,0) -> C[0,0] += A[0,0]*B[0,0] + A[0,1]*B[0,1]... no
    
    # This is getting complicated. Let me just return the rank-44 solution.
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    if S_U[r][li * 2 + la] != 0:
                        U[li * m + la][col] = S_U[r][li * 2 + la]
            for la in range(2):
                for lk in range(2):
                    if S_V[r][la * 2 + lk] != 0:
                        V[la * p + (k0 + lk)][col] = S_V[r][la * 2 + lk]
            for lk in range(2):
                for li in range(2):
                    if S_W[r][lk * 2 + li] != 0:
                        W[(k0 + lk) * n + li][col] = S_W[r][lk * 2 + li]
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    for i in range(2):
        for k in range(8):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}
    if _validate(candidate):
        return candidate
    return _schoolbook()
