import time
import sys
from fractions import Fraction

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
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = 0
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        u_val = u_row[c]
                        v_val = v_row[c]
                        w_val = w_row[c]
                        if u_val == 0 or v_val == 0 or w_val == 0:
                            continue
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i*2 + a]
                        v_val = S_V[r][a*2 + k]
                        w_val = S_W[r][k*2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Build 44-term candidate
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Precompute Strassen index mappings for speed
    strassen_U = []  # list of (u_idx, val) for each r
    strassen_V = []  # list of (v_idx_base, val) for each r (base offset for k0)
    strassen_W = []  # list of (w_idx_base, val) for each r (base offset for k0)
    for r in range(7):
        su = []
        for li in range(2):
            for la in range(2):
                u_val = S_U[r][li*2 + la]
                if u_val != 0:
                    su.append((li * m + la, u_val))
        strassen_U.append(su)
        
        sv = []
        for la in range(2):
            for lk in range(2):
                v_val = S_V[r][la*2 + lk]
                if v_val != 0:
                    sv.append((la * p + lk, v_val))  # base index, add k0*n later
        strassen_V.append(sv)
        
        sw = []
        for lk in range(2):
            for li in range(2):
                w_val = S_W[r][lk*2 + li]
                if w_val != 0:
                    sw.append((lk * n + li, w_val))  # base index, add k0*n later
        strassen_W.append(sw)
    
    col = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for u_idx, u_val in strassen_U[r]:
                U[u_idx][col] += u_val
            for v_idx_base, v_val in strassen_V[r]:
                V[v_idx_base + k0][col] += v_val
            for w_idx_base, w_val in strassen_W[r]:
                W[w_idx_base + k0 * n][col] += w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}
    
    if _validate(candidate):
        best_candidate = candidate
    else:
        return _schoolbook()
        
    return best_candidate
