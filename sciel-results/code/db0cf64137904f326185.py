"""Tensor decomposition for 4x4 x 4x6 matrix multiplication."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time

def _verify(U, V, W, n, m, p, rank):
    """Check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    u = U[i*m+j][r]
                    if u == 0:
                        continue
                    v = V[j*p+k][r]
                    if v == 0:
                        continue
                    s += u * v * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _block_strassen(n, m, p, bs=2):
    """Block Strassen: split into bs x bs blocks, use Strassen-7 per 2x2x2."""
    na = n // bs
    nc = m // bs
    nb = p // bs
    rank = na * nc * nb * 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    strassen = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1}),
        ({(0,0):1}, {(1,0):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1}, {(0,1):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1,(1,1):-1}),
    ]
    col = 0
    for a in range(na):
        for c in range(nc):
            for b in range(nb):
                for (ac, bc, cc) in strassen:
                    for (ar, acol), coef in ac.items():
                        gi = a*bs + ar
                        gj = c*bs + acol
                        U[gi*m + gj][col] += coef
                    for (br, bcol), coef in bc.items():
                        gi = c*bs + br
                        gj = b*bs + bcol
                        V[gi*p + gj][col] += coef
                    for (cr, ccol), coef in cc.items():
                        gi = b*bs + ccol
                        gj = a*bs + cr
                        W[gi*n + gj][col] += coef
                    col += 1
    assert col == rank
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 140
    n, m, p = DIMENSIONS

    # Schoolbook fallback
    rank_school = n*m*p
    U_s = [[0]*rank_school for _ in range(n*m)]
    V_s = [[0]*rank_school for _ in range(m*p)]
    W_s = [[0]*rank_school for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j][col] = 1
                V_s[j*p+k][col] = 1
                W_s[k*n+i][col] = 1
                col += 1
    best = {"U": U_s, "V": V_s, "W": W_s, "rank": rank_school}

    # Block Strassen rank 84
    U, V, W, rank = _block_strassen(n, m, p, bs=2)
    if _verify(U, V, W, n, m, p, rank):
        best = {"U": U, "V": V, "W": W, "rank": rank}

    if time.time() > deadline:
        return {"U": best["U"], "V": best["V"], "W": best["W"]}

    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
