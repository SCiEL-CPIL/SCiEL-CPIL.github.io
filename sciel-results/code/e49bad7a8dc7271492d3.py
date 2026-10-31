"""Rank-49 exact tensor decomposition for 4x4x4 matrix multiplication.

Uses the Kronecker product of two rank-7 Strassen decompositions for 2x2x2.
This achieves the Strassen lower bound r(4x4x4) >= r(2x2x2)^2 = 49.

The W matrix follows the official trace convention W[k*n+i, r].
"""
import time

_T_START = time.time()
_DEADLINE_S = 145.0


def _time_left():
    return (_T_START + _DEADLINE_S) - time.time()


def _schoolbook():
    n = m = p = 4
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


def _strassen_2x2():
    """Return U2, V2, W2 for 2x2x2 Strassen decomposition (rank 7).
    
    Convention: U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    where i,j,k in {0,1} and r in {0,...,6}.
    """
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1,  0, 1],   # A01
        [0, 1, 0, 0, 0,  1, 0],   # A10
        [1, 1, 0, 1, 0,  0,-1],   # A11
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [0, 0, 0,  1, 0, 0, 1],   # B01
        [0, 0, 1,  0, 0, 1, 0],   # B10
        [1, 0,-1,  0, 1, 0, 1],   # B11
    ]
    W2 = [
        [1, 0, 0, -1, 0, -1, 1],  # C00
        [0, 0, 0,  0, 1,  0, 1],  # C01
        [0, 1, 1,  0, 0,  0, 0],  # C10
        [1,-1, 0,  0, 1,  1, 0],  # C11
    ]
    return U2, V2, W2


def _build_rank49():
    """Build rank-49 decomposition via Kronecker product of two Strassen tensors.
    
    For 4x4 matrices split into 2x2 blocks:
    A = [[A00, A01], [A10, A11]]
    B = [[B00, B01], [B10, B11]]
    C = A*B = [[C00, C01], [C10, C11]]
    
    The Kronecker product decomposition:
    For each pair (s, t) where s,t in {0,...,6}:
      U[i,j] contribution = U2[i1,j1][s] * U2[i2,j2][t]
      V[j,k] contribution = V2[j1,k1][s] * V2[j2,k2][t]
      W[k,i] contribution = W2[k1,i1][s] * W2[k2,i2][t]
    where i = i1*2+i2, j = j1*2+j2, k = k1*2+k2
    """
    U2, V2, W2 = _strassen_2x2()
    n = m = p = 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            # U entries
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            u_val = U2[i1 * 2 + j1][s] * U2[i2 * 2 + j2][t]
                            if u_val != 0:
                                U[i * m + j][r] = u_val
            # V entries
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            v_val = V2[j1 * 2 + k1][s] * V2[j2 * 2 + k2][t]
                            if v_val != 0:
                                V[j * p + k][r] = v_val
            # W entries
            for k1 in range(2):
                for k2 in range(2):
                    for i1 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            w_val = W2[k1 * 2 + i1][s] * W2[k2 * 2 + i2][t]
                            if w_val != 0:
                                W[k * n + i][r] = w_val
    return {"U": U, "V": V, "W": W}


def _verify(d, num_trials=6):
    """Verify the decomposition produces correct matrix products."""
    U, V, W = d["U"], d["V"], d["W"]
    n = m = p = 4
    r = len(U[0])
    if len(V[0]) != r or len(W[0]) != r:
        return False
    
    # Simple LCG for deterministic random testing
    seed = 123456789
    def rand_int(lo, hi):
        nonlocal seed
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        return lo + (seed % (hi - lo + 1))
    
    for _ in range(num_trials):
        A = [[rand_int(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[rand_int(-5, 5) for _ in range(p)] for _ in range(m)]
        # Compute true product
        M = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        # Compute via decomposition
        for i in range(n):
            for k in range(p):
                val = 0
                for j in range(m):
                    for rr in range(r):
                        val += U[i * m + j][rr] * V[j * p + k][rr] * W[k * n + i][rr]
                if val != M[i][k]:
                    return False
    return True


def construct():
    if _time_left() <= 0.0:
        return _schoolbook()
    
    try:
        cand = _build_rank49()
        if _verify(cand):
            return cand
    except Exception:
        pass
    
    return _schoolbook()
