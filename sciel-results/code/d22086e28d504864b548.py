"""Search for a low-rank exact tensor decomposition of (2,2,8) matrix multiplication."""
import time
from fractions import Fraction
import numpy as np

N, M, P = 2, 2, 8
NM, MP, PN = N * M, M * P, P * N  # 4, 16, 16

def _build_strassen_222():
    """Return U, V, W for M_{2,2,2} (rank 7) with standard Strassen formula."""
    # Strassen terms: (u, v, w) where u, v, w are length-4 vectors (row-major 2x2)
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    for t in range(7):
        u, v, w = terms[t]
        for i in range(4):
            U[i][t] = u[i]
            V[i][t] = v[i]
            W[i][t] = w[i]
    return U, V, W

def _build_strassen_228():
    """
    Build rank-28 decomposition for M_{2,2,8} by applying 4 independent
    Strassen M_{2,2,2} blocks to the 4 column-pairs of the 2x8 matrix B.
    
    A is 2x2, B is 2x8, C is 2x8.
    Split B into 4 blocks of 2x2 columns: B[:, 0:2], B[:, 2:4], B[:, 4:6], B[:, 6:8]
    C is computed block-wise: C[:, 2b:2b+2] = A * B[:, 2b:2b+2]
    
    Indexing:
      U: rows indexed by (i, j) in A, i.e., a = i*M + j, i in [0,2), j in [0,2)
      V: rows indexed by (j, k) in B, i.e., b = j*P + k, j in [0,2), k in [0,8)
      W: rows indexed by (k, i) in C, i.e., c = k*N + i, k in [0,8), i in [0,2)
    """
    U_local, V_local, W_local = _build_strassen_222()
    
    U = [[0] * 28 for _ in range(NM)]  # 4 x 28
    V = [[0] * 28 for _ in range(MP)]  # 16 x 28
    W = [[0] * 28 for _ in range(PN)]  # 16 x 28
    
    for block in range(4):
        for t in range(7):
            col = block * 7 + t
            # U is same for all blocks (A is 2x2)
            for a in range(4):
                U[a][col] = U_local[a][t]
            
            # V: for block, local k is 0..1, global k = 2*block + local_k
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * block + k_local
                    b_idx = j * P + k_global  # j*8 + k_global
                    v_idx = j * 2 + k_local   # local index in 2x2
                    V[b_idx][col] = V_local[v_idx][t]
            
            # W: for block, local k is 0..1, global k = 2*block + local_k
            for i in range(2):
                for k_local in range(2):
                    k_global = 2 * block + k_local
                    c_idx = k_global * N + i  # k_global*2 + i
                    w_idx = k_local * 2 + i   # local index in 2x2 output (k_local, i)
                    W[c_idx][col] = W_local[w_idx][t]
    
    return U, V, W

def verify_decomp(U, V, W, n=N, m=M, p=P):
    """Verify exact decomposition. U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank."""
    nm, mp, pn = n * m, m * p, p * n
    rank = len(U[0]) if U else 0
    if len(U) != nm or len(V) != mp or len(W) != pn:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
    
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i * m + j, j * p + k, k * n + i))
    
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                expected = 1 if (a, b, c) in valid else 0
                if s != expected:
                    return False
    return True

def schoolbook():
    """Rank-32 schoolbook decomposition."""
    rank = N * M * P
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    return U, V, W

def construct():
    """Entry point: return the best decomposition found."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try rank 28 Strassen decomposition
    try:
        U, V, W = _build_strassen_228()
        if verify_decomp(U, V, W):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook rank 32
    U, V, W = schoolbook()
    return {"U": U, "V": V, "W": W}
