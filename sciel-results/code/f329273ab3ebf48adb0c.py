"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    if time.time() < deadline:
        result = _try_rank_26(n, m, p)
        if result is not None:
            return result
            
    return _rank_27_fallback(n, m, p)

def _verify_decomposition(U, V, W, n, m, p):
    nm, mp, pn = n*m, m*p, p*n
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T_target = np.zeros((nm, mp, pn), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1
                
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    return np.allclose(T_recon, T_target, atol=1e-9)

def _try_rank_26(n, m, p):
    """
    Rank-26 decomposition via 2+1 block partitioning.
    
    A = [[A2, a], [aT, a3]]  where A2 is 2x2, a is 2x1, aT is 1x2, a3 is scalar
    B = [[B2, b], [bT, b3]]  similarly
    C = A*B = [[A2*B2 + a*bT, A2*b + a*b3],
               [aT*B2 + a3*bT, aT*b + a3*b3]]
    
    Index mapping (i*3+j for A,B; k*3+i for C):
    A2: {(0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4}
    a:  {(0,2)->2, (1,2)->5}
    aT: {(2,0)->6, (2,1)->7}
    a3: {(2,2)->8}
    (Same for B)
    C2: {(0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4}  [C[k,i] at k*3+i]
    c:  {(0,2)->2, (1,2)->5}
    cT: {(2,0)->6, (2,1)->7}
    c3: {(2,2)->8}
    """
    nm, mp, pn = n*m, m*p, p*n
    rank = 26
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    
    # Block 1: A2 * B2 -> C2 (2x2 * 2x2 = 2x2), Rank 7 (Strassen)
    # Using standard Strassen formulas verified:
    # M1=(A00+A11)(B00+B11), M2=(A10+A11)B00, M3=A00(B10-B00)
    # M4=A11(B01+B11), M5=(A00+A01)B11, M6=(A10-A00)(B00+B01)
    # M7=(A01-A11)(B10+B11)
    # C00=M1+M2-M3-M5, C01=M4+M5-M7, C10=M2+M3-M6, C11=M1-M4+M6+M7
    
    strassen_terms = [
        ({0:1, 4:1}, {0:1, 4:1}, {0:1, 4:1}),   # M1: C00+=1, C11+=1
        ({3:1, 4:1}, {0:1},     {0:1, 3:1}),     # M2: C00+=1, C10+=1
        ({0:1},      {3:1, 0:-1}, {0:-1, 3:1}),  # M3: C00-=1, C10+=1
        ({4:1},      {1:1, 4:1}, {1:1, 4:-1}),   # M4: C01+=1, C11-=1
        ({0:1, 1:1}, {4:1},     {0:-1, 1:1}),    # M5: C00-=1, C01+=1
        ({3:1, 0:-1},{0:1, 1:1}, {3:-1, 4:1}),   # M6: C10-=1, C11+=1
        ({1:1, 4:-1},{3:1, 4:1}, {1:-1, 4:1}),   # M7: C01-=1, C11+=1
    ]
    
    for ud, vd, wd in strassen_terms:
        for idx, val in ud.items():
            U[idx, col] += val
        for idx, val in vd.items():
            V[idx, col] += val
        for idx, val in wd.items():
            W[idx, col] += val
        col += 1
    
    # Block 2: a * bT -> adds to C2 (2x1 * 1x2 = 2x2), Rank 4
    # C2[0,0]+=a0*b0, C2[0,1]+=a0*b1, C2[1,0]+=a1*b0, C2[1,1]+=a1*b1
    terms = [
        (2, 6, 0), (2, 7, 1), (5, 6, 3), (5, 7, 4)
    ]
    for u_idx, v_idx, w_idx in terms:
        U[u_idx, col] = 1; V[v_idx, col] = 1; W[w_idx, col] = 1; col += 1
    
    # Block 3: A2 * b -> c (2x2 * 2x1 = 2x1), Rank 4
    # c[0]=A00*b0+A01*b1, c[1]=A10*b0+A11*b1
    terms = [
        (0, 2, 2), (1, 5, 2), (3, 2, 5), (4, 5, 5)
    ]
    for u_idx, v_idx, w_idx in terms:
        U[u_idx, col] = 1; V[v_idx, col] = 1; W[w_idx, col] = 1; col += 1
    
    # Block 4: a * b3 -> adds to c (2x1 * scalar = 2x1), Rank 2
    # c[0]+=a0*b3, c[1]+=a1*b3
    terms = [
        (2, 8, 2), (5, 8, 5)
    ]
    for u_idx, v_idx, w_idx in terms:
        U[u_idx, col] = 1; V[v_idx, col] = 1; W[w_idx, col] = 1; col += 1
    
    # Block 5: aT * B2 -> cT (1x2 * 2x2 = 1x2), Rank 4
    # cT[0]=a0*B00+a1*B10, cT[1]=a0*B01+a1*B11
    terms = [
        (6, 0, 6), (7, 3, 6), (6, 1, 7), (7, 4, 7)
    ]
    for u_idx, v_idx, w_idx in terms:
        U[u_idx, col] = 1; V[v_idx, col] = 1; W[w_idx, col] = 1; col += 1
    
    # Block 6: a3 * bT -> adds to cT (scalar * 1x2 = 1x2), Rank 2
    # cT[0]+=a3*b0, cT[1]+=a3*b1
    terms = [
        (8, 6, 6), (8, 7, 7)
    ]
    for u_idx, v_idx, w_idx in terms:
        U[u_idx, col] = 1; V[v_idx, col] = 1; W[w_idx, col] = 1; col += 1
    
    # Block 7: aT * b -> c3 (1x2 * 2x1 = scalar), Rank 2
    # c3=a0*b0+a1*b1
    terms = [
        (6, 2, 8), (7, 5, 8)
    ]
    for u_idx, v_idx, w_idx in terms:
        U[u_idx, col] = 1; V[v_idx, col] = 1; W[w_idx, col] = 1; col += 1
    
    # Block 8: a3 * b3 -> c3 (scalar * scalar = scalar), Rank 1
    U[8, col] = 1; V[8, col] = 1; W[8, col] = 1; col += 1
    
    # Total: 7+4+4+2+4+2+2+1 = 26
    if col != rank:
        return None
    
    if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def _rank_27_fallback(n, m, p):
    nm, mp, pn = n*m, m*p, p*n
    rank = 27
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][r] = 1
                V[b][r] = 1
                W[c][r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
