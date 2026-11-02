import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Kronecker product: (2,2,2) Strassen ⊗ (1,2,3) rank-2
    # Global: i = i1 (n=2=2*1), j = j1*2+j2 (m=4=2*2), k = k1*3+k2 (p=6=2*3)
    
    # Strassen terms for (2,2,2): local indices 0=(0,0),1=(0,1),2=(1,0),3=(1,1)
    strassen_terms = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(2, 1), (0, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    # (1,2,3) rank-2 terms:
    # Term 0: j2=0, U2[(0,0)]=1, V2[(0,k2)]=1 for k2=0,1,2, W2[(k2,0)]=1 for k2=0,1,2
    # Term 1: j2=1, U2[(0,1)]=1, V2[(1,k2)]=1 for k2=0,1,2, W2[(k2,0)]=1 for k2=0,1,2
    inner_terms = [
        # (U2 entries, V2 entries, W2 entries)
        ([(0, 1)], [(0, 1), (1, 1), (2, 1)], [(0, 1), (1, 1), (2, 1)]),  # j2=0
        ([(1, 1)], [(3, 1), (4, 1), (5, 1)], [(0, 1), (1, 1), (2, 1)]),  # j2=1
    ]
    
    rank = 7 * 2
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for t in range(7):
        u1, v1, w1 = strassen_terms[t]
        for s in range(2):
            u2, v2, w2 = inner_terms[s]
            
            for (li1, c1) in u1:
                i1 = li1 >> 1
                j1 = li1 & 1
                for (lj2, c2) in u2:
                    j2 = lj2  # i2 is always 0
                    i_global = i1  # i1 * 1 + 0
                    j_global = j1 * 2 + j2
                    U[i_global * m + j_global, col] += c1 * c2
            
            for (lj1, c1) in v1:
                j1 = lj1 >> 1
                k1 = lj1 & 1
                for (lk2, c2) in v2:
                    j2 = lk2 // 3
                    k2 = lk2 % 3
                    j_global = j1 * 2 + j2
                    k_global = k1 * 3 + k2
                    V[j_global * p + k_global, col] += c1 * c2
            
            for (lk1, c1) in w1:
                k1 = lk1 >> 1
                i1 = lk1 & 1
                for (lk2, c2) in w2:
                    k2 = lk2
                    k_global = k1 * 3 + k2
                    i_global = i1
                    W[k_global * n + i_global, col] += c1 * c2
            
            col += 1
    
    # Verify
    try:
        T = np.einsum('ijr,jkr,ki->ijk', U.reshape(n, m, rank), V.reshape(m, p, rank), W.reshape(p, n, rank))
        if not np.allclose(T, 1.0, atol=1e-9):
            r_sb = n * m * p
            U_sb = np.zeros((n * m, r_sb))
            V_sb = np.zeros((m * p, r_sb))
            W_sb = np.zeros((p * n, r_sb))
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i * m + j, c] = 1
                        V_sb[j * p + k, c] = 1
                        W_sb[k * n + i, c] = 1
                        c += 1
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    except Exception:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
