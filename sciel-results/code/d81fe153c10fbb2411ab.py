import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def _get_strassen_2x2():
    """
    Returns U, V, W for 2x2 matrix multiplication with rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing:
    U[i*2+j, r] corresponds to A[i,j]
    V[j*2+k, r] corresponds to B[j,k]
    W[k*2+i, r] corresponds to C[i,k] -> W index is k*2+i (col * n + row)
    """
    n = 2
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # M2 = (A10+A11)B00 -> C10 += 1
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[1,1] = 1  # C10: i=1, k=0 -> 0*2+1 = 1
    
    # M3 = A00(B01-B11) -> C01 += 1, C11 -= 1
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[2,2] = 1  # C01: i=0, k=1 -> 1*2+0 = 2
    W[3,2] = -1 # C11: i=1, k=1 -> 1*2+1 = 3
    
    # M4 = A11(B10-B00) -> C00 += 1, C10 += 1
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1  # C00
    W[1,3] = 1  # C10
    
    # M5 = (A00+A01)B11 -> C00 -= 1, C01 += 1
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1 # C00
    W[2,4] = 1  # C01
    
    # M6 = (A10-A00)(B00+B01) -> C11 += 1
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1  # C11
    
    # M7 = (A01-A11)(B10+B11) -> C00 += 1, C11 -= 1
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1  # C00
    W[3,6] = -1 # C11
    
    return U, V, W

def _strassen_4x4():
    n = 4
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    U2, V2, W2 = _get_strassen_2x2()
    
    # Block structure:
    # A is 4x4. Blocks A00, A01, A10, A11 are 2x2.
    # A[bi*2+a, bj*2+b] is element (a,b) of block (bi,bj).
    # U index for A[i,j] is i*4+j.
    # V index for B[j,k] is j*4+k.
    # W index for C[i,k] is k*4+i.
    
    # Strassen block terms for 2x2 block multiplication:
    # Each term rb (0..6) defines:
    # P_blocks: list of (bi, bj, coeff)
    # Q_blocks: list of (bi, bj, coeff)
    # C_blocks: list of (ci, cj, coeff)
    
    # M1: P=A00+A11, Q=B00+B11, C: C00+1, C11+1
    # M2: P=A10+A11, Q=B00,     C: C10+1
    # M3: P=A00,     Q=B01-B11, C: C01+1, C11-1
    # M4: P=A11,     Q=B10-B00, C: C00+1, C10+1
    # M5: P=A00+A01, Q=B11,     C: C00-1, C01+1
    # M6: P=A10-A00, Q=B00+B01, C: C11+1
    # M7: P=A01-A11, Q=B10+B11, C: C00+1, C11-1
    
    terms = [
        [{'P': [(0,0,1), (1,1,1)], 'Q': [(0,0,1), (1,1,1)], 'C': [(0,0,1), (1,1,1)]}],
        [{'P': [(1,0,1), (1,1,1)], 'Q': [(0,0,1)], 'C': [(1,0,1)]}],
        [{'P': [(0,0,1)], 'Q': [(0,1,1), (1,1,-1)], 'C': [(0,1,1), (1,1,-1)]}],
        [{'P': [(1,1,1)], 'Q': [(1,0,1), (0,0,-1)], 'C': [(0,0,1), (1,0,1)]}],
        [{'P': [(0,0,1), (0,1,1)], 'Q': [(1,1,1)], 'C': [(0,0,-1), (0,1,1)]}],
        [{'P': [(1,0,1), (0,0,-1)], 'Q': [(0,0,1), (0,1,1)], 'C': [(1,1,1)]}],
        [{'P': [(0,1,1), (1,1,-1)], 'Q': [(1,0,1), (1,1,1)], 'C': [(0,0,1), (1,1,-1)]}]
    ]
    
    for rb, term_data in enumerate(terms):
        t = term_data[0]
        P_list = t['P']
        Q_list = t['Q']
        C_list = t['C']
        
        for rs in range(7):
            r = rb * 7 + rs
            
            # Fill U
            for bi, bj, pc in P_list:
                for a in range(2):
                    for b in range(2):
                        coeff = pc * U2[a*2+b, rs]
                        if coeff != 0:
                            ai = bi * 2 + a
                            aj = bj * 2 + b
                            U[ai*4+aj, r] += coeff
            
            # Fill V
            for bi, bj, qc in Q_list:
                for a in range(2):
                    for b in range(2):
                        coeff = qc * V2[a*2+b, rs]
                        if coeff != 0:
                            vi = bi * 2 + a
                            vj = bj * 2 + b
                            V[vi*4+vj, r] += coeff
            
            # Fill W
            for ci, cj, cc in C_list:
                for a in range(2):
                    for b in range(2):
                        coeff = cc * W2[a*2+b, rs]
                        if coeff != 0:
                            crow = ci * 2 + a
                            ccol = cj * 2 + b
                            W[ccol*4+crow, r] += coeff

    return U, V, W

def _verify(U, V, W, n, m, p):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = 100.0
    
    # 1. Try Strassen 4x4 (Rank 49)
    try:
        U, V, W = _strassen_4x4()
        if _verify(U, V, W, 4, 4, 4):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    except Exception:
        pass
    
    if time.time() > deadline:
        pass

    # 2. Fallback: Schoolbook (Rank 64)
    n, m, p = 4, 4, 4
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
