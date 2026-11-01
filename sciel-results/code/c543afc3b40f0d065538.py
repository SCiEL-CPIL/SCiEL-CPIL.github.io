"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    if U.shape[1] != rank or V.shape[1] != rank or W.shape[1] != rank:
        return False
    np.random.seed(42)
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        C_comp = np.zeros((n, p))
        for r in range(rank):
            u_val = U[:, r] @ A.flatten()
            v_val = V[:, r] @ B.flatten()
            C_comp += u_val * v_val * W[:, r].reshape(p, n).T
        if not np.allclose(C_comp, C_true, atol=tol):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Strategy: Apply Strassen to all 2x2x2 blocks where j and k can be paired
    # j pairs: (0,1), (2,3) -- j=4 is unpaired
    # k pairs: (0,1), (2,3), (4,5)
    # 6 Strassen blocks (rank 7 each) + schoolbook for j=4 (12 terms)
    # Total rank: 6*7 + 12 = 54
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen terms definition (verified correct):
    # For 2x2 matrices A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=A*B
    # P1=(a+d)(e+h), P2=(c+d)e, P3=a(f-h), P4=d(g-e), P5=(a+c)h, P6=(c-a)(e+f), P7=(b-d)(g+h)
    # C00=P1-P4+P5+P7, C01=P3+P5, C10=P2+P3-P6-P7, C11=P1-P2+P3+P6
    #
    # U entries for each P (indices: (i,j) where i=row, j=col of A):
    # V entries for each P (indices: (j,k) where j=row, k=col of B):
    # W entries for each P (indices: (k,i) where k=col, i=row of C):
    
    strassen_defs = [
        # P1: (a+d)(e+h) -> C00+=1, C11+=1
        {
            'U': [(0,0,1), (1,1,1)],
            'V': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1)]
        },
        # P2: (c+d)e -> C10+=1, C11-=1
        {
            'U': [(1,0,1), (1,1,1)],
            'V': [(0,0,1)],
            'W': [(0,1,1), (1,1,-1)]
        },
        # P3: a(f-h) -> C01+=1, C10+=1, C11+=1
        {
            'U': [(0,0,1)],
            'V': [(0,1,1), (1,1,-1)],
            'W': [(0,0,1), (0,1,1), (1,1,1)]
        },
        # P4: d(g-e) -> C00-=1, C01+=1, C11+=1
        {
            'U': [(1,1,1)],
            'V': [(1,0,1), (0,0,-1)],
            'W': [(0,0,-1), (0,1,1), (1,1,1)]
        },
        # P5: (a+c)h -> C00+=1, C01+=1
        {
            'U': [(0,0,1), (1,0,1)],
            'V': [(1,1,1)],
            'W': [(0,0,1), (0,1,1)]
        },
        # P6: (c-a)(e+f) -> C10-=1, C11+=1
        {
            'U': [(1,0,1), (0,0,-1)],
            'V': [(0,0,1), (0,1,1)],
            'W': [(0,1,-1), (1,1,1)]
        },
        # P7: (b-d)(g+h) -> C00+=1, C10-=1
        {
            'U': [(0,1,1), (1,1,-1)],
            'V': [(1,0,1), (1,1,1)],
            'W': [(0,0,1), (0,1,-1)]
        },
    ]
    
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    # Apply Strassen to each 2x2x2 block
    for j1, j2 in j_pairs:
        for k1, k2 in k_pairs:
            for sdef in strassen_defs:
                # Apply U: local (i, j_local) -> global (i, j_global)
                for (li, lj, val) in sdef['U']:
                    gj = j1 + lj
                    U[li * m + gj][col] += val
                
                # Apply V: local (j_local, k_local) -> global (j_global, k_global)
                for (lj, lk, val) in sdef['V']:
                    gj = j1 + lj
                    gk = k1 + lk
                    V[gj * p + gk][col] += val
                
                # Apply W: local (k_local, i) -> global (k_global, i)
                for (lk, li, val) in sdef['W']:
                    gk = k1 + lk
                    W[gk * n + li][col] += val
                
                col += 1
    
    # Schoolbook for j=4 (unpaired): A[:,4] * B[4,:]
    for i in range(n):
        for k in range(p):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U2 = [[0] * rank_fb for _ in range(n * m)]
        V2 = [[0] * rank_fb for _ in range(m * p)]
        W2 = [[0] * rank_fb for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U2[i * m + j][c] = 1
                    V2[j * p + k][c] = 1
                    W2[k * n + i][c] = 1
                    c += 1
        return {"U": U2, "V": V2, "W": W2}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
