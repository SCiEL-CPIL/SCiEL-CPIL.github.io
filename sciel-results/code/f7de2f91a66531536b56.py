import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    start_time = time.time()
    deadline = start_time + 140

    def _strassen_2x2():
        """Return Strassen rank-7 decomposition for 2x2*2x2.
        U: (4,7) rows indexed i*m+j for A[i,j]
        V: (4,7) rows indexed j*2+k for B[j,k] (local 2x2)
        W: (4,7) rows indexed k*2+i for C[k,i] (local 2x2, trace convention)
        """
        U = np.zeros((4, 7), dtype=np.int64)
        V = np.zeros((4, 7), dtype=np.int64)
        W = np.zeros((4, 7), dtype=np.int64)
        
        # M1 = (a+d)(e+h)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1  # C[0,0]
        W[3, 0] = 1  # C[1,1]
        
        # M2 = (c+d)e
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1  # C[1,0]
        W[3, 1] = -1 # C[1,1]
        
        # M3 = a(f-h)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1  # C[0,1]
        W[3, 2] = 1  # C[1,1]
        
        # M4 = d(g-e)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1  # C[0,0]
        W[2, 3] = 1  # C[1,0]
        
        # M5 = (a+b)h
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1 # C[0,0]
        W[1, 4] = 1  # C[0,1]
        
        # M6 = (c-a)(e+f)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1  # C[1,1]
        
        # M7 = (b-d)(g+h)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1  # C[0,0]
        
        return U, V, W

    def verify(decomp):
        U = np.array(decomp["U"], dtype=float)
        V = np.array(decomp["V"], dtype=float)
        W = np.array(decomp["W"], dtype=float)
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def strassen_56():
        """8 independent Strassen blocks for 2x2 * 2x16, rank 56."""
        num_blocks = p // 2  # 8
        rank = num_blocks * 7  # 56
        U = np.zeros((n * m, rank), dtype=np.int64)
        V = np.zeros((m * p, rank), dtype=np.int64)
        W = np.zeros((p * n, rank), dtype=np.int64)
        
        U_core, V_core, W_core = _strassen_2x2()
        
        for t in range(num_blocks):
            base = t * 7
            # U: same pattern for all blocks
            U[:, base:base+7] = U_core
            
            # V: B[j, 2t+l] → row j*p + (2t+l)
            # V_core rows: 0=(j=0,l=0), 1=(j=0,l=1), 2=(j=1,l=0), 3=(j=1,l=1)
            V[0*p + 2*t, base:base+7] = V_core[0]
            V[0*p + 2*t+1, base:base+7] = V_core[1]
            V[1*p + 2*t, base:base+7] = V_core[2]
            V[1*p + 2*t+1, base:base+7] = V_core[3]
            
            # W: C[k, i] → row k*n + i, where k=2t+l, i in {0,1}
            # W_core rows: 0=(k=0,i=0), 1=(k=0,i=1), 2=(k=1,i=0), 3=(k=1,i=1)
            W[(2*t)*n + 0, base:base+7] = W_core[0]
            W[(2*t)*n + 1, base:base+7] = W_core[1]
            W[(2*t+1)*n + 0, base:base+7] = W_core[2]
            W[(2*t+1)*n + 1, base:base+7] = W_core[3]
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try Strassen rank-56 first
    try:
        decomp = strassen_56()
        if verify(decomp):
            return decomp
    except Exception:
        pass

    # Fallback
    return schoolbook()
