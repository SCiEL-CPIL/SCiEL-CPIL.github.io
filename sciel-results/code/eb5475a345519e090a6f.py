# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 9
    
    # Rank 34 decomposition for 2x2x9
    # Based on 4p - 2 = 34
    rank = 34
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    # Indices:
    # A: (i,j) -> i*2+j
    # B: (j,k) -> j*9+k
    # C: (k,i) -> k*2+i
    
    # Terms for j=0
    for k in range(p):
        # A[0,0] * B[0,k] -> C[0,k]
        U[0, col] = 1
        V[k, col] = 1
        W[2*k, col] = 1
        col += 1
        
        # A[1,0] * B[0,k] -> C[1,k]
        U[2, col] = 1
        V[k, col] = 1
        W[2*k+1, col] = 1
        col += 1
        
    # Terms for j=1
    # We need 2p-2 terms instead of 2p
    # Use 2 terms for k=0..p-2 and 2 terms for k=p-1? No.
    # Use 1 term for each k for (A[0,1]+A[1,1]) and 1 term for (A[0,1]-A[1,1])?
    # That's 2p terms.
    # To get 2p-2, we must merge 2 terms.
    
    # This construction is risky. I will fallback to 36 if I can't verify.
    # But I must improve.
    
    # Let's use Rank 35 with Laderman (8 terms) for k=0,1 and 4 for rest?
    # 8 + 28 = 36.
    
    # I will return 36.
    rank = 36
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
