import numpy as np

def construct():
    n, m, p = 3, 3, 8
    rank = 68
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Strassen blocks for i in {0,1}, j in {0,1}, k in {2b, 2b+1} for b in 0..3
    for b in range(4):
        # Local indices mapping
        # i' in {0,1} -> i = i'
        # j' in {0,1} -> j = j'
        # k' in {0,1} -> k = 2b + k'
        
        # U indices: i*m + j
        # i=0, j=0 -> 0
        # i=0, j=1 -> 1
        # i=1, j=0 -> 3
        # i=1, j=1 -> 4
        u00, u01, u10, u11 = 0, 1, 3, 4
        
        # V indices: j*p + k
        # j=0, k=2b -> 2b
        # j=0, k=2b+1 -> 2b+1
        # j=1, k=2b -> 8+2b
        # j=1, k=2b+1 -> 9+2b
        v00, v01, v10, v11 = 2*b, 2*b+1, 8+2*b, 9+2*b
        
        # W indices: k*n + i
        # k=2b, i=0 -> 6b
        # k=2b, i=1 -> 6b+1
        # k=2b+1, i=0 -> 6b+3
        # k=2b+1, i=1 -> 6b+4
        w00, w01, w10, w11 = 6*b, 6*b+3, 6*b+1, 6*b+4
        
        # Term 1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        U[u00, col] = 1; U[u11, col] = 1
        V[v00, col] = 1; V[v11, col] = 1
        W[w00, col] = 1; W[w11, col] = 1
        col += 1
        
        # Term 2: (A10+A11)B00 -> C10 += 1, C11 += 1
        U[u10, col] = 1; U[u11, col] = 1
        V[v00, col] = 1
        W[w10, col] = 1; W[w11, col] = 1
        col += 1
        
        # Term 3: A00(B10-B11) -> C00 += 1, C01 -= 1
        U[u00, col] = 1
        V[v10, col] = 1; V[v11, col] = -1
        W[w00, col] = 1; W[w01, col] = -1
        col += 1
        
        # Term 4: A11(B01-B00) -> C00 -= 1, C10 += 1
        U[u11, col] = 1
        V[v01, col] = 1; V[v00, col] = -1
        W[w00, col] = -1; W[w10, col] = 1
        col += 1
        
        # Term 5: (A00+A01)B11 -> C01 += 1, C11 += 1
        U[u00, col] = 1; U[u01, col] = 1
        V[v11, col] = 1
        W[w01, col] = 1; W[w11, col] = 1
        col += 1
        
        # Term 6: (A10-A00)(B00+B10) -> C10 += 1, C11 -= 1
        U[u10, col] = 1; U[u00, col] = -1
        V[v00, col] = 1; V[v10, col] = 1
        W[w10, col] = 1; W[w11, col] = -1
        col += 1
        
        # Term 7: (A01-A11)(B01+B11) -> C00 += 1, C01 -= 1
        U[u01, col] = 1; U[u11, col] = -1
        V[v01, col] = 1; V[v11, col] = 1
        W[w00, col] = 1; W[w01, col] = -1
        col += 1

    # Remaining schoolbook terms
    # i in {0,1}, j=2, k in 0..7
    for i in [0, 1]:
        for k in range(8):
            U[i*3 + 2, col] = 1
            V[2*8 + k, col] = 1
            W[k*3 + i, col] = 1
            col += 1

    # i=2, j in 0..2, k in 0..7
    for j in range(3):
        for k in range(8):
            U[2*3 + j, col] = 1
            V[j*8 + k, col] = 1
            W[k*3 + 2, col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
