"""Matrix multiplication 2,3,5 — rank-29 decomposition via Strassen 2x2x2 block."""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 2x3 * 3x5 matrix multiplication using rank-29 ansatz.
    
    Strategy: Replace the 8 schoolbook terms for the (i,j,k) sub-block
    with i in {0,1}, j in {0,1}, k in {0,1} using Strassen's 7-term
    decomposition for 2x2x2. The remaining 22 terms stay as schoolbook.
    Total rank: 7 + 22 = 29.
    """
    n, m, p = 2, 3, 5
    rank = 29
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    c = 0
    
    # Strassen 2x2x2 block for i in {0,1}, j in {0,1}, k in {0,1}
    # m1 = (a00+a11)*(b00+b11), contributes to c00(+1), c11(+1)
    U[0*3+0][c] = 1; U[1*3+1][c] = 1
    V[0*5+0][c] = 1; V[1*5+1][c] = 1
    W[0*2+0][c] = 1; W[1*2+1][c] = 1
    c += 1
    
    # m2 = (a10+a11)*b00, contributes to c10(+1), c11(-1)
    U[1*3+0][c] = 1; U[1*3+1][c] = 1
    V[0*5+0][c] = 1
    W[0*2+1][c] = 1; W[1*2+1][c] = -1
    c += 1
    
    # m3 = a00*(b01-b11), contributes to c01(+1), c11(+1)
    U[0*3+0][c] = 1
    V[0*5+1][c] = 1; V[1*5+1][c] = -1
    W[1*2+0][c] = 1; W[1*2+1][c] = 1
    c += 1
    
    # m4 = a11*(b10-b00), contributes to c00(+1), c10(+1)
    U[1*3+1][c] = 1
    V[1*5+0][c] = 1; V[0*5+0][c] = -1
    W[0*2+0][c] = 1; W[0*2+1][c] = 1
    c += 1
    
    # m5 = (a00+a01)*b11, contributes to c00(-1), c01(+1)
    U[0*3+0][c] = 1; U[0*3+1][c] = 1
    V[1*5+1][c] = 1
    W[0*2+0][c] = -1; W[1*2+0][c] = 1
    c += 1
    
    # m6 = (a10-a00)*(b00+b01), contributes to c11(+1)
    U[1*3+0][c] = 1; U[0*3+0][c] = -1
    V[0*5+0][c] = 1; V[0*5+1][c] = 1
    W[1*2+1][c] = 1
    c += 1
    
    # m7 = (a01-a11)*(b10+b11), contributes to c00(+1)
    U[0*3+1][c] = 1; U[1*3+1][c] = -1
    V[1*5+0][c] = 1; V[1*5+1][c] = 1
    W[0*2+0][c] = 1
    c += 1
    
    # Remaining 22 schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip the 8 terms we already handled with Strassen
                if i in (0, 1) and j in (0, 1) and k in (0, 1):
                    continue
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    
    assert c == rank, f"Expected rank {rank}, got {c}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
