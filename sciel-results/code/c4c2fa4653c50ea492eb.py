DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2 x 2x2
    # Applied to 5 independent blocks of 2x2 columns in B
    
    for q in range(5):
        # Columns of B involved: 2q, 2q+1
        # Indices in V:
        # B[0, 2q] -> 0 * 10 + 2q = 2q
        # B[0, 2q+1] -> 0 * 10 + 2q+1 = 2q+1
        # B[1, 2q] -> 1 * 10 + 2q = 10+2q
        # B[1, 2q+1] -> 1 * 10 + 2q+1 = 11+2q
        
        # Indices in W:
        # C[0, 2q] -> 2q * 2 + 0 = 4q
        # C[0, 2q+1] -> (2q+1) * 2 + 0 = 4q+2
        # C[1, 2q] -> 2q * 2 + 1 = 4q+1
        # C[1, 2q+1] -> (2q+1) * 2 + 1 = 4q+3
        
        # Indices in U:
        # A[0,0] -> 0
        # A[0,1] -> 1
        # A[1,0] -> 2
        # A[1,1] -> 3
        
        # s=0: M1 = (A00+A11)(B0,2q+B1,2q+1)
        r = 7*q + 0
        U[0][r] += 1
        U[3][r] += 1
        V[2*q][r] += 1
        V[11+2*q][r] += 1
        W[4*q][r] += 1
        W[4*q+3][r] += 1
        
        # s=1: M2 = (A10+A11)B0,2q
        r = 7*q + 1
        U[2][r] += 1
        U[3][r] += 1
        V[2*q][r] += 1
        W[4*q+1][r] += 1
        W[4*q+3][r] += -1
        
        # s=2: M3 = A00(B0,2q+1-B1,2q+1)
        r = 7*q + 2
        U[0][r] += 1
        V[2*q+1][r] += 1
        V[11+2*q][r] += -1
        W[4*q+2][r] += 1
        W[4*q+3][r] += 1
        
        # s=3: M4 = A11(B1,2q-B0,2q)
        r = 7*q + 3
        U[3][r] += 1
        V[10+2*q][r] += 1
        V[2*q][r] += -1
        W[4*q][r] += 1
        W[4*q+1][r] += 1
        
        # s=4: M5 = (A00+A01)B1,2q+1
        r = 7*q + 4
        U[0][r] += 1
        U[1][r] += 1
        V[11+2*q][r] += 1
        W[4*q][r] += -1
        W[4*q+2][r] += 1
        
        # s=5: M6 = (A10-A00)(B0,2q+B0,2q+1)
        r = 7*q + 5
        U[2][r] += 1
        U[0][r] += -1
        V[2*q][r] += 1
        V[2*q+1][r] += 1
        W[4*q+3][r] += 1
        
        # s=6: M7 = (A01-A11)(B1,2q+B1,2q+1)
        r = 7*q + 6
        U[1][r] += 1
        U[3][r] += -1
        V[10+2*q][r] += 1
        V[11+2*q][r] += 1
        W[4*q][r] += 1

    return {"U": U, "V": V, "W": W}
