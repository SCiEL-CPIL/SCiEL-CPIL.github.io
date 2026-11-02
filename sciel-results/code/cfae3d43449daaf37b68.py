import time

def construct():
    n, m, p = 2, 4, 6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # Local indices 0,1
    # U[i, j], V[j, k], W[k, i]
    strassen_terms = [
        # M1
        { 'U': [(0,0,1), (1,1,1)], 'V': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)] },
        # M2
        { 'U': [(1,0,1), (1,1,1)], 'V': [(0,0,1)], 'W': [(0,1,1)] },
        # M3
        { 'U': [(0,0,1)], 'V': [(0,1,1), (1,1,-1)], 'W': [(1,0,1)] },
        # M4
        { 'U': [(1,1,1)], 'V': [(1,0,1), (0,0,-1)], 'W': [(0,1,1), (0,0,1)] },
        # M5
        { 'U': [(0,0,1), (0,1,1)], 'V': [(1,1,1)], 'W': [(1,0,1), (0,0,-1)] },
        # M6
        { 'U': [(1,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)], 'W': [(1,1,1)] },
        # M7
        { 'U': [(0,1,1), (1,1,-1)], 'V': [(1,0,1), (1,1,1)], 'W': [(0,0,1)] },
    ]
    
    col = 0
    for i0 in [0, 1]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                for term in strassen_terms:
                    # U
                    for (i, j, val) in term['U']:
                        U[i0 + i][j0 + j] # Wait, U index is i*m + j
                        # Correct index: (i0+i)*m + (j0+j)
                        U[(i0+i)*m + (j0+j)][col] += val
                    # V
                    for (j, k, val) in term['V']:
                        # V index is j*p + k
                        V[(j0+j)*p + (k0+k)][col] += val
                    # W
                    for (k, i, val) in term['W']:
                        # W index is k*n + i
                        W[(k0+k)*n + (i0+i)][col] += val
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
