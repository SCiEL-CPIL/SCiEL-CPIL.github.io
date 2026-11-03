"""Matrix multiplication 4x4, 4x7 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]


def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (U, V, W) where U, V, W are 4-element lists.
    U[i*2+j], V[j*2+k], W[k*2+i].
    """
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # Flatten: A[0]=a,A[1]=b,A[2]=c,A[3]=d
    #          B[0]=e,B[1]=f,B[2]=g,B[3]=h
    #          C[0]=i,C[1]=j,C[2]=k,C[3]=l
    # C[0] = M1 - M4 + M5 - M7
    # C[1] = M3 + M5
    # C[2] = M2 + M4
    # C[3] = M1 - M2 + M6 - M7
    terms = []
    # M1 = (a+d)(e+h)
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # M2 = (c+d)(e)
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    # M3 = a(f-h)
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
    # M4 = d(g-e)
    terms.append(([0,0,0,1], [0,0,1,-1], [-1,0,1,0]))
    # M5 = (a+b)(h)
    terms.append(([1,1,0,0], [0,0,0,1], [1,1,0,0]))
    # M6 = (c-a)(e+f)
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    # M7 = (b-d)(g+h)
    terms.append(([0,1,0,-1], [0,0,1,1], [-1,0,0,-1]))
    return terms


def _strassen_4x4x4_terms():
    """Generate 49 Strassen terms for 4x4x4 using Kronecker product of 2x2 Strassen."""
    base = _strassen_2x2_terms()
    terms = []
    for c1 in range(7):
        for c2 in range(7):
            U4 = [0] * 16
            V4 = [0] * 16
            W4 = [0] * 16
            for I in range(4):
                for J in range(4):
                    I1, I0 = I // 2, I % 2
                    J1, J0 = J // 2, J % 2
                    U4[I*4+J] = base[c1][0][I1*2+J1] * base[c2][0][I0*2+J0]
            for J in range(4):
                for K in range(4):
                    J1, J0 = J // 2, J % 2
                    K1, K0 = K // 2, K % 2
                    V4[J*4+K] = base[c1][1][J1*2+K1] * base[c2][1][J0*2+K0]
            for K in range(4):
                for I in range(4):
                    K1, K0 = K // 2, K % 2
                    I1, I0 = I // 2, I % 2
                    W4[K*4+I] = base[c1][2][K1*2+I1] * base[c2][2][K0*2+I0]
            terms.append((U4, V4, W4))
    return terms


def _make_decomposition():
    """Create a rank-98 decomposition for 4x4x7.
    Block 1: columns 0-3 of B -> columns 0-3 of C (49 terms, full 4x4x4 Strassen)
    Block 2: columns 4-6 of B -> columns 4-6 of C (49 terms, 4x4x3 restricted)
    """
    terms_4x4 = _strassen_4x4x4_terms()
    n, m, p = 4, 4, 7
    rank = 98
    
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    for idx, (U4, V4, W4) in enumerate(terms_4x4):
        # Block 1: cols 0-3
        c1 = idx
        for I in range(4):
            for J in range(4):
                U[I*m+J, c1] = U4[I*4+J]
        for J in range(4):
            for K in range(4):
                V[J*p+K, c1] = V4[J*4+K]
        for K in range(4):
            for I in range(4):
                W[K*n+I, c1] = W4[K*4+I]
        
        # Block 2: cols 4-6 (map internal col 0,1,2 -> actual col 4,5,6)
        c2 = idx + 49
        for I in range(4):
            for J in range(4):
                U[I*m+J, c2] = U4[I*4+J]
        for J in range(4):
            for K in range(3):  # only 3 columns
                V[J*p+(K+4), c2] = V4[J*4+K]
        for K in range(3):  # only 3 columns
            for I in range(4):
                W[(K+4)*n+I, c2] = W4[K*4+I]
    
    return U.tolist(), V.tolist(), W.tolist()


def _verify(U, V, W, n, m, p):
    """Verify decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
                if s != 1:
                    return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin under 120s
    
    n, m, p = DIMENSIONS
    
    # Build the rank-98 Strassen-based decomposition
    U, V, W = _make_decomposition()
    
    # Verify (fast with numpy)
    if time.time() < deadline - 5:
        if not _verify(U, V, W, n, m, p):
            # Fall back to schoolbook
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
