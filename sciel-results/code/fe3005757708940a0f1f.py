import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def _verify(U, V, W, n=2, m=3, p=9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    np.random.seed(42)
    A = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B = np.random.randint(-5, 6, size=(m, p)).astype(float)
    C_expected = A @ B
    
    C_reconstructed = np.zeros((n, p))
    rank = len(U[0])
    for t in range(rank):
        alpha = np.array([sum(U[i * m + k][t] * A[i, k] for k in range(m)) for i in range(n)])
        beta = np.array([sum(V[k * p + j][t] * B[k, j] for k in range(m)) for j in range(p)])
        for i in range(n):
            for j in range(p):
                C_reconstructed[i, j] += W[j * n + i][t] * alpha[i] * beta[j]
    
    return np.allclose(C_reconstructed, C_expected, atol=1e-9)

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x9 matrix multiplication.
    Uses Strassen for 2x2 blocks to achieve rank 50.
    """
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # 6 x 50
    V = [[0] * rank for _ in range(m * p)]  # 27 x 50
    W = [[0] * rank for _ in range(p * n)]  # 18 x 50
    
    col = 0
    
    # Part 1: A1 (2x2) * B1 (2x9)
    # A1 indices: i in {0,1}, k in {0,1}. U row: i*3 + k
    # B1 indices: k in {0,1}, j in {0..8}. V row: k*9 + j
    # C indices: i in {0,1}, j in {0..8}. W row: j*2 + i
    
    # Blocks of B1 columns: [0,1], [2,3], [4,5], [6,7], [8]
    blocks = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    for j0, j1 in blocks:
        # 7 Strassen terms
        # Term 1: M1 = (a+d)(e+h)
        U[0*3+0][col] = 1; U[1*3+1][col] = 1
        V[0*9+j0][col] = 1; V[1*9+j1][col] = 1
        W[j0*2+0][col] = 1; W[j1*2+1][col] = 1
        col += 1
        
        # Term 2: M2 = (c+d)(e+f)
        U[1*3+0][col] = 1; U[1*3+1][col] = 1
        V[0*9+j0][col] = 1; V[0*9+j1][col] = 1
        W[j0*2+1][col] = 1; W[j1*2+1][col] = -1
        col += 1
        
        # Term 3: M3 = (a+b)(g+h)
        U[0*3+0][col] = 1; U[0*3+1][col] = 1
        V[1*9+j0][col] = 1; V[1*9+j1][col] = 1
        W[j1*2+0][col] = 1; W[j1*2+1][col] = 1
        col += 1
        
        # Term 4: M4 = (c-a)(e+g)
        U[1*3+0][col] = 1; U[0*3+0][col] = -1
        V[0*9+j0][col] = 1; V[1*9+j0][col] = 1
        W[j0*2+0][col] = 1; W[j0*2+1][col] = 1
        col += 1
        
        # Term 5: M5 = (b-d)(f+h)
        U[0*3+1][col] = 1; U[1*3+1][col] = -1
        V[0*9+j1][col] = 1; V[1*9+j1][col] = 1
        W[j0*2+0][col] = -1; W[j1*2+0][col] = 1
        col += 1
        
        # Term 6: M6 = (a+d)(e-g)
        U[0*3+0][col] = 1; U[1*3+1][col] = 1
        V[0*9+j0][col] = 1; V[1*9+j0][col] = -1
        W[j1*2+1][col] = 1
        col += 1
        
        # Term 7: M7 = (b+c)(h-f)
        U[0*3+1][col] = 1; U[1*3+0][col] = 1
        V[1*9+j1][col] = 1; V[0*9+j1][col] = -1
        W[j0*2+0][col] = 1
        col += 1

    # Tail block: column 8 (2x1)
    j2 = 8
    # Schoolbook 2x1
    # C0, j2 = A0,0 B0, j2 + A0,1 B1, j2
    U[0*3+0][col] = 1
    V[0*9+j2][col] = 1
    W[j2*2+0][col] = 1
    col += 1
    
    U[0*3+1][col] = 1
    V[1*9+j2][col] = 1
    W[j2*2+0][col] = 1
    col += 1
    
    # C1, j2 = A1,0 B0, j2 + A1,1 B1, j2
    U[1*3+0][col] = 1
    V[0*9+j2][col] = 1
    W[j2*2+1][col] = 1
    col += 1
    
    U[1*3+1][col] = 1
    V[1*9+j2][col] = 1
    W[j2*2+1][col] = 1
    col += 1

    # Part 2: A2 (2x1) * B2 (1x9)
    # A2 is column 2 of A. U row: i*3 + 2
    # B2 is row 2 of B. V row: 2*9 + j
    # C indices: i in {0,1}, j in {0..8}. W row: j*2 + i
    
    for i in range(2):
        for j in range(9):
            U[i*3+2][col] = 1
            V[2*9+j][col] = 1
            W[j*2+i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(_verify(res['U'], res['V'], res['W']))
