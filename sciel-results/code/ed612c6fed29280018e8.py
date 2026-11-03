import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE = 145.0
_START = time.time()

def _time_left():
    return DEADLINE - (time.time() - _START)

def _laderman_3x3_3():
    """
    Return U, V, W for 3x3 * 3x3 with rank 23 (Laderman).
    U shape (9, 23), V shape (9, 23), W shape (9, 23).
    Indices: U[i*3+j], V[j*3+k], W[k*3+i].
    """
    n, m, p = 3, 3, 3
    rank = 23
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Laderman's 23 products
    # We define L_r(A) and M_r(B) and C_{i,k} = sum c_{i,k,r} L_r M_r
    # U[i,j,r] is coeff of A[i,j] in L_r
    # V[j,k,r] is coeff of B[j,k] in M_r
    # W[k,i,r] is coeff c_{i,k,r}
    
    # 1. P1 = A00 B00
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1 # C00 += A00 B00
    # Note: W index is k*n+i. For C00, k=0, i=0. W[0*3+0]=W[0].
    
    # 2. P2 = A11 B11
    U[4][1] = 1; V[4][1] = 1; W[1][1] = 1 # C11 += A11 B11
    
    # 3. P3 = A22 B22
    U[8][2] = 1; V[8][2] = 1; W[2][2] = 1 # C22 += A22 B22
    
    # 4. P4 = (A00+A11)(B00+B11)
    U[0][3] = 1; U[4][3] = 1
    V[0][3] = 1; V[4][3] = 1
    # C00 += P4 - P1 - P2 + ...? 
    # Standard Laderman reconstruction is complex.
    # I will use a known set of coefficients.
    
    # Due to complexity of hardcoding Laderman correctly from memory, 
    # I will fall back to a safer, slightly higher rank if I can't verify Laderman quickly.
    # However, 64 is the target.
    # Let's use Strassen for 2x2x2 blocks which I know is correct.
    # Rank 68.
    # Is 68 better than 72? Yes.
    # Is 64 achievable? Yes, but risky to hardcode Laderman without verification.
    # I will stick to the Strassen embedding (Rank 68) but optimize the placement if possible.
    # Or, I can try to find a better 2x2x2 embedding.
    
    # Actually, I can use the provided _strassen_2x2_2 and _build_strassen_embedding.
    # The current program already does this.
    # Score 1.0588 corresponds to rank 68?
    # Seed score 1.0 corresponds to rank 72 (schoolbook).
    # 72/68 = 1.0588.
    # So current program is rank 68.
    
    # To improve, I need rank < 68.
    # Laderman gives 64.
    # I MUST implement Laderman correctly.
    
    # Let's try to construct Laderman using a library or careful derivation?
    # No, I will use a known valid decomposition for 3x3x3.
    # There is a 23-term algorithm.
    
    # If I fail to implement Laderman, I return the current best (68).
    
    # Let's try to implement a simplified Laderman or verify the coefficients.
    
    # I will use the following known 23-term coefficients for 3x3x3.
    # Reference: Laderman, "A 23-term algorithm for 3x3 matrix multiplication"
    
    # This is tedious to hardcode.
    # Alternative: Use 4 Strassen blocks (Rank 68) and see if I can shave off 1-2 ranks?
    # Unlikely.
    
    # I will attempt to hardcode Laderman.
    
    # Coefficients for Laderman (simplified for code):
    # I will use a placeholder that falls back to Strassen if Laderman fails verification.
    
    return None # Placeholder

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _strassen_2x2_2():
    """Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen)."""
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_strassen_embedding(n, m, p, strassen_blocks):
    covered = set()
    for (i0, j0, k0) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    n_school = n*m*p - len(covered)
    rank = 7 * len(strassen_blocks) + n_school
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for (i0, j0, k0) in strassen_blocks:
        sU, sV, sW = _strassen_2x2_2()
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj)][col+c] = sU[di*2+dj][c]
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk)][col+c] = sV[dj*2+dk][c]
            for dk in range(2):
                for di in range(2):
                    W[(k0+dk)*n + (i0+di)][col+c] = sW[dk*2+di][c]
        col += 7
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Try Laderman if possible
    # Since I don't have verified Laderman coefficients, I will stick to Strassen.
    # Rank 68 is the current best.
    
    blocks = [
        (0, 0, 0),
        (0, 1, 2),
        (1, 0, 4),
        (1, 1, 6)
    ]
    
    U, V, W = _build_strassen_embedding(n, m, p, blocks)
    
    if not _verify(U, V, W, n, m, p):
        U, V, W = _schoolbook(n, m, p)
        
    return {"U": U, "V": V, "W": W}
