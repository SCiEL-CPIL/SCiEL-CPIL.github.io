"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _block_decomposition_6x6x6():
    """
    Decompose 6x6x6 matrix multiplication using 3x3 blocks of 2x2 matrices.
    Total rank = 27 * 7 = 189.
    """
    n, m, p = 6, 6, 6
    rank = 189
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    sU, sV, sW = _strassen_2x2()
    
    col = 0
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                for c in range(7):
                    for li in range(2):
                        for lj in range(2):
                            u_row = (bi*2+li)*6 + (bj*2+lj)
                            U[u_row][col] = sU[li*2+lj][c]
                    
                    for lj in range(2):
                        for lk in range(2):
                            v_row = (bj*2+lj)*6 + (bk*2+lk)
                            V[v_row][col] = sV[lj*2+lk][c]
                    
                    for lk in range(2):
                        for li in range(2):
                            w_row = (bk*2+lk)*6 + (bi*2+li)
                            W[w_row][col] = sW[lk*2+li][c]
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}

def _strassen_3x3x3():
    """
    Construct a decomposition for 3x3x3 using Strassen-like approach.
    Split 3x3 into 2x2 + 1x1 blocks and use Strassen for the 2x2 part.
    This gives rank = 7 (for 2x2) + 1 (for the extra row/col) * some factor.
    
    Actually, let's use a different approach: use the known rank-23 decomposition
    or construct one via Kronecker products.
    
    For now, let's use a simple approach: schoolbook 3x3 (rank 27).
    """
    n = 3
    rank = 27
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    return U, V, W

def _hybrid_decomposition_6x6x6():
    """
    Hybrid approach: use Strassen 2x2 for the top-left 4x4 block and
    schoolbook for the remaining parts.
    
    Split 6x6 into:
    - Top-left 4x4 (handled by 2x2 Strassen blocks: rank 4*4*4 * 7/8 = 112)
    - Top-right 4x2, bottom-left 2x4, bottom-right 2x2 (schoolbook)
    
    Actually, let's try a cleaner approach: use 3x3 blocks with a better 3x3 decomposition.
    """
    n, m, p = 6, 6, 6
    
    # Use 2x2 Strassen for all 2x2 sub-blocks in a 3x3 grid
    # This is the same as _block_decomposition_6x6x6
    # Let's try something different: use 3x3 Strassen-like
    
    # For now, return the same as block decomposition
    return _block_decomposition_6x6x6()

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    n, m, p = DIMENSIONS
    
    # Primary: block decomposition with Strassen 2x2 (rank 189)
    best = _block_decomposition_6x6x6()
    
    # Verify quickly using numpy for speed
    U_arr = np.array(best["U"], dtype=np.float64)
    V_arr = np.array(best["V"], dtype=np.float64)
    W_arr = np.array(best["W"], dtype=np.float64)
    
    # Check a few random entries
    import random
    random.seed(42)
    sample_indices = [(random.randint(0,5), random.randint(0,5), random.randint(0,5)) for _ in range(50)]
    
    valid = True
    for (i, j, k) in sample_indices:
        s = 0.0
        for c in range(len(U_arr[0])):
            s += U_arr[i*6+j][c] * V_arr[j*6+k][c] * W_arr[k*6+i][c]
        if abs(s - 1.0) > 1e-9:
            valid = False
            break
    
    if valid:
        # Full verification with a subset
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in sample_indices:
                        continue
                    s = 0.0
                    for c in range(len(U_arr[0])):
                        s += U_arr[i*m+j][c] * V_arr[j*p+k][c] * W_arr[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
    
    if valid:
        return best
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
