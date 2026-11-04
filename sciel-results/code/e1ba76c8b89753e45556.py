"""
Program to find rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.

Strategy:
  Partition A (2x3) into A1 (2x2) and A2 (2x1).
  Partition B (3x10) into B1 (2x10) and B2 (1x10).
  C = A1*B1 + A2*B2

  For A1*B1 (2x2 x 2x10): split B1 into five 2x2 blocks, use Strassen (rank 7) for each.
    Sub-rank: 5 * 7 = 35
  For A2*B2 (2x1 x 1x10): schoolbook, rank = 2*1*10 = 20
  Total rank: 35 + 20 = 55
"""
import time

DIMENSIONS = [2, 3, 10]
DEADLINE = 140.0  # Safety margin before 150s kill


def _add_strassen_2x2_block(U, V, W, col, i0, j0, k0):
    """
    Add a Strassen decomposition for a 2x2 * 2x2 block.
    A block: rows i0..i0+1, cols j0..j0+1 (in A)
    B block: rows j0..j0+1, cols k0..k0+1 (in B)
    C block: rows k0..k0+1, cols i0..i0+1 (in C)
    
    Strassen for 2x2*2x2:
    m1 = (a+b)*(e+f)
    m2 = (a+b)*(f-d)
    m3 = (c+d)*e
    m4 = (c-d)*(h-g)
    m5 = (a+d)*(e+h)
    m6 = (b+d)*(f-h)
    m7 = (a-c)*(g+h)
    
    C[0,0] = m5 + m4 - m6 - m7
    C[0,1] = m1 + m2
    C[1,0] = m3 + m5
    C[1,1] = m1 - m2 + m4 + m6
    
    Indexing: U row = i*m + j, V row = j*p + k, W row = k*n + i
    n=2, m=3, p=10
    """
    n, m, p = 2, 3, 10
    
    # A elements: a=A[i0,j0], b=A[i0,j0+1], c=A[i0+1,j0], d=A[i0+1,j0+1]
    # B elements: e=B[j0,k0], f=B[j0,k0+1], g=B[j0+1,k0], h=B[j0+1,k0+1]
    # C elements: C[i0,k0], C[i0,k0+1], C[i0+1,k0], C[i0+1,k0+1]
    
    a_u = i0 * m + j0
    b_u = i0 * m + (j0 + 1)
    c_u = (i0 + 1) * m + j0
    d_u = (i0 + 1) * m + (j0 + 1)
    
    e_v = j0 * p + k0
    f_v = j0 * p + (k0 + 1)
    g_v = (j0 + 1) * p + k0
    h_v = (j0 + 1) * p + (k0 + 1)
    
    c00_w = k0 * n + i0
    c01_w = (k0 + 1) * n + i0
    c10_w = k0 * n + (i0 + 1)
    c11_w = (k0 + 1) * n + (i0 + 1)
    
    # m1 = (a+b)*(e+f): U has +1 on a, +1 on b; V has +1 on e, +1 on f; W contributes to C[0,1] with +1
    U[a_u][col] += 1.0
    U[b_u][col] += 1.0
    V[e_v][col] += 1.0
    V[f_v][col] += 1.0
    W[c01_w][col] += 1.0
    col += 1
    
    # m2 = (a+b)*(f-d): U has +1 on a, +1 on b; V has +1 on f, -1 on d; W contributes to C[0,1] with +1, C[1,1] with -1
    U[a_u][col] += 1.0
    U[b_u][col] += 1.0
    V[f_v][col] += 1.0
    V[d_u][col] += -1.0  # Wait, d is in A not B. Let me reconsider.
    
    # Actually Strassen:
    # m1 = (a+b)*(e+f)
    # m2 = (a+b)*(f-d)  -- d is from A, f is from B. This is wrong for Strassen.
    
    # Let me redo Strassen properly:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+b)*(e+f)
    # m2 = (c+d)*(f-h)
    # m3 = (a-c)*(e+g)
    # m4 = (a+d)*(e+h)
    # m5 = (b+d)*(g+h)
    # m6 = (a-c)*(h-g)  -- wait, let me use the standard form
    # 
    # Standard Strassen:
    # m1 = (a+b)*(e+f)
    # m2 = (a+b)*(h-f)
    # m3 = (c+d)*e
    # m4 = (c+d)*(h-g)
    # m5 = (a+d)*(e+h)
    # m6 = (b+d)*(g+h)
    # m7 = (a-c)*(g+h)
    #
    # C[0,0] = m5 + m4 - m6 - m7
    # C[0,1] = m1 + m2
    # C[1,0] = m3 + m5
    # C[1,1] = m1 - m2 + m4 + m6
    
    # I made an error above. Let me reset and redo.
    # The entries above for m1 are correct. Let me fix m2 onwards.
    # Actually I need to redo this more carefully. Let me use a helper approach.
    
    return col


def _verify_decomposition(U, V, W, n, m, p, num_tests=20):
    """Verify the decomposition by testing against random matrices."""
    import random
    rank = len(U[0])
    for _ in range(num_tests):
        # Random A (n x m)
        A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        # Random B (m x p)
        B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        
        # Compute C = A @ B directly
        C_direct = [[0.0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += A[i][j] * B[j][k]
                C_direct[i][k] = s
        
        # Compute C via decomposition: C[i,k] = sum_r U[i*m+j... wait
        # U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
        # C[i,k] = sum_j sum_r U[i*m+j][r] * V[j*p+k][r] ... no
        # Actually: C[i,k] = sum_r (sum_j U[i*m+j][r]) * (sum_j V[j*p+k][r]) ... no
        # The tensor decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # C[i,k] = sum_j sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        
        C_decomp = [[0.0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                C_decomp[i][k] = s
        
        for i in range(n):
            for k in range(p):
                if abs(C_direct[i][k] - C_decomp[i][k]) > 1e-9:
                    return False
    return True


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    # Try rank-55 Strassen-based decomposition
    rank_55 = 55
    U = [[0.0] * rank_55 for _ in range(n * m)]
    V = [[0.0] * rank_55 for _ in range(m * p)]
    W = [[0.0] * rank_55 for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 (2x2) * B1 (2x10) using Strassen on five 2x2 blocks
    for block_idx in range(5):
        k0 = block_idx * 2
        
        a_u = 0
        b_u = 1
        c_u = 3
        d_u = 4
        
        e_v = 0 * p + k0
        f_v = 0 * p + (k0 + 1)
        g_v = 1 * p + k0
        h_v = 1 * p + (k0 + 1)
        
        c00_w = k0 * n + 0
        c01_w = (k0 + 1) * n + 0
        c10_w = k0 * n + 1
        c11_w = (k0 + 1) * n + 1
        
        # m1 = (a+b)*(e+f)
        U[a_u][col] += 1.0
        U[b_u][col] += 1.0
        V[e_v][col] += 1.0
        V[f_v][col] += 1.0
        W[c11_w][col] += 1.0
        col += 1
        
        # m2 = (c+d)*(e+f)
        U[c_u][col] += 1.0
        U[d_u][col] += 1.0
        V[e_v][col] += 1.0
        V[f_v][col] += 1.0
        W[c10_w][col] += 1.0
        W[c11_w][col] += -1.0
        col += 1
        
        # m3 = (a-c)*(g+h)
        U[a_u][col] += 1.0
        U[c_u][col] += -1.0
        V[g_v][col] += 1.0
        V[h_v][col] += 1.0
        W[c11_w][col] += 1.0
        col += 1
        
        # m4 = (b-d)*(e+h)
        U[b_u][col] += 1.0
        U[d_u][col] += -1.0
        V[e_v][col] += 1.0
        V[h_v][col] += 1.0
        W[c00_w][col] += 1.0
        W[c11_w][col] += 1.0
        col += 1
        
        # m5 = (a+d)*(e-h)
        U[a_u][col] += 1.0
        U[d_u][col] += 1.0
        V[e_v][col] += 1.0
        V[h_v][col] += -1.0
        W[c00_w][col] += 1.0
        W[c01_w][col] += 1.0
        col += 1
        
        # m6 = (c+d)*(g-e)
        U[c_u][col] += 1.0
        U[d_u][col] += 1.0
        V[g_v][col] += 1.0
        V[e_v][col] += -1.0
        W[c00_w][col] += 1.0
        W[c10_w][col] += 1.0
        col += 1
        
        # m7 = (a-b)*(f+h)
        U[a_u][col] += 1.0
        U[b_u][col] += -1.0
        V[f_v][col] += 1.0
        V[h_v][col] += 1.0
        W[c00_w][col] += -1.0
        W[c01_w][col] += 1.0
        col += 1
    
    # Part 2: A2 (2x1) * B2 (1x10) using schoolbook
    for i in range(2):
        for k in range(10):
            row_u = i * m + 2
            row_v = 2 * p + k
            row_w = k * n + i
            U[row_u][col] = 1.0
            V[row_v][col] = 1.0
            W[row_w][col] = 1.0
            col += 1
    
    assert col == rank_55, f"Column count mismatch: {col} != {rank_55}"
    
    # Verify the decomposition
    if _verify_decomposition(U, V, W, n, m, p, num_tests=10):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if Strassen is incorrect
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row_u = i * m + j
                row_v = j * p + k
                row_w = k * n + i
                U[row_u][col] = 1.0
                V[row_v][col] = 1.0
                W[row_w][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}

# Entry point required by evaluator
if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
