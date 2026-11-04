"""
Rewritten program for Matrix Multiplication 2x3 x 3x10.
Approach: Split middle dimension m=3 into 2+1.
Part 1: 2x2 * 2x10 decomposed into 5 independent 2x2 multiplications using Strassen (Rank 7 each -> 35 total).
Part 2: 2x1 * 1x10 decomposed into 20 scalar multiplications (Rank 20).
Total Rank: 55.
"""

def construct():
    n, m, p = 2, 3, 10
    rank = 55
    
    # Initialize matrices with zeros
    # U: (n*m, rank) = (6, 55)
    # V: (m*p, rank) = (30, 55)
    # W: (p*n, rank) = (20, 55)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    idx = 0
    
    # Part 1: Strassen for the 2x2 block of A (cols 0,1) and 2x10 block of B (rows 0,1)
    # We iterate over the 5 pairs of columns in B (and thus C)
    for l in range(5):
        # Column indices in B and C
        k0 = 2 * l
        k1 = 2 * l + 1
        
        # Indices in U for A[0,0], A[0,1], A[1,0], A[1,1]
        # A is 2x3, so A[i,j] is at index i*3 + j
        u_a00 = 0 * 3 + 0  # 0
        u_a01 = 0 * 3 + 1  # 1
        u_a10 = 1 * 3 + 0  # 3
        u_a11 = 1 * 3 + 1  # 4
        
        # Indices in V for B[0,k0], B[0,k1], B[1,k0], B[1,k1]
        # B is 3x10, so B[j,k] is at index j*10 + k
        v_b0k0 = 0 * 10 + k0
        v_b0k1 = 0 * 10 + k1
        v_b1k0 = 1 * 10 + k0
        v_b1k1 = 1 * 10 + k1
        
        # Indices in W for C[0,k0], C[0,k1], C[1,k0], C[1,k1]
        # C is 2x10, convention W[k*n + i]
        w_c0k0 = k0 * 2 + 0
        w_c0k1 = k1 * 2 + 0
        w_c1k0 = k0 * 2 + 1
        w_c1k1 = k1 * 2 + 1
        
        # Strassen M1 = (a00 + a11) * (b0k0 + b1k1)
        U[u_a00][idx] += 1
        U[u_a11][idx] += 1
        V[v_b0k0][idx] += 1
        V[v_b1k1][idx] += 1
        W[w_c0k0][idx] += 1
        W[w_c1k1][idx] += 1
        idx += 1
        
        # Strassen M2 = (a10 + a11) * b0k0
        U[u_a10][idx] += 1
        U[u_a11][idx] += 1
        V[v_b0k0][idx] += 1
        W[w_c1k0][idx] += 1
        W[w_c1k1][idx] -= 1
        idx += 1
        
        # Strassen M3 = a00 * (b0k1 - b1k1)
        U[u_a00][idx] += 1
        V[v_b0k1][idx] += 1
        V[v_b1k1][idx] -= 1
        W[w_c0k1][idx] += 1
        W[w_c1k1][idx] += 1
        idx += 1
        
        # Strassen M4 = a11 * (b1k0 - b0k0)
        U[u_a11][idx] += 1
        V[v_b1k0][idx] += 1
        V[v_b0k0][idx] -= 1
        W[w_c0k0][idx] += 1
        W[w_c1k0][idx] += 1
        idx += 1
        
        # Strassen M5 = (a00 + a01) * b1k1
        U[u_a00][idx] += 1
        U[u_a01][idx] += 1
        V[v_b1k1][idx] += 1
        W[w_c0k0][idx] -= 1
        W[w_c0k1][idx] += 1
        idx += 1
        
        # Strassen M6 = (a10 - a00) * (b0k0 + b0k1)
        U[u_a10][idx] += 1
        U[u_a00][idx] -= 1
        V[v_b0k0][idx] += 1
        V[v_b0k1][idx] += 1
        W[w_c1k1][idx] += 1
        idx += 1
        
        # Strassen M7 = (a01 - a11) * (b1k0 + b1k1)
        U[u_a01][idx] += 1
        U[u_a11][idx] -= 1
        V[v_b1k0][idx] += 1
        V[v_b1k1][idx] += 1
        W[w_c0k0][idx] += 1
        idx += 1

    # Part 2: Schoolbook for the remaining 2x1 * 1x10 block
    # A[0,2] * B[2,k] -> C[0,k]
    # A[1,2] * B[2,k] -> C[1,k]
    
    u_a02 = 0 * 3 + 2  # 2
    u_a12 = 1 * 3 + 2  # 5
    
    for k in range(10):
        v_b2k = 2 * 10 + k
        
        w_c0k = k * 2 + 0
        w_c1k = k * 2 + 1
        
        # Term: A[0,2] * B[2,k]
        U[u_a02][idx] += 1
        V[v_b2k][idx] += 1
        W[w_c0k][idx] += 1
        idx += 1
        
        # Term: A[1,2] * B[2,k]
        U[u_a12][idx] += 1
        V[v_b2k][idx] += 1
        W[w_c1k][idx] += 1
        idx += 1

    return {'U': U, 'V': V, 'W': W}
