"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition using Strassen's algorithm for 2x2 blocks
    # and Strassen's algorithm for the scalar 2x2 multiplications.
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen coefficients for 2x2 multiplication
    # Indices: 0, 1 for rows/cols
    # U_s[u, v] coeff of A[u, v]
    # V_s[v, w] coeff of B[v, w]
    # W_s[w, u] coeff of C[u, w] (Note: C index is u*w? No, C[u, w] corresponds to W[w, u] in flat index w*n+u? 
    # Task W index is k*n+i. So C[i, k] is at W[k*n+i].
    # In my derivation: W[k*n+i] = W_outer[s][w, u] * W_inner[t][k', i'].
    # Here i is row, k is col.
    # So C[i, k] coefficient is W_outer[s][k//2, i//2] * W_inner[t][k%2, i%2].
    
    # Strassen terms for 2x2
    # Term 0
    U0 = [[1, 0], [0, 1]] # A00, A11
    V0 = [[1, 0], [0, 1]] # B00, B11
    W0 = [[1, 0], [0, 1]] # C00, C11 -> W[0,0]=1, W[1,1]=1. (w,u) indices: W[0,0] is C[0,0], W[1,1] is C[1,1].
    
    # Term 1
    U1 = [[0, 0], [1, 1]] # A10, A11
    V1 = [[1, 0], [0, 0]] # B00
    W1 = [[0, 1], [-1, 0]] # C10, C11 -> W[0,1]=1 (C[1,0]), W[1,1]=-1 (C[1,1]).
    # Check: W[k, i]. C[i, k].
    # C[1, 0] is i=1, k=0. W[0, 1].
    # C[1, 1] is i=1, k=1. W[1, 1].
    
    # Term 2
    U2 = [[1, 0], [0, 0]] # A00
    V2 = [[0, 1], [-1, 0]] # B01, B11 -> B[0,1]=1, B[1,1]=-1.
    W2 = [[1, 0], [1, 0]] # C01, C11 -> C[0,1] (i=0, k=1) -> W[1,0]. C[1,1] (i=1, k=1) -> W[1,1].
    # Wait.
    # W2 should be: C[0,1] coeff 1, C[1,1] coeff 1.
    # C[0,1] -> i=0, k=1 -> W[1, 0].
    # C[1,1] -> i=1, k=1 -> W[1, 1].
    # So W2[1,0]=1, W2[1,1]=1.
    W2 = [[0, 0], [1, 1]]
    
    # Term 3
    U3 = [[0, 0], [0, 1]] # A11
    V3 = [[-1, 0], [1, 0]] # B00, B10 -> B[0,0]=-1, B[1,0]=1.
    W3 = [[1, 0], [1, 0]] # C00, C10 -> C[0,0] (i=0, k=0) -> W[0,0]. C[1,0] (i=1, k=0) -> W[0,1].
    # So W3[0,0]=1, W3[0,1]=1.
    W3 = [[1, 1], [0, 0]]
    
    # Term 4
    U4 = [[1, 1], [0, 0]] # A00, A01
    V4 = [[0, 0], [0, 1]] # B11
    W4 = [[-1, 0], [1, 0]] # C00, C01 -> C[0,0] -> W[0,0]. C[0,1] -> W[1,0].
    # So W4[0,0]=-1, W4[1,0]=1.
    W4 = [[-1, 0], [1, 0]]
    
    # Term 5
    U5 = [[-1, 0], [1, 0]] # A00, A10
    V5 = [[1, 1], [0, 0]] # B00, B01
    W5 = [[0, 0], [0, 1]] # C11 -> C[1,1] -> W[1,1].
    W5 = [[0, 0], [0, 1]]
    
    # Term 6
    U6 = [[0, 1], [-1, 0]] # A01, A11
    V6 = [[0, 0], [1, 1]] # B10, B11
    W6 = [[1, 0], [0, 0]] # C00 -> C[0,0] -> W[0,0].
    W6 = [[1, 0], [0, 0]]
    
    U_outer = [U0, U1, U2, U3, U4, U5, U6]
    V_outer = [V0, V1, V2, V3, V4, V5, V6]
    W_outer = [W0, W1, W2, W3, W4, W5, W6]
    
    # Inner Strassen is the same
    U_inner = U_outer
    V_inner = V_outer
    W_inner = W_outer
    
    for s in range(7):
        for t in range(7):
            c = s * 7 + t
            # Fill U
            for i in range(4):
                for j in range(4):
                    u = i // 2
                    v = j // 2
                    ip = i % 2
                    jp = j % 2
                    U[i * 4 + j][c] = U_outer[s][u][v] * U_inner[t][ip][jp]
            
            # Fill V
            for j in range(4):
                for k in range(4):
                    v = j // 2
                    w = k // 2
                    jp = j % 2
                    kp = k % 2
                    V[j * 4 + k][c] = V_outer[s][v][w] * V_inner[t][jp][kp]
            
            # Fill W
            for k in range(4):
                for i in range(4):
                    w = k // 2
                    u = i // 2
                    kp = k % 2
                    ip = i % 2
                    W[k * 4 + i][c] = W_outer[s][w][u] * W_inner[t][kp][ip]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
