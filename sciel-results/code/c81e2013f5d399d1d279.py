"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split the 5 intermediate dimension into 1 + 4
    m1, m2 = 1, 4
    p1, p2 = 6, 6  # p is shared
    
    # Sub-problem 1: 2x1 times 1x6
    n1, m1_s, p1_s = n, m1, p1
    # Sub-problem 2: 2x4 times 4x6
    n2, m2_s, p2_s = n, m2, p2
    
    def trivial_decomp(n_s, m_s, p_s):
        r = n_s * m_s * p_s
        U_s = [[0] * r for _ in range(n_s * m_s)]
        V_s = [[0] * r for _ in range(m_s * p_s)]
        W_s = [[0] * r for _ in range(p_s * n_s)]
        col = 0
        for i_s in range(n_s):
            for j_s in range(m_s):
                for k_s in range(p_s):
                    U_s[i_s * m_s + j_s][col] = 1
                    V_s[j_s * p_s + k_s][col] = 1
                    W_s[k_s * n_s + i_s][col] = 1
                    col += 1
        return U_s, V_s, W_s
    
    U1, V1, W1 = trivial_decomp(n1, m1_s, p1_s)
    U2, V2, W2 = trivial_decomp(n2, m2_s, p2_s)
    
    rank = len(U1[0]) + len(U2[0])
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Combine U1
    for i in range(n1):
        for j in range(m1_s):
            row = i * m + j
            for c in range(len(U1[0])):
                U[row][c] = U1[i * m1_s + j][c]
    
    # Combine U2 (offset rows by m1)
    for i in range(n2):
        for j in range(m2_s):
            row = i * m + (m1 + j)
            for c in range(len(U2[0])):
                U[row][len(U1[0]) + c] = U2[i * m2_s + j][c]
    
    # Combine V1
    for j in range(m1_s):
        for k in range(p1_s):
            row = j * p + k
            for c in range(len(V1[0])):
                V[row][c] = V1[j * p1_s + k][c]
    
    # Combine V2 (offset rows by m1)
    for j in range(m2_s):
        for k in range(p2_s):
            row = (m1 + j) * p + k
            for c in range(len(V2[0])):
                V[row][len(V1[0]) + c] = V2[j * p2_s + k][c]
    
    # Combine W1
    for i in range(n1):
        for k in range(p1_s):
            row = k * n + i
            for c in range(len(W1[0])):
                W[row][c] = W1[k * n1 + i][c]
    
    # Combine W2
    for i in range(n2):
        for k in range(p2_s):
            row = k * n + i
            for c in range(len(W2[0])):
                W[row][len(W1[0]) + c] = W2[k * n2 + i][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
