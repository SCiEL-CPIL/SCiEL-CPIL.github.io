import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will construct a rank-85 decomposition.
    # It consists of:
    # 1. 4x4x4 Strassen part (rank 49)
    # 2. 4x4x1 Schoolbook part (rank 16)
    # 3. 4x1x4 Schoolbook part (rank 16)
    # 4. 4x1x1 Schoolbook part (rank 4)
    
    rank = 49 + 16 + 16 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get Strassen 2x2x2 matrices
    # U, V, W are 4x7
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    # M2
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1
    # M3
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1
    # M4
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[1][3] = 1
    # M5
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = 1; W2[2][4] = 1
    # M6
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    # M7
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # Kronecker product for 4x4x4
    # U4 = U2 x U2 (16x49)
    # We need to map indices correctly.
    # U4[i*4+j, k*7+l] = U2[i*2+row1, k*7+col1] * U2[j*2+row2, l*7+col2]?
    # No, the Kronecker product of the factors gives the factors for the tensor product.
    # The tensor $T = T_2 \otimes T_2$.
    # $U_4 = U_2 \otimes U_2$.
    # $V_4 = V_2 \otimes V_2$.
    # $W_4 = W_2 \otimes W_2$.
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for i in range(4):
        for j in range(4):
            for k in range(7):
                for l in range(7):
                    # Index in U4
                    u_idx = i * 4 + j
                    # Index in U2
                    u2_idx_1 = i * 2 + (j // 2) # This is wrong.
                    
    # Let's do it properly.
    # U4 is 16x49.
    # Row i*4+j of U4 corresponds to A_{i,j} in the 4x4 matrix.
    # This A_{i,j} is composed of A_{i//2, j//2} and A_{i%2, j%2} in the 2x2 blocks?
    # Actually, the Kronecker product structure:
    # $A_{4 \times 4} = \begin{pmatrix} A_{00} & A_{01} \\ A_{10} & A_{11} \end{pmatrix}$ where blocks are $2 \times 2$.
    # The Strassen algorithm applies to the $2 \times 2$ block matrix.
    # So the input to the first Strassen layer is the 4x4 matrix viewed as 2x2 blocks.
    # The factors $U_2$ operate on the block indices.
    # The factors $U_2$ also operate on the internal 2x2 indices.
    # So $U_4$ row $(i,j)$ (where $i,j \in 0..3$) is $U_2[i//2, col1] \times U_2[i%2, col2]$?
    # No.
    # $U_4$ row $r$ (0..15). $r = i \times 4 + j$.
    # $i = r // 4$, $j = r \% 4$.
    # $i$ is row in 4x4. $j$ is col in 4x4.
    # In the 2x2 block structure:
    # Block row $I = i // 2$. Block col $J = j // 2$.
    # Internal row $ii = i \% 2$. Internal col $jj = j \% 2$.
    # The rank index is $K = k \times 7 + l$.
    # $k$ is the block rank index (0..6). $l$ is the internal rank index (0..6).
    # $U_4[r, K] = U_2[I \times 2 + ii, k] \times U_2[J \times 2 + jj, l]$?
    # Wait, $U_2$ has rows indexed by $2 \times 2$ entries.
    # Row $2 \times 2$ entry $a_{bc}$ is at index $b \times 2 + c$.
    # So $U_2$ row for block $I, J$? No, $U_2$ row is for a specific $A_{bc}$.
    # In the Kronecker product, $U_4 = U_2 \otimes U_2$.
    # Row $(i,j)$ of $U_4$ corresponds to $(I, ii)$ and $(J, jj)$?
    # Actually, standard Kronecker:
    # $U_4$ row $u_4$ (0..15). $u_4 = u_{2,1} \times 4 + u_{2,2}$?
    # If $U_2$ is 4x7, $U_4$ is 16x49.
    # $U_4[i \times 4 + j, k \times 7 + l] = U_2[i \times 2 + (j//2)? No.
    
    # Let's use NumPy for the Kronecker product to be safe.
    U2_np = np.array(U2)
    V2_np = np.array(V2)
    W2_np = np.array(W2)
    
    U4_np = np.kron(U2_np, U2_np)
    V4_np = np.kron(V2_np, V2_np)
    W4_np = np.kron(W2_np, W2_np)
    
    # Fill global U, V, W for sub-problem 1 (4x4x4)
    # A_L is A[:, 0..3]. B_TL is B[0..3, 0..3]. C_TL is C[:, 0..3].
    for u_row in range(16):
        i = u_row // 4
        j = u_row % 4
        # Global U index: i * 5 + j
        g_u = i * 5 + j
        for k in range(49):
            U[g_u][k] += U4_np[u_row, k]
            
    for v_row in range(16):
        j = v_row // 4
        k = v_row % 4
        # Global V index: j * 5 + k
        g_v = j * 5 + k
        for l in range(49):
            V[g_v][l] += V4_np[v_row, l]
            
    for w_row in range(16):
        k = w_row // 4
        i = w_row % 4
        # Global W index: k * 4 + i
        g_w = k * 4 + i
        for m in range(49):
            W[g_w][m] += W4_np[w_row, m]
            
    # Sub-problem 2: 4x4x1 (rank 16)
    # A_L (4x4) * B_TR (4x1) -> C_TR (4x1)
    # Schoolbook
    # Rank indices 49..64
    for i in range(4):
        for j in range(4):
            r = 49 + i * 4 + j
            # U: A[i, j] -> i*5 + j
            U[i * 5 + j][r] = 1
            # V: B[j, 0] -> j*5 + 0
            V[j * 5 + 0][r] = 1
            # W: C[i, 0] -> 0*4 + i = i
            W[0 * 4 + i][r] = 1

    # Sub-problem 3: 4x1x4 (rank 16)
    # A_R (4x1) * B_BL (1x4) -> C_BL (4x4)
    # Schoolbook
    # Rank indices 65..80
    for i in range(4):
        for k in range(4):
            r = 65 + i * 4 + k
            # U: A[i, 4] -> i*5 + 4
            U[i * 5 + 4][r] = 1
            # V: B[4, k] -> 4*5 + k = 20 + k
            V[4 * 5 + k][r] = 1
            # W: C[i, k] -> k*4 + i
            W[k * 4 + i][r] = 1

    # Sub-problem 4: 4x1x1 (rank 4)
    # A_R (4x1) * B_BR (1x1) -> C_BR (4x1)
    # Schoolbook
    # Rank indices 81..84
    for i in range(4):
        r = 81 + i
        # U: A[i, 4] -> i*5 + 4
        U[i * 5 + 4][r] = 1
        # V: B[4, 4] -> 4*5 + 4 = 24
        V[4 * 5 + 4][r] = 1
        # W: C[i, 4] -> 4*4 + i = 16 + i
        W[4 * 4 + i][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
