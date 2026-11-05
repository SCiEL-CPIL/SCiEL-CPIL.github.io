DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 54
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: 1x4x7 (Rank 4)
    # i=0
    for r in range(4):
        U[0][col + r] = 1
        for k in range(7):
            W[k * n][col + r] = 1
            V[r * p + k][col + r] = 1
    col += 4
    
    # Block 2a: i=1,2; j=0,1; k=0..6 (Rank 25)
    # Block 2b: i=1,2; j=2,3; k=0..6 (Rank 25)
    
    def add_2x2x7(i_base, j_base, col_start):
        # i_base: 1, j_base: 0 or 2
        # U rows: i_base, i_base+1
        # V rows: j_base, j_base+1
        # W rows: k*n + i_base, k*n + i_base+1
        
        u00 = i_base * m + j_base
        u01 = i_base * m + (j_base + 1)
        u10 = (i_base + 1) * m + j_base
        u11 = (i_base + 1) * m + (j_base + 1)
        
        # Split k into 0,1; 2,3; 4,5; 6
        # 3 Strassen 2x2x2 blocks and 1 2x2x1 block
        
        # k=0,1
        v00 = j_base * p + 0
        v01 = j_base * p + 1
        v10 = (j_base + 1) * p + 0
        v11 = (j_base + 1) * p + 1
        
        w00 = 0 * n + i_base
        w01 = 1 * n + i_base
        w10 = 0 * n + (i_base + 1)
        w11 = 1 * n + (i_base + 1)
        
        # M1
        U[u00][col_start] += 1; U[u11][col_start] += 1
        V[v00][col_start] += 1; V[v11][col_start] += 1
        W[w00][col_start] += 1; W[w11][col_start] += 1
        col_start += 1
        # M2
        U[u10][col_start] += 1; U[u11][col_start] += 1
        V[v00][col_start] += 1
        W[w10][col_start] += 1
        col_start += 1
        # M3
        U[u00][col_start] += 1
        V[v10][col_start] += 1; V[v11][col_start] -= 1
        W[w01][col_start] += 1
        col_start += 1
        # M4
        U[u11][col_start] += 1
        V[v01][col_start] += 1; V[v00][col_start] -= 1
        W[w00][col_start] -= 1; W[w01][col_start] += 1
        col_start += 1
        # M5
        U[u00][col_start] += 1; U[u01][col_start] += 1
        V[v11][col_start] += 1
        W[w00][col_start] += 1; W[w10][col_start] += 1
        col_start += 1
        # M6
        U[u10][col_start] += 1; U[u00][col_start] -= 1
        V[v00][col_start] += 1; V[v01][col_start] += 1
        W[w11][col_start] -= 1
        col_start += 1
        # M7
        U[u01][col_start] += 1; U[u11][col_start] -= 1
        V[v10][col_start] += 1; V[v11][col_start] += 1
        W[w00][col_start] -= 1; W[w11][col_start] += 1
        col_start += 1
        
        # k=2,3
        v00 = j_base * p + 2
        v01 = j_base * p + 3
        v10 = (j_base + 1) * p + 2
        v11 = (j_base + 1) * p + 3
        
        w00 = 2 * n + i_base
        w01 = 3 * n + i_base
        w10 = 2 * n + (i_base + 1)
        w11 = 3 * n + (i_base + 1)
        
        # M1
        U[u00][col_start] += 1; U[u11][col_start] += 1
        V[v00][col_start] += 1; V[v11][col_start] += 1
        W[w00][col_start] += 1; W[w11][col_start] += 1
        col_start += 1
        # M2
        U[u10][col_start] += 1; U[u11][col_start] += 1
        V[v00][col_start] += 1
        W[w10][col_start] += 1
        col_start += 1
        # M3
        U[u00][col_start] += 1
        V[v10][col_start] += 1; V[v11][col_start] -= 1
        W[w01][col_start] += 1
        col_start += 1
        # M4
        U[u11][col_start] += 1
        V[v01][col_start] += 1; V[v00][col_start] -= 1
        W[w00][col_start] -= 1; W[w01][col_start] += 1
        col_start += 1
        # M5
        U[u00][col_start] += 1; U[u01][col_start] += 1
        V[v11][col_start] += 1
        W[w00][col_start] += 1; W[w10][col_start] += 1
        col_start += 1
        # M6
        U[u10][col_start] += 1; U[u00][col_start] -= 1
        V[v00][col_start] += 1; V[v01][col_start] += 1
        W[w11][col_start] -= 1
        col_start += 1
        # M7
        U[u01][col_start] += 1; U[u11][col_start] -= 1
        V[v10][col_start] += 1; V[v11][col_start] += 1
        W[w00][col_start] -= 1; W[w11][col_start] += 1
        col_start += 1
        
        # k=4,5
        v00 = j_base * p + 4
        v01 = j_base * p + 5
        v10 = (j_base + 1) * p + 4
        v11 = (j_base + 1) * p + 5
        
        w00 = 4 * n + i_base
        w01 = 5 * n + i_base
        w10 = 4 * n + (i_base + 1)
        w11 = 5 * n + (i_base + 1)
        
        # M1
        U[u00][col_start] += 1; U[u11][col_start] += 1
        V[v00][col_start] += 1; V[v11][col_start] += 1
        W[w00][col_start] += 1; W[w11][col_start] += 1
        col_start += 1
        # M2
        U[u10][col_start] += 1; U[u11][col_start] += 1
        V[v00][col_start] += 1
        W[w10][col_start] += 1
        col_start += 1
        # M3
        U[u00][col_start] += 1
        V[v10][col_start] += 1; V[v11][col_start] -= 1
        W[w01][col_start] += 1
        col_start += 1
        # M4
        U[u11][col_start] += 1
        V[v01][col_start] += 1; V[v00][col_start] -= 1
        W[w00][col_start] -= 1; W[w01][col_start] += 1
        col_start += 1
        # M5
        U[u00][col_start] += 1; U[u01][col_start] += 1
        V[v11][col_start] += 1
        W[w00][col_start] += 1; W[w10][col_start] += 1
        col_start += 1
        # M6
        U[u10][col_start] += 1; U[u00][col_start] -= 1
        V[v00][col_start] += 1; V[v01][col_start] += 1
        W[w11][col_start] -= 1
        col_start += 1
        # M7
        U[u01][col_start] += 1; U[u11][col_start] -= 1
        V[v10][col_start] += 1; V[v11][col_start] += 1
        W[w00][col_start] -= 1; W[w11][col_start] += 1
        col_start += 1
        
        # k=6 (2x1)
        # Rank 4: sum_{j=0}^1 A_i_j B_j_6
        # C_i_6 = A_i_0 B_0_6 + A_i_1 B_1_6
        # Terms:
        # 1: A_0_0 B_0_6 -> U[u00]=1, V[v00]=1, W[w00]=1
        # 2: A_0_1 B_1_6 -> U[u01]=1, V[v10]=1, W[w00]=1
        # 3: A_1_0 B_0_6 -> U[u10]=1, V[v00]=1, W[w10]=1
        # 4: A_1_1 B_1_6 -> U[u11]=1, V[v10]=1, W[w10]=1
        
        v06 = j_base * p + 6
        v16 = (j_base + 1) * p + 6
        
        w06 = 6 * n + i_base
        w16 = 6 * n + (i_base + 1)
        
        U[u00][col_start] += 1
        V[v06][col_start] += 1
        W[w06][col_start] += 1
        col_start += 1
        
        U[u01][col_start] += 1
        V[v16][col_start] += 1
        W[w06][col_start] += 1
        col_start += 1
        
        U[u10][col_start] += 1
        V[v06][col_start] += 1
        W[w16][col_start] += 1
        col_start += 1
        
        U[u11][col_start] += 1
        V[v16][col_start] += 1
        W[w16][col_start] += 1
        col_start += 1
        
        return col_start

    col = add_2x2x7(1, 0, col)
    col = add_2x2x7(1, 2, col)
    
    return {"U": U, "V": V, "W": W}
