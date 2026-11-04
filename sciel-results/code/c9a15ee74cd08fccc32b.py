"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 33 decomposition:
    # Split m=3 into m1=1 and m2=2.
    # Part 1: A1 (2x1) * B1 (1x6) -> Rank 12
    # Part 2: A2 (2x2) * B2 (2x6) -> Split B2 into three 2x2 blocks.
    #         Each 2x2*2x2 uses Strassen (Rank 7). Total 3 * 7 = 21.
    # Total Rank = 12 + 21 = 33.
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)] # 6 x 33
    V = [[0] * rank for _ in range(m * p)] # 18 x 33
    W = [[0] * rank for _ in range(p * n)] # 12 x 33
    
    # Part 1: 2x1x6 multiplication
    # A1 is A[0,0] and A[1,0] (indices 0 and 1 in U)
    # B1 is B[0, 0..5] (indices 0..5 in V)
    # Output C is C[0..5, 0] and C[0..5, 1] (indices k*n+i in W)
    
    col = 0
    for i in range(n): # 2 rows
        for j in range(p): # 6 cols
            # Term: A[i, 0] * B[0, j]
            U[i * m + 0][col] = 1
            V[0 * p + j][col] = 1
            W[j * n + i][col] = 1
            col += 1
            
    # Part 2: 2x2x6 multiplication
    # A2 is A[i, 1] and A[i, 2] for i in 0..1
    # B2 is B[k, j] for k in 1..2, j in 0..5
    # Split B2 into 3 blocks of 2 columns each.
    
    for block_idx in range(3):
        # Strassen for 2x2x2
        # A2 = [[a, b], [c, d]]
        # a = A[0,1] (idx 1), b = A[0,2] (idx 2)
        # c = A[1,1] (idx 4), d = A[1,2] (idx 5)
        
        # B2 block = [[x, y], [u, v]]
        # x = B[1, 2*block_idx], y = B[1, 2*block_idx+1]
        # u = B[2, 2*block_idx], v = B[2, 2*block_idx+1]
        
        # Indices in V:
        # x_idx = 1*p + 2*block_idx = 6 + 2*block_idx
        # y_idx = 1*p + 2*block_idx+1 = 6 + 2*block_idx + 1
        # u_idx = 2*p + 2*block_idx = 12 + 2*block_idx
        # v_idx = 2*p + 2*block_idx+1 = 12 + 2*block_idx + 1
        
        x_idx = 6 + 2 * block_idx
        y_idx = x_idx + 1
        u_idx = 12 + 2 * block_idx
        v_idx = u_idx + 1
        
        # Strassen terms
        # m1 = (a+d)(x+u)
        # m2 = (c+d)x
        # m3 = a(y-u)
        # m4 = d(v-u)
        # m5 = (a+b)u
        # m6 = (c-a)(x+v)
        # m7 = (b-d)(y+v)
        
        # m1
        U[1][col] = 1; U[5][col] = 1
        V[x_idx][col] = 1; V[u_idx][col] = 1
        # C00 += m1, C11 += m1
        W[0 * n + 0][col] = 1 # C[0,0] of block -> global C[2*block_idx, 0]
        W[(2 * block_idx + 1) * n + 1][col] = 1 # C[1,1] of block -> global C[2*block_idx+1, 1]
        col += 1
        
        # m2
        U[4][col] = 1; U[5][col] = 1
        V[x_idx][col] = 1
        # C10 += m2, C11 -= m2
        W[(2 * block_idx) * n + 1][col] = 1 # C[0,1] of block? No.
        # Strassen: C10 = m2 + m4. C11 = m1 - m2 + m3 + m6.
        # C10 is global C[2*block_idx, 1].
        # C11 is global C[2*block_idx+1, 1].
        W[(2 * block_idx) * n + 1][col] = 1
        W[(2 * block_idx + 1) * n + 1][col] = -1
        col += 1
        
        # m3
        U[1][col] = 1
        V[y_idx][col] = 1; V[u_idx][col] = -1
        # C01 += m3, C11 += m3
        W[(2 * block_idx + 1) * n + 0][col] = 1 # C[1,0] of block -> global C[2*block_idx+1, 0]
        W[(2 * block_idx + 1) * n + 1][col] = 1
        col += 1
        
        # m4
        U[5][col] = 1
        V[v_idx][col] = 1; V[u_idx][col] = -1
        # C00 += m4, C10 += m4
        W[(2 * block_idx) * n + 0][col] = 1
        W[(2 * block_idx) * n + 1][col] = 1
        col += 1
        
        # m5
        U[1][col] = 1; U[2][col] = 1
        V[u_idx][col] = 1
        # C00 -= m5, C01 += m5
        W[(2 * block_idx) * n + 0][col] = -1
        W[(2 * block_idx + 1) * n + 0][col] = 1
        col += 1
        
        # m6
        U[4][col] = 1; U[1][col] = -1
        V[x_idx][col] = 1; V[v_idx][col] = 1
        # C11 += m6
        W[(2 * block_idx + 1) * n + 1][col] = 1
        col += 1
        
        # m7
        U[2][col] = 1; U[5][col] = -1
        V[y_idx][col] = 1; V[v_idx][col] = 1
        # C00 += m7
        W[(2 * block_idx) * n + 0][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
