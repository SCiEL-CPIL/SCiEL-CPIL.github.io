"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56
    
    # Strassen 2x2x2 base decomposition
    def strassen_2x2x2():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        # T0: (A00+A11)(B00+B11) -> C00+=1, C11+=1
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        # T1: (A10+A11)B00 -> C10+=1, C11-=1
        U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1; W[3][1]=-1
        # T2: A00(B10-B11) -> C01+=1, C11+=1
        U[0][2]=1; V[2][2]=1; V[3][2]=-1; W[1][2]=1; W[3][2]=1
        # T3: A11(B01-B00) -> C00+=1, C10+=1
        U[3][3]=1; V[1][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
        # T4: (A00+A01)B11 -> C00-=1, C01+=1
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
        # T5: (A10-A00)(B00+B10) -> C11+=1
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[2][5]=1; W[3][5]=1
        # T6: (A01-A11)(B01+B11) -> C00+=1
        U[1][6]=1; U[3][6]=-1; V[1][6]=1; V[3][6]=1; W[0][6]=1
        return U, V, W
    
    sU, sV, sW = strassen_2x2x2()
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for a_block in range(2):
        for b_col_block in range(4):
            for s in range(7):
                for i in range(2):
                    for jl in range(2):
                        j = 2 * a_block + jl
                        U[i * m + j][col] = sU[i * 2 + jl][s]
                for jl in range(2):
                    for kl in range(2):
                        j = 2 * a_block + jl
                        k = 2 * b_col_block + kl
                        V[j * p + k][col] = sV[jl * 2 + kl][s]
                for kl in range(2):
                    for i in range(2):
                        k = 2 * b_col_block + kl
                        W[k * n + i][col] = sW[kl * 2 + i][s]
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
