"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 decomposition
    U_s = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, -1],
        [1, 1, 0, 1, 0, 0, 0]
    ]
    V_s = [
        [1, 1, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ]
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    rank_s = 7
    
    # Schoolbook 2x2x3 decomposition
    U_sb = [[0]*12 for _ in range(4)]
    V_sb = [[0]*12 for _ in range(6)]
    W_sb = [[0]*12 for _ in range(6)]
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U_sb[i*2+j][col] = 1
                V_sb[j*3+k][col] = 1
                W_sb[k*2+i][col] = 1
                col += 1
    rank_sb = 12
    
    total_rank = rank_s * rank_sb
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    for c_s in range(rank_s):
        for c_sb in range(rank_sb):
            c = c_s * rank_sb + c_sb
            
            for bi in range(2):
                for ei in range(2):
                    for bj in range(2):
                        for ej in range(2):
                            coeff = U_s[bi*2+bj][c_s] * U_sb[ei*2+ej][c_sb]
                            if coeff != 0:
                                row_u = (bi*2+ei)*m + (bj*2+ej)
                                U[row_u][c] += coeff
            
            for bj in range(2):
                for ej in range(2):
                    for ck in range(3):
                        for ek in range(3):
                            coeff = V_s[bj*2+ck][c_s] * V_sb[ej*3+ek][c_sb]
                            if coeff != 0:
                                row_v = (bj*2+ej)*p + (ck*3+ek)
                                V[row_v][c] += coeff
            
            for bi in range(2):
                for ei in range(2):
                    for ck in range(3):
                        for ek in range(3):
                            coeff = W_s[bi*2+ck][c_s] * W_sb[ei*3+ek][c_sb]
                            if coeff != 0:
                                row_w = (ck*3+ek)*n + (bi*2+ei)
                                W[row_w][c] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
