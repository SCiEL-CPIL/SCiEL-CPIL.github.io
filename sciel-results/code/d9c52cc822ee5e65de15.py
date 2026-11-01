import numpy as np

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for block in range(4):
        base_col = block * 14
        v_b11 = 2 * block
        v_b12 = 2 * block + 1
        v_b21 = 8 + 2 * block
        v_b22 = 9 + 2 * block
        v_b31 = 16 + 2 * block
        v_b32 = 17 + 2 * block
        v_b41 = 24 + 2 * block
        v_b42 = 25 + 2 * block
        
        w_c11 = 2 * block
        w_c12 = 2 * block + 1
        w_c21 = 2 * block + 2
        w_c22 = 2 * block + 3
        
        col = base_col + 0
        U[0][col] += 1
        U[5][col] += 1
        V[v_b11][col] += 1
        V[v_b22][col] += 1
        W[w_c11][col] += 1
        W[w_c22][col] += 1
        
        col = base_col + 1
        U[4][col] += 1
        U[5][col] += 1
        V[v_b11][col] += 1
        W[w_c21][col] += 1
        W[w_c22][col] += -1
        
        col = base_col + 2
        U[0][col] += 1
        V[v_b12][col] += 1
        V[v_b22][col] += -1
        W[w_c12][col] += 1
        W[w_c22][col] += 1
        
        col = base_col + 3
        U[5][col] += 1
        V[v_b21][col] += 1
        V[v_b11][col] += -1
        W[w_c11][col] += 1
        W[w_c21][col] += 1
        
        col = base_col + 4
        U[0][col] += 1
        U[1][col] += 1
        V[v_b22][col] += 1
        W[w_c11][col] += -1
        W[w_c12][col] += 1
        
        col = base_col + 5
        U[4][col] += 1
        U[0][col] += -1
        V[v_b11][col] += 1
        V[v_b12][col] += 1
        W[w_c22][col] += -1
        
        col = base_col + 6
        U[1][col] += 1
        U[5][col] += -1
        V[v_b21][col] += 1
        V[v_b22][col] += 1
        W[w_c11][col] += 1
        
        col = base_col + 7
        U[2][col] += 1
        U[7][col] += 1
        V[v_b31][col] += 1
        V[v_b42][col] += 1
        W[w_c11][col] += 1
        W[w_c22][col] += 1
        
        col = base_col + 8
        U[6][col] += 1
        U[7][col] += 1
        V[v_b31][col] += 1
        W[w_c21][col] += 1
        W[w_c22][col] += -1
        
        col = base_col + 9
        U[2][col] += 1
        V[v_b32][col] += 1
        V[v_b42][col] += -1
        W[w_c12][col] += 1
        W[w_c22][col] += 1
        
        col = base_col + 10
        U[7][col] += 1
        V[v_b41][col] += 1
        V[v_b31][col] += -1
        W[w_c11][col] += 1
        W[w_c21][col] += 1
        
        col = base_col + 11
        U[2][col] += 1
        U[3][col] += 1
        V[v_b42][col] += 1
        W[w_c11][col] += -1
        W[w_c12][col] += 1
        
        col = base_col + 12
        U[6][col] += 1
        U[2][col] += -1
        V[v_b31][col] += 1
        V[v_b32][col] += 1
        W[w_c22][col] += -1
        
        col = base_col + 13
        U[3][col] += 1
        U[7][col] += -1
        V[v_b41][col] += 1
        V[v_b42][col] += 1
        W[w_c11][col] += 1
        
    return {"U": U, "V": V, "W": W}
