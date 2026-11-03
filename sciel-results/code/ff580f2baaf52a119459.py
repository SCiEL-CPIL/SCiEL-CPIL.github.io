"""Matrix multiplication 4x4x6 with Strassen-based rank-84 decomposition."""
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 6
    rank = 84
    U = [[0]*rank for _ in range(n*m)]  # 16 x 84
    V = [[0]*rank for _ in range(m*p)]  # 24 x 84
    W = [[0]*rank for _ in range(p*n)]  # 24 x 84
    
    def add_strassen(a_row, a_col, b_row, b_col, c_row, c_col, col_start):
        a11 = a_row*4 + a_col
        a12 = a_row*4 + a_col + 1
        a21 = (a_row+1)*4 + a_col
        a22 = (a_row+1)*4 + a_col + 1
        b11 = b_row*6 + b_col
        b12 = b_row*6 + b_col + 1
        b21 = (b_row+1)*6 + b_col
        b22 = (b_row+1)*6 + b_col + 1
        c11 = c_col*4 + c_row
        c12 = (c_col+1)*4 + c_row
        c21 = c_col*4 + (c_row+1)
        c22 = (c_col+1)*4 + (c_row+1)
        
        U[a11][col_start] = 1; U[a22][col_start] = 1
        V[b11][col_start] = 1; V[b22][col_start] = 1
        W[c11][col_start] = 1; W[c22][col_start] = 1
        
        U[a21][col_start+1] = 1; U[a22][col_start+1] = 1
        V[b11][col_start+1] = 1
        W[c21][col_start+1] = 1; W[c22][col_start+1] = -1
        
        U[a11][col_start+2] = 1
        V[b12][col_start+2] = 1; V[b22][col_start+2] = -1
        W[c12][col_start+2] = 1; W[c22][col_start+2] = 1
        
        U[a22][col_start+3] = 1
        V[b21][col_start+3] = 1; V[b11][col_start+3] = -1
        W[c11][col_start+3] = 1; W[c21][col_start+3] = 1
        
        U[a11][col_start+4] = 1; U[a12][col_start+4] = 1
        V[b22][col_start+4] = 1
        W[c11][col_start+4] = -1; W[c12][col_start+4] = 1
        
        U[a21][col_start+5] = 1; U[a11][col_start+5] = -1
        V[b11][col_start+5] = 1; V[b12][col_start+5] = 1
        W[c22][col_start+5] = 1
        
        U[a12][col_start+6] = 1; U[a22][col_start+6] = -1
        V[b21][col_start+6] = 1; V[b22][col_start+6] = 1
        W[c11][col_start+6] = 1; W[c12][col_start+6] = 1
    
    blocks = [
        (0,0, 0,0, 0,0),
        (0,2, 2,0, 0,0),
        (0,0, 0,2, 0,2),
        (0,2, 2,2, 0,2),
        (0,0, 0,4, 0,4),
        (0,2, 2,4, 0,4),
        (2,0, 0,0, 2,0),
        (2,2, 2,0, 2,0),
        (2,0, 0,2, 2,2),
        (2,2, 2,2, 2,2),
        (2,0, 0,4, 2,4),
        (2,2, 2,4, 2,4),
    ]
    
    for idx, (ar, ac, br, bc, cr, cc) in enumerate(blocks):
        add_strassen(ar, ac, br, bc, cr, cc, idx * 7)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
