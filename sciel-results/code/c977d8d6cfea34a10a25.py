DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    rank = 84  # 12 Strassen multiplications * 7 each
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84
    
    def a_idx(ai, aj, r, c):
        return (ai * 2 + r) * m + (aj * 2 + c)
    
    def b_idx(bi, bj, r, c):
        return (bi * 2 + r) * p + (bj * 2 + c)
    
    def c_idx(ci, cj, r, c):
        return (cj * 2 + c) * n + (ci * 2 + r)
    
    def add_strassen(ai, aj, bi, bj, ci, cj, col_offset):
        nonlocal col
        # M1 = (A11+A22)(B11+B22)
        U[a_idx(ai,aj,0,0)][col] += 1
        U[a_idx(ai,aj,1,1)][col] += 1
        V[b_idx(bi,bj,0,0)][col] += 1
        V[b_idx(bi,bj,1,1)][col] += 1
        W[c_idx(ci,cj,0,0)][col] += 1
        W[c_idx(ci,cj,1,1)][col] += 1
        col += 1
        
        # M2 = (A21+A22)B11
        U[a_idx(ai,aj,1,0)][col] += 1
        U[a_idx(ai,aj,1,1)][col] += 1
        V[b_idx(bi,bj,0,0)][col] += 1
        W[c_idx(ci,cj,1,0)][col] += 1
        col += 1
        
        # M3 = A11(B12-B22)
        U[a_idx(ai,aj,0,0)][col] += 1
        V[b_idx(bi,bj,0,1)][col] += 1
        V[b_idx(bi,bj,1,1)][col] += -1
        W[c_idx(ci,cj,0,1)][col] += 1
        W[c_idx(ci,cj,1,1)][col] += 1
        col += 1
        
        # M4 = A22(B21-B11)
        U[a_idx(ai,aj,1,1)][col] += 1
        V[b_idx(bi,bj,1,0)][col] += 1
        V[b_idx(bi,bj,0,0)][col] += -1
        W[c_idx(ci,cj,0,0)][col] += 1
        W[c_idx(ci,cj,1,0)][col] += 1
        col += 1
        
        # M5 = (A11+A12)B22
        U[a_idx(ai,aj,0,0)][col] += 1
        U[a_idx(ai,aj,0,1)][col] += 1
        V[b_idx(bi,bj,1,1)][col] += 1
        W[c_idx(ci,cj,0,0)][col] += -1
        W[c_idx(ci,cj,0,1)][col] += 1
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[a_idx(ai,aj,1,0)][col] += 1
        U[a_idx(ai,aj,0,0)][col] += -1
        V[b_idx(bi,bj,0,0)][col] += 1
        V[b_idx(bi,bj,0,1)][col] += 1
        W[c_idx(ci,cj,1,1)][col] += 1
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[a_idx(ai,aj,0,1)][col] += 1
        U[a_idx(ai,aj,1,1)][col] += -1
        V[b_idx(bi,bj,1,0)][col] += 1
        V[b_idx(bi,bj,1,1)][col] += 1
        W[c_idx(ci,cj,0,0)][col] += 1
        W[c_idx(ci,cj,1,1)][col] += -1
        col += 1
    
    col = 0
    for ci in range(2):
        for cj in range(3):
            # C[ci][cj] = A[ci][0]*B[0][cj] + A[ci][1]*B[1][cj]
            add_strassen(ci, 0, 0, cj, ci, cj, col)
            add_strassen(ci, 1, 1, cj, ci, cj, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
