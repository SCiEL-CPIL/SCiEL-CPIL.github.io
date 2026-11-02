"""Strassen-based rank-35 decomposition for 2x2 * 2x10 matrix multiplication."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    num_blocks = p // m   # 5 blocks of size 2x2
    rank = num_blocks * 7 # 35
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 35)
    V = [[0] * rank for _ in range(m * p)]  # (20, 35)
    W = [[0] * rank for _ in range(p * n)]  # (20, 35)
    
    for b in range(num_blocks):
        base = 7 * b
        # For block b, the B columns are at positions 2b and 2b+1
        # k indices: 2b (col 0 of block) and 2b+1 (col 1 of block)
        
        # Strassen's 7 products:
        # s=0: M1 = (a00+a11)*(b00+b11)
        # s=1: M2 = (a10+a11)*b00
        # s=2: M3 = a00*(b01-b11)
        # s=3: M4 = a11*(b10-b00)
        # s=4: M5 = (a00+a01)*b11
        # s=5: M6 = (a10-a00)*(b00+b01)
        # s=6: M7 = (a01-a11)*(b10+b11)
        
        # U coefficients: row i*m + j, for i=0,1 and j=0,1
        # A = [[a00, a01], [a10, a11]]
        # U row 0 = i=0,j=0 = a00
        # U row 1 = i=0,j=1 = a01
        # U row 2 = i=1,j=0 = a10
        # U row 3 = i=1,j=1 = a11
        
        # M1: (a00+a11) -> U[0]=1, U[3]=1
        # M2: (a10+a11) -> U[2]=1, U[3]=1
        # M3: a00 -> U[0]=1
        # M4: a11 -> U[3]=1
        # M5: (a00+a01) -> U[0]=1, U[1]=1
        # M6: (a10-a00) -> U[2]=1, U[0]=-1
        # M7: (a01-a11) -> U[1]=1, U[3]=-1
        
        u_coeffs = [
            [1, 0, 1, 0, 1, -1, 0],   # a00 (row 0)
            [0, 0, 0, 0, 1,  0, 1],   # a01 (row 1)
            [0, 1, 0, 0, 0,  1, 0],   # a10 (row 2)
            [1, 1, 0, 1, 0,  0,-1],   # a11 (row 3)
        ]
        
        # V coefficients: row j*p + k
        # For block b, k = 2b (b00, b10) and k = 2b+1 (b01, b11)
        # V row j*p + k where j is column index of A (row index of B)
        # B_b = [[b00, b01], [b10, b11]]
        # j=0, k=2b: b00
        # j=0, k=2b+1: b01
        # j=1, k=2b: b10
        # j=1, k=2b+1: b11
        
        # M1: (b00+b11) -> V[0*p+2b]=1, V[1*p+2b+1]=1
        # M2: b00 -> V[0*p+2b]=1
        # M3: (b01-b11) -> V[0*p+2b+1]=1, V[1*p+2b+1]=-1
        # M4: (b10-b00) -> V[1*p+2b]=1, V[0*p+2b]=-1
        # M5: b11 -> V[1*p+2b+1]=1
        # M6: (b00+b01) -> V[0*p+2b]=1, V[0*p+2b+1]=1
        # M7: (b10+b11) -> V[1*p+2b]=1, V[1*p+2b+1]=1
        
        k0 = 2 * b
        k1 = 2 * b + 1
        
        v_coeffs = {
            (0, k0): [1, 1, 0, -1, 0, 1, 0],   # b00
            (0, k1): [0, 0, 1,  0, 0, 1, 0],   # b01
            (1, k0): [0, 0, 0,  1, 0, 0, 1],   # b10
            (1, k1): [1, 0,-1,  0, 1, 0, 1],   # b11
        }
        
        # W coefficients: row k*n + i, for k=0..p-1, i=0,1
        # C = A*B, so C[i][k] = sum_j A[i][j]*B[j][k]
        # W[k*n + i, c] is coefficient for C[i][k]
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For block b:
        # C[0][2b] = M1 + M4 - M5 - M7
        # C[0][2b+1] = M3 + M5
        # C[1][2b] = M2 + M4
        # C[1][2b+1] = M1 - M2 + M3 + M6
        
        # W row (2b)*2 + 0 = 4b: C[0][2b]
        # W row (2b)*2 + 1 = 4b+1: C[1][2b]
        # W row (2b+1)*2 + 0 = 4b+2: C[0][2b+1]
        # W row (2b+1)*2 + 1 = 4b+3: C[1][2b+1]
        
        w_coeffs = {
            (2*b, 0): [1,  0, 0, 1, -1, 0, -1],  # C00 = M1+M4-M5-M7
            (2*b, 1): [0,  1, 0, 1,  0, 0,  0],  # C10 = M2+M4
            (2*b+1, 0): [0, 0, 1, 0, 1,  0, 0],  # C01 = M3+M5
            (2*b+1, 1): [1, -1, 1, 0, 0, 1, 0],  # C11 = M1-M2+M3+M6
        }
        
        for s in range(7):
            c = base + s
            for i in range(n):
                for j in range(m):
                    U[i*m+j][c] = u_coeffs[i*m+j][s]
            for (j_idx, k_idx), coeff_list in v_coeffs.items():
                V[j_idx * p + k_idx][c] = coeff_list[s]
            for (k_idx, i_idx), coeff_list in w_coeffs.items():
                W[k_idx * n + i_idx][c] = coeff_list[s]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
