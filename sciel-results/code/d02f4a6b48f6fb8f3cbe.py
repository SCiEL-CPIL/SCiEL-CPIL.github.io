"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 29 decomposition:
    # 1. M_{2,1,5} part (j=0): Rank 10
    # 2. M_{2,2,5} part (j=1,2) split into:
    #    a. M_{2,2,2} (k=0,1) using Strassen: Rank 7
    #    b. M_{2,2,3} (k=2,3,4) using Schoolbook: Rank 12
    
    rank = 29
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j=0
    # C[i,k] = A[i,0] * B[0,k]
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2a: Strassen for A[:, 1:3] * B[1:3, 0:2]
    # Strassen components for 2x2x2
    # Indices in local 2x2 space:
    # A: (0,0), (0,1), (1,0), (1,1) -> Global A: (0,1), (0,2), (1,1), (1,2)
    # B: (0,0), (0,1), (1,0), (1,1) -> Global B: (1,0), (1,1), (2,0), (2,1)
    # C: (0,0), (0,1), (1,0), (1,1) -> Global C: (0,0), (0,1), (1,0), (1,1)
    
    # Strassen Multiplications M1..M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Output C:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_coeffs = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u_coeffs for A (local 2x2): [A00, A01, A10, A11]
        # v_coeffs for B (local 2x2): [B00, B01, B10, B11]
        # w_coeffs for C (local 2x2): [C00, C01, C10, C11]
        
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 0, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]),
    ]
    
    # Mapping local indices to global
    # A local (li, lj) -> Global (li, 1+lj)
    # B local (li, lj) -> Global (1+li, lj)
    # C local (li, lj) -> Global (li, lj)
    
    for u_c, v_c, w_c in strassen_coeffs:
        for li in range(2):
            for lj in range(2):
                if u_c[li*2+lj] != 0:
                    U[li * m + (1 + lj)][col] += u_c[li*2+lj]
                if v_c[li*2+lj] != 0:
                    V[(1 + li) * p + lj][col] += v_c[li*2+lj]
                if w_c[li*2+lj] != 0:
                    W[lj * n + li][col] += w_c[li*2+lj]
        col += 1
        
    # Part 2b: Schoolbook for A[:, 1:3] * B[1:3, 2:5]
    # i in 0..1, j in 1..2, k in 2..4
    for i in range(n):
        for j in range(1, m):
            for k in range(2, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
