import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank calculation
    # 4 Strassen blocks (2x2x2x2) -> 4 * 7 = 28
    # A12 (2x1) * BB (1x8) -> 2 * 8 = 16
    # A21 (1x2) * BT (2x8) -> 2 * 8 = 16
    # A22 (1x1) * BB (1x8) -> 1 * 8 = 8
    # Total = 68
    
    rank = 68
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1

    # Part 1: 4 Strassen Blocks
    # A11 is A[0:2, 0:2]
    # BT is B[0:2, 0:8] split into 4 blocks of 2x2
    # Block j uses B columns 2j, 2j+1
    # Output C rows 0,1 columns 2j, 2j+1
    
    for j in range(4):
        # U indices for A11
        # a00: A[0,0] -> 0
        # a01: A[0,1] -> 1
        # a10: A[1,0] -> 3
        # a11: A[1,1] -> 4
        
        # V indices for BT_j
        # b00: B[0, 2j] -> 2j
        # b01: B[0, 2j+1] -> 2j+1
        # b10: B[1, 2j] -> 8+2j
        # b11: B[1, 2j+1] -> 9+2j
        
        # W indices for C block
        # c00: C[0, 2j] -> (2j)*3 + 0 = 6j
        # c01: C[0, 2j+1] -> (2j+1)*3 + 0 = 6j+1
        # c10: C[1, 2j] -> (2j)*3 + 1 = 6j+3
        # c11: C[1, 2j+1] -> (2j+1)*3 + 1 = 6j+4
        
        u00, u01, u10, u11 = 0, 1, 3, 4
        v00, v01, v10, v11 = 2*j, 2*j+1, 8+2*j, 9+2*j
        w00, w01, w10, w11 = 6*j, 6*j+1, 6*j+3, 6*j+4
        
        # M1 = (a00 + a11)(b00 + b11)
        # U: u00+1, u11+1
        # V: v00+1, v11+1
        # W: w00+1, w11+1
        U[u00][col] += 1; U[u11][col] += 1
        V[v00][col] += 1; V[v11][col] += 1
        W[w00][col] += 1; W[w11][col] += 1
        col += 1
        
        # M2 = (a10 + a11)b00
        # U: u10+1, u11+1
        # V: v00+1
        # W: w10+1, w11-1
        U[u10][col] += 1; U[u11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1; W[w11][col] -= 1
        col += 1
        
        # M3 = a00(b01 - b11)
        # U: u00+1
        # V: v01+1, v11-1
        # W: w01+1, w11+1
        U[u00][col] += 1
        V[v01][col] += 1; V[v11][col] -= 1
        W[w01][col] += 1; W[w11][col] += 1
        col += 1
        
        # M4 = a11(b10 - b00)
        # U: u11+1
        # V: v10+1, v00-1
        # W: w00+1, w10+1
        U[u11][col] += 1
        V[v10][col] += 1; V[v00][col] -= 1
        W[w00][col] += 1; W[w10][col] += 1
        col += 1
        
        # M5 = (a00 + a01)b11
        # U: u00+1, u01+1
        # V: v11+1
        # W: w00-1, w01+1
        U[u00][col] += 1; U[u01][col] += 1
        V[v11][col] += 1
        W[w00][col] -= 1; W[w01][col] += 1
        col += 1
        
        # M6 = (a10 - a00)(b00 + b01)
        # U: u10+1, u00-1
        # V: v00+1, v01+1
        # W: w11+1
        U[u10][col] += 1; U[u00][col] -= 1
        V[v00][col] += 1; V[v01][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M7 = (a01 - a11)(b10 + b11)
        # U: u01+1, u11-1
        # V: v10+1, v11+1
        # W: w00+1
        U[u01][col] += 1; U[u11][col] -= 1
        V[v10][col] += 1; V[v11][col] += 1
        W[w00][col] += 1
        col += 1

    # Part 2: A12 * BB
    # A12 is A[0:2, 2]
    # a0: A[0,2] -> 2
    # a1: A[1,2] -> 5
    # BB is B[2, 0:8]
    # bk: B[2, k] -> 16+k
    # Output C[0:2, 0:8]
    # c0k: C[0, k] -> 3k
    # c1k: C[1, k] -> 3k+1
    
    for k in range(8):
        # a0 * bk -> c0k
        U[2][col] += 1
        V[16+k][col] += 1
        W[3*k][col] += 1
        col += 1
        
        # a1 * bk -> c1k
        U[5][col] += 1
        V[16+k][col] += 1
        W[3*k+1][col] += 1
        col += 1

    # Part 3: A21 * BT
    # A21 is A[2, 0:2]
    # a0: A[2,0] -> 6
    # a1: A[2,1] -> 7
    # BT is B[0:2, 0:8]
    # b0k: B[0, k] -> k
    # b1k: B[1, k] -> 8+k
    # Output C[2, 0:8]
    # c2k: C[2, k] -> 3k+2
    
    for k in range(8):
        # a0 * b0k -> c2k
        U[6][col] += 1
        V[k][col] += 1
        W[3*k+2][col] += 1
        col += 1
        
        # a1 * b1k -> c2k
        U[7][col] += 1
        V[8+k][col] += 1
        W[3*k+2][col] += 1
        col += 1

    # Part 4: A22 * BB
    # A22 is A[2, 2]
    # a: A[2,2] -> 8
    # BB is B[2, 0:8]
    # bk: B[2, k] -> 16+k
    # Output C[2, 0:8]
    # c2k: C[2, k] -> 3k+2
    
    for k in range(8):
        # a * bk -> c2k
        U[8][col] += 1
        V[16+k][col] += 1
        W[3*k+2][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
