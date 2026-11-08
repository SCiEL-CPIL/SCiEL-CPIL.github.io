"""Matrix multiplication 2,4,5 — rank-minimal tensor decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0

    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_exact(U_list, V_list, W_list, R):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r]
                    if s != 1:
                        return False
        return True

    # Strassen Rank 7 for 2x2x2
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    def get_strassen_222():
        U = [
            [0, 0, 0, 0, 0, 0, 0], # 00
            [0, 0, 0, 0, 0, 0, 0], # 01
            [0, 0, 0, 0, 0, 0, 0], # 10
            [0, 0, 0, 0, 0, 0, 0], # 11
        ]
        V = [
            [0, 0, 0, 0, 0, 0, 0], # 00
            [0, 0, 0, 0, 0, 0, 0], # 01
            [0, 0, 0, 0, 0, 0, 0], # 10
            [0, 0, 0, 0, 0, 0, 0], # 11
        ]
        W = [
            [0, 0, 0, 0, 0, 0, 0], # 00
            [0, 0, 0, 0, 0, 0, 0], # 01
            [0, 0, 0, 0, 0, 0, 0], # 10
            [0, 0, 0, 0, 0, 0, 0], # 11
        ]
        
        # M1 = (X00 + X11)(Y00 + Y11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = -1 # Z00 += M1, Z01 += M1, Z10 += M1, Z11 -= M1? 
        # Wait, Strassen:
        # Z00 = M1 + M4 - M5 + M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        
        # Let's map carefully.
        # M1: (X00+X11)(Y00+Y11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1 # Z00
        W[1][0] = 0 # Z01
        W[2][0] = 0 # Z10
        W[3][0] = 1 # Z11

        # M2: (X10+X11)Y00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1 # Z10
        W[3][1] = -1 # Z11

        # M3: X00(Y01-Y11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1 # Z01
        W[3][2] = 1 # Z11

        # M4: X11(Y10-Y00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1 # Z00
        W[2][3] = 1 # Z10

        # M5: (X00+X01)Y11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1 # Z00
        W[1][4] = 1 # Z01

        # M6: (X10-X00)(Y00+Y01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1 # Z11

        # M7: (X01-X11)(Y10+Y11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1 # Z00

        return U, V, W

    # Rank 11 for 2x2x3
    # U: (4, 11), V: (6, 11), W: (6, 11)
    # Split into Strassen (7) + 4 terms for the 3rd column
    def get_rank11_223():
        U_str, V_str, W_str = get_strassen_222()
        U = [row[:] for row in U_str]
        V = [row[:] for row in V_str]
        W = [row[:] for row in W_str]
        
        # Extend to 11 columns
        for i in range(4):
            U[i] = U[i] + [0, 0, 0, 0]
        for i in range(6):
            V[i] = V[i] + [0, 0, 0, 0]
        for i in range(6):
            W[i] = W[i] + [0, 0, 0, 0]
            
        # The 3rd column of Y (indices 2, 5 in V, 2, 5 in W)
        # Z02 = X00*Y02 + X01*Y12
        # Z12 = X10*Y02 + X11*Y12
        
        # Term 8: X00*Y02
        U[0][7] = 1
        V[2][7] = 1
        W[2][7] = 1 # Z02
        
        # Term 9: X01*Y12
        U[1][8] = 1
        V[5][8] = 1
        W[2][8] = 1 # Z02
        
        # Term 10: X10*Y02
        U[2][9] = 1
        V[2][9] = 1
        W[5][9] = 1 # Z12
        
        # Term 11: X11*Y12
        U[3][10] = 1
        V[5][10] = 1
        W[5][10] = 1 # Z12
        
        return U, V, W

    def assemble():
        # Total Rank = 22 + 14 = 36
        R = 36
        U = [[0] * R for _ in range(n * m)] # 8 x 36
        V = [[0] * R for _ in range(m * p)] # 20 x 36
        W = [[0] * R for _ in range(p * n)] # 10 x 36
        
        # Part 1: A B3 (Rank 22)
        # A1 B31 (Rank 11)
        U1, V1, W1 = get_rank11_223()
        # A1 is rows 0,1,2,3 of U (cols 0,1 of A)
        # B31 is rows 0,1,2,3,4,5 of V (cols 0,1 of B rows, cols 0,1,2 of B cols)
        # W1 is cols 0,1,2 of W (rows 0,1 of W)
        
        for r in range(11):
            for u_idx in range(4):
                val = U1[u_idx][r]
                if val != 0:
                    # Map u_idx (0..3) to U row (0..3)
                    U[u_idx][r] += val
            for v_idx in range(6):
                val = V1[v_idx][r]
                if val != 0:
                    # Map v_idx (0..5) to V row (0..5)
                    V[v_idx][r] += val
            for w_idx in range(6):
                val = W1[w_idx][r]
                if val != 0:
                    # Map w_idx (0..5) to W row (0..5)
                    # W index k*n+i. k in 0..2, i in 0..1.
                    # W1 is 2x3 output? No, W is p x n.
                    # W1 is for 2x2x3. Output is 2x3.
                    # W1 indices: k (0..2) * n (2) + i (0..1).
                    # So w_idx 0..5 maps to W rows 0..5.
                    W[w_idx][r] += val

        # A2 B32 (Rank 11)
        # A2 is rows 4,5,6,7 of U (cols 2,3 of A)
        # B32 is rows 6,7,8,9,10,11 of V (rows 2,3 of B, cols 0,1,2 of B cols)
        # W2 is cols 0,1,2 of W (rows 0,1 of W)
        
        for r in range(11):
            for u_idx in range(4):
                val = U1[u_idx][r]
                if val != 0:
                    U[4 + u_idx][11 + r] += val
            for v_idx in range(6):
                val = V1[v_idx][r]
                if val != 0:
                    V[6 + v_idx][11 + r] += val
            for w_idx in range(6):
                val = W1[w_idx][r]
                if val != 0:
                    W[w_idx][11 + r] += val

        # Part 2: A B2 (Rank 14)
        # A1 B21 (Rank 7)
        U2, V2, W2 = get_strassen_222()
        # A1 is rows 0..3
        # B21 is rows 12..15 of V (rows 0,1 of B, cols 3,4 of B cols)
        # W2 is cols 3,4 of W (rows 0,1 of W)
        
        for r in range(7):
            for u_idx in range(4):
                val = U2[u_idx][r]
                if val != 0:
                    U[u_idx][22 + r] += val
            for v_idx in range(4):
                val = V2[v_idx][r]
                if val != 0:
                    # V2 is 2x2. v_idx 0..3.
                    # B21 corresponds to B rows 0,1 and cols 3,4.
                    # V index = j*p + k. j in 0,1. k in 3,4.
                    # v_idx 0 -> j=0,k=0? No.
                    # V2 is for 2x2 multiplication.
                    # V2 indices: j (0..1) * 2 + k (0..1).
                    # Map to V: j=0..1, k=3..4.
                    j = v_idx // 2
                    k = v_idx % 2
                    V[j * p + (3 + k)][22 + r] += val
            for w_idx in range(4):
                val = W2[w_idx][r]
                if val != 0:
                    # W2 indices: k (0..1) * 2 + i (0..1).
                    # Map to W: k=3..4, i=0..1.
                    k = w_idx // 2
                    i = w_idx % 2
                    W[(3 + k) * n + i][22 + r] += val

        # A2 B22 (Rank 7)
        # A2 is rows 4..7
        # B22 is rows 16..19 of V (rows 2,3 of B, cols 3,4 of B cols)
        # W2 is cols 3,4 of W (rows 0,1 of W)
        
        for r in range(7):
            for u_idx in range(4):
                val = U2[u_idx][r]
                if val != 0:
                    U[4 + u_idx][29 + r] += val
            for v_idx in range(4):
                val = V2[v_idx][r]
                if val != 0:
                    j = v_idx // 2
                    k = v_idx % 2
                    V[(2 + j) * p + (3 + k)][29 + r] += val
            for w_idx in range(4):
                val = W2[w_idx][r]
                if val != 0:
                    k = w_idx // 2
                    i = w_idx % 2
                    W[(3 + k) * n + i][29 + r] += val

        return {"U": U, "V": V, "W": W}

    result = assemble()
    
    # Verify
    if verify_exact(result["U"], result["V"], result["W"], 36):
        return result
    else:
        return make_schoolbook()

# EVOLVE-BLOCK-END
