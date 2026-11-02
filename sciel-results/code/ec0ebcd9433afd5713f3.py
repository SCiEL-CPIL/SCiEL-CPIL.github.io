import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # rank = 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (x11+x22)(y11+y22)
    # M2 = (x21+x22)y11
    # M3 = x11(y12-y22)
    # M4 = x22(y21-y11)
    # M5 = (x11+x12)y22
    # M6 = (x21-x11)(y11+y12)
    # M7 = (x12-x22)(y21+y22)
    
    # Output reconstruction
    # z11 = M1 + M4 - M5 + M7
    # z12 = M3 + M5
    # z21 = M2 + M4
    # z22 = M1 - M2 + M3 + M6
    
    # Indices in U for A_L (j=0,1) and A_R (j=2,3)
    # A_L: x11=U[0], x12=U[1], x21=U[4], x22=U[5]
    # A_R: x11=U[2], x12=U[3], x21=U[6], x22=U[7]
    
    # Indices in V for B blocks
    # B is 4x8.
    # For k in 0..3:
    # B_1k is rows 0,1 cols 2k, 2k+1.
    # y11=V[2k], y12=V[2k+1], y21=V[8+2k], y22=V[8+2k+1]
    # B_2k is rows 2,3 cols 2k, 2k+1.
    # y11=V[16+2k], y12=V[16+2k+1], y21=V[24+2k], y22=V[24+2k+1]
    
    # Indices in W for C
    # C is 2x8.
    # C_0, 2k is W[2k], C_0, 2k+1 is W[2k+1]
    # C_1, 2k is W[4+2k], C_1, 2k+1 is W[4+2k+1]
    # Wait, W index is k*n + i.
    # W[col, row] in standard, but here W[k*n+i, r].
    # So W[0] is C[0,0], W[1] is C[0,1], W[2] is C[1,0], W[3] is C[1,1].
    # For block k:
    # C[0, 2k] -> W[2k]
    # C[0, 2k+1] -> W[2k+1]
    # C[1, 2k] -> W[2*2 + 2k] = W[4+2k]
    # C[1, 2k+1] -> W[4+2k+1]
    
    for k in range(4):
        # Base index for this block in U/V/W columns
        base_col = k * 14
        
        # --- Part 1: A_L * B_1k ---
        # U indices for A_L
        u_x11 = 0
        u_x12 = 1
        u_x21 = 4
        u_x22 = 5
        
        # V indices for B_1k
        v_y11 = 2 * k
        v_y12 = 2 * k + 1
        v_y21 = 8 + 2 * k
        v_y22 = 8 + 2 * k + 1
        
        # W indices for C block k
        w_z11 = 2 * k
        w_z12 = 2 * k + 1
        w_z21 = 4 + 2 * k
        w_z22 = 4 + 2 * k + 1
        
        # M1
        c = base_col + 0
        U[u_x11][c] += 1
        U[u_x22][c] += 1
        V[v_y11][c] += 1
        V[v_y22][c] += 1
        W[w_z11][c] += 1
        W[w_z22][c] += 1
        
        # M2
        c = base_col + 1
        U[u_x21][c] += 1
        U[u_x22][c] += 1
        V[v_y11][c] += 1
        W[w_z21][c] += 1
        W[w_z22][c] -= 1
        
        # M3
        c = base_col + 2
        U[u_x11][c] += 1
        V[v_y12][c] += 1
        V[v_y22][c] -= 1
        W[w_z12][c] += 1
        W[w_z22][c] += 1
        
        # M4
        c = base_col + 3
        U[u_x22][c] += 1
        V[v_y21][c] += 1
        V[v_y11][c] -= 1
        W[w_z11][c] += 1
        W[w_z21][c] += 1
        
        # M5
        c = base_col + 4
        U[u_x11][c] += 1
        U[u_x12][c] += 1
        V[v_y22][c] += 1
        W[w_z11][c] -= 1
        W[w_z12][c] += 1
        
        # M6
        c = base_col + 5
        U[u_x21][c] += 1
        U[u_x11][c] -= 1
        V[v_y11][c] += 1
        V[v_y12][c] += 1
        W[w_z22][c] += 1
        
        # M7
        c = base_col + 6
        U[u_x12][c] += 1
        U[u_x22][c] -= 1
        V[v_y21][c] += 1
        V[v_y22][c] += 1
        W[w_z11][c] += 1
        
        # --- Part 2: A_R * B_2k ---
        # U indices for A_R
        u_x11 = 2
        u_x12 = 3
        u_x21 = 6
        u_x22 = 7
        
        # V indices for B_2k
        v_y11 = 16 + 2 * k
        v_y12 = 16 + 2 * k + 1
        v_y21 = 24 + 2 * k
        v_y22 = 24 + 2 * k + 1
        
        # M1
        c = base_col + 7
        U[u_x11][c] += 1
        U[u_x22][c] += 1
        V[v_y11][c] += 1
        V[v_y22][c] += 1
        W[w_z11][c] += 1
        W[w_z22][c] += 1
        
        # M2
        c = base_col + 8
        U[u_x21][c] += 1
        U[u_x22][c] += 1
        V[v_y11][c] += 1
        W[w_z21][c] += 1
        W[w_z22][c] -= 1
        
        # M3
        c = base_col + 9
        U[u_x11][c] += 1
        V[v_y12][c] += 1
        V[v_y22][c] -= 1
        W[w_z12][c] += 1
        W[w_z22][c] += 1
        
        # M4
        c = base_col + 10
        U[u_x22][c] += 1
        V[v_y21][c] += 1
        V[v_y11][c] -= 1
        W[w_z11][c] += 1
        W[w_z21][c] += 1
        
        # M5
        c = base_col + 11
        U[u_x11][c] += 1
        U[u_x12][c] += 1
        V[v_y22][c] += 1
        W[w_z11][c] -= 1
        W[w_z12][c] += 1
        
        # M6
        c = base_col + 12
        U[u_x21][c] += 1
        U[u_x11][c] -= 1
        V[v_y11][c] += 1
        V[v_y12][c] += 1
        W[w_z22][c] += 1
        
        # M7
        c = base_col + 13
        U[u_x12][c] += 1
        U[u_x22][c] -= 1
        V[v_y21][c] += 1
        V[v_y22][c] += 1
        W[w_z11][c] += 1

    return {"U": U, "V": V, "W": W}
