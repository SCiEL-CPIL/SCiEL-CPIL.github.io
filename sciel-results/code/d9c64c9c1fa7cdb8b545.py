import numpy as np
import time

def construct():
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8
    
    rank = 28
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    def add_strassen_block(col_off, u_idx, v_idx, w_idx):
        # u_idx = [x00, x01, x10, x11]
        # v_idx = [y00, y01, y10, y11]
        # w_idx = [z00, z01, z10, z11]
        
        # M1 = (x00 + x11)(y00 + y11)
        U[u_idx[0], col_off+0] += 1
        U[u_idx[3], col_off+0] += 1
        V[v_idx[0], col_off+0] += 1
        V[v_idx[3], col_off+0] += 1
        W[w_idx[0], col_off+0] += 1
        W[w_idx[3], col_off+0] += 1
        
        # M2 = (x10 + x11)y00
        U[u_idx[2], col_off+1] += 1
        U[u_idx[3], col_off+1] += 1
        V[v_idx[0], col_off+1] += 1
        W[w_idx[1], col_off+1] += 1
        W[w_idx[3], col_off+1] -= 1
        
        # M3 = x00(y10 - y00)
        U[u_idx[0], col_off+2] += 1
        V[v_idx[2], col_off+2] += 1
        V[v_idx[0], col_off+2] -= 1
        W[w_idx[1], col_off+2] += 1
        W[w_idx[3], col_off+2] += 1
        
        # M4 = x11(y01 - y11)
        U[u_idx[3], col_off+3] += 1
        V[v_idx[1], col_off+3] += 1
        V[v_idx[3], col_off+3] -= 1
        W[w_idx[0], col_off+3] += 1
        W[w_idx[2], col_off+3] += 1
        
        # M5 = (x00 + x01)y11
        U[u_idx[0], col_off+4] += 1
        U[u_idx[1], col_off+4] += 1
        V[v_idx[3], col_off+4] += 1
        W[w_idx[0], col_off+4] -= 1
        W[w_idx[1], col_off+4] += 1
        
        # M6 = (x10 - x00)(y00 + y01)
        U[u_idx[2], col_off+5] += 1
        U[u_idx[0], col_off+5] -= 1
        V[v_idx[0], col_off+5] += 1
        V[v_idx[1], col_off+5] += 1
        W[w_idx[3], col_off+5] += 1
        
        # M7 = (x01 - x11)(y10 + y11)
        U[u_idx[1], col_off+6] += 1
        U[u_idx[3], col_off+6] -= 1
        V[v_idx[2], col_off+6] += 1
        V[v_idx[3], col_off+6] += 1
        W[w_idx[0], col_off+6] += 1
        
    # Block 1: A_L * B_T_left
    # A_L: U[0,1,4,5]
    # B_T_left: V[0,1,4,5]
    # C_00: W[0,1,4,5]
    add_strassen_block(0, [0, 1, 4, 5], [0, 1, 4, 5], [0, 1, 4, 5])
    
    # Block 2: A_L * B_T_right
    # A_L: U[0,1,4,5]
    # B_T_right: V[2,3,6,7]
    # C_01: W[2,3,6,7]
    add_strassen_block(7, [0, 1, 4, 5], [2, 3, 6, 7], [2, 3, 6, 7])
    
    # Block 3: A_R * B_B_left
    # A_R: U[2,3,6,7]
    # B_B_left: V[8,9,12,13]
    # C_10: W[0,1,4,5]
    add_strassen_block(14, [2, 3, 6, 7], [8, 9, 12, 13], [0, 1, 4, 5])
    
    # Block 4: A_R * B_B_right
    # A_R: U[2,3,6,7]
    # B_B_right: V[10,11,14,15]
    # C_11: W[2,3,6,7]
    add_strassen_block(21, [2, 3, 6, 7], [10, 11, 14, 15], [2, 3, 6, 7])
    
    return {"U": U, "V": V, "W": W}
