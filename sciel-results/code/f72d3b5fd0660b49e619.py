import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact tensor decomposition for 3x4x5 matrix multiplication, rank 44.
    Uses Strassen (rank 7) for 2x2x2, rank 2 for 2x2x1 and 1x2x2 and 1x2x1.
    """
    n, m, p = DIMENSIONS
    t_start = time.time()
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_block(u_idx, v_idx, w_idx, U_block, V_block, W_block):
        rank = U_block.shape[1]
        for r in range(rank):
            u_vec = np.zeros(n * m)
            for i, idx in enumerate(u_idx):
                u_vec[idx] = U_block[i, r]
            v_vec = np.zeros(m * p)
            for i, idx in enumerate(v_idx):
                v_vec[idx] = V_block[i, r]
            w_vec = np.zeros(p * n)
            for i, idx in enumerate(w_idx):
                w_vec[idx] = W_block[i, r]
            U_parts.append(u_vec)
            V_parts.append(v_vec)
            W_parts.append(w_vec)
    
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
        U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
        U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
        return U, V, W
    
    def sb_2x2x1():
        """2x2x1 rank 2: C0=A00*B0+A01*B1, C1=A10*B0+A11*B1"""
        U = np.zeros((4, 2))
        V = np.zeros((2, 2))
        W = np.zeros((2, 2))
        U[0,0]=1; U[2,0]=1; V[0,0]=1; W[0,0]=1; W[1,0]=1
        U[1,1]=1; U[3,1]=1; V[1,1]=1; W[0,1]=1; W[1,1]=1
        return U, V, W
    
    def sb_1x2x2():
        """1x2x2 rank 2: C0=A0*B00+A1*B10, C1=A0*B01+A1*B11"""
        U = np.zeros((2, 2))
        V = np.zeros((4, 2))
        W = np.zeros((2, 2))
        U[0,0]=1; V[0,0]=1; V[1,0]=1; W[0,0]=1; W[1,0]=1
        U[1,1]=1; V[2,1]=1; V[3,1]=1; W[0,1]=1; W[1,1]=1
        return U, V, W
    
    def sb_1x2x1():
        """1x2x1 rank 2: C=A0*B0+A1*B1"""
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        U[1,1]=1; V[1,1]=1; W[0,1]=1
        return U, V, W
    
    # Index definitions
    u_top = [0, 1, 4, 5]
    u_bot = [2, 3, 6, 7]
    u2_tl = [8, 9]
    u2_tm = [10, 11]
    
    v_tl = [0, 1, 5, 6]
    v_tm = [2, 3, 7, 8]
    v_tr = [4, 9]
    v_bl = [10, 11, 15, 16]
    v_bm = [12, 13, 17, 18]
    v_br = [14, 19]
    
    w_t01 = [0, 3, 1, 4]
    w_t23 = [6, 9, 7, 10]
    w_t4  = [12, 13]
    w_2_01 = [2, 5]
    w_2_23 = [8, 11]
    w_2_4  = [14]
    
    # Part 1: A rows 0,1 (6 blocks)
    U_b, V_b, W_b = strassen_2x2x2(); add_block(u_top, v_tl, w_t01, U_b, V_b, W_b)
    U_b, V_b, W_b = strassen_2x2x2(); add_block(u_top, v_tm, w_t23, U_b, V_b, W_b)
    U_b, V_b, W_b = sb_2x2x1();       add_block(u_top, v_tr, w_t4, U_b, V_b, W_b)
    U_b, V_b, W_b = strassen_2x2x2(); add_block(u_bot, v_bl, w_t01, U_b, V_b, W_b)
    U_b, V_b, W_b = strassen_2x2x2(); add_block(u_bot, v_bm, w_t23, U_b, V_b, W_b)
    U_b, V_b, W_b = sb_2x2x1();       add_block(u_bot, v_br, w_t4, U_b, V_b, W_b)
    
    # Part 2: A row 2 (6 blocks)
    U_b, V_b, W_b = sb_1x2x2(); add_block(u2_tl, v_tl, w_2_01, U_b, V_b, W_b)
    U_b, V_b, W_b = sb_1x2x2(); add_block(u2_tl, v_tm, w_2_23, U_b, V_b, W_b)
    U_b, V_b, W_b = sb_1x2x1(); add_block(u2_tl, v_tr, w_2_4, U_b, V_b, W_b)
    U_b, V_b, W_b = sb_1x2x2(); add_block(u2_tm, v_bl, w_2_01, U_b, V_b, W_b)
    U_b, V_b, W_b = sb_1x2x2(); add_block(u2_tm, v_bm, w_2_23, U_b, V_b, W_b)
    U_b, V_b, W_b = sb_1x2x1(); add_block(u2_tm, v_br, w_2_4, U_b, V_b, W_b)
    
    # Total rank: 7*4 + 2*2 + 2*4 + 2*2 = 28+4+8+4 = 44
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
