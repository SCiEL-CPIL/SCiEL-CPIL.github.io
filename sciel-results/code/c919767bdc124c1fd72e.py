import time
import numpy as np

def _strassen_2x2x2():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    U[0,0]=1; U[1,0]=1; V[0,0]=1; V[1,0]=1; W[0,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[2,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[1,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[3,6]=1
    return U, V, W

def _naive_2x2x1():
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[1,1]=1; W[0,1]=1
    U[2,2]=1; V[0,2]=1; W[1,2]=1
    U[3,3]=1; V[1,3]=1; W[1,3]=1
    return U, V, W

def _naive_2x1x2():
    U = np.zeros((2, 4))
    V = np.zeros((2, 4))
    W = np.zeros((4, 4))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[0,1]=1; V[1,1]=1; W[1,1]=1
    U[1,2]=1; V[0,2]=1; W[2,2]=1
    U[1,3]=1; V[1,3]=1; W[3,3]=1
    return U, V, W

def _naive_2x1x1():
    U = np.zeros((2, 2))
    V = np.zeros((1, 2))
    W = np.zeros((2, 2))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[0,1]=1; W[1,1]=1
    return U, V, W

def _construct():
    n, m, p = 4, 5, 5
    total_rank = 92
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    
    def add_block(i_off, j_off, k_off, U_b, V_b, W_b, na, nb, nc):
        nonlocal col
        rank_b = U_b.shape[1]
        
        # U mapping: A[i_off + r//na_cols, j_off + r%na_cols]
        na_rows = na
        na_cols = U_b.shape[0] // na_rows
        for r in range(U_b.shape[0]):
            i_loc = r // na_cols
            j_loc = r % na_cols
            g_u = (i_off + i_loc) * m + (j_off + j_loc)
            U[g_u, col:col+rank_b] = U_b[r, :]
            
        # V mapping: B[j_off + r//nb_cols, k_off + r%nb_cols]
        nb_rows = nb
        nb_cols = V_b.shape[0] // nb_rows
        for r in range(V_b.shape[0]):
            j_loc = r // nb_cols
            k_loc = r % nb_cols
            g_v = (j_off + j_loc) * p + (k_off + k_loc)
            V[g_v, col:col+rank_b] = V_b[r, :]
            
        # W mapping: C[i_off + r//nc_cols, k_off + r%nc_cols]
        nc_rows = nc
        nc_cols = W_b.shape[0] // nc_rows
        for r in range(W_b.shape[0]):
            i_loc = r // nc_cols
            k_loc = r % nc_cols
            g_w = (k_off + k_loc) * n + (i_off + i_loc)
            W[g_w, col:col+rank_b] = W_b[r, :]
            
        col += rank_b

    # 8 blocks of 2x2x2
    blocks_222 = [
        (0, 0, 0), (0, 2, 0),
        (0, 0, 2), (0, 2, 2),
        (2, 0, 0), (2, 2, 0),
        (2, 0, 2), (2, 2, 2),
    ]
    for i, j, k in blocks_222:
        U_b, V_b, W_b = _strassen_2x2x2()
        add_block(i, j, k, U_b, V_b, W_b, 2, 2, 2)

    # 4 blocks of 2x2x1
    blocks_221 = [
        (0, 0, 4), (0, 2, 4),
        (2, 0, 4), (2, 2, 4),
    ]
    for i, j, k in blocks_221:
        U_b, V_b, W_b = _naive_2x2x1()
        add_block(i, j, k, U_b, V_b, W_b, 2, 2, 2) # A is 2x2, B is 2x1, C is 2x1
        # Wait, B is 2x1, so nb=2, nb_cols=1.
        # My add_block infers nb_cols from V_b.shape[0] // nb.
        # V_b is 2x4. nb=2. nb_cols = 1. Correct.
        # C is 2x1, so nc=2, nc_cols=1.
        # W_b is 2x4. nc=2. nc_cols = 1. Correct.
        # A is 2x2, so na=2, na_cols=2.
        # U_b is 4x4. na=2. na_cols = 2. Correct.

    # 4 blocks of 2x1x2
    blocks_212 = [
        (0, 4, 0), (0, 4, 2),
        (2, 4, 0), (2, 4, 2),
    ]
    for i, j, k in blocks_212:
        U_b, V_b, W_b = _naive_2x1x2()
        add_block(i, j, k, U_b, V_b, W_b, 2, 1, 2)
        # A is 2x1. na=2, na_cols=1. U_b 2x4. Correct.
        # B is 1x2. nb=1, nb_cols=2. V_b 2x4. Correct.
        # C is 2x2. nc=2, nc_cols=2. W_b 4x4. Correct.

    # 2 blocks of 2x1x1
    blocks_211 = [
        (0, 4, 4),
        (2, 4, 4),
    ]
    for i, j, k in blocks_211:
        U_b, V_b, W_b = _naive_2x1x1()
        add_block(i, j, k, U_b, V_b, W_b, 2, 1, 2)
        # A is 2x1. na=2, na_cols=1. U_b 2x2. Correct.
        # B is 1x1. nb=1, nb_cols=1. V_b 1x2. Correct.
        # C is 2x1. nc=2, nc_cols=1. W_b 2x2. Correct.

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    start_time = time.time()
    result = _construct()
    return result
