import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def _strassen_2x2x2():
    """
    Returns U, V, W (each 4x7) for the rank-7 Strassen decomposition of 2x2x2.
    U indices: 0:A00, 1:A01, 2:A10, 3:A11
    V indices: 0:B00, 1:B01, 2:B10, 3:B11
    W indices: 0:C00, 1:C01, 2:C10, 3:C11
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _schoolbook_2x2x1():
    """
    Returns U (4x4), V (2x4), W (2x4) for rank-4 schoolbook 2x2x1.
    U indices: 0:A00, 1:A01, 2:A10, 3:A11
    V indices: 0:B0, 1:B1
    W indices: 0:C0, 1:C1
    """
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
    U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
    U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
    
    return U, V, W

def _schoolbook_1x2x2():
    """
    Returns U (2x4), V (4x4), W (2x4) for rank-4 schoolbook 1x2x2.
    U indices: 0:A0, 1:A1
    V indices: 0:B00, 1:B01, 2:B10, 3:B11
    W indices: 0:C0, 1:C1
    """
    U = np.zeros((2, 4))
    V = np.zeros((4, 4))
    W = np.zeros((2, 4))
    
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
    U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
    U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
    
    return U, V, W

def _schoolbook_1x2x1():
    """
    Returns U (2x2), V (2x2), W (1x2) for rank-2 schoolbook 1x2x1.
    U indices: 0:A0, 1:A1
    V indices: 0:B0, 1:B1
    W indices: 0:C0
    """
    U = np.zeros((2, 2))
    V = np.zeros((2, 2))
    W = np.zeros((1, 2))
    
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
    
    return U, V, W

def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Uses Strassen's algorithm for 2x2x2 blocks to reduce the rank from the schoolbook 60
    to 56 by splitting the 3x4x5 multiplication into structured blocks.
    """
    n, m, p = DIMENSIONS
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Precompute block decompositions
    S_U, S_V, S_W = _strassen_2x2x2()
    SB21_U, SB21_V, SB21_W = _schoolbook_2x2x1()
    SB12_U, SB12_V, SB12_W = _schoolbook_1x2x2()
    SB11_U, SB11_V, SB11_W = _schoolbook_1x2x1()
    
    def add_block(U_blk, V_blk, W_blk, u_idx, v_idx, w_idx):
        rank = U_blk.shape[1]
        for r in range(rank):
            u_vec = np.zeros(n * m)
            for i, idx in enumerate(u_idx):
                u_vec[idx] = U_blk[i, r]
            U_parts.append(u_vec)
            
            v_vec = np.zeros(m * p)
            for i, idx in enumerate(v_idx):
                v_vec[idx] = V_blk[i, r]
            V_parts.append(v_vec)
            
            w_vec = np.zeros(p * n)
            for i, idx in enumerate(w_idx):
                w_vec[idx] = W_blk[i, r]
            W_parts.append(w_vec)
    
    # Indices for U (3x4):
    # Row 0: 0,1,2,3
    # Row 1: 4,5,6,7
    # Row 2: 8,9,10,11
    
    # Indices for V (4x5):
    # Row 0: 0,1,2,3,4
    # Row 1: 5,6,7,8,9
    # Row 2: 10,11,12,13,14
    # Row 3: 15,16,17,18,19
    
    # Indices for W (5x3) with W[k*n+i]:
    # k=0: 0,1,2
    # k=1: 3,4,5
    # k=2: 6,7,8
    # k=3: 9,10,11
    # k=4: 12,13,14
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (2x2x2 Strassen)
    # U: A00=0, A01=1, A10=4, A11=5
    # V: B00=0, B01=1, B10=5, B11=6
    # W: C00=0, C01=3, C10=1, C11=4
    add_block(S_U, S_V, S_W, [0,1,4,5], [0,1,5,6], [0,3,1,4])
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] (2x2x2 Strassen)
    # U: A00=0, A01=1, A10=4, A11=5
    # V: B00=2, B01=3, B10=7, B11=8
    # W: C00=6, C01=9, C10=7, C11=10
    add_block(S_U, S_V, S_W, [0,1,4,5], [2,3,7,8], [6,9,7,10])
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5] (2x2x1 Schoolbook)
    # U: A00=0, A01=1, A10=4, A11=5
    # V: B0=4, B1=9
    # W: C0=12, C1=13
    add_block(SB21_U, SB21_V, SB21_W, [0,1,4,5], [4,9], [12,13])
    
    # Block 4: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2] (2x2x2 Strassen)
    # U: A00=2, A01=3, A10=6, A11=7
    # V: B00=10, B01=11, B10=15, B11=16
    # W: C00=0, C01=3, C10=1, C11=4
    add_block(S_U, S_V, S_W, [2,3,6,7], [10,11,15,16], [0,3,1,4])
    
    # Block 5: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4] (2x2x2 Strassen)
    # U: A00=2, A01=3, A10=6, A11=7
    # V: B00=12, B01=13, B10=17, B11=18
    # W: C00=6, C01=9, C10=7, C11=10
    add_block(S_U, S_V, S_W, [2,3,6,7], [12,13,17,18], [6,9,7,10])
    
    # Block 6: A[0:2, 2:4] * B[2:4, 4:5] -> C[0:2, 4:5] (2x2x1 Schoolbook)
    # U: A00=2, A01=3, A10=6, A11=7
    # V: B0=14, B1=19
    # W: C0=12, C1=13
    add_block(SB21_U, SB21_V, SB21_W, [2,3,6,7], [14,19], [12,13])
    
    # Block 7: A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2] (1x2x2 Schoolbook)
    # U: A0=8, A1=9
    # V: B00=0, B01=1, B10=5, B11=6
    # W: C0=2, C1=5
    add_block(SB12_U, SB12_V, SB12_W, [8,9], [0,1,5,6], [2,5])
    
    # Block 8: A[2, 0:2] * B[0:2, 2:4] -> C[2, 2:4] (1x2x2 Schoolbook)
    # U: A0=8, A1=9
    # V: B00=2, B01=3, B10=7, B11=8
    # W: C0=8, C1=11
    add_block(SB12_U, SB12_V, SB12_W, [8,9], [2,3,7,8], [8,11])
    
    # Block 9: A[2, 0:2] * B[0:2, 4:5] -> C[2, 4:5] (1x2x1 Schoolbook)
    # U: A0=8, A1=9
    # V: B0=4, B1=9
    # W: C0=14
    add_block(SB11_U, SB11_V, SB11_W, [8,9], [4,9], [14])
    
    # Block 10: A[2, 2:4] * B[2:4, 0:2] -> C[2, 0:2] (1x2x2 Schoolbook)
    # U: A0=10, A1=11
    # V: B00=10, B01=11, B10=15, B11=16
    # W: C0=2, C1=5
    add_block(SB12_U, SB12_V, SB12_W, [10,11], [10,11,15,16], [2,5])
    
    # Block 11: A[2, 2:4] * B[2:4, 2:4] -> C[2, 2:4] (1x2x2 Schoolbook)
    # U: A0=10, A1=11
    # V: B00=12, B01=13, B10=17, B11=18
    # W: C0=8, C1=11
    add_block(SB12_U, SB12_V, SB12_W, [10,11], [12,13,17,18], [8,11])
    
    # Block 12: A[2, 2:4] * B[2:4, 4:5] -> C[2, 4:5] (1x2x1 Schoolbook)
    # U: A0=10, A1=11
    # V: B0=14, B1=19
    # W: C0=14
    add_block(SB11_U, SB11_V, SB11_W, [10,11], [14,19], [14])
    
    # Total rank: 7+7+4+7+7+4+4+4+2+4+4+2 = 56
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
