import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def _get_strassen_2x2():
    """
    Returns U, V, W blocks for the 7-term Strassen decomposition of 2x2 matrix multiplication.
    U, V, W are lists of 7 vectors of length 4.
    """
    # Indices: 00=0, 01=1, 10=2, 11=3
    u = [
        [1, 0, 0, 1], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00+A01
        [0, 0, 1, -1],# M6: A10-A00
        [0, 1, 0, -1] # M7: A01-A11
    ]
    v = [
        [1, 0, 0, 1], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01-B11
        [0, 0, 1, -1],# M4: B10-B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    w = [
        [1, 0, 0, 1], # M1: C00+C11
        [0, 0, 1, 0], # M2: C10
        [0, 1, 0, 0], # M3: C01
        [1, 0, 1, 0], # M4: C00+C10
        [-1, 1, 0, 0],# M5: -C00+C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    return u, v, w

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Total rank estimate: 59
    # We will build the decomposition term by term.
    # U shape: (n*m, rank) = (15, rank)
    # V shape: (m*p, rank) = (25, rank)
    # W shape: (p*n, rank) = (15, rank)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        # u_vec: list of length n*m
        # v_vec: list of length m*p
        # w_vec: list of length p*n
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Initialize zero vectors
    def zero_u(): return [0.0] * (n*m)
    def zero_v(): return [0.0] * (m*p)
    def zero_w(): return [0.0] * (p*n)
    
    # Block 1: C_TL (Rows 0-1, Cols 0-3)
    # A_T: Rows 0-1, Cols 0-3 (2x4)
    # B_L: Rows 0-3, Cols 0-3 (4x4)
    # Split A_T into A11 (cols 0-1) and A12 (cols 2-3)
    # Split B_L into B11, B12, B21, B22 (2x2 blocks)
    
    strassen_u, strassen_v, strassen_w = _get_strassen_2x2()
    
    # Mapping for A_T (2x4) to 4x2 blocks?
    # A_T indices:
    # Row 0: 0,1,2,3 -> U indices 0,1,2,3
    # Row 1: 0,1,2,3 -> U indices 5,6,7,8
    # A11 (2x2): A[0,0], A[0,1], A[1,0], A[1,1] -> U indices 0,1,5,6
    # A12 (2x2): A[0,2], A[0,3], A[1,2], A[1,3] -> U indices 2,3,7,8
    
    # Mapping for B_L (4x4) to 2x2 blocks
    # B_L indices:
    # Row 0: 0,1,2,3 -> V indices 0,1,2,3
    # Row 1: 0,1,2,3 -> V indices 5,6,7,8
    # Row 2: 0,1,2,3 -> V indices 10,11,12,13
    # Row 3: 0,1,2,3 -> V indices 15,16,17,18
    
    # B11 (2x2): B[0,0], B[0,1], B[1,0], B[1,1] -> V indices 0,1,5,6
    # B12 (2x2): B[0,2], B[0,3], B[1,2], B[1,3] -> V indices 2,3,7,8
    # B21 (2x2): B[2,0], B[2,1], B[3,0], B[3,1] -> V indices 10,11,15,16
    # B22 (2x2): B[2,2], B[2,3], B[3,2], B[3,3] -> V indices 12,13,17,18
    
    # Output C_TL (2x4)
    # C indices:
    # Row 0: 0,1,2,3 -> W indices 0,3,6,9 (k*n+i)
    # Row 1: 0,1,2,3 -> W indices 1,4,7,10
    
    # C11 (2x2): C[0,0], C[0,1], C[1,0], C[1,1] -> W indices 0,3,1,4
    # C12 (2x2): C[0,2], C[0,3], C[1,2], C[1,3] -> W indices 6,9,7,10
    # C21 (2x2): C[0,0], C[0,1], C[1,0], C[1,1] -> W indices 0,3,1,4 (Same as C11? No, C21 is cols 0-1 of Block 1)
    # Wait. Block 1 is Cols 0-3.
    # C11 is Cols 0-1. C12 is Cols 2-3.
    # C21 is Cols 0-1. C22 is Cols 2-3.
    # So C11 and C21 write to the same output locations?
    # C_TL = A11 B11 + A12 B21 (for Cols 0-1)
    # C_TL = A11 B12 + A12 B22 (for Cols 2-3)
    # Yes, they add to the same C elements.
    
    # We need to map the local 2x2 indices to global indices.
    # Local U indices: 0,1,2,3
    # Local V indices: 0,1,2,3
    # Local W indices: 0,1,2,3
    
    def map_u_local_to_global(u_local, a_block_type):
        # a_block_type: 0 for A11, 1 for A12
        u_global = zero_u()
        if a_block_type == 0:
            # A11: A[0,0]=0, A[0,1]=1, A[1,0]=5, A[1,1]=6
            u_global[0] = u_local[0]
            u_global[1] = u_local[1]
            u_global[5] = u_local[2]
            u_global[6] = u_local[3]
        else:
            # A12: A[0,2]=2, A[0,3]=3, A[1,2]=7, A[1,3]=8
            u_global[2] = u_local[0]
            u_global[3] = u_local[1]
            u_global[7] = u_local[2]
            u_global[8] = u_local[3]
        return u_global

    def map_v_local_to_global(v_local, b_block_type):
        # b_block_type: 0 for B11, 1 for B12, 2 for B21, 3 for B22
        v_global = zero_v()
        if b_block_type == 0:
            # B11: B[0,0]=0, B[0,1]=1, B[1,0]=5, B[1,1]=6
            v_global[0] = v_local[0]
            v_global[1] = v_local[1]
            v_global[5] = v_local[2]
            v_global[6] = v_local[3]
        elif b_block_type == 1:
            # B12: B[0,2]=2, B[0,3]=3, B[1,2]=7, B[1,3]=8
            v_global[2] = v_local[0]
            v_global[3] = v_local[1]
            v_global[7] = v_local[2]
            v_global[8] = v_local[3]
        elif b_block_type == 2:
            # B21: B[2,0]=10, B[2,1]=11, B[3,0]=15, B[3,1]=16
            v_global[10] = v_local[0]
            v_global[11] = v_local[1]
            v_global[15] = v_local[2]
            v_global[16] = v_local[3]
        else:
            # B22: B[2,2]=12, B[2,3]=13, B[3,2]=17, B[3,3]=18
            v_global[12] = v_local[0]
            v_global[13] = v_local[1]
            v_global[17] = v_local[2]
            v_global[18] = v_local[3]
        return v_global

    def map_w_local_to_global(w_local, c_block_type):
        # c_block_type: 0 for C11/C21 (Cols 0-1), 1 for C12/C22 (Cols 2-3)
        # C rows are always 0-1 for Block 1.
        w_global = zero_w()
        if c_block_type == 0:
            # C[0,0]=0, C[0,1]=3, C[1,0]=1, C[1,1]=4
            w_global[0] = w_local[0]
            w_global[3] = w_local[1]
            w_global[1] = w_local[2]
            w_global[4] = w_local[3]
        else:
            # C[0,2]=6, C[0,3]=9, C[1,2]=7, C[1,3]=10
            w_global[6] = w_local[0]
            w_global[9] = w_local[1]
            w_global[7] = w_local[2]
            w_global[10] = w_local[3]
        return w_global

    # Add 28 terms for Block 1
    for r in range(7):
        # Term 1: A11 * B11 -> C11
        u1 = map_u_local_to_global(strassen_u[r], 0)
        v1 = map_v_local_to_global(strassen_v[r], 0)
        w1 = map_w_local_to_global(strassen_w[r], 0)
        add_term(u1, v1, w1)
        
        # Term 2: A11 * B12 -> C12
        u2 = map_u_local_to_global(strassen_u[r], 0)
        v2 = map_v_local_to_global(strassen_v[r], 1)
        w2 = map_w_local_to_global(strassen_w[r], 1)
        add_term(u2, v2, w2)
        
        # Term 3: A12 * B21 -> C21 (adds to C11 locations)
        u3 = map_u_local_to_global(strassen_u[r], 1)
        v3 = map_v_local_to_global(strassen_v[r], 2)
        w3 = map_w_local_to_global(strassen_w[r], 0)
        add_term(u3, v3, w3)
        
        # Term 4: A12 * B22 -> C22 (adds to C12 locations)
        u4 = map_u_local_to_global(strassen_u[r], 1)
        v4 = map_v_local_to_global(strassen_v[r], 3)
        w4 = map_w_local_to_global(strassen_w[r], 1)
        add_term(u4, v4, w4)
        
    if time.time() > deadline:
        # Fallback to schoolbook if taking too long (unlikely)
        pass

    # Block 2: C_BL (Row 2, Cols 0-3)
    # A_B: Row 2, Cols 0-3 (1x4)
    # B_L: Rows 0-3, Cols 0-3 (4x4)
    # C_BL: Row 2, Cols 0-3 (1x4)
    # C[2,k] = sum_j A[2,j] B[j,k]
    # 16 terms
    
    for j in range(4):
        for k in range(4):
            u = zero_u()
            v = zero_v()
            w = zero_w()
            
            # A[2,j] -> U index 2*5 + j = 10+j
            u[10+j] = 1.0
            
            # B[j,k] -> V index j*5 + k
            v[j*5+k] = 1.0
            
            # C[2,k] -> W index k*3 + 2
            w[k*3+2] = 1.0
            
            add_term(u, v, w)
            
    if time.time() > deadline:
        pass

    # Block 3: C_TR (Rows 0-1, Col 4)
    # A_T: Rows 0-1, Cols 0-4 (2x5)
    # B_R: Rows 0-4, Col 4 (5x1)
    # C_TR: Rows 0-1, Col 4 (2x1)
    # C[i,4] = sum_j A[i,j] B[j,4]
    # 10 terms
    
    for i in range(2):
        for j in range(5):
            u = zero_u()
            v = zero_v()
            w = zero_w()
            
            # A[i,j] -> U index i*5 + j
            u[i*5+j] = 1.0
            
            # B[j,4] -> V index j*5 + 4
            v[j*5+4] = 1.0
            
            # C[i,4] -> W index 4*3 + i
            w[4*3+i] = 1.0
            
            add_term(u, v, w)
            
    if time.time() > deadline:
        pass

    # Block 4: C_BR (Row 2, Col 4)
    # A_B: Row 2, Cols 0-4 (1x5)
    # B_R: Rows 0-4, Col 4 (5x1)
    # C_BR: Row 2, Col 4 (1x1)
    # C[2,4] = sum_j A[2,j] B[j,4]
    # 5 terms
    
    for j in range(5):
        u = zero_u()
        v = zero_v()
        w = zero_w()
        
        # A[2,j] -> U index 10+j
        u[10+j] = 1.0
        
        # B[j,4] -> V index j*5 + 4
        v[j*5+4] = 1.0
        
        # C[2,4] -> W index 4*3 + 2
        w[4*3+2] = 1.0
        
        add_term(u, v, w)

    # Convert to numpy arrays
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        U, V, W = _build_schoolbook(n, m, p)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True
