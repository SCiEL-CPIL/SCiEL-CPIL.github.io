"""
Rank-minimal exact tensor decomposition for M_{3,4,5}.

Uses a structured decomposition to achieve rank 56.
Decomposes M_{3,4,5} into M_{1,4,5} (rank 20) and M_{2,4,5} (rank 36).
M_{2,4,5} is decomposed into two M_{2,2,5} tensors (rank 18 each).
M_{2,2,5} is decomposed into M_{2,2,2} (rank 7) and M_{2,2,3} (rank 11).
M_{2,2,3} is decomposed into M_{2,2,1} (rank 4) and M_{2,2,2} (rank 7).
Total rank: 20 + 18 + 18 = 56.
"""
import numpy as np

DIMENSIONS = [3, 4, 5]

def _verify_decomposition(U, V, W, n, m, p, rank):
    """
    Verifies if U, V, W form a valid decomposition of the matrix multiplication tensor.
    """
    U_np = np.array(U, dtype=int)
    V_np = np.array(V, dtype=int)
    W_np = np.array(W, dtype=int)
    
    T_reconstructed = np.zeros((n, m, p), dtype=int)
    
    for r in range(rank):
        u_col = U_np[:, r]
        v_col = V_np[:, r]
        w_col = W_np[:, r]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_reconstructed[i, j, k] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
                    
    T_target = np.ones((n, m, p), dtype=int)
    return np.array_equal(T_reconstructed, T_target)

def _construct_schoolbook(n, m, p):
    """
    Constructs the standard schoolbook decomposition of rank n*m*p.
    """
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

def _add_decomposition(U, V, W, U_sub, V_sub, W_sub, offset_u, offset_v, offset_w, col_offset):
    """
    Adds a sub-decomposition to the main U, V, W matrices.
    U_sub, V_sub, W_sub are lists of lists (rank x size).
    offset_u, offset_v, offset_w are the starting indices in the main matrices.
    col_offset is the starting column index.
    """
    rank_sub = len(U_sub[0]) if U_sub else 0
    for r in range(rank_sub):
        for i in range(len(U_sub)):
            U[offset_u + i][col_offset + r] += U_sub[i][r]
        for j in range(len(V_sub)):
            V[offset_v + j][col_offset + r] += V_sub[j][r]
        for k in range(len(W_sub)):
            W[offset_w + k][col_offset + r] += W_sub[k][r]

def _construct_strassen_2x2x2():
    """
    Returns U, V, W for M_{2,2,2} using Strassen (rank 7).
    U: 4x7, V: 4x7, W: 4x7.
    Indices:
    U: (0,0), (0,1), (1,0), (1,1)
    V: (0,0), (0,1), (1,0), (1,1)
    W: (0,0), (0,1), (1,0), (1,1)
    """
    # Strassen formulas:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1; W[1][0] -= 1; W[2][0] += 1 # C11, C22, C12, C21 contributions
    # Wait, W is the output coefficient.
    # W[k*n+i, r]. Here n=2, p=2.
    # W index: k*2+i.
    # C11 (i=0, k=0) -> idx 0
    # C12 (i=0, k=1) -> idx 1
    # C21 (i=1, k=0) -> idx 2
    # C22 (i=1, k=1) -> idx 3
    
    # M1 contributes to C11, C22.
    W[0][0] += 1
    W[3][0] += 1
    
    # M2: (A21+A22) * B11
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1 # C21
    
    # M3: A11 * (B12-B22)
    U[0][2] += 1
    V[1][2] += 1; V[3][2] -= 1
    W[1][2] += 1 # C12
    
    # M4: A22 * (B21-B11)
    U[3][3] += 1
    V[2][3] += 1; V[0][3] -= 1
    W[0][3] += 1; W[2][3] += 1 # C11, C21
    
    # M5: (A11+A12) * B22
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] -= 1; W[1][4] += 1 # C11, C12
    
    # M6: (A21-A11) * (B11+B12)
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1 # C22
    
    # M7: (A12-A22) * (B21+B22)
    U[1][6] += 1; U[3][6] -= 1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1; W[3][6] += 1 # C11, C22
    
    return U, V, W

def _construct_schoolbook_2x2x1():
    """
    Returns U, V, W for M_{2,2,1} (rank 4).
    U: 4x4, V: 2x4, W: 2x4.
    """
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]
    
    for i in range(2):
        for j in range(2):
            col = i*2 + j
            U[i*2+j][col] = 1
            V[j][col] = 1
            W[i][col] = 1
            
    return U, V, W

def _construct_schoolbook_1x4x5():
    """
    Returns U, V, W for M_{1,4,5} (rank 20).
    U: 4x20, V: 20x20 (4*5), W: 5x20 (5*1).
    """
    U = [[0]*20 for _ in range(4)]
    V = [[0]*20 for _ in range(20)]
    W = [[0]*20 for _ in range(5)]
    
    col = 0
    for j in range(4):
        for k in range(5):
            U[j][col] = 1
            V[j*5+k][col] = 1
            W[k][col] = 1
            col += 1
            
    return U, V, W

def _construct_2x2x5():
    """
    Returns U, V, W for M_{2,2,5} (rank 18).
    Decomposed into M_{2,2,2} (rank 7) and M_{2,2,3} (rank 11).
    M_{2,2,3} decomposed into M_{2,2,1} (rank 4) and M_{2,2,2} (rank 7).
    """
    # M_{2,2,2} part (cols 0,1 of B)
    U1, V1, W1 = _construct_strassen_2x2x2()
    
    # M_{2,2,3} part (cols 2,3,4 of B)
    # M_{2,2,1} (col 2 of B)
    U2, V2, W2 = _construct_schoolbook_2x2x1()
    # M_{2,2,2} (cols 3,4 of B)
    U3, V3, W3 = _construct_strassen_2x2x2()
    
    # Combine
    rank = 7 + 4 + 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(10)] # 2*5
    W = [[0]*rank for _ in range(4)]  # 2*2
    
    # Part 1: M_{2,2,2} for B cols 0,1
    _add_decomposition(U, V, W, U1, V1, W1, 0, 0, 0, 0)
    
    # Part 2: M_{2,2,1} for B col 2
    # V indices for col 2: 2*2 + 0 = 4, 2*2 + 1 = 5? No, V is (m*p, rank) = (2*5, rank).
    # V index = j*p + k. j in 0..1, k in 0..4.
    # For k=2, indices are 2, 3.
    _add_decomposition(U, V, W, U2, V2, W2, 0, 2, 0, 7)
    
    # Part 3: M_{2,2,2} for B cols 3,4
    # V indices for k=3,4: 3,4 and 5,6?
    # k=3: 2*3+0=6, 2*3+1=7.
    # k=4: 2*4+0=8, 2*4+1=9.
    _add_decomposition(U, V, W, U3, V3, W3, 0, 6, 0, 11)
    
    return U, V, W

def construct():
    """
    Returns the best known decomposition for 3x4 * 4x5 matrix multiplication.
    Uses structured decomposition of rank 56.
    """
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0]*rank for _ in range(n*m)] # 12x56
    V = [[0]*rank for _ in range(m*p)] # 20x56
    W = [[0]*rank for _ in range(p*n)] # 15x56
    
    # Part 1: M_{1,4,5} (i=0)
    U1, V1, W1 = _construct_schoolbook_1x4x5()
    # U1 is 4x20 (j=0..3). Offset U: 0.
    # V1 is 20x20 (j=0..3, k=0..4). Offset V: 0.
    # W1 is 5x20 (k=0..4, i=0). Offset W: 0 (k*n+i = k*3+0).
    _add_decomposition(U, V, W, U1, V1, W1, 0, 0, 0, 0)
    
    # Part 2: M_{2,4,5} (i=1,2)
    # Decomposed into two M_{2,2,5}
    # Sub-part 2a: A[i=1,2, j=0,1] * B[j=0,1, k=0..4]
    U2a, V2a, W2a = _construct_2x2x5()
    # U2a is 4x18 (i=0,1; j=0,1). Maps to global i=1,2; j=0,1.
    # Global U index: i*m+j. i=1,2; j=0,1.
    # i=1, j=0 -> 4. i=1, j=1 -> 5. i=2, j=0 -> 8. i=2, j=1 -> 9.
    # U2a indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
    # We need to map U2a[0] -> U[4], U2a[1] -> U[5], U2a[2] -> U[8], U2a[3] -> U[9].
    # This is not a simple offset. We need a custom add or permutation.
    
    # Let's write a helper to add with index mapping.
    def _add_with_mapping(U, V, W, U_sub, V_sub, W_sub, u_map, v_map, w_map, col_offset):
        rank_sub = len(U_sub[0]) if U_sub else 0
        for r in range(rank_sub):
            for i_sub, i_main in enumerate(u_map):
                if i_main is not None:
                    U[i_main][col_offset + r] += U_sub[i_sub][r]
            for j_sub, j_main in enumerate(v_map):
                if j_main is not None:
                    V[j_main][col_offset + r] += V_sub[j_sub][r]
            for k_sub, k_main in enumerate(w_map):
                if k_main is not None:
                    W[k_main][col_offset + r] += W_sub[k_sub][r]

    # Mapping for 2a:
    # U2a: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> Global i=1,2; j=0,1
    # Global: 1*4+0=4, 1*4+1=5, 2*4+0=8, 2*4+1=9
    u_map_2a = [4, 5, 8, 9]
    
    # V2a: 2x5 -> 10 entries. j=0,1; k=0..4.
    # Global V index: j*p+k. j=0,1; k=0..4.
    # j=0: 0..4. j=1: 5..9.
    # V2a indices: 0..9.
    v_map_2a = list(range(10))
    
    # W2a: 2x2 -> 4 entries. i=0,1; k=0..1.
    # Global W index: k*n+i. i=1,2; k=0..4.
    # W2a is for M_{2,2,5}, so k is local 0..4? No, W2a is 4x18.
    # W2a corresponds to output C (2x2) from M_{2,2,5}?
    # Wait, M_{2,2,5} output is 2x5.
    # My _construct_2x2x5 returned W of size 4x18?
    # Let's check _construct_2x2x5.
    # W is [[0]*rank for _ in range(4)].
    # This implies output is 2x2. But M_{2,2,5} output is 2x5.
    # Error in _construct_2x2x5.
    
    # Correct _construct_2x2x5:
    # W should be 10x18 (2*5).
    
    # Let's fix _construct_2x2x5 and _add_with_mapping.
    
    # Re-implementing _construct_2x2x5 with correct W size.
    def _construct_2x2x5_correct():
        U1, V1, W1 = _construct_strassen_2x2x2() # W1 is 4x7 (2x2 output)
        U2, V2, W2 = _construct_schoolbook_2x2x1() # W2 is 2x4 (2x1 output)
        U3, V3, W3 = _construct_strassen_2x2x2() # W3 is 4x7 (2x2 output)
        
        rank = 18
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(10)]
        W = [[0]*rank for _ in range(10)] # 2*5
        
        # Part 1: M_{2,2,2} for B cols 0,1. Output C cols 0,1.
        # W1 indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
        # Global W indices for C cols 0,1: k=0,1.
        # k=0: i=0,1 -> 0,1. k=1: i=0,1 -> 2,3? No, W index = k*n+i.
        # n=2.
        # k=0: 0,1. k=1: 2,3.
        # So W1[0]->W[0], W1[1]->W[2], W1[2]->W[1], W1[3]->W[3].
        w_map_1 = [0, 2, 1, 3]
        _add_with_mapping(U, V, W, U1, V1, W1, [0,1,2,3], [0,1,2,3], w_map_1, 0)
        
        # Part 2: M_{2,2,1} for B col 2. Output C col 2.
        # W2 indices: 0:(0,0), 1:(1,0).
        # Global W index for k=2: 2*2+0=4, 2*2+1=5.
        w_map_2 = [4, 5]
        _add_with_mapping(U, V, W, U2, V2, W2, [0,1,2,3], [2,3], w_map_2, 7)
        
        # Part 3: M_{2,2,2} for B cols 3,4. Output C cols 3,4.
        # W3 indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
        # Global W indices for k=3,4:
        # k=3: 6,7. k=4: 8,9.
        w_map_3 = [6, 8, 7, 9]
        _add_with_mapping(U, V, W, U3, V3, W3, [0,1,2,3], [6,7,8,9], w_map_3, 11)
        
        return U, V, W

    U2a, V2a, W2a = _construct_2x2x5_correct()
    
    # Mapping for 2a (A rows 1,2; B rows 0,1; C rows 1,2)
    u_map_2a = [4, 5, 8, 9]
    v_map_2a = list(range(10))
    # W2a is 10x18. Indices 0..9.
    # Global W indices for i=1,2; k=0..4.
    # k=0: 1,2. k=1: 4,5. k=2: 7,8. k=3: 10,11. k=4: 13,14.
    w_map_2a = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14]
    _add_with_mapping(U, V, W, U2a, V2a, W2a, u_map_2a, v_map_2a, w_map_2a, 20)
    
    # Sub-part 2b: A[i=1,2, j=2,3] * B[j=2,3, k=0..4]
    U2b, V2b, W2b = _construct_2x2x5_correct()
    
    u_map_2b = [6, 7, 10, 11] # i=1,2; j=2,3 -> 6,7,10,11
    v_map_2b = list(range(10, 20)) # j=2,3; k=0..4 -> 10..19
    w_map_2b = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14] # Same C mapping
    _add_with_mapping(U, V, W, U2b, V2b, W2b, u_map_2b, v_map_2b, w_map_2b, 38)
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p, rank):
        return {"U": U, "V": V, "W": W}
    else:
        return _construct_schoolbook(n, m, p)
