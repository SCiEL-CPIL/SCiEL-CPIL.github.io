"""Program for matrix multiplication 3x3x4 tensor decomposition."""
DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    rank = 32
    U = [[0] * rank for _ in range(n * m)]  # 9 x 32
    V = [[0] * rank for _ in range(m * p)]  # 12 x 32
    W = [[0] * rank for _ in range(p * n)]  # 12 x 32
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for idx in u_indices:
            U[idx][col] += 1
        for idx in v_indices:
            V[idx][col] += 1
        for idx in w_indices:
            W[idx][col] += 1
        col += 1
    
    def add_term_signed(u_terms, v_terms, w_terms):
        nonlocal col
        for idx, val in u_terms:
            U[idx][col] += val
        for idx, val in v_terms:
            V[idx][col] += val
        for idx, val in w_terms:
            W[idx][col] += val
        col += 1

    # Strassen Left (A11 * B1L)
    # A11 indices: 0,1,3,4
    # B1L indices: 0,1,4,5
    # C1L indices: 0,3,1,4
    
    # P1
    add_term([0, 4], [0, 5], [0, 4])
    
    # P2
    add_term_signed([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (4, -1)])
    
    # P3
    add_term_signed([(0, 1)], [(1, 1), (5, -1)], [(3, 1), (4, 1)])
    
    # P4
    add_term_signed([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)])
    
    # P5
    add_term_signed([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (3, 1)])
    
    # P6
    add_term_signed([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(4, 1)])
    
    # P7
    add_term_signed([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)])
    
    # A12 * B2L
    # A12 indices: 2, 5
    # B2L indices: 8, 9
    # C1L contribution:
    # C[0,0]->0, C[0,1]->3, C[1,0]->1, C[1,1]->4
    
    # T1: A[0,2] B[2,0] C[0,0]
    add_term([2], [8], [0])
    
    # T2: A[0,2] B[2,1] C[0,1]
    add_term([2], [9], [3])
    
    # T3: A[1,2] B[2,0] C[1,0]
    add_term([5], [8], [1])
    
    # T4: A[1,2] B[2,1] C[1,1]
    add_term([5], [9], [4])
    
    # A21 * B1L
    # A21 indices: 6, 7
    # B1L indices: 0,1,4,5
    # C2L contribution:
    # C[2,0]->2, C[2,1]->5
    
    # T1: A[2,0] B[0,0] C[2,0]
    add_term([6], [0], [2])
    
    # T2: A[2,0] B[0,1] C[2,1]
    add_term([6], [1], [5])
    
    # T3: A[2,1] B[1,0] C[2,0]
    add_term([7], [4], [2])
    
    # T4: A[2,1] B[1,1] C[2,1]
    add_term([7], [5], [5])
    
    # A22 * B2L
    # A22 index: 8
    # B2L indices: 8, 9
    # C2L contribution:
    # C[2,0]->2, C[2,1]->5
    
    # T1: A[2,2] B[2,0] C[2,0]
    add_term([8], [8], [2])
    
    # T2: A[2,2] B[2,1] C[2,1]
    add_term([8], [9], [5])
    
    # Strassen Right (A11 * B1R)
    # A11 indices: 0,1,3,4
    # B1R indices: 2,3,6,7
    # C1R indices: 6,9,7,10
    
    # P1
    add_term([0, 4], [2, 7], [6, 10])
    
    # P2
    add_term_signed([(3, 1), (4, 1)], [(2, 1)], [(7, 1), (10, -1)])
    
    # P3
    add_term_signed([(0, 1)], [(3, 1), (7, -1)], [(9, 1), (10, 1)])
    
    # P4
    add_term_signed([(4, 1)], [(6, 1), (2, -1)], [(6, 1), (7, 1)])
    
    # P5
    add_term_signed([(0, 1), (1, 1)], [(7, 1)], [(6, -1), (9, 1)])
    
    # P6
    add_term_signed([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(10, 1)])
    
    # P7
    add_term_signed([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(6, 1)])
    
    # A12 * B2R
    # A12 indices: 2, 5
    # B2R indices: 10, 11
    # C1R contribution:
    # C[0,2]->6, C[0,3]->9, C[1,2]->7, C[1,3]->10
    
    # T1: A[0,2] B[2,2] C[0,2]
    add_term([2], [10], [6])
    
    # T2: A[0,2] B[2,3] C[0,3]
    add_term([2], [11], [9])
    
    # T3: A[1,2] B[2,2] C[1,2]
    add_term([5], [10], [7])
    
    # T4: A[1,2] B[2,3] C[1,3]
    add_term([5], [11], [10])
    
    # A21 * B1R
    # A21 indices: 6, 7
    # B1R indices: 2,3,6,7
    # C2R contribution:
    # C[2,2]->8, C[2,3]->11
    
    # T1: A[2,0] B[0,2] C[2,2]
    add_term([6], [2], [8])
    
    # T2: A[2,0] B[0,3] C[2,3]
    add_term([6], [3], [11])
    
    # T3: A[2,1] B[1,2] C[2,2]
    add_term([7], [6], [8])
    
    # T4: A[2,1] B[1,3] C[2,3]
    add_term([7], [7], [11])
    
    # A22 * B2R
    # A22 index: 8
    # B2R indices: 10, 11
    # C2R contribution:
    # C[2,2]->8, C[2,3]->11
    
    # T1: A[2,2] B[2,2] C[2,2]
    add_term([8], [10], [8])
    
    # T2: A[2,2] B[2,3] C[2,3]
    add_term([8], [11], [11])
    
    return {"U": U, "V": V, "W": W}
