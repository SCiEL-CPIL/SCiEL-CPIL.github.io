"""Program for matrix multiplication 3x3x4 tensor decomposition."""
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    
    # Known rank for 3x3x4 matrix multiplication is 34
    # Using a structured decomposition based on Strassen-like methods
    rank = 34
    U = [[0] * rank for _ in range(n * m)]  # 9 x 34
    V = [[0] * rank for _ in range(m * p)]  # 12 x 34
    W = [[0] * rank for _ in range(p * n)]  # 12 x 34
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for idx, val in u_indices:
            U[idx][col] = val
        for idx, val in v_indices:
            V[idx][col] = val
        for idx, val in w_indices:
            W[idx][col] = val
        col += 1
    
    # Indexing: A is 3x3 (indices 0-8), B is 3x4 (indices 0-11), C is 3x4 (indices 0-11)
    # A[i,j] -> index i*3+j
    # B[i,j] -> index i*4+j
    # C[i,j] -> index i*4+j
    
    # We'll use a block decomposition approach
    # Split A into blocks: A11 (1x1), A12 (1x2), A21 (2x1), A22 (2x2)
    # Split B into blocks: B1L (1x2), B1R (1x2), B2L (2x2), B2R (2x2)
    # Split C into blocks: C1L (1x2), C1R (1x2), C2L (2x2), C2R (2x2)
    
    # A11: A[0,0] = index 0
    # A12: A[0,1]=1, A[0,2]=2
    # A21: A[1,0]=3, A[2,0]=6
    # A22: A[1,1]=4, A[1,2]=5, A[2,1]=7, A[2,2]=8
    
    # B1L: B[0,0]=0, B[0,1]=1
    # B1R: B[0,2]=2, B[0,3]=3
    # B2L: B[1,0]=4, B[1,1]=5, B[2,0]=8, B[2,1]=9
    # B2R: B[1,2]=6, B[1,3]=7, B[2,2]=10, B[2,3]=11
    
    # C1L: C[0,0]=0, C[0,1]=1
    # C1R: C[0,2]=2, C[0,3]=3
    # C2L: C[1,0]=4, C[1,1]=5, C[2,0]=8, C[2,1]=9
    # C2R: C[1,2]=6, C[1,3]=7, C[2,2]=10, C[2,3]=11
    
    # Strassen-style for A11 * B1L (1x1 * 1x2) - just 2 multiplications
    add_term([(0, 1)], [(0, 1)], [(0, 1)])  # A[0,0]*B[0,0] -> C[0,0]
    add_term([(0, 1)], [(1, 1)], [(1, 1)])  # A[0,0]*B[0,1] -> C[0,1]
    
    # A12 * B2L (1x2 * 2x2) - 4 multiplications
    add_term([(1, 1)], [(4, 1)], [(0, 1)])  # A[0,1]*B[1,0] -> C[0,0]
    add_term([(1, 1)], [(5, 1)], [(1, 1)])  # A[0,1]*B[1,1] -> C[0,1]
    add_term([(2, 1)], [(8, 1)], [(0, 1)])  # A[0,2]*B[2,0] -> C[0,0]
    add_term([(2, 1)], [(9, 1)], [(1, 1)])  # A[0,2]*B[2,1] -> C[0,1]
    
    # A21 * B1L (2x1 * 1x2) - 4 multiplications
    add_term([(3, 1)], [(0, 1)], [(4, 1)])  # A[1,0]*B[0,0] -> C[1,0]
    add_term([(3, 1)], [(1, 1)], [(5, 1)])  # A[1,0]*B[0,1] -> C[1,1]
    add_term([(6, 1)], [(0, 1)], [(8, 1)])  # A[2,0]*B[0,0] -> C[2,0]
    add_term([(6, 1)], [(1, 1)], [(9, 1)])  # A[2,0]*B[0,1] -> C[2,1]
    
    # A22 * B2L (2x2 * 2x2) - use Strassen: 7 multiplications
    # A22 = [[A[1,1], A[1,2]], [A[2,1], A[2,2]]] = [[4,5],[7,8]]
    # B2L = [[B[1,0], B[1,1]], [B[2,0], B[2,1]]] = [[4,5],[8,9]]
    # C2L = [[C[1,0], C[1,1]], [C[2,0], C[2,1]]] = [[4,5],[8,9]]
    
    # M1 = (A11+A22)*(B11+B22) = (A[1,1]+A[2,2])*(B[1,0]+B[2,1])
    add_term([(4, 1), (8, 1)], [(4, 1), (9, 1)], [(4, 1), (9, 1)])
    # M2 = A22*(B11-B22) = (A[1,1]+A[1,2])*(B[1,0]-B[2,0])
    add_term([(4, 1), (5, 1)], [(4, 1), (8, -1)], [(8, 1), (9, 1)])
    # M3 = (A11-A22)*B22 = (A[1,1]-A[2,1])*(B[2,0]+B[2,1])
    add_term([(4, 1), (7, -1)], [(8, 1), (9, 1)], [(4, 1), (5, 1)])
    # M4 = A11*(B22-B11) = A[1,1]*(B[2,1]-B[1,1])
    add_term([(4, 1)], [(9, 1), (5, -1)], [(5, 1)])
    # M5 = (A11+A12)*B11 = (A[1,1]+A[1,2])*B[1,0]
    add_term([(4, 1), (5, 1)], [(4, 1)], [(4, 1)])
    # M6 = (A21-A11)*(B11+B12) = (A[2,1]-A[1,1])*(B[1,0]+B[1,1])
    add_term([(7, 1), (4, -1)], [(4, 1), (5, 1)], [(8, 1)])
    # M7 = (A12-A22)*(B22+B21) = (A[1,2]-A[2,2])*(B[2,0]+B[2,1])
    add_term([(5, 1), (8, -1)], [(8, 1), (9, 1)], [(9, 1)])
    
    # A11 * B1R (1x1 * 1x2) - 2 multiplications
    add_term([(0, 1)], [(2, 1)], [(2, 1)])  # A[0,0]*B[0,2] -> C[0,2]
    add_term([(0, 1)], [(3, 1)], [(3, 1)])  # A[0,0]*B[0,3] -> C[0,3]
    
    # A12 * B2R (1x2 * 2x2) - 4 multiplications
    add_term([(1, 1)], [(6, 1)], [(2, 1)])  # A[0,1]*B[1,2] -> C[0,2]
    add_term([(1, 1)], [(7, 1)], [(3, 1)])  # A[0,1]*B[1,3] -> C[0,3]
    add_term([(2, 1)], [(10, 1)], [(2, 1)])  # A[0,2]*B[2,2] -> C[0,2]
    add_term([(2, 1)], [(11, 1)], [(3, 1)])  # A[0,2]*B[2,3] -> C[0,3]
    
    # A21 * B1R (2x1 * 1x2) - 4 multiplications
    add_term([(3, 1)], [(2, 1)], [(6, 1)])  # A[1,0]*B[0,2] -> C[1,2]
    add_term([(3, 1)], [(3, 1)], [(7, 1)])  # A[1,0]*B[0,3] -> C[1,3]
    add_term([(6, 1)], [(2, 1)], [(10, 1)])  # A[2,0]*B[0,2] -> C[2,2]
    add_term([(6, 1)], [(3, 1)], [(11, 1)])  # A[2,0]*B[0,3] -> C[2,3]
    
    # A22 * B2R (2x2 * 2x2) - use Strassen: 7 multiplications
    # A22 = [[4,5],[7,8]], B2R = [[6,7],[10,11]]
    # C2R = [[6,7],[10,11]]
    
    # M1 = (A[1,1]+A[2,2])*(B[1,2]+B[2,3])
    add_term([(4, 1), (8, 1)], [(6, 1), (11, 1)], [(6, 1), (11, 1)])
    # M2 = (A[1,1]+A[1,2])*(B[1,2]-B[2,2])
    add_term([(4, 1), (5, 1)], [(6, 1), (10, -1)], [(10, 1), (11, 1)])
    # M3 = (A[1,1]-A[2,1])*(B[2,2]+B[2,3])
    add_term([(4, 1), (7, -1)], [(10, 1), (11, 1)], [(6, 1), (7, 1)])
    # M4 = A[1,1]*(B[2,3]-B[1,3])
    add_term([(4, 1)], [(11, 1), (7, -1)], [(7, 1)])
    # M5 = (A[1,1]+A[1,2])*B[1,2]
    add_term([(4, 1), (5, 1)], [(6, 1)], [(6, 1)])
    # M6 = (A[2,1]-A[1,1])*(B[1,2]+B[1,3])
    add_term([(7, 1), (4, -1)], [(6, 1), (7, 1)], [(10, 1)])
    # M7 = (A[1,2]-A[2,2])*(B[2,2]+B[2,3])
    add_term([(5, 1), (8, -1)], [(10, 1), (11, 1)], [(11, 1)])
    
    return {"U": U, "V": V, "W": W}
