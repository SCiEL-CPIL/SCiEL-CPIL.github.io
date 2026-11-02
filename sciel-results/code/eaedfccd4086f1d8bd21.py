import time
import sys

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs, c):
        for idx, val in u_coeffs:
            U[idx][c] += val
        for idx, val in v_coeffs:
            V[idx][c] += val
        for idx, val in w_coeffs:
            W[idx][c] += val

    # Strassen terms for A[0:2, 0:2] and B[0:2, 0:2]
    # A indices: 0,1,2,3 -> (0,0), (0,1), (1,0), (1,1)
    # B indices: 0,1,2,3,4,5 -> (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
    # W indices: 0..5 -> (0,0), (0,1), (1,0), (1,1), (2,0), (2,1)
    
    # M1
    add_term([(0,1), (3,1)], [(0,1), (4,1)], [(0,1), (3,1)], 0)
    # M2
    add_term([(2,1), (3,1)], [(0,1)], [(2,1), (3,-1)], 1)
    # M3
    add_term([(0,1)], [(3,1), (0,-1)], [(1,1), (2,1)], 2)
    # M4
    add_term([(3,1)], [(1,1), (3,-1)], [(0,-1), (1,1)], 3)
    # M5
    add_term([(0,1), (1,1)], [(4,1)], [(0,1), (2,-1)], 4)
    # M6
    add_term([(2,1), (0,-1)], [(0,1), (1,1)], [(2,1), (3,1)], 5)
    # M7
    add_term([(1,1), (3,-1)], [(3,1), (4,1)], [(0,-1), (1,1)], 6)
    
    # Terms for A[:, 2] (col 2, indices 2,5 in U) and B[2, :] (row 2, indices 6,7,8 in V)
    # A[0,2]=2, A[1,2]=5
    # B[2,0]=6, B[2,1]=7, B[2,2]=8
    # C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3, C[0,2]=4, C[1,2]=5
    
    # M8: A[0,2]B[2,0]
    add_term([(2,1)], [(6,1)], [(0,1)], 7)
    # M9: A[0,2]B[2,1]
    add_term([(2,1)], [(7,1)], [(1,1)], 8)
    # M10: A[1,2]B[2,0]
    add_term([(5,1)], [(6,1)], [(2,1)], 9)
    # M11: A[1,2]B[2,1]
    add_term([(5,1)], [(7,1)], [(3,1)], 10)
    # M12: A[0,2]B[2,2]
    add_term([(2,1)], [(8,1)], [(4,1)], 11)
    # M13: A[1,2]B[2,2]
    add_term([(5,1)], [(8,1)], [(5,1)], 12)
    
    # Terms for A[0:2, 0:2] and B[:, 2] (col 2, indices 2,5 in V? No, B[0,2]=2, B[1,2]=5)
    # B[0,2] is index 2. B[1,2] is index 5.
    # A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    
    # M14: A[0,0]B[0,2]
    add_term([(0,1)], [(2,1)], [(4,1)], 13)
    # M15: A[0,1]B[1,2]
    add_term([(1,1)], [(5,1)], [(4,1)], 14)
    # M16: A[1,0]B[0,2]
    add_term([(2,1)], [(2,1)], [(5,1)], 15)
    # M17: A[1,1]B[1,2]
    add_term([(3,1)], [(5,1)], [(5,1)], 16)
    
    return {"U": U, "V": V, "W": W}
