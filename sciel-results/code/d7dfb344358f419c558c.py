"""Program for 3x4x5 matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 4, 5]

# Strassen's 2x2x2 rank-7 decomposition factors
STRASSEN_U = [
    [1, 0, 0, 1, 0, 0, 0],  # A11
    [0, 1, 1, 0, 1, 0, 1],  # A12
    [1, 0, 0, 0, 0, 1, -1], # A21
    [0, 1, 0, 0, 1, 1, 0],  # A22
]

STRASSEN_V = [
    [1, 1, 0, 1, 0, 1, 1],  # B11
    [0, 0, 1, 0, -1, 0, 0], # B12
    [1, 1, -1, 0, 0, 0, 0], # B21
    [1, 0, -1, 0, 0, 1, 1], # B22
]

STRASSEN_W = [
    [1, 0, 0, 1, -1, 0, 1],  # C11
    [0, 0, 1, 1, 0, 1, 0],   # C12
    [0, 1, 0, 1, 1, 0, 0],   # C21
    [1, -1, 1, 0, 0, 1, 0],  # C22
]


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    total_rank = 58
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2 decomposition
    def add_strassen(u_map, v_map, w_map, start_col):
        """u_map: list of 4 global U indices for local A (0,0),(0,1),(1,0),(1,1)
           v_map: list of 4 global V indices for local B (0,0),(0,1),(1,0),(1,1)
           w_map: list of 4 global W indices for local C (0,0),(0,1),(1,0),(1,1)
        """
        for r in range(7):
            for li, ui in enumerate(u_map):
                if STRASSEN_U[li][r] != 0:
                    U[ui][start_col + r] += STRASSEN_U[li][r]
            for li, vi in enumerate(v_map):
                if STRASSEN_V[li][r] != 0:
                    V[vi][start_col + r] += STRASSEN_V[li][r]
            for li, wi in enumerate(w_map):
                if STRASSEN_W[li][r] != 0:
                    W[wi][start_col + r] += STRASSEN_W[li][r]
        return start_col + 7
    
    # Helper to add schoolbook 2x2x3 decomposition
    def add_schoolbook_223(u_map, v_map, w_map, start_col):
        """u_map: list of 4 global U indices for local A (0,0),(0,1),(1,0),(1,1)
           v_map: list of 6 global V indices for local B (0,0),(0,1),(0,2),(1,0),(1,1),(1,2)
           w_map: list of 6 global W indices for local C (0,0),(0,1),(0,2),(1,0),(1,1),(1,2)
        """
        # 2x2x3 schoolbook: 12 terms
        # A is 2x2, B is 2x3, C is 2x3
        # C[i][k] = sum_j A[i][j] * B[j][k]
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[u_map[i*2+j]][start_col] = 1
                    V[v_map[j*3+k]][start_col] = 1
                    W[w_map[i*3+k]][start_col] = 1
                    start_col += 1
        return start_col
    
    # Helper to add schoolbook 1x2x2 decomposition
    def add_schoolbook_122(u_map, v_map, w_map, start_col):
        """u_map: list of 2 global U indices for local A (0,0),(0,1)
           v_map: list of 4 global V indices for local B (0,0),(0,1),(1,0),(1,1)
           w_map: list of 2 global W indices for local C (0,0),(0,1)
        """
        for j in range(2):
            for k in range(2):
                U[u_map[j]][start_col] = 1
                V[v_map[j*2+k]][start_col] = 1
                W[w_map[k]][start_col] = 1
                start_col += 1
        return start_col
    
    # Helper to add schoolbook 1x2x3 decomposition
    def add_schoolbook_123(u_map, v_map, w_map, start_col):
        """u_map: list of 2 global U indices for local A (0,0),(0,1)
           v_map: list of 6 global V indices for local B (0,0),(0,1),(0,2),(1,0),(1,1),(1,2)
           w_map: list of 3 global W indices for local C (0,0),(0,1),(0,2)
        """
        for j in range(2):
            for k in range(3):
                U[u_map[j]][start_col] = 1
                V[v_map[j*3+k]][start_col] = 1
                W[w_map[k]][start_col] = 1
                start_col += 1
        return start_col
    
    # Define global index mappings
    # U indices: U[i*m+j] for A[i][j], i in 0..2, j in 0..3
    # A11: i=0,1 j=0,1 -> [0,1,4,5]
    # A12: i=0,1 j=2,3 -> [2,3,6,7]
    # A21: i=2 j=0,1 -> [8,9]
    # A22: i=2 j=2,3 -> [10,11]
    
    # V indices: V[j*p+k] for B[j][k], j in 0..3, k in 0..4
    # B11: j=0,1 k=0,1 -> [0,1,5,6]
    # B12: j=0,1 k=2,3,4 -> [2,3,4,7,8,9]
    # B21: j=2,3 k=0,1 -> [10,11,15,16]
    # B22: j=2,3 k=2,3,4 -> [12,13,14,17,18,19]
    
    # W indices: W[k*n+i] for C[i][k], i in 0..2, k in 0..4
    # C11: i=0,1 k=0,1 -> [0,1,3,4]
    # C12: i=0,1 k=2,3,4 -> [6,7,8,9,10,11]
    # C21: i=2 k=0,1 -> [2,5]
    # C22: i=2 k=2,3,4 -> [8,11,14]
    
    A11_u = [0, 1, 4, 5]
    A12_u = [2, 3, 6, 7]
    A21_u = [8, 9]
    A22_u = [10, 11]
    
    B11_v = [0, 1, 5, 6]
    B12_v = [2, 3, 4, 7, 8, 9]
    B21_v = [10, 11, 15, 16]
    B22_v = [12, 13, 14, 17, 18, 19]
    
    C11_w = [0, 1, 3, 4]
    C12_w = [6, 7, 8, 9, 10, 11]
    C21_w = [2, 5]
    C22_w = [8, 11, 14]
    
    # Sub-problem 1: A11(2x2) * B11(2x2) -> C11(2x2), Strassen rank 7
    col = add_strassen(A11_u, B11_v, C11_w, col)
    
    # Sub-problem 2: A12(2x2) * B21(2x2) -> C11(2x2), Strassen rank 7
    col = add_strassen(A12_u, B21_v, C11_w, col)
    
    # Sub-problem 3: A11(2x2) * B12(2x3) -> C12(2x3), schoolbook rank 12
    col = add_schoolbook_223(A11_u, B12_v, C12_w, col)
    
    # Sub-problem 4: A12(2x2) * B22(2x3) -> C12(2x3), schoolbook rank 12
    col = add_schoolbook_223(A12_u, B22_v, C12_w, col)
    
    # Sub-problem 5: A21(1x2) * B11(2x2) -> C21(1x2), schoolbook rank 4
    col = add_schoolbook_122(A21_u, B11_v, C21_w, col)
    
    # Sub-problem 6: A22(1x2) * B21(2x2) -> C21(1x2), schoolbook rank 4
    col = add_schoolbook_122(A22_u, B21_v, C21_w, col)
    
    # Sub-problem 7: A21(1x2) * B12(2x3) -> C22(1x3), schoolbook rank 6
    col = add_schoolbook_123(A21_u, B12_v, C22_w, col)
    
    # Sub-problem 8: A22(1x2) * B22(2x3) -> C22(1x3), schoolbook rank 6
    col = add_schoolbook_123(A22_u, B22_v, C22_w, col)
    
    assert col == total_rank, f"Expected {total_rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
