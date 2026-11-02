import numpy as np
import time

def construct():
    n, m, p = 2, 3, 7
    
    # Rank 39 hybrid decomposition:
    # Split A into A1 (2x2, cols 0,1) and A2 (2x1, col 2)
    # Split B into B1 (2x7, rows 0,1) and B2 (1x7, row 2)
    # C = A1*B1 + A2*B2
    #
    # For A1*B1: Use Strassen for 2x2x2 blocks on k={0,1}, {2,3}, {4,5}
    #             Use schoolbook for k={6}
    # For A2*B2: Use schoolbook (14 terms)
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]  # 6 x 39
    V = [[0] * rank for _ in range(m * p)]  # 21 x 39
    W = [[0] * rank for _ in range(p * n)]  # 14 x 39
    
    col = 0
    
    def add_strassen_2x2x2(i0, j0, k0):
        """Add Strassen decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[k0:k0+2, i0:i0+2]"""
        nonlocal col
        # A indices: A[i, j] at index i*m + j
        A00 = i0 * m + j0
        A01 = i0 * m + j0 + 1
        A10 = (i0 + 1) * m + j0
        A11 = (i0 + 1) * m + j0 + 1
        
        # B indices: B[j, k] at index j*p + k
        B00 = j0 * p + k0
        B01 = j0 * p + k0 + 1
        B10 = (j0 + 1) * p + k0
        B11 = (j0 + 1) * p + k0 + 1
        
        # C indices: C[k, i] at index k*n + i (trace convention W[k*n+i, r])
        C00 = k0 * n + i0
        C01 = k0 * n + i0 + 1
        C10 = (k0 + 1) * n + i0
        C11 = (k0 + 1) * n + i0 + 1
        
        # M1 = (A00 + A11)(B00 + B11)
        U[A00][col] = 1; U[A11][col] = 1
        V[B00][col] = 1; V[B11][col] = 1
        W[C00][col] = 1; W[C01][col] = 1; W[C10][col] = 1; W[C11][col] = 1
        col += 1
        
        # M2 = (A10 + A11)B00
        U[A10][col] = 1; U[A11][col] = 1
        V[B00][col] = 1
        W[C10][col] = 1; W[C11][col] = 1
        col += 1
        
        # M3 = A00(B10 - B11)
        U[A00][col] = 1
        V[B10][col] = 1; V[B11][col] = -1
        W[C01][col] = 1; W[C11][col] = 1
        col += 1
        
        # M4 = A11(B01 - B00)
        U[A11][col] = 1
        V[B01][col] = 1; V[B00][col] = -1
        W[C10][col] = 1; W[C00][col] = 1
        col += 1
        
        # M5 = (A00 + A01)B11
        U[A00][col] = 1; U[A01][col] = 1
        V[B11][col] = 1
        W[C01][col] = 1; W[C11][col] = 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[A10][col] = 1; U[A00][col] = -1
        V[B00][col] = 1; V[B01][col] = 1
        W[C10][col] = 1; W[C11][col] = 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[A01][col] = 1; U[A11][col] = -1
        V[B10][col] = 1; V[B11][col] = 1
        W[C00][col] = 1; W[C10][col] = 1
        col += 1
    
    def add_schoolbook_2x2x1(i0, j0, k0):
        """Add schoolbook for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+1] -> C[k0:k0+1, i0:i0+2]"""
        nonlocal col
        for i in range(2):
            for j in range(2):
                for k in range(1):
                    a_idx = (i0 + i) * m + (j0 + j)
                    b_idx = (j0 + j) * p + (k0 + k)
                    c_idx = (k0 + k) * n + (i0 + i)
                    U[a_idx][col] = 1
                    V[b_idx][col] = 1
                    W[c_idx][col] = 1
                    col += 1
    
    def add_schoolbook_2x1x7(i0, j0):
        """Add schoolbook for A[i0:i0+2, j0:j0+1] * B[j0:j0+1, 0:7] -> C[0:7, i0:i0+2]"""
        nonlocal col
        for i in range(2):
            for k in range(7):
                a_idx = (i0 + i) * m + j0
                b_idx = j0 * p + k
                c_idx = k * n + (i0 + i)
                U[a_idx][col] = 1
                V[b_idx][col] = 1
                W[c_idx][col] = 1
                col += 1
    
    # Part 1: A1 (2x2) * B1 (2x7)
    # k=0,1: Strassen
    add_strassen_2x2x2(0, 0, 0)   # 7 terms
    # k=2,3: Strassen
    add_strassen_2x2x2(0, 0, 2)   # 7 terms
    # k=4,5: Strassen
    add_strassen_2x2x2(0, 0, 4)   # 7 terms
    # k=6: Schoolbook
    add_schoolbook_2x2x1(0, 0, 6)  # 4 terms
    
    # Part 2: A2 (2x1) * B2 (1x7)
    add_schoolbook_2x1x7(0, 2)    # 14 terms
    
    # Total: 7 + 7 + 7 + 4 + 14 = 39
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
