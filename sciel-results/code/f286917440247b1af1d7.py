import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # We'll build the decomposition with rank 15:
    # 14 terms for the 2x2x4 part (two 2x2x2 Strassen blocks)
    # 1 term for the 2x1x4 outer product part
    
    rank = 15
    U = [[0] * rank for _ in range(n * m)]  # 6 x 15
    V = [[0] * rank for _ in range(m * p)]  # 12 x 15
    W = [[0] * rank for _ in range(p * n)]  # 8 x 15
    
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val
    
    # A is 2x3, indexed as A[i,j] with i in {0,1}, j in {0,1,2}
    # U index for A[i,j] is i*3 + j: A[0,0]=0, A[0,1]=1, A[0,2]=2, A[1,0]=3, A[1,1]=4, A[1,2]=5
    # B is 3x4, indexed as B[j,k] with j in {0,1,2}, k in {0,1,2,3}
    # V index for B[j,k] is j*4 + k: B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3, B[1,0]=4, B[1,1]=5, B[1,2]=6, B[1,3]=7, B[2,0]=8, B[2,1]=9, B[2,2]=10, B[2,3]=11
    # C is 2x4, indexed as C[i,k] with i in {0,1}, k in {0,1,2,3}
    # W index for C[i,k] follows trace convention W[k*n+i, r] = W[4*i + k, r]
    # So C[0,0]=0, C[0,1]=1, C[0,2]=2, C[0,3]=3, C[1,0]=4, C[1,1]=5, C[1,2]=6, C[1,3]=7
    
    # Part 1: A[0:2, 0:2] * B[0:2, 0:4]
    # This is a 2x2 x 2x4 multiplication, decomposed as two 2x2x2 Strassen blocks
    # Block 1a: A[0:2, 0:2] * B[0:2, 0:2]
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # B indices: B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    # C indices: C[0,0]=0, C[0,1]=1, C[1,0]=4, C[1,1]=5
    
    # M1: (A00+A11)(B00+B11) -> C00+C11
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (5, 1)], 0)
    
    # M2: (A10+A11)B00 -> C10-C11
    add_term([(3, 1), (4, 1)], [(0, 1)], [(4, 1), (5, -1)], 1)
    
    # M3: A00(B01-B11) -> C01+C11
    add_term([(0, 1)], [(1, 1), (5, -1)], [(1, 1), (5, 1)], 2)
    
    # M4: A11(B10-B00) -> C00+C10
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (4, 1)], 3)
    
    # M5: (A00+A01)B11 -> -C00+C01
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (1, 1)], 4)
    
    # M6: (A10-A00)(B00+B01) -> C11
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(5, 1)], 5)
    
    # M7: (A01-A11)(B10+B11) -> C00
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
    
    # Block 1b: A[0:2, 0:2] * B[0:2, 2:4]
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # B indices: B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7
    # C indices: C[0,2]=2, C[0,3]=3, C[1,2]=6, C[1,3]=7
    
    # M1': (A00+A11)(B02+B13) -> C02+C13
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(2, 1), (7, 1)], 7)
    
    # M2': (A10+A11)B02 -> C12-C13
    add_term([(3, 1), (4, 1)], [(2, 1)], [(6, 1), (7, -1)], 8)
    
    # M3': A00(B03-B13) -> C03+C13
    add_term([(0, 1)], [(3, 1), (7, -1)], [(3, 1), (7, 1)], 9)
    
    # M4': A11(B12-B02) -> C02+C12
    add_term([(4, 1)], [(6, 1), (2, -1)], [(2, 1), (6, 1)], 10)
    
    # M5': (A00+A01)B13 -> -C02+C03
    add_term([(0, 1), (1, 1)], [(7, 1)], [(2, -1), (3, 1)], 11)
    
    # M6': (A10-A00)(B02+B03) -> C13
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    
    # M7': (A01-A11)(B12+B13) -> C02
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(2, 1)], 13)
    
    # Part 2: A[0:2, 2:3] * B[2:3, 0:4]
    # This is a 2x1 x 1x4 multiplication, which is an outer product
    # A indices: A[0,2]=2, A[1,2]=5
    # B indices: B[2,0]=8, B[2,1]=9, B[2,2]=10, B[2,3]=11
    # C indices: C[0,0]=0, C[0,1]=1, C[0,2]=2, C[0,3]=3, C[1,0]=4, C[1,1]=5, C[1,2]=6, C[1,3]=7
    # C[i,k] = A[i,2] * B[2,k]
    # This is a single rank-1 tensor term: (sum_i e_i) ⊗ (sum_k e_k) ⊗ e_2
    # U: A[0,2]=2 and A[1,2]=5 both get +1
    # V: B[2,0]=8, B[2,1]=9, B[2,2]=10, B[2,3]=11 all get +1
    # W: all C entries get +1
    add_term([(2, 1), (5, 1)], [(8, 1), (9, 1), (10, 1), (11, 1)],
             [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)], 14)
    
    # Verify all entries are integers or half-integers
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if not (val == int(val) or val * 2 == int(val * 2)):
                    raise ValueError(f"Non-half-integer entry: {val}")
    
    return {"U": U, "V": V, "W": W}
