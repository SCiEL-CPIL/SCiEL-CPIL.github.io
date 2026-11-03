"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return the 7 rank-1 terms for 2x2 matrix multiplication.
    Each term is (u, v, w) where u is in R^4 (A indices), v in R^4 (B indices), w in R^4 (C indices).
    Indexing: row-major, index = r * n + c for an n x n matrix.
    """
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    # Strassen's 7 products:
    # p1 = (a00 + a11) * (b00 + b11)
    # p2 = (a10 + a11) * b00
    # p3 = a00 * (b01 - b11)
    # p4 = a11 * (b10 - b00)
    # p5 = (a00 + a01) * b11
    # p6 = (a10 - a00) * (b00 + b01)
    # p7 = (a01 - a11) * (b10 + b11)
    #
    # c00 = p1 + p4 - p5 + p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p6
    
    terms = []
    # p1
    terms.append(((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)))
    # p2
    terms.append(((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, 0)))
    # p3
    terms.append(((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, 0)))
    # p4
    terms.append(((0, 0, 0, 1), (0, 0, 1, -1), (0, 0, 1, 1)))
    # p5
    terms.append(((1, 1, 0, 0), (0, 0, 0, 1), (0, 0, -1, 1)))
    # p6
    terms.append(((0, 0, 1, -1), (1, 1, 0, 0), (0, 0, 0, 1)))
    # p7
    terms.append(((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, 1)))
    
    return terms

def _decompose_3x3x3():
    """Return rank-1 terms for 3x3 matrix multiplication using Strassen on 2x2 blocks.
    We split 3x3 into a 2x2 block and handle the third row/column separately.
    
    A = [[A2x2, a_col], [a_row, a33]]  where A2x2 is 2x2, a_col is 2x1, a_row is 1x2, a33 is scalar
    B = [[B2x2, b_col], [b_row, b33]]
    C = [[C2x2, c_col], [c_row, c33]]
    
    C2x2 = A2x2 * B2x2  (rank 7 via Strassen)
    c_col = A2x2 * b_col + a_col * b33
    c_row = a_row * B2x2 + a33 * b_row
    c33 = a33 * b33
    
    For c_col: A2x2 * b_col is 2x1 * 1x1... wait, A2x2 is 2x2, b_col is 2x1.
    Actually, let me reconsider.
    
    A (3x3) * B (3x3) = C (3x3)
    
    Let's use a different decomposition. We can use the fact that:
    R(3,3,3) <= R(2,2,2) + R(2,2,1) + R(1,2,2) + R(1,1,1) + cross terms...
    
    Actually, the known rank for 3x3x3 is 21. Let me just use the schoolbook for 3x3x3 to be safe, 
    but that's rank 27. 
    
    Let me try a different approach: use Strassen on the top-left 2x2, and handle the rest.
    
    A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    B = [B11 B12; B21 B22] where B11 is 2x2, B12 is 2x1, B21 is 1x2, B22 is 1x1
    
    C11 = A11*B11 + A12*B21  (2x2)
    C12 = A11*B12 + A12*B22  (2x1)
    C21 = A21*B11 + A22*B21  (1x2)
    C22 = A21*B12 + A22*B22  (1x1)
    
    C11: A11*B11 is rank 7 (Strassen), A12*B21 is 2x1 * 1x2 = rank 1 (outer product)
    C12: A11*B12 is 2x2 * 2x1, which is like multiplying 2x2 by a 2x1 matrix. This can be done with rank... 
         Actually, A11*B12 where A11 is 2x2 and B12 is 2x1. This is a 2x1 result.
         We can write this as: for each row i of the result, sum_j A11[i,j] * B12[j,0].
         This is just 2*2 = 4 multiplications, but we can share some.
         Actually, this is equivalent to multiplying a 2x2 matrix by a 2x1 matrix, which has rank 4 (schoolbook) 
         or possibly less.
         
    This is getting complicated. Let me just use schoolbook for 3x3x3 (rank 27) and see if I can improve.
    
    Actually, let me just implement the full schoolbook for 3x3x7 and see if I can find a better decomposition.
    """
    # For now, return schoolbook terms for 3x3x3
    terms = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u = [0.0] * 9
                v = [0.0] * 9
                w = [0.0] * 9
                u[i * 3 + j] = 1.0
                v[j * 3 + k] = 1.0
                w[k * 3 + i] = 1.0
                terms.append((u, v, w))
    return terms

def construct():
    # Schoolbook decomposition of rank n*m*p
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
