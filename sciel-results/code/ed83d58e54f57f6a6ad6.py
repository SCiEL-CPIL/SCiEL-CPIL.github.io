import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 54
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_222(i_a, j_a, j_b, k_b):
        nonlocal col
        # i_a: start row of A block (0 or 2)
        # j_a: start col of A block (0 or 2)
        # j_b: start row of B block (0 or 2)
        # k_b: start col of B block (0, 2, or 4)
        
        # Indices in U, V, W
        # U[i*m + j]
        # V[j*p + k]
        # W[k*n + i]
        
        # Strassen M1: (a+d)(x+u)
        # a: U[i_a*m + j_a], d: U[(i_a+1)*m + (j_a+1)]
        # x: V[j_b*p + k_b], u: V[(j_b+1)*p + (k_b+1)]
        # p: W[k_b*n + i_a], s: W[(k_b+1)*n + (i_a+1)]
        
        U[i_a*m + j_a][col] = 1
        U[(i_a+1)*m + (j_a+1)][col] = 1
        V[j_b*p + k_b][col] = 1
        V[(j_b+1)*p + (k_b+1)][col] = 1
        W[k_b*n + i_a][col] = 1
        W[(k_b+1)*n + (i_a+1)][col] = 1
        col += 1
        
        # M2: (c+d)x
        # c: U[(i_a+1)*m + j_a], d: U[(i_a+1)*m + (j_a+1)]
        # x: V[j_b*p + k_b]
        # r: W[k_b*n + (i_a+1)]
        
        U[(i_a+1)*m + j_a][col] = 1
        U[(i_a+1)*m + (j_a+1)][col] = 1
        V[j_b*p + k_b][col] = 1
        W[k_b*n + (i_a+1)][col] = 1
        col += 1
        
        # M3: a(y-u)
        # a: U[i_a*m + j_a]
        # y: V[j_b*p + (k_b+1)], u: V[(j_b+1)*p + k_b]
        # q: W[(k_b+1)*n + i_a]
        
        U[i_a*m + j_a][col] = 1
        V[j_b*p + (k_b+1)][col] = 1
        V[(j_b+1)*p + k_b][col] = -1
        W[(k_b+1)*n + i_a][col] = 1
        col += 1
        
        # M4: d(v-x)
        # d: U[(i_a+1)*m + (j_a+1)]
        # v: V[(j_b+1)*p + (k_b+1)], x: V[j_b*p + k_b]
        # p: W[k_b*n + i_a], r: W[k_b*n + (i_a+1)]
        
        U[(i_a+1)*m + (j_a+1)][col] = 1
        V[(j_b+1)*p + (k_b+1)][col] = 1
        V[j_b*p + k_b][col] = -1
        W[k_b*n + i_a][col] = 1
        W[k_b*n + (i_a+1)][col] = 1
        col += 1
        
        # M5: (a+b)v
        # a: U[i_a*m + j_a], b: U[i_a*m + (j_a+1)]
        # v: V[(j_b+1)*p + (k_b+1)]
        # p: W[k_b*n + i_a], q: W[(k_b+1)*n + i_a]
        
        U[i_a*m + j_a][col] = 1
        U[i_a*m + (j_a+1)][col] = 1
        V[(j_b+1)*p + (k_b+1)][col] = 1
        W[k_b*n + i_a][col] = -1
        W[(k_b+1)*n + i_a][col] = 1
        col += 1
        
        # M6: (c-a)(x+y)
        # c: U[(i_a+1)*m + j_a], a: U[i_a*m + j_a]
        # x: V[j_b*p + k_b], y: V[j_b*p + (k_b+1)]
        # q: W[(k_b+1)*n + i_a], s: W[(k_b+1)*n + (i_a+1)]
        
        U[(i_a+1)*m + j_a][col] = 1
        U[i_a*m + j_a][col] = -1
        V[j_b*p + k_b][col] = 1
        V[j_b*p + (k_b+1)][col] = 1
        W[(k_b+1)*n + i_a][col] = 1
        W[(k_b+1)*n + (i_a+1)][col] = 1
        col += 1
        
        # M7: (b-d)(u+v)
        # b: U[i_a*m + (j_a+1)], d: U[(i_a+1)*m + (j_a+1)]
        # u: V[(j_b+1)*p + k_b], v: V[(j_b+1)*p + (k_b+1)]
        # p: W[k_b*n + i_a], s: W[(k_b+1)*n + (i_a+1)]
        
        U[i_a*m + (j_a+1)][col] = 1
        U[(i_a+1)*m + (j_a+1)][col] = -1
        V[(j_b+1)*p + k_b][col] = 1
        V[(j_b+1)*p + (k_b+1)][col] = 1
        W[k_b*n + i_a][col] = 1
        W[(k_b+1)*n + (i_a+1)][col] = 1
        col += 1

    def add_dot_122(i_a, j_a, j_b, k_b):
        nonlocal col
        # i_a: row of A (2)
        # j_a: start col of A (0 or 2)
        # j_b: start row of B (0 or 2)
        # k_b: start col of B (0, 2, 4)
        
        # Term 1: (a+b) * (x+u) -> p
        # a: U[i_a*m + j_a], b: U[i_a*m + (j_a+1)]
        # x: V[j_b*p + k_b], u: V[(j_b+1)*p + k_b]
        # p: W[k_b*n + i_a]
        
        U[i_a*m + j_a][col] = 1
        U[i_a*m + (j_a+1)][col] = 1
        V[j_b*p + k_b][col] = 1
        V[(j_b+1)*p + k_b][col] = 1
        W[k_b*n + i_a][col] = 1
        col += 1
        
        # Term 2: (a+b) * (y+v) -> q
        # y: V[j_b*p + (k_b+1)], v: V[(j_b+1)*p + (k_b+1)]
        # q: W[(k_b+1)*n + i_a]
        
        U[i_a*m + j_a][col] = 1
        U[i_a*m + (j_a+1)][col] = 1
        V[j_b*p + (k_b+1)][col] = 1
        V[(j_b+1)*p + (k_b+1)][col] = 1
        W[(k_b+1)*n + i_a][col] = 1
        col += 1

    # Block L: A[:, 0:2] * B[0:2, :]
    # Top: A[0:2, 0:2] * B[0:2, 0:6]
    add_strassen_222(0, 0, 0, 0)
    add_strassen_222(0, 0, 0, 2)
    add_strassen_222(0, 0, 0, 4)
    
    # Bot: A[2:3, 0:2] * B[0:2, 0:6]
    add_dot_122(2, 0, 0, 0)
    add_dot_122(2, 0, 0, 2)
    add_dot_122(2, 0, 0, 4)
    
    # Block R: A[:, 2:4] * B[2:4, :]
    # Top: A[0:2, 2:4] * B[2:4, 0:6]
    add_strassen_222(0, 2, 2, 0)
    add_strassen_222(0, 2, 2, 2)
    add_strassen_222(0, 2, 2, 4)
    
    # Bot: A[2:3, 2:4] * B[2:4, 0:6]
    add_dot_122(2, 2, 2, 0)
    add_dot_122(2, 2, 2, 2)
    add_dot_122(2, 2, 2, 4)
    
    return {"U": U, "V": V, "W": W}
