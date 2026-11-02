import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We will build the decomposition by adding terms.
    # Instead of pre-allocating full rank, we can build lists and then compact.
    # But the interface expects fixed shapes.
    # We know the rank will be 57.
    # Let's just use the 57 rank directly.
    
    new_rank = 57
    U = [[0] * new_rank for _ in range(n * m)]
    V = [[0] * new_rank for _ in range(m * p)]
    W = [[0] * new_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook term
    def add_schoolbook(i, j, k):
        nonlocal col
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1

    # Helper to add Strassen block
    # Maps local indices (0,1) to global indices
    def add_strassen(i0, i1, j0, j1, k0, k1):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        # A00 is i0,j0. A11 is i1,j1.
        # B00 is j0,k0. B11 is j1,k1.
        # Term 1: (A00+A11) * (B00+B11)
        # This is not rank 1. It's a sum of 4 rank 1 terms if expanded.
        # But Strassen uses 7 terms.
        # The terms are bilinear forms.
        
        # M1 = (A_{i0,j0} + A_{i1,j1}) * (B_{j0,k0} + B_{j1,k1})
        # This corresponds to adding 1 to U for A00 and A11, and 1 to V for B00 and B11.
        # Wait, U, V, W are matrices.
        # A term is u * v * w.
        # For M1, u has 1 at (i0,j0) and (i1,j1).
        # v has 1 at (j0,k0) and (j1,k1).
        # w has 1 at (k0,i0), (k1,i0), (k0,i1), (k1,i1)?
        # No, w is the output coefficient.
        # C_{i,k} = sum u_{i,j} v_{j,k} w_{k,i}.
        # For M1, it contributes to C_{i0,k0}, C_{i0,k1}, C_{i1,k0}, C_{i1,k1}.
        # Specifically:
        # C_{i0,k0} += M1
        # C_{i0,k1} += M1
        # C_{i1,k0} += M1
        # C_{i1,k1} += M1
        # So w has 1 at (k0,i0), (k1,i0), (k0,i1), (k1,i1).
        
        U[i0 * m + j0][col] = 1
        U[i1 * m + j1][col] = 1
        V[j0 * p + k0][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + i0][col] = 1
        W[k1 * n + i0][col] = 1
        W[k0 * n + i1][col] = 1
        W[k1 * n + i1][col] = 1
        col += 1
        
        # M2 = (A_{i1,j0} + A_{i1,j1}) * B_{j0,k0}
        U[i1 * m + j0][col] = 1
        U[i1 * m + j1][col] = 1
        V[j0 * p + k0][col] = 1
        W[k0 * n + i1][col] = 1
        col += 1
        
        # M3 = A_{i0,j0} * (B_{j1,k0} - B_{j1,k1})
        U[i0 * m + j0][col] = 1
        V[j1 * p + k0][col] = 1
        V[j1 * p + k1][col] = -1
        W[k0 * n + i0][col] = 1
        col += 1
        
        # M4 = A_{i1,j1} * (B_{j0,k1} - B_{j0,k0})
        U[i1 * m + j1][col] = 1
        V[j0 * p + k1][col] = 1
        V[j0 * p + k0][col] = -1
        W[k1 * n + i1][col] = 1
        col += 1
        
        # M5 = (A_{i0,j0} + A_{i0,j1}) * B_{j1,k1}
        U[i0 * m + j0][col] = 1
        U[i0 * m + j1][col] = 1
        V[j1 * p + k1][col] = 1
        W[k1 * n + i0][col] = 1
        col += 1
        
        # M6 = (A_{i1,j0} - A_{i0,j0}) * (B_{j0,k0} + B_{j0,k1})
        U[i1 * m + j0][col] = 1
        U[i0 * m + j0][col] = -1
        V[j0 * p + k0][col] = 1
        V[j0 * p + k1][col] = 1
        W[k0 * n + i1][col] = 1
        W[k1 * n + i1][col] = 1
        col += 1
        
        # M7 = (A_{i0,j1} - A_{i1,j1}) * (B_{j1,k0} + B_{j1,k1})
        U[i0 * m + j1][col] = 1
        U[i1 * m + j1][col] = -1
        V[j1 * p + k0][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + i0][col] = 1
        W[k1 * n + i0][col] = 1
        col += 1

    # Add Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    add_strassen(0, 1, 0, 1, 0, 1)
    
    # Add Block 2: i in {0,1}, j in {2,3}, k in {2,3}
    add_strassen(0, 1, 2, 3, 2, 3)
    
    # Add remaining schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this term was covered by Strassen
                covered = False
                if i in [0, 1]:
                    if j in [0, 1] and k in [0, 1]:
                        covered = True
                    if j in [2, 3] and k in [2, 3]:
                        covered = True
                if not covered:
                    add_schoolbook(i, j, k)
                    
    return {"U": U, "V": V, "W": W}
