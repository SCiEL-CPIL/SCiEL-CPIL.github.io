"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe
    
    n, m, p = DIMENSIONS
    
    def add_strassen(U, V, W, i0, j0, k0, col):
        # M1
        U[i0 * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M2
        U[(i0 + 1) * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] -= 1
        col += 1
        
        # M3
        U[i0 * m + j0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] -= 1
        W[(k0 + 1) * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M4
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0][col] += 1
        V[j0 * p + k0][col] -= 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        col += 1
        
        # M5
        U[i0 * m + j0][col] += 1
        U[i0 * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0][col] += 1
        W[k0 * n + i0][col] -= 1
        col += 1
        
        # M6
        U[(i0 + 1) * m + j0][col] += 1
        U[i0 * m + j0][col] -= 1
        V[j0 * p + k0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M7
        U[i0 * m + j0 + 1][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] -= 1
        V[(j0 + 1) * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1
        
        return col

    def add_schoolbook(U, V, W, i_range, j_range, k_range, col):
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return col

    # Total rank estimation
    # Part 1a: 3 Strassen (21) + 4 Schoolbook = 25
    # Part 1b: 14 Schoolbook
    # Part 2a: 3 Strassen (21) + 4 Schoolbook = 25
    # Part 2b: 14 Schoolbook
    # Part 2c: 21 Schoolbook
    # Total: 25 + 14 + 25 + 14 + 21 = 99
    
    rank = 99
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j in {0, 1}
    # Sub-part 1a: i in {0, 1}
    # Strassen blocks for k in {0,1}, {2,3}, {4,5}
    for k0 in [0, 2, 4]:
        if time.time() > deadline: break
        col = add_strassen(U, V, W, 0, 0, k0, col)
    
    # Schoolbook for i in {0,1}, j in {0,1}, k in {6}
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    col = add_schoolbook(U, V, W, [0, 1], [0, 1], [6], col)
    
    # Sub-part 1b: i in {2}
    # Schoolbook for i in {2}, j in {0,1}, k in {0..6}
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    col = add_schoolbook(U, V, W, [2], [0, 1], range(7), col)
    
    # Part 2: j in {2, 3, 4}
    # Sub-part 2a: i in {0, 1}, j in {2, 3}
    # Strassen blocks for k in {0,1}, {2,3}, {4,5}
    for k0 in [0, 2, 4]:
        if time.time() > deadline: break
        col = add_strassen(U, V, W, 0, 2, k0, col)
    
    # Schoolbook for i in {0,1}, j in {2,3}, k in {6}
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    col = add_schoolbook(U, V, W, [0, 1], [2, 3], [6], col)
    
    # Sub-part 2b: i in {0, 1}, j in {4}
    # Schoolbook for i in {0,1}, j in {4}, k in {0..6}
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    col = add_schoolbook(U, V, W, [0, 1], [4], range(7), col)
    
    # Sub-part 2c: i in {2}, j in {2, 3, 4}
    # Schoolbook for i in {2}, j in {2,3,4}, k in {0..6}
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    col = add_schoolbook(U, V, W, [2], [2, 3, 4], range(7), col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
