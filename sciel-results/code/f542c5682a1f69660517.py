"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Constructs a rank-36 tensor decomposition for matrix multiplication C = A @ B
    where A is 2x4 and B is 4x5.
    Uses a block structure with Strassen for 2x2x2 blocks and direct multiplication for 2x2x1.
    """
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    timeout = 100.0
    
    rank = 36
    
    # Shapes
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    def u_idx(i, j): return i*m + j
    def v_idx(j, k): return j*p + k
    def w_idx(k, i): return k*n + i
    
    col = 0
    
    def add_strassen_2x2x2(ai0, ai1, aj0, aj1, bj0, bj1, bk0, bk1):
        nonlocal col
        a00 = u_idx(ai0, aj0)
        a01 = u_idx(ai0, aj1)
        a10 = u_idx(ai1, aj0)
        a11 = u_idx(ai1, aj1)
        
        b00 = v_idx(bj0, bk0)
        b01 = v_idx(bj0, bk1)
        b10 = v_idx(bj1, bk0)
        b11 = v_idx(bj1, bk1)
        
        c00 = w_idx(bk0, ai0)
        c01 = w_idx(bk1, ai0)
        c10 = w_idx(bk0, ai1)
        c11 = w_idx(bk1, ai1)
        
        # M1
        U[a00][col] += 1; U[a11][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        
        # M2
        U[a10][col] += 1; U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1; W[c11][col] -= 1
        col += 1
        
        # M3
        U[a00][col] += 1
        V[b01][col] += 1; V[b11][col] -= 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1
        
        # M4
        U[a11][col] += 1
        V[b10][col] += 1; V[b00][col] -= 1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1
        
        # M5
        U[a00][col] += 1; U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1; W[c01][col] += 1
        col += 1
        
        # M6
        U[a10][col] += 1; U[a00][col] -= 1
        V[b00][col] += 1; V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7
        U[a01][col] += 1; U[a11][col] -= 1
        V[b10][col] += 1; V[b11][col] += 1
        W[c00][col] += 1
        col += 1
        
    def add_vec_mat_2x2x1(ai0, ai1, aj0, aj1, bj0, bj1, bk):
        nonlocal col
        a00 = u_idx(ai0, aj0)
        a01 = u_idx(ai0, aj1)
        a10 = u_idx(ai1, aj0)
        a11 = u_idx(ai1, aj1)
        
        b0 = v_idx(bj0, bk)
        b1 = v_idx(bj1, bk)
        
        c0 = w_idx(bk, ai0)
        c1 = w_idx(bk, ai1)
        
        U[a00][col] += 1; V[b0][col] += 1; W[c0][col] += 1; col += 1
        U[a01][col] += 1; V[b1][col] += 1; W[c0][col] += 1; col += 1
        U[a10][col] += 1; V[b0][col] += 1; W[c1][col] += 1; col += 1
        U[a11][col] += 1; V[b1][col] += 1; W[c1][col] += 1; col += 1

    add_strassen_2x2x2(0, 1, 0, 1, 0, 1, 0, 1)
    add_strassen_2x2x2(0, 1, 0, 1, 0, 1, 2, 3)
    add_vec_mat_2x2x1(0, 1, 0, 1, 0, 1, 4)
    add_strassen_2x2x2(0, 1, 2, 3, 2, 3, 0, 1)
    add_strassen_2x2x2(0, 1, 2, 3, 2, 3, 2, 3)
    add_vec_mat_2x2x1(0, 1, 2, 3, 2, 3, 4)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
