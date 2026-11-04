"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]
MAX_STEPS = 8

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    base_rank = n * m * p
    best = _schoolbook(n, m, p)
    best_rank = base_rank
    
    return best

# EVOLVE-BLOCK-START
def _construct_rank58():
    n, m, p = 3, 4, 5
    cols = []
    
    def zero_u(): return [0.0] * 12
    def zero_v(): return [0.0] * 20
    def zero_w(): return [0.0] * 15
    
    def set_u(vec, i, j, val): vec[i*4 + j] = val
    def set_v(vec, j, k, val): vec[j*5 + k] = val
    def set_w(vec, k, i, val): vec[k*3 + i] = val
    
    def add_term(u, v, w):
        cols.append((u, v, w))
        
    def add_strassen(i0, j0, k0):
        # M1
        u, v, w = zero_u(), zero_v(), zero_w()
        set_u(u, i0, j0, 1); set_u(u, i0+1, j0+1, 1)
        set_v(v, j0, k0, 1); set_v(v, j0+1, k0+1, 1)
        set_w(w, k0, i0, 1); set_w(w, k0+1, i0+1, 1)
        add_term(u, v, w)
        
        # M2
        u, v, w = zero_u(), zero_v(), zero_w()
        set_u(u, i0+1, j0, 1); set_u(u, i0+1, j0+1, 1)
        set_v(v, j0, k0, 1)
        set_w(w, k0+1, i0, 1); set_w(w, k0+1, i0+1, -1)
        add_term(u, v, w)
        
        # M3
        u, v, w = zero_u(), zero_v(), zero_w()
        set_u(u, i0, j0, 1)
        set_v(v, j0, k0+1, 1); set_v(v, j0+1, k0+1, -1)
        set_w(w, k0, i0+1, 1); set_w(w, k0+1, i0+1, 1)
        add_term(u, v, w)
        
        # M4
        u, v, w = zero_u(), zero_v(), zero_w()
        set_u(u, i0+1, j0+1, 1)
        set_v(v, j0+1, k0, 1); set_v(v, j0, k0, -1)
        set_w(w, k0, i0, 1); set_w(w, k0+1, i0, 1)
        add_term(u, v, w)
        
        # M5
        u, v, w = zero_u(), zero_v(), zero_w()
        set_u(u, i0, j0, 1); set_u(u, i0+1, j0, 1)
        set_v(v, j0+1, k0+1, 1)
        set_w(w, k0, i0, -1); set_w(w, k0, i0+1, 1)
        add_term(u, v, w)
        
        # M6
        u, v, w = zero_u(), zero_v(), zero_w()
        set_u(u, i0, j0+1, 1); set_u(u, i0, j0, -1)
        set_v(v, j0, k0, 1); set_v(v, j0+1, k0, 1)
        set_w(w, k0+1, i0+1, 1)
        add_term(u, v, w)
        
        # M7
        u, v, w = zero_u(), zero_v(), zero_w()
        set_u(u, i0+1, j0, 1); set_u(u, i0+1, j0+1, -1)
        set_v(v, j0, k0+1, 1); set_v(v, j0+1, k0+1, 1)
        set_w(w, k0, i0, 1)
        add_term(u, v, w)
        
    def add_schoolbook(i0, j0, k0, ni, mj, pk):
        for i in range(ni):
            for j in range(mj):
                for k in range(pk):
                    u, v, w = zero_u(), zero_v(), zero_w()
                    set_u(u, i0+i, j0+j, 1)
                    set_v(v, j0+j, k0+k, 1)
                    set_w(w, k0+k, i0+i, 1)
                    add_term(u, v, w)
                    
    add_strassen(0, 0, 0)
    add_strassen(0, 2, 0)
    add_schoolbook(0, 0, 2, 2, 2, 3)
    add_schoolbook(0, 2, 2, 2, 2, 3)
    add_schoolbook(2, 0, 0, 1, 2, 2)
    add_schoolbook(2, 2, 0, 1, 2, 2)
    add_schoolbook(2, 0, 2, 1, 2, 3)
    add_schoolbook(2, 2, 2, 1, 2, 3)
    
    rank = len(cols)
    U = [[0.0] * rank for _ in range(12)]
    V = [[0.0] * rank for _ in range(20)]
    W = [[0.0] * rank for _ in range(15)]
    
    for r, (u, v, w) in enumerate(cols):
        for i in range(12):
            U[i][r] = u[i]
        for j in range(20):
            V[j][r] = v[j]
        for k in range(15):
            W[k][r] = w[k]
            
    return {"U": U, "V": V, "W": W}

def construct():
    return _construct_rank58()
# EVOLVE-BLOCK-END
