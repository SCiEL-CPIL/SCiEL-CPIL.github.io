import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    def add_strassen_block(a_row, a_col, b_row, b_col, c_row, c_col):
        nonlocal r
        # Map local indices to global
        # A indices: i in 0..1, j in 0..1
        # Global A index: (a_row + i) * m + (a_col + j)
        # B indices: j in 0..1, k in 0..1
        # Global B index: (b_row + j) * p + (b_col + k)
        # C indices: k in 0..1, i in 0..1
        # Global C index: (c_col + k) * n + (c_row + i)
        
        def get_u(i, j): return (a_row + i) * m + (a_col + j)
        def get_v(j, k): return (b_row + j) * p + (b_col + k)
        def get_w(k, i): return (c_col + k) * n + (c_row + i)
        
        # M1 = (A11 - A00) * (B10 + B11)
        U[get_u(1,1)][r] = 1
        U[get_u(0,0)][r] = -1
        V[get_v(1,0)][r] = 1
        V[get_v(1,1)][r] = 1
        W[get_w(0,0)][r] = 1
        W[get_w(0,1)][r] = -1
        W[get_w(1,0)][r] = 1
        W[get_w(1,1)][r] = -1
        r += 1
        
        # M2 = (A11 + A00) * B11
        U[get_u(1,1)][r] = 1
        U[get_u(0,0)][r] = 1
        V[get_v(1,1)][r] = 1
        W[get_w(0,1)][r] = 1
        W[get_w(1,0)][r] = -1
        r += 1
        
        # M3 = A00 * (B01 - B11)
        U[get_u(0,0)][r] = 1
        V[get_v(0,1)][r] = 1
        V[get_v(1,1)][r] = -1
        W[get_w(0,0)][r] = -1
        W[get_w(1,1)][r] = 1
        r += 1
        
        # M4 = A11 * (B00 + B10)
        U[get_u(1,1)][r] = 1
        V[get_v(0,0)][r] = 1
        V[get_v(1,0)][r] = 1
        W[get_w(0,0)][r] = 1
        W[get_w(1,1)][r] = -1
        r += 1
        
        # M5 = (A00 + A01) * B00
        U[get_u(0,0)][r] = 1
        U[get_u(0,1)][r] = 1
        V[get_v(0,0)][r] = 1
        W[get_w(0,1)][r] = 1
        W[get_w(1,0)][r] = 1
        r += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[get_u(1,0)][r] = 1
        U[get_u(0,0)][r] = -1
        V[get_v(0,0)][r] = 1
        V[get_v(0,1)][r] = 1
        W[get_w(1,1)][r] = 1
        r += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[get_u(0,1)][r] = 1
        U[get_u(1,1)][r] = -1
        V[get_v(1,0)][r] = 1
        V[get_v(1,1)][r] = 1
        W[get_w(0,0)][r] = 1
        W[get_w(1,1)][r] = -1
        r += 1

    def add_tail_block(a_row, a_col, b_row, b_col, c_row, c_col):
        nonlocal r
        # 2x2x1 block
        # A is 2x2, B is 2x1, C is 2x1
        def get_u(i, j): return (a_row + i) * m + (a_col + j)
        def get_v(j, k): return (b_row + j) * p + b_col
        def get_w(k, i): return c_col * n + (c_row + i)
        
        # C00 = A00 B00
        U[get_u(0,0)][r] = 1
        V[get_v(0,0)][r] = 1
        W[get_w(0,0)][r] = 1
        r += 1
        
        # C00 += A01 B10
        U[get_u(0,1)][r] = 1
        V[get_v(1,0)][r] = 1
        W[get_w(0,0)][r] = 1
        r += 1
        
        # C10 = A10 B00
        U[get_u(1,0)][r] = 1
        V[get_v(0,0)][r] = 1
        W[get_w(0,1)][r] = 1
        r += 1
        
        # C10 += A11 B10
        U[get_u(1,1)][r] = 1
        V[get_v(1,0)][r] = 1
        W[get_w(0,1)][r] = 1
        r += 1

    # Blocks:
    # m=0: a_row=0, b_row=0
    # m=1: a_row=0, b_row=2
    # p=0: a_col=0, b_col=0, c_col=0
    # p=1: a_col=2, b_col=2, c_col=2
    # Tail: b_col=4, c_col=4 (only 1 col)
    
    # Note: A cols are split 2,2. So a_col is 0 or 2.
    # B rows are split 2,2. So b_row is 0 or 2.
    # B cols are split 2,2,1. So b_col is 0, 2, 4.
    # C cols are split 2,2,1. So c_col is 0, 2, 4.
    # A rows 0..1. C rows 0..1.
    
    # Main blocks (2x2x2)
    for a_col in [0, 2]:
        for b_row in [0, 2]:
            for p_col in [0, 2]:
                add_strassen_block(0, a_col, b_row, p_col, 0, p_col)
                
    # Tail blocks (2x2x1)
    for b_row in [0, 2]:
        add_tail_block(0, 0 if b_row==0 else 2, b_row, 4, 0, 4)

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(res["U"][0])
