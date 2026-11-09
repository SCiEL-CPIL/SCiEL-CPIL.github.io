import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    term = 0
    
    # Helper to add a Strassen block
    def add_strassen_block(h, j):
        nonlocal term
        # A indices
        # A[r, k] -> U[r*4 + k]
        # h=0: k=0,1. h=1: k=2,3.
        # r=0,1
        a00 = 0 * 4 + (h * 2 + 0)
        a01 = 0 * 4 + (h * 2 + 1)
        a10 = 1 * 4 + (h * 2 + 0)
        a11 = 1 * 4 + (h * 2 + 1)
        
        # B indices
        # B[k, c] -> V[k*8 + c]
        # h=0: k=0,1. h=1: k=2,3.
        # j=0: c=0,1. j=1: c=2,3. etc.
        b00 = (h * 2 + 0) * 8 + (j * 2 + 0)
        b01 = (h * 2 + 0) * 8 + (j * 2 + 1)
        b10 = (h * 2 + 1) * 8 + (j * 2 + 0)
        b11 = (h * 2 + 1) * 8 + (j * 2 + 1)
        
        # C indices
        # C[r, c] -> W[c*2 + r]
        # c = 2j, 2j+1
        # r = 0, 1
        c00 = (j * 2 + 0) * 2 + 0
        c01 = (j * 2 + 1) * 2 + 0
        c10 = (j * 2 + 0) * 2 + 1
        c11 = (j * 2 + 1) * 2 + 1
        
        # Term 1: M1 = (A00 + A11)(B00 + B11)
        U[a00][term] += 1
        U[a11][term] += 1
        V[b00][term] += 1
        V[b11][term] += 1
        W[c00][term] += 1
        W[c11][term] += 1
        term += 1
        
        # Term 2: M2 = (A10 + A11)B00
        U[a10][term] += 1
        U[a11][term] += 1
        V[b00][term] += 1
        W[c10][term] += 1
        W[c11][term] += -1
        term += 1
        
        # Term 3: M3 = A00(B01 - B11)
        U[a00][term] += 1
        V[b01][term] += 1
        V[b11][term] += -1
        W[c00][term] += 1
        W[c01][term] += 1
        term += 1
        
        # Term 4: M4 = A11(B10 - B00)
        U[a11][term] += 1
        V[b10][term] += 1
        V[b00][term] += -1
        W[c00][term] += 1
        W[c10][term] += 1
        term += 1
        
        # Term 5: M5 = (A00 + A01)B11
        U[a00][term] += 1
        U[a01][term] += 1
        V[b11][term] += 1
        W[c00][term] += -1
        W[c01][term] += 1
        term += 1
        
        # Term 6: M6 = (A10 - A00)(B00 + B01)
        U[a10][term] += 1
        U[a00][term] += -1
        V[b00][term] += 1
        V[b01][term] += 1
        W[c11][term] += 1
        term += 1
        
        # Term 7: M7 = (A01 - A11)(B10 + B11)
        U[a01][term] += 1
        U[a11][term] += -1
        V[b10][term] += 1
        V[b11][term] += 1
        W[c00][term] += 1
        term += 1
        
    for h in range(2):
        for j in range(4):
            add_strassen_block(h, j)
            
    return {"U": U, "V": V, "W": W}
