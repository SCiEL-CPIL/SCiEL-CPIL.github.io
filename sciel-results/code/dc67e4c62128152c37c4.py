import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    def build_rank39():
        rank = 39
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        
        def add_strassen_block(k_start):
            nonlocal col
            # B indices: row j, col k -> j*p+k
            b00 = 0 * p + k_start
            b01 = 0 * p + (k_start + 1)
            b10 = 1 * p + k_start
            b11 = 1 * p + (k_start + 1)
            # W indices: col k, row i -> k*n+i
            c00 = k_start * n + 0
            c01 = (k_start + 1) * n + 0
            c10 = k_start * n + 1
            c11 = (k_start + 1) * n + 1
            
            # M1: (A00+A11)(B00+B11) -> C00, C11
            U[0, col] = 1; U[4, col] = 1
            V[b00, col] = 1; V[b11, col] = 1
            W[c00, col] = 1; W[c11, col] = 1
            col += 1
            
            # M2: (A10+A11)B00 -> C10, -C11
            U[3, col] = 1; U[4, col] = 1
            V[b00, col] = 1
            W[c10, col] = 1; W[c11, col] = -1
            col += 1
            
            # M3: A00(B01-B11) -> C01, C11
            U[0, col] = 1
            V[b01, col] = 1; V[b11, col] = -1
            W[c01, col] = 1; W[c11, col] = 1
            col += 1
            
            # M4: A11(B10-B00) -> C00, C10
            U[4, col] = 1
            V[b10, col] = 1; V[b00, col] = -1
            W[c00, col] = 1; W[c10, col] = 1
            col += 1
            
            # M5: (A00+A01)B11 -> -C00, C01
            U[0, col] = 1; U[1, col] = 1
            V[b11, col] = 1
            W[c00, col] = -1; W[c01, col] = 1
            col += 1
            
            # M6: (A10-A00)(B00+B01) -> C11
            U[3, col] = 1; U[0, col] = -1
            V[b00, col] = 1; V[b01, col] = 1
            W[c11, col] = 1
            col += 1
            
            # M7: (A01-A11)(B10+B11) -> C00
            U[1, col] = 1; U[4, col] = -1
            V[b10, col] = 1; V[b11, col] = 1
            W[c00, col] = 1
            col += 1
        
        add_strassen_block(0)
        add_strassen_block(2)
        add_strassen_block(4)
        
        # k=6, j=0,1 schoolbook
        for i in range(n):
            for j in range(2):
                U[i*m+j, col] = 1
                V[j*p+6, col] = 1
                W[6*n+i, col] = 1
                col += 1
        
        # j=2, all k
        for k in range(p):
            for i in range(n):
                U[i*m+2, col] = 1
                V[2*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
        
        return U, V, W
    
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j] * V[j*p+k], W[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    U, V, W = build_rank39()
    
    if not verify(U, V, W):
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
