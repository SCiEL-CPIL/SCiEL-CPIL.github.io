import numpy as np

def construct():
    n = 2
    m = 4
    p = 5
    
    max_rank = 36
    U = [[0] * max_rank for _ in range(n * m)]
    V = [[0] * max_rank for _ in range(m * p)]
    W = [[0] * max_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        A00 = (i_base+0)*m + (j_base+0)
        A01 = (i_base+0)*m + (j_base+1)
        A10 = (i_base+1)*m + (j_base+0)
        A11 = (i_base+1)*m + (j_base+1)
        
        B00 = (j_base+0)*p + (k_base+0)
        B01 = (j_base+0)*p + (k_base+1)
        B10 = (j_base+1)*p + (k_base+0)
        B11 = (j_base+1)*p + (k_base+1)
        
        C00 = (k_base+0)*n + (i_base+0)
        C10 = (k_base+0)*n + (i_base+1)
        C01 = (k_base+1)*n + (i_base+0)
        C11 = (k_base+1)*n + (i_base+1)
        
        # M1 = (A00+A11)(B00+B11)
        U[A00][col] += 1; U[A11][col] += 1
        V[B00][col] += 1; V[B11][col] += 1
        W[C00][col] += 1
        W[C11][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[A10][col] += 1; U[A11][col] += 1
        V[B00][col] += 1
        W[C10][col] += 1
        W[C11][col] += -1
        col += 1
        
        # M3 = A00(B01-B11)
        U[A00][col] += 1
        V[B01][col] += 1; V[B11][col] += -1
        W[C01][col] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[A11][col] += 1
        V[B10][col] += 1; V[B00][col] += -1
        W[C00][col] += -1
        W[C10][col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[A00][col] += 1; U[A01][col] += 1
        V[B11][col] += 1
        W[C00][col] += 1
        W[C01][col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[A10][col] += 1; U[A00][col] += -1
        V[B00][col] += 1; V[B01][col] += 1
        W[C11][col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[A01][col] += 1; U[A11][col] += -1
        V[B10][col] += 1; V[B11][col] += 1
        W[C00][col] += -1
        W[C11][col] += 1
        col += 1

    def add_2x2x1(i_base, j_base, k_base):
        nonlocal col
        A00 = (i_base+0)*m + (j_base+0)
        A01 = (i_base+0)*m + (j_base+1)
        A10 = (i_base+1)*m + (j_base+0)
        A11 = (i_base+1)*m + (j_base+1)
        
        B0 = (j_base+0)*p + k_base
        B1 = (j_base+1)*p + k_base
        
        C0 = k_base*n + (i_base+0)
        C1 = k_base*n + (i_base+1)
        
        U[A00][col] += 1; V[B0][col] += 1; W[C0][col] += 1; col += 1
        U[A01][col] += 1; V[B1][col] += 1; W[C0][col] += 1; col += 1
        
        U[A10][col] += 1; V[B0][col] += 1; W[C1][col] += 1; col += 1
        U[A11][col] += 1; V[B1][col] += 1; W[C1][col] += 1; col += 1

    # C1: A1 B1a + A2 B2a
    add_strassen(0, 0, 0)
    add_strassen(0, 2, 0)
    
    # C2: A1 B1b + A2 B2b
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 2)
    
    # C3: A1 B1c + A2 B2c
    add_2x2x1(0, 0, 4)
    add_2x2x1(0, 2, 4)
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    return {'U': U_np, 'V': V_np, 'W': W_np}
