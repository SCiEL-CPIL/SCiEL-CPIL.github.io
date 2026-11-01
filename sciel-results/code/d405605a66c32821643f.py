import numpy as np

def construct():
    n, m, p = 2, 2, 3
    R = 11
    
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Indices:
    # U: A00=0, A01=1, A10=2, A11=3
    # V: B00=0, B01=1, B02=2, B10=3, B11=4, B12=5
    # W: C00=0, C10=1, C01=2, C11=3, C02=4, C12=5
    
    # Strassen M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # Strassen M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # Strassen M3
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Strassen M4
    U[3, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = -1
    
    # Strassen M5
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Strassen M6
    U[1, 5] = 1; U[2, 5] = -1
    V[0, 5] = 1; V[3, 5] = 1
    W[2, 5] = -1; W[3, 5] = 1
    
    # Strassen M7
    U[2, 6] = 1; U[0, 6] = -1
    V[1, 6] = 1; V[4, 6] = 1
    W[0, 6] = -1; W[1, 6] = 1
    
    # Schoolbook for column 2
    # M7 = A00 * B02
    U[0, 7] = 1
    V[2, 7] = 1
    W[4, 7] = 1
    
    # M8 = A01 * B12
    U[1, 8] = 1
    V[5, 8] = 1
    W[4, 8] = 1
    
    # M9 = A10 * B02
    U[2, 9] = 1
    V[2, 9] = 1
    W[5, 9] = 1
    
    # M10 = A11 * B12
    U[3, 10] = 1
    V[5, 10] = 1
    W[5, 10] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
