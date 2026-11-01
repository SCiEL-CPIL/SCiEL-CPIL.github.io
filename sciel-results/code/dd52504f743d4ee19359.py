import numpy as np

def construct():
    n, m, p = 2, 2, 3
    rank = 11
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to set values in U, V, W
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    def set_u(i, j, r, val):
        U[i * m + j, r] = val
        
    def set_v(j, k, r, val):
        V[j * p + k, r] = val
        
    def set_w(k, i, r, val):
        W[k * n + i, r] = val

    # Strassen-like decomposition for the 2x2 part (columns 0 and 1 of B)
    # This covers C[0,0], C[0,1], C[1,0], C[1,1]
    
    # Term 0: (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1); set_w(1, 1, 0, 1)

    # Term 1: (A[1,0] + A[1,1]) * B[0,0]
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1); set_w(1, 1, 1, -1)

    # Term 2: A[0,0] * (B[0,1] - B[1,1])
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1); set_w(1, 1, 2, 1)

    # Term 3: A[1,1] * (B[1,0] - B[0,0])
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1); set_w(0, 1, 3, 1)

    # Term 4: (A[0,0] + A[0,1]) * B[1,1]
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1); set_w(1, 0, 4, 1)

    # Term 5: (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    # Term 6: (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # Schoolbook for the 3rd column (column 2 of B)
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2]
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2]
    
    # Term 7: A[0,0] * B[0,2]
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    # Term 8: A[0,1] * B[1,2]
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    # Term 9: A[1,0] * B[0,2]
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    # Term 10: A[1,1] * B[1,2]
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}
