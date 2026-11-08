import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank = 17
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to set values
    def set_val(mat, idx, col, val):
        mat[idx, col] = val

    # 1. Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: 0,1,3,4
    # B indices: 0,1,3,4
    # C indices: 0,2,1,3 (W[k*n+i])
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    set_val(U, 0, 0, 1); set_val(U, 4, 0, 1)
    set_val(V, 0, 0, 1); set_val(V, 4, 0, 1)
    set_val(W, 0, 0, 1); set_val(W, 3, 0, 1)
    
    # M2: (A10+A11)B00 -> C10, -C11
    set_val(U, 3, 1, 1); set_val(U, 4, 1, 1)
    set_val(V, 0, 1, 1)
    set_val(W, 1, 1, 1); set_val(W, 3, 1, -1)
    
    # M3: A00(B01-B11) -> C01, C11
    set_val(U, 0, 2, 1)
    set_val(V, 1, 2, 1); set_val(V, 4, 2, -1)
    set_val(W, 2, 2, 1); set_val(W, 3, 2, 1)
    
    # M4: A11(B10-B00) -> C00, C10
    set_val(U, 4, 3, 1)
    set_val(V, 3, 3, 1); set_val(V, 0, 3, -1)
    set_val(W, 0, 3, 1); set_val(W, 1, 3, 1)
    
    # M5: (A00+A01)B11 -> -C00, C01
    set_val(U, 0, 4, 1); set_val(U, 1, 4, 1)
    set_val(V, 4, 4, 1)
    set_val(W, 0, 4, -1); set_val(W, 2, 4, 1)
    
    # M6: (A10-A00)(B00+B01) -> C11
    set_val(U, 3, 5, 1); set_val(U, 0, 5, -1)
    set_val(V, 0, 5, 1); set_val(V, 1, 5, 1)
    set_val(W, 3, 5, 1)
    
    # M7: (A01-A11)(B10+B11) -> C00
    set_val(U, 1, 6, 1); set_val(U, 4, 6, -1)
    set_val(V, 3, 6, 1); set_val(V, 4, 6, 1)
    set_val(W, 0, 6, 1)

    # 2. A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    # A: 2, 5. B: 6, 7. C: 0, 2, 1, 3
    r = 7
    set_val(U, 2, r, 1); set_val(V, 6, r, 1); set_val(W, 0, r, 1)
    r += 1
    set_val(U, 2, r, 1); set_val(V, 7, r, 1); set_val(W, 2, r, 1)
    r += 1
    set_val(U, 5, r, 1); set_val(V, 6, r, 1); set_val(W, 1, r, 1)
    r += 1
    set_val(U, 5, r, 1); set_val(V, 7, r, 1); set_val(W, 3, r, 1)
    r += 1
    
    # 3. A[0:2, 0:2] * B[0:1, 2] -> C[0:2, 2]
    # A: 0,1,3,4. B: 2,5. C: 4,5
    set_val(U, 0, r, 1); set_val(V, 2, r, 1); set_val(W, 4, r, 1)
    r += 1
    set_val(U, 1, r, 1); set_val(V, 5, r, 1); set_val(W, 4, r, 1)
    r += 1
    set_val(U, 3, r, 1); set_val(V, 2, r, 1); set_val(W, 5, r, 1)
    r += 1
    set_val(U, 4, r, 1); set_val(V, 5, r, 1); set_val(W, 5, r, 1)
    r += 1
    
    # 4. A[0:2, 2] * B[2, 2] -> C[0:2, 2]
    # A: 2,5. B: 8. C: 4,5
    set_val(U, 2, r, 1); set_val(V, 8, r, 1); set_val(W, 4, r, 1)
    r += 1
    set_val(U, 5, r, 1); set_val(V, 8, r, 1); set_val(W, 5, r, 1)
    r += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
