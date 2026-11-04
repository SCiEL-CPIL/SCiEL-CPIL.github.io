import numpy as np

def construct():
    # Correct Strassen factors for 2x2 matrices
    # U2 rows: A00, A01, A10, A11
    # V2 rows: B00, B01, B10, B11
    # W2 rows: C00, C01, C10, C11
    
    U2 = np.array([
        [1, 0, 1, 0, 1, 0, 0], # A00
        [0, 0, 0, 0, 1, 1, 1], # A01
        [0, 1, 0, 0, 0, -1, 0],# A10
        [1, 1, 0, 1, 0, 0, -1] # A11
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ], dtype=int)
    
    # Kronecker products
    Kron_U = np.kron(U2, U2)
    Kron_V = np.kron(V2, V2)
    Kron_W = np.kron(W2, W2)
    
    # Permute rows to match standard indexing
    # U: A is 4x4. Standard index s = r*4 + c.
    # Kron index I = 8*r1 + 4*c1 + 2*r2 + c2.
    # s = 8*r1 + 4*r2 + 2*c1 + c2.
    U = np.zeros_like(Kron_U)
    for s in range(16):
        r1 = s // 8
        r2 = (s // 4) % 2
        c1 = (s // 2) % 2
        c2 = s % 2
        I = 8 * r1 + 4 * c1 + 2 * r2 + c2
        U[s, :] = Kron_U[I, :]
        
    # V: B is 4x4. Standard index s = k*4 + j.
    # Kron index J = 8*c1 + 4*b1 + 2*c2 + b2. (c=k, b=j)
    # s = 8*c1 + 4*c2 + 2*b1 + b2.
    V = np.zeros_like(Kron_V)
    for s in range(16):
        c1 = s // 8
        c2 = (s // 4) % 2
        b1 = (s // 2) % 2
        b2 = s % 2
        J = 8 * c1 + 4 * b1 + 2 * c2 + b2
        V[s, :] = Kron_V[J, :]
        
    # W: C is 4x4. Convention W[k*n+i, r] with n=4.
    # Index k_w = j*4 + i.
    # Kron index L = 8*i1 + 4*j1 + 2*i2 + j2.
    # k_w = 8*j1 + 4*j2 + 2*i1 + i2.
    W = np.zeros_like(Kron_W)
    for k in range(16):
        j1 = k // 8
        j2 = (k // 4) % 2
        i1 = (k // 2) % 2
        i2 = k % 2
        L = 8 * i1 + 4 * j1 + 2 * i2 + j2
        W[k, :] = Kron_W[L, :]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
