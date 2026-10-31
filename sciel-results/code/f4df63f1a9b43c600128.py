import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 4, 4)
    
    # 2x2 Strassen Factors
    # U2: 4x7, row i*2+j
    # V2: 4x7, row j*2+k
    # W2: 4x7, row k*2+i
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1 = (a00+a11)(b00+b11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1 # C00, C11
    
    # M2 = (a10+a11)b00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1 # C10, C11
    
    # M3 = a00(b01-b11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1 # C01, C11
    
    # M4 = a11(b10-b00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1 # C00, C10
    
    # M5 = (a00+a01)b11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1 # C00, C01
    
    # M6 = (a10-a00)(b00+b01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # C11
    
    # M7 = (a01-a11)(b10+b11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C00
    
    # Compute Kronecker products
    # U_km: 16x49. Row index a_km = (i0*2+j0)*4 + (i1*2+j1)
    U_km = np.kron(U2, U2)
    V_km = np.kron(V2, V2)
    W_km = np.kron(W2, W2)
    
    # Permute to row-major indices
    U_rm = np.zeros((16, 49), dtype=int)
    V_rm = np.zeros((16, 49), dtype=int)
    W_rm = np.zeros((16, 49), dtype=int)
    
    # Mapping for A (U)
    # a_km = (i0*2+j0)*4 + (i1*2+j1)
    # a_rm = (2*i0+i1)*4 + (2*j0+j1)
    for i0 in range(2):
        for i1 in range(2):
            for j0 in range(2):
                for j1 in range(2):
                    a_km = (i0*2 + j0)*4 + (i1*2 + j1)
                    a_rm = (2*i0 + i1)*4 + (2*j0 + j1)
                    U_rm[a_rm, :] = U_km[a_km, :]
                    
    # Mapping for B (V)
    # b_km = (j0*2+k0)*4 + (j1*2+k1)
    # b_rm = (2*j0+j1)*4 + (2*k0+k1)
    for j0 in range(2):
        for j1 in range(2):
            for k0 in range(2):
                for k1 in range(2):
                    b_km = (j0*2 + k0)*4 + (j1*2 + k1)
                    b_rm = (2*j0 + j1)*4 + (2*k0 + k1)
                    V_rm[b_rm, :] = V_km[b_km, :]
                    
    # Mapping for C (W)
    # c_km = (k0*2+i0)*4 + (k1*2+i1)
    # c_rm = (2*k0+k1)*4 + (2*i0+i1)
    for k0 in range(2):
        for k1 in range(2):
            for i0 in range(2):
                for i1 in range(2):
                    c_km = (k0*2 + i0)*4 + (k1*2 + i1)
                    c_rm = (2*k0 + k1)*4 + (2*i0 + i1)
                    W_rm[c_rm, :] = W_km[c_km, :]
    
    # Verify
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a = i*4 + j
                b = j*4 + k
                c = k*4 + i
                val = np.sum(U_rm[a, :] * V_rm[b, :] * W_rm[c, :])
                if val != 1:
                    # Debug print if needed, but raise error
                    raise ValueError(f"Verification failed at i={i}, j={j}, k={k}, val={val}")
                    
    return {"U": U_rm.tolist(), "V": V_rm.tolist(), "W": W_rm.tolist()}
