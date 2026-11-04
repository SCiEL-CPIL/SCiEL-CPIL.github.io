import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    # Strassen 2x2 decomposition
    # U2, V2, W2 are 4x7
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])

    # Rank-11 2x3 decomposition
    # U23, V23, W23
    # U23 is 4x11
    U23 = np.zeros((4, 11))
    U23[:, :7] = U2
    # Last 4 columns for A * B_right (schoolbook)
    U23[0, 7] = 1
    U23[1, 8] = 1
    U23[2, 9] = 1
    U23[3, 10] = 1

    # V23 is 6x11
    # Rows: B00, B01, B02, B10, B11, B12
    V23 = np.zeros((6, 11))
    # First 7 cols from V2, mapped to rows 0,1,3,4
    V23[0, :7] = V2[0, :7]
    V23[1, :7] = V2[1, :7]
    V23[3, :7] = V2[2, :7]
    V23[4, :7] = V2[3, :7]
    # Rows 2 and 5 are 0 for first 7 cols
    
    # Last 4 cols
    # A00*B02 -> V col 7 has B02=1 (row 2)
    V23[2, 7] = 1
    # A01*B12 -> V col 8 has B12=1 (row 5)
    V23[5, 8] = 1
    # A10*B02 -> V col 9 has B02=1 (row 2)
    V23[2, 9] = 1
    # A11*B12 -> V col 10 has B12=1 (row 5)
    V23[5, 10] = 1

    # W23 is 6x11
    # Rows: C00, C01, C02, C10, C11, C12
    W23 = np.zeros((6, 11))
    # First 7 cols from W2, mapped to rows 0,1,3,4
    W23[0, :7] = W2[0, :7]
    W23[1, :7] = W2[1, :7]
    W23[3, :7] = W2[2, :7]
    W23[4, :7] = W2[3, :7]
    # Rows 2 and 5 are 0 for first 7 cols

    # Last 4 cols
    # C02 = A00*B02 + A01*B12
    W23[2, 7] = 1
    W23[2, 8] = 1
    # C12 = A10*B02 + A11*B12
    W23[5, 9] = 1
    W23[5, 10] = 1

    # Kronecker product
    U = np.kron(U2, U23)
    V = np.kron(V2, V23)
    W = np.kron(W2, W23)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
