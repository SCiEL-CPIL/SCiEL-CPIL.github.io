import numpy as np

def construct():
    # Strassen factors for 2x2 matrices
    # Indices: 0:A, 1:B, 2:C, 3:D for left matrix
    #          0:E, 1:F, 2:G, 3:H for right matrix
    #          0:C11, 1:C12, 2:C21, 3:C22 for output
    
    U2 = np.array([
        [1, 0, 1, 0, 1, 0, 1], # A+D, C+D, A, D, A+B, C+D, A+B
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 1, 0]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, 0, 0, 0, 1], # E+H, E, F-H, G-F, H, G+H, E+F
        [0, 0, 1, -1, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 0, -1, 0, 1, 1, 0]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C11 = P1 + P4 - P5 + P7
        [0, 0, 1, 0, 1, 0, 0],  # C12 = P3 + P5
        [0, 1, 0, 1, 0, 0, 0],  # C21 = P2 + P4
        [1, -1, 1, 0, 0, 1, 0]  # C22 = P1 - P2 + P3 + P6
    ], dtype=int)
    
    # Kronecker product for 4x4
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Convert to list of lists for return
    U = U4.tolist()
    V = V4.tolist()
    W = W4.tolist()
    
    return {"U": U, "V": V, "W": W}
