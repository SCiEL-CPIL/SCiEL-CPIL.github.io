import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # Base 2x2 Strassen decomposition
    # U_2[i, alpha] where i = 2*i2 + i1, alpha in 0..6
    U_2 = np.array([
        [1, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, -1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    
    # V_2[j, alpha] where j = 2*j2 + j1, alpha in 0..6
    V_2 = np.array([
        [1, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 1]
    ])
    
    # W_2[k, alpha] where k = 2*k2 + k1, alpha in 0..6
    # W_2[k2*2+k1, alpha] gives the coefficient for C_{k2,k1} from product alpha
    W_2 = np.array([
        [1, 0, 0, 0, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Construct 4x4 decomposition via Kronecker products
    # The 4x4 matrix multiplication tensor is the tensor product of two 2x2 tensors
    # U_4 = U_2 ⊗ U_2, V_4 = V_2 ⊗ V_2, W_4 = W_2 ⊗ W_2
    U_4 = np.kron(U_2, U_2)
    V_4 = np.kron(V_2, V_2)
    W_4 = np.kron(W_2, W_2)
    
    # Convert to lists of lists with integer values
    U = [[int(x) for x in row] for row in U_4.tolist()]
    V = [[int(x) for x in row] for row in V_4.tolist()]
    W = [[int(x) for x in row] for row in W_4.tolist()]
    
    return {"U": U, "V": V, "W": W}
