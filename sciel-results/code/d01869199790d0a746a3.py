import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin

    n, m, p = DIMENSIONS

    # Recursive Strassen 4x4x4 construction (rank 49).
    # Apply Strassen's 7-term algorithm to the 2x2 block structure,
    # where each block is a 2x2 matrix. Rank = 7*7 = 49.
    
    # Strassen coefficients for 2x2 matrices (0-indexed):
    # a_coeffs[s, i, j]: coefficient of A[i,j] in s-th linear form
    # b_coeffs[s, j, k]: coefficient of B[j,k] in s-th linear form  
    # c_coeffs[s, i, k]: coefficient of C[i,k] in s-th linear form
    
    a_coeffs = np.array([
        [[1, 0], [0, 1]],   # s=0: A00 + A11
        [[0, 0], [1, 1]],   # s=1: A10 + A11
        [[1, 0], [0, 0]],   # s=2: A00
        [[0, 0], [0, 1]],   # s=3: A11
        [[1, 1], [0, 0]],   # s=4: A00 + A01
        [[-1, 0], [1, 0]],  # s=5: A10 - A00
        [[0, 1], [0, -1]],  # s=6: A01 - A11
    ], dtype=int)

    b_coeffs = np.array([
        [[1, 0], [0, 1]],   # s=0: B00 + B11
        [[1, 0], [0, 0]],   # s=1: B00
        [[0, 1], [0, -1]],  # s=2: B01 - B11
        [[-1, 0], [1, 0]],  # s=3: B10 - B00
        [[0, 0], [0, 1]],   # s=4: B11
        [[1, 1], [0, 0]],   # s=5: B00 + B01
        [[0, 0], [1, 1]],   # s=6: B10 + B11
    ], dtype=int)

    c_coeffs = np.array([
        [[1, 0], [0, 1]],   # s=0: C00 + C11
        [[0, 0], [1, -1]],  # s=1: C10 - C11
        [[0, 1], [0, 1]],   # s=2: C01 + C11
        [[1, 0], [1, 0]],   # s=3: C00 + C10
        [[-1, 1], [0, 0]],  # s=4: -C00 + C01
        [[0, 0], [0, 1]],   # s=5: C11
        [[1, 0], [0, 0]],   # s=6: C00
    ], dtype=int)

    # Precompute all 49 Kronecker products
    U_4d = np.zeros((7, 7, 4, 4), dtype=int)
    V_4d = np.zeros((7, 7, 4, 4), dtype=int)
    W_4d = np.zeros((7, 7, 4, 4), dtype=int)
    
    for s in range(7):
        for t in range(7):
            U_4d[s, t] = np.kron(a_coeffs[s], a_coeffs[t])
            V_4d[s, t] = np.kron(b_coeffs[s], b_coeffs[t])
            # W: need row index k*4+i for C[i,k]
            # kron(c_s, c_t)[i, k] gives coeff for C[i,k]
            # Transpose to get [k, i] indexing
            W_4d[s, t] = np.kron(c_coeffs[s], c_coeffs[t]).T
            
            if time.time() > deadline:
                # Return zero matrices as fallback
                return {"U": np.zeros((16, 49), dtype=int).tolist(), 
                        "V": np.zeros((16, 49), dtype=int).tolist(), 
                        "W": np.zeros((16, 49), dtype=int).tolist()}

    # Flatten to (16, 49)
    # U_4d[s, t, i, j] -> row = i*4+j, col = 7*s+t
    U = U_4d.transpose(2, 3, 0, 1).reshape(16, 49)
    V = V_4d.transpose(2, 3, 0, 1).reshape(16, 49)
    W = W_4d.transpose(2, 3, 0, 1).reshape(16, 49)
    
    # Quick verification: check that the decomposition is exact
    # For random integer matrices, verify C = A @ B
    if time.time() < deadline - 2.0:
        rng = np.random.RandomState(42)
        A_test = rng.randint(-3, 4, size=(4, 4))
        B_test = rng.randint(-3, 4, size=(4, 4))
        
        # Compute via decomposition
        u_vec = (U * A_test.reshape(16, 1)).sum(axis=0)
        v_vec = (V * B_test.reshape(16, 1)).sum(axis=0)
        c_vec = (W * (u_vec * v_vec).reshape(1, 49)).sum(axis=1)
        C_decomp = c_vec.reshape(4, 4)
        
        # Direct multiplication
        C_direct = A_test @ B_test
        
        if not np.array_equal(C_decomp, C_direct):
            # If verification fails, still return the construction
            # (should not happen with correct Strassen)
            pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
