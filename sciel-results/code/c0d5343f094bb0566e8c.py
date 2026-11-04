"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen's 2x2 decomposition coefficients
    # U, V, W are 4x7 matrices (flattened 2x2 indices 0..3)
    # Indices: 00, 01, 10, 11
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0], # r=1..7 for U
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    
    V2 = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ])
    
    W2 = np.array([
        [1, 0, 0, -1, 0, -1, 1],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 1, -1, 0, -1, 0, 0]
    ])
    
    # Verify Strassen 2x2 correctness mentally?
    # C00 = W2[0] dot P = 1*M1 - 1*M4 - 1*M6 + 1*M7. Correct.
    # C01 = W2[1] dot P = 1*M3 + 1*M7. Correct.
    # C10 = W2[2] dot P = 1*M2 + 1*M4. Correct.
    # C11 = W2[3] dot P = 1*M1 - 1*M3 - 1*M5 + 1*M2. Correct.
    
    # Construct 4x4 decomposition via Kronecker product
    # Rank = 7 * 7 = 49
    rank = 49
    
    U4 = np.zeros((16, rank))
    V4 = np.zeros((16, rank))
    W4 = np.zeros((16, rank))
    
    col = 0
    for r in range(7):
        for s in range(7):
            # Kronecker product of column r and column s
            # np.kron(u_r, u_s)
            u_kron = np.kron(U2[:, r], U2[:, s])
            v_kron = np.kron(V2[:, r], V2[:, s])
            w_kron = np.kron(W2[:, r], W2[:, s])
            
            U4[:, col] = u_kron
            V4[:, col] = v_kron
            W4[:, col] = w_kron
            col += 1
            
    # Convert to lists of integers/half-integers as required
    # The problem states "integer/half-integer entries".
    # Since inputs are integers, outputs are integers.
    
    U_list = [[int(x) for x in row] for row in U4.tolist()]
    V_list = [[int(x) for x in row] for row in V4.tolist()]
    W_list = [[int(x) for x in row] for row in W4.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
