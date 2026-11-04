import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 115.0

def construct():
    n, m, p = DIMENSIONS
    
    # 2x2 Strassen decomposition factors
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    # Kronecker product for 4x4
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Permutation to map Kronecker row order to standard row-major order
    # Standard: index = i*4 + j, where i = 2*i1+i2, j = 2*j1+j2
    # Kronecker: index = r1*4 + r2, where r1 = i1*2+j1, r2 = i2*2+j2
    perm = np.zeros(16, dtype=int)
    for S in range(16):
        R = S // 4
        C = S % 4
        r_in = R % 2
        r_out = R // 2
        c_in = C % 2
        c_out = C // 2
        K_in = r_in * 2 + c_in
        K_out = r_out * 2 + c_out
        K = K_out * 4 + K_in
        perm[S] = K
    
    U4_fixed = U4[perm, :]
    V4_fixed = V4[perm, :]
    W4_fixed = W4[perm, :]
    
    # Verify correctness
    U_arr = np.array(U4_fixed, dtype=float)
    V_arr = np.array(V4_fixed, dtype=float)
    W_arr = np.array(W4_fixed, dtype=float)
    
    # Build target tensor
    T = np.zeros((16, 16, 16), dtype=float)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                T[i*4+j, j*4+k, k*4+i] = 1.0
    
    T_decomp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    max_diff = np.max(np.abs(T_decomp - T))
    
    if max_diff > 1e-9:
        # Fallback to schoolbook if verification fails
        rank = 64
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        column = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i * 4 + j][column] = 1
                    V[j * 4 + k][column] = 1
                    W[k * 4 + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {
        "U": U4_fixed.tolist(),
        "V": V4_fixed.tolist(),
        "W": W4_fixed.tolist()
    }
# EVOLVE-BLOCK-END
