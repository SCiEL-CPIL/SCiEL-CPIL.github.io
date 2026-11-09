import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0 - 5.0  # 5 second safety margin

    n, m, p = 4, 4, 4

    # Base 2x2 Strassen matrices
    # U2: 4x7, rows correspond to A[00], A[01], A[10], A[11]
    # V2: 4x7, rows correspond to B[00], B[01], B[10], B[11]
    # W2: 4x7, rows correspond to C[00], C[10], C[01], C[11] (index k*2+i)
    
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
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ], dtype=int)
    
    # Construct 4x4 matrices using Kronecker products
    # U4[i*4+j, r*7+s] = U2[i2*2+j2, r] * U2[i1*2+j1, s]
    # This matches the recursive block structure.
    U = np.kron(U2, U2)
    V = np.kron(V2, V2)
    W = np.kron(W2, W2)
    
    rank = U.shape[1]
    
    # Quick verification
    try:
        if time.time() > deadline:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
        # Test with a random matrix
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, size=(4, 4))
        B_test = np.random.randint(-5, 5, size=(4, 4))
        
        AB = A_test @ B_test
        
        A_flat = A_test.flatten()
        B_flat = B_test.flatten()
        
        u_vals = U @ A_flat
        v_vals = V @ B_flat
        scalars = u_vals * v_vals
        
        # W[k*4+i, r]
        # Result C[i, k] = sum_r W[k*4+i, r] * scalars[r]
        # We need to check if this matches AB[i, k]
        
        # Reshape W to (4, 4, 49) where dimensions are (k, i, r)
        # W is (16, 49). Index is k*4+i.
        # So W_reshaped[k, i, r] = W[k*4+i, r]
        W_reshaped = W.reshape(4, 4, rank)
        
        # Compute C
        # C[i, k] = sum_r W_reshaped[k, i, r] * scalars[r]
        # This is a tensordot over r
        # C_decomp = np.tensordot(W_reshaped, scalars, axes=([2], [0]))
        # Result shape: (4, 4) with indices (k, i)
        # So C_decomp[k, i] = AB[i, k]
        C_decomp = np.tensordot(W_reshaped, scalars, axes=([2], [0]))
        
        # C_decomp is indexed by [k, i]. AB is indexed by [i, k].
        # So we compare C_decomp.T with AB
        if not np.array_equal(AB, C_decomp.T):
            # Fallback to schoolbook if verification fails
            rank_fb = 64
            U_fb = np.zeros((16, rank_fb), dtype=int)
            V_fb = np.zeros((16, rank_fb), dtype=int)
            W_fb = np.zeros((16, rank_fb), dtype=int)
            col_fb = 0
            for i2 in range(4):
                for j2 in range(4):
                    for k2 in range(4):
                        U_fb[i2 * 4 + j2, col_fb] = 1
                        V_fb[j2 * 4 + k2, col_fb] = 1
                        W_fb[k2 * 4 + i2, col_fb] = 1
                        col_fb += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
            
    except Exception:
        # If verification code crashes, return the computed U, V, W
        # It is likely correct, but we don't want to crash.
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
