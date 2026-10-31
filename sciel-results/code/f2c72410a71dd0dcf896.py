import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin

    n, m, p = 4, 4, 4
    rank = 49

    # Precompute the 2x2 Strassen factor matrices as numpy arrays
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # M0 = (A00 + A11) * (B00 + B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # M1 = (A10 + A11) * B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1

    # M2 = A00 * (B01 - B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1

    # M3 = A11 * (B10 - B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1

    # M4 = (A00 + A01) * B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1

    # M5 = (A10 - A00) * (B00 + B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    # M6 = (A01 - A11) * (B10 + B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Build the 49-rank decomposition using Kronecker product of Strassen factors
    # U[i*4+j, s*7+t] = U2[I*2+J, s] * U2[ii*2+jj, t]
    # V[k*4+l, s*7+t] = V2[K*2+L, s] * V2[kk*2+ll, t]
    # W[j*4+i, s*7+t] = W2[I*2+J, s] * W2[ii*2+jj, t]
    # where I=i//2, J=j//2, ii=i%2, jj=j%2 (and similarly for k,l)
    # For W: I=i//2 (row block of C), J=j//2 (col block of C)

    # Precompute the Kronecker product structure using outer products
    # For each s, t, the rank index is r = s*7 + t
    # U[:, r] = (U2[:, s] ⊗ U2[:, t]) reshaped appropriately
    # Actually, let's be more careful with the indexing.
    
    # U flat index: i*4+j where i = 2*I+ii, j = 2*J+jj
    # So U[2*I+ii, 2*J+jj] in 4x4 view = U2[I*2+J, s] * U2[ii*2+jj, t]
    
    # We can precompute for each s, the vector u_s = U2[:, s] (length 4)
    # For each t, the vector v_t = U2[:, t] (length 4)
    # Then U[:, s*7+t] is the Kronecker product of u_s and v_t, but with specific reshaping.
    
    # Let's just use the vectorized approach from inspiration 1 which was clean.
    U_arr = np.zeros((16, 49), dtype=int)
    V_arr = np.zeros((16, 49), dtype=int)
    W_arr = np.zeros((16, 49), dtype=int)
    
    # Precompute index mappings for speed
    I_map = np.array([i // 2 for i in range(4)])
    J_map = np.array([j // 2 for j in range(4)])
    ii_map = np.array([i % 2 for i in range(4)])
    jj_map = np.array([j % 2 for j in range(4)])
    
    # Vectorized construction
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            
            # U: U[i*4+j, r] = U2[I_map[i]*2 + J_map[j], s] * U2[ii_map[i]*2 + jj_map[j], t]
            # We can create a 4x4 matrix for this rank term
            # U_term[i, j] = U2[I_map[i]*2 + J_map[j], s] * U2[ii_map[i]*2 + jj_map[j], t]
            
            # Let's build it using broadcasting
            # Block part: U2[:, s] has shape (4,). We need to map (I, J) -> I*2+J
            # U2_block[i, j] = U2[I_map[i]*2 + J_map[j], s]
            # U2_inner[i, j] = U2[ii_map[i]*2 + jj_map[j], t]
            
            # Create 4x4 matrices
            block_idx = I_map[:, None] * 2 + J_map[None, :]  # (4, 4)
            inner_idx = ii_map[:, None] * 2 + jj_map[None, :]  # (4, 4)
            
            U_term = U2[block_idx, s] * U2[inner_idx, t]
            U_arr[:, r] = U_term.flatten()
            
            V_term = V2[block_idx, s] * V2[inner_idx, t]
            V_arr[:, r] = V_term.flatten()
            
            # For W: W[j*4+i, r] = W2[I_map[i]*2 + J_map[j], s] * W2[ii_map[i]*2 + jj_map[j], t]
            # Note: W row index is j*4+i, so W_arr[j*4+i, r] = W_term[j, i] where W_term is 4x4
            # But W_term[i, j] would be W2[I_map[i]*2 + J_map[j], s] * W2[ii_map[i]*2 + jj_map[j], t]
            # So W_arr[:, r] is W_term transposed and flattened in row-major order of (j, i)?
            # W_arr[j*4+i, r] corresponds to element (j, i) of a 4x4 matrix.
            # If we create W_term[i, j] = ..., then W_arr[:, r] = W_term.T.flatten()
            
            W_term = W2[block_idx, s] * W2[inner_idx, t]
            W_arr[:, r] = W_term.T.flatten()
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Convert to list of lists
    U = U_arr.tolist()
    V = V_arr.tolist()
    W = W_arr.tolist()

    return {"U": U, "V": V, "W": W}
