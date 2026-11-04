import numpy as np
import time

def construct():
    start_time = time.time()
    
    # Strassen 2x2 matrices (rank-7 decomposition)
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
    
    # Kronecker product gives rank-49 decomposition for 4x4
    U4 = np.kron(U2, U2)  # shape (16, 49)
    V4 = np.kron(V2, V2)  # shape (16, 49)
    W4 = np.kron(W2, W2)  # shape (16, 49)
    
    # The Kronecker product uses interleaved index ordering:
    # For 4x4: row index in kron = (r_outer * 2 + r_inner) * 4 + (c_outer * 2 + c_inner)
    # Standard row-major: row index = r * 4 + c
    # We need to find permutation P such that W4_permuted = W4[P, :] matches the tensor convention
    # W[k*n+i, r] where k is column of B, i is row of A (output C[i,k])
    
    # Build the canonical tensor pattern for W:
    # W[k*4+i, r] should be 1 for each (i,j,k) triple at rank position corresponding to (j,k)
    # Actually, let's verify by checking the full tensor identity:
    # sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] = 1 if (i,j,k) valid, else 0
    
    # The kron ordering: for 2x2 blocks, kron(U2,U2) has rows indexed as:
    # row = i1*4 + i2*2 + j1*2 + j2 where i1,i2 are row bits, j1,j2 are col bits
    # Wait, let me think more carefully.
    # np.kron(A,B) where A is 4x7 and B is 4x7 gives 16x49
    # Row index in kron = row_A * 4 + row_B
    # row_A = i1*2 + j1 (for 2x2 A matrix, row = i*2+j)
    # row_B = i2*2 + j2
    # So kron row = (i1*2+j1)*4 + (i2*2+j2) = i1*8 + j1*4 + i2*2 + j2
    
    # Standard 4x4 row-major: row = i*4 + j where i = i1*2+i2, j = j1*2+j2
    # = (i1*2+i2)*4 + (j1*2+j2) = i1*8 + i2*4 + j1*2 + j2
    
    # So we need perm such that kron_row = i1*8 + j1*4 + i2*2 + j2 maps to std_row = i1*8 + i2*4 + j1*2 + j2
    # For each std_row s, find the kron_row k such that they represent the same (i,j)
    
    # Compute permutation: for each standard index s, find corresponding kron index
    perm = np.zeros(16, dtype=int)
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    std_idx = (i1*2+i2)*4 + (j1*2+j2)
                    kron_idx = (i1*2+j1)*4 + (i2*2+j2)
                    perm[std_idx] = kron_idx
    
    # Apply permutation to reorder rows to standard convention
    U4_fixed = U4[perm, :]
    V4_fixed = V4[perm, :]
    W4_fixed = W4[perm, :]
    
    # Verify the decomposition is correct
    # Check: sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] should equal 1 for all i,j,k
    U_np = U4_fixed.astype(float)
    V_np = V4_fixed.astype(float)
    W_np = W4_fixed.astype(float)
    
    error = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = np.dot(np.dot(U_np[i*4+j], V_np[j*4+k]), W_np[k*4+i])
                if abs(val - 1.0) > 1e-9:
                    error += abs(val - 1.0)
    
    if error > 1e-6:
        # Fallback: use schoolbook rank-64 if something went wrong
        rank = 64
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        for i in range(4):
            for j in range(4):
                c = i*4+j
                U[c][c] = 1
                for k in range(4):
                    V[j*4+k][c] = 1
                    W[k*4+i][c] = 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to lists
    U_list = U4_fixed.tolist()
    V_list = V4_fixed.tolist()
    W_list = W4_fixed.tolist()
    
    return {
        "U": U_list,
        "V": V_list,
        "W": W_list
    }
