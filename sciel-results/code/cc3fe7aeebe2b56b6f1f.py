import numpy as np
import time

def construct():
    # Deadline guard
    t_start = time.time()
    deadline = 120.0
    
    # Strassen 2x2 decomposition (Rank 7)
    # Indices for U: (i, j) where i is row of A, j is col of A
    # Indices for V: (j, k) where j is row of B, k is col of B
    # Indices for W: (k, i) where k is row of C (col of B), i is col of C (row of A)
    # Convention: W[k*n+i, r]
    
    # S_U, S_V, S_W are 4x7 matrices for 2x2 case
    # Linear index for 2x2: idx = x*2 + y
    S_U = np.array([
        [1, 0, 1, 0, 1, -1, 0], # (0,0)
        [0, 0, 0, 0, 1, 0, 1],  # (0,1)
        [0, 1, 0, 0, 0, 1, 0],  # (1,0)
        [1, 1, 0, 1, 0, 0, -1]  # (1,1)
    ], dtype=float)
    
    S_V = np.array([
        [1, 1, 0, -1, 0, 1, 0], # (0,0)
        [0, 0, 1, 0, 0, 1, 0],  # (0,1)
        [0, 0, 0, 1, 0, 0, 1],  # (1,0)
        [1, 0, -1, 0, 1, 0, 1]  # (1,1)
    ], dtype=float)
    
    S_W = np.array([
        [1, 0, 0, 1, -1, 0, 1], # (0,0) -> k=0, i=0
        [0, 1, 0, 1, 0, 0, 0],  # (0,1) -> k=0, i=1
        [0, 0, 1, 0, 1, 0, 0],  # (1,0) -> k=1, i=0
        [1, -1, 1, 0, 0, 1, 0]  # (1,1) -> k=1, i=1
    ], dtype=float)
    
    # Verify 2x2 Strassen
    # T2[i,j,k] = sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
    # Should be 1 for all i,j,k in 0..1
    T2_check = np.einsum('ir,jr,kr->ijk', S_U.reshape(2,2,7), S_V.reshape(2,2,7), S_W.reshape(2,2,7))
    # Note: Reshape order matters.
    # S_U is 4x7. Row 0 is (0,0), Row 1 is (0,1), Row 2 is (1,0), Row 3 is (1,1).
    # Reshape(2,2,7) maps [i, j, r].
    if not np.all(T2_check == 1):
        # Fallback if basic Strassen is broken (should not happen)
        return _schoolbook_4x4()

    # Build 4x4 via Kronecker Product
    # U4_raw = kron(S_U, S_U)
    # Rows of kron are (i_o, j_o, i_i, j_i) flattened as (i_o*2+j_o)*4 + (i_i*2+j_i)
    # We want rows to be (i, j) flattened as i*4+j where i = i_o*2+i_i, j = j_o*2+j_i
    # So target index = (i_o*2+i_i)*4 + (j_o*2+j_i) = i_o*8 + i_i*4 + j_o*2 + j_i
    
    U4_raw = np.kron(S_U, S_U)
    V4_raw = np.kron(S_V, S_V)
    W4_raw = np.kron(S_W, S_W)
    
    # Generate permutations
    # For U: linear index l = i*4+j. i = l//4, j = l%4.
    # i = i_o*2 + i_i => i_o = i//2, i_i = i%2
    # j = j_o*2 + j_i => j_o = j//2, j_i = j%2
    # kron_index = (i_o*2 + j_o)*4 + (i_i*2 + j_i)
    
    def get_perm(n_dim):
        perm = []
        for lin in range(n_dim * n_dim):
            i = lin // n_dim
            j = lin % n_dim
            i_o, i_i = i // 2, i % 2
            j_o, j_i = j // 2, j % 2
            kron_idx = (i_o * 2 + j_o) * 4 + (i_i * 2 + j_i)
            perm.append(kron_idx)
        return np.array(perm, dtype=int)
        
    # Note: The permutation for V and W might be different depending on how indices map.
    # V indices: j, k. j is "row of B" (which is "col of A" in the first slot of U).
    # In U, the second index is j. In V, the first index is j.
    # The structure is symmetric for the "middle" index j in the tensor contraction.
    # However, the linearization in U is i*4+j. In V is j*4+k.
    # The Kronecker structure for V is kron(S_V, S_V).
    # Row of S_V is (j, k).
    # Kron row is (j_o, k_o, j_i, k_i).
    # Target linear for V is j*4+k.
    # j = j_o*2 + j_i, k = k_o*2 + k_i.
    # Target = (j_o*2+j_i)*4 + (k_o*2+k_i) = j_o*8 + j_i*4 + k_o*2 + k_i.
    # Kron index from raw (j_o*2+k_o)*4 + (j_i*2+k_i).
    # This is the same mapping as U if we map (i,j) to (j,k).
    # So the permutation array is the same for U, V, and W?
    # Let's check W.
    # W indices: k, i.
    # Row of S_W is (k, i).
    # Kron row is (k_o, i_o, k_i, i_i).
    # Target linear for W is k*4+i.
    # k = k_o*2 + k_i, i = i_o*2 + i_i.
    # Target = (k_o*2+k_i)*4 + (i_o*2+i_i) = k_o*8 + k_i*4 + i_o*2 + i_i.
    # Kron index = (k_o*2+i_o)*4 + (k_i*2+i_i).
    # Yes, the permutation mapping from (out, in) pair to linear index is identical in structure.
    # So one permutation array works for all if we define it as mapping (x, y) to linear.
    
    perm_idx = get_perm(4)
    
    U4_fixed = U4_raw[perm_idx, :]
    V4_fixed = V4_raw[perm_idx, :]
    W4_fixed = W4_raw[perm_idx, :]
    
    # Verify 4x4
    if time.time() - t_start < deadline - 5:
        # Reshape to 3D for einsum verification
        # U4_fixed is 16x49. Reshape to (4, 4, 49) where first dim is i, second is j.
        U_3d = U4_fixed.reshape(4, 4, 49)
        V_3d = V4_fixed.reshape(4, 4, 49)
        W_3d = W4_fixed.reshape(4, 4, 49)
        
        # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        # Note: W is indexed by (k, i) in the row.
        # Our W_3d is reshaped from row k*4+i. So W_3d[k, i, r] is correct.
        
        T_4 = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        if not np.all(T_4 == 1):
            return _schoolbook_4x4()
            
    return {
        "U": U4_fixed.tolist(),
        "V": V4_fixed.tolist(),
        "W": W4_fixed.tolist()
    }

def _schoolbook_4x4():
    n, m, p = 4, 4, 4
    rank = 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
