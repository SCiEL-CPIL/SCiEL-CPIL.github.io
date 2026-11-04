import numpy as np
from functools import reduce
import operator

def _verify_decomposition(U, V, W):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = U.shape[1]
    # Build the decomposition: Z = sum_r W_r * (U_r^T X) * (V_r^T Y)
    # But we need to check against the actual multiplication tensor.
    # The tensor T_{I,J,K} = sum_r U[I,r] * V[J,r] * W[K,r]
    # For matrix multiplication, T_{I,J,K} should be 1 if I and J correspond to
    # a valid multiplication step that produces K, and 0 otherwise.
    n = 6
    # Build the target tensor
    # I = i*6+j, J = j*6+k, K = k*6+i (based on W[k*n+i])
    # Actually, let's think about this more carefully.
    # X is 6x6, Y is 6x6, Z = X*Y
    # Z[i,k] = sum_j X[i,j] * Y[j,k]
    # I = i*6+j (flattened X)
    # J = j*6+k (flattened Y)
    # K = k*6+i (flattened Z, but with the trace convention W[k*n+i])
    # So T[I,J,K] = 1 if the indices match a valid multiplication term
    
    # Compute the decomposition tensor
    # T_decomp[I,J,K] = sum_r U[I,r] * V[J,r] * W[K,r]
    # We can compute this as: T_decomp = U @ V^T @ W^T ? No.
    # T_decomp[I,J,K] = sum_r U[I,r] * V[J,r] * W[K,r]
    # This is a 3D tensor. Let's build it by iterating over r.
    
    n = 6
    T_decomp = np.zeros((n*n, n*n, n*n), dtype=np.float64)
    for r in range(rank):
        T_decomp += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    
    # Build the target tensor
    T_target = np.zeros((n*n, n*n, n*n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                I = i * n + j
                J = j * n + k
                K = k * n + i  # W[k*n+i]
                T_target[I, J, K] = 1.0
    
    # Check if they match
    diff = np.max(np.abs(T_decomp - T_target))
    return diff

def construct():
    """
    Construct a rank-189 decomposition for 6x6 matrix multiplication
    using the Kronecker product of Strassen's 2x2 algorithm (rank 7)
    and schoolbook 3x3 algorithm (rank 27).
    
    Returns:
        dict with keys 'U', 'V', 'W' - factor matrices of shapes (36, 189), (36, 189), (36, 189)
    """
    n = 6
    
    # Strassen's 2x2 decomposition (rank 7)
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = A*B, C = [[C00, C01], [C10, C11]]
    # 
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # 
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U_S: 4x7 (blocks 00, 01, 10, 11)
    U_S = np.array([
        [1, 0, 1, 0, 1, 1, 0],  # A00
        [0, 0, 1, 0, 1, 0, -1], # A01
        [0, 1, 0, 0, 0, -1, 0], # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ], dtype=np.float64)
    
    # V_S: 4x7
    V_S = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [1, 0, 1, 0, 0, 1, 1],   # B01
        [0, 0, -1, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ], dtype=np.float64)
    
    # W_S: 4x7
    W_S = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ], dtype=np.float64)
    
    # Schoolbook 3x3 decomposition (rank 27)
    # For each (i,j,k) in 0..2:
    # U_33[i*3+j, i*9+j*3+k] = 1
    # V_33[j*3+k, i*9+j*3+k] = 1
    # W_33[k*3+i, i*9+j*3+k] = 1
    U_33 = np.zeros((9, 27), dtype=np.float64)
    V_33 = np.zeros((9, 27), dtype=np.float64)
    W_33 = np.zeros((9, 27), dtype=np.float64)
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r = i * 9 + j * 3 + k
                I = i * 3 + j
                J = j * 3 + k
                K = k * 3 + i
                U_33[I, r] = 1.0
                V_33[J, r] = 1.0
                W_33[K, r] = 1.0
    
    # Kronecker product: U_kron = kron(U_S, U_33)
    # This gives a 36x189 matrix
    U_kron = np.kron(U_S, U_33)
    V_kron = np.kron(V_S, V_33)
    W_kron = np.kron(W_S, W_33)
    
    # Now we need to permute the rows to match the standard indexing.
    # U_kron has rows indexed by (block, local) where block is 0..3 and local is 0..8.
    # We need to map to global index I = i*6+j where i,j are 0..5.
    # Block b = bi*2 + bj where bi is block row (0 or 1) and bj is block col (0 or 1)
    # Local index l = li*3 + lj where li, lj are 0..2
    # Global: i = bi*3 + li, j = bj*3 + lj
    # So global I = i*6 + j = (bi*3+li)*6 + (bj*3+lj)
    
    # Build the permutation for U and V (same structure)
    # Current row index in kron: block * 9 + local
    # block = bi*2 + bj, local = li*3 + lj
    # We want to reorder so that row g = (bi*3+li)*6 + (bj*3+lj)
    
    perm = np.zeros(36, dtype=np.int32)
    for bi in range(2):
        for bj in range(2):
            for li in range(3):
                for lj in range(3):
                    block = bi * 2 + bj
                    local = li * 3 + lj
                    kron_idx = block * 9 + local
                    g = (bi * 3 + li) * 6 + (bj * 3 + lj)
                    perm[g] = kron_idx
    
    U_final = U_kron[perm, :]
    V_final = V_kron[perm, :]
    
    # For W, the indexing is different.
    # W[k*n+i] means the row index is k*6 + i.
    # In the kron product, W_kron has rows indexed by (block_c, local_w).
    # block_c corresponds to the C block: C00, C01, C10, C11
    # C_block = ci*2 + ck where ci is the C row block and ck is the C col block
    # local_w = lwk*3 + lwi where lwk is local k and lwi is local i
    # Global: i = ci*3 + lwi, k = ck*3 + lwk
    # W row index = k*6 + i = (ck*3+lwk)*6 + (ci*3+lwi)
    
    perm_w = np.zeros(36, dtype=np.int32)
    for ci in range(2):
        for ck in range(2):
            for lwi in range(3):
                for lwk in range(3):
                    block_c = ci * 2 + ck
                    local_w = lwk * 3 + lwi
                    kron_idx = block_c * 9 + local_w
                    g = (ck * 3 + lwk) * 6 + (ci * 3 + lwi)
                    perm_w[g] = kron_idx
    
    W_final = W_kron[perm_w, :]
    
    # Verify the decomposition
    diff = _verify_decomposition(U_final, V_final, W_final)
    
    return {"U": U_final, "V": V_final, "W": W_final}
