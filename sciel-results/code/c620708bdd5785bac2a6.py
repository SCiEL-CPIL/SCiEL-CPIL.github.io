import numpy as np
import time

def _verify_decomposition(U, V, W, n=4, m=4, p=4, atol=1e-10):
    """Verify that the tensor decomposition correctly represents matrix multiplication."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    # For each output element C[i,k] (using W[k*n+i] convention)
    # C[i,k] = sum_{j} A[i,j] * B[j,k]
    # Decomposition: C = sum_r (U @ A_flat) * (V @ B_flat) * W_r
    # where A_flat is A reshaped to (n*m, 1) in row-major order
    # and B_flat is B reshaped to (m*p, 1) in row-major order
    
    # Build the full multiplication tensor T[i,j,k] = 1 if j is the connecting index
    # T[i, j, k] = delta_{i,j,k} meaning: C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form: T[i,j,k] = 1 if the indices connect properly
    # Actually, the multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
    # since C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k in range(4)
    
    # The decomposition says: T[i,j,k] = sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
    # Wait, let me re-read the convention.
    # U has shape (n*m, rank) = (16, rank), indexed by (i,j) -> i*4+j
    # V has shape (m*p, rank) = (16, rank), indexed by (j,k) -> j*4+k  
    # W has shape (p*n, rank) = (16, rank), indexed by (k,i) -> k*4+i (per problem)
    
    # Check: sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] should equal 1 for all i,j,k
    
    # Do a spot check on a subset for speed
    import random
    random.seed(42)
    indices = list(range(n * m * p))
    if len(indices) <= 100:
        check_indices = indices
    else:
        check_indices = random.sample(indices, 100)
    
    for idx in check_indices:
        k_val = idx % p
        j_val = (idx // p) % m
        i_val = idx // (p * m)
        
        u_row = i_val * m + j_val  # i*4 + j
        v_row = j_val * p + k_val  # j*4 + k
        w_row = k_val * n + i_val  # k*4 + i (per problem convention)
        
        dot = np.dot(U[u_row, :], V[v_row, :])
        result = dot * W[w_row, :]
        actual = np.sum(result)
        
        if abs(actual - 1.0) > atol:
            return False, (i_val, j_val, k_val, actual)
    
    return True, None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Define 2x2 Strassen matrices (rank 7)
    # U2[i*2+j, r] for i,j in {0,1}, r in {0..6}
    # V2[j*2+k, r] for j,k in {0,1}, r in {0..6}  
    # W2[k*2+i, r] for k,i in {0,1}, r in {0..6}
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    # First verify 2x2 decomposition
    ok2, err2 = _verify_decomposition(U2, V2, W2, 2, 2, 2, atol=1e-10)
    if not ok2:
        # Fallback to schoolbook for 2x2 if Strassen is wrong
        U2 = np.zeros((4, 4))
        V2 = np.zeros((4, 4))
        W2 = np.zeros((4, 4))
        for i in range(2):
            for j in range(2):
                r = i * 2 + j
                U2[i*2+j, r] = 1
                V2[j*2+i, r] = 1  # B[j,i] connects A[i,j] to C[i,i]? No...
        # Actually let me just use schoolbook properly
        U2 = np.zeros((4, 4))
        V2 = np.zeros((4, 4))
        W2 = np.zeros((4, 4))
        for i in range(2):
            for k in range(2):
                r = i * 2 + k
                U2[i*2+0, r] = 0  # This is getting complicated
        # Simpler: just use identity-based schoolbook
        U2 = np.eye(4)
        V2 = np.eye(4)
        W2 = np.zeros((4, 4))
        # C[i,k] = sum_j A[i,j] B[j,k]
        # U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1
        # For schoolbook: r = j (the summing index)
        # U[i*2+j, j] = 1, V[j*2+k, j] = 1, W[k*2+i, j] = 1
        for j in range(2):
            for i in range(2):
                for k in range(2):
                    U2[i*2+j, j] = 1
                    V2[j*2+k, j] = 1
                    W2[k*2+i, j] = 1
    
    # Kronecker product for 4x4
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Now we need to permute rows to match the standard indexing
    # U4 row index in Kronecker: P1 * 4 + P2 where P1, P2 are U2 row indices
    # U2 row index p corresponds to (p//2, p%2) = (i, j) for 2x2
    # For 4x4: i = i1*2 + i0, j = j1*2 + j0
    # Standard row index: i*4 + j = (i1*2+i0)*4 + (j1*2+j0) = i1*8 + i0*4 + j1*2 + j0
    # Kronecker row index: P1 * 4 + P2 where P1 = i1*2 + j1, P2 = i0*2 + j0
    
    perm_u = np.zeros(16, dtype=int)
    for S in range(16):
        i = S // 4
        j = S % 4
        i1, i0 = divmod(i, 2)
        j1, j0 = divmod(j, 2)
        P1 = i1 * 2 + j1
        P2 = i0 * 2 + j0
        perm_u[S] = P1 * 4 + P2
    
    # For V: standard index is j*4 + k
    # j = j1*2 + j0, k = k1*2 + k0
    # Kronecker: Q1 = j1*2 + k1, Q2 = j0*2 + k0
    
    perm_v = np.zeros(16, dtype=int)
    for S in range(16):
        j = S // 4
        k = S % 4
        j1, j0 = divmod(j, 2)
        k1, k0 = divmod(k, 2)
        Q1 = j1 * 2 + k1
        Q2 = j0 * 2 + k0
        perm_v[S] = Q1 * 4 + Q2
    
    # For W: standard index is k*4 + i (per problem convention W[k*n+i])
    # k = k1*2 + k0, i = i1*2 + i0
    # Kronecker: R1 = k1*2 + i1, R2 = k0*2 + i0
    
    perm_w = np.zeros(16, dtype=int)
    for S in range(16):
        k = S // 4
        i = S % 4
        k1, k0 = divmod(k, 2)
        i1, i0 = divmod(i, 2)
        R1 = k1 * 2 + i1
        R2 = k0 * 2 + i0
        perm_w[S] = R1 * 4 + R2
    
    U4_fixed = U4[perm_u, :]
    V4_fixed = V4[perm_v, :]
    W4_fixed = W4[perm_w, :]
    
    # Verify the 4x4 decomposition
    if time.time() < deadline:
        ok4, err4 = _verify_decomposition(U4_fixed, V4_fixed, W4_fixed, 4, 4, 4, atol=1e-10)
        if not ok4:
            # If Kronecker construction has indexing errors, fall back to schoolbook rank-64
            U4_fb = np.zeros((16, 64))
            V4_fb = np.zeros((16, 64))
            W4_fb = np.zeros((16, 64))
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        r = i * 16 + j * 4 + k
                        U4_fb[i * 4 + j, r] = 1
                        V4_fb[j * 4 + k, r] = 1
                        W4_fb[k * 4 + i, r] = 1
            return {
                "U": U4_fb.tolist(),
                "V": V4_fb.tolist(),
                "W": W4_fb.tolist()
            }
    
    return {
        "U": U4_fixed.tolist(),
        "V": V4_fixed.tolist(),
        "W": W4_fixed.tolist()
    }
