<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140  # 10s safety margin before 150s kill

def _time_left():
    return _DEADLINE - time.time()

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15

    if _time_left() < 5:
        # Emergency: return a trivial decomposition
        rank = n * m * p  # 60
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # We use a block decomposition.
    # 7 pairs of columns (0-1, 2-3, ..., 12-13) using Strassen (rank 7 each)
    # 1 single column (14) using trivial (rank 4)
    # Total rank = 7 * 7 + 4 = 53

    rank = 53
    U = [[0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * rank for _ in range(p * n)]  # 30 x 53

    # Helper to set Strassen block for column pair (2k, 2k+1)
    # A = [[a, b], [c, d]] (indices 0,1,2,3 in U)
    # B columns k0, k1: B = [[b0k0, b0k1], [b1k0, b1k1]]
    # V indices: row0 = B[0, col], row1 = B[1, col]
    #   So V[col] = B[0, col], V[p+col] = B[1, col]
    # W indices: W[k*n + r] for C[r, k]
    #   C is 2x15, so C[r, k] -> W[k*2 + r]
    #   C[0, k] -> W[2k], C[1, k] -> W[2k+1]

    # Strassen for 2x2 * 2x2:
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    # where:
    # M1 = (a + d)(b0k0 + b1k1)
    # M2 = (c + d) * b0k0
    # M3 = a * (b0k1 - b1k1)
    # M4 = d * (b1k0 - b0k0)
    # M5 = (a + b) * b1k1
    # M6 = (c - a) * (b0k0 + b0k1)
    # M7 = (b - d) * (b1k0 + b1k1)

    def add_strassen_block(base_r, k0, k1):
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        r5 = base_r + 4
        r6 = base_r + 5
        r7 = base_r + 6

        # M1 = (a + d)(B[0,k0] + B[1,k1])
        U[0][r1] = 1
        U[3][r1] = 1
        V[k0][r1] = 1
        V[p + k1][r1] = 1

        # M2 = (c + d) * B[0,k0]
        U[2][r2] = 1
        U[3][r2] = 1
        V[k0][r2] = 1

        # M3 = a * (B[0,k1] - B[1,k1])
        U[0][r3] = 1
        V[k1][r3] = 1
        V[p + k1][r3] = -1

        # M4 = d * (B[1,k0] - B[0,k0])
        U[3][r4] = 1
        V[p + k0][r4] = 1
        V[k0][r4] = -1

        # M5 = (a + b) * B[1,k1]
        U[0][r5] = 1
        U[1][r5] = 1
        V[p + k1][r5] = 1

        # M6 = (c - a) * (B[0,k0] + B[0,k1])
        U[2][r6] = 1
        U[0][r6] = -1
        V[k0][r6] = 1
        V[k1][r6] = 1

        # M7 = (b - d) * (B[1,k0] + B[1,k1])
        U[1][r7] = 1
        U[3][r7] = -1
        V[p + k0][r7] = 1
        V[p + k1][r7] = 1

        # C[0, k0] = M1 + M4 - M5 + M7
        W[2 * k0][r1] = 1
        W[2 * k0][r4] = 1
        W[2 * k0][r5] = -1
        W[2 * k0][r7] = 1

        # C[1, k0] = M2 + M4
        W[2 * k0 + 1][r2] = 1
        W[2 * k0 + 1][r4] = 1

        # C[0, k1] = M3 + M5
        W[2 * k1][r3] = 1
        W[2 * k1][r5] = 1

        # C[1, k1] = M1 - M2 + M3 + M6
        W[2 * k1 + 1][r1] = 1
        W[2 * k1 + 1][r2] = -1
        W[2 * k1 + 1][r3] = 1
        W[2 * k1 + 1][r6] = 1

    for i in range(7):
        k0 = 2 * i
        k1 = 2 * i + 1
        base_r = 7 * i
        add_strassen_block(base_r, k0, k1)

    # Trivial block for column 14
    # C[0, 14] = a*B[0,14] + b*B[1,14]
    # C[1, 14] = c*B[0,14] + d*B[1,14]
    base_r = 49
    r1 = base_r
    r2 = base_r + 1
    r3 = base_r + 2
    r4 = base_r + 3

    # a * B[0,14]
    U[0][r1] = 1
    V[14][r1] = 1
    W[28][r1] = 1

    # b * B[1,14]
    U[1][r2] = 1
    V[p + 14][r2] = 1
    W[28][r2] = 1

    # c * B[0,14]
    U[2][r3] = 1
    V[14][r3] = 1
    W[29][r3] = 1

    # d * B[1,14]
    U[3][r4] = 1
    V[p + 14][r4] = 1
    W[29][r4] = 1

    # Verify correctness using numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check tensor decomposition: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # T[i,j,k] should be 1 iff the (i,j) pair contributes to output k
    # A flattened: 0=a, 1=b, 2=c, 3=d
    # B flattened: 0=B[0,0], 1=B[1,0], ..., 2k=B[0,k], 2k+1=B[1,k], ...
    # C flattened: 0=C[0,0], 1=C[1,0], ..., 2k=C[0,k], 2k+1=C[1,k], ...
    
    # Build expected tensor
    T_expected = np.zeros((n*m, m*p, p*n))
    for k in range(p):
        # C[0,k] = a*B[0,k] + b*B[1,k]
        T_expected[0, k, 2*k] = 1      # a * B[0,k] -> C[0,k]
        T_expected[1, p+k, 2*k] = 1    # b * B[1,k] -> C[0,k]
        # C[1,k] = c*B[0,k] + d*B[1,k]
        T_expected[2, k, 2*k+1] = 1    # c * B[0,k] -> C[1,k]
        T_expected[3, p+k, 2*k+1] = 1  # d * B[1,k] -> C[1,k]
    
    # Compute actual tensor from decomposition
    # T_actual[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    T_actual = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
    
    max_err = np.max(np.abs(T_actual - T_expected))
    if max_err > 1e-10:
        # Fallback to trivial decomposition
        rank_triv = n * m * p
        U_t = np.zeros((n*m, rank_triv))
        V_t = np.zeros((m*p, rank_triv))
        W_t = np.zeros((p*n, rank_triv))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U_t[i*m+j, r] = 1
                    V_t[j*p+k, r] = 1
                    W_t[k*n+i, r] = 1
        return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140  # 10s safety margin before 150s kill

def _time_left():
    return _DEADLINE - time.time()

def _build_expected_tensor(n, m, p):
    """Build the expected matrix multiplication tensor."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    return T

def _verify_decomposition(U, V, W, T_expected, tol=1e-10):
    """Verify tensor decomposition correctness."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    T_actual = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
    return np.max(np.abs(T_actual - T_expected)) <= tol

def _trivial_decomposition(n, m, p):
    """Return the trivial (schoolbook) decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i*m*p + j*p + k
                U[i*m+j, r] = 1
                V[j*p+k, r] = 1
                W[k*n+i, r] = 1
    return U.tolist(), V.tolist(), W.tolist()

def _strassen_2x2x2_block(U, V, W, base_r, k0, k1, p):
    """Add a Strassen block for 2x2 * 2x2 (columns k0, k1)."""
    r1, r2, r3, r4, r5, r6, r7 = base_r + np.arange(7)
    
    # M1 = (a + d)(B[0,k0] + B[1,k1])
    U[0, r1] = 1; U[3, r1] = 1
    V[k0, r1] = 1; V[p + k1, r1] = 1
    # M2 = (c + d) * B[0,k0]
    U[2, r2] = 1; U[3, r2] = 1
    V[k0, r2] = 1
    # M3 = a * (B[0,k1] - B[1,k1])
    U[0, r3] = 1
    V[k1, r3] = 1; V[p + k1, r3] = -1
    # M4 = d * (B[1,k0] - B[0,k0])
    U[3, r4] = 1
    V[p + k0, r4] = 1; V[k0, r4] = -1
    # M5 = (a + b) * B[1,k1]
    U[0, r5] = 1; U[1, r5] = 1
    V[p + k1, r5] = 1
    # M6 = (c - a) * (B[0,k0] + B[0,k1])
    U[2, r6] = 1; U[0, r6] = -1
    V[k0, r6] = 1; V[k1, r6] = 1
    # M7 = (b - d) * (B[1,k0] + B[1,k1])
    U[1, r7] = 1; U[3, r7] = -1
    V[p + k0, r7] = 1; V[p + k1, r7] = 1
    
    # C[0, k0] = M1 + M4 - M5 + M7
    W[2*k0, r1] = 1; W[2*k0, r4] = 1; W[2*k0, r5] = -1; W[2*k0, r7] = 1
    # C[1, k0] = M2 + M4
    W[2*k0+1, r2] = 1; W[2*k0+1, r4] = 1
    # C[0, k1] = M3 + M5
    W[2*k1, r3] = 1; W[2*k1, r5] = 1
    # C[1, k1] = M1 - M2 + M3 + M6
    W[2*k1+1, r1] = 1; W[2*k1+1, r2] = -1; W[2*k1+1, r3] = 1; W[2*k1+1, r6] = 1

def _schoolbook_2x2x1_block(U, V, W, base_r, k, p):
    """Add a schoolbook block for 2x2 * 2x1 (column k)."""
    r1, r2, r3, r4 = base_r + np.arange(4)
    U[0, r1] = 1; V[k, r1] = 1; W[2*k, r1] = 1
    U[1, r2] = 1; V[p+k, r2] = 1; W[2*k, r2] = 1
    U[2, r3] = 1; V[k, r3] = 1; W[2*k+1, r3] = 1
    U[3, r4] = 1; V[p+k, r4] = 1; W[2*k+1, r4] = 1

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    T_expected = _build_expected_tensor(n, m, p)

    if _time_left() < 5:
        U, V, W = _trivial_decomposition(n, m, p)
        return {"U": U, "V": V, "W": W}

    # Strategy: Try different block decompositions and pick the best valid one.
    # Option 1: 7 Strassen blocks (2x2x2, rank 7) + 1 schoolbook (2x2x1, rank 4) = 53
    # Option 2: 5 blocks of 2x2x3 (try to find low-rank decomp)
    # Option 3: 3 blocks of 2x2x5 (try to find low-rank decomp)
    
    best_result = None
    best_rank = n * m * p  # 60 fallback
    
    # Option 1: Current approach - rank 53
    rank1 = 7 * 7 + 4  # 53
    U1 = np.zeros((n*m, rank1))
    V1 = np.zeros((m*p, rank1))
    W1 = np.zeros((p*n, rank1))
    for i in range(7):
        _strassen_2x2x2_block(U1, V1, W1, 7*i, 2*i, 2*i+1, p)
    _schoolbook_2x2x1_block(U1, V1, W1, 49, 14, p)
    
    if _verify_decomposition(U1.tolist(), V1.tolist(), W1.tolist(), T_expected):
        best_result = (U1.tolist(), V1.tolist(), W1.tolist())
        best_rank = rank1
    
    # Option 2: Try to find a rank-9 decomposition for 2x2x3
    # If we can decompose 2x2x3 with rank 9, then 5 blocks = rank 45
    # Let's try to find it computationally
    if _time_left() > 30 and best_rank > 45:
        # Build the 2x2x3 tensor
        p3 = 3
        T_223 = np.zeros((4, 6, 6))
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    T_223[i*2+j, j*3+k, k*2+i] = 1
        
        # Try rank 9 decomposition using iterative refinement
        # Initialize with random matrices and use ALS-style optimization
        found_223 = False
        for trial in range(5):
            if _time_left() < 15:
                break
            np.random.seed(42 + trial)
            r = 9
            U_223 = np.random.randn(4, r)
            V_223 = np.random.randn(6, r)
            W_223 = np.random.randn(6, r)
            
            # Normalize to avoid scale issues
            U_223 /= np.linalg.norm(U_223, axis=0, keepdims=True)
            V_223 /= np.linalg.norm(V_223, axis=0, keepdims=True)
            W_223 /= np.linalg.norm(W_223, axis=0, keepdims=True)
            
            # Iterative optimization: minimize ||T - sum_r U_r ⊗ V_r ⊗ W_r||
            for it in range(200):
                if _time_left() < 10:
                    break
                # Update U
                for i in range(4):
                    # T[i,:,:] = sum_r U[i,r] * V_r ⊗ W_r
                    # Solve for U[i,:] given V, W
                    # T[i,:,:] = U[i,:] @ (V ⊙ W)^T where ⊙ is Khatri-Rao
                    VW = np.kron(V_223, W_223)  # 36 x r
                    Ti = T_223[i].reshape(-1)
                    U_223[i] = np.linalg.lstsq(VW, Ti, rcond=None)[0]
                
                # Update V
                for j in range(6):
                    UW = np.kron(U_223, W_223)
                    Tj = T_223[:, j, :].reshape(-1)
                    V_223[j] = np.linalg.lstsq(UW, Tj, rcond=None)[0]
                
                # Update W
                for k in range(6):
                    UV = np.kron(U_223, V_223)
                    Tk = T_223[:, :, k].reshape(-1)
                    W_223[k] = np.linalg.lstsq(UV, Tk, rcond=None)[0]
            
            # Check if we got an exact decomposition
            T_actual = np.einsum('ir,jr,kr->ijk', U_223, V_223, W_223)
            err = np.max(np.abs(T_actual - T_223))
            if err < 1e-8:
                # Round to nearest integers/half-integers
                U_223 = np.round(U_223 * 2) / 2
                V_223 = np.round(V_223 * 2) / 2
                W_223 = np.round(W_223 * 2) / 2
                T_actual = np.einsum('ir,jr,kr->ijk', U_223, V_223, W_223)
                if np.max(np.abs(T_actual - T_223)) < 1e-8:
                    found_223 = True
                    break
        
        if found_223:
            rank2 = 5 * 9  # 45
            U2 = np.zeros((n*m, rank2))
            V2 = np.zeros((m*p, rank2))
            W2 = np.zeros((p*n, rank2))
            for block_i in range(5):
                base_r = 9 * block_i
                # Copy 2x2x3 block to columns 3*block_i to 3*block_i+2
                for r in range(9):
                    U2[:, base_r + r] = U_223[:, r]
                    # V: column j in 2x2x3 maps to column 3*block_i + j in full
                    for j in range(6):
                        V2[j + 3*block_i*2, base_r + r] = V_223[j, r]
                    # Wait, V is indexed by (j, k) where j is row of B, k is col
                    # In 2x2x3: V index is j*3+k for j in {0,1}, k in {0,1,2}
                    # In full: V index is j*p+k for j in {0,1}, k in {0,...,14}
                    # So for block starting at column 3*block_i:
                    #   V_223[j, r] -> V2[j*15 + 3*block_i + k, base_r + r] where j*3+k = index in V_223
                    pass
                # Let me redo this mapping
                # V_223 has shape (6, 9), indexed by (j*3+k, r) for j in {0,1}, k in {0,1,2}
                # V2 has shape (30, 45), indexed by (j*15+k, r) for j in {0,1}, k in {0,...,14}
                # For block_i, columns are 3*block_i, 3*block_i+1, 3*block_i+2
                for j in range(2):
                    for k in range(3):
                        v_idx = j*3 + k  # index in V_223
                        v2_idx = j*15 + 3*block_i + k  # index in V2
                        V2[v2_idx, base_r:base_r+9] = V_223[v_idx, :]
                # W_223 has shape (6, 9), indexed by (k*2+i, r) for k in {0,1,2}, i in {0,1}
                # W2 has shape (30, 45), indexed by (k*2+i, r) for k in {0,...,14}, i in {0,1}
                for k in range(3):
                    for i in range(2):
                        w_idx = k*2 + i  # index in W_223
                        w2_idx = (3*block_i + k)*2 + i  # index in W2
                        W2[w2_idx, base_r:base_r+9] = W_223[w_idx, :]
            
            if _verify_decomposition(U2.tolist(), V2.tolist(), W2.tolist(), T_expected):
                best_result = (U2.tolist(), V2.tolist(), W2.tolist())
                best_rank = rank2
    
    if best_result is not None:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    
    # Fallback
    U, V, W = _trivial_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I made an error in the V mapping. Let me fix that. The V_223 tensor is indexed by (j, k) where j is the row of B (0 or 1) and k is the column within the block (0, 1, or 2). The flattened index is j*3+k. In the full problem, V is indexed by (j, k_full) where k_full is 0..14, flattened as j*15+k_full.

Let me rewrite this more carefully:

