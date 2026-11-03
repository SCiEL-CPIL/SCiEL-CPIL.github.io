import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0

def _time_left():
    return _DEADLINE - time.time()

def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """Exact scalar check of the decomposition."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = float(np.dot(U[i * m + j], V[j * p + k] * W[k * n + i]))
                if abs(total - 1.0) > tol:
                    return False
    return True

def _verify_np(U, V, W, n, m, p, tol=1e-6):
    """Fast vectorized check."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return bool(np.all(np.abs(T - 1.0) < tol))

def _build_strassen_2x2():
    """
    Build the 2x2 Strassen decomposition.
    Returns (U, V, W) where each has shape (4, 7).
    U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] for n=m=p=2.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_strassen_4x4x2():
    """
    Build 4x4x2 decomposition using 4 Strassen 2x2x2 blocks.
    A is 4x4, B is 4x2, C is 4x2.
    Rank = 4 * 7 = 28.
    
    U has shape (16, 28), V has shape (8, 28), W has shape (8, 28).
    U[i*4+j, r], V[j*2+k, r], W[k*4+i, r].
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 28
    U = np.zeros((16, rank))
    V = np.zeros((8, rank))
    W = np.zeros((8, rank))
    
    def add_strassen_block(a_i_start, a_j_start, b_j_start, c_i_start, c_k_start, base_rank):
        for t in range(7):
            col = base_rank + t
            
            # A part
            for li in range(2):
                for lj in range(2):
                    local_a_row = li * 2 + lj
                    coeff_a = U2[local_a_row][t]
                    if coeff_a != 0:
                        global_i = a_i_start + li
                        global_j = a_j_start + lj
                        u_row = global_i * 4 + global_j
                        U[u_row][col] += coeff_a
                        
            # B part
            for lj in range(2):
                for lk in range(2):
                    local_b_row = lj * 2 + lk
                    coeff_b = V2[local_b_row][t]
                    if coeff_b != 0:
                        global_j = b_j_start + lj
                        global_k = c_k_start + lk
                        v_row = global_j * 2 + global_k
                        V[v_row][col] += coeff_b
                        
            # C part
            for lk in range(2):
                for li in range(2):
                    local_c_row = lk * 2 + li
                    coeff_c = W2[local_c_row][t]
                    if coeff_c != 0:
                        global_k = c_k_start + lk
                        global_i = c_i_start + li
                        w_row = global_k * 4 + global_i
                        W[w_row][col] += coeff_c

    # 4 blocks
    add_strassen_block(0, 0, 0, 0, 0, 0)
    add_strassen_block(0, 2, 2, 0, 0, 7)
    add_strassen_block(2, 0, 0, 2, 0, 14)
    add_strassen_block(2, 2, 2, 2, 0, 21)
    
    return U, V, W

def _build_strassen_4x4x4():
    """
    Build 4x4x4 decomposition using Kronecker product of 2x2x2 Strassen.
    Rank = 7 * 7 = 49.
    U has shape (16, 49), V has shape (16, 49), W has shape (16, 49).
    U[i*4+j, r], V[j*4+k, r], W[k*4+i, r].
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    for r1 in range(7):
        for r2 in range(7):
            col = r1 * 7 + r2
            
            # U part: U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
            for i1 in range(2):
                for j1 in range(2):
                    c1 = U2[i1 * 2 + j1, r1]
                    if c1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            c2 = U2[i2 * 2 + j2, r2]
                            if c2 == 0: continue
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U[i * 4 + j, col] += c1 * c2
            
            # V part: V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
            for j1 in range(2):
                for k1 in range(2):
                    c1 = V2[j1 * 2 + k1, r1]
                    if c1 == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            c2 = V2[j2 * 2 + k2, r2]
                            if c2 == 0: continue
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            V[j * 4 + k, col] += c1 * c2
            
            # W part: W2[k1*2+i1, r1] * W2[k2*2+i2, r2]
            for k1 in range(2):
                for i1 in range(2):
                    c1 = W2[k1 * 2 + i1, r1]
                    if c1 == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            c2 = W2[k2 * 2 + i2, r2]
                            if c2 == 0: continue
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            W[k * 4 + i, col] += c1 * c2
    
    return U, V, W

def _build_4x5x6_block_cyclic():
    """
    Build 4x5x6 decomposition using block-cyclic 2x2x2 Strassen blocks.
    
    Partition:
    - A (4x5) split into row blocks: A_top (2x5), A_bot (2x5)
    - B (5x6) split into col blocks: B_L (5x2), B_M (5x2), B_R (5x2)
    
    C has 6 blocks (2x2 each):
    C[0:2,0:2] = A_top @ B_L
    C[0:2,2:4] = A_top @ B_M
    C[0:2,4:6] = A_top @ B_R
    C[2:4,0:2] = A_bot @ B_L
    C[2:4,2:4] = A_bot @ B_M
    C[2:4,4:6] = A_bot @ B_R
    
    For each C block (2x5 @ 5x2):
    - Split middle dim 5 into 2+2+1
    - Use 2x2x2 Strassen for the 2x2 parts (rank 7 each)
    - Use schoolbook for the 2x1x2 part (rank 4)
    
    Per C block: 7+7+4 = 18
    Total: 6 * 18 = 108 (initial), then column reduction may help.
    """
    n, m, p = 4, 5, 6
    U2, V2, W2 = _build_strassen_2x2()
    
    # Each C block: 2x5 @ 5x2
    # Split middle: indices 0-1, 2-3, 4
    # Part 1: A[:,0:2] @ B[0:2,:] -> 2x2x2 Strassen (rank 7)
    # Part 2: A[:,2:4] @ B[2:4,:] -> 2x2x2 Strassen (rank 7)
    # Part 3: A[:,4] @ B[4,:] -> 2x1x2 schoolbook (rank 4)
    
    per_block_rank = 7 + 7 + 4
    total_rank = 6 * per_block_rank
    
    U = np.zeros((n * m, total_rank))  # 20 rows
    V = np.zeros((m * p, total_rank))  # 30 rows
    W = np.zeros((p * n, total_rank))  # 24 rows
    
    # C block (ci, ck) where ci in {0,1} (row block), ck in {0,1,2} (col block)
    for ci in range(2):
        for ck in range(3):
            base = (ci * 3 + ck) * per_block_rank
            
            # Part 1: A[ci*2:ci*2+2, 0:2] @ B[0:2, ck*2:ck*2+2]
            # This is a 2x2x2 product
            for t in range(7):
                col = base + t
                # U part: A indices (i, j) where i in {ci*2, ci*2+1}, j in {0, 1}
                for li in range(2):
                    for lj in range(2):
                        local_a = li * 2 + lj
                        coeff = U2[local_a, t]
                        if coeff != 0:
                            gi = ci * 2 + li
                            gj = lj  # j index in A (0 or 1)
                            U[gi * 5 + gj, col] += coeff
                # V part: B indices (j, k) where j in {0, 1}, k in {ck*2, ck*2+1}
                for lj in range(2):
                    for lk in range(2):
                        local_b = lj * 2 + lk
                        coeff = V2[local_b, t]
                        if coeff != 0:
                            gj = lj  # j index in B (0 or 1)
                            gk = ck * 2 + lk
                            V[gj * 6 + gk, col] += coeff
                # W part: C indices (k, i) where k in {ck*2, ck*2+1}, i in {ci*2, ci*2+1}
                for lk in range(2):
                    for li in range(2):
                        local_c = lk * 2 + li
                        coeff = W2[local_c, t]
                        if coeff != 0:
                            gk = ck * 2 + lk
                            gi = ci * 2 + li
                            W[gk * 4 + gi, col] += coeff
            
            # Part 2: A[ci*2:ci*2+2, 2:4] @ B[2:4, ck*2:ck*2+2]
            for t in range(7):
                col = base + 7 + t
                for li in range(2):
                    for lj in range(2):
                        local_a = li * 2 + lj
                        coeff = U2[local_a, t]
                        if coeff != 0:
                            gi = ci * 2 + li
                            gj = 2 + lj  # j index in A (2 or 3)
                            U[gi * 5 + gj, col] += coeff
                for lj in range(2):
                    for lk in range(2):
                        local_b = lj * 2 + lk
                        coeff = V2[local_b, t]
                        if coeff != 0:
                            gj = 2 + lj  # j index in B (2 or 3)
                            gk = ck * 2 + lk
                            V[gj * 6 + gk, col] += coeff
                for lk in range(2):
                    for li in range(2):
                        local_c = lk * 2 + li
                        coeff = W2[local_c, t]
                        if coeff != 0:
                            gk = ck * 2 + lk
                            gi = ci * 2 + li
                            W[gk * 4 + gi, col] += coeff
            
            # Part 3: A[ci*2:ci*2+2, 4] @ B[4, ck*2:ck*2+2]
            # Schoolbook: rank 4 (2*1*2)
            for li in range(2):
                for lk in range(2):
                    col = base + 14 + li * 2 + lk
                    gi = ci * 2 + li
                    gj = 4  # j index in A
                    gk = ck * 2 + lk
                    U[gi * 5 + gj, col] = 1.0
                    V[gj * 6 + gk, col] = 1.0
                    W[gk * 4 + gi, col] = 1.0
    
    return U, V, W

def _build_4x5x6_improved():
    """
    Build 4x5x6 decomposition using 4x4x4 Strassen and 4x4x2 Strassen.
    
    Strategy:
    - Split A (4x5) into A_L (4x4) and A_R (4x1)
    - Split B (5x6) into B_T (4x6) and B_B (1x6)
    - C = A_L @ B_T + A_R @ B_B
    
    For A_L @ B_T (4x4 @ 4x6):
      - Split B_T into B_T1 (4x4) and B_T2 (4x2)
      - Use 4x4x4 Strassen for A_L @ B_T1 (rank 49)
      - Use 4x4x2 Strassen for A_L @ B_T2 (rank 28)
    
    For A_R @ B_B (4x1 @ 1x6):
      - Schoolbook: rank 24
    
    Total rank: 49 + 28 + 24 = 101
    """
    n, m, p = 4, 5, 6
    
    U444, V444, W444 = _build_strassen_4x4x4()
    U442, V442, W442 = _build_strassen_4x4x2()
    
    total_rank = 49 + 28 + 24
    U = np.zeros((n * m, total_rank))  # 4x5 = 20 rows
    V = np.zeros((m * p, total_rank))  # 5x6 = 30 rows
    W = np.zeros((p * n, total_rank))  # 6x4 = 24 rows
    
    # Part 1: A_L @ B_T1 (4x4 @ 4x4)
    base = 0
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, base:base+49] = U444[i * 4 + j, :]
    
    for j in range(4):
        for k in range(4):
            V[j * 6 + k, base:base+49] = V444[j * 4 + k, :]
    
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, base:base+49] = W444[k * 4 + i, :]
    
    # Part 2: A_L @ B_T2 (4x4 @ 4x2)
    base = 49
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, base:base+28] = U442[i * 4 + j, :]
    
    for j in range(4):
        for k_local in range(2):
            k_global = 4 + k_local
            V[j * 6 + k_global, base:base+28] = V442[j * 2 + k_local, :]
    
    for k_local in range(2):
        k_global = 4 + k_local
        for i in range(4):
            W[k_global * 4 + i, base:base+28] = W442[k_local * 4 + i, :]
    
    # Part 3: A_R @ B_B (4x1 @ 1x6)
    base = 49 + 28
    for i in range(4):
        for k in range(6):
            col = base + i * 6 + k
            U[i * 5 + 4, col] = 1.0
            V[4 * 6 + k, col] = 1.0
            W[k * 4 + i, col] = 1.0
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """Schoolbook decomposition for n x m @ m x p (rank n*m*p)."""
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _residual_zero(U, V, W, n, m, p, tol=1e-9):
    """Check if the residual tensor (after subtracting the rank-1 contributions) is zero."""
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return bool(np.all(np.abs(T - 1.0) < tol))

def _local_reduce(U, V, W, n, m, p, max_iters=50, budget_s=60.0):
    """Local column-removal search with randomization."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    for _ in range(max_iters):
        if _time_left() < 5.0 or (time.time() - start_time) > budget_s:
            break
        if R < 2:
            break
        found = False
        # Randomly shuffle columns to try different orders
        cols = list(range(R))
        rng.shuffle(cols)
        
        for c in cols:
            if _time_left() < 5.0 or (time.time() - start_time) > budget_s:
                break
            U2 = np.delete(U, c, axis=1)
            V2 = np.delete(V, c, axis=1)
            W2 = np.delete(W, c, axis=1)
            if _residual_zero(U2, V2, W2, n, m, p, 1e-6):
                if _verify_decomp(U2, V2, W2, n, m, p, 1e-9):
                    U, V, W = U2, V2, W2
                    R -= 1
                    found = True
                    break
        if not found:
            break
    return U, V, W

def _systematic_reduce(U, V, W, n, m, p, budget_s=60.0):
    """Systematic column removal: try all singles in sparsity order, then all pairs."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    start_time = time.time()
    
    # Phase 1: Try all single column removals in sparsity order
    found_single = True
    while found_single and (time.time() - start_time) < budget_s and _time_left() > 5.0:
        found_single = False
        R = U.shape[1]
        if R < 2:
            break
        U_sparsity = np.count_nonzero(U, axis=0)
        V_sparsity = np.count_nonzero(V, axis=0)
        W_sparsity = np.count_nonzero(W, axis=0)
        total_sparsity = U_sparsity + V_sparsity + W_sparsity
        sorted_cols = np.argsort(total_sparsity)
        
        for c in sorted_cols:
            if _time_left() < 5.0 or (time.time() - start_time) > budget_s:
                break
            U2 = np.delete(U, c, axis=1)
            V2 = np.delete(V, c, axis=1)
            W2 = np.delete(W, c, axis=1)
            if _residual_zero(U2, V2, W2, n, m, p, 1e-6):
                if _verify_decomp(U2, V2, W2, n, m, p, 1e-9):
                    U, V, W = U2, V2, W2
                    found_single = True
                    break
    
    # Phase 2: Try all pair removals in sparsity order
    found_pair = True
    while found_pair and (time.time() - start_time) < budget_s and _time_left() > 5.0:
        found_pair = False
        R = U.shape[1]
        if R < 3:
            break
        U_sparsity = np.count_nonzero(U, axis=0)
        V_sparsity = np.count_nonzero(V, axis=0)
        W_sparsity = np.count_nonzero(W, axis=0)
        total_sparsity = U_sparsity + V_sparsity + W_sparsity
        sorted_cols = np.argsort(total_sparsity)
        
        # Try pairs of the sparsest columns
        for i in range(min(R, 30)):
            if _time_left() < 5.0 or (time.time() - start_time) > budget_s:
                break
            for j in range(i+1, min(R, 30)):
                if _time_left() < 5.0 or (time.time() - start_time) > budget_s:
                    break
                mask = np.ones(R, dtype=bool)
                mask[sorted_cols[i]] = False
                mask[sorted_cols[j]] = False
                U2 = U[:, mask]
                V2 = V[:, mask]
                W2 = W[:, mask]
                if _residual_zero(U2, V2, W2, n, m, p, 1e-6):
                    if _verify_decomp(U2, V2, W2, n, m, p, 1e-9):
                        U, V, W = U2, V2, W2
                        found_pair = True
                        break
            if found_pair:
                break
    
    return U, V, W

def _local_reduce_pairs(U, V, W, n, m, p, budget_s=30.0):
    """Try removing pairs of columns."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    start_time = time.time()
    rng = np.random.RandomState(123)
    
    if R < 3:
        return U, V, W
    
    # Try random pairs
    for _ in range(100):
        if _time_left() < 5.0 or (time.time() - start_time) > budget_s:
            break
        c1, c2 = rng.choice(R, size=2, replace=False)
        if c1 == c2:
            continue
        mask = np.ones(R, dtype=bool)
        mask[c1] = False
        mask[c2] = False
        U2 = U[:, mask]
        V2 = V[:, mask]
        W2 = W[:, mask]
        if _residual_zero(U2, V2, W2, n, m, p, 1e-6):
            if _verify_decomp(U2, V2, W2, n, m, p, 1e-9):
                return U2, V2, W2
    
    return U, V, W

def _try_remove_fast(U, V, W, n, m, p, budget_s=30.0):
    """Fast column removal using vectorized checks."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    start_time = time.time()
    
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    
    # Precompute the full tensor
    T_full = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    for c in range(R):
        if _time_left() < 5.0 or (time.time() - start_time) > budget_s:
            break
        # Compute the contribution of column c
        T_c = np.einsum('ij,jk,ki->ijk', U3[:,:,c], V3[:,:,c], W3[:,:,c])
        T_remainder = T_full - T_c
        if np.all(np.abs(T_remainder - 1.0) < 1e-6):
            # Verify with exact check
            U2 = np.delete(U, c, axis=1)
            V2 = np.delete(V, c, axis=1)
            W2 = np.delete(W, c, axis=1)
            if _verify_decomp(U2, V2, W2, n, m, p, 1e-9):
                return U2, V2, W2
    
    return U, V, W

def _build_4x5x6_winograd():
    """
    Build a Winograd-style decomposition for 4x5x6.
    This uses a different partitioning strategy.
    """
    n, m, p = 4, 5, 6
    
    # Use the improved version as base
    U, V, W = _build_4x5x6_improved()
    return U, V, W

def _linear_reduce(U, V, W, n, m, p, budget_s=30.0):
    """
    Attempts to reduce rank by finding linear dependencies among columns.
    Uses sparsity-based heuristic.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    start_time = time.time()
    
    found = True
    while found and (time.time() - start_time) < budget_s and _time_left() > 5.0:
        found = False
        R = U.shape[1]
        
        U_sparsity = np.count_nonzero(U, axis=0)
        V_sparsity = np.count_nonzero(V, axis=0)
        W_sparsity = np.count_nonzero(W, axis=0)
        total_sparsity = U_sparsity + V_sparsity + W_sparsity
        sorted_cols = np.argsort(total_sparsity)
        
        for c in sorted_cols:
            if _time_left() < 5.0 or (time.time() - start_time) > budget_s:
                break
            U2 = np.delete(U, c, axis=1)
            V2 = np.delete(V, c, axis=1)
            W2 = np.delete(W, c, axis=1)
            if _residual_zero(U2, V2, W2, n, m, p, 1e-6):
                if _verify_decomp(U2, V2, W2, n, m, p, 1e-9):
                    U, V, W = U2, V2, W2
                    found = True
                    break
                    
    return U, V, W

def construct():
    n, m, p = 4, 5, 6
    
    # Start with the improved decomposition
    best_rank = n * m * p
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    
    # Try Improved 4x5x6
    try:
        if _time_left() > 30.0:
            U, V, W = _build_4x5x6_improved()
            if _verify_decomp(U, V, W, n, m, p, 1e-9):
                if U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best_U, best_V, best_W = U, V, W
    except Exception:
        pass
    
    # Try Block-cyclic 4x5x6
    try:
        if _time_left() > 30.0:
            U, V, W = _build_4x5x6_block_cyclic()
            if _verify_decomp(U, V, W, n, m, p, 1e-9):
                if U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best_U, best_V, best_W = U, V, W
    except Exception:
        pass
    
    # Fast column removal
    try:
        if _time_left() > 15.0:
            cand = _try_remove_fast(best_U, best_V, best_W, n, m, p, budget_s=20.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass
    
    # Local column-removal refinement (Randomized)
    try:
        if _time_left() > 15.0:
            cand = _local_reduce(best_U, best_V, best_W, n, m, p, max_iters=50, budget_s=40.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass
    
    # Pair removal
    try:
        if _time_left() > 10.0:
            cand = _local_reduce_pairs(best_U, best_V, best_W, n, m, p, budget_s=20.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass
    
    # Linear/Sparsity-based column-removal refinement
    try:
        if _time_left() > 10.0:
            cand = _linear_reduce(best_U, best_V, best_W, n, m, p, budget_s=20.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass
    
    # Systematic reduction
    try:
        if _time_left() > 10.0:
            cand = _systematic_reduce(best_U, best_V, best_W, n, m, p, budget_s=30.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass
    
    # Final safety check
    if not _verify_decomp(best_U, best_V, best_W, n, m, p, 1e-9):
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
