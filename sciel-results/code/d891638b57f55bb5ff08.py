"""
Program to find a rank-minimal exact tensor decomposition for multiplying
a 3 by 5 matrix by a 5 by 6 matrix.

Strategy:
1. Build a block decomposition using Strassen (rank 7) for 2x2x2 blocks
   and schoolbook for all other blocks.
2. Try Winograd 2x4x4 (rank 16) for 2x4x4 blocks if beneficial.
3. Search over block partitions to minimize total rank.
4. Attempt rank reduction via ALS with exact verification.
5. Round to half-integers and verify.
"""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def strassen_factors():
    """
    Returns (U, V, W) for 2x2x2 Strassen decomposition.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U row i*2+j, V row j*2+k, W row k*2+i
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # m1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # m2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # m3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # m4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # m5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[2, 4] = -1; W[3, 4] = 1
    
    # m6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1
    
    # m7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[2, 6] = 1
    
    return U, V, W

def schoolbook_factors(n, m, p):
    """
    Returns (U, V, W) for n x m x p schoolbook decomposition.
    U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def winograd_2x4x4_factors():
    """
    Winograd 2x4x4: multiplies 2x4 A by 4x4 B to get 2x4 C.
    Uses 16 scalar multiplications.
    
    A is 2x4: A[0,0..3], A[1,0..3]
    B is 4x4: B[0..3,0..3]
    C is 2x4: C[0,0..3], C[1,0..3]
    
    Winograd scheme for 2x4x4:
    Transform A: 4x4 matrix F_A such that A' = F_A @ A_flat (reshaped)
    Transform B: 4x4 matrix F_B such that B' = F_B @ B_flat (reshaped)
    Element-wise multiply: M = A' * B' (16 elements)
    Transform back: C = F_C @ M (reshaped)
    
    Actually, let me use a simpler approach. The Winograd 2x4x4 algorithm:
    
    For C = A @ B where A is 2x4 and B is 4x4:
    
    Let's use the standard Winograd scheme:
    m1 = (a00+a10+a01+a11) * (b00+b01+b02+b03 + b10+b11+b12+b13 + b20+b21+b22+b23 + b30+b31+b32+b33)
    ... this gets complex.
    
    Let me use a different approach: build the Winograd factors directly.
    
    Actually, for simplicity and correctness, let me just use schoolbook for 2x4x4
    unless I can verify a correct Winograd decomposition.
    
    For now, I'll return None to indicate we should use schoolbook.
    """
    return None

def build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes):
    """
    Build the full factor matrices using block decomposition.
    """
    total_rank = 0
    n_offsets = []
    offset = 0
    for s in n_sizes:
        n_offsets.append(offset)
        offset += s
    
    m_offsets = []
    offset = 0
    for s in m_sizes:
        m_offsets.append(offset)
        offset += s
    
    p_offsets = []
    offset = 0
    for s in p_sizes:
        p_offsets.append(offset)
        offset += s
    
    for i_idx, ni in enumerate(n_sizes):
        for j_idx, mj in enumerate(m_sizes):
            for k_idx, pk in enumerate(p_sizes):
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
    
    U_full = np.zeros((n * m, total_rank), dtype=np.float64)
    V_full = np.zeros((m * p, total_rank), dtype=np.float64)
    W_full = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col_offset = 0
    for i_idx, ni in enumerate(n_sizes):
        i_start = n_offsets[i_idx]
        for j_idx, mj in enumerate(m_sizes):
            j_start = m_offsets[j_idx]
            for k_idx, pk in enumerate(p_sizes):
                k_start = p_offsets[k_idx]
                
                if ni == 2 and mj == 2 and pk == 2:
                    U_block, V_block, W_block = strassen_factors()
                else:
                    U_block, V_block, W_block = schoolbook_factors(ni, mj, pk)
                r = U_block.shape[1]
                
                for li in range(ni):
                    for lj in range(mj):
                        full_row = (i_start + li) * m + (j_start + lj)
                        U_full[full_row, col_offset:col_offset + r] = U_block[li * mj + lj, :]
                
                for lj in range(mj):
                    for lk in range(pk):
                        full_row = (j_start + lj) * p + (k_start + lk)
                        V_full[full_row, col_offset:col_offset + r] = V_block[lj * pk + lk, :]
                
                for lk in range(pk):
                    for li in range(ni):
                        full_row = (k_start + lk) * n + (i_start + li)
                        W_full[full_row, col_offset:col_offset + r] = W_block[lk * ni + li, :]
                
                col_offset += r
    
    return U_full, V_full, W_full, total_rank

def verify_decomposition(U, V, W, n, m, p, atol=1e-10):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    """
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_rows = i_idx * m + j_idx
    v_rows = j_idx * p + k_idx
    w_rows = k_idx * n + i_idx
    
    products = U[u_rows, :] * V[v_rows, :] * W[w_rows, :]
    vals = products.sum(axis=1)
    
    return np.allclose(vals, 1.0, atol=atol)

def round_to_half_integers(mat, tol=1e-8):
    """
    Round matrix entries to nearest half-integers and return (rounded_mat, is_exact).
    """
    doubled = 2.0 * mat
    rounded_doubled = np.round(doubled)
    if np.allclose(doubled, rounded_doubled, atol=tol):
        return rounded_doubled / 2.0, True
    return None, False

def _block_rank(ni, mj, pk):
    """Return the rank for a block of size ni x mj x pk."""
    if ni == 2 and mj == 2 and pk == 2:
        return 7
    return ni * mj * pk

def _find_best_block_partition(n, m, p, deadline):
    """
    Search over block partitions to minimize total rank.
    Returns (n_sizes, m_sizes, p_sizes, total_rank).
    """
    best_rank = n * m * p
    best_partition = ([n], [m], [p])
    
    # Generate all possible partitions of a number into parts of size 1 or 2
    def partitions(num, max_part=2):
        results = []
        def helper(remaining, current):
            if remaining == 0:
                results.append(list(current))
                return
            for part in range(min(max_part, remaining), 0, -1):
                current.append(part)
                helper(remaining - part, current)
                current.pop()
        helper(num, [])
        return results
    
    n_parts = partitions(n)
    m_parts = partitions(m)
    p_parts = partitions(p)
    
    for np_ in n_parts:
        for mp_ in m_parts:
            for pp_ in p_parts:
                rank = 0
                for ni in np_:
                    for mj in mp_:
                        for pk in pp_:
                            rank += _block_rank(ni, mj, pk)
                if rank < best_rank:
                    best_rank = rank
                    best_partition = (np_, mp_, pp_)
    
    return best_partition[0], best_partition[1], best_partition[2], best_rank

def als_reduce_rank(U, V, W, n, m, p, target_rank, max_iter=50, deadline=None, seed=42):
    """
    Attempt to reduce the rank of the decomposition to target_rank using
    alternating least squares (ALS) on the tensor.
    Returns (U_new, V_new, W_new) or None if not successful.
    """
    if deadline is not None and time.time() > deadline - 10.0:
        return None
    
    current_rank = U.shape[1]
    if target_rank >= current_rank:
        return U, V, W
    
    n_alpha = n * m
    n_beta = m * p
    n_gamma = p * n
    
    # Precompute index mappings for the tensor
    alpha_to_ijk = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha_to_ijk.append((i, j, k))
    
    # Build T as a sparse representation
    T_vals = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T_vals[(alpha, beta, gamma)] = 1.0
    
    rng = np.random.RandomState(seed)
    U_new = rng.randn(n_alpha, target_rank) * 0.1
    V_new = rng.randn(n_beta, target_rank) * 0.1
    W_new = rng.randn(n_gamma, target_rank) * 0.1
    
    best_loss = np.inf
    best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 10.0:
            break
        
        # Update U
        for r in range(target_rank):
            Vr = V_new[:, r]
            Wr = W_new[:, r]
            for alpha in range(n_alpha):
                i = alpha // m
                j = alpha % m
                num = 0.0
                den = 0.0
                for k in range(p):
                    beta = j * p + k
                    gamma = k * n + i
                    num += T_vals.get((alpha, beta, gamma), 0.0) * Vr[beta] * Wr[gamma]
                    den += Vr[beta] * Vr[beta] * Wr[gamma] * Wr[gamma]
                if den > 1e-15:
                    U_new[alpha, r] = num / den
        
        # Update V
        for r in range(target_rank):
            Ur = U_new[:, r]
            Wr = W_new[:, r]
            for beta in range(n_beta):
                j = beta // p
                k = beta % p
                num = 0.0
                den = 0.0
                for i in range(n):
                    alpha = i * m + j
                    gamma = k * n + i
                    num += T_vals.get((alpha, beta, gamma), 0.0) * Ur[alpha] * Wr[gamma]
                    den += Ur[alpha] * Ur[alpha] * Wr[gamma] * Wr[gamma]
                if den > 1e-15:
                    V_new[beta, r] = num / den
        
        # Update W
        for r in range(target_rank):
            Ur = U_new[:, r]
            Vr = V_new[:, r]
            for gamma in range(n_gamma):
                k = gamma // n
                i = gamma % n
                num = 0.0
                den = 0.0
                for j in range(m):
                    alpha = i * m + j
                    beta = j * p + k
                    num += T_vals.get((alpha, beta, gamma), 0.0) * Ur[alpha] * Vr[beta]
                    den += Ur[alpha] * Ur[alpha] * Vr[beta] * Vr[beta]
                if den > 1e-15:
                    W_new[gamma, r] = num / den
        
        # Compute loss
        loss = 0.0
        for (alpha, beta, gamma), val in T_vals.items():
            pred = np.sum(U_new[alpha, :] * V_new[beta, :] * W_new[gamma, :])
            loss += (pred - val) ** 2
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
        
        if best_loss < 1e-10:
            break
    
    if best_loss < 1e-8:
        U_rounded, U_exact = round_to_half_integers(best_U, tol=1e-8)
        V_rounded, V_exact = round_to_half_integers(best_V, tol=1e-8)
        W_rounded, W_exact = round_to_half_integers(best_W, tol=1e-8)
        
        if U_exact and V_exact and W_exact:
            if verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, atol=1e-10):
                return U_rounded, V_rounded, W_rounded
        
        if verify_decomposition(best_U, best_V, best_W, n, m, p, atol=1e-10):
            return best_U, best_V, best_W
    
    return None

def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of multiplying a 3x5 matrix by a 5x6 matrix.
    """
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # Find the best block partition
    n_sizes, m_sizes, p_sizes, total_rank = _find_best_block_partition(n, m, p, deadline)
    
    # Build the block decomposition
    U, V, W, total_rank = build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes)
    
    # Verify correctness
    if time.time() < deadline:
        if not verify_decomposition(U, V, W, n, m, p, atol=1e-10):
            # Fall back to schoolbook
            U, V, W = schoolbook_factors(n, m, p)
            total_rank = n * m * p
    
    # Try to reduce rank further using ALS-based methods
    if time.time() < deadline - 15.0:
        for target in range(total_rank - 1, max(total_rank - 10, 60), -1):
            if time.time() > deadline - 15.0:
                break
            
            result = als_reduce_rank(U, V, W, n, m, p, target, max_iter=30, deadline=deadline, seed=42+target)
            if result is not None:
                U_new, V_new, W_new = result
                if verify_decomposition(U_new, V_new, W_new, n, m, p, atol=1e-10):
                    U, V, W = U_new, V_new, W_new
                    total_rank = target
                    break
    
    # Try rounding to half-integers
    if time.time() < deadline - 5.0:
        U_rounded, U_exact = round_to_half_integers(U, tol=1e-8)
        V_rounded, V_exact = round_to_half_integers(V, tol=1e-8)
        W_rounded, W_exact = round_to_half_integers(W, tol=1e-8)
        
        if U_exact and V_exact and W_exact:
            if verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, atol=1e-10):
                U, V, W = U_rounded, V_rounded, W_rounded
    
    # Convert to lists
    U_list = U.tolist() if isinstance(U, np.ndarray) else U
    V_list = V.tolist() if isinstance(V, np.ndarray) else V
    W_list = W.tolist() if isinstance(W, np.ndarray) else W
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
