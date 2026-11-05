import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _build_block_decomposition():
    """
    Build a block decomposition for 5x5x5 matrix multiplication.
    Partition: 5 = 2 + 2 + 1
    - 2x2x2 blocks: use rank-7 Strassen
    - 2x2x1, 2x1x2, 1x2x2, 1x1x2, 2x1x1, 1x2x1, 1x1x1 blocks: use schoolbook
    
    Returns U, V, W matrices.
    """
    n, m, p = 5, 5, 5
    
    # Block boundaries: [0, 2, 4, 5]
    splits = [0, 2, 4, 5]
    num_blocks = len(splits) - 1  # 3 blocks
    
    # Collect all terms
    U_terms = []  # Each term: (row_indices, coeffs)
    V_terms = []
    W_terms = []
    
    def add_term(U_idx, U_coeff, V_idx, V_coeff, W_idx, W_coeff):
        U_terms.append((U_idx, U_coeff))
        V_terms.append((V_idx, V_coeff))
        W_terms.append((W_idx, W_coeff))
    
    def get_block_start(b):
        return splits[b]
    
    def get_block_size(b):
        return splits[b + 1] - splits[b]
    
    # Strassen 2x2 terms (local indices 0..3)
    def strassen_2x2():
        terms = []
        # M1 = (a00+a11)(b00+b11)
        terms.append(([0, 3], [1, 1], [0, 3], [1, 1], [0, 2, 3], [1, 0, 1]))
        # M2 = (a10+a11)b00
        terms.append(([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]))
        # M3 = a00(b01-b11)
        terms.append(([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]))
        # M4 = a11(b10-b00)
        terms.append(([3], [1], [2, 0], [1, -1], [0, 2], [1, 1]))
        # M5 = (a00+a01)b11
        terms.append(([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]))
        # M6 = (a10-a00)(b00+b01)
        terms.append(([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]))
        # M7 = (a01-a11)(b10+b11)
        terms.append(([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]))
        return terms
    
    # For each combination of blocks (i, j, k) where i is A's row block,
    # j is B's row block (= A's col block), k is B's col block (= C's col block)
    for i_block in range(num_blocks):
        for j_block in range(num_blocks):
            for k_block in range(num_blocks):
                # A block: rows i_block, cols j_block
                # B block: rows j_block, cols k_block
                # C block: rows i_block, cols k_block
                
                ai = get_block_start(i_block)
                aj = get_block_start(j_block)
                ak = get_block_start(k_block)
                
                si = get_block_size(i_block)
                sj = get_block_size(j_block)
                sk = get_block_size(k_block)
                
                # If all three blocks are size 2, use Strassen
                if si == 2 and sj == 2 and sk == 2:
                    for (ui, uc, vi, vc, wi, wc) in strassen_2x2():
                        # Map local indices to global
                        ui_global = [ai * m + aj + x for x in ui]
                        vi_global = [aj * p + ak + x for x in vi]
                        wi_global = [ak * n + ai + x for x in wi]
                        add_term(ui_global, uc, vi_global, vc, wi_global, wc)
                else:
                    # Schoolbook for this block combination
                    for i in range(si):
                        for j in range(sj):
                            for k in range(sk):
                                a_idx = (ai + i) * m + (aj + j)
                                b_idx = (aj + j) * p + (ak + k)
                                c_idx = (ak + k) * n + (ai + i)
                                add_term([a_idx], [1], [b_idx], [1], [c_idx], [1])
    
    rank = len(U_terms)
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    for r in range(rank):
        ui, uc = U_terms[r]
        vi, vc = V_terms[r]
        wi, wc = W_terms[r]
        
        for idx, coeff in zip(ui, uc):
            U[idx, r] = coeff
        for idx, coeff in zip(vi, vc):
            V[idx, r] = coeff
        for idx, coeff in zip(wi, wc):
            W[idx, r] = coeff
    
    return U, V, W, rank


def _verify_decomposition(U, V, W, n=5, m=5, p=5, tol=1e-9):
    """
    Verify that the tensor decomposition is exact.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: for all (i,k): sum_c W[k*n+i, c] * (sum_j U[i*m+j, c]*A[i,j]) * (sum_j V[j*p+k, c]*B[j,k]) == (AB)[i,k]
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    rank = U.shape[1]
    
    np.random.seed(12345)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_decomp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            u_vals = np.zeros(rank)
            for j in range(m):
                u_vals += U[i * m + j, :] * A[i, j]
            v_vals = np.zeros(rank)
            for j in range(m):
                v_vals += V[j * p + k, :] * B[j, k]
            C_decomp[i, k] = np.dot(W[k * n + i, :], u_vals * v_vals)
    
    err = np.max(np.abs(C_decomp - C_expected))
    return err < tol, err


def construct():
    n, m, p = DIMENSIONS
    
    deadline = time.time() + 140.0
    
    def get_schoolbook():
        """Return the schoolbook decomposition (rank 125)."""
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U.tolist(), V.tolist(), W.tolist()
    
    # Start with schoolbook as baseline
    U_best, V_best, W_best = get_schoolbook()
    best_rank = n * m * p
    
    # Try block decomposition with 2+2+1 partition
    if time.time() < deadline - 20.0:
        try:
            U_block, V_block, W_block, rank_block = _build_block_decomposition()
            valid, err = _verify_decomposition(U_block, V_block, W_block, n, m, p)
            if valid and rank_block < best_rank:
                U_best = U_block.tolist()
                V_best = V_block.tolist()
                W_best = W_block.tolist()
                best_rank = rank_block
        except Exception:
            pass
    
    return {"U": U_best, "V": V_best, "W": W_best}

# EVOLVE-BLOCK-END
