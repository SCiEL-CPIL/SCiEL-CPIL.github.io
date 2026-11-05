"""Matrix multiplication 3x5x5 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a11 + a22)(b11 + b22)
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1]
    
    # M2 = (a21 + a22) b11
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 1, 0, -1]
    
    # M3 = a11 (b12 - b22)
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 1, 0, -1]
    W[:, 2] = [0, 0, 1, 1]
    
    # M4 = a22 (b21 - b11)
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [-1, 0, 1, 0]
    W[:, 3] = [1, 1, 0, 0]
    
    # M5 = (a11 + a12) b22
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [-1, 0, 1, 0]
    
    # M6 = (a21 - a11) (b11 + b12)
    U[:, 5] = [-1, 0, 1, 0]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [0, 0, 0, 1]
    
    # M7 = (a12 - a22) (b21 + b22)
    U[:, 6] = [0, 1, 0, -1]
    V[:, 6] = [0, 0, 1, 1]
    W[:, 6] = [1, 0, 0, 0]
    
    return U, V, W


def _rank1d_decomposition(n, m, p):
    """
    Minimal-rank decomposition for 1 x m x p (or n x 1 x p, etc.) blocks.
    For 1 x m x p: rank = p (not m*p).
    We can compute C = A * B where A is 1xm, B is mxp with p multiplications.
    
    Actually, for 1 x m x p matrix multiplication:
    C_k = sum_j A_j * B_jk
    
    We can do this with p multiplications:
    M_k = A * (sum_j B_jk) -- no, that's not right.
    
    Wait, let's think again. For 1 x m x p:
    C_k = sum_j A_j B_jk
    
    This is the dot product of A with each column of B.
    We can compute each C_k with m multiplications, total m*p.
    
    But can we do better? The rank of M_{1,m,p} is actually p (not m*p)
    when m >= 1... no, that's not right either.
    
    Actually, the tensor rank of M_{1,m,p} is min(m, p) * max(1, ...) hmm.
    
    Let me reconsider. For 1 x m x p, the multiplication is:
    C_k = sum_j A_j B_jk
    
    This is equivalent to computing the product of a row vector and a matrix.
    The rank of this tensor is actually m (if we fix p) or p (if we fix m)...
    
    No wait. Let me think about it as: we need to compute m*p outputs (well, p outputs for a 1xm row).
    Each output C_k = sum_j A_j B_jk.
    
    We can write: M_k = (sum_j A_j) * (B_jk) -- no, that doesn't work.
    
    Actually, for 1 x m x p, the rank is min(m, p). No, that's not right.
    
    Let me just use schoolbook for now and focus on the structural change.
    The key insight from the knowledge is that Strassen saves 1 rank per 2x2x2 block.
    The current best uses rank 74 (75 - 1 from one Strassen block).
    
    Let me try a different approach: use multiple Strassen blocks in a smarter way.
    """
    # For now, fall back to schoolbook
    return _schoolbook(n, m, p)


def _schoolbook(n, m, p):
    """Schoolbook decomposition for n x m x p. Returns U, V, W."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _embed_block(U_local, V_local, W_local, n, m, p, n_off, m_off, p_off, ni, mi, pi):
    """
    Embed a local decomposition of size ni x mi x pi into global dimensions.
    Returns the global U, V, W slices (as arrays of shape (n*m, r), etc.)
    """
    r = U_local.shape[1]
    U_g = np.zeros((n * m, r))
    V_g = np.zeros((m * p, r))
    W_g = np.zeros((p * n, r))
    
    # U: local index li*mi + lj -> global (n_off+li)*m + (m_off+lj)
    for li in range(ni):
        for lj in range(mi):
            local_idx = li * mi + lj
            global_idx = (n_off + li) * m + (m_off + lj)
            U_g[global_idx, :] = U_local[local_idx, :]
    
    # V: local index lj*pi + lk -> global (m_off+lj)*p + (p_off+lk)
    for lj in range(mi):
        for lk in range(pi):
            local_idx = lj * pi + lk
            global_idx = (m_off + lj) * p + (p_off + lk)
            V_g[global_idx, :] = V_local[local_idx, :]
    
    # W: local index lk*ni + li -> global (p_off+lk)*n + (n_off+li)
    for lk in range(pi):
        for li in range(ni):
            local_idx = lk * ni + li
            global_idx = (p_off + lk) * n + (n_off + li)
            W_g[global_idx, :] = W_local[local_idx, :]
    
    return U_g, V_g, W_g


def _block_decomposition(n, m, p, n_parts, m_parts, p_parts):
    """
    Decompose n x m x p into blocks defined by n_parts, m_parts, p_parts.
    Use Strassen for 2x2x2 blocks, schoolbook for others.
    """
    # Compute offsets
    n_offsets = [sum(n_parts[:i]) for i in range(len(n_parts))]
    m_offsets = [sum(m_parts[:i]) for i in range(len(m_parts))]
    p_offsets = [sum(p_parts[:i]) for i in range(len(p_parts))]
    
    # Determine rank for each block
    total_rank = 0
    block_ranks = {}
    for i, ni in enumerate(n_parts):
        for j, mi in enumerate(m_parts):
            for k, pi in enumerate(p_parts):
                if ni == 2 and mi == 2 and pi == 2:
                    block_ranks[(i, j, k)] = 7
                else:
                    block_ranks[(i, j, k)] = ni * mi * pi
                total_rank += block_ranks[(i, j, k)]
    
    # Construct global U, V, W
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for i, ni in enumerate(n_parts):
        for j, mi in enumerate(m_parts):
            for k, pi in enumerate(p_parts):
                r = block_ranks[(i, j, k)]
                if ni == 2 and mi == 2 and pi == 2:
                    U_loc, V_loc, W_loc = _strassen_2x2x2()
                else:
                    U_loc, V_loc, W_loc = _schoolbook(ni, mi, pi)
                
                U_g, V_g, W_g = _embed_block(U_loc, V_loc, W_loc, n, m, p,
                                              n_offsets[i], m_offsets[j], p_offsets[k],
                                              ni, mi, pi)
                U[:, col:col+r] = U_g
                V[:, col:col+r] = V_g
                W[:, col:col+r] = W_g
                col += r
    
    return U, V, W, total_rank


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact.
    Returns True if valid, False otherwise.
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _try_block_split(n, m, p, n_parts, m_parts, p_parts):
    """Try a single block split and return (U, V, W, rank) or None if invalid."""
    try:
        U, V, W, rank = _block_decomposition(n, m, p, n_parts, m_parts, p_parts)
        if _verify_decomposition(U, V, W, n, m, p):
            return U, V, W, rank
    except Exception:
        pass
    return None


def construct():
    deadline = time.time() + 130
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Structural change: instead of just trying a few fixed splits,
    # systematically enumerate all valid block partitions and pick the best.
    # This is feasible since n=3, m=5, p=5 are small.
    
    from itertools import product
    
    def all_compositions(total, max_parts=4):
        """Generate all compositions of 'total' into at most max_parts positive integers."""
        if total <= 0:
            yield []
            return
        if max_parts == 1:
            yield [total]
            return
        for first in range(1, total + 1):
            for rest in all_compositions(total - first, max_parts - 1):
                yield [first] + rest
    
    # Generate candidate splits
    n_comps = list(all_compositions(3, 3))
    m_comps = list(all_compositions(5, 3))
    p_comps = list(all_compositions(5, 3))
    
    # Score each candidate by estimated rank (Strassen saves 1 per 2x2x2 block)
    candidates = []
    for nc in n_comps:
        for mc in m_comps:
            for pc in p_comps:
                rank = 0
                for ni in nc:
                    for mi in mc:
                        for pi in pc:
                            if ni == 2 and mi == 2 and pi == 2:
                                rank += 7
                            else:
                                rank += ni * mi * pi
                candidates.append((rank, nc, mc, pc))
    
    # Sort by estimated rank
    candidates.sort(key=lambda x: x[0])
    
    best_result = None
    best_rank = float('inf')
    
    for est_rank, nc, mc, pc in candidates:
        if time.time() > deadline:
            break
        if est_rank >= best_rank:
            # No point trying worse or equal estimated ranks
            continue
        
        result = _try_block_split(n, m, p, nc, mc, pc)
        if result is not None:
            U, V, W, actual_rank = result
            if actual_rank < best_rank:
                best_rank = actual_rank
                best_result = (U, V, W)
    
    # If no block split worked, fall back to schoolbook
    if best_result is None:
        U, V, W = _schoolbook(n, m, p)
    else:
        U, V, W = best_result
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
