"""Matrix multiplication 3x5x5 rank-minimal tensor decomposition."""
import time
import numpy as np
from itertools import product

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

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W correctly decompose the n x m x p matrix multiplication tensor.
    Uses einsum for speed: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    Returns True if correct.
    """
    rank = U.shape[1]
    # Build the full tensor via einsum
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # We need T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Reshape U to (n, m, rank)
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # Use einsum: 'ijc,jkc,kic->ijk'
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    
    # The matrix multiplication tensor T[i,j,k] should be 1 if i,j,k form a valid 
    # (row, col, inner) triple, else 0. Actually for matmul C = A*B where A is n x m, B is m x p,
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 for all valid i in [0,n), j in [0,m), k in [0,p)
    # Wait, that's not right. The tensor for matrix multiplication is:
    # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i,j,k in their respective ranges.
    
    # Actually, the standard matrix multiplication tensor T has T[i,j,k] = 1 for all 
    # i in [0,n), j in [0,m), k in [0,p). This is because:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T is defined such that C = T(A,B) where T contracts over j.
    # T[i,j,k] = 1 for all valid indices.
    
    expected = np.ones((n, m, p))
    
    return np.allclose(T, expected, atol=1e-10)
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
    W[:, 0] = [1, 0, 0, 1]  # C11=1, C21=0, C12=0, C22=1
    
    # M2 = (a21 + a22) b11
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 1, 0, -1]  # C11=0, C21=1, C12=0, C22=-1
    
    # M3 = a11 (b12 - b22)
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 1, 0, -1]
    W[:, 2] = [0, 0, 1, 1]  # C11=0, C21=0, C12=1, C22=1
    
    # M4 = a22 (b21 - b11)
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [-1, 0, 1, 0]
    W[:, 3] = [1, 1, 0, 0]  # C11=1, C21=1, C12=0, C22=0
    
    # M5 = (a11 + a12) b22
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [-1, 0, 1, 0]  # C11=-1, C21=0, C12=1, C22=0
    
    # M6 = (a21 - a11) (b11 + b12)
    U[:, 5] = [-1, 0, 1, 0]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [0, 0, 0, 1]  # C11=0, C21=0, C12=0, C22=1
    
    # M7 = (a12 - a22) (b21 + b22)
    U[:, 6] = [0, 1, 0, -1]
    V[:, 6] = [0, 0, 1, 1]
    W[:, 6] = [1, 0, 0, 0]  # C11=1, C21=0, C12=0, C22=0
    
    return U, V, W


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


def _embed_block_into(U, V, W, U_local, V_local, W_local, n, m, p, n_off, m_off, p_off, ni, mi, pi, col_offset):
    """
    Embed a local decomposition of size ni x mi x pi into global U, V, W arrays
    starting at column col_offset. Modifies U, V, W in place.
    """
    r = U_local.shape[1]
    
    # U: local index li*mi + lj -> global (n_off+li)*m + (m_off+lj)
    for li in range(ni):
        for lj in range(mi):
            local_idx = li * mi + lj
            global_idx = (n_off + li) * m + (m_off + lj)
            U[global_idx, col_offset:col_offset+r] = U_local[local_idx, :]
    
    # V: local index lj*pi + lk -> global (m_off+lj)*p + (p_off+lk)
    for lj in range(mi):
        for lk in range(pi):
            local_idx = lj * pi + lk
            global_idx = (m_off + lj) * p + (p_off + lk)
            V[global_idx, col_offset:col_offset+r] = V_local[local_idx, :]
    
    # W: local index lk*ni + li -> global (p_off+lk)*n + (n_off+li)
    for lk in range(pi):
        for li in range(ni):
            local_idx = lk * ni + li
            global_idx = (p_off + lk) * n + (n_off + li)
            W[global_idx, col_offset:col_offset+r] = W_local[local_idx, :]


_STRASSEN_CACHE = None

def _get_strassen():
    """Return cached Strassen 2x2x2 decomposition."""
    global _STRASSEN_CACHE
    if _STRASSEN_CACHE is None:
        _STRASSEN_CACHE = _strassen_2x2x2()
    return _STRASSEN_CACHE


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
    block_ranks = []
    for i, ni in enumerate(n_parts):
        for j, mi in enumerate(m_parts):
            for k, pi in enumerate(p_parts):
                if ni == 2 and mi == 2 and pi == 2:
                    block_ranks.append(7)
                else:
                    block_ranks.append(ni * mi * pi)
                total_rank += block_ranks[-1]
    
    # Construct global U, V, W
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    idx = 0
    for i, ni in enumerate(n_parts):
        for j, mi in enumerate(m_parts):
            for k, pi in enumerate(p_parts):
                r = block_ranks[idx]
                idx += 1
                if ni == 2 and mi == 2 and pi == 2:
                    U_loc, V_loc, W_loc = _get_strassen()
                else:
                    U_loc, V_loc, W_loc = _schoolbook(ni, mi, pi)
                
                _embed_block_into(U, V, W, U_loc, V_loc, W_loc, n, m, p,
                                  n_offsets[i], m_offsets[j], p_offsets[k],
                                  ni, mi, pi, col)
                col += r
    
    return U, V, W, total_rank


def _estimate_rank(n_parts, m_parts, p_parts):
    """Estimate the rank of a block decomposition."""
    total = 0
    for ni in n_parts:
        for mi in m_parts:
            for pi in p_parts:
                if ni == 2 and mi == 2 and pi == 2:
                    total += 7
                else:
                    total += ni * mi * pi
    return total


def _try_direct_optimization(n, m, p, deadline):
    """
    Try to find a better decomposition by exploring non-uniform block splits
    that exploit the rectangular structure. For 3x5x5, try splits like
    n=[1,2], m=[2,3], p=[2,3] and variants.
    """
    best_rank = float('inf')
    best_U = best_V = best_W = None
    
    # Try specific promising splits
    split_candidates = [
        # (n_parts, m_parts, p_parts)
        ([1, 2], [2, 3], [2, 3]),
        ([1, 2], [3, 2], [2, 3]),
        ([1, 2], [2, 3], [3, 2]),
        ([1, 2], [3, 2], [3, 2]),
        ([2, 1], [2, 3], [2, 3]),
        ([2, 1], [3, 2], [2, 3]),
        ([2, 1], [2, 3], [3, 2]),
        ([2, 1], [3, 2], [3, 2]),
        ([1, 1, 1], [2, 3], [2, 3]),
        ([1, 1, 1], [3, 2], [2, 3]),
        ([1, 1, 1], [2, 3], [3, 2]),
        ([1, 1, 1], [3, 2], [3, 2]),
        ([1, 2], [1, 1, 3], [2, 3]),
        ([1, 2], [1, 1, 3], [3, 2]),
        ([1, 2], [1, 3, 1], [2, 3]),
        ([1, 2], [1, 3, 1], [3, 2]),
        ([1, 2], [3, 1, 1], [2, 3]),
        ([1, 2], [3, 1, 1], [3, 2]),
        ([2, 1], [1, 1, 3], [2, 3]),
        ([2, 1], [1, 1, 3], [3, 2]),
        ([2, 1], [1, 3, 1], [2, 3]),
        ([2, 1], [1, 3, 1], [3, 2]),
        ([2, 1], [3, 1, 1], [2, 3]),
        ([2, 1], [3, 1, 1], [3, 2]),
    ]
    
    for n_parts, m_parts, p_parts in split_candidates:
        if time.time() > deadline:
            break
        try:
            est = _estimate_rank(n_parts, m_parts, p_parts)
            if est >= best_rank:
                continue
            U, V, W, rank = _block_decomposition(n, m, p, n_parts, m_parts, p_parts)
            if rank < best_rank:
                best_rank = rank
                best_U, best_V, best_W = U, V, W
        except Exception:
            continue
    
    if best_U is not None:
        return best_U, best_V, best_W, best_rank
    return None


def _generate_candidates():
    """Generate all candidate block splits for 3x5x5."""
    candidates = []
    
    # Generate all compositions of 3, 5, 5
    def compositions(n, max_parts=4):
        """Generate all compositions of n with at most max_parts parts."""
        if max_parts == 1:
            yield [n]
        return
        
    # For small numbers, just enumerate manually or use a helper
    def all_compositions(n, max_parts):
        if max_parts == 1:
            yield [n]
            return
        for first in range(1, n + 1):
            for rest in all_compositions(n - first, max_parts - 1):
                yield [first] + rest
    
    n_comps = list(all_compositions(3, 3))
    m_comps = list(all_compositions(5, 3))
    p_comps = list(all_compositions(5, 3))
    
    for nc in n_comps:
        for mc in m_comps:
            for pc in p_comps:
                candidates.append((nc, mc, pc))
    
    return candidates


def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # First try targeted optimizations
    result = _try_direct_optimization(n, m, p, deadline)
    if result is not None:
        best_U, best_V, best_W, best_rank = result
    else:
        best_U = best_V = best_W = None
        best_rank = float('inf')
    
    # Then try all candidate block splits
    candidates = _generate_candidates()
    
    # Sort by estimated rank (lower first)
    candidates.sort(key=lambda x: _estimate_rank(x[0], x[1], x[2]))
    
    for n_parts, m_parts, p_parts in candidates:
        if time.time() > deadline:
            break
        try:
            est_rank = _estimate_rank(n_parts, m_parts, p_parts)
            if est_rank >= best_rank:
                continue  # Skip if estimated rank is already worse than best found
            
            U, V, W, rank = _block_decomposition(n, m, p, n_parts, m_parts, p_parts)
            if rank < best_rank:
                best_rank = rank
                best_U, best_V, best_W = U, V, W
        except Exception:
            continue
    
    # Fallback to schoolbook if something went wrong
    if best_U is None:
        U, V, W = _schoolbook(n, m, p)
    else:
        U, V, W = best_U, best_V, best_W
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
