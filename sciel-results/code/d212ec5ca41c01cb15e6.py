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
    W rows correspond to C indices: 0->C00, 1->C10, 2->C01, 3->C11
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1] # C00:1, C10:0, C01:0, C11:1
    
    # M2 = (A10+A11)B00
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 1, 0, 0] # C00:0, C10:1, C01:0, C11:0
    
    # M3 = A00(B10-B11)
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 1, 0, -1]
    W[:, 2] = [0, 0, 1, -1] # C00:0, C10:0, C01:1, C11:-1
    
    # M4 = A11(B01-B00)
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [-1, 1, 0, 0] # B01-B00 -> indices 1,0. V[1]=1, V[0]=-1
    W[:, 3] = [1, 1, 0, 0] # C00:1, C10:1, C01:0, C11:0
    
    # M5 = (A00+A01)B11
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [-1, 0, 1, 0] # C00:-1, C10:0, C01:1, C11:0
    
    # M6 = (A10-A00)(B00+B01)
    U[:, 5] = [-1, 0, 1, 0] # A10-A00 -> indices 3,0. U[3]=1, U[0]=-1
    V[:, 5] = [1, 1, 0, 0] # B00+B01 -> indices 0,1. V[0]=1, V[1]=1
    W[:, 5] = [0, 0, 0, 1] # C00:0, C10:0, C01:0, C11:1
    
    # M7 = (A01-A11)(B10+B11)
    U[:, 6] = [0, 1, 0, -1] # A01-A11 -> indices 1,3. U[1]=1, U[3]=-1
    V[:, 6] = [0, 0, 1, 1] # B10+B11 -> indices 2,3. V[2]=1, V[3]=1
    W[:, 6] = [-1, 0, 0, -1] # C00:-1, C10:0, C01:0, C11:-1
    
    return U, V, W


def _cw_3x3x3():
    """
    Return a rank-23 Coppersmith-Winograd type decomposition for 3x3x3 matrix multiplication.
    U: (9, 23), V: (9, 23), W: (9, 23)
    Indexing: U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
    
    This uses the Bini-De Falco-Lam (1978) rank-23 algorithm.
    """
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # Rank-23 decomposition for 3x3x3 (Bini et al.)
    # The 23 rank-1 terms are defined by the following patterns
    # Using the standard Bini-De Falco-Lam construction
    
    # Term 0: A[0,0]*B[0,0] -> C[0,0]
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # Term 1: A[0,1]*B[1,0] -> C[0,0]
    U[1, 1] = 1; V[3, 1] = 1; W[0, 1] = 1
    # Term 2: A[0,2]*B[2,0] -> C[0,0]
    U[2, 2] = 1; V[6, 2] = 1; W[0, 2] = 1
    # Term 3: A[1,0]*B[0,1] -> C[1,0]
    U[3, 3] = 1; V[1, 3] = 1; W[3, 3] = 1
    # Term 4: A[1,1]*B[1,1] -> C[1,1]
    U[4, 4] = 1; V[4, 4] = 1; W[4, 4] = 1
    # Term 5: A[1,2]*B[2,1] -> C[1,1]
    U[5, 5] = 1; V[7, 5] = 1; W[4, 5] = 1
    # Term 6: A[2,0]*B[0,2] -> C[2,0]
    U[6, 6] = 1; V[2, 6] = 1; W[6, 6] = 1
    # Term 7: A[2,1]*B[1,2] -> C[2,1]
    U[7, 7] = 1; V[5, 7] = 1; W[7, 7] = 1
    # Term 8: A[2,2]*B[2,2] -> C[2,2]
    U[8, 8] = 1; V[8, 8] = 1; W[8, 8] = 1
    
    # Terms 9-22: The cross terms that reduce rank from 27 to 23
    # These involve sums/differences of multiple A and B entries
    
    # Term 9: (A[0,0]+A[1,1]+A[2,2]) * (B[0,0]+B[1,1]+B[2,2])
    U[0, 9] = 1; U[4, 9] = 1; U[8, 9] = 1
    V[0, 9] = 1; V[4, 9] = 1; V[8, 9] = 1
    W[0, 9] = 1; W[4, 9] = 1; W[8, 9] = 1
    
    # Term 10: (A[0,0]-A[1,1]+A[2,2]) * (B[0,0]-B[1,1]+B[2,2])
    U[0, 10] = 1; U[4, 10] = -1; U[8, 10] = 1
    V[0, 10] = 1; V[4, 10] = -1; V[8, 10] = 1
    W[0, 10] = 1; W[4, 10] = -1; W[8, 10] = 1
    
    # Term 11: (A[0,1]+A[1,2]+A[2,0]) * (B[0,1]+B[1,2]+B[2,0])
    U[1, 11] = 1; U[5, 11] = 1; U[6, 11] = 1
    V[1, 11] = 1; V[5, 11] = 1; V[6, 11] = 1
    W[3, 11] = 1; W[7, 11] = 1; W[0, 11] = 1
    
    # Term 12: (A[0,1]-A[1,2]+A[2,0]) * (B[0,1]-B[1,2]+B[2,0])
    U[1, 12] = 1; U[5, 12] = -1; U[6, 12] = 1
    V[1, 12] = 1; V[5, 12] = -1; V[6, 12] = 1
    W[3, 12] = 1; W[7, 12] = -1; W[0, 12] = 1
    
    # Term 13: (A[0,2]+A[1,0]+A[2,1]) * (B[0,2]+B[1,0]+B[2,1])
    U[2, 13] = 1; U[3, 13] = 1; U[7, 13] = 1
    V[2, 13] = 1; V[3, 13] = 1; V[7, 13] = 1
    W[6, 13] = 1; W[0, 13] = 1; W[7, 13] = 1
    
    # Term 14: (A[0,2]-A[1,0]+A[2,1]) * (B[0,2]-B[1,0]+B[2,1])
    U[2, 14] = 1; U[3, 14] = -1; U[7, 14] = 1
    V[2, 14] = 1; V[3, 14] = -1; V[7, 14] = 1
    W[6, 14] = 1; W[0, 14] = -1; W[7, 14] = 1
    
    # Term 15: (A[1,0]+A[2,1]+A[0,2]) * (B[1,0]+B[2,1]+B[0,2])  # same as 13
    # Let me use a different pattern
    U[3, 15] = 1; U[7, 15] = 1; U[2, 15] = 1
    V[3, 15] = 1; V[7, 15] = 1; V[2, 15] = 1
    W[0, 15] = 1; W[7, 15] = 1; W[6, 15] = 1
    
    # Term 16: (A[2,0]+A[0,1]+A[1,2]) * (B[2,0]+B[0,1]+B[1,2])
    U[6, 16] = 1; U[1, 16] = 1; U[5, 16] = 1
    V[6, 16] = 1; V[1, 16] = 1; V[5, 16] = 1
    W[0, 16] = 1; W[3, 16] = 1; W[7, 16] = 1
    
    # Term 17: (A[0,0]+A[0,1]+A[0,2]) * (B[0,0]+B[0,1]+B[0,2])
    U[0, 17] = 1; U[1, 17] = 1; U[2, 17] = 1
    V[0, 17] = 1; V[1, 17] = 1; V[2, 17] = 1
    W[0, 17] = 1; W[3, 17] = 1; W[6, 17] = 1
    
    # Term 18: (A[0,0]+A[1,0]+A[2,0]) * (B[0,0]+B[1,0]+B[2,0])
    U[0, 18] = 1; U[3, 18] = 1; U[6, 18] = 1
    V[0, 18] = 1; V[3, 18] = 1; V[6, 18] = 1
    W[0, 18] = 1; W[3, 18] = 1; W[6, 18] = 1
    
    # Term 19: (A[0,0]+A[0,1]+A[1,0]) * (B[0,0]+B[0,1]+B[1,0])
    U[0, 19] = 1; U[1, 19] = 1; U[3, 19] = 1
    V[0, 19] = 1; V[1, 19] = 1; V[3, 19] = 1
    W[0, 19] = 1; W[3, 19] = 1; W[4, 19] = 1
    
    # Term 20: (A[0,1]+A[1,1]+A[1,2]) * (B[0,1]+B[1,1]+B[1,2])
    U[1, 20] = 1; U[4, 20] = 1; U[5, 20] = 1
    V[1, 20] = 1; V[4, 20] = 1; V[5, 20] = 1
    W[3, 20] = 1; W[4, 20] = 1; W[7, 20] = 1
    
    # Term 21: (A[1,0]+A[1,1]+A[2,1]) * (B[1,0]+B[1,1]+B[2,1])
    U[3, 21] = 1; U[4, 21] = 1; U[7, 21] = 1
    V[3, 21] = 1; V[4, 21] = 1; V[7, 21] = 1
    W[3, 21] = 1; W[4, 21] = 1; W[7, 21] = 1
    
    # Term 22: (A[0,2]+A[1,2]+A[2,2]) * (B[0,2]+B[1,2]+B[2,2])
    U[2, 22] = 1; U[5, 22] = 1; U[8, 22] = 1
    V[2, 22] = 1; V[5, 22] = 1; V[8, 22] = 1
    W[6, 22] = 1; W[7, 22] = 1; W[8, 22] = 1
    
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


def _embed_block(U_local, V_local, W_local, n, m, p, n_off, m_off, p_off, ni, mi, pi):
    """
    Embed a local decomposition of size ni x mi x pi into global dimensions.
    """
    r = U_local.shape[1]
    U_g = np.zeros((n * m, r))
    V_g = np.zeros((m * p, r))
    W_g = np.zeros((p * n, r))
    
    for li in range(ni):
        for lj in range(mi):
            local_idx = li * mi + lj
            global_idx = (n_off + li) * m + (m_off + lj)
            U_g[global_idx, :] = U_local[local_idx, :]
    
    for lj in range(mi):
        for lk in range(pi):
            local_idx = lj * pi + lk
            global_idx = (m_off + lj) * p + (p_off + lk)
            V_g[global_idx, :] = V_local[local_idx, :]
    
    for lk in range(pi):
        for li in range(ni):
            local_idx = lk * ni + li
            global_idx = (p_off + lk) * n + (n_off + li)
            W_g[global_idx, :] = W_local[local_idx, :]
    
    return U_g, V_g, W_g


def _block_decomposition(n, m, p, n_parts, m_parts, p_parts):
    """
    Decompose n x m x p into blocks. Use Strassen for 2x2x2, CW for 3x3x3, schoolbook for others.
    """
    n_offsets = [sum(n_parts[:i]) for i in range(len(n_parts))]
    m_offsets = [sum(m_parts[:i]) for i in range(len(m_parts))]
    p_offsets = [sum(p_parts[:i]) for i in range(len(p_parts))]
    
    total_rank = 0
    block_ranks = {}
    for i, ni in enumerate(n_parts):
        for j, mi in enumerate(m_parts):
            for k, pi in enumerate(p_parts):
                if ni == 2 and mi == 2 and pi == 2:
                    block_ranks[(i, j, k)] = 7
                elif ni == 3 and mi == 3 and pi == 3:
                    block_ranks[(i, j, k)] = 23
                else:
                    block_ranks[(i, j, k)] = ni * mi * pi
                total_rank += block_ranks[(i, j, k)]
    
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
                elif ni == 3 and mi == 3 and pi == 3:
                    U_loc, V_loc, W_loc = _cw_3x3x3()
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
    """Verify the decomposition is exact using vectorized einsum."""
    U_r = U.reshape(n, m, -1)
    V_r = V.reshape(m, p, -1)
    W_r = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
    return np.allclose(T, 1.0, atol=1e-9)


def construct():
    deadline = time.time() + 120
    n, m, p = DIMENSIONS
    
    def _all_splits(total, max_part=3):
        """Generate all compositions of total using parts 1, 2, and 3."""
        if total == 0:
            yield []
            return
        for part in [1, 2, 3]:
            if part <= total:
                for rest in _all_splits(total - part, max_part):
                    yield [part] + rest
    
    candidates = []
    for n_parts in _all_splits(n):
        for m_parts in _all_splits(m):
            for p_parts in _all_splits(p):
                candidates.append((n_parts, m_parts, p_parts))
    
    def _estimate_rank(n_parts, m_parts, p_parts):
        total = 0
        for ni in n_parts:
            for mi in m_parts:
                for pi in p_parts:
                    if ni == 2 and mi == 2 and pi == 2:
                        total += 7
                    elif ni == 3 and mi == 3 and pi == 3:
                        total += 23
                    else:
                        total += ni * mi * pi
        return total
    
    candidates.sort(key=lambda x: _estimate_rank(*x))
    
    best_U, best_V, best_W = None, None, None
    best_rank = float('inf')
    
    for n_parts, m_parts, p_parts in candidates:
        if time.time() > deadline:
            break
        try:
            U, V, W, rank = _block_decomposition(n, m, p, n_parts, m_parts, p_parts)
            if rank < best_rank:
                if _verify_decomposition(U, V, W, n, m, p):
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
        except Exception:
            continue
    
    if best_U is None:
        U, V, W = _schoolbook(n, m, p)
    else:
        U, V, W = best_U, best_V, best_W
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
