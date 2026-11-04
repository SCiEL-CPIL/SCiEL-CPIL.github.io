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


def construct():
    deadline = time.time() + 130  # Safety margin before 150s limit
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Strategy: Use block decomposition with multiple splits and pick the best rank.
    # We'll try several different block splits and return the one with lowest rank.
    
    candidates = []
    
    # Split 1: n=[1,2], m=[2,3], p=[2,3]
    # Blocks: (1,2,2)=4, (1,3,3)=9, (2,2,2)=7, (2,3,3)=18 -> total 38
    candidates.append(([1, 2], [2, 3], [2, 3]))
    
    # Split 2: n=[1,2], m=[3,2], p=[3,2]
    # Blocks: (1,3,3)=9, (1,2,2)=4, (2,3,3)=18, (2,2,2)=7 -> total 38
    candidates.append(([1, 2], [3, 2], [3, 2]))
    
    # Split 3: n=[2,1], m=[2,3], p=[2,3]
    # Blocks: (2,2,2)=7, (2,3,3)=18, (1,2,2)=4, (1,3,3)=9 -> total 38
    candidates.append(([2, 1], [2, 3], [2, 3]))
    
    # Split 4: n=[1,1,1], m=[2,3], p=[2,3]
    # 6 blocks: (1,2,2)=4, (1,3,3)=9, (1,2,2)=4, (1,3,3)=9, (1,2,2)=4, (1,3,3)=9 -> total 39
    candidates.append(([1, 1, 1], [2, 3], [2, 3]))
    
    # Split 6: n=[1,2], m=[2,1,2], p=[2,1,2]
    # i=0: (1,2,2)=4, (1,1,1)=1, (1,2,2)=4
    # i=1: (2,2,2)=7, (2,1,1)=2, (2,2,2)=7
    # Total: 4+1+4+7+2+7 = 25
    candidates.append(([1, 2], [2, 1, 2], [2, 1, 2]))
    
    # Split 7: n=[1,2], m=[1,2,2], p=[1,2,2]
    # i=0: (1,1,1)=1, (1,2,2)=4, (1,2,2)=4
    # i=1: (2,1,1)=2, (2,2,2)=7, (2,2,2)=7
    # Total: 1+4+4+2+7+7 = 25
    candidates.append(([1, 2], [1, 2, 2], [1, 2, 2]))
    
    # Split 8: n=[1,1,1], m=[1,2,2], p=[1,2,2]
    # 9 blocks: 1+4+4 + 1+4+4 + 1+4+4 = 27
    candidates.append(([1, 1, 1], [1, 2, 2], [1, 2, 2]))
    
    # Split 9: n=[1,2], m=[2,2,1], p=[2,2,1]
    # i=0: (1,2,2)=4, (1,2,2)=4, (1,1,1)=1
    # i=1: (2,2,2)=7, (2,2,2)=7, (2,1,1)=2
    # Total: 4+4+1+7+7+2 = 25
    candidates.append(([1, 2], [2, 2, 1], [2, 2, 1]))
    
    # Split 10: n=[1,2], m=[1,1,3], p=[1,1,3]
    # i=0: (1,1,1)=1, (1,1,1)=1, (1,3,3)=9
    # i=1: (2,1,1)=2, (2,1,1)=2, (2,3,3)=18
    # Total: 1+1+9+2+2+18 = 33
    candidates.append(([1, 2], [1, 1, 3], [1, 1, 3]))
    
    # New splits to explore lower rank:
    
    # Split 11: n=[1,1,1], m=[2,2,1], p=[2,2,1]
    # 9 blocks: each row of n has blocks (1,2,2)=4, (1,2,2)=4, (1,1,1)=1
    # Total: 3*(4+4+1) = 27
    candidates.append(([1, 1, 1], [2, 2, 1], [2, 2, 1]))
    
    # Split 12: n=[1,1,1], m=[1,2,2], p=[2,2,1]
    # i=0: (1,1,2)=2, (1,2,2)=4, (1,2,1)=2
    # i=1: (1,1,2)=2, (1,2,2)=4, (1,2,1)=2
    # i=2: (1,1,2)=2, (1,2,2)=4, (1,2,1)=2
    # Total: 3*(2+4+2) = 24
    candidates.append(([1, 1, 1], [1, 2, 2], [2, 2, 1]))
    
    # Split 13: n=[1,1,1], m=[2,2,1], p=[1,2,2]
    # i=0: (1,2,1)=2, (1,2,2)=4, (1,1,2)=2
    # i=1: (1,2,1)=2, (1,2,2)=4, (1,1,2)=2
    # i=2: (1,2,1)=2, (1,2,2)=4, (1,1,2)=2
    # Total: 3*(2+4+2) = 24
    candidates.append(([1, 1, 1], [2, 2, 1], [1, 2, 2]))
    
    # Split 14: n=[1,1,1], m=[1,1,3], p=[1,1,3]
    # i=0: (1,1,1)=1, (1,1,1)=1, (1,3,3)=9
    # i=1: (1,1,1)=1, (1,1,1)=1, (1,3,3)=9
    # i=2: (1,1,1)=1, (1,1,1)=1, (1,3,3)=9
    # Total: 3*(1+1+9) = 33
    candidates.append(([1, 1, 1], [1, 1, 3], [1, 1, 3]))
    
    # Split 15: n=[1,2], m=[2,1,2], p=[1,2,2]
    # i=0: (1,2,1)=2, (1,1,2)=2, (1,2,2)=4
    # i=1: (2,2,1)=4, (2,1,2)=4, (2,2,2)=7
    # Total: 2+2+4+4+4+7 = 23
    candidates.append(([1, 2], [2, 1, 2], [1, 2, 2]))
    
    # Split 16: n=[1,2], m=[1,2,2], p=[2,1,2]
    # i=0: (1,1,2)=2, (1,2,1)=2, (1,2,2)=4
    # i=1: (2,1,2)=4, (2,2,1)=4, (2,2,2)=7
    # Total: 2+2+4+4+4+7 = 23
    candidates.append(([1, 2], [1, 2, 2], [2, 1, 2]))
    
    # Split 17: n=[2,1], m=[2,1,2], p=[1,2,2]
    # i=0: (2,2,1)=4, (2,1,2)=4, (2,2,2)=7
    # i=1: (1,2,1)=2, (1,1,2)=2, (1,2,2)=4
    # Total: 4+4+7+2+2+4 = 23
    candidates.append(([2, 1], [2, 1, 2], [1, 2, 2]))
    
    # Split 18: n=[2,1], m=[1,2,2], p=[2,1,2]
    # i=0: (2,1,2)=4, (2,2,1)=4, (2,2,2)=7
    # i=1: (1,1,2)=2, (1,2,1)=2, (1,2,2)=4
    # Total: 4+4+7+2+2+4 = 23
    candidates.append(([2, 1], [1, 2, 2], [2, 1, 2]))
    
    # Split 19: n=[1,1,1], m=[2,1,2], p=[2,1,2]
    # 9 blocks: each (1,2,2)=4, (1,1,1)=1, (1,2,2)=4
    # Total: 3*(4+1+4) = 27
    candidates.append(([1, 1, 1], [2, 1, 2], [2, 1, 2]))
    
    # Split 20: n=[1,2], m=[2,2,1], p=[1,2,2]
    # i=0: (1,2,1)=2, (1,2,2)=4, (1,1,2)=2
    # i=1: (2,2,1)=4, (2,2,2)=7, (2,1,2)=4
    # Total: 2+4+2+4+7+4 = 23
    candidates.append(([1, 2], [2, 2, 1], [1, 2, 2]))
    
    # Split 21: n=[1,2], m=[1,2,2], p=[2,2,1]
    # i=0: (1,1,2)=2, (1,2,2)=4, (1,2,1)=2
    # i=1: (2,1,2)=4, (2,2,2)=7, (2,2,1)=4
    # Total: 2+4+2+4+7+4 = 23
    candidates.append(([1, 2], [1, 2, 2], [2, 2, 1]))
    
    # Split 22: n=[2,1], m=[2,2,1], p=[1,2,2]
    # i=0: (2,2,1)=4, (2,2,2)=7, (2,1,2)=4
    # i=1: (1,2,1)=2, (1,2,2)=4, (1,1,2)=2
    # Total: 4+7+4+2+4+2 = 23
    candidates.append(([2, 1], [2, 2, 1], [1, 2, 2]))
    
    # Split 23: n=[2,1], m=[1,2,2], p=[2,2,1]
    # i=0: (2,1,2)=4, (2,2,2)=7, (2,2,1)=4
    # i=1: (1,1,2)=2, (1,2,2)=4, (1,2,1)=2
    # Total: 4+7+4+2+4+2 = 23
    candidates.append(([2, 1], [1, 2, 2], [2, 2, 1]))
    
    # Split 24: n=[1,1,1], m=[1,2,2], p=[1,2,2]
    # Already covered by split 8, skip
    
    # Split 25: n=[1,2], m=[1,1,3], p=[1,2,2]
    # i=0: (1,1,1)=1, (1,1,2)=2, (1,3,2)=6
    # i=1: (2,1,1)=2, (2,1,2)=4, (2,3,2)=12
    # Total: 1+2+6+2+4+12 = 27
    candidates.append(([1, 2], [1, 1, 3], [1, 2, 2]))
    
    # Split 26: n=[1,2], m=[1,2,2], p=[1,1,3]
    # i=0: (1,1,1)=1, (1,2,1)=2, (1,2,3)=6
    # i=1: (2,1,1)=2, (2,2,1)=4, (2,2,3)=12
    # Total: 1+2+6+2+4+12 = 27
    candidates.append(([1, 2], [1, 2, 2], [1, 1, 3]))
    
    # Split 27: n=[1,1,1], m=[1,1,3], p=[1,2,2]
    # i=0: (1,1,1)=1, (1,1,2)=2, (1,3,2)=6
    # i=1: (1,1,1)=1, (1,1,2)=2, (1,3,2)=6
    # i=2: (1,1,1)=1, (1,1,2)=2, (1,3,2)=6
    # Total: 3*(1+2+6) = 27
    candidates.append(([1, 1, 1], [1, 1, 3], [1, 2, 2]))
    
    # Split 28: n=[1,1,1], m=[1,2,2], p=[1,1,3]
    # i=0: (1,1,1)=1, (1,2,1)=2, (1,2,3)=6
    # i=1: (1,1,1)=1, (1,2,1)=2, (1,2,3)=6
    # i=2: (1,1,1)=1, (1,2,1)=2, (1,2,3)=6
    # Total: 3*(1+2+6) = 27
    candidates.append(([1, 1, 1], [1, 2, 2], [1, 1, 3]))
    
    # Split 29: n=[1,2], m=[2,1,2], p=[2,2,1]
    # i=0: (1,2,2)=4, (1,1,2)=2, (1,2,1)=2
    # i=1: (2,2,2)=7, (2,1,2)=4, (2,2,1)=4
    # Total: 4+2+2+7+4+4 = 23
    candidates.append(([1, 2], [2, 1, 2], [2, 2, 1]))
    
    # Split 30: n=[1,2], m=[1,2,2], p=[1,2,2]
    # Already covered by split 7, skip
    
    # Split 31: n=[2,1], m=[2,1,2], p=[2,2,1]
    # i=0: (2,2,2)=7, (2,1,2)=4, (2,2,1)=4
    # i=1: (1,2,2)=4, (1,1,2)=2, (1,2,1)=2
    # Total: 7+4+4+4+2+2 = 23
    candidates.append(([2, 1], [2, 1, 2], [2, 2, 1]))
    
    # Split 32: n=[2,1], m=[1,2,2], p=[1,2,2]
    # i=0: (2,1,1)=2, (2,2,2)=7, (2,2,2)=7
    # i=1: (1,1,1)=1, (1,2,2)=4, (1,2,2)=4
    # Total: 2+7+7+1+4+4 = 25
    candidates.append(([2, 1], [1, 2, 2], [1, 2, 2]))
    
    # Split 33: n=[1,2], m=[1,2,2], p=[2,1,2]
    # Already covered by split 16, skip
    
    # Split 34: n=[1,2], m=[2,2,1], p=[2,1,2]
    # i=0: (1,2,2)=4, (1,2,1)=2, (1,1,2)=2
    # i=1: (2,2,2)=7, (2,2,1)=4, (2,1,2)=4
    # Total: 4+2+2+7+4+4 = 23
    candidates.append(([1, 2], [2, 2, 1], [2, 1, 2]))
    
    # Split 35: n=[2,1], m=[2,2,1], p=[2,1,2]
    # i=0: (2,2,2)=7, (2,2,1)=4, (2,1,2)=4
    # i=1: (1,2,2)=4, (1,2,1)=2, (1,1,2)=2
    # Total: 7+4+4+4+2+2 = 23
    candidates.append(([2, 1], [2, 2, 1], [2, 1, 2]))
    
    # Split 36: n=[1,1,1], m=[2,2,1], p=[1,1,3]
    # i=0: (1,2,1)=2, (1,2,1)=2, (1,1,3)=3
    # i=1: (1,2,1)=2, (1,2,1)=2, (1,1,3)=3
    # i=2: (1,2,1)=2, (1,2,1)=2, (1,1,3)=3
    # Total: 3*(2+2+3) = 21
    candidates.append(([1, 1, 1], [2, 2, 1], [1, 1, 3]))
    
    # Split 37: n=[1,1,1], m=[1,1,3], p=[2,2,1]
    # i=0: (1,1,2)=2, (1,1,2)=2, (1,3,1)=3
    # i=1: (1,1,2)=2, (1,1,2)=2, (1,3,1)=3
    # i=2: (1,1,2)=2, (1,1,2)=2, (1,3,1)=3
    # Total: 3*(2+2+3) = 21
    candidates.append(([1, 1, 1], [1, 1, 3], [2, 2, 1]))
    
    # Split 38: n=[1,1,1], m=[2,1,2], p=[1,1,3]
    # i=0: (1,2,1)=2, (1,1,1)=1, (1,2,3)=6
    # i=1: (1,2,1)=2, (1,1,1)=1, (1,2,3)=6
    # i=2: (1,2,1)=2, (1,1,1)=1, (1,2,3)=6
    # Total: 3*(2+1+6) = 27
    candidates.append(([1, 1, 1], [2, 1, 2], [1, 1, 3]))
    
    # Split 39: n=[1,1,1], m=[1,1,3], p=[2,1,2]
    # i=0: (1,1,2)=2, (1,1,1)=1, (1,3,2)=6
    # i=1: (1,1,2)=2, (1,1,1)=1, (1,3,2)=6
    # i=2: (1,1,2)=2, (1,1,1)=1, (1,3,2)=6
    # Total: 3*(2+1+6) = 27
    candidates.append(([1, 1, 1], [1, 1, 3], [2, 1, 2]))
    
    # Split 40: n=[1,2], m=[2,1,2], p=[1,1,3]
    # i=0: (1,2,1)=2, (1,1,1)=1, (1,2,3)=6
    # i=1: (2,2,1)=4, (2,1,1)=2, (2,2,3)=12
    # Total: 2+1+6+4+2+12 = 27
    candidates.append(([1, 2], [2, 1, 2], [1, 1, 3]))
    
    # Split 41: n=[1,2], m=[1,1,3], p=[2,1,2]
    # i=0: (1,1,2)=2, (1,1,1)=1, (1,3,2)=6
    # i=1: (2,1,2)=4, (2,1,1)=2, (2,3,2)=12
    # Total: 2+1+6+4+2+12 = 27
    candidates.append(([1, 2], [1, 1, 3], [2, 1, 2]))
    
    # Split 42: n=[2,1], m=[2,1,2], p=[1,1,3]
    # i=0: (2,2,1)=4, (2,1,1)=2, (2,2,3)=12
    # i=1: (1,2,1)=2, (1,1,1)=1, (1,2,3)=6
    # Total: 4+2+12+2+1+6 = 27
    candidates.append(([2, 1], [2, 1, 2], [1, 1, 3]))
    
    # Split 43: n=[2,1], m=[1,1,3], p=[2,1,2]
    # i=0: (2,1,2)=4, (2,1,1)=2, (2,3,2)=12
    # i=1: (1,1,2)=2, (1,1,1)=1, (1,3,2)=6
    # Total: 4+2+12+2+1+6 = 27
    candidates.append(([2, 1], [1, 1, 3], [2, 1, 2]))
    
    # Split 44: n=[1,1,1], m=[1,2,2], p=[1,1,3]
    # Already covered by split 28, skip
    
    # Split 45: n=[1,1,1], m=[1,1,3], p=[1,2,2]
    # Already covered by split 27, skip
    
    # Split 46: n=[1,2], m=[1,1,3], p=[1,1,3]
    # i=0: (1,1,1)=1, (1,1,1)=1, (1,3,3)=9
    # i=1: (2,1,1)=2, (2,1,1)=2, (2,3,3)=18
    # Total: 1+1+9+2+2+18 = 33
    candidates.append(([1, 2], [1, 1, 3], [1, 1, 3]))
    
    # Split 47: n=[2,1], m=[1,1,3], p=[1,1,3]
    # i=0: (2,1,1)=2, (2,1,1)=2, (2,3,3)=18
    # i=1: (1,1,1)=1, (1,1,1)=1, (1,3,3)=9
    # Total: 2+2+18+1+1+9 = 33
    candidates.append(([2, 1], [1, 1, 3], [1, 1, 3]))
    
    # Split 48: n=[1,1,1], m=[2,2,1], p=[2,1,2]
    # i=0: (1,2,2)=4, (1,2,1)=2, (1,1,2)=2
    # i=1: (1,2,2)=4, (1,2,1)=2, (1,1,2)=2
    # i=2: (1,2,2)=4, (1,2,1)=2, (1,1,2)=2
    # Total: 3*(4+2+2) = 24
    candidates.append(([1, 1, 1], [2, 2, 1], [2, 1, 2]))
    
    # Split 49: n=[1,1,1], m=[1,2,2], p=[2,2,1]
    # Already covered by split 12, skip
    
    # Split 50: n=[1,1,1], m=[2,1,2], p=[2,2,1]
    # i=0: (1,2,2)=4, (1,1,2)=2, (1,2,1)=2
    # i=1: (1,2,2)=4, (1,1,2)=2, (1,2,1)=2
    # i=2: (1,2,2)=4, (1,1,2)=2, (1,2,1)=2
    # Total: 3*(4+2+2) = 24
    candidates.append(([1, 1, 1], [2, 1, 2], [2, 2, 1]))
    
    # Split 51: n=[1,1,1], m=[1,2,2], p=[1,2,2]
    # Already covered by split 8, skip
    
    # Split 52: n=[1,1,1], m=[2,2,1], p=[1,2,2]
    # Already covered by split 13, skip
    
    # Split 53: n=[1,1,1], m=[1,1,3], p=[1,1,3]
    # Already covered by split 14, skip
    
    # Split 54: n=[1,1,1], m=[1,1,3], p=[2,1,2]
    # Already covered by split 39, skip
    
    # Split 55: n=[1,1,1], m=[2,1,2], p=[1,2,2]
    # i=0: (1,2,1)=2, (1,1,2)=2, (1,2,2)=4
    # i=1: (1,2,1)=2, (1,1,2)=2, (1,2,2)=4
    # i=2: (1,2,1)=2, (1,1,2)=2, (1,2,2)=4
    # Total: 3*(2+2+4) = 24
    candidates.append(([1, 1, 1], [2, 1, 2], [1, 2, 2]))
    
    # Split 56: n=[1,1,1], m=[1,2,2], p=[2,1,2]
    # i=0: (1,1,2)=2, (1,2,1)=2, (1,2,2)=4
    # i=1: (1,1,2)=2, (1,2,1)=2, (1,2,2)=4
    # i=2: (1,1,2)=2, (1,2,1)=2, (1,2,2)=4
    # Total: 3*(2+2+4) = 24
    candidates.append(([1, 1, 1], [1, 2, 2], [2, 1, 2]))
    
    # Split 57: n=[1,1,1], m=[1,1,3], p=[1,2,2]
    # Already covered by split 27, skip
    
    # Split 58: n=[1,1,1], m=[1,2,2], p=[1,1,3]
    # Already covered by split 28, skip
    
    # Try all candidates and pick the best
    best_U, best_V, best_W = None, None, None
    best_rank = float('inf')
    
    # Sort candidates by estimated rank (lower first) to find good solutions early
    def estimate_rank(n_parts, m_parts, p_parts):
        total = 0
        for ni in n_parts:
            for mi in m_parts:
                for pi in p_parts:
                    if ni == 2 and mi == 2 and pi == 2:
                        total += 7
                    else:
                        total += ni * mi * pi
        return total
    
    sorted_candidates = sorted(candidates, key=lambda x: estimate_rank(x[0], x[1], x[2]))
    
    for n_parts, m_parts, p_parts in sorted_candidates:
        if time.time() > deadline:
            break
        try:
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
