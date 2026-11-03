import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U: shape (4, 7) - indexed by (i, j)
    V: shape (4, 7) - indexed by (j, k)
    W: shape (4, 7) - indexed by (k, i)
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = A00 * B01
    U[0, 0] = 1; V[1, 0] = 1
    W[1, 0] = 1; W[1, 1] = 1; W[3, 0] = 1; W[3, 1] = 1
    
    # M2 = A10 * (B00 + B01)
    U[2, 1] = 1; V[0, 1] = 1; V[1, 1] = 1
    W[2, 0] = 1; W[3, 0] = 1
    
    # M3 = (A00 + A01) * B10
    U[0, 2] = 1; U[1, 2] = 1; V[2, 2] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M4 = (A10 - A00) * (B11 - B00)
    U[2, 3] = 1; U[0, 3] = -1; V[3, 3] = 1; V[0, 3] = -1
    W[0, 0] = 1; W[1, 1] = 1; W[2, 1] = 1; W[3, 0] = 1
    
    # M5 = (A00 + A10) * B00
    U[0, 4] = 1; U[2, 4] = 1; V[0, 4] = 1
    W[0, 0] = 1; W[1, 0] = 1
    
    # M6 = A01 * (B10 - B00)
    U[1, 5] = 1; V[2, 5] = 1; V[0, 5] = -1
    W[0, 1] = 1; W[1, 1] = 1
    
    # M7 = (A11 - A01) * (B00 + B10)
    U[3, 6] = 1; U[1, 6] = -1; V[0, 6] = 1; V[2, 6] = 1
    W[0, 0] = 1; W[2, 0] = 1; W[3, 1] = 1
    
    return U, V, W

def _naive_decomp(n, m, p):
    """Return U, V, W for n x m x p matrix multiplication using naive method.
    U: shape (n*m, n*m*p)
    V: shape (m*p, n*m*p)
    W: shape (p*n, n*m*p)
    """
    r = n * m * p
    U = np.zeros((n*m, r), dtype=int)
    V = np.zeros((m*p, r), dtype=int)
    W = np.zeros((p*n, r), dtype=int)
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, c] = 1
                V[j*p + k, c] = 1
                W[k*n + i, c] = 1
                c += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Split dimensions into blocks
    # n = 3 -> [2, 1]
    # m = 5 -> [2, 2, 1]
    # p = 5 -> [2, 2, 1]
    
    n_blocks = [2, 1]
    m_blocks = [2, 2, 1]
    p_blocks = [2, 2, 1]
    
    # Precompute offsets
    n_offsets = [0]
    for i in range(len(n_blocks) - 1):
        n_offsets.append(n_offsets[-1] + n_blocks[i])
    
    m_offsets = [0]
    for i in range(len(m_blocks) - 1):
        m_offsets.append(m_offsets[-1] + m_blocks[i])
    
    p_offsets = [0]
    for i in range(len(p_blocks) - 1):
        p_offsets.append(p_offsets[-1] + p_blocks[i])
    
    # First pass: compute total rank
    total_rank = 0
    for r in range(len(n_blocks)):
        for c in range(len(m_blocks)):
            for k in range(len(p_blocks)):
                ns = n_blocks[r]
                ms = m_blocks[c]
                ps = p_blocks[k]
                if ns == 2 and ms == 2 and ps == 2:
                    total_rank += 7
                else:
                    total_rank += ns * ms * ps
    
    # Allocate arrays
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    col = 0
    for r in range(len(n_blocks)):
        for c in range(len(m_blocks)):
            for k in range(len(p_blocks)):
                ns = n_blocks[r]
                ms = m_blocks[c]
                ps = p_blocks[k]
                
                if ns == 2 and ms == 2 and ps == 2:
                    U_sub, V_sub, W_sub = _strassen_2x2x2()
                    sub_rank = 7
                else:
                    U_sub, V_sub, W_sub = _naive_decomp(ns, ms, ps)
                    sub_rank = ns * ms * ps
                
                # Add to global arrays
                for i in range(ns):
                    for j in range(ms):
                        for t in range(sub_rank):
                            val = U_sub[i * ms + j, t]
                            if val != 0:
                                U[n_offsets[r] + i, m_offsets[c] + j] * 0  # no-op
                                U[(n_offsets[r] + i) * m + (m_offsets[c] + j), col + t] += val
                
                for j in range(ms):
                    for kk in range(ps):
                        for t in range(sub_rank):
                            val = V_sub[j * ps + kk, t]
                            if val != 0:
                                V[(m_offsets[c] + j) * p + (p_offsets[k] + kk), col + t] += val
                
                for kk in range(ps):
                    for i in range(ns):
                        for t in range(sub_rank):
                            val = W_sub[kk * ns + i, t]
                            if val != 0:
                                W[(p_offsets[k] + kk) * n + (n_offsets[r] + i), col + t] += val
                
                col += sub_rank
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
