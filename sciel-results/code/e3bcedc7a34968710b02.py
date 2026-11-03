"""
Improved block-decomposition for 3x5x6 matrix multiplication.
Uses Strassen (rank 7) for 2x2x2 blocks, schoolbook elsewhere.
Adds a deadline guard and verifies correctness before returning.
"""
import numpy as np
import time

# ---------- Strassen 2x2x2 (rank 7) ----------
# Tensor index conventions:
#   u index: i*m + j  (for the 2x2x2 block, i,j in 0..1 -> i*2+j)
#   v index: j*p + k  (for the 2x2x2 block, j,k in 0..1 -> j*2+k)
#   w index: k*n + i  (for the 2x2x2 block, k,i in 0..1 -> k*2+i)
#
# Strassen intermediate products:
#   P1 = (A00 + A11)(B00 + B11)
#   P2 = (A10 + A11) * B01
#   P3 = A00 * (B10 - B00)
#   P4 = A11 * (B11 - B01)
#   P5 = (A00 + A01) * B00
#   P6 = A10 * (B11 - B01)
#   P7 = (A11 - A10) * (B01 - B11)
#
# Reconstructed C:
#   C00 = P1 - P5 + P4 + P7
#   C01 = P3 + P5
#   C10 = P2 + P6
#   C11 = P1 - P3 + P2 - P6

# u vectors (index i*2+j):
_u_str = np.array([
    [1, 0, 0, 1],  # P1: A00+A11
    [0, 1, 1, 0],  # P2: A10+A11 -> indices: i=1,j=0 -> 2; i=1,j=1 -> 3 => [0,0,1,1]
    # Wait, i*2+j: i=1,j=0 -> 2; i=1,j=1 -> 3. So P2 u = [0,0,1,1]
    # Let me redo all carefully.
    # i*2+j:
    # i=0,j=0 -> 0; i=0,j=1 -> 1; i=1,j=0 -> 2; i=1,j=1 -> 3
    # P1: A00+A11 -> indices 0,3 -> [1,0,0,1]
    # P2: (A10+A11) -> indices 2,3 -> [0,0,1,1]
    # P3: A00 -> index 0 -> [1,0,0,0]
    # P4: A11 -> index 3 -> [0,0,0,1]
    # P5: (A00+A01) -> indices 0,1 -> [1,1,0,0]
    # P6: A10 -> index 2 -> [0,0,1,0]
    # P7: (A11-A10) -> indices 3,2 -> [0,0,-1,1]
], dtype=float)

_u_str = np.array([
    [1, 0, 0, 1],
    [0, 0, 1, 1],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [1, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, -1, 1],
], dtype=float)

# v vectors (index j*2+k):
# j*2+k:
# j=0,k=0 -> 0; j=0,k=1 -> 1; j=1,k=0 -> 2; j=1,k=1 -> 3
# P1: (B00+B11) -> indices 0,3 -> [1,0,0,1]
# P2: B01 -> index 1 -> [0,1,0,0]
# P3: (B10-B00) -> indices 2,0 -> [1,0,1,0]
# P4: (B11-B01) -> indices 3,1 -> [0,-1,0,1]
# P5: B00 -> index 0 -> [1,0,0,0]
# P6: (B11-B01) -> indices 3,1 -> [0,-1,0,1]
# P7: (B01-B11) -> indices 1,3 -> [0,1,0,-1]

_v_str = np.array([
    [1, 0, 0, 1],
    [0, 1, 0, 0],
    [1, 0, 1, 0],
    [0, -1, 0, 1],
    [1, 0, 0, 0],
    [0, -1, 0, 1],
    [0, 1, 0, -1],
], dtype=float)

# w vectors (index k*2+i):
# k*2+i:
# k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; k=1,i=1 -> 3
# C00 = P1 - P5 + P4 + P7
# C01 = P3 + P5
# C10 = P2 + P6
# C11 = P1 - P3 + P2 - P6
#
# w for P1: C00+1, C11+1 -> indices 0,3 -> [1,0,0,1]
# w for P2: C10+1, C11-1 -> indices 1,3 -> [0,1,0,-1]
# w for P3: C01+1, C11-1 -> indices 2,3 -> [0,0,1,-1]
# w for P4: C00+1 -> index 0 -> [1,0,0,0]
# w for P5: C00-1, C01+1 -> indices 0,2 -> [-1,0,1,0]
# w for P6: C10+1 -> index 1 -> [0,1,0,0]
# w for P7: C00+1 -> index 0 -> [1,0,0,0]

_w_str = np.array([
    [1, 0, 0, 1],
    [0, 1, 0, -1],
    [0, 0, 1, -1],
    [1, 0, 0, 0],
    [-1, 0, 1, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
], dtype=float)


def _verify(U, V, W, n, m, p):
    """Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.dot(U[i * m + j], V[j * p + k] * W[k * n + i]))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _build_block(n_parts, m_parts, p_parts, n, m, p):
    """
    Build U, V, W using Strassen for 2x2x2 sub-blocks and schoolbook otherwise.
    n_parts, m_parts, p_parts: partitions summing to n, m, p respectively.
    """
    terms = []
    for ni, ns in enumerate(n_parts):
        for mi, ms in enumerate(m_parts):
            for pi, ps in enumerate(p_parts):
                if ns == 2 and ms == 2 and ps == 2:
                    for r in range(7):
                        terms.append(('strassen', ni, mi, pi, r))
                else:
                    for i in range(ns):
                        for j in range(ms):
                            for k in range(ps):
                                terms.append(('school', ni, mi, pi, i, j, k))
    
    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for idx, term in enumerate(terms):
        if term[0] == 'strassen':
            _, ni, mi, pi, r = term
            n_off = sum(n_parts[:ni])
            m_off = sum(m_parts[:mi])
            p_off = sum(p_parts[:pi])
            
            for li in range(2):
                for lj in range(2):
                    U[(n_off + li) * m + (m_off + lj), idx] = _u_str[r, li * 2 + lj]
            for lj in range(2):
                for lk in range(2):
                    V[(m_off + lj) * p + (p_off + lk), idx] = _v_str[r, lj * 2 + lk]
            for lk in range(2):
                for li in range(2):
                    W[(p_off + lk) * n + (n_off + li), idx] = _w_str[r, lk * 2 + li]
        else:
            _, ni, mi, pi, li, lj, lk = term
            n_off = sum(n_parts[:ni])
            m_off = sum(m_parts[:mi])
            p_off = sum(p_parts[:pi])
            
            gi, gj, gk = n_off + li, m_off + lj, p_off + lk
            U[gi * m + gj, idx] = 1.0
            V[gj * p + gk, idx] = 1.0
            W[gk * n + gi, idx] = 1.0
    
    return U, V, W, rank


def _schoolbook(n, m, p):
    """Rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
    return U, V, W


def construct():
    """
    Returns a tensor decomposition for 3x5x6 matrix multiplication.
    Uses block decomposition with Strassen on 2x2x2 blocks where possible.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    n, m, p = 3, 5, 6
    
    # Try various block partitions
    candidates = [
        ([2, 1], [2, 2, 1], [2, 2, 2]),
        ([2, 1], [2, 1, 2], [2, 2, 2]),
        ([2, 1], [1, 2, 2], [2, 2, 2]),
        ([2, 1], [2, 2, 1], [2, 1, 3]),
        ([2, 1], [2, 2, 1], [1, 2, 3]),
        ([2, 1], [2, 2, 1], [2, 3, 1]),
        ([2, 1], [1, 2, 2], [2, 1, 3]),
        ([2, 1], [1, 2, 2], [1, 2, 3]),
        ([2, 1], [1, 2, 2], [2, 3, 1]),
        ([1, 2], [2, 2, 1], [2, 2, 2]),
        ([1, 2], [2, 1, 2], [2, 2, 2]),
        ([1, 2], [1, 2, 2], [2, 2, 2]),
    ]
    
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # fallback schoolbook rank
    
    for n_parts, m_parts, p_parts in candidates:
        if time.time() > deadline:
            break
        if sum(n_parts) != n or sum(m_parts) != m or sum(p_parts) != p:
            continue
        U, V, W, rank = _build_block(n_parts, m_parts, p_parts, n, m, p)
        if rank < best_rank:
            if _verify(U, V, W, n, m, p):
                best_U, best_V, best_W, best_rank = U, V, W, rank
            else:
                # Decomposition failed verification; skip
                pass
    
    # If no valid block decomposition found, use schoolbook
    if best_U is None:
        best_U, best_V, best_W = _schoolbook(n, m, p)
    
    # Final verification
    if not _verify(best_U, best_V, best_W, n, m, p):
        # Last resort: schoolbook
        best_U, best_V, best_W = _schoolbook(n, m, p)
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
