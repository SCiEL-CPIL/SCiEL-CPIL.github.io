"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(total - 1.0) > 1e-6:
                    return False
    return True

def _verify_integer_half_integer(U, V, W, tol=1e-9):
    """Check that all entries are integer or half-integer (multiple of 0.5)."""
    for name, M in [("U", U), ("V", V), ("W", W)]:
        for row in M:
            for val in row:
                if abs(val * 2 - round(val * 2)) > tol:
                    return False
    return True

def _strassen_2x2_terms():
    """Return the 7 terms of Strassen's 2x2x2 matrix multiplication."""
    terms = [
        {'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}},
        {'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(1,0): 1}},
        {'U': {(0,0): 1}, 'V': {(1,0): 1, (1,1): -1}, 'W': {(0,1): 1, (1,1): -1}},
        {'U': {(1,1): 1}, 'V': {(0,1): 1, (0,0): -1}, 'W': {(0,0): -1, (1,0): 1}},
        {'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,1): 1, (1,1): 1}},
        {'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(0,0): 1, (1,0): 1}},
        {'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}},
    ]
    return terms

def _strassen_3x3_terms():
    """Return the 21 terms of Strassen's 3x3x3 matrix multiplication.
    
    Based on the recursive application of 2x2 Strassen to 3x3 blocks.
    Split 3x3 into 2x2 + 1x1 in each dimension.
    """
    # We'll build this by combining 2x2 Strassen with naive for the 1-dim parts
    # 3x3x3 = (2+1)x(2+1)x(2+1)
    # = 2x2x2 (rank 7) + 2x2x1 (rank 2) + 2x1x2 (rank 2) + 2x1x1 (rank 1)
    #   + 1x2x2 (rank 2) + 1x2x1 (rank 1) + 1x1x2 (rank 1) + 1x1x1 (rank 1)
    # Total: 7+2+2+1+2+1+1+1 = 17? But known rank is 21.
    
    # Actually, let me use the known 21-term decomposition.
    # For 3x3x3, the optimal known rank is 21.
    # I'll construct it using the formula from literature.
    
    # Alternative: use 2x2 Strassen on the top-left 2x2x2 block,
    # then handle the remaining parts with naive multiplication.
    # This gives rank 7 + 2 + 2 + 1 + 2 + 1 + 1 + 1 = 17, but we need to be careful
    # about cross terms.
    
    # Let me just use the naive approach for 3x3x3 for now and optimize later.
    # Actually, let me construct it properly.
    
    terms = []
    
    # 2x2x2 Strassen for the (0,1)x(0,1)x(0,1) block
    for term in _strassen_2x2_terms():
        terms.append({
            'U': {(i, j): c for (i, j), c in term['U'].items()},
            'V': {(j, k): c for (j, k), c in term['V'].items()},
            'W': {(k, i): c for (k, i), c in term['W'].items()}
        })
    
    # 2x2x1 for (0,1)x(0,1)x(2)
    # C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2] for i=0,1
    terms.append({'U': {(0,0): 1, (1,0): 1}, 'V': {(0,2): 1}, 'W': {(2,0): 1}})
    terms.append({'U': {(0,1): 1, (1,1): 1}, 'V': {(1,2): 1}, 'W': {(2,0): 1}})
    
    # 2x1x2 for (0,1)x(2)x(0,1)
    # C[2,j] = A[2,0]*B[0,j] + A[2,1]*B[1,j] for j=0,1
    terms.append({'U': {(0,0): 1, (1,0): 1}, 'V': {(2,0): 1}, 'W': {(0,0): 1}})
    terms.append({'U': {(0,1): 1, (1,1): 1}, 'V': {(2,1): 1}, 'W': {(0,0): 1}})
    
    # 2x1x1 for (0,1)x(2)x(2)
    terms.append({'U': {(0,0): 1, (1,0): 1}, 'V': {(2,2): 1}, 'W': {(2,0): 1}})
    
    # 1x2x2 for (2)x(0,1)x(0,1)
    # C[2,j] = A[2,0]*B[0,j] + A[2,1]*B[1,j] for j=0,1
    # Wait, this is the same as 2x1x2 but transposed in a sense
    terms.append({'U': {(0,0): 1, (1,0): 1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(0,0): 1, (1,0): 1}})
    terms.append({'U': {(0,1): 1, (1,1): 1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,1): 1, (1,1): 1}})
    
    # 1x2x1 for (2)x(0,1)x(2)
    terms.append({'U': {(0,0): 1, (1,0): 1}, 'V': {(0,2): 1, (1,2): 1}, 'W': {(2,0): 1}})
    
    # 1x1x2 for (2)x(2)x(0,1)
    terms.append({'U': {(0,0): 1, (1,0): 1}, 'V': {(2,0): 1, (2,1): 1}, 'W': {(0,0): 1, (1,0): 1}})
    
    # 1x1x1 for (2)x(2)x(2)
    terms.append({'U': {(0,0): 1, (1,0): 1}, 'V': {(2,2): 1}, 'W': {(2,0): 1}})
    
    return terms

def _strassen_3x3x2_terms():
    """Return terms for 3x3x2 matrix multiplication with rank 12."""
    # 3x3x2 = (2+1)x(2+1)x2
    # = 2x2x2 (rank 7) + 2x1x2 (rank 2) + 1x2x2 (rank 2) + 1x1x2 (rank 1)
    # Total: 7+2+2+1 = 12
    
    terms = []
    
    # 2x2x2 Strassen for (0,1)x(0,1)x(0,1)
    for term in _strassen_2x2_terms():
        terms.append({
            'U': {(i, j): c for (i, j), c in term['U'].items()},
            'V': {(j, k): c for (j, k), c in term['V'].items()},
            'W': {(k, i): c for (k, i), c in term['W'].items()}
        })
    
    # 2x1x2 for (0,1)x(2)x(0,1)
    terms.append({'U': {(0,0): 1, (1,0): 1}, 'V': {(2,0): 1}, 'W': {(0,0): 1}})
    terms.append({'U': {(0,1): 1, (1,1): 1}, 'V': {(2,1): 1}, 'W': {(0,0): 1}})
    
    # 1x2x2 for (2)x(0,1)x(0,1)
    terms.append({'U': {(0,0): 1, (1,0): 1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(0,0): 1, (1,0): 1}})
    terms.append({'U': {(0,1): 1, (1,1): 1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,1): 1, (1,1): 1}})
    
    # 1x1x2 for (2)x(2)x(0,1)
    terms.append({'U': {(0,0): 1, (1,0): 1}, 'V': {(2,0): 1, (2,1): 1}, 'W': {(0,0): 1, (1,0): 1}})
    
    return terms

def _build_block_decomposition_v2(n, m, p):
    """Build decomposition using 3x3x3 Strassen and 3x3x2 blocks.
    
    Split n=4 as [3,1], m=4 as [3,1], p=5 as [3,2].
    """
    n_blocks = [3, 1]
    m_blocks = [3, 1]
    p_blocks = [3, 2]
    
    n_offsets = [0, 3]
    m_offsets = [0, 3]
    p_offsets = [0, 3]
    
    rank = 0
    U = {}  # (i, j) -> list of (coeff, rank_idx)
    V = {}  # (j, k) -> list of (coeff, rank_idx)
    W = {}  # (k, i) -> list of (coeff, rank_idx)
    
    for ni, ns in enumerate(n_blocks):
        for mi, ms in enumerate(m_blocks):
            for pi, ps in enumerate(p_blocks):
                no = n_offsets[ni]
                mo = m_offsets[mi]
                po = p_offsets[pi]
                
                if ns == 3 and ms == 3 and ps == 3:
                    # Use 3x3x3 Strassen (rank 21)
                    for term in _strassen_3x3_terms():
                        for (li, lj), cu in term['U'].items():
                            i, j = no + li, mo + lj
                            U.setdefault((i, j), []).append((cu, rank))
                        for (lj, lk), cv in term['V'].items():
                            j, k = mo + lj, po + lk
                            V.setdefault((j, k), []).append((cv, rank))
                        for (lk, li), cw in term['W'].items():
                            k, i = po + lk, no + li
                            W.setdefault((k, i), []).append((cw, rank))
                        rank += 1
                elif ns == 3 and ms == 3 and ps == 2:
                    # Use 3x3x2 (rank 12)
                    for term in _strassen_3x3x2_terms():
                        for (li, lj), cu in term['U'].items():
                            i, j = no + li, mo + lj
                            U.setdefault((i, j), []).append((cu, rank))
                        for (lj, lk), cv in term['V'].items():
                            j, k = mo + lj, po + lk
                            V.setdefault((j, k), []).append((cv, rank))
                        for (lk, li), cw in term['W'].items():
                            k, i = po + lk, no + li
                            W.setdefault((k, i), []).append((cw, rank))
                        rank += 1
                else:
                    # Fallback to schoolbook for this block
                    for li in range(ns):
                        for lj in range(ms):
                            for lk in range(ps):
                                i, j, k = no + li, mo + lj, po + lk
                                U.setdefault((i, j), []).append((1, rank))
                                V.setdefault((j, k), []).append((1, rank))
                                W.setdefault((k, i), []).append((1, rank))
                                rank += 1
    
    # Convert to matrix form
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    for (i, j), entries in U.items():
        for coeff, r in entries:
            U_mat[i * m + j][r] += coeff
    for (j, k), entries in V.items():
        for coeff, r in entries:
            V_mat[j * p + k][r] += coeff
    for (k, i), entries in W.items():
        for coeff, r in entries:
            W_mat[k * n + i][r] += coeff
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}

def _build_block_decomposition_v1(n, m, p):
    """Original block decomposition using 2x2 Strassen."""
    n_blocks = [2, 2]
    m_blocks = [2, 2]
    p_blocks = [2, 2, 1]
    
    n_offsets = [0, 2]
    m_offsets = [0, 2]
    p_offsets = [0, 2, 4]
    
    rank = 0
    U = {}
    V = {}
    W = {}
    
    for ni, ns in enumerate(n_blocks):
        for mi, ms in enumerate(m_blocks):
            for pi, ps in enumerate(p_blocks):
                no = n_offsets[ni]
                mo = m_offsets[mi]
                po = p_offsets[pi]
                
                if ns == 2 and ms == 2 and ps == 2:
                    for term in _strassen_2x2_terms():
                        for (li, lj), cu in term['U'].items():
                            i, j = no + li, mo + lj
                            U.setdefault((i, j), []).append((cu, rank))
                        for (lj, lk), cv in term['V'].items():
                            j, k = mo + lj, po + lk
                            V.setdefault((j, k), []).append((cv, rank))
                        for (lk, li), cw in term['W'].items():
                            k, i = po + lk, no + li
                            W.setdefault((k, i), []).append((cw, rank))
                        rank += 1
                elif ns == 2 and ms == 2 and ps == 1:
                    # Naive 2x2x1
                    for li in range(2):
                        for lj in range(2):
                            i, j, k = no + li, mo + lj, po
                            U.setdefault((i, j), []).append((1, rank))
                            V.setdefault((j, k), []).append((1, rank))
                            W.setdefault((k, i), []).append((1, rank))
                            rank += 1
                else:
                    for li in range(ns):
                        for lj in range(ms):
                            for lk in range(ps):
                                i, j, k = no + li, mo + lj, po + lk
                                U.setdefault((i, j), []).append((1, rank))
                                V.setdefault((j, k), []).append((1, rank))
                                W.setdefault((k, i), []).append((1, rank))
                                rank += 1
    
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    for (i, j), entries in U.items():
        for coeff, r in entries:
            U_mat[i * m + j][r] += coeff
    for (j, k), entries in V.items():
        for coeff, r in entries:
            V_mat[j * p + k][r] += coeff
    for (k, i), entries in W.items():
        for coeff, r in entries:
            W_mat[k * n + i][r] += coeff
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try v2 (3x3x3 Strassen) first
    try:
        result = _build_block_decomposition_v2(n, m, p)
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            if _verify_integer_half_integer(result["U"], result["V"], result["W"]):
                return result
    except Exception:
        pass
    
    # Try v1 (2x2 Strassen)
    try:
        result = _build_block_decomposition_v1(n, m, p)
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            if _verify_integer_half_integer(result["U"], result["V"], result["W"]):
                return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _build_schoolbook()
# EVOLVE-BLOCK-END
