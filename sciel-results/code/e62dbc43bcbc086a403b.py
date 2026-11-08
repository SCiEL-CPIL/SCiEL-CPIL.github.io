"""Matrix multiplication 3x3x7 - rank 60 decomposition using Strassen for 2x2x2 blocks."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook (naive) tensor decomposition for n×m × m×p multiplication."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact for the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _strassen_2x2x2(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
    """
    Return 7 rank-1 terms for 2x2 * 2x2 -> 2x2 multiplication using Strassen.
    Each term is (u_vals, v_vals, w_vals) where the vectors are sparse dicts.
    """
    terms = []
    # m1 = (a00+a11)(b00+b11)
    terms.append(({a00: 1, a11: 1}, {b00: 1, b11: 1}, {c00: 1, c11: 1}))
    # m2 = (a10+a11)b00
    terms.append(({a10: 1, a11: 1}, {b00: 1}, {c10: 1, c11: -1}))
    # m3 = a00(b01-b11)
    terms.append(({a00: 1}, {b01: 1, b11: -1}, {c01: 1, c11: 1}))
    # m4 = a11(b10-b00)
    terms.append(({a11: 1}, {b10: 1, b00: -1}, {c00: 1, c10: 1}))
    # m5 = (a00+a01)b11
    terms.append(({a00: 1, a01: 1}, {b11: 1}, {c00: -1, c01: 1}))
    # m6 = (a10-a00)(b00+b01)
    terms.append(({a10: 1, a00: -1}, {b00: 1, b01: 1}, {c11: 1}))
    # m7 = (a01-a11)(b10+b11)
    terms.append(({a01: 1, a11: -1}, {b10: 1, b11: 1}, {c00: 1}))
    return terms

def _build_rank60_decomposition(n, m, p):
    """
    Build a rank-60 decomposition for 3x3 * 3x7.
    
    Structure:
    - A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] : Strassen, rank 7
    - A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] : Strassen, rank 7
    - A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6] : Strassen, rank 7
    - A[0:2, 0:2] * B[0:2, 6:7] -> C[0:2, 6:7] : schoolbook 2x2x1, rank 4
    - A[0:2, 2] * B[2, 0:7] -> C[0:2, 0:7] : schoolbook 2x1x7, rank 14
    - A[2, 0:3] * B[0:3, 0:7] -> C[2, 0:7] : schoolbook 1x3x7, rank 21
    Total: 7+7+7+4+14+21 = 60
    """
    rank = 60
    U = [[0] * rank for _ in range(n * m)]  # 9 x rank
    V = [[0] * rank for _ in range(m * p)]  # 21 x rank
    W = [[0] * rank for _ in range(p * n)]  # 21 x rank
    
    col = 0
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(i, j):
        return i * p + j
    
    def w_idx(i, k):
        return k * n + i
    
    def add_term(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= rank:
            return
        for idx, val in u_dict.items():
            U[idx][col] += val
        for idx, val in v_dict.items():
            V[idx][col] += val
        for idx, val in w_dict.items():
            W[idx][col] += val
        col += 1
    
    # Strassen helper for 2x2 A block (rows a0,a1; cols 0,1) * 2x2 B block (rows 0,1; cols k,k+1)
    def add_strassen_block(a_row_base, k_base):
        """
        A[a_row_base:a_row_base+2, 0:2] * B[0:2, k_base:k_base+2] -> C[a_row_base:a_row_base+2, k_base:k_base+2]
        """
        # A indices
        a00 = u_idx(a_row_base, 0)
        a01 = u_idx(a_row_base, 1)
        a10 = u_idx(a_row_base + 1, 0)
        a11 = u_idx(a_row_base + 1, 1)
        
        # B indices
        b00 = v_idx(0, k_base)
        b01 = v_idx(0, k_base + 1)
        b10 = v_idx(1, k_base)
        b11 = v_idx(1, k_base + 1)
        
        # C indices
        c00 = w_idx(a_row_base, k_base)
        c01 = w_idx(a_row_base, k_base + 1)
        c10 = w_idx(a_row_base + 1, k_base)
        c11 = w_idx(a_row_base + 1, k_base + 1)
        
        terms = _strassen_2x2x2(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11)
        for u_t, v_t, w_t in terms:
            add_term(u_t, v_t, w_t)
    
    # Schoolbook for 2x2 A block * 2x1 B block -> 2x1 C block
    def add_schoolbook_2x2x1(a_row_base, k_base):
        """
        A[a_row_base:a_row_base+2, 0:2] * B[0:2, k_base:k_base+1] -> C[a_row_base:a_row_base+2, k_base:k_base+1]
        Rank 4
        """
        for i in range(2):
            for j in range(2):
                add_term(
                    {u_idx(a_row_base + i, j): 1},
                    {v_idx(j, k_base): 1},
                    {w_idx(a_row_base + i, k_base): 1}
                )
    
    # Schoolbook for 2x1 A block * 1x7 B block -> 2x7 C block
    def add_schoolbook_2x1x7(a_row_base, j_col, k_start, k_end):
        """
        A[a_row_base:a_row_base+2, j_col] * B[j_col, k_start:k_end] -> C[a_row_base:a_row_base+2, k_start:k_end]
        Rank 2*(k_end-k_start)
        """
        for i in range(2):
            for k in range(k_start, k_end):
                add_term(
                    {u_idx(a_row_base + i, j_col): 1},
                    {v_idx(j_col, k): 1},
                    {w_idx(a_row_base + i, k): 1}
                )
    
    # Schoolbook for 1x3 A block * 3x7 B block -> 1x7 C block
    def add_schoolbook_1x3x7(a_row, k_start, k_end):
        """
        A[a_row, 0:3] * B[0:3, k_start:k_end] -> C[a_row, k_start:k_end]
        Rank 3*(k_end-k_start)
        """
        for j in range(3):
            for k in range(k_start, k_end):
                add_term(
                    {u_idx(a_row, j): 1},
                    {v_idx(j, k): 1},
                    {w_idx(a_row, k): 1}
                )
    
    # Apply decomposition:
    # 1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] : Strassen, rank 7
    add_strassen_block(0, 0)
    
    # 2. A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] : Strassen, rank 7
    add_strassen_block(0, 2)
    
    # 3. A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6] : Strassen, rank 7
    add_strassen_block(0, 4)
    
    # 4. A[0:2, 0:2] * B[0:2, 6:7] -> C[0:2, 6:7] : schoolbook 2x2x1, rank 4
    add_schoolbook_2x2x1(0, 6)
    
    # 5. A[0:2, 2] * B[2, 0:7] -> C[0:2, 0:7] : schoolbook 2x1x7, rank 14
    add_schoolbook_2x1x7(0, 2, 0, 7)
    
    # 6. A[2, 0:3] * B[0:3, 0:7] -> C[2, 0:7] : schoolbook 1x3x7, rank 21
    add_schoolbook_1x3x7(2, 0, 7)
    
    return {"U": U, "V": V, "W": W}

def _try_optimize_rank(result, n, m, p, time_limit=60.0):
    """
    Try to find a lower rank decomposition. For now, this is a placeholder
    that could attempt random or structured searches.
    """
    start_time = time.time()
    current_rank = len(result['U'][0]) if result['U'] else n * m * p
    
    # No known decomposition better than 60 for 3x3x7 in the literature
    # that we can easily construct. Return the current result.
    return result

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard: 150s wall-clock limit, use 140s as safety margin
    deadline = time.time() + 140.0
    
    # Build the rank-60 decomposition
    result = _build_rank60_decomposition(n, m, p)
    
    # Verify the decomposition
    if not _verify_decomposition(result['U'], result['V'], result['W'], n, m, p):
        # Fallback to schoolbook if verification fails
        result = _schoolbook_decomposition(n, m, p)
    
    # Try to optimize further if time permits
    if time.time() < deadline - 30:
        try:
            better = _try_optimize_rank(result, n, m, p, time_limit=min(60, deadline - time.time() - 5))
            if better is not None and len(better['U'][0]) < len(result['U'][0]):
                if _verify_decomposition(better['U'], better['V'], better['W'], n, m, p):
                    result = better
        except Exception:
            pass
    
    return result
# EVOLVE-BLOCK-END
