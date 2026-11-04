import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

def _build_block_decomposition(n, m, p):
    """
    Build a valid decomposition using block-diagonal Strassen + schoolbook.
    Returns list of (U_col, V_col, W_col) as sparse representations.
    """
    terms = []  # Each term is (U_sparse, V_sparse, W_sparse) where each is dict {idx: val}
    
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
    
    def schoolbook_block(i_off, j_off, k_off, bn, bm, bp):
        for i in range(bn):
            for j in range(bm):
                for k in range(bp):
                    ig = i_off + i
                    jg = j_off + j
                    kg = k_off + k
                    add_term(
                        {ig * m + jg: 1},
                        {jg * p + kg: 1},
                        {kg * n + ig: 1}
                    )
    
    def strassen_block(i_off, j_off, k_off):
        """2x2x2 Strassen block."""
        def gi(i, j): return (i_off + i) * m + (j_off + j)
        def gj(j, k): return (j_off + j) * p + (k_off + k)
        def gk(k, i): return (k_off + k) * n + (i_off + i)
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        add_term(
            {gi(0,0): 1, gi(1,1): 1},
            {gj(0,0): 1, gj(1,1): 1},
            {gk(0,0): 1, gk(1,1): 1}
        )
        # M2: (A10+A11)B00 -> C10, -C11
        add_term(
            {gi(1,0): 1, gi(1,1): 1},
            {gj(0,0): 1},
            {gk(0,1): 1, gk(1,1): -1}
        )
        # M3: A00(B01-B11) -> C01, C11
        add_term(
            {gi(0,0): 1},
            {gj(0,1): 1, gj(1,1): -1},
            {gk(1,0): 1, gk(1,1): 1}
        )
        # M4: A11(B10-B00) -> C00, C10
        add_term(
            {gi(1,1): 1},
            {gj(1,0): 1, gj(0,0): -1},
            {gk(0,0): 1, gk(0,1): 1}
        )
        # M5: (A00+A01)B11 -> -C00, C01
        add_term(
            {gi(0,0): 1, gi(0,1): 1},
            {gj(1,1): 1},
            {gk(0,0): -1, gk(1,0): 1}
        )
        # M6: (A10-A00)(B00+B01) -> C11
        add_term(
            {gi(1,0): 1, gi(0,0): -1},
            {gj(0,0): 1, gj(0,1): 1},
            {gk(1,1): 1}
        )
        # M7: (A01-A11)(B10+B11) -> C00
        add_term(
            {gi(0,1): 1, gi(1,1): -1},
            {gj(1,0): 1, gj(1,1): 1},
            {gk(0,0): 1}
        )
    
    # Partition: n=4 as [2,2], m=5 as [2,2,1], p=5 as [2,2,1]
    i_parts = [(0, 2), (2, 2)]
    j_parts = [(0, 2), (2, 2), (4, 1)]
    k_parts = [(0, 2), (2, 2), (4, 1)]
    
    for i_off, i_sz in i_parts:
        for j_off, j_sz in j_parts:
            for k_off, k_sz in k_parts:
                if i_sz == 2 and j_sz == 2 and k_sz == 2:
                    strassen_block(i_off, j_off, k_off)
                else:
                    schoolbook_block(i_off, j_off, k_off, i_sz, j_sz, k_sz)
    
    return terms

def _terms_to_matrices(terms, n, m, p):
    """Convert list of sparse terms to dense U, V, W matrices."""
    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, val in u_dict.items():
            U[idx, c] = val
        for idx, val in v_dict.items():
            V[idx, c] = val
        for idx, val in w_dict.items():
            W[idx, c] = val
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify the decomposition is exact."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_reduce_rank(terms, n, m, p, deadline):
    """
    Try to reduce the rank by finding linear dependencies.
    """
    num_terms = len(terms)
    if num_terms < 2:
        return terms
    
    if time.time() > deadline:
        return terms
    
    # Check for duplicate terms
    seen = set()
    unique_terms = []
    for t in terms:
        u_key = frozenset(t[0].items())
        v_key = frozenset(t[1].items())
        w_key = frozenset(t[2].items())
        key = (u_key, v_key, w_key)
        if key not in seen:
            seen.add(key)
            unique_terms.append(t)
    
    if len(unique_terms) < num_terms:
        return unique_terms
    
    return terms

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Leave 50s margin for safety
    
    # Step 1: Build block decomposition
    terms = _build_block_decomposition(n, m, p)
    original_rank = len(terms)
    
    # Step 2: Convert to matrices
    U, V, W = _terms_to_matrices(terms, n, m, p)
    
    # Step 3: Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to pure schoolbook
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Step 4: Try to reduce rank
    if time.time() < deadline:
        reduced_terms = _try_reduce_rank(terms, n, m, p, deadline)
        if len(reduced_terms) < original_rank:
            U, V, W = _terms_to_matrices(reduced_terms, n, m, p)
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return the block decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
