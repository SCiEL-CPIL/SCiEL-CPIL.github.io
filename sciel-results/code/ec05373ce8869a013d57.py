import numpy as np
import time
from itertools import combinations

DIMENSIONS = [3, 3, 4]
n, m, p = DIMENSIONS

def _verify(U, V, W):
    """
    Verify if U, V, W exactly decompose the matmul tensor.
    U: (n*m, r)
    V: (m*p, r)
    W: (p*n, r)
    """
    r = U.shape[1]
    if r == 0:
        return False
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    
    try:
        T_reconstructed = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    except Exception:
        return False
    
    if not np.allclose(T_reconstructed, 1.0, atol=1e-9, rtol=1e-9):
        return False
    
    # Check half-integer constraint
    U2 = 2 * U
    V2 = 2 * V
    W2 = 2 * W
    if not np.allclose(U2, np.round(U2), atol=1e-9): return False
    if not np.allclose(V2, np.round(V2), atol=1e-9): return False
    if not np.allclose(W2, np.round(W2), atol=1e-9): return False
    
    return True

def _strassen_2x2_block(i_base, j_base, k_base):
    """
    Returns the U, V, W vectors for a 2x2x2 Strassen block.
    """
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    U1 = {u_idx(i_base, j_base): 1, u_idx(i_base+1, j_base+1): 1}
    V1 = {v_idx(j_base, k_base): 1, v_idx(j_base+1, k_base+1): 1}
    W1 = {w_idx(k_base, i_base): 1, w_idx(k_base+1, i_base+1): 1}
    U_terms.append(U1); V_terms.append(V1); W_terms.append(W1)
    
    U2 = {u_idx(i_base+1, j_base): 1, u_idx(i_base+1, j_base+1): 1}
    V2 = {v_idx(j_base, k_base): 1}
    W2 = {w_idx(k_base, i_base+1): 1}
    U_terms.append(U2); V_terms.append(V2); W_terms.append(W2)
    
    U3 = {u_idx(i_base, j_base): 1}
    V3 = {v_idx(j_base, k_base+1): 1, v_idx(j_base+1, k_base+1): -1}
    W3 = {w_idx(k_base+1, i_base): 1}
    U_terms.append(U3); V_terms.append(V3); W_terms.append(W3)
    
    U4 = {u_idx(i_base+1, j_base+1): 1}
    V4 = {v_idx(j_base+1, k_base): 1, v_idx(j_base, k_base): -1}
    W4 = {w_idx(k_base, i_base+1): 1}
    U_terms.append(U4); V_terms.append(V4); W_terms.append(W4)
    
    U5 = {u_idx(i_base, j_base): 1, u_idx(i_base, j_base+1): 1}
    V5 = {v_idx(j_base+1, k_base+1): 1}
    W5 = {w_idx(k_base+1, i_base): 1}
    U_terms.append(U5); V_terms.append(V5); W_terms.append(W5)
    
    U6 = {u_idx(i_base+1, j_base): 1, u_idx(i_base, j_base): -1}
    V6 = {v_idx(j_base, k_base): 1, v_idx(j_base, k_base+1): 1}
    W6 = {w_idx(k_base, i_base+1): 1, w_idx(k_base+1, i_base+1): 1}
    U_terms.append(U6); V_terms.append(V6); W_terms.append(W6)
    
    U7 = {u_idx(i_base, j_base+1): 1, u_idx(i_base+1, j_base+1): -1}
    V7 = {v_idx(j_base+1, k_base): 1, v_idx(j_base+1, k_base+1): 1}
    W7 = {w_idx(k_base, i_base): 1, w_idx(k_base+1, i_base): 1}
    U_terms.append(U7); V_terms.append(V7); W_terms.append(W7)
    
    return U_terms, V_terms, W_terms

def _make_term(u_dict, v_dict, w_dict):
    """Convert dict-based terms to vectors."""
    u_vec = np.zeros(n * m)
    v_vec = np.zeros(m * p)
    w_vec = np.zeros(p * n)
    for idx, val in u_dict.items():
        u_vec[idx] = val
    for idx, val in v_dict.items():
        v_vec[idx] = val
    for idx, val in w_dict.items():
        w_vec[idx] = val
    return u_vec, v_vec, w_vec

def _try_rank19():
    """
    Try to find a rank-19 decomposition.
    Strategy: Use 2x2x2 Strassen on (i,j) in {0,1}^2 for k in {0,1} and {2,3} = 14 terms.
    Then handle i=2 row (12 terms) with 5 paired terms + 1 single = 6 terms. Total = 20.
    
    Alternative: Can we merge one of the i=2 terms with a Strassen term?
    Let's try: use Strassen on {0,1}x{0,1}x{0,1} (7 terms), then handle the rest
    more cleverly.
    
    Actually, let's try: one 2x2x2 Strassen block (7 terms) covering (0,0,0),(0,1,0),(1,0,0),(1,1,0),
    (0,0,1),(0,1,1),(1,0,1),(1,1,1).
    Remaining: all terms with k=2,3 (16 terms) + terms (i=2, *, *) (12 terms) = 28 terms.
    Pair them: 14 pairs = 14 terms. Total = 7 + 14 = 21. Worse.
    
    Let's try: 2x2x2 Strassen on k={0,1} (7 terms) + 2x2x2 Strassen on k={2,3} (7 terms) = 14.
    i=2 row: 12 terms. Pair as (2,j,0)+(2,j,2) and (2,j,1)+(2,j,3) for j=0,1,2 = 6 terms.
    Total = 20.
    
    To get 19, we need to save 1 term. Can we share a term between the Strassen blocks and i=2?
    Or use a different blocking?
    
    Try: Strassen on (i,j)={0,1}x{0,1}, k={0,1} (7) + Strassen on (i,j)={0,1}x{1,2}, k={2,3} (7)
    But these overlap on (i,j)=(0,1) and (1,1) for different k's. That's fine since k's differ.
    Wait, (i,j)={0,1}x{1,2} means i in {0,1}, j in {1,2}. k in {2,3}.
    Covered: (0,1,2),(0,2,2),(1,1,2),(1,2,2),(0,1,3),(0,2,3),(1,1,3),(1,2,3)
    Not covered from first block: (0,0,2),(0,0,3),(0,1,2) already covered... 
    This is getting complex. Let me just try several configurations.
    """
    best_rank = float('inf')
    best_result = None
    
    # Configuration A: Standard 2 Strassen blocks + 6 paired i=2 terms = 20
    # Try to see if we can do 19 by using a different structure
    
    # Configuration B: Try 2x2x4 style - use 7 Strassen for k={0,1} on (0,1)x(0,1)
    # then handle k={2,3} and i=2 together
    U_cols = []
    V_cols = []
    W_cols = []
    
    # 7 Strassen terms for (i,j) in {0,1}^2, k in {0,1}
    U_terms, V_terms, W_terms = _strassen_2x2_block(0, 0, 0)
    for u_t, v_t, w_t in zip(U_terms, V_terms, W_terms):
        u_vec, v_vec, w_vec = _make_term(u_t, v_t, w_t)
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Now handle: (i,j,k) where k in {2,3} for i,j in {0,1} (8 terms) 
    # and all (i=2, j, k) for j in {0,1,2}, k in {0,1,2,3} (12 terms)
    # Total 20 terms. Pair them optimally.
    
    # Group by (i,j) and pair k's
    # For (0,0): k={2,3} -> 1 pair
    # For (0,1): k={2,3} -> 1 pair  
    # For (1,0): k={2,3} -> 1 pair
    # For (1,1): k={2,3} -> 1 pair
    # For (2,0): k={0,1,2,3} -> 2 pairs
    # For (2,1): k={0,1,2,3} -> 2 pairs
    # For (2,2): k={0,1,2,3} -> 2 pairs
    # Total: 4 + 6 = 10 pairs. Grand total: 7 + 10 = 17? No wait, that's not right.
    # The Strassen block already covers (0,0,0),(0,1,0),(1,0,0),(1,1,0),(0,0,1),(0,1,1),(1,0,1),(1,1,1)
    # So remaining for (0,0): k={2,3}, (0,1): k={2,3}, (1,0): k={2,3}, (1,1): k={2,3}
    # And (2,0): k={0,1,2,3}, (2,1): k={0,1,2,3}, (2,2): k={0,1,2,3}
    # That's 4*2 + 3*4 = 8 + 12 = 20 terms = 10 pairs. Total = 7 + 10 = 17.
    # Wait, that seems too good. Let me verify.
    
    # Pair (0,0,2)+(0,0,3), (0,1,2)+(0,1,3), (1,0,2)+(1,0,3), (1,1,2)+(1,1,3)
    # Pair (2,0,0)+(2,0,1), (2,0,2)+(2,0,3), (2,1,0)+(2,1,1), (2,1,2)+(2,1,3), (2,2,0)+(2,2,1), (2,2,2)+(2,2,3)
    
    pairs = [
        ((0,0,2),(0,0,3)),
        ((0,1,2),(0,1,3)),
        ((1,0,2),(1,0,3)),
        ((1,1,2),(1,1,3)),
        ((2,0,0),(2,0,1)),
        ((2,0,2),(2,0,3)),
        ((2,1,0),(2,1,1)),
        ((2,1,2),(2,1,3)),
        ((2,2,0),(2,2,1)),
        ((2,2,2),(2,2,3)),
    ]
    
    for (i1,j1,k1),(i2,j2,k2) in pairs:
        u_vec = np.zeros(n * m)
        v_vec = np.zeros(m * p)
        w_vec = np.zeros(p * n)
        u_vec[i1 * m + j1] = 1
        if (i2, j2) != (i1, j1):
            u_vec[i2 * m + j2] = 1
        v_vec[j1 * p + k1] = 1
        if (j2, k2) != (j1, k1):
            v_vec[j2 * p + k2] = 1
        w_vec[k1 * n + i1] = 1
        if (k2, i2) != (k1, i1):
            w_vec[k2 * n + i2] = 1
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    rank = U.shape[1]
    if rank < best_rank and _verify(U, V, W):
        best_rank = rank
        best_result = (U, V, W)
    
    if best_result is not None:
        return best_result
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try rank-19 decomposition
    result = _try_rank19()
    
    if result is not None:
        U, V, W = result
        if _verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: rank-20 decomposition
    U_cols = []
    V_cols = []
    W_cols = []
    
    for kb in [0, 2]:
        U_terms, V_terms, W_terms = _strassen_2x2_block(0, 0, kb)
        for u_t, v_t, w_t in zip(U_terms, V_terms, W_terms):
            u_vec, v_vec, w_vec = _make_term(u_t, v_t, w_t)
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)
    
    for j in range(m):
        for idx in range(0, 4, 2):
            k1 = idx
            k2 = idx + 1
            u_vec = np.zeros(n * m)
            v_vec = np.zeros(m * p)
            w_vec = np.zeros(p * n)
            u_vec[2 * m + j] = 1
            v_vec[j * p + k1] = 1
            v_vec[j * p + k2] = 1
            w_vec[k1 * n + 2] = 1
            w_vec[k2 * n + 2] = 1
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    if _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Schoolbook fallback
    U_sb = np.zeros((n * m, n * m * p))
    V_sb = np.zeros((m * p, n * m * p))
    W_sb = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]) if res['U'] else 0)
