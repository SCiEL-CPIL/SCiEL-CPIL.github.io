import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

def _strassen_2x2x2(i_offset, j_offset, k_offset):
    """
    Returns list of (u_vec, v_vec, w_vec) for Strassen 2x2x2.
    """
    def gI(i_l, j_l): return (i_offset + i_l) * 5 + (j_offset + j_l)
    def gJ(j_l, k_l): return (j_offset + j_l) * 5 + (k_offset + k_l)
    def gK(k_l, i_l): return (k_offset + k_l) * 4 + (i_offset + i_l)
    
    terms = []
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    terms.append((
        [(gI(0,0), 1), (gI(1,1), 1)],
        [(gJ(0,0), 1), (gJ(1,1), 1)],
        [(gK(0,0), 1), (gK(1,1), 1)]
    ))
    
    # M2: (A10+A11)B00 -> C10, -C11
    terms.append((
        [(gI(1,0), 1), (gI(1,1), 1)],
        [(gJ(0,0), 1)],
        [(gK(0,1), 1), (gK(1,1), -1)]
    ))
    
    # M3: A00(B01-B11) -> C01, C11
    terms.append((
        [(gI(0,0), 1)],
        [(gJ(0,1), 1), (gJ(1,1), -1)],
        [(gK(1,0), 1), (gK(1,1), 1)]
    ))
    
    # M4: A11(B10-B00) -> C00, C10
    terms.append((
        [(gI(1,1), 1)],
        [(gJ(1,0), 1), (gJ(0,0), -1)],
        [(gK(0,0), 1), (gK(0,1), 1)]
    ))
    
    # M5: (A00+A01)B11 -> -C00, C01
    terms.append((
        [(gI(0,0), 1), (gI(0,1), 1)],
        [(gJ(1,1), 1)],
        [(gK(0,0), -1), (gK(1,0), 1)]
    ))
    
    # M6: (A10-A00)(B00+B01) -> C11
    terms.append((
        [(gI(1,0), 1), (gI(0,0), -1)],
        [(gJ(0,0), 1), (gJ(0,1), 1)],
        [(gK(1,1), 1)]
    ))
    
    # M7: (A01-A11)(B10+B11) -> C00
    terms.append((
        [(gI(0,1), 1), (gI(1,1), -1)],
        [(gJ(1,0), 1), (gJ(1,1), 1)],
        [(gK(0,0), 1)]
    ))
    
    return terms

def _schoolbook(i_size, j_size, k_size, i_offset, j_offset, k_offset):
    """
    Returns list of (u_vec, v_vec, w_vec) for schoolbook.
    """
    terms = []
    for i_loc in range(i_size):
        for j_loc in range(j_size):
            for k_loc in range(k_size):
                i_glob = i_offset + i_loc
                j_glob = j_offset + j_loc
                k_glob = k_offset + k_loc
                
                I = i_glob * 5 + j_glob
                J = j_glob * 5 + k_glob
                K = k_glob * 4 + i_glob
                
                terms.append(([(I, 1)], [(J, 1)], [(K, 1)]))
    return terms

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Partitioning:
    # I: [0,1], [2,3]
    # J: [0,1], [2,3], [4]
    # K: [0,1], [2,3], [4]
    
    i_groups = [(0, 2), (2, 2)]
    j_groups = [(0, 2), (2, 2), (4, 1)]
    k_groups = [(0, 2), (2, 2), (4, 1)]
    
    all_terms = []
    
    for i_start, i_size in i_groups:
        if time.time() > deadline:
            break
        for j_start, j_size in j_groups:
            if time.time() > deadline:
                break
            for k_start, k_size in k_groups:
                if time.time() > deadline:
                    break
                if i_size == 2 and j_size == 2 and k_size == 2:
                    terms = _strassen_2x2x2(i_start, j_start, k_start)
                else:
                    terms = _schoolbook(i_size, j_size, k_size, i_start, j_start, k_start)
                
                for t in terms:
                    all_terms.append(t)
    
    if not all_terms:
        return {"U": [[0]*100 for _ in range(20)], "V": [[0]*100 for _ in range(25)], "W": [[0]*100 for _ in range(20)]}
    
    # Construct matrices
    U = np.zeros((n_m, len(all_terms)))
    V = np.zeros((m_p, len(all_terms)))
    W = np.zeros((p_n, len(all_terms)))
    
    for idx, (u_vec, v_vec, w_vec) in enumerate(all_terms):
        for i, val in u_vec:
            U[i, idx] = val
        for j, val in v_vec:
            V[j, idx] = val
        for k, val in w_vec:
            W[k, idx] = val
    
    # Check for linear dependence in columns to reduce rank
    # We can try to remove columns that are linear combinations of others.
    # Since the entries are small integers, we can use Gaussian elimination over rationals (Fractions)
    # or just float with tolerance, but float is risky for exactness.
    # However, the task requires exact decomposition.
    # A column c is redundant if it is in the span of the previous columns.
    # We can maintain a basis of the column space.
    
    # To ensure speed, we'll only attempt reduction if the rank is high enough and we have time.
    # Current rank is 98.
    # Let's try to reduce.
    
    # We will keep a list of active columns indices.
    active_indices = list(range(len(all_terms)))
    
    # We can't easily do exact Gaussian elimination on the full 20x98, 25x98, 20x98 system simultaneously
    # to find the minimal rank subset, because the "rank" of the tensor decomposition is not just
    # the column rank of U, V, or W individually. It's the minimal r such that U, V, W exist.
    # If we remove a column from U, V, and W, we must ensure the decomposition still holds.
    # The decomposition holds if sum_{l=1}^r u_l v_l w_l^T = T.
    # If column l is a linear combination of other columns in U, i.e., u_l = sum c_i u_i,
    # then we can't just remove it unless we can adjust the other columns.
    # Actually, if u_l = sum c_i u_i, we can replace v_l with v_l - sum c_i v_i ? No.
    # The term is u_l v_l w_l^T.
    # If we have a set of terms, we can combine them if the u-vectors are linearly dependent.
    # Suppose u_1 = u_2. Then u_1 v_1 w_1^T + u_2 v_2 w_2^T = u_1 (v_1 w_1^T + v_2 w_2^T).
    # This doesn't reduce rank unless the resulting v or w becomes zero or dependent in a way that allows further reduction.
    # Generally, rank reduction of a tensor is hard.
    
    # However, there is a trivial reduction: if a column in U is zero, we can remove it.
    # In our construction, no columns are zero.
    
    # Let's check for duplicate columns in U, V, W simultaneously?
    # If (u_i, v_i, w_i) is a scalar multiple of (u_j, v_j, w_j), we can combine them.
    # But Strassen terms are distinct.
    
    # Given the time constraints and complexity, and that the current score is 1.087 (rank 98),
    # and the theoretical minimum is around 48-50, there is significant room for improvement.
    # But implementing BCL-R is complex.
    
    # Let's stick to the current result for now, as it is valid and fast.
    # If we have time, we could try to identify if any Strassen blocks can be merged.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
