"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _rank1_decomposition(n, m, p):
    """Schoolbook rank n*m*p decomposition for n x m * m * p."""
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
    return U, V, W

def _decompose_tensor_to_rank1(U, V, W):
    """
    Convert matrix decomposition back to rank-1 tensor terms.
    Each column c gives a rank-1 term with support on U[:,c], V[:,c], W[:,c].
    """
    n_rows_U = len(U)
    m_rows_V = len(V)
    p_rows_W = len(W)
    rank = len(U[0]) if U else 0
    
    terms = []
    for c in range(rank):
        u_nonzero = [(i, U[i][c]) for i in range(n_rows_U) if U[i][c] != 0]
        v_nonzero = [(j, V[j][c]) for j in range(m_rows_V) if V[j][c] != 0]
        w_nonzero = [(k, W[k][c]) for k in range(p_rows_W) if W[k][c] != 0]
        if u_nonzero and v_nonzero and w_nonzero:
            terms.append((u_nonzero, v_nonzero, w_nonzero))
    return terms

def _term_support(term):
    """Get the set of (U_idx, V_idx, W_idx) tuples with nonzero values in a term."""
    u_set = set(u[0] for u in term[0])
    v_set = set(v[0] for v in term[1])
    w_set = set(w[0] for w in term[2])
    return u_set, v_set, w_set

def _check_term_redundant(term, other_terms, n_rows_U, m_rows_V, p_rows_W):
    """
    Check if a term can be expressed as a linear combination of other terms.
    Returns True if redundant, False otherwise.
    """
    u_dict = {u[0]: u[1] for u in term[0]}
    v_dict = {v[0]: v[1] for v in term[1]}
    w_dict = {w[0]: w[1] for w in term[2]}
    
    target_positions = []
    for i, ui in u_dict.items():
        for j, vj in v_dict.items():
            for k, wk in w_dict.items():
                target_value = ui * vj * wk
                target_positions.append((i, j, k, target_value))
    
    if not target_positions:
        return True
    
    A_rows = []
    b_vec = []
    for (i, j, k, target_val) in target_positions:
        row = []
        for other_term in other_terms:
            ou_dict = {u[0]: u[1] for u in other_term[0]}
            ov_dict = {v[0]: v[1] for v in other_term[1]}
            ow_dict = {w[0]: w[1] for w in other_term[2]}
            if i in ou_dict and j in ov_dict and k in ow_dict:
                row.append(ou_dict[i] * ov_dict[j] * ow_dict[k])
            else:
                row.append(0)
        A_rows.append(row)
        b_vec.append(target_val)
    
    A = np.array(A_rows, dtype=float)
    b = np.array(b_vec, dtype=float)
    
    try:
        coeffs, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=1e-10)
        if len(residuals) > 0:
            residual = residuals[0]
        else:
            residual = np.linalg.norm(A @ coeffs - b)
        
        b_norm = np.linalg.norm(b)
        if b_norm == 0:
            return True
        return residual < 1e-8 * b_norm
    except Exception:
        return False

def _try_reduce_rank(U, V, W, n, m, p, deadline, max_iterations=500):
    """
    Try to reduce rank by eliminating redundant terms.
    Returns (U, V, W) with potentially reduced rank.
    """
    rank = len(U[0]) if U else 0
    current_rank = rank
    
    terms = _decompose_tensor_to_rank1(U, V, W)
    
    eliminated = set()
    iteration = 0
    
    while iteration < max_iterations:
        if time.time() > deadline:
            break
        
        found_elimination = False
        for idx in range(len(terms)):
            if idx in eliminated:
                continue
            if time.time() > deadline:
                break
            
            other_terms = [terms[j] for j in range(len(terms)) if j != idx and j not in eliminated]
            
            if _check_term_redundant(terms[idx], other_terms, n*m, m*p, p*n):
                eliminated.add(idx)
                found_elimination = True
                break
        
        if not found_elimination:
            break
        
        iteration += 1
    
    remaining_indices = [i for i in range(len(terms)) if i not in eliminated]
    new_rank = len(remaining_indices)
    
    if new_rank >= current_rank:
        return U, V, W
    
    new_U = [[0] * new_rank for _ in range(n * m)]
    new_V = [[0] * new_rank for _ in range(m * p)]
    new_W = [[0] * new_rank for _ in range(p * n)]
    
    for new_col, old_idx in enumerate(remaining_indices):
        for (i, val) in terms[old_idx][0]:
            new_U[i][new_col] = val
        for (j, val) in terms[old_idx][1]:
            new_V[j][new_col] = val
        for (k, val) in terms[old_idx][2]:
            new_W[k][new_col] = val
    
    return new_U, new_V, new_W

def _strassen_2x2x2_terms():
    """
    Return the 7 rank-1 terms for Strassen's M_{2,2,2} decomposition.
    Each term is (u_dict, v_dict, w_dict) where keys are (i,j), (j,k), (k,i)
    in local 2x2x2 coordinates.
    """
    # Strassen products:
    # P1 = (A11+A22)(B11+B22) -> C11 += P1, C22 += P1
    # P2 = (A21+A22)B11       -> C21 += P2, C22 -= P2
    # P3 = A11(B12-B22)       -> C12 += P3, C22 += P3
    # P4 = A22(B21-B11)       -> C11 += P4, C21 += P4
    # P5 = (A11+A12)B22       -> C11 -= P5, C12 += P5
    # P6 = (A21-A11)(B11+B12) -> C22 += P6
    # P7 = (A12-A22)(B21+B22) -> C11 += P7
    
    # For each product, the U, V, W vectors:
    # U is over (i,j) in {0,1}x{0,1} = {(0,0),(0,1),(1,0),(1,1)}
    # V is over (j,k) in {0,1}x{0,1} = {(0,0),(0,1),(1,0),(1,1)}
    # W is over (k,i) in {0,1}x{0,1} = {(0,0),(0,1),(1,0),(1,1)}
    
    # P1: (A11+A22)(B11+B22)
    # U: A11=1, A22=1 -> {(0,0):1, (1,1):1}
    # V: B11=1, B22=1 -> {(0,0):1, (1,1):1}
    # W: contributes to C11 and C22 -> C11 at (0,0), C22 at (1,1)
    # So W: {(0,0):1, (1,1):1}
    
    # P2: (A21+A22)B11
    # U: A21=1, A22=1 -> {(1,0):1, (1,1):1}
    # V: B11=1 -> {(0,0):1}
    # W: C21 at (1,0) with +1, C22 at (1,1) with -1
    # So W: {(1,0):1, (1,1):-1}
    
    # P3: A11(B12-B22)
    # U: A11=1 -> {(0,0):1}
    # V: B12=1, B22=-1 -> {(0,1):1, (1,1):-1}
    # W: C12 at (0,1) with +1, C22 at (1,1) with +1
    # So W: {(0,1):1, (1,1):1}
    
    # P4: A22(B21-B11)
    # U: A22=1 -> {(1,1):1}
    # V: B21=1, B11=-1 -> {(1,0):1, (0,0):-1}
    # W: C11 at (0,0) with +1, C21 at (1,0) with +1
    # So W: {(0,0):1, (1,0):1}
    
    # P5: (A11+A12)B22
    # U: A11=1, A12=1 -> {(0,0):1, (0,1):1}
    # V: B22=1 -> {(1,1):1}
    # W: C11 at (0,0) with -1, C12 at (0,1) with +1
    # So W: {(0,0):-1, (0,1):1}
    
    # P6: (A21-A11)(B11+B12)
    # U: A21=1, A11=-1 -> {(1,0):1, (0,0):-1}
    # V: B11=1, B12=1 -> {(0,0):1, (0,1):1}
    # W: C22 at (1,1) with +1
    # So W: {(1,1):1}
    
    # P7: (A12-A22)(B21+B22)
    # U: A12=1, A22=-1 -> {(0,1):1, (1,1):-1}
    # V: B21=1, B22=1 -> {(1,0):1, (1,1):1}
    # W: C11 at (0,0) with +1
    # So W: {(0,0):1}
    
    terms = []
    
    # P1
    u1 = {(0,0): 1, (1,1): 1}
    v1 = {(0,0): 1, (1,1): 1}
    w1 = {(0,0): 1, (1,1): 1}
    terms.append((u1, v1, w1))
    
    # P2
    u2 = {(1,0): 1, (1,1): 1}
    v2 = {(0,0): 1}
    w2 = {(1,0): 1, (1,1): -1}
    terms.append((u2, v2, w2))
    
    # P3
    u3 = {(0,0): 1}
    v3 = {(0,1): 1, (1,1): -1}
    w3 = {(0,1): 1, (1,1): 1}
    terms.append((u3, v3, w3))
    
    # P4
    u4 = {(1,1): 1}
    v4 = {(1,0): 1, (0,0): -1}
    w4 = {(0,0): 1, (1,0): 1}
    terms.append((u4, v4, w4))
    
    # P5
    u5 = {(0,0): 1, (0,1): 1}
    v5 = {(1,1): 1}
    w5 = {(0,0): -1, (0,1): 1}
    terms.append((u5, v5, w5))
    
    # P6
    u6 = {(1,0): 1, (0,0): -1}
    v6 = {(0,0): 1, (0,1): 1}
    w6 = {(1,1): 1}
    terms.append((u6, v6, w6))
    
    # P7
    u7 = {(0,1): 1, (1,1): -1}
    v7 = {(1,0): 1, (1,1): 1}
    w7 = {(0,0): 1}
    terms.append((u7, v7, w7))
    
    return terms

def _construct_2x3x9_strassen():
    """
    Construct decomposition for 2x3 * 3x9 using Strassen on 2x2 blocks.
    
    The multiplication is:
    A (2x3) * B (3x9) = C (2x9)
    
    We can view this as:
    For each pair of rows i1,i2 in {0,1} (only one pair since n=2)
    For each pair of columns k1,k2 in {0..8}
    For the middle index j in {0,1,2}
    
    Actually, let's think differently. The tensor is M_{2,3,9}.
    
    We can decompose the middle dimension j={0,1,2} and output dimension k={0..8}.
    
    One approach: partition the 9 output columns into blocks.
    - 4 blocks of 2 columns: {0,1}, {2,3}, {4,5}, {6,7}
    - 1 block of 1 column: {8}
    
    For each pair of middle indices (j1,j2) and each pair of output indices (k1,k2),
    we have a sub-problem M_{2,2,2} if we use Strassen, or M_{2,2,1} etc.
    
    Actually, let's be more systematic.
    
    The full decomposition:
    For i in {0,1}, j in {0,1,2}, k in {0..8}:
      C[i,k] += A[i,j] * B[j,k]
    
    We can group by (j1,j2) pairs and (k1,k2) pairs.
    
    For a fixed pair (j1,j2) and (k1,k2), the sub-problem is:
    C[i,k1] += A[i,j1]*B[j1,k1] + A[i,j2]*B[j2,k1]
    C[i,k2] += A[i,j1]*B[j1,k2] + A[i,j2]*B[j2,k2]
    
    This is M_{2,2,2} with:
    A_sub = [[A[0,j1], A[0,j2]], [A[1,j1], A[1,j2]]]
    B_sub = [[B[j1,k1], B[j1,k2]], [B[j2,k1], B[j2,k2]]]
    C_sub = [[C[0,k1], C[0,k2]], [C[1,k1], C[1,k2]]]
    
    So for each pair (j1,j2) and (k1,k2), we use 7 terms.
    
    Number of (j1,j2) pairs from {0,1,2}: C(3,2) = 3 pairs: (0,1), (0,2), (1,2)
    Number of (k1,k2) pairs from {0..8}: C(9,2) = 36 pairs
    
    But wait, this would give 3 * 36 * 7 = 756 terms, which is way more than 54.
    
    The issue is that Strassen's algorithm works for computing a full 2x2 product from 2x2 inputs.
    We can't just apply it to arbitrary pairs of middle and output indices independently.
    
    Let me reconsider. The standard approach for M_{n,m,p} using Strassen-like decompositions
    is to block the matrices.
    
    For 2x3 * 3x9:
    We can write:
    A = [A_1 | A_2] where A_1 is 2x2 (columns 0,1) and A_2 is 2x1 (column 2)
    B = [B_1; B_2; B_3] where B_1 is 2x9 (rows 0,1), B_2 is 1x9 (row 2)
    
    Wait, that's not quite right either.
    
    Let's use a different blocking:
    A (2x3): rows are indexed by i in {0,1}, columns by j in {0,1,2}
    B (3x9): rows by j in {0,1,2}, columns by k in {0..8}
    
    We can partition j into {0,1} and {2}:
    A = [A_{j<2} | A_{j=2}] where A_{j<2} is 2x2, A_{j=2} is 2x1
    B = [B_{j<2}; B_{j=2}] where B_{j<2} is 2x9, B_{j=2} is 1x9
    
    Then C = A_{j<2} * B_{j<2} + A_{j=2} * B_{j=2}
    
    The first term is 2x2 * 2x9, which we can further decompose.
    The second term is 2x1 * 1x9, which is rank 2*1*9 = 18.
    
    For 2x2 * 2x9:
    Partition the 9 output columns into {0,1}, {2,3}, {4,5}, {6,7}, {8}
    For each pair of output columns (k1,k2) from the first four pairs, we have a 2x2 * 2x2 sub-problem.
    For the last column, we have a 2x2 * 2x1 sub-problem.
    
    2x2 * 2x2: Strassen gives 7 terms.
    2x2 * 2x1: This is M_{2,2,1}. Let's compute its rank.
    
    M_{2,2,1}: A(2x2) * B(2x1) = C(2x1)
    C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[1] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    
    This is like multiplying a 2x2 by a 2x1. The rank is at most 4 (schoolbook).
    Can we do better? 
    C[0] = a*d + b*e (using different notation)
    Actually, let's just use schoolbook for M_{2,2,1}: 4 terms.
    
    So for 2x2 * 2x9:
    4 pairs of (k1,k2) from {0,1},{2,3},{4,5},{6,7}: 4 * 7 = 28 terms
    1 single k=8: 4 terms
    Total: 32 terms
    
    For 2x1 * 1x9:
    A is 2x1, B is 1x9.
    C[i,k] = A[i,0] * B[0,k]
    This is rank 2*1*9 = 18 terms (schoolbook, and it's minimal since each term is independent).
    
    Total: 32 + 18 = 50 terms.
    
    This matches the earlier calculation. Let's implement this.
    """
    n, m, p = 2, 3, 9
    
    # We'll build U, V, W directly.
    # U has shape (n*m, rank) = (6, rank)
    # V has shape (m*p, rank) = (27, rank)
    # W has shape (p*n, rank) = (18, rank)
    
    # Indexing:
    # U[i*m + j, col] corresponds to A[i,j]
    # V[j*p + k, col] corresponds to B[j,k]
    # W[k*n + i, col] corresponds to C[i,k]
    
    U_terms = []  # List of dicts: {(i,j): coeff}
    V_terms = []  # List of dicts: {(j,k): coeff}
    W_terms = []  # List of dicts: {(i,k): coeff}
    
    # Part 1: A_{j<2} * B_{j<2} where j in {0,1}
    # A_sub is 2x2 (i in {0,1}, j in {0,1})
    # B_sub is 2x9 (j in {0,1}, k in {0..8})
    
    # Partition k into pairs: (0,1), (2,3), (4,5), (6,7) and single (8)
    k_pairs = [(0,1), (2,3), (4,5), (6,7)]
    k_single = [8]
    
    strassen_terms = _strassen_2x2x2_terms()
    
    for (k1, k2) in k_pairs:
        # For this pair of output columns, we have a 2x2 * 2x2 sub-problem
        # A_sub: rows i in {0,1}, cols j in {0,1}
        # B_sub: rows j in {0,1}, cols k in {k1,k2}
        # C_sub: rows i in {0,1}, cols k in {k1,k2}
        
        # Apply Strassen's 7 terms
        for (u_dict, v_dict, w_dict) in strassen_terms:
            # u_dict keys are (i, j) in local {0,1}x{0,1}
            # v_dict keys are (j, k) in local {0,1}x{0,1}
            # w_dict keys are (i, k) in local {0,1}x{0,1}
            
            # Map local to global:
            # local (i, j) -> global (i, j) since i in {0,1}, j in {0,1}
            # local (j, k_local) -> global (j, k1 + k_local)
            # local (i, k_local) -> global (i, k1 + k_local)
            
            U_term = {}
            for (li, lj), coeff in u_dict.items():
                U_term[(li, lj)] = coeff
            
            V_term = {}
            for (lj, lk), coeff in v_dict.items():
                V_term[(lj, k1 + lk)] = coeff
            
            W_term = {}
            for (li, lk), coeff in w_dict.items():
                W_term[(li, k1 + lk)] = coeff
            
            U_terms.append(U_term)
            V_terms.append(V_term)
            W_terms.append(W_term)
    
    for k in k_single:
        # 2x2 * 2x1 sub-problem: schoolbook
        # A_sub: 2x2 (i in {0,1}, j in {0,1})
        # B_sub: 2x1 (j in {0,1}, k=k)
        # C_sub: 2x1 (i in {0,1}, k=k)
        
        for i in range(2):
            for j in range(2):
                U_term = {(i, j): 1}
                V_term = {(j, k): 1}
                W_term = {(i, k): 1}
                U_terms.append(U_term)
                V_terms.append(V_term)
                W_terms.append(W_term)
    
    # Part 2: A_{j=2} * B_{j=2}
    # A_sub: 2x1 (i in {0,1}, j=2)
    # B_sub: 1x9 (j=2, k in {0..8})
    # C_sub: 2x9 (i in {0,1}, k in {0..8})
    
    for i in range(2):
        for k in range(9):
            U_term = {(i, 2): 1}
            V_term = {(2, k): 1}
            W_term = {(i, k): 1}
            U_terms.append(U_term)
            V_terms.append(V_term)
            W_terms.append(W_term)
    
    rank = len(U_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for (i, j), coeff in U_terms[col].items():
            U[i * m + j][col] = coeff
        for (j, k), coeff in V_terms[col].items():
            V[j * p + k][col] = coeff
        for (i, k), coeff in W_terms[col].items():
            W[k * n + i][col] = coeff
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0]) if U else 0
    
    # Build the tensor T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This should equal 1 if the indices form a valid multiplication term, 0 otherwise.
    # Actually, the matrix multiplication tensor is:
    # T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    #           = 0 otherwise
    
    # Let's check by computing the full tensor and comparing.
    T_computed = np.zeros((n*m, m*p, p*n))
    for c in range(rank):
        u_vec = np.array([U[i][c] for i in range(n*m)])
        v_vec = np.array([V[i][c] for i in range(m*p)])
        w_vec = np.array([W[i][c] for i in range(p*n)])
        T_computed += np.einsum('i,j,k->ijk', u_vec, v_vec, w_vec)
    
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1
    
    diff = np.max(np.abs(T_computed - T_expected))
    return diff < 1e-10

def _construct_2x3x9():
    """
    Construct rank-minimal decomposition for 2x3 * 3x9.
    Uses Strassen-based decomposition to achieve rank 50.
    """
    n, m, p = 2, 3, 9
    start = time.time()
    deadline = start + 130  # Safety margin
    
    try:
        U, V, W = _construct_2x3x9_strassen()
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            # Try to reduce further
            U, V, W = _try_reduce_rank(U, V, W, n, m, p, deadline)
            return U, V, W
        else:
            # Fallback to schoolbook if Strassen decomposition is incorrect
            return _rank1_decomposition(n, m, p)
    except Exception:
        return _rank1_decomposition(n, m, p)

def construct():
    """
    Returns tensor decomposition for matrix multiplication 2x3 * 3x9.
    Uses Strassen-based decomposition to achieve rank 50.
    """
    start = time.time()
    deadline = start + 135  # Safety margin before 150s kill
    
    try:
        result = _construct_2x3x9()
        return {"U": result[0], "V": result[1], "W": result[2]}
    except Exception as e:
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
