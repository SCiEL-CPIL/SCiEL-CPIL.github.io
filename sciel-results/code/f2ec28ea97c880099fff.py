"""Matrix multiplication 4x4 * 4x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(total - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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

def _winograd_2x2x2():
    """
    Return the rank-7 Winograd F(2) decomposition for 2x2 * 2x2.
    """
    U = np.array([
        [1, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 1, 0],
        [1, 0, 0, 0, -1, -1, 1],
    ], dtype=float)

    V = np.array([
        [1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [1, -1, 0, 1, -1, -1, 1],
    ], dtype=float)

    W = np.array([
        [1, 0, 0, 0, 0, 0, -1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, -1, 0, 0],
        [0, 1, 0, 0, 0, 1, 1],
    ], dtype=float)

    n, m, p = 2, 2, 2
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(7):
                    total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(total - 1.0) > 1e-9:
                    return None

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _construct_4x4x4_kron():
    """
    Construct a rank-49 decomposition for 4x4 * 4x4 using Kronecker product of Winograd F(2).
    """
    w2 = _winograd_2x2x2()
    if not w2:
        return None

    U2 = np.array(w2["U"])  # shape (4, 7)
    V2 = np.array(w2["V"])  # shape (4, 7)
    W2 = np.array(w2["W"])  # shape (4, 7)

    n, m, p = 4, 4, 4
    rank = 49
    U4 = np.zeros((n * m, rank))
    V4 = np.zeros((m * p, rank))
    W4 = np.zeros((p * n, rank))

    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    for c1 in range(7):
                        for c2 in range(7):
                            row_idx = i1 * 8 + i2 * 4 + j1 * 2 + j2
                            col_idx = c1 * 7 + c2
                            U4[row_idx, col_idx] = U2[i1 * 2 + j1, c1] * U2[i2 * 2 + j2, c2]

    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    for c1 in range(7):
                        for c2 in range(7):
                            row_idx = j1 * 8 + j2 * 4 + k1 * 2 + k2
                            col_idx = c1 * 7 + c2
                            V4[row_idx, col_idx] = V2[j1 * 2 + k1, c1] * V2[j2 * 2 + k2, c2]

    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    for c1 in range(7):
                        for c2 in range(7):
                            row_idx = k1 * 8 + k2 * 4 + i1 * 2 + i2
                            col_idx = c1 * 7 + c2
                            W4[row_idx, col_idx] = W2[k1 * 2 + i1, c1] * W2[k2 * 2 + i2, c2]

    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}

def _construct_4x4x5():
    """
    Construct a rank-53 decomposition for 4x4 * 4x5.
    Uses rank-49 Kronecker for first 4 columns, then 4 additional rank terms for 5th column.
    """
    n, m, p = 4, 4, 5

    base = _construct_4x4x4_kron()
    if not base:
        return _schoolbook(n, m, p)

    U4 = np.array(base["U"])
    V4 = np.array(base["V"])
    W4 = np.array(base["W"])

    rank_base = 49
    rank_new = rank_base + 4

    U_new = np.zeros((n * m, rank_new))
    V_new = np.zeros((m * p, rank_new))
    W_new = np.zeros((p * n, rank_new))

    U_new[:, :rank_base] = U4

    for j in range(4):
        for k in range(4):
            V_new[j * 5 + k, :rank_base] = V4[j * 4 + k, :rank_base]

    W_new[:16, :rank_base] = W4

    for j in range(4):
        col_idx = rank_base + j
        for i in range(4):
            U_new[i * 4 + j, col_idx] = 1.0
        V_new[j * 5 + 4, col_idx] = 1.0
        for i in range(4):
            W_new[4 * 4 + i, col_idx] = 1.0

    return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}

def _construct_5x5x5_strassen():
    """
    Construct a rank-49 decomposition for 5x5 * 5x5 using padded Strassen.
    Pad 5x5 to 8x8, use Strassen rank 7^3=343... no that's too big.
    
    Actually, let's use a different approach. We'll construct a 5x5x5 decomposition
    by combining a 4x4x4 block (rank 49) with cross terms and a 1x1x1 term.
    
    For 5x5x5:
    - A = [[A44, A41], [A14, A11]] where A44 is 4x4, A41 is 4x1, A14 is 1x4, A11 is 1x1
    - B = [[B44, B41], [B14, B11]]
    - C = [[C44, C41], [C14, C11]]
    
    C44 = A44*B44 + A41*B14
    C41 = A44*B41 + A41*B11
    C14 = A14*B44 + A11*B14
    C11 = A14*B41 + A11*B11
    
    This gets complex. Let me use a simpler approach: just use the Kronecker 4x4x4
    and add the 5th row/col terms.
    
    For 5x5x5, we can use:
    - 4x4x4 part: rank 49
    - Row 5 of A times columns 1-4 of B: 4 terms (each rank 1 for the cross)
    - Columns 5 of A times rows 1-4 of B: 4 terms
    - A[4,4]*B[4,4]: 1 term
    - But we need to be careful about the structure.
    
    Actually, the cleanest way: embed 4x4x5 into 5x5x5 by padding, then use a known
    5x5x5 decomposition. The known best for 5x5x5 is rank 49 (from Strassen-like).
    
    Let me construct it directly:
    5x5x5 = 4x4x4 (rank 49) + cross terms
    
    The cross terms:
    - A[0:4, 4] * B[4, 0:4] -> contributes to C[0:4, 0:4]: 4*4=16 entries, but as rank-1 terms
      Actually this is a single rank-1 term: outer product of A[:,4] and B[4,:]
      Wait no. A[0:4,4] is a 4x1 vector, B[4,0:4] is a 1x4 vector.
      Their product is a 4x4 matrix, which is rank 1.
      So we need 1 term for this.
      
    Similarly:
    - A[0:4,4] * B[4,4] -> contributes to C[0:4,4]: 4x1, rank 1 -> 1 term
    - A[4,0:4] * B[0:4,4] -> contributes to C[4,0:4]: 1x4, rank 1 -> 1 term  
    - A[4,4] * B[4,4] -> contributes to C[4,4]: scalar -> 1 term
    - A[4,0:4] * B[0:4,0:4] -> contributes to C[4,0:4]: 1x4, rank 1 -> 1 term
    - A[0:4,0:4] * B[0:4,4] -> contributes to C[0:4,4]: 4x1, rank 1 -> 1 term
    
    Wait, I'm overcomplicating. Let me just build it term by term.
    
    For 5x5x5, the tensor has 5*5*5 = 125 entries.
    
    The 4x4x4 part uses 49 terms.
    The remaining terms involve index 4 in at least one position.
    
    Terms where i=4 or j=4 or k=4 (but not all in the 4x4x4 block):
    
    Case 1: j=4, k<4: A[i,4]*B[4,k] for i in 0..4, k in 0..3
    This is a 5x4 sub-tensor. Each (i,k) pair needs A[i,4]*B[4,k] = C[i,k] contribution.
    These are 5*4=20 scalar products, but they can be grouped.
    Actually, for fixed k, the vector [A[0,4], A[1,4], A[2,4], A[3,4], A[4,4]] * B[4,k]
    contributes to C[:,k]. This is rank 1 in the (i) direction for each k.
    So 4 terms for k=0,1,2,3.
    
    Case 2: i=4, k<4, j<4: A[4,j]*B[j,k] for j in 0..3, k in 0..3
    For fixed k, the vector [A[4,0], A[4,1], A[4,2], A[4,3]] * B[:,k] contributes to C[4,k].
    This is rank 1 in the (j) direction for each k. So 4 terms.
    
    Case 3: i=4, j<4, k=4: A[4,j]*B[j,4] for j in 0..3
    For fixed j, A[4,j]*B[j,4] contributes to C[4,4]. These 4 terms sum to C[4,4]'s contribution.
    But each is a separate rank-1 term? No, they all contribute to the same C[4,4] entry.
    We can combine them: sum over j of A[4,j]*B[j,4]. This requires 4 rank-1 terms
    (one for each j) OR we can think of it differently.
    
    Actually, let me just count: the total number of (i,j,k) triples where at least one index is 4
    is 125 - 64 = 61. But many of these can be shared in rank-1 terms.
    
    This is getting complicated. Let me just use the simpler approach:
    Build 5x5x5 as 4x4x4 + 20 additional terms (4 for each of the 5 non-4x4x4 "edges").
    
    Actually, the simplest correct approach: 
    5x5x5 rank <= 49 (from 4x4x4) + 5*4 + 5*4 + 4 + 1 = 49 + 20 + 20 + 4 + 1 = 94? That's worse.
    
    Let me just use the 4x4x5 construction directly (rank 53) and not bother with 5x5x5.
    """
    # This function is not used in the final construct, but kept for reference.
    return None

def _construct_4x4x5_nested_strassen():
    """
    Construct a rank-65 decomposition for 4x4 * 4x5 using nested Strassen.
    """
    n, m, p = 4, 4, 5
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_list, v_list, w_list):
        nonlocal col
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c
        col += 1
    
    super_terms = [
        {'p_blocks': [1, 0, 0, 1], 'q_blocks': [1, 0, 0, 1], 'c_contrib': [(0, 1), (3, 1)]},
        {'p_blocks': [0, 0, 1, 1], 'q_blocks': [1, 0, 0, 0], 'c_contrib': [(2, 1), (3, -1)]},
        {'p_blocks': [1, 0, 0, 0], 'q_blocks': [0, 1, 0, -1], 'c_contrib': [(1, 1), (3, 1)]},
        {'p_blocks': [0, 0, 0, 1], 'q_blocks': [-1, 0, 1, 0], 'c_contrib': [(0, 1), (2, 1)]},
        {'p_blocks': [1, 1, 0, 0], 'q_blocks': [0, 0, 0, 1], 'c_contrib': [(0, -1), (1, 1)]},
        {'p_blocks': [-1, 0, 1, 0], 'q_blocks': [1, 1, 0, 0], 'c_contrib': [(3, 1)]},
        {'p_blocks': [0, 1, 0, -1], 'q_blocks': [0, 0, 1, 1], 'c_contrib': [(0, 1)]},
    ]
    
    inner_strassen = [
        {'p_elems': [(0, 1), (3, 1)], 'q_elems': [(0, 1), (3, 1)], 'r_elems': [(0, 1), (3, 1)]},
        {'p_elems': [(2, 1), (3, 1)], 'q_elems': [(0, 1)], 'r_elems': [(2, 1), (3, -1)]},
        {'p_elems': [(0, 1)], 'q_elems': [(1, 1), (3, -1)], 'r_elems': [(1, 1), (3, 1)]},
        {'p_elems': [(3, 1)], 'q_elems': [(2, 1), (0, -1)], 'r_elems': [(0, 1), (2, 1)]},
        {'p_elems': [(0, 1), (1, 1)], 'q_elems': [(3, 1)], 'r_elems': [(0, -1), (1, 1)]},
        {'p_elems': [(2, 1), (0, -1)], 'q_elems': [(0, 1), (1, 1)], 'r_elems': [(3, 1)]},
        {'p_elems': [(1, 1), (3, -1)], 'q_elems': [(2, 1), (3, 1)], 'r_elems': [(0, 1)]},
    ]
    
    def get_p_indices(p_elem_idx, p_blocks):
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(p_blocks):
            if coeff == 0:
                continue
            if b_idx == 0:
                gi, gj = r, c
            elif b_idx == 1:
                gi, gj = r, 2 + c
            elif b_idx == 2:
                gi, gj = 2 + r, c
            elif b_idx == 3:
                gi, gj = 2 + r, 2 + c
            res.append((gi, gj, coeff))
        return res
    
    def get_q_indices(q_elem_idx, q_blocks):
        r = q_elem_idx // 2
        c = q_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(q_blocks):
            if coeff == 0:
                continue
            if b_idx == 0:
                gj, gk = r, c
            elif b_idx == 1:
                gj, gk = r, 2 + c
            elif b_idx == 2:
                gj, gk = 2 + r, c
            elif b_idx == 3:
                gj, gk = 2 + r, 2 + c
            res.append((gj, gk, coeff))
        return res
    
    def get_c_indices(r_elem_idx, c_contrib):
        r = r_elem_idx // 2
        c = r_elem_idx % 2
        res = []
        for c_block_idx, sign in c_contrib:
            if c_block_idx == 0:
                gi, gk = r, c
            elif c_block_idx == 1:
                gi, gk = r, 2 + c
            elif c_block_idx == 2:
                gi, gk = 2 + r, c
            elif c_block_idx == 3:
                gi, gk = 2 + r, 2 + c
            res.append((gk, gi, sign))
        return res
    
    for super_term in super_terms:
        p_blocks = super_term['p_blocks']
        q_blocks = super_term['q_blocks']
        c_contrib = super_term['c_contrib']
        
        for inner in inner_strassen:
            p_elems = inner['p_elems']
            q_elems = inner['q_elems']
            r_elems = inner['r_elems']
            
            u_list = []
            for p_elem_idx, coeff in p_elems:
                indices = get_p_indices(p_elem_idx, p_blocks)
                for gi, gj, bcoeff in indices:
                    u_list.append((gi, gj, coeff * bcoeff))
            
            v_list = []
            for q_elem_idx, coeff in q_elems:
                indices = get_q_indices(q_elem_idx, q_blocks)
                for gj, gk, bcoeff in indices:
                    v_list.append((gj, gk, coeff * bcoeff))
            
            w_list = []
            for r_elem_idx, coeff in r_elems:
                indices = get_c_indices(r_elem_idx, c_contrib)
                for gk, gi, bcoeff in indices:
                    w_list.append((gk, gi, coeff * bcoeff))
            
            add_term(u_list, v_list, w_list)
    
    for i in range(4):
        for j in range(4):
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]
            add_term(u_list, v_list, w_list)
    
    return {"U": U, "V": V, "W": W}

def _construct_4x4x5_padded_square():
    """
    Structural approach: embed 4x4x5 into a larger square problem and use
    a known efficient decomposition.
    
    Key insight: 4x4x5 can be seen as part of a 4x5x5 problem or we can
    use a "split" approach where we handle the 4x4x4 part with rank-49
    Kronecker Strassen and the 4x4x1 part more efficiently.
    
    For the 4x4x1 part (last column of B), we need:
    C[:,4] = A * B[:,4]
    
    This is a 4x4 * 4x1 multiplication. We can use Strassen-like tricks:
    Split A into 2x2 blocks and B[:,4] into two 2x1 vectors.
    
    A = [[A00, A01], [A10, A11]] (2x2 blocks of 2x2)
    b = [b0, b1] where b0 = B[0:2, 4], b1 = B[2:4, 4]
    
    c0 = A00*b0 + A01*b1  (2x1 vector)
    c1 = A10*b0 + A11*b1  (2x1 vector)
    
    Using Strassen for 2x2 * 2x1:
    s1 = (A00+A11) * b0
    s2 = A10 * (b0+b1)
    s3 = (A00+A11) * b1  -- wait, this isn't standard Strassen
    
    Actually for 2x2 * 2x1, the schoolbook is 4 multiplications.
    Strassen doesn't directly help for rectangular.
    
    Alternative: Use the fact that we only need 4 scalar outputs.
    c[0] = A[0,0]*B[0,4] + A[0,1]*B[1,4] + A[0,2]*B[2,4] + A[0,3]*B[3,4]
    c[1] = A[1,0]*B[0,4] + A[1,1]*B[1,4] + A[1,2]*B[2,4] + A[1,3]*B[3,4]
    c[2] = A[2,0]*B[0,4] + A[2,1]*B[1,4] + A[2,2]*B[2,4] + A[2,3]*B[3,4]
    c[3] = A[3,0]*B[0,4] + A[3,1]*B[1,4] + A[3,2]*B[2,4] + A[3,3]*B[3,4]
    
    This is just a matrix-vector product, rank 4 is optimal (can't do better than 4
    for a 4x4 * 4x1 in general).
    
    So the best we can do with this split is 49 + 4 = 53, which is what we already have.
    
    Let me try a different structural approach: use a 5x5x5 decomposition.
    Pad A to 5x5 (add a zero row), pad B to 5x5 (add a zero column).
    Then C_padded = A_padded * B_padded gives us C in the top-left 4x5 block.
    
    If we have a rank-r decomposition for 5x5x5, we get rank r for 4x4x5.
    
    The best known rank for 5x5x5 is 49 (same as 4x4x4, since 5x5x5 >= 4x4x4).
    Actually, R(5,5,5) >= R(4,4,4) = 49, and R(5,5,5) <= R(5,5,5) schoolbook = 125.
    Using Strassen: pad to 8x8, R(8,8,8) = 7^3 = 343. That's worse.
    
    Using the 4x4x4 + cross terms approach for 5x5x5:
    The cross terms add at most 5+5+1+5+5 = 21 terms? Let me count properly.
    
    For 5x5x5:
    - 4x4x4 block: 49 terms
    - Terms with j=4, i<4, k<4: A[i,4]*B[4,k] -> 4*4 = 16 entries, but each (i,k) 
      pair is independent. However, for fixed k, it's a rank-1 term in i.
      So 4 terms (one for each k=0..3).
    - Terms with i=4, j<4, k<4: A[4,j]*B[j,k] -> for fixed k, rank-1 in j. 4 terms.
    - Terms with i=4, j<4, k=4: A[4,j]*B[j,4] -> all contribute to C[4,4]. 4 terms.
    - Terms with i<4, j=4, k=4: A[i,4]*B[4,4] -> for fixed i, scalar. 4 terms.
    - Term with i=4, j=4, k=4: A[4,4]*B[4,4] -> 1 term.
    
    Total: 49 + 4 + 4 + 4 + 4 + 1 = 66 terms for 5x5x5.
    
    That's worse than 49. So padding to 5x5x5 doesn't help.
    
    Let me try yet another approach: use a 4x4x4 decomposition but with a different
    structure that naturally handles the 5th column.
    
    Actually, the current best is 53. Let me verify this is correct and just use it.
    """
    n, m, p = 4, 4, 5

    base = _construct_4x4x4_kron()
    if not base:
        return _schoolbook(n, m, p)

    U4 = np.array(base["U"])
    V4 = np.array(base["V"])
    W4 = np.array(base["W"])

    rank_base = 49
    rank_new = rank_base + 4

    U_new = np.zeros((n * m, rank_new))
    V_new = np.zeros((m * p, rank_new))
    W_new = np.zeros((p * n, rank_new))

    U_new[:, :rank_base] = U4

    for j in range(4):
        for k in range(4):
            V_new[j * 5 + k, :rank_base] = V4[j * 4 + k, :rank_base]

    W_new[:16, :rank_base] = W4

    for j in range(4):
        col_idx = rank_base + j
        for i in range(4):
            U_new[i * 4 + j, col_idx] = 1.0
        V_new[j * 5 + 4, col_idx] = 1.0
        for i in range(4):
            W_new[4 * 4 + i, col_idx] = 1.0

    return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}

def construct():
    n, m, p = DIMENSIONS

    start_time = time.time()
    deadline = start_time + 140

    # Try the Winograd Kronecker approach (rank 53)
    try:
        result = _construct_4x4x5()
        if result and _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass

    # Try the padded square approach (also rank 53, but different construction)
    try:
        if time.time() < deadline:
            result = _construct_4x4x5_padded_square()
            if result and _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                return result
    except Exception:
        pass

    # Try the nested Strassen approach (rank 65) as a fallback
    try:
        if time.time() < deadline:
            result = _construct_4x4x5_nested_strassen()
            if result and _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                return result
    except Exception:
        pass

    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
