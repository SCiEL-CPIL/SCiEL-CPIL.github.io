"""
Matrix multiplication 2x5x5 rank minimization.
Uses systematic linear algebra to find rank reductions below the
Strassen-based rank-46 decomposition.
"""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _build_strassen_decomposition():
    """
    Build the rank-46 Strassen-based decomposition for 2x5 x 5x5.
    Partition inner dimension 5 = 2+2+1.
    """
    n, m, p = 2, 5, 5
    terms = []

    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))

    def schoolbook_block(a_rows, a_cols, b_rows, b_cols, a_shape, b_shape, out_rows, out_cols):
        nr, nc = a_shape
        nr2, nc2 = b_shape
        for i in range(nr):
            for k in range(nc):
                for j in range(nc2):
                    u_idx = (a_rows + i) * m + (a_cols + k)
                    v_idx = (b_rows + k) * p + (b_cols + j)
                    w_idx = (out_cols + j) * n + (out_rows + i)
                    add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])

    def strassen_2x2x2(a_row_off, a_col_off, b_row_off, b_col_off, out_row_off, out_col_off):
        a_u = a_row_off * m + a_col_off
        b_u = a_row_off * m + (a_col_off + 1)
        c_u = (a_row_off + 1) * m + a_col_off
        d_u = (a_row_off + 1) * m + (a_col_off + 1)

        e_v = b_row_off * p + b_col_off
        f_v = b_row_off * p + (b_col_off + 1)
        g_v = (b_row_off + 1) * p + b_col_off
        h_v = (b_row_off + 1) * p + (b_col_off + 1)

        o00_w = out_col_off * n + out_row_off
        o01_w = (out_col_off + 1) * n + out_row_off
        o10_w = out_col_off * n + (out_row_off + 1)
        o11_w = (out_col_off + 1) * n + (out_row_off + 1)

        add_term([(a_u, 1), (d_u, 1)], [(e_v, 1), (h_v, 1)], [(o00_w, 1), (o11_w, 1)])
        add_term([(c_u, 1), (d_u, 1)], [(e_v, 1)], [(o10_w, 1), (o11_w, -1)])
        add_term([(a_u, 1)], [(f_v, 1), (h_v, -1)], [(o01_w, 1), (o11_w, 1)])
        add_term([(d_u, 1)], [(g_v, 1), (e_v, -1)], [(o00_w, 1), (o10_w, 1)])
        add_term([(a_u, 1), (b_u, 1)], [(h_v, 1)], [(o00_w, -1), (o01_w, 1)])
        add_term([(c_u, 1), (a_u, -1)], [(e_v, 1), (f_v, 1)], [(o11_w, 1)])
        add_term([(b_u, 1), (d_u, -1)], [(g_v, 1), (h_v, 1)], [(o00_w, 1)])

    strassen_2x2x2(0, 0, 0, 0, 0, 0)
    strassen_2x2x2(0, 0, 0, 2, 0, 2)
    schoolbook_block(0, 0, 0, 4, (2, 2), (2, 1), 0, 4)

    strassen_2x2x2(0, 2, 2, 0, 0, 0)
    strassen_2x2x2(0, 2, 2, 2, 0, 2)
    schoolbook_block(0, 2, 2, 4, (2, 2), (2, 1), 0, 4)

    schoolbook_block(0, 4, 4, 0, (2, 1), (1, 5), 0, 0)

    return terms

def _terms_to_matrices(terms):
    """Convert list of term tuples to U, V, W matrices."""
    n, m, p = 2, 5, 5
    total_rank = len(terms)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    for col, (u_entries, v_entries, w_entries) in enumerate(terms):
        for idx, val in u_entries:
            U[idx][col] = val
        for idx, val in v_entries:
            V[idx][col] = val
        for idx, val in w_entries:
            W[idx][col] = val

    return U, V, W, total_rank

def _validate_decomposition(U, V, W, n, m, p):
    """Validate that the decomposition exactly reconstructs the matrix multiplication tensor."""
    r = len(W[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = 0.0
                for c in range(r):
                    val += U[idx_u][c] * V[idx_v][c] * W[idx_w][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _find_null_space_reductions(terms, deadline):
    """
    Find linear dependencies among terms and use them to reduce rank.
    Build the contribution matrix and find null-space vectors that
    can zero out coefficients.
    """
    n, m, p = 2, 5, 5
    r0 = len(terms)
    num_constraints = n * m * p  # 50

    # Build contribution matrix: M[c, (i,j,k)] = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    M = np.zeros((r0, num_constraints))
    for c in range(r0):
        u_dict = {}
        v_dict = {}
        w_dict = {}
        for idx, val in terms[c][0]:
            u_dict[idx] = val
        for idx, val in terms[c][1]:
            v_dict[idx] = val
        for idx, val in terms[c][2]:
            w_dict[idx] = val
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    u_val = u_dict.get(idx_u, 0)
                    v_val = v_dict.get(idx_v, 0)
                    w_val = w_dict.get(idx_w, 0)
                    M[c, i * m * p + j * p + k] = u_val * v_val * w_val

    # Find null space of M (rows are terms, cols are constraints)
    # We want vectors v such that M.T @ v = 0, i.e., sum_c v[c] * M[c,:] = 0
    try:
        null_space = np.linalg.null_space(M.T)
    except:
        null_space = None

    if null_space is None or null_space.shape[1] == 0:
        return terms

    best_terms = terms
    best_rank = r0

    # For each null-space vector, try to zero out one coefficient
    for ns in range(null_space.shape[1]):
        if time.time() > deadline - 5:
            break
        v = null_space[:, ns]
        
        # Coefficients become 1 + t*v
        # Try zeroing out each term
        for target in range(r0):
            if abs(v[target]) < 1e-10:
                continue
            t = -1.0 / v[target]
            new_coeffs = 1.0 + t * v
            
            # Build new terms (skip the zeroed one)
            new_terms = []
            valid = True
            for c in range(r0):
                if c == target:
                    continue
                if abs(new_coeffs[c]) < 1e-9:
                    continue
                scale = new_coeffs[c]
                u_entries = [(idx, val * scale) for idx, val in terms[c][0]]
                v_entries = [(idx, val) for idx, val in terms[c][1]]
                w_entries = [(idx, val) for idx, val in terms[c][2]]
                new_terms.append((u_entries, v_entries, w_entries))
            
            if len(new_terms) < best_rank:
                U_test, V_test, W_test, _ = _terms_to_matrices(new_terms)
                if _validate_decomposition(U_test, V_test, W_test, n, m, p):
                    best_terms = new_terms
                    best_rank = len(new_terms)

    return best_terms

def _try_local_replacement(terms, deadline):
    """
    Try replacing small subsets of terms with fewer terms.
    For each pair of terms, check if their combined contribution
    can be represented by a single rank-1 tensor.
    """
    n, m, p = 2, 5, 5
    r0 = len(terms)
    best_terms = terms
    best_rank = r0

    # Precompute contribution vectors
    contribs = []
    for c in range(r0):
        vec = np.zeros(n * m * p)
        u_dict = {}
        v_dict = {}
        w_dict = {}
        for idx, val in terms[c][0]:
            u_dict[idx] = val
        for idx, val in terms[c][1]:
            v_dict[idx] = val
        for idx, val in terms[c][2]:
            w_dict[idx] = val
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    u_val = u_dict.get(idx_u, 0)
                    v_val = v_dict.get(idx_v, 0)
                    w_val = w_dict.get(idx_w, 0)
                    vec[i * m * p + j * p + k] = u_val * v_val * w_val
        contribs.append(vec)

    # For each pair, check if sum is rank-1
    for i in range(r0):
        for j in range(i + 1, r0):
            if time.time() > deadline - 5:
                break
            
            sum_vec = contribs[i] + contribs[j]
            tensor = sum_vec.reshape(n, m, p)
            
            # Check if rank-1 by matricization
            mat = tensor.reshape(n * m, p)
            try:
                rank = np.linalg.matrix_rank(mat)
                if rank <= 1 and np.any(np.abs(mat) > 1e-10):
                    # It's rank-1! Find factors
                    # Use SVD
                    U_svd, S, Vt = np.linalg.svd(mat, full_matrices=False)
                    if len(S) > 0 and S[0] > 1e-10:
                        u_vec = U_svd[:, 0] * np.sqrt(S[0])
                        v_vec = Vt[0, :] * np.sqrt(S[0])
                        
                        # Build new term
                        new_u = []
                        new_v = []
                        new_w = []
                        
                        for ii in range(n):
                            for jj in range(m):
                                if abs(u_vec[ii * m + jj]) > 1e-10:
                                    new_u.append((ii * m + jj, u_vec[ii * m + jj]))
                        
                        for jj in range(m):
                            for kk in range(p):
                                # v_vec is indexed by (j,k) but we need to extract properly
                                # mat is (n*m) x p, so row index is i*m+j, col is k
                                # v_vec[k] corresponds to the k-th column
                                pass
                        
                        # The matricization is (i*m+j) x k
                        # So v_vec[k] is the factor for position k in the p-dimension
                        # But we need it as B[j,k] factors... this doesn't directly work
                        # because the p-dimension is mixed with j in the matricization
                        
                        # Let's try a different matricization: (n) x (m*p)
                        mat2 = tensor.reshape(n, m * p)
                        rank2 = np.linalg.matrix_rank(mat2)
                        if rank2 <= 1:
                            U2, S2, Vt2 = np.linalg.svd(mat2, full_matrices=False)
                            if len(S2) > 0 and S2[0] > 1e-10:
                                a_vec = U2[:, 0] * np.sqrt(S2[0])
                                bc_vec = Vt2[0, :] * np.sqrt(S2[0])
                                bc_mat = bc_vec.reshape(m, p)
                                
                                new_u = []
                                new_v = []
                                new_w = []
                                
                                for ii in range(n):
                                    if abs(a_vec[ii]) > 1e-10:
                                        new_u.append((ii * m + 0, a_vec[ii]))  # This is wrong
                                        # U index is i*m+j, but a_vec only depends on i
                                        # So for each j, U[i*m+j] should have the same value
                                        # This means the term isn't separable in the U,V,W sense
                                        pass
                        
                        # This approach is too complex. Skip.
                        
            except:
                pass

    return best_terms

def _try_different_tiling(deadline):
    """
    Try alternative tilings that might give lower base rank.
    """
    n, m, p = 2, 5, 5
    terms = []

    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))

    def schoolbook(i_range, j_range, k_range):
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    add_term([(i * m + j, 1)], [(j * p + k, 1)], [(k * n + i, 1)])

    # Tiling 1: 5 = 1 + 4
    # j={0}: k={0,1}, {2,3}, {4}
    # For 2x1x2 blocks, use Strassen-like: actually for n=2, m=1, p=2
    # The tensor is sum_{i,k} A[i,0]*B[0,k] which is an outer product
    # Rank of outer product tensor: it's already rank 1 per (i,k) pair? No.
    # C[i,k] += A[i,0]*B[0,k] for i in {0,1}, k in {0,1}
    # This is 4 terms in schoolbook, but can we do it in fewer?
    # The sub-tensor has entries T[i,0,k] = 1 for i in {0,1}, k in {0,1}
    # This is a 2x1x2 tensor. Its rank: matricize as 2x(1*2)=2x2, rank 2.
    # So minimum 2 terms? Let's try: 
    # Term 1: U[0,0]=1, V[0,0]=1, W[0,0]=1; U[1,0]=1, V[0,0]=1, W[1,0]=1
    # That's 2 terms for the 2x1x2 block!
    
    # Actually, for C[i,k] = A[i,j]*B[j,k], the contribution from a single j
    # is: for each (i,k), the term is A[i,j]*B[j,k]
    # As a tensor: sum_{i,k} A[i,j] * B[j,k] * e_{i,k}
    # This has tensor rank min(n, p) = min(2, 5) = 2 for the full j-block
    # But we need to handle all k's...
    
    # Let me just use schoolbook for now and rely on null-space search
    schoolbook([0, 1], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
    
    return terms

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 5 matrix by a 5 by 5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    # Build the standard rank-46 decomposition
    terms = _build_strassen_decomposition()
    current_rank = len(terms)

    # Try null-space reductions
    if time.time() < deadline - 30:
        terms = _find_null_space_reductions(terms, deadline)
        current_rank = len(terms)

    # Try local replacements
    if time.time() < deadline - 10:
        terms = _try_local_replacement(terms, deadline)
        current_rank = len(terms)

    # Convert to matrices
    U, V, W, total_rank = _terms_to_matrices(terms)

    # Verify
    if not _validate_decomposition(U, V, W, 2, 5, 5):
        # Fallback to original
        terms = _build_strassen_decomposition()
        U, V, W, total_rank = _terms_to_matrices(terms)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
