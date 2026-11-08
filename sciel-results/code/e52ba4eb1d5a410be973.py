import numpy as np
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 4, 7
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))

    def strassen_2x2_block(i_off, j_off, k_off, a_scale=1, b_scale=1, c_scale=1):
        # 1
        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        # 2
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = a_scale
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        w = [0] * (p * n)
        w[k_off * n + (i_off+1)] = c_scale
        w[(k_off+1) * n + (i_off+1)] = -c_scale
        add_term(u, v, w)

        # 3
        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        v = [0] * (m * p)
        v[j_off * p + (k_off+1)] = b_scale
        v[(j_off+1) * p + (k_off+1)] = -b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = c_scale
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        # 4
        u = [0] * (n * m)
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = b_scale
        v[j_off * p + k_off] = -b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        w[k_off * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        # 5
        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        u[i_off * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = c_scale
        w[k_off * n + i_off] = -c_scale
        add_term(u, v, w)

        # 6
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = a_scale
        u[i_off * m + j_off] = -a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        v[j_off * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        # 7
        u = [0] * (n * m)
        u[i_off * m + (j_off+1)] = a_scale
        u[(i_off+1) * m + (j_off+1)] = -a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = b_scale
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        add_term(u, v, w)

    def strassen_4x4_head():
        outer_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        inner_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        def get_block_indices(block_idx, local_idx, dim_a):
            br = block_idx // 2
            bc = block_idx % 2
            lr = local_idx // 2
            lc = local_idx % 2
            row = br * 2 + lr
            col = bc * 2 + lc
            if dim_a == 'A':
                return row * m + col
            elif dim_a == 'B':
                return row * p + col
            elif dim_a == 'C':
                return col * n + row

        for outer_a, outer_b, outer_c in outer_terms:
            for inner_x, inner_y, inner_z in inner_terms:
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                for x_local, x_coeff in inner_x:
                    for a_block, a_coeff in outer_a:
                        idx = get_block_indices(a_block, x_local, 'A')
                        u[idx] += a_coeff * x_coeff
                for y_local, y_coeff in inner_y:
                    for b_block, b_coeff in outer_b:
                        idx = get_block_indices(b_block, y_local, 'B')
                        v[idx] += b_coeff * y_coeff
                for z_local, z_coeff in inner_z:
                    for c_block, c_coeff in outer_c:
                        idx = get_block_indices(c_block, z_local, 'C')
                        w[idx] += c_coeff * z_coeff
                add_term(u, v, w)

    def strassen_4x4x2_tail():
        for i_b in range(2):
            for j_b in range(2):
                i_off = i_b * 2
                j_off = j_b * 2
                k_off = 4
                strassen_2x2_block(i_off, j_off, k_off)

    def schoolbook_4x4x1_tail():
        for i in range(n):
            for j in range(m):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + 6] = 1
                w = [0] * (p * n)
                w[6 * n + i] = 1
                add_term(u, v, w)

    strassen_4x4_head()
    strassen_4x4x2_tail()
    schoolbook_4x4x1_tail()
    
    if time.time() > deadline - 5:
        return {"U": [], "V": [], "W": []}

    num_terms = len(terms)
    
    if num_terms == 0:
        return {"U": [], "V": [], "W": []}

    # Build the tensor matrix for exact rank computation
    # Each term is a rank-1 tensor u ⊗ v ⊗ w
    # Flatten into a vector of length (n*m) * (m*p) * (p*n)
    
    N = (n * m) * (m * p) * (p * n)
    
    # Use Fraction for exact arithmetic
    # Build matrix as list of lists of Fractions
    # This might be slow, but let's try
    
    # Actually, let's use numpy for speed but with a fallback to exact arithmetic
    # First, try to find dependencies using integer Gaussian elimination
    
    # Build the matrix as a numpy array of integers
    tensor_matrix = np.zeros((N, num_terms), dtype=np.int64)
    
    for t_idx, (u, v, w) in enumerate(terms):
        u_arr = np.array(u, dtype=np.int64)
        v_arr = np.array(v, dtype=np.int64)
        w_arr = np.array(w, dtype=np.int64)
        term_vec = np.einsum('i,j,c->ijc', u_arr, v_arr, w_arr).flatten()
        tensor_matrix[:, t_idx] = term_vec

    # Perform Gaussian elimination to find rank and pivot columns
    # We'll work with the matrix as a list of integer vectors
    # and use Fraction for exact division
    
    # Convert to list of lists of Fractions
    # This is expensive, so let's do it carefully
    
    # Actually, let's use a different approach: 
    # We can find the rank using SVD first (fast), and if rank < num_terms,
    # then do exact Gaussian elimination to find a basis
    
    # Quick SVD to estimate rank
    try:
        U_svd, s, Vt = np.linalg.svd(tensor_matrix.astype(np.float64), full_matrices=False)
        tol = np.finfo(np.float64).eps * max(N, num_terms) * s[0] if s.size > 0 else 0
        svd_rank = int(np.sum(s > tol))
    except Exception:
        svd_rank = num_terms

    if svd_rank >= num_terms:
        # No dependencies found, return original terms
        rank = num_terms
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for c, (u, v, w) in enumerate(terms):
            for r in range(n * m):
                if u[r] != 0:
                    U[r][c] = u[r]
            for r in range(m * p):
                if v[r] != 0:
                    V[r][c] = v[r]
            for r in range(p * n):
                if w[r] != 0:
                    W[r][c] = w[r]
        return {"U": U, "V": V, "W": W}

    # Dependencies exist, do exact Gaussian elimination
    # Convert to Fractions
    mat = [[Fraction(tensor_matrix[i, j]) for j in range(num_terms)] for i in range(N)]
    
    pivot_cols = []
    row = 0
    for col in range(num_terms):
        if time.time() > deadline - 10:
            break
        # Find pivot
        pivot = -1
        for i in range(row, N):
            if mat[i][col] != 0:
                pivot = i
                break
        if pivot == -1:
            continue
        # Swap rows
        mat[row], mat[pivot] = mat[pivot], mat[row]
        # Eliminate
        for i in range(N):
            if i != row and mat[i][col] != 0:
                factor = mat[i][col] / mat[row][col]
                for j in range(num_terms):
                    mat[i][j] -= factor * mat[row][j]
        pivot_cols.append(col)
        row += 1
        if row >= N:
            break

    rank = len(pivot_cols)
    
    if rank == 0 or rank >= num_terms:
        # Fallback to original terms
        rank = num_terms
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for c, (u, v, w) in enumerate(terms):
            for r in range(n * m):
                if u[r] != 0:
                    U[r][c] = u[r]
            for r in range(m * p):
                if v[r] != 0:
                    V[r][c] = v[r]
            for r in range(p * n):
                if w[r] != 0:
                    W[r][c] = w[r]
        return {"U": U, "V": V, "W": W}

    # Now, express each non-pivot column as a linear combination of pivot columns
    # The reduced row echelon form has 1s at pivot positions
    # For each non-pivot column, the entries in pivot rows give the coefficients
    
    # Actually, we need to be more careful. The RREF has the property that
    # each pivot column has a 1 at the pivot row and 0s elsewhere.
    # For a non-pivot column j, the entry in row i (where i is a pivot row)
    # gives the coefficient of the pivot column for row i.
    
    # Let's build the coefficient matrix
    # coeffs[non_pivot_col][pivot_row_index] = coefficient
    
    non_pivot_cols = [c for c in range(num_terms) if c not in pivot_cols]
    
    # For each non-pivot column, find its expression in terms of pivot columns
    # In RREF, column j has entries: for each pivot row i, mat[i][j] is the coefficient
    # of the pivot column corresponding to row i.
    
    # Map pivot row index to pivot column
    pivot_row_to_col = {}
    for pr_idx, pc in enumerate(pivot_cols):
        pivot_row_to_col[pr_idx] = pc
    
    # Coefficients for non-pivot columns
    # coeff[non_pivot_col][pivot_row_idx] = mat[pivot_row_idx][non_pivot_col]
    
    # Now, T = sum_{t=0}^{num_terms-1} term_t
    # = sum_{t in pivot_cols} term_t + sum_{t in non_pivot_cols} term_t
    # = sum_{t in pivot_cols} term_t + sum_{t in non_pivot_cols} sum_{pr} coeff[t][pr] * term_{pivot_cols[pr]}
    # = sum_{pr} (1 + sum_{t in non_pivot_cols} coeff[t][pr]) * term_{pivot_cols[pr]}
    
    # So the coefficient of term_{pivot_cols[pr]} is 1 + sum_{t in non_pivot_cols} coeff[t][pr]
    
    final_coeffs = [Fraction(1) for _ in range(rank)]
    for t in non_pivot_cols:
        for pr_idx in range(rank):
            final_coeffs[pr_idx] += mat[pr_idx][t]
    
    # Now, T = sum_{pr} final_coeffs[pr] * term_{pivot_cols[pr]}
    # Each term is a rank-1 tensor, so T is a sum of rank rank-1 tensors.
    # The new U, V, W have columns that are final_coeffs[pr] * (original term's vectors)
    
    # Check if all final_coeffs are integers or half-integers
    # If not, we might need to scale
    
    # For now, assume they are rational and construct U, V, W with Fraction entries
    # Then convert to float if needed, but the task allows integer/half-integer
    
    # Let's check if all entries are integers or half-integers
    all_ok = True
    for fc in final_coeffs:
        if fc.denominator not in (1, 2):
            all_ok = False
            break
    
    if not all_ok:
        # Scale to make all entries integers
        # Find LCM of all denominators
        from math import gcd
        def lcm(a, b):
            return a * b // gcd(a, b)
        
        lcm_den = 1
        for fc in final_coeffs:
            lcm_den = lcm(lcm_den, fc.denominator)
        
        # Scale all coefficients by lcm_den
        final_coeffs = [fc * lcm_den for fc in final_coeffs]
        # But this changes the decomposition: T = sum (fc * lcm_den) * term / lcm_den
        # We need to divide one of U, V, W by lcm_den
        # Let's divide W by lcm_den
        scale_factor = Fraction(1, lcm_den)
    else:
        scale_factor = Fraction(1)

    # Construct U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for pr_idx, pc in enumerate(pivot_cols):
        u, v, w = terms[pc]
        coeff = final_coeffs[pr_idx]
        for r in range(n * m):
            val = u[r] * coeff
            if val != 0:
                U[r][pr_idx] = float(val)
        for r in range(m * p):
            val = v[r] * coeff
            if val != 0:
                V[r][pr_idx] = float(val)
        for r in range(p * n):
            val = w[r] * coeff * scale_factor
            if val != 0:
                W[r][pr_idx] = float(val)

    return {"U": U, "V": V, "W": W}
