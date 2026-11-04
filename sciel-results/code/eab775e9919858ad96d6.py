"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify decomposition using matrix multiplication check."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for c in range(rank):
        u_col = U_arr[:, c].reshape(n, m)
        v_col = V_arr[:, c].reshape(m, p)
        w_col = W_arr[:, c].reshape(p, n)
        prod = u_col @ v_col
        if not np.allclose(prod, w_col.T, atol=1e-9):
            return False
    return True

def _strassen_2x2x2(i0, j0, k0, n, m, p, U, V, W, col):
    """Apply Strassen to 2x2x2 block. Returns new column index."""
    # Strassen's 7 multiplications for 2x2 matrix multiply
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define the 7 terms as (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: list of (di, dj, sign) for U
    # b_coeffs: list of (dj, dk, sign) for V
    # c_coeffs: list of (dk, di, sign) for W
    
    terms = [
        # M1
        {
            'U': [(0,0,1), (1,1,1)],
            'V': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1)]
        },
        # M2
        {
            'U': [(1,0,1), (1,1,1)],
            'V': [(0,0,1)],
            'W': [(0,1,1), (1,1,-1)]
        },
        # M3
        {
            'U': [(0,0,1)],
            'V': [(1,0,1), (1,1,-1)],
            'W': [(0,1,1), (1,1,1)]
        },
        # M4
        {
            'U': [(1,1,1)],
            'V': [(0,1,1), (0,0,-1)],
            'W': [(0,0,1), (0,1,1)]
        },
        # M5
        {
            'U': [(0,0,1), (0,1,1)],
            'V': [(1,1,1)],
            'W': [(0,0,-1), (0,1,1)]
        },
        # M6
        {
            'U': [(1,0,1), (0,0,-1)],
            'V': [(0,0,1), (0,1,1)],
            'W': [(1,1,1)]
        },
        # M7
        {
            'U': [(0,1,1), (1,1,-1)],
            'V': [(1,0,1), (1,1,1)],
            'W': [(0,0,1)]
        },
    ]
    
    for t in terms:
        c = col + t if isinstance(t, int) else col
        c_idx = col
        for (di, dj, sign) in t['U']:
            U[(i0 + di) * m + (j0 + dj)][c_idx] += sign
        for (dj, dk, sign) in t['V']:
            V[(j0 + dj) * p + (k0 + dk)][c_idx] += sign
        for (dk, di, sign) in t['W']:
            W[(k0 + dk) * n + (i0 + di)][c_idx] += sign
        col += 1
    
    return col

def _schoolbook_block(i0, j0, k0, hi, hj, hk, n, m, p, U, V, W, col):
    """Schoolbook for hi x hj x hk block. Returns new column index."""
    for di in range(hi):
        for dj in range(hj):
            for dk in range(hk):
                U[(i0 + di) * m + (j0 + dj)][col] = 1
                V[(j0 + dj) * p + (k0 + dk)][col] = 1
                W[(k0 + dk) * n + (i0 + di)][col] = 1
                col += 1
    return col

def _construct_block_decomposition(splits_i, splits_j, splits_k):
    """Construct using block decomposition with Strassen for 2x2x2 blocks."""
    n, m, p = DIMENSIONS
    
    # Calculate total rank needed
    total_rank = 0
    i_offsets = [0]
    for s in splits_i[:-1]:
        i_offsets.append(i_offsets[-1] + s)
    j_offsets = [0]
    for s in splits_j[:-1]:
        j_offsets.append(j_offsets[-1] + s)
    k_offsets = [0]
    for s in splits_k[:-1]:
        k_offsets.append(k_offsets[-1] + s)
    
    for ii in range(len(splits_i)):
        for jj in range(len(splits_j)):
            for kk in range(len(splits_k)):
                hi = splits_i[ii]
                hj = splits_j[jj]
                hk = splits_k[kk]
                if hi == 2 and hj == 2 and hk == 2:
                    total_rank += 7
                else:
                    total_rank += hi * hj * hk
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for ii in range(len(splits_i)):
        for jj in range(len(splits_j)):
            for kk in range(len(splits_k)):
                hi = splits_i[ii]
                hj = splits_j[jj]
                hk = splits_k[kk]
                i0 = i_offsets[ii]
                j0 = j_offsets[jj]
                k0 = k_offsets[kk]
                
                if hi == 2 and hj == 2 and hk == 2:
                    col = _strassen_2x2x2(i0, j0, k0, n, m, p, U, V, W, col)
                else:
                    col = _schoolbook_block(i0, j0, k0, hi, hj, hk, n, m, p, U, V, W, col)
    
    return {"U": U, "V": V, "W": W}

def _try_all_splits():
    """Try various block splits and return the best one."""
    n, m, p = DIMENSIONS
    best_result = None
    best_rank = float('inf')
    
    # Generate possible splits for each dimension
    def gen_splits(dim):
        """Generate all compositions of dim into parts of size 1 or 2."""
        results = []
        def helper(remaining, current):
            if remaining == 0:
                results.append(list(current))
                return
            if remaining >= 1:
                current.append(1)
                helper(remaining - 1, current)
                current.pop()
            if remaining >= 2:
                current.append(2)
                helper(remaining - 2, current)
                current.pop()
        helper(dim, [])
        return results
    
    splits_i_options = gen_splits(n)
    splits_j_options = gen_splits(m)
    splits_k_options = gen_splits(p)
    
    for si in splits_i_options:
        for sj in splits_j_options:
            for sk in splits_k_options:
                result = _construct_block_decomposition(si, sj, sk)
                rank = len(result['U'][0])
                
                # Verify
                if _verify(result['U'], result['V'], result['W'], n, m, p):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = result
    
    return best_result, best_rank

def _construct_schoolbook():
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

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Start with schoolbook
    best_result = _construct_schoolbook()
    best_rank = 3 * 5 * 7
    
    # Try block decomposition
    if time.time() < deadline:
        block_result, block_rank = _try_all_splits()
        if block_result is not None and block_rank < best_rank:
            best_result = block_result
            best_rank = block_rank
    
    return best_result
# EVOLVE-BLOCK-END
