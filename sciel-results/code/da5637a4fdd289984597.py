"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]
import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START

def _verify_decomp(decomp, n, m, p, tol=1e-9):
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    rank = len(U[0]) if U else 0
    
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U) or any(len(row) != rank for row in V) or any(len(row) != rank for row in W):
        return False
    
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _recover_rational_matrix(M, max_den=8, tol=1e-6):
    """Recover a matrix with small rational entries."""
    n_rows, n_cols = M.shape
    result = np.zeros_like(M)
    for i in range(n_rows):
        for j in range(n_cols):
            val = M[i, j]
            if abs(val) < tol:
                result[i, j] = 0.0
                continue
            # Try to find p/q approximation
            frac = Fraction(val).limit_denominator(max_den)
            if abs(float(frac) - val) < 1e-4:
                result[i, j] = float(frac)
            else:
                # Try next few denominators
                found = False
                for d in range(2, max_den + 1):
                    num = round(val * d)
                    if abs(num / d - val) < 1e-4:
                        result[i, j] = num / d
                        found = True
                        break
                if not found:
                    result[i, j] = val
    return result

def _cp_decompose_tensor(T, n, m, p, rank, max_iters=200, tol=1e-8):
    """
    CP decomposition of the matmul tensor T[i,j,k] = 1.
    T is indexed as (i*m+j, j*p+k, k*n+i) for the three modes.
    Uses ALS with SVD-based initialization.
    """
    # Reshape T into three matricizations
    # Mode-1: (n*m) x (m*p * p*n) -- flatten other two modes
    # Mode-2: (m*p) x (n*m * p*n)
    # Mode-3: (p*n) x (n*m * m*p)
    
    dim1 = n * m
    dim2 = m * p
    dim3 = p * n
    
    # Build the tensor as a 3D array
    T3 = np.zeros((dim1, dim2, dim3))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T3[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Initialize factors randomly
    rng = np.random.RandomState(42)
    U = rng.randn(dim1, rank) * 0.1
    V = rng.randn(dim2, rank) * 0.1
    W = rng.randn(dim3, rank) * 0.1
    
    # Better initialization: use SVD of mode-1 matricization
    T1 = T3.reshape(dim1, dim2 * dim3)
    _, _, Vt = np.linalg.svd(T1, full_matrices=False)
    U = Vt[:rank, :].T * np.sqrt(dim2 * dim3 / rank)
    
    T2 = T3.transpose(1, 0, 2).reshape(dim2, dim1 * dim3)
    _, _, Vt2 = np.linalg.svd(T2, full_matrices=False)
    V = Vt2[:rank, :].T * np.sqrt(dim1 * dim3 / rank)
    
    T3mat = T3.transpose(2, 0, 1).reshape(dim3, dim1 * dim2)
    _, _, Wt = np.linalg.svd(T3mat, full_matrices=False)
    W = Wt[:rank, :].T * np.sqrt(dim1 * dim2 / rank)
    
    for it in range(max_iters):
        # Update U: for each column c, U[:,c] = (T1 * (V[:,c] ⊗ W[:,c])) / (||V[:,c]||^2 * ||W[:,c]||^2)
        for c in range(rank):
            # T1 * (V[:,c] ⊗ W[:,c]) means: sum over (j,k) of T3[i,j,k] * V[j,c] * W[k,c]
            # This is the mode-1 unfolding multiplied by the outer product
            vw = np.outer(V[:, c], W[:, c])  # (dim2*dim3,)
            T1_flat = T3.reshape(dim1, dim2 * dim3)
            U[:, c] = T1_flat @ vw
            norm_sq = np.dot(V[:, c], V[:, c]) * np.dot(W[:, c], W[:, c])
            if norm_sq > 1e-12:
                U[:, c] /= norm_sq
        
        # Update V
        for c in range(rank):
            uw = np.outer(U[:, c], W[:, c])  # (dim1*dim3,)
            T2_flat = T3.transpose(1, 0, 2).reshape(dim2, dim1 * dim3)
            V[:, c] = T2_flat @ uw
            norm_sq = np.dot(U[:, c], U[:, c]) * np.dot(W[:, c], W[:, c])
            if norm_sq > 1e-12:
                V[:, c] /= norm_sq
        
        # Update W
        for c in range(rank):
            uv = np.outer(U[:, c], V[:, c])  # (dim1*dim2,)
            T3_flat = T3.transpose(2, 0, 1).reshape(dim3, dim1 * dim2)
            W[:, c] = T3_flat @ uv
            norm_sq = np.dot(U[:, c], U[:, c]) * np.dot(V[:, c], V[:, c])
            if norm_sq > 1e-12:
                W[:, c] /= norm_sq
        
        # Check convergence
        if it > 0:
            # Compute residual
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        err += (s - 1.0)**2
            if err < tol:
                break
    
    return U, V, W, err

def _try_rank(rank, n, m, p, max_den=8):
    """Try to find a valid decomposition at the given rank."""
    U, V, W, err = _cp_decompose_tensor(np.zeros((n*m, m*p, p*n)), n, m, p, rank)
    
    if err > 1e-4:
        return None
    
    # Recover rational entries
    U_r = _recover_rational_matrix(U, max_den)
    V_r = _recover_rational_matrix(V, max_den)
    W_r = _recover_rational_matrix(W, max_den)
    
    # Convert to lists
    U_list = U_r.tolist()
    V_list = V_r.tolist()
    W_list = W_r.tolist()
    
    decomp = {"U": U_list, "V": V_list, "W": W_list}
    
    if _verify_decomp(decomp, n, m, p, tol=1e-6):
        return decomp
    
    return None

def _strassen_6x6x6():
    """
    Strassen 2x2 block decomposition for 6x6x6 matrix multiplication.
    Split into 3x3 blocks. Each 3x3*3x3*3x3 has rank 27 (schoolbook).
    7 Strassen steps -> rank 7*27 = 189.
    """
    n = m = p = 6
    nh = mh = ph = 3
    block_rank = nh * mh * ph
    total_rank = 7 * block_rank
    
    steps = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}),
        ({0: 1}, {1: 1, 3: -1}),
        ({3: 1}, {2: 1, 0: -1}),
        ({0: 1, 1: 1}, {3: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}),
    ]
    
    c_combos = [
        {0: 1, 3: -1, 5: 1, 6: -1},
        {2: 1, 4: 1},
        {1: 1, 3: 1},
        {0: 1, 1: -1, 2: 1, 6: 1},
    ]
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for s in range(7):
        a_coeffs, b_coeffs = steps[s]
        for ai, a_sign in a_coeffs.items():
            for bi, b_sign in b_coeffs.items():
                a_row_start = (ai // 2) * nh
                a_col_start = (ai % 2) * mh
                b_row_start = (bi // 2) * mh
                b_col_start = (bi % 2) * ph
                
                for i in range(nh):
                    for j in range(mh):
                        for k in range(ph):
                            gi = a_row_start + i
                            gj = a_col_start + j
                            gk = b_col_start + k
                            
                            c_block_row = gi // nh
                            c_block_col = gk // ph
                            c_block_idx = c_block_row * 2 + c_block_col
                            
                            c_sign = c_combos[c_block_idx].get(s, 0)
                            if c_sign == 0:
                                continue
                            
                            col = s * block_rank + i * (mh * ph) + j * ph + k
                            
                            U[gi * m + gj][col] = a_sign
                            V[gj * p + gk][col] = b_sign
                            W[gk * n + gi][col] = c_sign
    
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    best_decomp = None
    best_rank = float('inf')
    
    # First, get a valid baseline from Strassen (rank 189)
    try:
        decomp = _strassen_6x6x6()
        rank = len(decomp['U'][0])
        if _verify_decomp(decomp, n, m, p):
            best_decomp = decomp
            best_rank = rank
    except Exception:
        pass
    
    # Try numerical CP decomposition at various ranks
    # The matmul tensor for 6x6x6 has dimension 36x36x36
    # Known bounds: rank is between ~150 and 216
    # Try ranks below 189 to see if we can beat Strassen
    
    ranks_to_try = [180, 175, 170, 165, 160, 155, 150]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
        try:
            decomp = _try_rank(rank, n, m, p)
            if decomp is not None:
                actual_rank = len(decomp['U'][0])
                if actual_rank < best_rank:
                    best_decomp = decomp
                    best_rank = actual_rank
                    break  # Found a better one, stop
        except Exception:
            continue
    
    # Fall back to schoolbook (rank 216) if nothing else works
    if best_decomp is None:
        best_decomp = _schoolbook(n, m, p)
    
    return best_decomp

# EVOLVE-BLOCK-END
