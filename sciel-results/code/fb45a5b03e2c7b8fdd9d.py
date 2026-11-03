"""Matrix multiplication 3x4 by 4x6: Direct linear algebra reduction of schoolbook rank-72."""
import time
import numpy as np
import random

n, m, p = 3, 4, 6

_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _time_ok():
    return _time_left() > 5.0

def _build_schoolbook():
    """Build the rank-72 schoolbook decomposition. Returns (U, V, W) as numpy int arrays."""
    rank = n * m * p  # 72
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def _compute_contributions(U, V, W):
    """
    Compute the vectorized tensor contribution for each column r.
    Returns a matrix of shape (n*m*p, rank) where each column r is 
    the vector v_r such that sum_r v_r = target_tensor.
    """
    R = U.shape[1]
    contributions = np.zeros((n * m * p, R), dtype=int)
    
    for r in range(R):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        
        for i in range(n):
            for j in range(m):
                u_val = u_col[i * m + j]
                if u_val == 0: continue
                for k in range(p):
                    v_val = v_col[j * p + k]
                    if v_val == 0: continue
                    w_val = w_col[k * n + i]
                    if w_val == 0: continue
                    val = u_val * v_val * w_val
                    contributions[i * m * p + j * p + k, r] = val
    
    return contributions

def _exact_rank_reduction(contrib_matrix):
    """
    Perform exact integer Gaussian elimination to find a minimal set of linearly independent columns.
    Returns a list of column indices that form a basis.
    """
    # Work with fractions for exact arithmetic
    from fractions import Fraction
    
    n_rows, n_cols = contrib_matrix.shape
    
    # Convert to Fraction matrix
    mat = [[Fraction(contrib_matrix[i, j]) for j in range(n_cols)] for i in range(n_rows)]
    
    # Track which original columns are in the basis
    basis_cols = []
    pivot_rows = []
    
    # Gaussian elimination
    row = 0
    for col in range(n_cols):
        if row >= n_rows:
            break
        
        # Find a non-zero entry in this column from row onwards
        pivot_idx = None
        for i in range(row, n_rows):
            if mat[i][col] != 0:
                pivot_idx = i
                break
        
        if pivot_idx is None:
            continue
        
        # Swap rows
        mat[row], mat[pivot_idx] = mat[pivot_idx], mat[row]
        
        # Normalize the pivot row
        pivot_val = mat[row][col]
        for j in range(col, n_cols):
            mat[row][j] = mat[row][j] / pivot_val
        
        # Eliminate this column from other rows
        for i in range(n_rows):
            if i != row and mat[i][col] != 0:
                factor = mat[i][col]
                for j in range(col, n_cols):
                    mat[i][j] = mat[i][j] - factor * mat[row][j]
        
        basis_cols.append(col)
        pivot_rows.append(row)
        row += 1
    
    return basis_cols

def _verify(U, V, W):
    """Check exactness and return (ok, effective_rank)."""
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    
    # Reconstruct tensor
    recon = np.zeros((n, m, p), dtype=int)
    for r in range(U.shape[1]):
        if not (np.any(U[:, r]) and np.any(V[:, r]) and np.any(W[:, r])):
            continue
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        for i in range(n):
            for j in range(m):
                u_val = u_col[i * m + j]
                if u_val == 0: continue
                for k in range(p):
                    v_val = v_col[j * p + k]
                    if v_val == 0: continue
                    w_val = w_col[k * n + i]
                    if w_val == 0: continue
                    recon[i, j, k] += u_val * v_val * w_val
    
    target = np.ones((n, m, p), dtype=int)
    ok = np.array_equal(recon, target)
    
    # Count effective rank
    rank = 0
    for r in range(U.shape[1]):
        if np.any(U[:, r]) and np.any(V[:, r]) and np.any(W[:, r]):
            rank += 1
            
    return ok, rank

def _try_strassen_base():
    """Build the rank-66 Strassen-based decomposition as a fallback."""
    rank = 66
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    col = 0

    def add_schoolbook(i, j, k):
        nonlocal col
        U[i * m + j, col] = 1
        V[j * p + k, col] = 1
        W[k * n + i, col] = 1
        col += 1

    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        a = i_base * m + j_base
        b = i_base * m + (j_base + 1)
        c = (i_base + 1) * m + j_base
        d = (i_base + 1) * m + (j_base + 1)
        e = j_base * p + k_base
        f = j_base * p + (k_base + 1)
        g = (j_base + 1) * p + k_base
        h = (j_base + 1) * p + (k_base + 1)
        P = k_base * n + i_base
        Q = (k_base + 1) * n + i_base
        R = k_base * n + (i_base + 1)
        S = (k_base + 1) * n + (i_base + 1)

        # P1 = (a+d)(e+h)
        U[a, col] = 1; U[d, col] = 1
        V[e, col] = 1; V[h, col] = 1
        W[P, col] = 1; W[S, col] = 1
        col += 1
        # P2 = (c+d)e
        U[c, col] = 1; U[d, col] = 1
        V[e, col] = 1
        W[R, col] = 1; W[S, col] = -1
        col += 1
        # P3 = a(f-h)
        U[a, col] = 1
        V[f, col] = 1; V[h, col] = -1
        W[Q, col] = 1; W[S, col] = 1
        col += 1
        # P4 = d(g-e)
        U[d, col] = 1
        V[g, col] = 1; V[e, col] = -1
        W[P, col] = 1; W[R, col] = 1
        col += 1
        # P5 = (a+b)h
        U[a, col] = 1; U[b, col] = 1
        V[h, col] = 1
        W[P, col] = -1; W[Q, col] = 1
        col += 1
        # P6 = (c-a)(e+f)
        U[c, col] = 1; U[a, col] = -1
        V[e, col] = 1; V[f, col] = 1
        W[S, col] = 1
        col += 1
        # P7 = (b-d)(g+h)
        U[b, col] = 1; U[d, col] = -1
        V[g, col] = 1; V[h, col] = 1
        W[P, col] = 1
        col += 1

    for j in range(m):
        for k in range(p):
            add_schoolbook(0, j, k)

    for k_block in range(p // 2):
        for j_block in range(m // 2):
            add_strassen(1, 2 * j_block, 2 * k_block)

    return U, V, W

def construct():
    # Strategy 1: Try direct reduction of schoolbook rank-72
    school_U, school_V, school_W = _build_schoolbook()
    
    # Compute contributions
    contribs = _compute_contributions(school_U, school_V, school_W)
    
    # Find basis columns
    basis = _exact_rank_reduction(contribs)
    basis_rank = len(basis)
    
    # Create reduced matrices
    reduced_U = np.zeros((n * m, basis_rank), dtype=int)
    reduced_V = np.zeros((m * p, basis_rank), dtype=int)
    reduced_W = np.zeros((p * n, basis_rank), dtype=int)
    
    for new_col, old_col in enumerate(basis):
        reduced_U[:, new_col] = school_U[:, old_col]
        reduced_V[:, new_col] = school_V[:, old_col]
        reduced_W[:, new_col] = school_W[:, old_col]
    
    # Verify
    ok, actual_rank = _verify(reduced_U, reduced_V, reduced_W)
    
    if ok:
        # Check if we can do even better with Strassen base
        strassen_U, strassen_V, strassen_W = _try_strassen_base()
        strassen_ok, strassen_rank = _verify(strassen_U, strassen_V, strassen_W)
        
        if strassen_ok and strassen_rank < basis_rank:
            return {"U": strassen_U.tolist(), "V": strassen_V.tolist(), "W": strassen_W.tolist()}
        
        return {"U": reduced_U.tolist(), "V": reduced_V.tolist(), "W": reduced_W.tolist()}
    
    # Fallback to Strassen if reduction failed
    strassen_U, strassen_V, strassen_W = _try_strassen_base()
    strassen_ok, strassen_rank = _verify(strassen_U, strassen_V, strassen_W)
    
    if strassen_ok:
        return {"U": strassen_U.tolist(), "V": strassen_V.tolist(), "W": strassen_W.tolist()}
    
    # Last resort: return schoolbook
    return {"U": school_U.tolist(), "V": school_V.tolist(), "W": school_W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
