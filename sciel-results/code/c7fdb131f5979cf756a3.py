import time
import numpy as np
from fractions import Fraction

N, M, P = 3, 4, 4
NM, MP, PN = N * M, M * P, P * N

def _verify(U, V, W, tol=1e-10):
    """
    Verify the tensor decomposition using floating point arithmetic.
    T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] should be 1.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float).reshape(N, M, rank)
    V_arr = np.array(V, dtype=float).reshape(M, P, rank)
    W_arr = np.array(W, dtype=float).reshape(P, N, rank)
    
    # Compute the tensor T
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err

def _strassen_44():
    """
    Constructs a rank-44 decomposition for 3x4 * 4x4 matrix multiplication.
    This uses Strassen's algorithm for the top 2x2 block and standard multiplication for the rest.
    """
    rank = 44
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    
    # Strassen steps for 2x2 block multiplication
    # Each step is (U_coeffs, V_coeffs, W_coeffs)
    # Indices are local (0,1)
    strassen_steps = [
        # M1 = (X00 + X11)(Y00 + Y11) -> Z00 += 1, Z11 += 1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (X10 + X11)Y00 -> Z10 += 1, Z11 -= 1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = X00(Y01 - Y11) -> Z01 += 1, Z11 += 1
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = X11(Y10 - Y00) -> Z00 += 1, Z10 += 1
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (X00 + X01)Y11 -> Z01 += 1, Z00 -= 1
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        # M6 = (X10 - X00)(Y00 + Y01) -> Z11 += 1
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (X01 - X11)(Y10 + Y11) -> Z00 += 1
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    blocks = [
        (0, 0, 0, 0, 0, 0),
        (0, 2, 2, 0, 0, 0),
        (0, 0, 0, 2, 0, 2),
        (0, 2, 2, 2, 0, 2)
    ]
    
    col = 0
    for (xi, xj, yj, yk, zi, zk) in blocks:
        for u_c, v_c, w_c in strassen_steps:
            for (a, b), val in u_c.items():
                U[(a + xi) * M + (b + xj)][col] += val
            for (b, c), val in v_c.items():
                V[(b + yj) * P + (c + yk)][col] += val
            for (a, c), val in w_c.items():
                W[(c + zk) * N + (a + zi)][col] += val
            col += 1
    
    # Naive part for the bottom row of C (i=2)
    # C[2, k] = sum_j A[2, j] * B[j, k]
    for j in range(M):
        for k in range(P):
            U[2 * M + j][col] = 1
            V[j * P + k][col] = 1
            W[k * N + 2][col] = 1
            col += 1
    
    assert col == rank, f"Rank mismatch: {col} vs {rank}"
    return {'U': U, 'V': V, 'W': W}

def _gaussian_elim_rational(matrix, deadline):
    """
    Perform Gaussian elimination over rationals.
    Returns the row-echelon form and the list of pivot columns.
    matrix is a list of lists of Fractions.
    """
    if not matrix or not matrix[0]:
        return matrix, []
    
    nrows = len(matrix)
    ncols = len(matrix[0])
    M = [row[:] for row in matrix]
    pivot_cols = []
    row_idx = 0
    
    for col in range(ncols):
        if row_idx >= nrows:
            break
        if time.time() > deadline:
            break
        
        # Find pivot
        pivot = None
        for r in range(row_idx, nrows):
            if M[r][col] != 0:
                pivot = r
                break
        
        if pivot is None:
            continue
        
        # Swap rows
        M[row_idx], M[pivot] = M[pivot], M[row_idx]
        
        # Normalize pivot row
        pv = M[row_idx][col]
        if pv != 1:
            M[row_idx] = [x / pv for x in M[row_idx]]
        
        # Eliminate other rows
        for r in range(nrows):
            if r == row_idx:
                continue
            factor = M[r][col]
            if factor != 0:
                M[r] = [M[r][c] - factor * M[row_idx][c] for c in range(ncols)]
        
        pivot_cols.append(col)
        row_idx += 1
    
    return M, pivot_cols

def _find_null_vector(U_frac, V_frac, W_frac, deadline):
    """
    Find a non-zero vector alpha such that U*alpha = 0, V*alpha = 0, W*alpha = 0.
    Uses the stacked matrix approach.
    Returns the index of a column that can be eliminated and the coefficients, or None.
    """
    rank = len(U_frac[0])
    
    # Stack U, V, W vertically
    combined = []
    for row in U_frac:
        combined.append(row[:])
    for row in V_frac:
        combined.append(row[:])
    for row in W_frac:
        combined.append(row[:])
    
    # Find the rank and identify free variables
    echelon, pivot_cols = _gaussian_elim_rational(combined, deadline)
    
    if time.time() > deadline:
        return None
    
    if len(pivot_cols) == rank:
        return None  # Full rank, no dependencies
    
    # Free variables are those not in pivot_cols
    free_vars = [c for c in range(rank) if c not in pivot_cols]
    
    if not free_vars:
        return None
    
    # For each free variable, find a null vector by setting that free var to 1
    # and solving for pivot variables
    for fv in free_vars:
        if time.time() > deadline:
            break
        
        alpha = [Fraction(0)] * rank
        alpha[fv] = Fraction(1)
        
        # Back-substitute: for each pivot row (in reverse order of pivot_cols)
        for pi in reversed(range(len(pivot_cols))):
            col = pivot_cols[pi]
            # Row pi in echelon form: M[pi][col] = 1, and other entries
            # alpha[col] = -sum_{c > col} echelon[pi][c] * alpha[c]
            val = Fraction(0)
            for c in range(col + 1, rank):
                if alpha[c] != 0:
                    val += echelon[pi][c] * alpha[c]
            alpha[col] = -val
        
        # Verify: check that combined * alpha = 0
        valid = True
        for r in range(len(combined)):
            s = Fraction(0)
            for c in range(rank):
                if combined[r][c] != 0 and alpha[c] != 0:
                    s += combined[r][c] * alpha[c]
            if s != 0:
                valid = False
                break
        
        if valid:
            return fv, alpha
    
    return None

def _try_reduce_by_linear_dependency(U, V, W, deadline):
    """
    Attempts to reduce rank by finding linear dependencies among columns
    using exact rational arithmetic.
    """
    rank = len(U[0])
    best_rank = rank
    
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    deadline = deadline - 2.0  # Safety margin
    
    max_iterations = 20
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        result = _find_null_vector(U_frac, V_frac, W_frac, deadline)
        if result is None:
            break
        
        elim_col, alpha = result
        
        # Eliminate column elim_col by adding alpha*column to other columns
        # Actually, we want to express column elim_col as a combination of others.
        # Since alpha[elim_col] = 1, we have:
        # column_elim_col = -sum_{c != elim_col} alpha[c] * column_c
        # So we can eliminate column elim_col by updating the other columns.
        
        # New columns: for c != elim_col, new_col_c = col_c + alpha[c] * col_elim_col
        # Then remove column elim_col.
        
        new_U = []
        new_V = []
        new_W = []
        
        for i in range(NM):
            new_row = []
            for c in range(rank):
                if c == elim_col:
                    continue
                new_val = U_frac[i][c] + alpha[c] * U_frac[i][elim_col]
                new_row.append(new_val)
            new_U.append(new_row)
        
        for i in range(MP):
            new_row = []
            for c in range(rank):
                if c == elim_col:
                    continue
                new_val = V_frac[i][c] + alpha[c] * V_frac[i][elim_col]
                new_row.append(new_val)
            new_V.append(new_row)
        
        for i in range(PN):
            new_row = []
            for c in range(rank):
                if c == elim_col:
                    continue
                new_val = W_frac[i][c] + alpha[c] * W_frac[i][elim_col]
                new_row.append(new_val)
            new_W.append(new_row)
        
        U_frac = new_U
        V_frac = new_V
        W_frac = new_W
        best_rank -= 1
    
    if best_rank == rank:
        return {'U': U, 'V': V, 'W': W}, rank
    
    return {
        'U': [[float(x) for x in row] for row in U_frac],
        'V': [[float(x) for x in row] for row in V_frac],
        'W': [[float(x) for x in row] for row in W_frac]
    }, best_rank

def construct():
    start = time.time()
    deadline = start + 140.0  # 10s safety margin
    
    # 1. Generate the base rank-44 decomposition
    result = _strassen_44()
    
    # 2. Verify it
    ok, err = _verify(result['U'], result['V'], result['W'])
    if not ok:
        # Fallback to schoolbook if something went wrong (should not happen)
        rank = NM * MP * PN
        U = [[0] * rank for _ in range(NM)]
        V = [[0] * rank for _ in range(MP)]
        W = [[0] * rank for _ in range(PN)]
        c = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j][c] = 1
                    V[j * P + k][c] = 1
                    W[k * N + i][c] = 1
                    c += 1
        return {'U': U, 'V': V, 'W': W}
    
    best_result = result
    best_rank = len(result['U'][0])
    
    # 3. Try to reduce rank using linear dependency elimination
    if time.time() < deadline - 5.0:
        reduced_result, reduced_rank = _try_reduce_by_linear_dependency(
            result['U'], result['V'], result['W'], deadline
        )
        if reduced_rank < best_rank:
            ok, err = _verify(reduced_result['U'], reduced_result['V'], reduced_result['W'])
            if ok:
                best_result = reduced_result
                best_rank = reduced_rank
    
    return best_result

if __name__ == "__main__":
    # Simple self-test
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
    ok, err = _verify(res['U'], res['V'], res['W'])
    print(f"Valid: {ok}, Max Error: {err}")
