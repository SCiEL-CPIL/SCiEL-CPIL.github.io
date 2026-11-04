import time
import fractions
import numpy as np

DIMENSIONS = [3, 3, 4]
Frac = fractions.Fraction

def _validate(U, V, W, n, m, p, rank):
    """Exact check that the tensor decomposition holds."""
    U_frac = [[Frac(U[i][c]) for c in range(rank)] for i in range(n * m)]
    V_frac = [[Frac(V[j][c]) for c in range(rank)] for j in range(m * p)]
    W_frac = [[Frac(W[k][c]) for c in range(rank)] for k in range(p * n)]
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(m):
                    s = Frac(0)
                    for c in range(rank):
                        s += U_frac[i * m + j][c] * V_frac[l * p + k][c] * W_frac[k * n + i][c]
                    expected = Frac(1) if j == l else Frac(0)
                    if s != expected:
                        return False
    return True

def _schoolbook(n, m, p):
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

def _try_random_search(n, m, p, target_rank, time_budget):
    """
    Randomized search for a tensor decomposition of rank target_rank.
    Strategy: Randomly initialize U and V with entries in {-1, 0, 1}.
    For each (i, k) pair, we have a linear system for W_row = W[k*n + i].
    The system is: sum_c U[i*m+j][c] * V[l*p+k][c] * W[c] = delta_{j,l}
    This is a system of m equations (for j) with m unknowns? No.
    The tensor equation is:
    sum_c U_{i,j,c} V_{l,k,c} W_{k,i,c} = delta_{j,l}
    
    For fixed (i, k), let W_{k,i,c} = w_c.
    Then for all j, l:
    sum_c w_c * (U_{i,j,c} * V_{l,k,c}) = delta_{j,l}
    
    This is a linear system in w_c:
    M_{(j,l), c} = U_{i,j,c} * V_{l,k,c}
    b_{(j,l)} = delta_{j,l}
    
    The system has m*m equations and target_rank unknowns.
    We need to check if a solution w exists.
    Since we want integer/half-integer solutions, we can solve over rationals.
    However, solving a potentially underdetermined or inconsistent system for 12 (i,k) pairs
    for every random trial is expensive.
    
    Optimization:
    1. Generate random U, V.
    2. For one (i, k) pair, check consistency. If inconsistent, reject.
    3. If consistent, find a particular solution w (e.g., using least squares over rationals, or just check rank).
       Actually, we just need existence.
    4. If all (i, k) pairs are consistent, we have a candidate. We then need to extract w values.
       We can solve for w for each (i, k) independently.
       If the system is consistent, a solution exists. We can pick the minimum norm rational solution or just any rational solution.
       If the solution has half-integer entries, we are good.
    
    Let's implement a fast consistency check using Gaussian elimination over Fractions.
    """
    start_time = time.time()
    n, m, p = 3, 3, 4
    
    # Pre-compute indices
    iks = [(i, k) for i in range(n) for k in range(p)]
    
    while time.time() - start_time < time_budget:
        # Generate random U, V
        # Using small integers increases chance of small solutions
        U = [[1 if np.random.rand() < 0.3 else (0 if np.random.rand() < 0.5 else -1) for _ in range(target_rank)] for _ in range(n * m)]
        V = [[1 if np.random.rand() < 0.3 else (0 if np.random.rand() < 0.5 else -1) for _ in range(target_rank)] for _ in range(m * p)]
        
        valid = True
        W_candidate = [[0] * target_rank for _ in range(p * n)]
        
        for (i, k) in iks:
            # Build system M w = b
            # M has m*m rows, target_rank cols
            # M[(j, l)] = U[i*m+j] * V[l*p+k] (element-wise product, flattened?)
            # Wait, the equation is:
            # sum_c U_{i,j,c} V_{l,k,c} W_{k,i,c} = delta_{j,l}
            # Let w_c = W_{k,i,c}
            # For a fixed (i, k), we have a system:
            # For each j (row of A), and each l (row of B):
            # sum_c (U[i*m+j][c] * V[l*p+k][c]) * w_c = delta_{j,l}
            
            # This is m*m equations.
            # Let's build the augmented matrix [M | b]
            # M is m*m x target_rank
            # b is m*m x 1
            
            # We can solve this using Gaussian elimination.
            # To speed up, we can check if the rank of M equals the rank of [M|b].
            
            # Let's use a helper function for rank and consistency
            # Since target_rank is small (e.g., 24), m*m = 9.
            
            # Build rows
            rows = []
            for j in range(m):
                for l in range(m):
                    row = [Frac(U[i*m+j][c]) * Frac(V[l*p+k][c]) for c in range(target_rank)]
                    b_val = Frac(1) if j == l else Frac(0)
                    rows.append(row + [b_val])
            
            # Gaussian elimination to check consistency
            # We want to see if the system is consistent.
            # Perform row reduction.
            matrix = [list(r) for r in rows]
            rank_m = 0
            rank_aug = 0
            num_rows = len(matrix)
            num_cols = target_rank
            num_aug_cols = target_rank + 1
            
            # Standard Gaussian elimination
            for col in range(num_cols):
                # Find pivot
                pivot_row = -1
                for r in range(rank_m, num_rows):
                    if matrix[r][col] != 0:
                        pivot_row = r
                        break
                
                if pivot_row == -1:
                    continue
                
                # Swap
                matrix[rank_m], matrix[pivot_row] = matrix[pivot_row], matrix[rank_m]
                
                # Normalize pivot
                pivot_val = matrix[rank_m][col]
                if pivot_val != 1:
                    matrix[rank_m] = [x / pivot_val for x in matrix[rank_m]]
                
                # Eliminate
                for r in range(num_rows):
                    if r == rank_m:
                        continue
                    factor = matrix[r][col]
                    if factor != 0:
                        matrix[r] = [matrix[r][c] - factor * matrix[rank_m][c] for c in range(num_aug_cols)]
                
                rank_m += 1
            
            # Check consistency: Any row with 0s in M part but non-zero in b part?
            for r in range(rank_m, num_rows):
                if matrix[r][num_aug_cols - 1] != 0:
                    valid = False
                    break
            
            if not valid:
                break
            
            # If consistent, we need to find a solution w.
            # We can extract it from the reduced matrix.
            # Since it might be underdetermined, we set free variables to 0.
            # Find the pivot columns
            pivot_cols = []
            temp_matrix = [list(r) for r in rows]
            # Redo elimination to track pivots? Or just use the reduced matrix.
            # The reduced matrix `matrix` is in RREF (mostly).
            # Let's just solve it simply.
            # We know it's consistent.
            # Set free vars to 0.
            w = [Frac(0)] * target_rank
            # We need to know which columns are pivots.
            # Let's re-do RREF properly to get pivot indices.
            
            # Actually, we can just use numpy to find a least-squares solution over reals,
            # then check if it's rational and has small denominator.
            # But we need EXACT match.
            
            # Better: Use the reduced matrix.
            # In the `matrix` from above, rows 0..rank_m-1 are pivot rows.
            # But we didn't track pivot columns explicitly in that loop.
            
            # Let's do a cleaner RREF.
            mat = [list(r) for r in rows]
            r = 0
            pivot_cols = []
            for c in range(target_rank):
                if r >= num_rows: break
                # Find pivot in column c from row r downwards
                pivot_r = -1
                for i in range(r, num_rows):
                    if mat[i][c] != 0:
                        pivot_r = i
                        break
                if pivot_r == -1:
                    continue
                
                mat[r], mat[pivot_r] = mat[pivot_r], mat[r]
                pivot_val = mat[r][c]
                mat[r] = [x / pivot_val for x in mat[r]]
                
                for i in range(num_rows):
                    if i == r: continue
                    factor = mat[i][c]
                    if factor != 0:
                        mat[i] = [mat[i][j] - factor * mat[r][j] for j in range(num_aug_cols)]
                
                pivot_cols.append(c)
                r += 1
            
            # Check consistency again on this mat
            consistent = True
            for i in range(r, num_rows):
                if mat[i][num_aug_cols - 1] != 0:
                    consistent = False
                    break
            if not consistent:
                valid = False
                break
            
            # Extract solution
            # Pivot columns have value 1 in their row, 0 elsewhere.
            # Free columns are 0 in pivot rows.
            # w[c] = mat[i][num_aug_cols-1] if c is the pivot for row i.
            # w[c] = 0 for free columns.
            w_sol = [Frac(0)] * target_rank
            for i, pc in enumerate(pivot_cols):
                w_sol[pc] = mat[i][num_aug_cols - 1]
            
            # Check if w_sol has half-integer entries
            ok = True
            for val in w_sol:
                if val * 2 != int(val * 2):
                    ok = False
                    break
            if not ok:
                valid = False
                break
            
            # Store W values
            idx = k * n + i
            for c in range(target_rank):
                W_candidate[idx][c] = w_sol[c]
        
        if valid:
            # We found a candidate!
            # Convert to int if possible
            U_int = [[int(x) for x in row] for row in U]
            V_int = [[int(x) for x in row] for row in V]
            W_int = [[int(x) for x in row] for row in W_candidate]
            
            # Final validation
            if _validate(U_int, V_int, W_int, n, m, p, target_rank):
                return {"U": U_int, "V": V_int, "W": W_int}
            
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Try rank 24 first (if Lader-like structure existed for 3x3x4, or just lucky)
    # Actually, rank 24 is 36 * 2/3.
    # Let's try a few ranks.
    
    for target_rank in [24, 25, 26]:
        if time.time() > deadline - 20:
            break
        result = _try_random_search(n, m, p, target_rank, time_budget=10)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
