import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

def _validate(U, V, W):
    """Check if U, V, W form a valid rank-R tensor decomposition."""
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    
    rank = len(V[0]) if V else 0
    U_arr = np.array(U, dtype=object)
    V_arr = np.array(V, dtype=object)
    W_arr = np.array(W, dtype=object)
    
    # Shapes:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if s != 1:
                    return False
    return True

def _is_valid_entry(x):
    """Check if entry is integer or half-integer."""
    if isinstance(x, Fraction):
        num, den = x.numerator, x.denominator
        return den in [1, 2]
    elif isinstance(x, int):
        return True
    elif isinstance(x, float):
        # Check if it's effectively an integer or half-integer
        x2 = x * 2
        return abs(x2 - round(x2)) < 1e-9
    return False

def _try_solve_for_W(U, V, R, n, m, p):
    """
    Given U and V of shape (n*m, R) and (m*p, R),
    solve for W of shape (p*n, R) such that
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    This is a linear system in the entries of W.
    Let W_flat be a vector of length (p*n)*R.
    The equations are linear in W.
    """
    # Number of unknowns: p * n * R
    # Number of equations: n * m * p
    
    # We can group equations by (i, k). For a fixed (i, k), the equation is:
    # sum_j sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    # Wait, W[k*n+i, c] is the same for all j in the inner sum?
    # No, the index of W is (k, i). The index of U is (i, j). The index of V is (j, k).
    # The term is U[i,j,c] * V[j,k,c] * W[k,i,c].
    # For a fixed (i, k), the variable is W[k,i,c] for each c.
    # The equation for (i, j, k) is:
    # sum_c ( U[i*m+j, c] * V[j*p+k, c] ) * W[k*n+i, c] = 1
    
    # This couples different j's? No, for a fixed (i, k), the LHS is a sum over c.
    # But the RHS is 1 for every j.
    # So for a fixed (i, k), we have m equations (one for each j):
    # sum_c A_{j,c} W_{c} = 1, where A_{j,c} = U[i*m+j, c] * V[j*p+k, c] and W_c = W[k*n+i, c].
    
    # So for each pair (i, k), we have a system of m equations in R unknowns (the vector W[k*n+i, :]).
    # If R >= m, we can solve it.
    # Here m=3, so we need R >= 3.
    
    W = [[Fraction(0) for _ in range(R)] for _ in range(p * n)]
    
    for k in range(p):
        for i in range(n):
            # Build system for W[k*n+i, :]
            # Equations: for j in range(m):
            # sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1
            
            # Matrix A of size m x R
            A = [[Fraction(0) for _ in range(R)] for _ in range(m)]
            for j in range(m):
                for c in range(R):
                    A[j][c] = Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c])
            
            # Solve A * W_vec = b, where b = [1, 1, 1]
            # We use Gaussian elimination over Fractions
            
            # Augmented matrix [A | b]
            M = [row[:] + [Fraction(1)] for row in A]
            
            # Gaussian Elimination
            rank = 0
            for col in range(R):
                # Find pivot
                pivot = -1
                for row in range(rank, m):
                    if M[row][col] != 0:
                        pivot = row
                        break
                if pivot == -1:
                    continue
                
                # Swap rows
                M[rank], M[pivot] = M[pivot], M[rank]
                
                # Normalize pivot row
                pivot_val = M[rank][col]
                for c in range(col, R + 1):
                    M[rank][c] /= pivot_val
                
                # Eliminate other rows
                for row in range(m):
                    if row != rank and M[row][col] != 0:
                        factor = M[row][col]
                        for c in range(col, R + 1):
                            M[row][c] -= factor * M[rank][c]
                
                rank += 1
            
            # Check for inconsistency
            for row in range(m):
                # If all coefficients are 0 but RHS is not 0, inconsistent
                is_zero_row = all(M[row][c] == 0 for c in range(R))
                if is_zero_row and M[row][R] != 0:
                    return None
            
            # Extract solution
            # If R > m, there are free variables. We set them to 0.
            # We need to check if the resulting W entries are valid (int/half-int)
            # Actually, we just need *a* solution.
            # The solution is not unique if R > m.
            # We can pick the solution with free variables = 0.
            
            W_vec = [Fraction(0) for _ in range(R)]
            # Back substitution or just read from reduced row echelon form
            # Since we did full elimination (Gauss-Jordan), the matrix is in RREF.
            # The columns corresponding to pivot positions give the solution directly if free vars are 0.
            
            # Identify pivot columns
            pivot_cols = []
            r = 0
            for c in range(R):
                if M[r][c] == 1 and all(M[rr][c] == 0 for rr in range(m) if rr != r):
                    # Check if this is a pivot column
                    # In RREF, pivot is 1, others 0
                    # We need to be careful with the loop above
                    pass
            
            # Simpler: Just solve A x = b using np.linalg.lstsq? No, needs exact.
            # Let's re-do the elimination to track pivots properly.
            
            # Reset M
            M = [row[:] + [Fraction(1)] for row in A]
            pivots = []
            r = 0
            for c in range(R):
                if r >= m:
                    break
                # Find pivot in column c
                pivot_row = -1
                for rr in range(r, m):
                    if M[rr][c] != 0:
                        pivot_row = rr
                        break
                if pivot_row == -1:
                    continue
                
                M[r], M[pivot_row] = M[pivot_row], M[r]
                pivots.append((r, c))
                
                # Normalize
                pv = M[r][c]
                for cc in range(c, R + 1):
                    M[r][cc] /= pv
                
                # Eliminate
                for rr in range(m):
                    if rr != r and M[rr][c] != 0:
                        f = M[rr][c]
                        for cc in range(c, R + 1):
                            M[rr][cc] -= f * M[r][cc]
                
                r += 1
            
            # Check consistency
            for rr in range(m):
                if all(M[rr][cc] == 0 for cc in range(R)) and M[rr][R] != 0:
                    return None
            
            # Construct solution
            # Free variables (columns not in pivots) are set to 0.
            W_vec = [Fraction(0) for _ in range(R)]
            for rr, cc in pivots:
                W_vec[cc] = M[rr][R]
            
            # Assign to W
            for c in range(R):
                W[k * n + i][c] = W_vec[c]
                
            # Check if entries are valid (int or half-int)
            # If not, this U,V pair failed for this (i,k)
            # Actually, if the solution has fractions with denom > 2, it's invalid for the "half-integer" constraint?
            # The task says "integer/half-integer entries".
            # If the exact solution requires 1/3, it's invalid.
            # So we must check.
            for c in range(R):
                if not _is_valid_entry(W[k * n + i][c]):
                    return None

    return W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Schoolbook baseline
    base_rank = n * m * p
    U_base = [[0] * base_rank for _ in range(n * m)]
    V_base = [[0] * base_rank for _ in range(m * p)]
    W_base = [[0] * base_rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U_base[i * m + j][c] = 1
                V_base[j * p + k][c] = 1
                W_base[k * n + i][c] = 1
    
    best_result = {"U": U_base, "V": V_base, "W": W_base}
    best_rank = base_rank
    
    # Try to find lower rank
    # We'll try ranks from 55 down to 40
    # For each rank, we try random sparse U and V, then solve for W.
    
    for R in range(55, 39, -1):
        if time.time() > deadline:
            break
            
        # Try a few random instances
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            # Generate random U and V with entries in {-1, 0, 1}
            # To ensure solvability, we want the submatrices A_{i,k} to be invertible or consistent.
            # Random sparse might be bad. Let's use random integers from {-2, -1, 1, 2}.
            
            U = [[random.randint(-2, 2) for _ in range(R)] for _ in range(n * m)]
            V = [[random.randint(-2, 2) for _ in range(R)] for _ in range(m * p)]
            
            # Convert to Fraction for exact arithmetic in solver
            U_frac = [[Fraction(x) for x in row] for row in U]
            V_frac = [[Fraction(x) for x in row] for row in V]
            
            W = _try_solve_for_W(U_frac, V_frac, R, n, m, p)
            
            if W is not None:
                # Verify the solution
                # Convert W to list of lists of numbers
                W_list = [[float(w) for w in row] for row in W]
                # Actually, keep as Fraction for validation?
                # _validate uses object, so it can handle Fraction.
                if _validate(U, V, W):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = R
                    break # Found one for this R, try lower R
        
    return best_result
