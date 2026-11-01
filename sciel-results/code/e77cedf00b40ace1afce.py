import numpy as np
from fractions import Fraction
import time
import random

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_schoolbook = nm * mp // p # n*m*p
    # Actually rank is n*m*p
    rank_schoolbook = n * m * p
    
    # Deadline guard
    start_time = time.time()
    timeout = 110.0 # Leave 10s margin
    
    # Fallback: Schoolbook solution
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_solution = None
    best_rank = rank_schoolbook
    
    # Try to find a solution with rank < 24
    for r in range(rank_schoolbook - 1, 14, -1):
        if time.time() - start_time > timeout:
            break
            
        # Try random sparse U, V
        # Number of trials
        trials = 500
        for _ in range(trials):
            if time.time() - start_time > timeout:
                break
                
            # Generate random U, V
            # Entries in {-1, 0, 1}
            # Ensure some density
            U = [[0] * r for _ in range(nm)]
            V = [[0] * r for _ in range(mp)]
            
            for i in range(nm):
                for c in range(r):
                    if random.random() < 0.3:
                        U[i][c] = random.choice([-1, 1])
                        
            for j in range(mp):
                for c in range(r):
                    if random.random() < 0.3:
                        V[j][c] = random.choice([-1, 1])
                        
            # Build the linear system for W
            # Variables: w_{k, i, c} for k in 0..p-1, i in 0..n-1, c in 0..r-1
            # Total vars: p * n * r
            num_vars = p * n * r
            num_eqs = n * m * p
            
            # Matrix A (num_eqs x num_vars)
            # Vector b (num_eqs)
            A = [[Fraction(0) for _ in range(num_vars)] for _ in range(num_eqs)]
            b = [Fraction(1) for _ in range(num_eqs)]
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Equation: sum_c u_{i,j,c} v_{j,k,c} w_{k,i,c} = 1
                        # Variable index for w_{k,i,c} is (k * n + i) * r + c
                        for c in range(r):
                            u_val = Fraction(U[i * m + j][c])
                            v_val = Fraction(V[j * p + k][c])
                            coeff = u_val * v_val
                            if coeff != 0:
                                var_idx = (k * n + i) * r + c
                                A[eq_idx][var_idx] += coeff
                        eq_idx += 1
                        
            # Solve A W = b using Gaussian Elimination with Fractions
            # We only care if a solution exists and if the particular solution (free vars = 0) has half-integer entries
            W_solution = solve_linear_system(A, b)
            
            if W_solution is not None:
                # Check if W_solution entries are half-integers
                valid = True
                for val in W_solution:
                    # val is Fraction
                    if val.denominator not in [1, 2]:
                        valid = False
                        break
                
                if valid:
                    # Construct U, V, W matrices
                    W_mat = [[0] * r for _ in range(pn)]
                    for k in range(p):
                        for i in range(n):
                            for c in range(r):
                                W_mat[k * n + i][c] = W_solution[(k * n + i) * r + c]
                    
                    best_solution = {"U": U, "V": V, "W": W_mat}
                    best_rank = r
                    # Found a solution, can we find a better one?
                    # Continue to lower ranks
                    break # Break inner trial loop, continue to next r
    
    if best_solution is not None:
        return best_solution
    else:
        return get_schoolbook()

def solve_linear_system(A, b):
    """
    Solves A x = b for x.
    Returns x if a solution exists, else None.
    Uses Gaussian Elimination.
    If the system is underdetermined, returns the particular solution with free variables set to 0.
    """
    n_rows = len(A)
    n_cols = len(A[0])
    
    # Augmented matrix
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    
    # Gaussian Elimination
    rank = 0
    for col in range(n_cols):
        # Find pivot
        pivot_row = -1
        for row in range(rank, n_rows):
            if M[row][col] != 0:
                pivot_row = row
                break
        
        if pivot_row == -1:
            continue
            
        # Swap
        M[rank], M[pivot_row] = M[pivot_row], M[rank]
        
        # Normalize pivot row
        pivot_val = M[rank][col]
        for j in range(col, n_cols + 1):
            M[rank][j] /= pivot_val
            
        # Eliminate other rows
        for row in range(n_rows):
            if row == rank:
                continue
            factor = M[row][col]
            if factor != 0:
                for j in range(col, n_cols + 1):
                    M[row][j] -= factor * M[rank][j]
                    
        rank += 1
        
    # Check for inconsistency
    for row in range(rank, n_rows):
        # Check if row is [0, ..., 0, non-zero]
        if M[row][n_cols] != 0:
            return None
            
    # Extract solution
    # Free variables are set to 0
    x = [Fraction(0) for _ in range(n_cols)]
    pivot_cols = []
    for col in range(n_cols):
        # Find if col is a pivot column
        # In reduced row echelon form, the pivot is the first non-zero in the row
        # We need to track which columns are pivots
        pass
        
    # Re-do to track pivot columns properly
    # Since we already did elimination, let's just back-substitute from the reduced matrix
    # But we need to know which variables are free.
    # In RREF, the columns with leading 1s are pivot columns.
    
    # Let's identify pivot columns from M
    pivot_cols = []
    for row in range(rank):
        for col in range(n_cols):
            if M[row][col] != 0:
                pivot_cols.append(col)
                break
                
    # Set pivot variables
    for row in range(rank):
        col = pivot_cols[row]
        # x[col] = M[row][n_cols] - sum (M[row][free_col] * x[free_col])
        # Since free vars are 0, x[col] = M[row][n_cols]
        x[col] = M[row][n_cols]
        
    return x

# EVOLVE-BLOCK-END
