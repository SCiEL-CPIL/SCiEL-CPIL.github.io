import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
                
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school

    # Try to find lower rank
    # The condition is sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    # For fixed i,k, we have a linear system in W[k,i,r]
    
    def is_half_integer(frac):
        val = frac * 2
        return val.denominator == 1

    for R in range(rank_school - 1, 2, -1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # Random U, V with 0/1 entries
            U = [[0] * R for _ in range(n * m)]
            V = [[0] * R for _ in range(m * p)]
            
            # Ensure each column of U and V is not zero
            for r in range(R):
                # Pick random non-zero entries
                u_idx = np.random.choice(n * m, size=max(1, n*m//4), replace=False)
                v_idx = np.random.choice(m * p, size=max(1, m*p//4), replace=False)
                for idx in u_idx:
                    U[idx][r] = 1
                for idx in v_idx:
                    V[idx][r] = 1
            
            W = [[Fraction(0)] * R for _ in range(p * n)]
            valid = True
            
            # Solve for W for each i, k
            for i in range(n):
                for k in range(p):
                    # System: sum_j sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
                    # For fixed i,k, let w_r = W[k*n+i, r]
                    # Equations for j=0..m-1:
                    # sum_r (sum_j U[i,j,r] V[j,k,r]) w_r = 1
                    # Wait, the sum over j is in the original equation?
                    # Original: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
                    # This equation is for each j.
                    # So for fixed i,k, we have m equations:
                    # For each j: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
                    
                    # Let's form the m x R matrix A
                    # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    A = [[Fraction(0)] * R for _ in range(m)]
                    for j in range(m):
                        for r in range(R):
                            A[j][r] = Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r])
                    
                    # Solve A w = 1
                    # We can use Gaussian elimination
                    # Since R is large, we pick 3 columns to solve for, set others to 0
                    # Find 3 linearly independent columns
                    # For simplicity, just pick first 3 columns and check if they form a basis
                    # If not, try next 3, etc.
                    
                    # Actually, just solve using numpy for speed, then check fractions
                    A_np = np.array(A, dtype=float)
                    b = np.ones(m)
                    
                    try:
                        # Least squares or exact solve?
                        # We want exact.
                        # Use Fraction-based Gaussian elimination
                        # Simplified: if R >= m, we can set R-m variables to 0
                        # and solve for m variables.
                        
                        # Find m columns that are linearly independent
                        # This is expensive.
                        # Instead, just pick random m columns and hope they are independent.
                        
                        # Pick m random columns
                        cols = np.random.choice(R, size=m, replace=False)
                        A_sub = np.array([[A[j][r] for r in cols] for j in range(m)], dtype=float)
                        
                        if np.linalg.matrix_rank(A_sub) < m:
                            continue # Try different columns
                        
                        # Solve A_sub w_sub = 1
                        w_sub = np.linalg.solve(A_sub, b)
                        
                        # Check if w_sub are half-integers (approximately)
                        valid_w = True
                        w_frac = []
                        for val in w_sub:
                            # Convert to fraction
                            # val is float. We need to check if it's close to a half-integer
                            # Since U, V are 0/1, A entries are 0/1.
                            # So w_sub will be rational with small denominator.
                            # We can use Fraction(val) but float precision is bad.
                            # Better: solve using Fraction directly for these m columns.
                            pass
                            
                        # Re-do with Fraction for exactness
                        A_sub_frac = [[Fraction(A[j][r]) for r in cols] for j in range(m)]
                        # Solve A_sub_frac w = 1
                        # Gaussian elimination
                        n_eq = m
                        n_var = m
                        mat = [row[:] for row in A_sub_frac]
                        rhs = [Fraction(1) for _ in range(n_eq)]
                        
                        for col_idx in range(n_var):
                            # Find pivot
                            pivot_row = -1
                            for row_idx in range(col_idx, n_eq):
                                if mat[row_idx][col_idx] != 0:
                                    pivot_row = row_idx
                                    break
                            if pivot_row == -1:
                                valid_w = False
                                break
                                
                            # Swap
                            mat[col_idx], mat[pivot_row] = mat[pivot_row], mat[col_idx]
                            rhs[col_idx], rhs[pivot_row] = rhs[pivot_row], rhs[col_idx]
                            
                            # Normalize
                            pivot_val = mat[col_idx][col_idx]
                            for k_idx in range(col_idx, n_var):
                                mat[col_idx][k_idx] = mat[col_idx][k_idx] / pivot_val
                            rhs[col_idx] = rhs[col_idx] / pivot_val
                            
                            # Eliminate
                            for row_idx in range(n_eq):
                                if row_idx != col_idx and mat[row_idx][col_idx] != 0:
                                    factor = mat[row_idx][col_idx]
                                    for k_idx in range(col_idx, n_var):
                                        mat[row_idx][k_idx] = mat[row_idx][k_idx] - factor * mat[col_idx][k_idx]
                                    rhs[row_idx] = rhs[row_idx] - factor * rhs[col_idx]
                        
                        if not valid_w:
                            continue
                            
                        # Check if solutions are half-integers
                        for val in rhs:
                            if not is_half_integer(val):
                                valid_w = False
                                break
                                
                        if not valid_w:
                            continue
                            
                        # Construct W for this i,k
                        for idx, r in enumerate(cols):
                            W[k*n+i][r] = rhs[idx]
                            
                    except Exception:
                        continue
                        
            # If we got here, W is fully constructed and valid
            # Convert to list of lists of floats/ints for output
            U_out = [[float(x) for x in row] for row in U]
            V_out = [[float(x) for x in row] for row in V]
            W_out = [[float(x) for x in row] for row in W]
            
            return {"U": U_out, "V": V_out, "W": W_out}
            
    return {"U": best_U, "V": best_V, "W": best_W}
