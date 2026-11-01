import time
import random
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=4
    
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
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_school
    
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    def validate(U, V, W, rank):
        """Check if U, V, W with given rank exactly decompose the multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                    if val != 1:
                        return False
        
        for a in range(n * m):
            i, j = divmod(a, m)
            for b in range(m * p):
                j2, k = divmod(b, p)
                if j2 != j:
                    for c in range(p * n):
                        val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                        if val != 0:
                            return False
                else:
                    valid_c = k * n + i
                    for c in range(p * n):
                        if c == valid_c:
                            continue
                        val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                        if val != 0:
                            return False
        
        return True

    def solve_for_W(U, V, r):
        """
        Given U and V, solve the linear system for W.
        Returns W as a list of lists if a solution exists, else None.
        """
        # Variables: W[c][r_idx] for c in [0, p*n), r_idx in [0, r)
        # Total variables: p * n * r
        num_vars = p * n * r
        
        # Build the linear system A * x = b
        # Each equation corresponds to a triple (a, b, c)
        # For valid (i,j,k): a=i*m+j, b=j*p+k, c=k*n+i -> sum_r U[a][r]*V[b][r]*W[c][r] = 1
        # For invalid: sum_r U[a][r]*V[b][r]*W[c][r] = 0
        
        equations = []
        rhs = []
        
        # Generate all (a, b, c) combinations
        for a in range(n * m):
            i_a, j_a = divmod(a, m)
            for b in range(m * p):
                j_b, k_b = divmod(b, p)
                for c in range(p * n):
                    # Determine if this is a valid equation
                    # Valid if j_a == j_b and c == k_b*n + i_a
                    if j_a == j_b:
                        valid_c = k_b * n + i_a
                        if c == valid_c:
                            # Valid equation: RHS = 1
                            eq = [0] * num_vars
                            for r_idx in range(r):
                                coeff = U[a][r_idx] * V[b][r_idx]
                                if coeff != 0:
                                    var_idx = c * r + r_idx
                                    eq[var_idx] = coeff
                            equations.append(eq)
                            rhs.append(1)
                        else:
                            # Invalid equation: RHS = 0
                            eq = [0] * num_vars
                            for r_idx in range(r):
                                coeff = U[a][r_idx] * V[b][r_idx]
                                if coeff != 0:
                                    var_idx = c * r + r_idx
                                    eq[var_idx] = coeff
                            equations.append(eq)
                            rhs.append(0)
                    else:
                        # j_a != j_b, always invalid: RHS = 0
                        eq = [0] * num_vars
                        for r_idx in range(r):
                            coeff = U[a][r_idx] * V[b][r_idx]
                            if coeff != 0:
                                var_idx = c * r + r_idx
                                eq[var_idx] = coeff
                        equations.append(eq)
                        rhs.append(0)
        
        # Remove duplicate/zero equations to reduce system size
        # Convert to numpy for efficient solving
        A = np.array(equations, dtype=float)
        b = np.array(rhs, dtype=float)
        
        # Check if system is consistent using least squares
        # We need an exact solution, so let's use rational arithmetic
        # For now, use numpy and verify
        
        try:
            # Use least squares to find a solution
            x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            
            # Check if the solution is exact
            A_x = A @ x
            if not np.allclose(A_x, b, atol=1e-10):
                return None
            
            # Check if the solution has rational entries (within tolerance)
            # For integer/half-integer entries, check if 2*x is close to integers
            x_2 = 2 * x
            x_rounded = np.round(x_2)
            if not np.allclose(x_2, x_rounded, atol=1e-10):
                # Try to snap to nearest half-integer
                # If it's close, use the rounded values
                if np.allclose(x_2, x_rounded, atol=1e-8):
                    x = x_rounded / 2.0
                else:
                    return None
            
            # Reconstruct W
            W = [[0] * r for _ in range(p * n)]
            for c in range(p * n):
                for r_idx in range(r):
                    W[c][r_idx] = x[c * r + r_idx]
            
            # Verify the solution exactly
            if validate(U, V, W, r):
                return W
            else:
                return None
                
        except np.linalg.LinAlgError:
            return None
    
    def try_find_decomposition(r, max_trials=100):
        """
        Try to find a rank-r decomposition using random search and exact linear algebra.
        """
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            
            # Generate random sparse U, V with entries in {-1, 0, 1}
            # Use a bias towards sparsity
            sparsity = 0.5  # 50% of entries are non-zero
            
            U = []
            for _ in range(n * m):
                row = [random.choice([-1, 0, 1]) for _ in range(r)]
                # Make it sparser
                for i in range(r):
                    if row[i] != 0 and random.random() > sparsity:
                        row[i] = 0
                U.append(row)
            
            V = []
            for _ in range(m * p):
                row = [random.choice([-1, 0, 1]) for _ in range(r)]
                for i in range(r):
                    if row[i] != 0 and random.random() > sparsity:
                        row[i] = 0
                V.append(row)
            
            # Solve for W
            W = solve_for_W(U, V, r)
            if W is not None:
                return U, V, W
        
        return None

    # Try to find a lower rank decomposition
    # Start from rank 8 and go up to rank_school - 1
    for r in range(8, rank_school):
        if time.time() > deadline:
            break
        
        # Try random trials
        result = try_find_decomposition(r, max_trials=50)
        if result is not None:
            U, V, W = result
            if validate(U, V, W, r):
                best_U = U
                best_V = V
                best_W = W
                best_rank = r
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
