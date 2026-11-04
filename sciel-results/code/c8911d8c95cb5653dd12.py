"""Matrix multiplication 3,3,4 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36

    # Build schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank), dtype=object)
    V_school = np.zeros((m * p, schoolbook_rank), dtype=object)
    W_school = np.zeros((p * n, schoolbook_rank), dtype=object)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = Fraction(1)
                V_school[j * p + k, col] = Fraction(1)
                W_school[k * n + i, col] = Fraction(1)
                col += 1

    def verify_exact(U_mat, V_mat, W_mat, rank):
        """Verify exact decomposition using Fractions"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if s != Fraction(1):
                        return False
        return True

    def solve_linear_system_exact(A, b):
        """Solve A*x = b over rationals using Gaussian elimination."""
        n_rows = len(A)
        if n_rows == 0:
            return [Fraction(0)] * len(b)
        n_unknowns = len(b)
        
        aug = [[A[i][j] for j in range(n_unknowns)] + [b[i]] for i in range(n_rows)]
        
        rank = 0
        pivot_cols = []
        for col in range(n_unknowns):
            if rank >= n_rows:
                break
            pivot_row = None
            for row in range(rank, n_rows):
                if aug[row][col] != 0:
                    pivot_row = row
                    break
            if pivot_row is None:
                continue
            
            aug[rank], aug[pivot_row] = aug[pivot_row], aug[rank]
            pivot_cols.append(col)
            
            pivot = aug[rank][col]
            for j in range(col, n_unknowns + 1):
                aug[rank][j] = aug[rank][j] / pivot
            
            for row in range(n_rows):
                if row != rank and aug[row][col] != 0:
                    factor = aug[row][col]
                    for j in range(col, n_unknowns + 1):
                        aug[row][j] = aug[row][j] - factor * aug[rank][j]
            
            rank += 1
        
        for row in range(rank, n_rows):
            if any(aug[row][j] != 0 for j in range(n_unknowns)):
                return None
        
        x = [Fraction(0)] * n_unknowns
        for row in range(rank):
            col = pivot_cols[row]
            x[col] = aug[row][n_unknowns]
        
        return x

    def try_rank_linear_solve(R, seed, max_attempts=30):
        """Try to find rank R by fixing V,W and solving for U"""
        rng = random.Random(seed)
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return None
            
            V = np.zeros((m*p, R), dtype=object)
            W = np.zeros((p*n, R), dtype=object)
            
            # Use a more structured initialization
            for r in range(R):
                # Ensure each rank term has some non-zero entries
                v_nonzero = rng.sample(range(m*p), min(3, m*p))
                w_nonzero = rng.sample(range(p*n), min(3, p*n))
                
                for idx in v_nonzero:
                    V[idx, r] = Fraction(rng.choice([-1, 1]))
                for idx in w_nonzero:
                    W[idx, r] = Fraction(rng.choice([-1, 1]))
            
            U = np.zeros((n*m, R), dtype=object)
            valid = True
            
            for j in range(m):
                n_eqs = n * p
                n_unknowns = n * R
                
                A = [[Fraction(0)] * n_unknowns for _ in range(n_eqs)]
                b = [Fraction(1)] * n_eqs
                
                eq_idx = 0
                for i in range(n):
                    for k in range(p):
                        for r in range(R):
                            u_idx = i * R + r
                            A[eq_idx][u_idx] = V[j*p+k, r] * W[k*n+i, r]
                        eq_idx += 1
                
                sol = solve_linear_system_exact(A, b)
                if sol is None:
                    valid = False
                    break
                
                for i in range(n):
                    for r in range(R):
                        U[i*m+j, r] = sol[i*R + r]
            
            if not valid:
                continue
            
            if verify_exact(U, V, W, R):
                return U, V, W, R
        
        return None

    best_result = None
    
    for R in range(25, 36):
        if time.time() > deadline - 20:
            break
            
        found = False
        for seed in range(200):
            if time.time() > deadline - 10:
                break
                
            result = try_rank_linear_solve(R, seed, max_attempts=5)
            if result:
                U_rat, V_rat, W_rat, rank = result
                best_result = (U_rat, V_rat, W_rat, rank)
                found = True
                break
        
        if found:
            break

    if best_result:
        U_rat, V_rat, W_rat, rank = best_result
        return {"U": [[float(x) for x in row] for row in U_rat], 
                "V": [[float(x) for x in row] for row in V_rat], 
                "W": [[float(x) for x in row] for row in W_rat]}
    else:
        return {"U": [[float(x) for x in row] for row in U_school], 
                "V": [[float(x) for x in row] for row in V_school], 
                "W": [[float(x) for x in row] for row in W_school]}
# EVOLVE-BLOCK-END
