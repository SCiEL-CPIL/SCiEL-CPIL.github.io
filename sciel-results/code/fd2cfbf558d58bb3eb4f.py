"""Rewritten program using randomized algebraic geometry approach for low-rank tensor decomposition."""
import numpy as np
from fractions import Fraction
import random
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor T has indices (i, j, k) -> T[i*m+j, j*p+k, k*n+i] = 1
    # We want to find rank r such that T = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c
    # where U is (n*m) x r, V is (m*p) x r, W is (p*n) x r
    
    # Schoolbook rank is n*m*p = 42
    # Lower bound is max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    
    def _schoolbook_decomposition(n, m, p):
        """Return the standard schoolbook decomposition of rank n*m*p."""
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
    
    def _verify_decomposition(U, V, W, n, m, p):
        """Verify that U ⊗ V ⊗ W equals the matrix multiplication tensor."""
        num_cols = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for c in range(num_cols):
                        total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                    if total != 1:
                        return False
        return True
    
    def _gauss_eliminate(aug):
        """Solve a system of linear equations using exact Gaussian elimination.
        aug is a list of lists representing the augmented matrix.
        Returns the solution vector or None if no unique solution."""
        num_rows = len(aug)
        if num_rows == 0:
            return []
        num_cols = len(aug[0]) - 1
        
        # Work with a copy
        A = [row[:] for row in aug]
        
        row = 0
        for col in range(num_cols):
            # Find pivot
            pivot_row = -1
            for i in range(row, num_rows):
                if A[i][col] != 0:
                    pivot_row = i
                    break
            
            if pivot_row == -1:
                continue
            
            # Swap rows
            A[row], A[pivot_row] = A[pivot_row], A[row]
            
            # Normalize pivot row
            pivot_val = A[row][col]
            if pivot_val != 1:
                A[row] = [x / pivot_val for x in A[row]]
            
            # Eliminate below and above
            for i in range(num_rows):
                if i == row:
                    continue
                factor = A[i][col]
                if factor != 0:
                    A[i] = [A[i][j] - factor * A[row][j] for j in range(num_cols + 1)]
            
            row += 1
            if row == num_rows:
                break
        
        # Check for inconsistency
        for i in range(row, num_rows):
            if any(A[i][j] != 0 for j in range(num_cols)):
                return None
        
        # Extract solution
        x = [Fraction(0)] * num_cols
        for i in range(row):
            # Find which variable is the leading one in row i
            for j in range(num_cols):
                if A[i][j] != 0:
                    x[j] = A[i][num_cols]
                    break
        
        return x
    
    def try_rank_reduction(target_rank, max_attempts=20):
        """
        Try to find a rank-target_rank decomposition using randomized approach.
        Fix U and V with random rational entries, then solve for W.
        """
        n, m, p = DIMENSIONS
        
        for attempt in range(max_attempts):
            # Initialize U and V with small random integer entries
            random.seed(attempt * 1000 + target_rank)
            
            U = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m * p)]
            
            # For each (i,j,k), we need:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            #
            # This is not linear in W because W appears multiplied by U*V.
            # Instead, let's think differently.
            #
            # Actually, for a fixed set of (U, V), the equation is:
            # sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1
            #
            # Let A_{(i,j,k), c} = U[i*m+j, c] * V[j*p+k, c]
            # Then we need: sum_c A_{(i,j,k), c} * W[k*n+i, c] = 1
            #
            # But W[k*n+i, c] depends on (k, i) and c. For a fixed (i,j,k),
            # the unknowns are W[k*n+i, c] for c=0..target_rank-1.
            #
            # So for each (i,j,k), we have one equation in target_rank unknowns.
            # But the unknowns W[k*n+i, c] are shared across different j values
            # (for the same i, k).
            #
            # Let's set up the full system. The unknowns are W[k*n+i, c] for 
            # all k, i, c. That's p*n*target_rank unknowns.
            # The equations are n*m*p = 42 equations.
            #
            # For target_rank = 41: 14*41 = 574 unknowns, 42 equations.
            # This is underdetermined, so there should be solutions.
            #
            # Let's build the system:
            # For each (i, j, k): sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            #
            # Unknown index: (k*n+i, c) -> k*n*target_rank + i*target_rank + c
            
            num_unknowns = p * n * target_rank
            num_eqs = n * m * p
            
            # Build augmented matrix
            aug = [[Fraction(0) for _ in range(num_unknowns + 1)] for _ in range(num_eqs)]
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(target_rank):
                            coeff = Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c])
                            if coeff != 0:
                                unknown_idx = k * n * target_rank + i * target_rank + c
                                aug[eq_idx][unknown_idx] += coeff
                        aug[eq_idx][num_unknowns] = 1
                        eq_idx += 1
            
            # Solve the system
            solution = _gauss_eliminate(aug)
            
            if solution is None:
                continue
            
            # Construct W from the solution
            W = [[0] * target_rank for _ in range(p * n)]
            for k in range(p):
                for i in range(n):
                    for c in range(target_rank):
                        unknown_idx = k * n * target_rank + i * target_rank + c
                        W[k*n+i][c] = solution[unknown_idx]
            
            # Verify
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try to find lower rank decompositions
    schoolbook_rank = n * m * p  # 42
    
    # Try ranks from schoolbook-1 down to a reasonable lower bound
    # Start with rank 41 and work down
    for target_rank in range(schoolbook_rank - 1, 20, -1):
        result = try_rank_reduction(target_rank, max_attempts=30)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)
# EVOLVE-BLOCK-END
