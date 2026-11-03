"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
from fractions import Fraction
import random
import itertools

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x10 matrix.
    Uses alternating minimization with exact rational arithmetic to search for low-rank solutions.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Precompute the index mappings
    # For each (i, j, k), we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    def check_solution(U, V, W):
        """Check if the solution satisfies all equations exactly."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    for c in range(len(U[0])):
                        total += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                    if total != 1:
                        return False
        return True
    
    def solve_linear_fixed(fixed_matrix, fixed_rows, target_rows, other1, other2, indices):
        """
        Solve linear system: for each row r in target_rows,
        sum_c X[r, c] * A[r, c] = b[r]
        where A and b are computed from the fixed matrices.
        """
        r = len(fixed_matrix[0])
        # Build the linear system: for each target row, we have r unknowns
        # But we need to handle this carefully
        
        # Actually, let's use a simpler approach: for each column c, the constraint is
        # sum_c X[r,c] * (V[v_idx,c] * W[w_idx,c]) = 1
        # This is a linear system in X[r,c] for each r.
        
        # For a fixed row r of X, we have:
        # sum_c X[r,c] * coeff[r,c] = 1
        # where coeff depends on which (i,j,k) correspond to row r.
        
        # But multiple (i,j,k) can map to the same row r of X (when i*m+j is the same).
        # So we need to satisfy multiple equations for the same row of X.
        
        # Let me restructure: group by rows of X.
        pass
    
    # Simpler approach: use alternating minimization with direct linear solves
    
    def alt_minimize(U, V, W, num_iters=50):
        """Alternating minimization with exact rational arithmetic."""
        r = len(U[0])
        
        for it in range(num_iters):
            # Fix V and W, solve for U
            # For each row u_idx of U, we have equations from all (i,j,k) where i*m+j == u_idx
            # The equation is: sum_c U[u_idx,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # But for a fixed u_idx = i*m+j, the equation involves k varying.
            # So for each (i,j), we have p equations (one for each k):
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for k=0,...,p-1
            
            new_U = [[Fraction(0)] * r for _ in range(nm)]
            for i in range(n):
                for j in range(m):
                    u_idx = i * m + j
                    # We have p equations in r unknowns U[u_idx, c]
                    # Build the system
                    # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    # b[k] = 1
                    # Solve A * x = b where x = U[u_idx, :]
                    
                    A = [[Fraction(0)] * r for _ in range(p)]
                    b = [Fraction(1)] * p
                    for k in range(p):
                        v_idx = j * p + k
                        w_idx = k * n + i
                        for c in range(r):
                            A[k][c] = V[v_idx][c] * W[w_idx][c]
                    
                    # Solve the linear system using Gaussian elimination with rationals
                    # A is p x r, we want to find x such that A*x = b
                    # If p <= r, we can pick p independent rows
                    # If p > r, we need least squares, but with exact arithmetic this is tricky
                    
                    if p <= r:
                        # Pick first p rows (or find independent rows)
                        # Simple approach: use first min(p,r) rows
                        num_eq = min(p, r)
                        A_sub = A[:num_eq]
                        b_sub = b[:num_eq]
                        
                        # Gaussian elimination
                        x = [Fraction(0)] * r
                        # Build augmented matrix
                        aug = [row[:] + [b_sub[i]] for i, row in enumerate(A_sub)]
                        
                        # Forward elimination
                        for col in range(num_eq):
                            # Find pivot
                            pivot = -1
                            for row in range(col, num_eq):
                                if aug[row][col] != 0:
                                    pivot = row
                                    break
                            if pivot == -1:
                                continue
                            if pivot != col:
                                aug[col], aug[pivot] = aug[pivot], aug[col]
                            # Eliminate
                            pivot_val = aug[col][col]
                            for row in range(col + 1, num_eq):
                                factor = aug[row][col] / pivot_val
                                for c2 in range(col, num_eq + 1):
                                    aug[row][c2] -= factor * aug[col][c2]
                        
                        # Back substitution
                        for row in range(num_eq - 1, -1, -1):
                            if aug[row][row] == 0:
                                continue
                            val = aug[row][num_eq]
                            for c2 in range(row + 1, num_eq):
                                val -= aug[row][c2] * x[c2]
                            x[row] = val / aug[row][row]
                        
                        new_U[u_idx] = x
                    else:
                        # p > r, use least squares with normal equations
                        # A^T A x = A^T b
                        A_TA = [[Fraction(0)] * r for _ in range(r)]
                        A_tb = [Fraction(0)] * r
                        for c1 in range(r):
                            for c2 in range(r):
                                for k in range(p):
                                    A_TA[c1][c2] += A[k][c1] * A[k][c2]
                            for k in range(p):
                                A_tb[c1] += A[k][c1] * b[k]
                        
                        # Solve A_TA * x = A_tb
                        aug = [A_TA[i][:] + [A_tb[i]] for i in range(r)]
                        for col in range(r):
                            pivot = -1
                            for row in range(col, r):
                                if aug[row][col] != 0:
                                    pivot = row
                                    break
                            if pivot == -1:
                                continue
                            if pivot != col:
                                aug[col], aug[pivot] = aug[pivot], aug[col]
                            pivot_val = aug[col][col]
                            for row in range(col + 1, r):
                                factor = aug[row][col] / pivot_val
                                for c2 in range(col, r + 1):
                                    aug[row][c2] -= factor * aug[col][c2]
                        
                        x = [Fraction(0)] * r
                        for row in range(r - 1, -1, -1):
                            if aug[row][row] == 0:
                                continue
                            val = aug[row][r]
                            for c2 in range(row + 1, r):
                                val -= aug[row][c2] * x[c2]
                            x[row] = val / aug[row][row]
                        
                        new_U[u_idx] = x
            
            U = new_U
            
            # Fix U and W, solve for V
            new_V = [[Fraction(0)] * r for _ in range(mp)]
            for j in range(m):
                for k in range(p):
                    v_idx = j * p + k
                    # Equations from all (i,j,k) where j*p+k == v_idx
                    # For fixed j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for i=0,...,n-1
                    # So n equations in r unknowns
                    A = [[Fraction(0)] * r for _ in range(n)]
                    b = [Fraction(1)] * n
                    for i in range(n):
                        u_idx = i * m + j
                        w_idx = k * n + i
                        for c in range(r):
                            A[i][c] = U[u_idx][c] * W[w_idx][c]
                    
                    if n <= r:
                        num_eq = min(n, r)
                        A_sub = A[:num_eq]
                        b_sub = b[:num_eq]
                        aug = [row[:] + [b_sub[i]] for i, row in enumerate(A_sub)]
                        for col in range(num_eq):
                            pivot = -1
                            for row in range(col, num_eq):
                                if aug[row][col] != 0:
                                    pivot = row
                                    break
                            if pivot == -1:
                                continue
                            if pivot != col:
                                aug[col], aug[pivot] = aug[pivot], aug[col]
                            pivot_val = aug[col][col]
                            for row in range(col + 1, num_eq):
                                factor = aug[row][col] / pivot_val
                                for c2 in range(col, num_eq + 1):
                                    aug[row][c2] -= factor * aug[col][c2]
                        x = [Fraction(0)] * r
                        for row in range(num_eq - 1, -1, -1):
                            if aug[row][row] == 0:
                                continue
                            val = aug[row][num_eq]
                            for c2 in range(row + 1, num_eq):
                                val -= aug[row][c2] * x[c2]
                            x[row] = val / aug[row][row]
                        new_V[v_idx] = x
                    else:
                        A_TA = [[Fraction(0)] * r for _ in range(r)]
                        A_tb = [Fraction(0)] * r
                        for c1 in range(r):
                            for c2 in range(r):
                                for i in range(n):
                                    A_TA[c1][c2] += A[i][c1] * A[i][c2]
                            for i in range(n):
                                A_tb[c1] += A[i][c1] * b[i]
                        aug = [A_TA[i][:] + [A_tb[i]] for i in range(r)]
                        for col in range(r):
                            pivot = -1
                            for row in range(col, r):
                                if aug[row][col] != 0:
                                    pivot = row
                                    break
                            if pivot == -1:
                                continue
                            if pivot != col:
                                aug[col], aug[pivot] = aug[pivot], aug[col]
                            pivot_val = aug[col][col]
                            for row in range(col + 1, r):
                                factor = aug[row][col] / pivot_val
                                for c2 in range(col, r + 1):
                                    aug[row][c2] -= factor * aug[col][c2]
                        x = [Fraction(0)] * r
                        for row in range(r - 1, -1, -1):
                            if aug[row][row] == 0:
                                continue
                            val = aug[row][r]
                            for c2 in range(row + 1, r):
                                val -= aug[row][c2] * x[c2]
                            x[row] = val / aug[row][row]
                        new_V[v_idx] = x
            
            V = new_V
            
            # Fix U and V, solve for W
            new_W = [[Fraction(0)] * r for _ in range(pn)]
            for k in range(p):
                for i in range(n):
                    w_idx = k * n + i
                    # Equations from all (i,j,k) where k*n+i == w_idx
                    # For fixed k,i: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for j=0,...,m-1
                    # So m equations in r unknowns
                    A = [[Fraction(0)] * r for _ in range(m)]
                    b = [Fraction(1)] * m
                    for j in range(m):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        for c in range(r):
                            A[j][c] = U[u_idx][c] * V[v_idx][c]
                    
                    if m <= r:
                        num_eq = min(m, r)
                        A_sub = A[:num_eq]
                        b_sub = b[:num_eq]
                        aug = [row[:] + [b_sub[i]] for i, row in enumerate(A_sub)]
                        for col in range(num_eq):
                            pivot = -1
                            for row in range(col, num_eq):
                                if aug[row][col] != 0:
                                    pivot = row
                                    break
                            if pivot == -1:
                                continue
                            if pivot != col:
                                aug[col], aug[pivot] = aug[pivot], aug[col]
                            pivot_val = aug[col][col]
                            for row in range(col + 1, num_eq):
                                factor = aug[row][col] / pivot_val
                                for c2 in range(col, num_eq + 1):
                                    aug[row][c2] -= factor * aug[col][c2]
                        x = [Fraction(0)] * r
                        for row in range(num_eq - 1, -1, -1):
                            if aug[row][row] == 0:
                                continue
                            val = aug[row][num_eq]
                            for c2 in range(row + 1, num_eq):
                                val -= aug[row][c2] * x[c2]
                            x[row] = val / aug[row][row]
                        new_W[w_idx] = x
                    else:
                        A_TA = [[Fraction(0)] * r for _ in range(r)]
                        A_tb = [Fraction(0)] * r
                        for c1 in range(r):
                            for c2 in range(r):
                                for j in range(m):
                                    A_TA[c1][c2] += A[j][c1] * A[j][c2]
                            for j in range(m):
                                A_tb[c1] += A[j][c1] * b[j]
                        aug = [A_TA[i][:] + [A_tb[i]] for i in range(r)]
                        for col in range(r):
                            pivot = -1
                            for row in range(col, r):
                                if aug[row][col] != 0:
                                    pivot = row
                                    break
                            if pivot == -1:
                                continue
                            if pivot != col:
                                aug[col], aug[pivot] = aug[pivot], aug[col]
                            pivot_val = aug[col][col]
                            for row in range(col + 1, r):
                                factor = aug[row][col] / pivot_val
                                for c2 in range(col, r + 1):
                                    aug[row][c2] -= factor * aug[col][c2]
                        x = [Fraction(0)] * r
                        for row in range(r - 1, -1, -1):
                            if aug[row][row] == 0:
                                continue
                            val = aug[row][r]
                            for c2 in range(row + 1, r):
                                val -= aug[row][c2] * x[c2]
                            x[row] = val / aug[row][row]
                        new_W[w_idx] = x
            
            W = new_W
        
        return U, V, W
    
    # Try ranks from 54 down to 1
    for rank in range(54, 0, -1):
        # Try multiple random initializations
        for trial in range(5):
            random.seed(trial * 1000 + rank)
            
            # Initialize with small random rationals
            U = [[random.randint(-3, 3) for _ in range(rank)] for _ in range(nm)]
            V = [[random.randint(-3, 3) for _ in range(rank)] for _ in range(mp)]
            W = [[random.randint(-3, 3) for _ in range(rank)] for _ in range(pn)]
            
            # Convert to Fractions
            U = [[Fraction(x) for x in row] for row in U]
            V = [[Fraction(x) for x in row] for row in V]
            W = [[Fraction(x) for x in row] for row in W]
            
            # Run alternating minimization
            U, V, W = alt_minimize(U, V, W, num_iters=20)
            
            # Check if solution is valid
            if check_solution(U, V, W):
                # Convert to list of lists of int/float
                U_out = [[float(x) for x in row] for row in U]
                V_out = [[float(x) for x in row] for row in V]
                W_out = [[float(x) for x in row] for row in W]
                return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback: schoolbook
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
# EVOLVE-BLOCK-END
