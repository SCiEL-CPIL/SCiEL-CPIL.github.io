"""Structural change: Use Exact Rational Arithmetic (fractions.Fraction) for Alternating Least Squares.
This avoids floating point errors and incorrect rounding by solving linear systems exactly.
If the solution is integer, it is a valid decomposition."""
import time
import random
from fractions import Fraction

try:
    import numpy as np
except Exception:
    np = None

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _verify_numpy(U, V, W, n, m, p):
    if np is None:
        return False
    R = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    for i in range(n):
        for k in range(p):
            M = np.zeros((m, m), dtype=np.float64)
            for c in range(R):
                wc = Wa[k * n + i, c]
                if wc == 0:
                    continue
                uc = Ua[i * m: i * m + m, c]
                vc = Va[k * p: k * p + m, c]
                M += np.outer(uc, vc) * wc
            target = np.eye(m, dtype=np.float64)
            if not np.allclose(M, target, atol=1e-9):
                return False
    return True

def _solve_linear_system_exact(A, b):
    """
    Solve A x = b where A is (m x R) and b is (m x 1) or (m x k).
    Returns x as list of Fractions or None if no unique solution/underdetermined.
    Actually, we want the minimum norm solution or any integer solution?
    We use Gaussian elimination to find a particular solution.
    """
    m = len(A)
    R = len(A[0])
    # Augment matrix
    # We will solve for x in R^R such that A x = b.
    # A is m x R. b is m x 1.
    # If m < R, the system is underdetermined. We can set free variables to 0.
    
    # Convert to Fractions
    A_frac = [[Fraction(x) for x in row] for row in A]
    b_frac = [Fraction(x) for x in b]
    
    # Gaussian Elimination
    # We have m equations, R variables.
    # We will perform elimination to find a solution.
    # To keep it simple, we assume we can solve for the first m variables in terms of the rest (set to 0).
    # This requires the leading m x m submatrix to be invertible.
    # We will try to pivot.
    
    # Copy for mutation
    mat = [row[:] for row in A_frac]
    vec = b_frac[:]
    
    # We want to reduce to row echelon form.
    # We will try to find a basis for the solution.
    # Let's just use a standard solver for square systems if m==R.
    # If m < R, we can set the last R-m variables to 0 and solve the first m.
    # This is a heuristic.
    
    # For robustness, let's just solve the square system formed by the first m columns if m <= R.
    # If m > R, it's overdetermined, we need least squares (hard with exact).
    # In our case, for U: m=5, R is the rank. If R >= 5, we can pick 5 columns.
    
    if m > R:
        return None # Overdetermined, hard to solve exactly without LS
        
    # We will solve for the first m variables, setting the rest to 0.
    # This requires the m x m submatrix of A (first m cols) to be invertible.
    # If not, we might need to swap columns.
    
    # Let's try to find any m linearly independent columns.
    # For simplicity, let's just try the first m.
    sub_A = [[mat[i][j] for j in range(m)] for i in range(m)]
    
    # Check determinant
    # Calculate determinant of sub_A
    det = Fraction(1)
    for i in range(m):
        det *= sub_A[i][i]
        # This is wrong, need proper determinant
        pass
    
    # Use Cramer's rule or Gaussian Elimination on the square system.
    # Let's implement a simple Gaussian Elimination for the square system.
    
    # Augment sub_A with vec
    aug = [[sub_A[i][j] for j in range(m)] + [vec[i]] for i in range(m)]
    
    # Elimination
    for col in range(m):
        # Find pivot
        pivot_row = -1
        for row in range(col, m):
            if aug[row][col] != 0:
                pivot_row = row
                break
        if pivot_row == -1:
            return None # Singular
            
        # Swap
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        
        # Normalize
        pivot_val = aug[col][col]
        for j in range(col, m + 1):
            aug[col][j] /= pivot_val
            
        # Eliminate
        for row in range(m):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0:
                continue
            for j in range(col, m + 1):
                aug[row][j] -= factor * aug[col][j]
                
    # Solution is in the last column
    x = [aug[i][m] for i in range(m)]
    
    # Construct full solution vector (R elements)
    full_x = [Fraction(0) for _ in range(R)]
    for i in range(m):
        full_x[i] = x[i]
        
    return full_x

def _solve_for_U_exact(V, W, n, m, p, R):
    U = [[0] * R for _ in range(n * m)]
    for i in range(n):
        for k in range(p):
            # System: sum_c U[i,j,c] * (W[k,i,c] * V[j,k,c]) = delta_{j,j'}
            # For each j, j', we have an equation.
            # Wait, the equation is sum_c U[i,j,c] A[j,c] = delta_{j,j'}?
            # No, U is m x R. We need to solve for U[i, :, :] (m vectors of length R).
            # The equations couple all j's.
            # M_{j, j'} = sum_c W[k,i,c] U[i,j,c] V[j',k,c] = delta_{j,j'}.
            # This is a linear system for U[i, :, :].
            # Variables: U[i, 0, 0] ... U[i, m-1, R-1]. Total m*R variables.
            # Equations: m*m equations.
            # If m*R >= m*m (i.e., R >= m), we can solve.
            
            A = []
            b = []
            for j in range(m):
                for jp in range(m):
                    # Equation for M[j, jp] = delta_{j,jp}
                    # sum_c W[k,i,c] U[i,j,c] V[jp,k,c] = 1 if j==jp else 0
                    row = []
                    for c in range(R):
                        coeff = W[k * n + i][c] * V[jp * p + k][c]
                        # The variable is U[i,j,c]
                        # But U[i,j,c] only appears in equations where the first index is j.
                        # So for fixed j, the variables are U[i,j,c] for c=0..R-1.
                        # But the equation sums over c.
                        # Wait, the equation is for a specific j and jp.
                        # It involves U[i,j,c] for all c.
                        # So for a fixed j, we have m equations (one for each jp) and R variables (U[i,j,c]).
                        # This decouples!
                        # For each j, we solve a system of m equations for R unknowns.
                        pass
                    break
                break
            
            # Let's re-structure.
            # For each j in 0..m-1:
            #   Variables: x_c = U[i,j,c] for c=0..R-1.
            #   Equations: for each jp in 0..m-1:
            #     sum_c x_c * (W[k,i,c] * V[jp,k,c]) = delta_{j,jp}
            
            for j in range(m):
                A_j = []
                b_j = []
                for jp in range(m):
                    row = []
                    for c in range(R):
                        row.append(W[k * n + i][c] * V[jp * p + k][c])
                    A_j.append(row)
                    b_j.append(1 if j == jp else 0)
                
                # Solve A_j x = b_j
                # A_j is m x R.
                x = _solve_linear_system_exact(A_j, b_j)
                if x is None:
                    return None # Failed to solve
                for c in range(R):
                    U[i * m + j][c] = x[c]
    
    # Check if all are integers
    for i in range(n * m):
        for c in range(R):
            if U[i][c].denominator != 1:
                return None
            U[i][c] = int(U[i][c])
            
    return U

def _solve_for_V_exact(U, W, n, m, p, R):
    V = [[0] * R for _ in range(m * p)]
    for k in range(p):
        for i in range(n):
            # Equations: sum_c W[k,i,c] U[i,j,c] V[jp,k,c] = delta_{j,jp}
            # For each jp, variables are V[jp,k,c] for c=0..R-1.
            # Equations for j=0..m-1.
            for jp in range(m):
                A_jp = []
                b_jp = []
                for j in range(m):
                    row = []
                    for c in range(R):
                        row.append(W[k * n + i][c] * U[i * m + j][c])
                    A_jp.append(row)
                    b_jp.append(1 if j == jp else 0)
                    
                x = _solve_linear_system_exact(A_jp, b_jp)
                if x is None:
                    return None
                for c in range(R):
                    V[jp * p + k][c] = x[c]
    
    for i in range(m * p):
        for c in range(R):
            if V[i][c].denominator != 1:
                return None
            V[i][c] = int(V[i][c])
            
    return V

def _solve_for_W_exact(U, V, n, m, p, R):
    W = [[0] * R for _ in range(p * n)]
    for i in range(n):
        for k in range(p):
            # Equations: sum_c W[k,i,c] U[i,j,c] V[jp,k,c] = delta_{j,jp}
            # Variables: W[k,i,c] for c=0..R-1.
            # Equations: m*m equations.
            A = []
            b = []
            for j in range(m):
                for jp in range(m):
                    row = []
                    for c in range(R):
                        row.append(U[i * m + j][c] * V[jp * p + k][c])
                    A.append(row)
                    b.append(1 if j == jp else 0)
                    
            # A is m*m x R.
            # We need to solve A x = b.
            # If m*m > R, it's overdetermined.
            # If m*m <= R, we can solve.
            # Note: m=5, so m*m=25.
            # If R >= 25, we can solve.
            
            x = _solve_linear_system_exact(A, b)
            if x is None:
                return None
            for c in range(R):
                W[k * n + i][c] = x[c]
                
    for i in range(p * n):
        for c in range(R):
            if W[i][c].denominator != 1:
                return None
            W[i][c] = int(W[i][c])
            
    return W

def _try_rank_exact(R, n, m, p, deadline, rng):
    # Initialize with random small integers
    U = [[rng.choice([-1, 0, 1]) for _ in range(R)] for _ in range(n * m)]
    V = [[rng.choice([-1, 0, 1]) for _ in range(R)] for _ in range(m * p)]
    W = [[rng.choice([-1, 0, 1]) for _ in range(R)] for _ in range(p * n)]
    
    # Convert to Fractions for solving
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    for it in range(10):
        if time.time() > deadline - 10.0:
            return None
            
        U_new = _solve_for_U_exact(V_frac, W_frac, n, m, p, R)
        if U_new is None: continue
        U_frac = U_new
        
        V_new = _solve_for_V_exact(U_frac, W_frac, n, m, p, R)
        if V_new is None: continue
        V_frac = V_new
        
        W_new = _solve_for_W_exact(U_frac, V_frac, n, m, p, R)
        if W_new is None: continue
        W_frac = W_new
        
        # Verify
        if _verify_numpy(U_frac, V_frac, W_frac, n, m, p):
            return U_frac, V_frac, W_frac
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS
    rng = random.Random(42)
    
    # Try ranks from 120 down to 60
    for R in range(120, 59, -1):
        if time.time() > deadline - 20.0:
            break
        result = _try_rank_exact(R, n, m, p, deadline, rng)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fall back to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
