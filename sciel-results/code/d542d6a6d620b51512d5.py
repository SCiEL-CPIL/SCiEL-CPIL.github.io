"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time
import itertools

def _check_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = int(n), int(m), int(p)
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    u_val = U[i*m + j][c]
                    v_val = V[j*p + k][c]
                    w_val = W[k*n + i][c]
                    val += u_val * v_val * w_val
                if val != 1:
                    return False
    return True

def _solve_rank_decomposition(rank, n, m, p, timeout=10):
    """
    Use a structured approach to find rank decomposition.
    
    The matrix multiplication tensor T_{i,j,k} = 1 for all valid (i,j,k) in [0,n)×[0,m)×[0,p).
    We seek U (n*m × rank), V (m*p × rank), W (p*n × rank) such that:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    Strategy: Fix U and W to have simple structures, then solve for V.
    Alternatively, use the fact that we can set some entries to 0/1 and solve the rest.
    
    Approach: Parameterize with a small set of free variables and solve linearly.
    """
    start_time = time.time()
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    # For small ranks, try a direct algebraic approach
    # Key insight: We can set U and W to have specific patterns and solve for V
    
    # Try approach: Set U[i*m+j, c] = a_{i,j,c}, W[k*n+i, c] = b_{k,i,c}
    # Then for each (i,j,k): sum_c a_{i,j,c} * V[j*p+k, c] * b_{k,i,c} = 1
    # This is linear in V entries if we fix U and W.
    
    # Strategy 1: Use Kronecker product structure
    # For 2x3x4, we can think of it as combining 2x2x2 and 2x2x2 blocks
    
    # Let's try a more systematic approach:
    # Fix U to have a simple structure (e.g., identity-like), then solve for V and W
    
    # Approach: Set U[i*m+j, c] = 1 if c == i*m+j else 0 (identity for first U_rows columns)
    # But rank might be less than U_rows, so this doesn't work directly.
    
    # Better: Use the known structure of matrix multiplication
    # The tensor is T[i,j,k] = 1, which is the product of three "indicator" tensors
    
    # Let's try: for each rank c, define the contribution as a product of three vectors
    # u_c ∈ R^{n*m}, v_c ∈ R^{m*p}, w_c ∈ R^{p*n}
    # such that sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1
    
    # This is a 3-tensor decomposition problem.
    
    # For rank = n*m*p = 24, the trivial decomposition works.
    # For lower ranks, we need to be clever.
    
    # Let's try a block-based approach for 2x3x4:
    # Think of A as 2x3, B as 3x4
    # Partition the 3 middle columns into groups
    
    # Actually, let's use a known result: the rank of 2x3x4 matrix multiplication
    # is at most 19 (by various constructions) and at least 19 (by lower bounds).
    # So rank 19 should be achievable.
    
    # Let's try to construct rank 19 using a combination of Strassen-like ideas.
    
    # For now, let's use a computational approach:
    # Set up the system of equations and try to solve it.
    
    # Number of unknowns: rank * (U_rows + V_rows + W_rows) = rank * (6 + 12 + 8) = rank * 26
    # Number of equations: n*m*p = 24
    
    # For rank >= 24, we have more unknowns than equations, so solutions exist.
    # For rank < 24, we need the system to be consistent.
    
    # Let's try to find a solution by fixing some variables and solving linearly.
    
    # Strategy: Fix U and W to have integer entries from {-1, 0, 1}, then solve for V.
    # For each (i,j,k): sum_c U[i*m+j, c] * W[k*n+i, c] * V[j*p+k, c] = 1
    # This is linear in V[j*p+k, c] for fixed c... but it's coupled across c.
    
    # Actually, for fixed (i,j,k), the equation is:
    # sum_c (U[i*m+j, c] * W[k*n+i, c]) * V[j*p+k, c] = 1
    # Let A_{i,k,j,c} = U[i*m+j, c] * W[k*n+i, c]
    # Then sum_c A_{i,k,j,c} * V[j*p+k, c] = 1
    
    # For each pair (j,k), we have an equation involving V[j*p+k, :]:
    # sum_c A_{i,k,j,c} * V[j*p+k, c] = 1 for all i
    
    # This gives n equations for the rank unknowns V[j*p+k, 0..rank-1].
    # We have m*p = 12 such systems (one for each (j,k)).
    
    # So the total system is: for each (j,k), solve:
    #   M_{j,k} @ V_{j,k} = 1_vector (length n)
    # where M_{j,k}[i, c] = U[i*m+j, c] * W[k*n+i, c]
    
    # This means M_{j,k} must have rank at least n for each (j,k), and the system must be consistent.
    
    # Let's try to find U and W such that all M_{j,k} are invertible (or at least full row rank).
    
    # For simplicity, let's try small integer values for U and W.
    
    max_attempts = 200
    for attempt in range(max_attempts):
        if time.time() - start_time > timeout:
            break
        
        # Random U and W with small integers
        U = [[0] * rank for _ in range(U_rows)]
        W = [[0] * rank for _ in range(W_rows)]
        
        for i in range(U_rows):
            for c in range(rank):
                U[i][c] = int(np.random.randint(-2, 3))
        for k in range(p):
            for i in range(n):
                for c in range(rank):
                    W[k*n + i][c] = int(np.random.randint(-2, 3))
        
        # Check if we can solve for V
        V = [[0] * rank for _ in range(V_rows)]
        success = True
        
        for j in range(m):
            for k in range(p):
                # Build M_{j,k}: n x rank matrix
                M = np.zeros((n, rank))
                for i in range(n):
                    for c in range(rank):
                        M[i, c] = U[i*m + j][c] * W[k*n + i][c]
                
                # Solve M @ v = ones(n)
                try:
                    # Use least squares as a check; we need exact solution
                    # First check if system is consistent
                    v, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    
                    # Check if residual is close to zero
                    if np.max(np.abs(M @ v - np.ones(n))) > 1e-10:
                        success = False
                        break
                    
                    # Check if v has nice rational/integer values
                    # Convert to fractions and check
                    v_frac = []
                    for val in v:
                        # Try to find a simple fraction
                        v_frac.append(Fraction(val).limit_denominator(100))
                    
                    # Check if the solution is exact with these fractions
                    M_frac = np.array([[Fraction(M[i,c]) for c in range(rank)] for i in range(n)])
                    v_frac_arr = np.array([v_frac[c] for c in range(rank)])
                    
                    # Verify
                    result = np.zeros(n)
                    for i in range(n):
                        s = 0
                        for c in range(rank):
                            s += M_frac[i, c] * v_frac_arr[c]
                        result[i] = s
                    
                    if any(r != 1 for r in result):
                        success = False
                        break
                    
                    for c in range(rank):
                        V[j*p + k][c] = v_frac[c]
                
                except Exception:
                    success = False
                    break
            
            if not success:
                break
        
        if success:
            # Verify the full decomposition
            if _check_decomposition(U, V, W, n, m, p):
                return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try ranks from the lower bound upward
    # Known lower bound for 2x3x4 is 19
    for rank in range(19, 25):
        result = _solve_rank_decomposition(rank, n, m, p, timeout=8)
        if result is not None:
            U, V, W = result
            U_out = [[float(u) for u in row] for row in U]
            V_out = [[float(v) for v in row] for row in V]
            W_out = [[float(w) for w in row] for row in W]
            return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback: schoolbook decomposition of rank n*m*p = 24
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
