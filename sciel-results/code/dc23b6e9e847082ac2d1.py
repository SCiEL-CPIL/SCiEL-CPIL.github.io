import numpy as np
import time
import itertools
from fractions import Fraction
from scipy.optimize import linprog

DIMENSIONS = [3, 3, 5]

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly represents matrix multiplication.
    U: (n*m) x R
    V: (m*p) x R
    W: (p*n) x R
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _check_half_integer(x):
    """Check if x is close to a half-integer (multiple of 0.5)."""
    return abs(2 * x - round(2 * x)) < 1e-6

def _find_rank_r(r, deadline, seed=0):
    """
    Attempt to find a rank-r decomposition.
    Strategy:
    1. Randomly initialize U and V with small integers/half-integers.
    2. Formulate the problem of finding W as a system of linear equations.
       For each (i, k), we have a system of m equations in r unknowns W[k*n+i, :].
       Actually, the equations are coupled by r.
       sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
       This is a system of n*m*p equations in p*n*r variables.
       
       Let's use the slice method more effectively.
       Fix U and V.
       The equations for W are linear.
       Let x be the vector of all W entries.
       A * x = b
       where b is a vector of ones (length n*m*p).
       
       We want a solution x such that all entries are half-integers.
       This is a linear feasibility problem with discrete constraints.
       
       Alternative: 
       Since we want half-integers, let W = 0.5 * Z where Z is integer.
       Then sum_r U_ij_r V_jk_r (0.5 Z_ki_r) = 1
       => sum_r U_ij_r V_jk_r Z_ki_r = 2
       
       So we are looking for an integer matrix Z (p*n x r) such that
       A * vec(Z) = 2 * 1
       where A is constructed from U and V.
       
       This is an integer linear feasibility problem.
       We can try to find a rational solution and check if it's integer,
       or use a heuristic.
       
       Given the small size (r <= 45, variables <= 15*r), we can try:
       1. Find a least squares solution for Z.
       2. Check if it's close to integer.
       3. If not, try to project to integer and verify.
       
       To increase success rate, we can restrict U and V to have entries in {0, 1, 2, 3}
       or similar small sets.
    """
    n, m, p = DIMENSIONS
    np.random.seed(seed)
    
    # Number of equations
    n_eq = n * m * p
    # Number of variables for Z (which is W * 2)
    n_var = p * n * r
    
    # Build A matrix: A[eq_idx, var_idx] = U[i*m+j, r] * V[j*p+k, r]
    # eq_idx = i*m*p + j*p + k
    # var_idx = (k*n + i) * r + r_idx
    
    # We will iterate over several random initializations of U and V.
    # To save time, we'll try a limited number of restarts.
    
    max_restarts = 100
    # Adjust restarts based on time remaining
    # But let's just run until deadline or max_restarts
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize U and V
        # Use small integers to keep numbers manageable
        # Entries in {-1, 0, 1, 2}
        U = np.random.randint(-1, 3, (n*m, r))
        V = np.random.randint(-1, 3, (m*p, r))
        
        # Construct A
        A = np.zeros((n_eq, n_var))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for r_idx in range(r):
                        var_idx = (k * n + i) * r + r_idx
                        A[eq_idx, var_idx] = U[i * m + j, r_idx] * V[j * p + k, r_idx]
        
        b = np.ones(n_eq) * 2  # Target is 2 because W = Z/2
        
        # Solve A * Z_vec = b
        # We want an integer solution.
        # First, check if a real solution exists.
        try:
            # Use least squares to get a real solution
            Z_vec, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            
            # Check if the real solution is close to integer
            Z_rounded = np.round(Z_vec)
            if np.allclose(A @ Z_rounded, b, atol=1e-6):
                # We found an integer solution for Z
                Z_mat = Z_rounded.reshape(p * n, r)
                W_mat = Z_mat / 2.0
                
                # Verify the solution
                if _verify(U.tolist(), V.tolist(), W_mat.tolist(), n, m, p):
                    return U, V, W_mat
            else:
                # The least squares solution is not close to integer.
                # This doesn't mean no integer solution exists, but it's a bad sign.
                # We could try to find a specific integer solution using other methods,
                # but for now, we'll just move to the next restart.
                pass
        except Exception:
            pass

    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    # Leave 10s margin for safety
    deadline = start_time + 140
    
    # Get schoolbook solution as fallback
    sb = _schoolbook()
    best_U = sb["U"]
    best_V = sb["V"]
    best_W = sb["W"]
    best_rank = n * m * p
    
    # Try to find lower ranks
    # Lower bound is max(n*p, m*p, n*m) = 15
    # Let's try ranks from 15 to 44
    # However, finding rank 15 is extremely hard.
    # Let's try a range of ranks.
    
    # Strategy: Try ranks in increasing order? Or decreasing?
    # Decreasing from 44 down to 15 allows us to stop early if we find a good one.
    # But lower ranks are harder.
    # Maybe try a few specific ranks first?
    # Let's just iterate downwards.
    
    # To optimize, we can skip some ranks if time is tight.
    # But let's try all.
    
    for r in range(best_rank - 1, 14, -1):
        if time.time() > deadline:
            break
            
        # Allocate time budget for this rank
        # Simple heuristic: more time for lower ranks? Or equal?
        # Let's just let the loop handle it with the global deadline.
        
        # Try to find a solution for this rank
        # Use multiple seeds to increase diversity
        found = False
        for seed in range(50): # 50 restarts per rank
            if time.time() > deadline:
                break
                
            res = _find_rank_r(r, deadline, seed=seed)
            if res is not None:
                U, V, W = res
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                best_rank = r
                found = True
                break
                
        if found:
            # If we found a solution, we might want to try lower ranks immediately
            # But the loop structure already does that (r decreases)
            pass

    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-START
# The code above is the implementation.
# EVOLVE-BLOCK-END
