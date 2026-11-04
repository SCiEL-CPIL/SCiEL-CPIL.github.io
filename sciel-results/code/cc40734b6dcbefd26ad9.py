"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import random
import time
from fractions import Fraction

def _schoolbook_solution():
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

def _verify_decomposition(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    # Check shapes
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
    # Verify tensor identity: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1) > 1e-9:
                    return False
    return True

def _check_half_integer_solution(U, V, W):
    """Check if all entries are integers or half-integers."""
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                # Check if val is integer or half-integer
                # Multiply by 2 and check if it's an integer
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True

def _solve_sparse_system(r, n, m, p, seed=None):
    """
    Try to find a rank-r decomposition with sparse support.
    Returns (U, V, W) if found, else None.
    """
    if seed is not None:
        random.seed(seed)
    
    nm = n * m
    mp = m * p
    pn = p * n
    num_equations = n * m * p
    
    # Number of non-zero entries to try per matrix
    # We need enough unknowns to solve the system
    # Each equation involves 3 variables (one from each matrix)
    # Let's try to have about 2*num_equations unknowns total
    target_unknowns = max(num_equations * 2, r * 3)
    
    # Distribute unknowns across U, V, W
    # Each rank column c has entries in U, V, W
    # Let's say each column has u_c, v_c, w_c non-zero entries
    # Total unknowns = sum_c (u_c + v_c + w_c)
    
    # Strategy: for each column, randomly choose how many non-zeros
    # in each matrix, ensuring total >= num_equations
    
    attempts = 50
    for attempt in range(attempts):
        # Choose sparsity pattern
        u_sparsity = []
        v_sparsity = []
        w_sparsity = []
        
        for c in range(r):
            # Randomly choose number of non-zeros per matrix per column
            u_nz = random.randint(1, 4)
            v_nz = random.randint(1, 4)
            w_nz = random.randint(1, 4)
            u_sparsity.append(u_nz)
            v_sparsity.append(v_nz)
            w_sparsity.append(w_nz)
        
        total_unknowns = sum(u_sparsity) + sum(v_sparsity) + sum(w_sparsity)
        if total_unknowns < num_equations:
            continue
        
        # Build the linear system
        # Variables: U[i*m+j, c] for each (i,j,c) where non-zero
        #            V[j*p+k, c] for each (j,k,c) where non-zero
        #            W[k*n+i, c] for each (k,i,c) where non-zero
        
        # This is a 3-linear system, not linear. We need to handle it differently.
        # Actually, the tensor identity is:
        # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # This is a system of polynomial equations (degree 3 in the variables).
        # Solving this directly is hard.
        
        # Alternative: fix two of the matrices and solve for the third.
        # If we fix U and V, then for each (i,j,k):
        # sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c]) = 1
        # This is linear in W!
        # Similarly, we can fix any two and solve for the third.
        
        # Let's try: fix U and V randomly, then solve for W.
        # But W has pn * r = 9*4*r = 36r variables, and we have n*m*p = 144 equations.
        # For each k, i: sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1 for all j
        # Wait, that's not quite right. Let me re-derive.
        
        # The identity is: for all i in [0,n), j in [0,m), k in [0,p):
        #   sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # If we fix U and V, then for fixed (k,i), the equation becomes:
        #   sum_{c=0}^{r-1} W[k*n+i, c] * (U[i*m+j, c] * V[j*p+k, c]) = 1, for all j
        # So for fixed (k,i), we have m equations in r unknowns (W[k*n+i, 0..r-1]).
        # Total equations: p*n*m = 9*4*4 = 144
        # Total unknowns: p*n*r = 9*4*r = 36r
        # We need 36r >= 144, so r >= 4.
        
        # Let's try this approach:
        # 1. Randomly initialize U and V with small integer/half-integer values
        # 2. For each (k,i), solve the linear system for W[k*n+i, :]
        # 3. Check if the solution is integer/half-integer
        
        # But the linear system for each (k,i) is:
        #   A_{j,c} * W_{k,i,c} = 1 for j=0..m-1
        # where A_{j,c} = U[i*m+j, c] * V[j*p+k, c]
        # This is an m x r system. If m < r, it's underdetermined.
        # We need the system to have a unique solution, so we need r <= m? No, that's not right.
        # Actually, we have m equations in r unknowns. For a unique solution, we need r <= m.
        # But r is the rank, which is at least 9 (lower bound). And m=4. So r > m, meaning
        # the system is underdetermined. We need additional constraints.
        
        # Alternative: we can add the constraint that W has a specific sparsity pattern.
        # Or we can use the fact that we want integer/half-integer solutions.
        
        # Let me try a different approach: use the fact that for small ranks, we can
        # enumerate possible patterns.
        
        # Actually, let's try a simpler approach: for small r (say r <= 20),
        # try random sparse patterns and solve the full 3-linear system using
        # iterative methods or by fixing one matrix.
        
        pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    # Start with schoolbook
    best_result = _schoolbook_solution()
    best_rank = len(best_result["U"][0])
    
    # Try to find lower rank solutions
    # Lower bound is max(n,m,p) = 9
    # Upper bound is schoolbook = 144
    
    # Try ranks from lower bound up to schoolbook
    # For each rank, try random sparse patterns
    
    n, m, p = DIMENSIONS
    lower_bound = max(n, m, p)
    
    for r in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple random attempts for this rank
        num_attempts = 100
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            # Randomly initialize U, V with small values
            # Then solve for W
            U = [[0] * r for _ in range(n * m)]
            V = [[0] * r for _ in range(m * p)]
            
            # Fill U and V with random small integers
            for i in range(n * m):
                for c in range(r):
                    if random.random() < 0.3:  # 30% non-zero
                        U[i][c] = random.choice([-2, -1, 1, 2])
            
            for j in range(m * p):
                for c in range(r):
                    if random.random() < 0.3:  # 30% non-zero
                        V[j][c] = random.choice([-2, -1, 1, 2])
            
            # Solve for W
            W = [[0] * r for _ in range(p * n)]
            success = True
            
            for k in range(p):
                for i in range(n):
                    # Build the m x r system: A_{j,c} * W_{k,i,c} = 1
                    A = [[0] * r for _ in range(m)]
                    b = [1] * m
                    
                    for j in range(m):
                        for c in range(r):
                            A[j][c] = U[i * m + j][c] * V[j * p + k][c]
                    
                    # Solve A * W_col = b where W_col = W[k*n+i, :]
                    # This is m equations in r unknowns
                    # Use least squares or check if consistent
                    try:
                        A_np = np.array(A, dtype=float)
                        b_np = np.array(b, dtype=float)
                        W_col, residuals, rank_a, sv = np.linalg.lstsq(A_np, b_np, rcond=None)
                        
                        # Check if the solution is close to integer/half-integer
                        W_col_rounded = np.round(W_col * 2) / 2
                        if np.allclose(W_col, W_col_rounded, atol=1e-6):
                            for c in range(r):
                                W[k * n + i][c] = W_col_rounded[c]
                        else:
                            success = False
                            break
                    except:
                        success = False
                        break
                
                if not success:
                    break
            
            if success:
                # Verify the solution
                if _verify_decomposition(U, V, W) and _check_half_integer_solution(U, V, W):
                    result = {"U": U, "V": V, "W": W}
                    return result
    
    return best_result
# EVOLVE-BLOCK-END
