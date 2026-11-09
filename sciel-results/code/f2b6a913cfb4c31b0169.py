"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p):
    """Check if the decomposition is exact using integer arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _verify_numpy(U, V, W, n, m, p):
    """Verify using numpy for speed."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U_np[i * m + j, c] * V_np[j * p + k, c] * W_np[k * n + i, c] for c in range(rank))
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_strassen_2x3x3(rank, deadline):
    """
    Try to find a rank-r decomposition using a structured ansatz.
    
    We use the following strategy:
    - Parameterize U, V, W with a small number of integer parameters
    - Use the structure of 2x3x3 to reduce the search space
    - Solve the resulting system of equations
    
    The key insight: we can decompose the 2x3x3 tensor as a combination of
    smaller tensor products, similar to Strassen's approach.
    """
    if time.time() > deadline:
        return None
    
    n, m, p = 2, 3, 3
    
    # Strategy: Build the decomposition by combining known sub-decompositions
    # and adding correction terms.
    
    # Base: schoolbook has 18 terms. We want to reduce this.
    # 
    # Approach: Use the fact that for 2x3, we can use a rank-5 decomposition
    # (from literature) and combine it with 3x3 multiplications.
    #
    # Actually, let's try a more direct approach:
    # Enumerate all possible "pattern" decompositions where each column
    # of U, V, W has entries in {-1, 0, 1} and check if they satisfy
    # the tensor equation.
    
    # For rank r, we have r columns. Each column c has:
    # U_c: a vector in {-1,0,1}^6 (for 2x3 = 6 entries)
    # V_c: a vector in {-1,0,1}^9 (for 3x3 = 9 entries)  
    # W_c: a vector in {-1,0,1}^6 (for 3x2 = 6 entries)
    
    # The constraint is: for all (i,j,k), sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] = 1
    
    # This is a combinatorial search. Let's be smart about it.
    
    # Key observation: We can fix U to have a specific structure, then
    # the problem becomes: find V and W such that the tensor equation holds.
    
    # Let's try: U has a specific sparse pattern, then solve for V, W.
    
    # Approach 1: Fix U to be the "identity-like" pattern where each row
    # has exactly one non-zero entry, then solve for V and W.
    
    # Actually, let's try a different approach: use the known rank-15 
    # decomposition for 2x3x3 from the literature.
    
    # The rank of M_{2,3,3} is known to be 15 (not 14 as I initially thought).
    # Let me try to construct it.
    
    # Construction idea:
    # M_{2,3,3} can be decomposed as:
    # - One 2x2x2 block (rank 7 via Strassen)
    # - Plus correction terms for the remaining dimensions
    
    # Let me try a direct enumeration approach with a smart pruning.
    
    # For rank 15, we have 15 columns. Let's try to find them.
    
    # We'll use a backtracking search with constraint propagation.
    
    # First, let's establish what the tensor equation requires.
    # T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1,2}
    
    # The decomposition is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # Let's try to find a solution with entries in {-1, 0, 1}.
    
    # We'll use a randomized search with local optimization.
    
    best_score = float('inf')
    best_solution = None
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Initialize with random sparse patterns
        np.random.seed(attempt)
        
        # Try different sparsity levels
        for sparsity in [0.3, 0.5, 0.7]:
            if time.time() > deadline:
                break
            
            U = np.zeros((n*m, rank), dtype=int)
            V = np.zeros((m*p, rank), dtype=int)
            W = np.zeros((p*n, rank), dtype=int)
            
            for c in range(rank):
                # Randomly assign non-zero entries
                for idx in range(n*m):
                    if np.random.random() < sparsity:
                        U[idx, c] = np.random.choice([-1, 1])
                for idx in range(m*p):
                    if np.random.random() < sparsity:
                        V[idx, c] = np.random.choice([-1, 1])
                for idx in range(p*n):
                    if np.random.random() < sparsity:
                        W[idx, c] = np.random.choice([-1, 1])
            
            # Now try to optimize by flipping signs and zeroing entries
            for iteration in range(100):
                if time.time() > deadline:
                    break
                
                # Compute current error
                errors = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            errors[i,j,k] = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                
                # Check if we have a solution
                if np.allclose(errors, 1.0):
                    # Convert to lists
                    U_list = [[int(U[i*m+j, c]) for c in range(rank)] for i in range(n) for j in range(m)]
                    V_list = [[int(V[j*p+k, c]) for c in range(rank)] for j in range(m) for k in range(p)]
                    W_list = [[int(W[k*n+i, c]) for c in range(rank)] for k in range(p) for i in range(n)]
                    return {"U": U_list, "V": V_list, "W": W_list}
                
                # Try to improve by flipping a random entry
                # Pick the (i,j,k) with largest error
                i, j, k = np.unravel_index(np.argmax(np.abs(errors - 1.0)), (n, m, p))
                
                # Find which column contributes most to this error
                contribs = np.array([U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank)])
                
                # Try flipping signs in U, V, or W for the most contributing column
                c_max = np.argmax(np.abs(contribs))
                
                # Try all three flips
                improved = False
                for matrix_name in ['U', 'V', 'W']:
                    if matrix_name == 'U':
                        old_val = U[i*m+j, c_max]
                        U[i*m+j, c_max] = -old_val
                    elif matrix_name == 'V':
                        old_val = V[j*p+k, c_max]
                        V[j*p+k, c_max] = -old_val
                    else:
                        old_val = W[k*n+i, c_max]
                        W[k*n+i, c_max] = -old_val
                    
                    # Recompute error for this (i,j,k)
                    new_error = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                    
                    if abs(new_error - 1.0) < abs(errors[i,j,k] - 1.0):
                        improved = True
                        break
                    else:
                        # Revert
                        if matrix_name == 'U':
                            U[i*m+j, c_max] = old_val
                        elif matrix_name == 'V':
                            V[j*p+k, c_max] = old_val
                        else:
                            W[k*n+i, c_max] = old_val
                
                if not improved:
                    break
    
    return None

def _try_direct_search(rank, deadline):
    """
    Try to find a rank-r decomposition by direct enumeration of sparse patterns.
    Uses a smarter approach: fix the structure of U, then solve for V and W.
    """
    if time.time() > deadline:
        return None
    
    n, m, p = 2, 3, 3
    
    # For a given rank r, we need to find U (6 x r), V (9 x r), W (6 x r)
    # such that for all (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Key insight: if we fix U, then for each (i,k), we have m equations in r unknowns (W[k*n+i, c]).
    # The equations are: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for j=0..m-1
    
    # This is a linear system in W[k*n+i, c] if U and V are fixed.
    # But V is also unknown.
    
    # Alternative approach: use the fact that the tensor has a specific structure.
    # Let's try to build the decomposition incrementally.
    
    # Start with the schoolbook decomposition (rank 18) and try to merge terms.
    
    # Schoolbook: for each (i,j,k), we have a term with U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1
    # where c = i*m*p + j*p + k
    
    # We want to find a way to combine these terms.
    
    # Idea: if two terms have the same "pattern" in some sense, we can combine them.
    
    # Let's try a different approach: use linear algebra.
    # The tensor T can be viewed as a 3D array. We want to find its rank.
    # 
    # Flatten T into a matrix: T_mat[a,b] where a = i*m*p + j*p + k (for U index)
    # and b = ... this doesn't quite work for tensor rank.
    
    # Let's try a numerical approach: use the fact that we can fix one factor
    # and solve for the others.
    
    # Fix U to be a specific matrix, then the problem becomes:
    # Find V, W such that for all (i,j,k): sum_c V[j*p+k,c] * W[k*n+i,c] = 1 / U[i*m+j,c]
    # This only works if U[i*m+j,c] != 0, which is problematic.
    
    # Better approach: use the tensor equation as a system of polynomial equations
    # and solve it using a numerical method.
    
    # For rank r, we have 6r + 9r + 6r = 21r unknowns and 18 equations.
    # For r >= 2, this is underdetermined, so there should be solutions.
    
    # Let's try to find a solution using a greedy approach.
    
    # Initialize U, V, W with random values
    np.random.seed(42)
    
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for trial in range(100):
        if time.time() > deadline:
            break
        
        # Random initialization with small integers
        U = np.random.randint(-1, 2, size=(n*m, rank))
        V = np.random.randint(-1, 2, size=(m*p, rank))
        W = np.random.randint(-1, 2, size=(p*n, rank))
        
        # Ensure at least some non-zero entries
        for c in range(rank):
            if np.all(U[:, c] == 0):
                U[np.random.randint(n*m), c] = 1
            if np.all(V[:, c] == 0):
                V[np.random.randint(m*p), c] = 1
            if np.all(W[:, c] == 0):
                W[np.random.randint(p*n), c] = 1
        
        # Local search: try to improve by changing individual entries
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            # Compute error
            errors = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        errors[i,j,k] = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
            
            total_error = np.sum((errors - 1.0)**2)
            
            if total_error < best_error:
                best_error = total_error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if total_error < 1e-10:
                # Found a solution!
                U_list = [[int(U[i*m+j, c]) for c in range(rank)] for i in range(n) for j in range(m)]
                V_list = [[int(V[j*p+k, c]) for c in range(rank)] for j in range(m) for k in range(p)]
                W_list = [[int(W[k*n+i, c]) for c in range(rank)] for k in range(p) for i in range(n)]
                return {"U": U_list, "V": V_list, "W": W_list}
            
            # Pick the worst (i,j,k)
            i, j, k = np.unravel_index(np.argmax(np.abs(errors - 1.0)), (n, m, p))
            
            # Try changing U[i*m+j, c], V[j*p+k, c], or W[k*n+i, c] for some c
            improved = False
            for c in range(rank):
                if time.time() > deadline:
                    break
                
                # Try changing U
                old = U[i*m+j, c]
                for new_val in [-1, 0, 1]:
                    if new_val == old:
                        continue
                    U[i*m+j, c] = new_val
                    new_error = sum(U[i*m+j, cc] * V[j*p+k, cc] * W[k*n+i, cc] for cc in range(rank))
                    if abs(new_error - 1.0) < abs(errors[i,j,k] - 1.0) - 0.1:
                        improved = True
                        break
                    U[i*m+j, c] = old
                if improved:
                    break
                
                # Try changing V
                old = V[j*p+k, c]
                for new_val in [-1, 0, 1]:
                    if new_val == old:
                        continue
                    V[j*p+k, c] = new_val
                    new_error = sum(U[i*m+j, cc] * V[j*p+k, cc] * W[k*n+i, cc] for cc in range(rank))
                    if abs(new_error - 1.0) < abs(errors[i,j,k] - 1.0) - 0.1:
                        improved = True
                        break
                    V[j*p+k, c] = old
                if improved:
                    break
                
                # Try changing W
                old = W[k*n+i, c]
                for new_val in [-1, 0, 1]:
                    if new_val == old:
                        continue
                    W[k*n+i, c] = new_val
                    new_error = sum(U[i*m+j, cc] * V[j*p+k, cc] * W[k*n+i, cc] for cc in range(rank))
                    if abs(new_error - 1.0) < abs(errors[i,j,k] - 1.0) - 0.1:
                        improved = True
                        break
                    W[k*n+i, c] = old
                if improved:
                    break
                
                if improved:
                    break
            
            if not improved:
                break
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    # Try to find a better solution using structured search
    for rank in range(15, rank_schoolbook + 1):
        if time.time() > deadline:
            break
        
        # Try Strassen-like approach
        result = _try_strassen_2x3x3(rank, deadline)
        if result is not None:
            # Verify the solution
            if _verify_numpy(result["U"], result["V"], result["W"], n, m, p):
                return result
        
        # Try direct search
        result = _try_direct_search(rank, deadline)
        if result is not None:
            # Verify the solution
            if _verify_numpy(result["U"], result["V"], result["W"], n, m, p):
                return result
    
    # Return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
