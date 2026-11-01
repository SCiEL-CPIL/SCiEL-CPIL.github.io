"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook fallback
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U, V, W
    
    # Lower bound: max(n*p, m*p, n*m) = max(14, 28, 8) = 28
    min_rank = max(n * p, m * p, n * m)
    
    for target_rank in range(min_rank, rank_schoolbook):
        if time.time() > deadline - 10:
            break
        
        # Try to find a decomposition of this rank
        # Use random U, V and check if W can be solved
        success, U_list, V_list, W_list = _try_rank_decomposition_linear(n, m, p, target_rank, deadline)
        if success:
            best_rank = target_rank
            best_U, best_V, best_W = U_list, V_list, W_list
            break
    
    # Return the best found
    return {"U": best_U, "V": best_V, "W": best_W}


def _try_rank_decomposition_linear(n, m, p, target_rank, deadline):
    """
    Try to find a decomposition by fixing U and V and solving for W.
    Constraint: For each (i, k), sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all j.
    This is a linear system M_ik * w_ik = 1, where M_ik[j, c] = U[i, j, c] V[j, k, c].
    """
    start_time = time.time()
    max_time = deadline - start_time - 5  # safety margin
    
    # Number of trials depends on time
    # Each trial is O(n * p * (m * target_rank)^2) roughly for consistency check
    # m=4, n=2, p=7.
    # M_ik is 4 x r.
    # Check consistency: rank(M) == rank([M|1])
    
    num_trials = 0
    last_trial_time = start_time
    
    while time.time() < deadline - 5:
        num_trials += 1
        
        # Random U and V with entries in {-1, 0, 1}
        # To increase density, maybe use {-1, 1} or {-1, 0, 1}
        # Let's try {-1, 0, 1}
        U = np.random.randint(-1, 2, size=(n * m, target_rank))
        V = np.random.randint(-1, 2, size=(m * p, target_rank))
        
        # Check if W exists
        W_valid = True
        W_solution = np.zeros((p * n, target_rank))
        
        for i in range(n):
            for k in range(p):
                # Construct M_ik: shape (m, target_rank)
                # M_ik[j, c] = U[i*m+j, c] * V[j*p+k, c]
                M_ik = np.zeros((m, target_rank))
                for j in range(m):
                    M_ik[j, :] = U[i * m + j, :] * V[j * p + k, :]
                
                # System: M_ik * w_ik = ones(m)
                b = np.ones(m)
                
                # Check consistency
                # If rank(M_ik) < rank([M_ik, b]), no solution.
                # We can use np.linalg.matrix_rank
                rank_M = np.linalg.matrix_rank(M_ik)
                rank_aug = np.linalg.matrix_rank(np.column_stack((M_ik, b)))
                
                if rank_M != rank_aug:
                    W_valid = False
                    break
                
                # If consistent, find a solution.
                # We want integer or half-integer solutions.
                # Find minimum norm solution
                w_ik, residuals, rank, s = np.linalg.lstsq(M_ik, b, rcond=None)
                
                # Check if w_ik is close to integer or half-integer
                # Multiply by 2 to check for integers
                w_ik_scaled = w_ik * 2
                # Check if w_ik_scaled is close to integer
                # Tolerance 1e-6
                if not np.all(np.isclose(w_ik_scaled, np.round(w_ik_scaled), atol=1e-6)):
                    W_valid = False
                    break
                
                # Store solution (scaled back to original)
                W_solution[k * n + i, :] = w_ik
        
        if W_valid:
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W_solution.tolist()
            
            # Final verification to be safe
            if _verify_decomposition(np.array(U_list), np.array(V_list), np.array(W_list), n, m, p):
                return True, U_list, V_list, W_list
        
        # If time is running out, break
        if time.time() > deadline - 5:
            break
        
    return False, None, None


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    # The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # T[i,j,k] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(U.shape[1]):
                    total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if not np.isclose(total, 1, atol=1e-6):
                    return False
    return True
# EVOLVE-BLOCK-END
