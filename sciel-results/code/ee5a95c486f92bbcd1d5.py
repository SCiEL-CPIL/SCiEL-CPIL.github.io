"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Start with the rank-56 Strassen decomposition
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2x2
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
    ]
    
    current_rank = 0
    
    # Group 1: A1 * Btop
    for q in range(4):
        for s in range(7):
            u_loc, v_loc, w_loc = strassen_terms[s]
            r = current_rank
            for i in range(2):
                for j_loc in range(2):
                    U[i * m + j_loc][r] = u_loc[i * 2 + j_loc]
            for j_loc in range(2):
                for k_loc in range(2):
                    V[j_loc * p + (2 * q + k_loc)][r] = v_loc[j_loc * 2 + k_loc]
            for i in range(2):
                for k_loc in range(2):
                    W[(2 * q + k_loc) * n + i][r] = w_loc[k_loc * 2 + i]
            current_rank += 1
            
    # Group 2: A2 * Bbottom
    for q in range(4):
        for s in range(7):
            u_loc, v_loc, w_loc = strassen_terms[s]
            r = current_rank
            for i in range(2):
                for j_loc in range(2):
                    U[i * m + (j_loc + 2)][r] = u_loc[i * 2 + j_loc]
            for j_loc in range(2):
                for k_loc in range(2):
                    V[(j_loc + 2) * p + (2 * q + k_loc)][r] = v_loc[j_loc * 2 + k_loc]
            for i in range(2):
                for k_loc in range(2):
                    W[(2 * q + k_loc) * n + i][r] = w_loc[k_loc * 2 + i]
            current_rank += 1
            
    # Try to reduce rank by finding linear dependencies
    # The tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # We want to find if any r can be eliminated
    
    # Check if the decomposition is valid first
    def verify_decomposition(U, V, W, n, m, p, rank):
        """Verify that the decomposition exactly represents matrix multiplication."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1 if (j == k) else 0
                    actual = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    if abs(actual - expected) > 1e-10:
                        return False
        return True
    
    # Try to reduce rank by zeroing out terms and checking validity
    # This is a heuristic approach - try removing each term one at a time
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try to find linear dependencies using SVD-like approach
    # For each pair of terms, check if they can be combined
    
    # Simple approach: try to remove terms one by one
    for r_remove in range(rank - 1, -1, -1):
        trial_rank = rank
        trial_U = U.copy()
        trial_V = V.copy()
        trial_W = W.copy()
        
        # Zero out the term to remove
        trial_U[:, r_remove] = 0
        trial_V[:, r_remove] = 0
        trial_W[:, r_remove] = 0
        
        # Check if still valid
        if verify_decomposition(trial_U, trial_V, trial_W, n, m, p, trial_rank):
            # We found a reducible term
            best_U, best_V, best_W, best_rank = trial_U, trial_V, trial_W, trial_rank
            U, V, W, rank = trial_U, trial_V, trial_W, trial_rank
    
    # Convert to list format as required
    U_list = [[int(U[i, r]) for r in range(rank)] for i in range(n * m)]
    V_list = [[int(V[i, r]) for r in range(rank)] for i in range(m * p)]
    W_list = [[int(W[i, r]) for r in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
