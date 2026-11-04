"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]
MAX_STEPS = 8

def _schoolbook(n, m, p):
    """
    Standard schoolbook matrix multiplication tensor decomposition.
    Rank = n * m * p.
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    Indexing:
    U[i*m+j, c]
    V[j*p+k, c]
    W[k*n+i, c]
    """
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

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill switch
    
    n, m, p = DIMENSIONS
    base_rank = n * m * p
    
    # 1. Establish the safe baseline (Schoolbook)
    best = _schoolbook(n, m, p)
    best_rank = base_rank
    
    def _verify_decomposition(U, V, W, n, m, p, rank):
        """Verify that the decomposition exactly computes matrix multiplication."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def _try_strassen_block(n, m, p, deadline):
        """
        Try to use Strassen's algorithm structure to reduce the rank.
        For 3x4x5, we can embed into 4x4x4 and use Strassen's rank-7 for 2x2x2 blocks.
        However, a direct rank reduction from 60 is non-trivial.
        
        Let's try a different approach: use known optimal decompositions for small matrices.
        The rank of 3x4x5 matrix multiplication is known to be at most 59 (from literature).
        
        Strategy: Use a structured search over small perturbations of the schoolbook method.
        """
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append((i, j, k))
        
        # Try to find a rank-59 decomposition by solving a linear system
        # For each term to eliminate, we need to find coefficients that compensate
        
        for elim_idx in range(base_rank):
            if time.time() > deadline - 30.0:
                return None
            
            i_e, j_e, k_e = terms[elim_idx]
            
            # Set up the problem: find adjustments to other terms that compensate for removing term elim_idx
            # The compensation must satisfy: for all (i,j,k), the sum of products equals 1
            
            # Identify terms that can contribute to the compensation
            # These are terms that share at least one index with (i_e, j_e, k_e)
            relevant_indices = []
            for idx in range(base_rank):
                if idx == elim_idx:
                    continue
                i_t, j_t, k_t = terms[idx]
                if i_t == i_e or j_t == j_e or k_t == k_e:
                    relevant_indices.append(idx)
            
            if len(relevant_indices) > 10:
                continue  # Too many variables, skip
            
            # Try to solve for coefficients using a simple approach
            # We'll use a greedy method: try to find a subset of relevant terms that can compensate
            
            # For each combination of relevant terms (up to 4), try to find coefficients
            from itertools import combinations
            
            max_combo_size = min(4, len(relevant_indices))
            found = False
            
            for combo_size in range(2, max_combo_size + 1):
                if time.time() > deadline - 15.0:
                    return None
                
                for combo in combinations(relevant_indices, combo_size):
                    # Try to find coefficients for this combination
                    # We need to solve: for all (i,j,k), sum_{c in combo} x_c * U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 - base_sum(i,j,k)
                    
                    # base_sum(i,j,k) is the sum from all terms except elim_idx and the combo
                    # For simplicity, let's just try small integer coefficients
                    
                    coeff_range = [-2, -1, 0, 1, 2]
                    
                    # Generate all coefficient combinations
                    for coeffs in product(coeff_range, repeat=combo_size):
                        if time.time() > deadline - 10.0:
                            return None
                        
                        # Build the modified decomposition
                        U_mod = [[0] * (base_rank - 1) for _ in range(n * m)]
                        V_mod = [[0] * (base_rank - 1) for _ in range(m * p)]
                        W_mod = [[0] * (base_rank - 1) for _ in range(p * n)]
                        
                        # Copy all terms except elim_idx
                        col = 0
                        for idx in range(base_rank):
                            if idx == elim_idx:
                                continue
                            i_t, j_t, k_t = terms[idx]
                            U_mod[i_t * m + j_t][col] = 1
                            V_mod[j_t * p + k_t][col] = 1
                            W_mod[k_t * n + i_t][col] = 1
                            col += 1
                        
                        # Apply adjustments
                        for idx, coeff in zip(combo, coeffs):
                            if coeff == 0:
                                continue
                            i_t, j_t, k_t = terms[idx]
                            # Find the column index for this term
                            col_idx = None
                            for c in range(base_rank - 1):
                                if U_mod[i_t * m + j_t][c] == 1 and V_mod[j_t * p + k_t][c] == 1 and W_mod[k_t * n + i_t][c] == 1:
                                    col_idx = c
                                    break
                            if col_idx is not None:
                                U_mod[i_t * m + j_t][col_idx] = 1 + coeff
                                V_mod[j_t * p + k_t][col_idx] = 1 + coeff
                                W_mod[k_t * n + i_t][col_idx] = 1 + coeff
                        
                        # Verify
                        if _verify_decomposition(U_mod, V_mod, W_mod, n, m, p, base_rank - 1):
                            found = True
                            break
                    
                    if found:
                        break
                
                if found:
                    break
            
            if found:
                return {"U": U_mod, "V": V_mod, "W": W_mod}
        
        return None
    
    # Try the Strassen-inspired search
    result = _try_strassen_block(n, m, p, deadline)
    if result is not None:
        best = result
        best_rank = base_rank - 1
    
    # Check time to ensure we don't accidentally timeout
    if time.time() > deadline - 5.0:
        return best

    return best
# EVOLVE-BLOCK-END
