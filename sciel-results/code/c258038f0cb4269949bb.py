"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from collections import defaultdict

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Schoolbook decomposition as guaranteed fallback
    schoolbook_rank = n * m * p  # 120
    
    def schoolbook_decomp():
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = schoolbook_decomp()
    best_rank = schoolbook_rank
    
    def verify(U, V, W):
        """Verify exact decomposition. Returns True if valid."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    # Strategy: Try structured low-rank decompositions
    # For 4x5x6, known lower bounds suggest rank around 60-80
    # Use a pattern-based approach: define which entries are nonzero, solve linear system
    
    # Approach 1: Try a "block" decomposition using 2x2 Strassen-like patterns
    # Partition 4x5 into blocks and 5x6 into blocks
    
    # Let's try a direct algebraic search for ranks from lower bound upward
    # Lower bound for matrix multiplication rank: max(n*p, m*p, n*m) = max(24,30,20) = 30
    # But practical lower bounds are higher. Let's try from 60 upward.
    
    # Key insight: Use a randomized algebraic search with verification
    # For each candidate rank, try to find U, V, W that satisfy the tensor equation
    
    # The tensor equation is: for all (i,j,k):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p = 120 equations in rank^3 unknowns (but structured)
    
    # Better approach: Fix W to be simple (e.g., permutation-like or sparse),
    # then solve for U and V
    
    # Let's try a structured approach: use a "diagonal" W matrix pattern
    # where W[k*n+i, r] is nonzero only for specific (k,i,r) combinations
    
    # Actually, let's try a more systematic approach:
    # Use the fact that we can always achieve rank = n*m*p (schoolbook)
    # and try to find linear dependencies to reduce rank
    
    # Method: Randomized search with algebraic verification
    # For each rank r, randomly initialize U, V with small integers/half-integers,
    # then solve for W using the linear system, and verify
    
    # For a given U and V, the equation for W is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    # This is linear in W for each (k,i) given U and V
    
    # Let's try: fix U and V randomly, solve for W, check if W has small entries
    
    random.seed(42)
    
    # Try ranks from a reasonable lower bound
    # For 4x5x6, let's try from 50 to best_rank-1
    for target_rank in range(50, best_rank):
        if time.time() > deadline:
            break
        
        # Try a few random initializations for this rank
        n_tries = 3
        found = False
        
        for attempt in range(n_tries):
            if time.time() > deadline:
                break
            
            # Randomly initialize U and V with small values
            # Use values from {-1, -0.5, 0, 0.5, 1} for simplicity
            U_trial = [[random.choice([0, 0, 0, 1, -1]) for _ in range(target_rank)] 
                      for _ in range(n * m)]
            V_trial = [[random.choice([0, 0, 0, 1, -1]) for _ in range(target_rank)] 
                      for _ in range(m * p)]
            
            # Now solve for W
            # For each (k, i), we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
            # This gives m equations for each (k,i) with target_rank unknowns W[k*n+i, r]
            # But it's not linear in W directly because of the product structure
            
            # Actually, for fixed U and V, the equation is:
            # For each (i, j, k): sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
            # Let A_{(i,j,k), r} = U[i*m+j, r] * V[j*p+k, r]
            # Then sum_r A_{(i,j,k), r} * W[k*n+i, r] = 1
            # This is a linear system in W!
            
            # Build the linear system
            # Unknowns: W[k*n+i, r] for k in range(p), i in range(n), r in range(target_rank)
            # Total unknowns: p * n * target_rank
            # Equations: n * m * p (one for each (i,j,k))
            
            num_unknowns = p * n * target_rank
            num_equations = n * m * p
            
            # Build matrix A (num_equations x num_unknowns) and vector b (num_equations)
            A = np.zeros((num_equations, num_unknowns))
            b = np.ones(num_equations)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(target_rank):
                            val = U_trial[i*m+j][r] * V_trial[j*p+k][r]
                            if val != 0:
                                # W index: k*n+i, r -> linear index
                                w_idx = (k * n + i) * target_rank + r
                                A[eq_idx, w_idx] = val
                        eq_idx += 1
            
            # Solve the linear system
            try:
                # Use least squares since the system may be underdetermined
                W_sol, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
                
                # Check if the solution is good
                if residuals.size > 0 and residuals[0] > 1e-8:
                    continue
                
                # Reshape W_sol
                W_trial = W_sol.reshape(p * n, target_rank)
                
                # Check if W has reasonably small entries (prefer simple values)
                # Round to nearest 0.5 and re-verify
                W_rounded = np.round(W_trial * 2) / 2
                W_trial_list = W_rounded.tolist()
                
                # Verify the rounded solution
                U_list = U_trial
                V_list = V_trial
                
                if verify(U_list, V_list, W_trial_list):
                    best_U, best_V, best_W = U_list, V_list, W_trial_list
                    best_rank = target_rank
                    found = True
                    break
                    
            except np.linalg.LinAlgError:
                continue
            except Exception:
                continue
        
        if found:
            # Found a valid decomposition at this rank, try to go lower
            continue
        else:
            # If we can't find one at this rank, try the next
            continue
    
    # Return the best result found
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
