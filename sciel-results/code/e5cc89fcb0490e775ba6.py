import numpy as np
import random
import time
from scipy.optimize import linear_sum_assignment

DIMENSIONS = [3, 4, 5]

def check_decomposition(U, V, W, n, m, p):
    """
    Check if U, V, W form a valid decomposition.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    Condition: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k
    """
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0
                for r in range(R):
                    val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                if val != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    base_rank = n * m * p  # 60
    
    # Try to find a rank-59 decomposition using matching-based approach
    # The idea: we want to cover all (i,j,k) triples with R rank-1 terms
    # Each term r contributes to (i,j,k) if U[i,j,r]*V[j,k,r]*W[k,i,r] != 0
    # We need sum over r of these products = 1 for each (i,j,k)
    
    # Start with schoolbook and try to merge one pair
    # For 3x4x5, rank 59 is known to be achievable
    
    # Construct schoolbook first
    U_school = np.zeros((n * m, base_rank), dtype=float)
    V_school = np.zeros((m * p, base_rank), dtype=float)
    W_school = np.zeros((p * n, base_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Try to find a rank-59 solution by removing one column and solving
    # We'll try each possible column to remove and see if we can adjust
    
    best_U, best_V, best_W = None, None, None
    
    # Time limit: 100 seconds
    start_time = time.time()
    
    # Try removing each schoolbook term and see if we can fix with half-integer adjustments
    for remove_col in range(base_rank):
        if time.time() - start_time > 100:
            break
            
        R = base_rank - 1  # 59
        U = np.zeros((n * m, R), dtype=float)
        V = np.zeros((m * p, R), dtype=float)
        W = np.zeros((p * n, R), dtype=float)
        
        # Copy all columns except the removed one
        col_src = 0
        for c in range(R):
            if col_src == remove_col:
                col_src += 1
            U[:, c] = U_school[:, col_src]
            V[:, c] = V_school[:, col_src]
            W[:, c] = W_school[:, col_src]
            col_src += 1
        
        # Now we need to adjust to compensate for the missing term
        # The missing term was for some (i0, j0, k0)
        # Find which (i0, j0, k0) was removed
        # In schoolbook, col corresponds to (i,j,k) where:
        # col = i*m*p + j*p + k? No, the order is i, j, k nested
        # Let's figure out: col increments for each (i,j,k) in order i outer, j middle, k inner
        # So col = i * (m*p) + j * p + k
        i0 = remove_col // (m * p)
        rem = remove_col % (m * p)
        j0 = rem // p
        k0 = rem % p
        
        # The missing contribution is 1 at (i0, j0, k0)
        # We need to distribute this 1 across the remaining terms
        # Each remaining term r has U[i,j,r], V[j,k,r], W[k,i,r]
        # We can add corrections
        
        # Simple approach: add 1 to U[i0, j0, r] for some r where V[j0, k0, r]=1 and W[k0, i0, r]=1
        # Or adjust multiple terms
        
        # Try: find a term r where V[j0, k0, r] = 1 and W[k0, i0, r] = 1
        # Then we can set U[i0, j0, r] += 1
        found = False
        for r in range(R):
            if V[j0 * p + k0, r] == 1 and W[k0 * n + i0, r] == 1:
                U[i0 * m + j0, r] += 1
                found = True
                break
        
        if found:
            # Verify
            if check_decomposition(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                break
    
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # If no rank-59 found, return schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

# EVOLVE-BLOCK-END
