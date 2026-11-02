"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    Uses a rank-26 decomposition based on Strassen's algorithm for the 2x2 core
    and standard multiplication for the remaining border terms.
    Attempts to find a lower rank decomposition via search.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = DIMENSIONS
    
    def idx_u(i, j):
        return i * m + j
    def idx_v(j, k):
        return j * p + k
    def idx_w(k, i):
        return k * n + i
    
    def verify(U, V, W, R):
        U_reshaped = U.reshape((n, m, R))
        V_reshaped = V.reshape((m, p, R))
        W_reshaped = W.reshape((p, n, R))
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        T_target = np.ones((n, m, p))
        return np.allclose(T_recon, T_target, atol=1e-9)
    
    # Build the rank-26 baseline
    def build_rank26():
        R = 26
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # Strassen 7 terms
        terms = [
            # (U_entries, V_entries, W_entries)
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]),
            ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
            ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)]),
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
        ]
        
        for r, (u_terms, v_terms, w_terms) in enumerate(terms):
            for (i, j, val) in u_terms:
                U[idx_u(i, j), r] = val
            for (j, k, val) in v_terms:
                V[idx_v(j, k), r] = val
            for (k, i, val) in w_terms:
                W[idx_w(k, i), r] = val
        
        # Standard 19 terms
        col = 7
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i < 2 and j < 2 and k < 2:
                        continue
                    U[idx_u(i, j), col] = 1
                    V[idx_v(j, k), col] = 1
                    W[idx_w(k, i), col] = 1
                    col += 1
        
        return U, V, W, R
    
    U, V, W, R = build_rank26()
    best_U, best_V, best_W, best_R = U.copy(), V.copy(), W.copy(), R
    
    if not verify(U, V, W, R):
        # Fallback to schoolbook
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[idx_u(i, j), col] = 1
                    V[idx_v(j, k), col] = 1
                    W[idx_w(k, i), col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a rank-24 or rank-25 decomposition
    # Strategy: exploit the structure of border terms by finding linear dependencies
    
    # The 19 border terms have indices:
    border_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                border_indices.append((i, j, k))
    
    # The border tensor has a specific structure:
    # - Terms with i=2: (2,j,k) for j,k in {0,1,2} -> 9 terms
    # - Terms with j=2: (i,2,k) for i,k in {0,1} -> 4 terms  
    # - Terms with k=2: (i,j,2) for i,j in {0,1} -> 4 terms
    # Total: 17 unique border positions (some overlap in counting)
    
    # Key insight: We can try to find if the border sub-tensor has rank < 19
    # by checking if we can represent it with fewer rank-1 terms
    
    # Approach: Use the fact that many border terms share indices
    # Try to find a valid decomposition where some border terms are combined
    
    # First, let's try a structured approach:
    # Group border terms by their "type" and see if we can merge within groups
    
    if time.time() < deadline:
        # Try to find rank-25 by checking if any two border terms can be merged
        # into a single non-trivial term that doesn't interfere with others
        
        # For each pair of border terms, check if there exists a rank-1 tensor
        # that covers both and zeros out all other non-covered positions
        
        found_rank25 = False
        for a_idx in range(len(border_indices)):
            if time.time() > deadline - 30:
                break
            if found_rank25:
                break
            for b_idx in range(a_idx + 1, len(border_indices)):
                (i1, j1, k1) = border_indices[a_idx]
                (i2, j2, k2) = border_indices[b_idx]
                
                # Skip if they share any index in a way that creates conflicts
                # A rank-1 term (u, v, w) contributes to position (i,j,k) as u[i,j]*v[j,k]*w[k,i]
                # For it to cover both (i1,j1,k1) and (i2,j2,k2) with value 1,
                # and zero elsewhere, we need specific conditions
                
                # Check if the two positions are "compatible" for a single rank-1 term
                # They must share either the same i, same j, or same k (otherwise 
                # the rank-1 term would need to be zero in one of them)
                
                if i1 != i2 and j1 != j2 and k1 != k2:
                    continue  # Can't be covered by a single rank-1 term
                
                # Try to construct a rank-1 term that covers both
                # This is complex; instead, let's try a different approach
                
                # Try to see if we can eliminate one border term by adjusting
                # the Strassen terms or other border terms
                
                # For now, just try random search with more attempts
                pass
    
    # More effective approach: Direct search for rank-25 by trying to
    # find a valid set of 18 border terms (instead of 19) that still
    # covers all required positions
    
    if time.time() < deadline:
        # The border tensor needs to satisfy:
        # For each (i,j,k) in border: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
        # For each (i,j,k) in core: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 0
        #   (already handled by Strassen terms)
        
        # Try to find if we can drop one border term and compensate with
        # non-trivial coefficients in other terms
        
        # This is a hard problem; let's use a more systematic approach
        # Try all subsets of size 18 from the 19 border terms
        
        np.random.seed(42)
        for target_R in [25, 24]:
            if time.time() > deadline - 20:
                break
            
            # For rank-25: try to find 18 border terms that work
            # We'll use a greedy approach: start with all 19, try to remove one
            # and see if the remaining 18 can be adjusted to still work
            
            for attempt in range(30):
                if time.time() > deadline - 20:
                    break
                
                U_try = np.zeros((n * m, target_R))
                V_try = np.zeros((m * p, target_R))
                W_try = np.zeros((p * n, target_R))
                
                # Initialize with the Strassen 7 terms
                terms = [
                    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
                    ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]),
                    ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
                    ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
                    ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)]),
                    ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
                    ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
                ]
                
                for r, (u_terms, v_terms, w_terms) in enumerate(terms):
                    if r >= target_R:
                        break
                    for (i, j, val) in u_terms:
                        U_try[idx_u(i, j), r] = val
                    for (j, k, val) in v_terms:
                        V_try[idx_v(j, k), r] = val
                    for (k, i, val) in w_terms:
                        W_try[idx_w(k, i), r] = val
                
                # For the remaining terms, try to assign border positions
                # with some non-trivial patterns
                remaining = target_R - 7
                for r in range(7, target_R):
                    # Choose a random border position to cover
                    border_pos = np.random.choice(len(border_indices))
                    i, j, k = border_indices[border_pos]
                    
                    # Assign 1 to this position
                    U_try[idx_u(i, j), r] = 1
                    V_try[idx_v(j, k), r] = 1
                    W_try[idx_w(k, i), r] = 1
                
                if verify(U_try, V_try, W_try, target_R):
                    best_U, best_V, best_W, best_R = U_try.copy(), V_try.copy(), W_try.copy(), target_R
                    break
    
    # Final verification
    if not verify(best_U, best_V, best_W, best_R):
        # Fallback to schoolbook
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[idx_u(i, j), col] = 1
                    V[idx_v(j, k), col] = 1
                    W[idx_w(k, i), col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
