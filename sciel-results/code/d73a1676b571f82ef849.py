"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    # Initialization: Rank matching
    # Heavy flows (high row sum A) to short distances (low row sum B)
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Sort facilities by descending flow
    heavy = np.argsort(-a_sums, kind="stable")
    # Sort locations by ascending distance
    central = np.argsort(b_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Helper to compute cost
    def compute_cost(p):
        # sum_{i,j} a[i,j] * b[p[i], p[j]]
        # Vectorized: (a * b[p[:, None], p[None, :]]).sum()
        # This creates an n x n matrix, which is 150x150 = 22500 elements. Fast.
        return np.sum(a * b[np.ix_(p, p)])

    current_cost = compute_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Precompute constants
    # We will use the vectorized 2-opt logic
    
    while time.time() < deadline - 5.0:  # Leave 5s for final check
        
        # Main 2-opt loop
        improved = False
        # We might do multiple passes if time permits
        # But a single pass with best-improvement per i is a good balance
        
        # Precompute M = A @ B_p.T
        # B_p is B[p, :]
        B_p = b[np.ix_(p, p)] # This is b[p][:, p]? No.
        # b[p, :] gives rows of b indexed by p.
        # We need b[p[i], p[j]] for M[i,j] = sum_k a[i,k] b[p[i], p[k]]?
        # Wait, M[i,j] in my derivation was A[i,:] @ B_p[j,:].
        # B_p[j,:] is the row of B corresponding to location p[j].
        # So B_p = b[p, :].
        
        B_p = b[p, :]
        M = a @ B_p.T
        DiagM = np.diag(M)
        DiagB_p = np.diag(B_p)
        
        for i in range(n):
            if time.time() > deadline - 2.0:
                break
            
            # Compute Delta for all j
            # D_raw = M[i, :] - M[i, i] - DiagM + M[:, i]
            D_raw = M[i, :] - M[i, i] - DiagM + M[:, i]
            
            # Correction term C
            Row_i = B_p[i, :]
            Col_pi = B_p[:, p[i]]
            # Gathered = Row_i[p] -> b[p[i], p[j]] for all j
            Gathered = Row_i[p]
            
            # C_j = A[i, j] * (B_p[j, p_i] - B_p[i, p_i] - DiagB_p[j] + B_p[i, p_j])
            # B_p[j, p_i] is Col_pi[j]
            # B_p[i, p_i] is Row_i[p[i]]
            # DiagB_p[j] is DiagB_p[j]
            # B_p[i, p_j] is Gathered[j]
            
            C = a[i, :] * (Col_pi - Row_i[p[i]] - DiagB_p + Gathered)
            
            Delta = D_raw + C
            Delta[i] = np.inf
            
            j_best = np.argmin(Delta)
            
            if Delta[j_best] < -1e-9:  # Small tolerance for floating point
                # Perform swap
                p[i], p[j_best] = p[j_best], p[i]
                
                # Update B_p
                B_p[i], B_p[j_best] = B_p[j_best], B_p[i]
                
                # Recompute M, DiagM, DiagB_p
                M = a @ B_p.T
                DiagM = np.diag(M)
                DiagB_p = np.diag(B_p)
                
                # Update cost
                current_cost += Delta[j_best]
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                    
                improved = True
                # Restart loop? Or continue?
                # Continuing is usually fine, but M is now stale for subsequent i's in this pass?
                # No, M is recomputed. But the loop index i moves forward.
                # The swap affected row i and j_best.
                # Subsequent i's will use the updated M.
                # However, we might miss improvements involving the new i if we don't re-evaluate?
                # Standard 2-opt usually restarts or continues. Continuing is faster.
                
        if not improved:
            # Local optimum reached.
            # Perturb and restart
            if time.time() > deadline - 10.0:
                break
            
            # Simple perturbation: swap 5 random pairs
            for _ in range(5):
                idx1 = np.random.randint(n)
                idx2 = np.random.randint(n)
                if idx1 != idx2:
                    p[idx1], p[idx2] = p[idx2], p[idx1]
            
            current_cost = compute_cost(p)
            # If perturbation made it worse, it's fine, we search from there.
            # If we want to be safer, we can keep best_p.
            
    # Final check
    if compute_cost(p) < best_cost:
        best_p = p.copy()
        
    return best_p.tolist()

# EVOLVE-BLOCK-END
