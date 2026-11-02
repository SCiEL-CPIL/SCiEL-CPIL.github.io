"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s safety margin
    
    # Target tensor: T[i*4+j, j*4+k, k*2+i] = 1 for all i,j,k
    # Shape: (n*m, m*p, p*n) = (8, 16, 8)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find low-rank decompositions
    # Start with schoolbook rank 32 as fallback
    best_rank = n * m * p
    best_decomp = None
    
    # Generate schoolbook decomposition
    U_school = np.zeros((n*m, best_rank))
    V_school = np.zeros((m*p, best_rank))
    W_school = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_decomp = (U_school, V_school, W_school)
    
    # Try to find better decompositions using numerical methods
    # Use alternating minimization / power iteration style approach
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Check if U,V,W exactly decompose T."""
        # Compute sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # and compare with T
        reconstructed = np.tensordot(U, V, axes=([1],[1]))  # (n*m, m*p) intermediate
        # Actually, let's do it properly
        # The tensor is sum_r U[:,r] V[:,r]^T ⊗ W[:,r] in some sense
        # More precisely: T_recon[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol
    
    def check_half_integer(U, V, W, tol=1e-6):
        """Check if entries are close to half-integers."""
        for mat in [U, V, W]:
            if not np.all(np.abs(mat * 2 - np.round(mat * 2)) < tol):
                return False
        return True
    
    def round_to_half_integer(mat):
        """Round entries to nearest half-integer."""
        return np.round(mat * 2) / 2.0
    
    # Try ranks from 16 upward
    for rank in range(16, best_rank):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 30:
                break
            
            # Random initialization
            np.random.seed(rank * 1000 + attempt)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Normalize to avoid divergence
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            # Alternating minimization
            converged = False
            for it in range(200):
                if time.time() > deadline - 30:
                    break
                
                # Update U: for each column r, U[:,r] should be proportional to (V[:,r] ⊗ W[:,r]) flattened
                # T_recon[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # For fixed V,W, optimal U[a,r] = T[a,b,c]*V[b,r]*W[c,r] summed over b,c
                U_new = np.einsum('abc,br,cr->ar', T, V, W)
                V_new = np.einsum('abc,ar,cr->br', T, U_new, W)
                W_new = np.einsum('abc,ar,br->cr', T, U_new, V_new)
                
                # Check convergence
                U = U_new
                V = V_new
                W = W_new
                
                # Normalize
                if np.linalg.norm(U) > 0:
                    U /= np.linalg.norm(U)
                if np.linalg.norm(V) > 0:
                    V /= np.linalg.norm(V)
                if np.linalg.norm(W) > 0:
                    W /= np.linalg.norm(W)
                
                # Check if converged
                if it > 10:
                    recon = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.max(np.abs(recon - T))
                    if err < 1e-8:
                        converged = True
                        break
            
            if converged:
                # Check if entries are close to half-integers
                if check_half_integer(U, V, W):
                    U_r = round_to_half_integer(U)
                    V_r = round_to_half_integer(V)
                    W_r = round_to_half_integer(W)
                    if verify_decomposition(U_r, V_r, W_r):
                        # Remove zero columns
                        mask = (np.abs(U_r).sum(axis=0) > 0) & (np.abs(V_r).sum(axis=0) > 0) & (np.abs(W_r).sum(axis=0) > 0)
                        U_r = U_r[:, mask]
                        V_r = V_r[:, mask]
                        W_r = W_r[:, mask]
                        actual_rank = U_r.shape[1]
                        if actual_rank < best_rank:
                            best_rank = actual_rank
                            best_decomp = (U_r, V_r, W_r)
                            break  # Found best for this rank, try lower
        
        if best_rank == rank:
            break  # Found the minimum
    
    U, V, W = best_decomp
    # Convert to lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
