"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    rank_school = n * m * p  # 40
    
    # Build the matrix multiplication tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    deadline = time.time() + 110  # Safety margin
    
    # Try to find a decomposition with rank < 40
    # Use iterative method: initialize with random, then iterate
    
    best_rank = rank_school
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with schoolbook
    U0 = np.zeros((n*m, rank_school))
    V0 = np.zeros((m*p, rank_school))
    W0 = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U0, V0, W0
    
    # Try lower ranks
    for target_rank in range(25, rank_school):
        if time.time() > deadline:
            break
        
        found = False
        for trial in range(3):
            if time.time() > deadline:
                break
            
            np.random.seed(12345 + trial * 100 + target_rank)
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Iterative refinement
            for it in range(300):
                if time.time() > deadline:
                    break
                
                # Update U
                # For each a: T[a,:,:] = sum_r U[a,r] * outer(V[:,r], W[:,r])
                # Let Q = V @ W.T  (shape m*p x p*n) -- no, that's sum_r V[:,r] @ W[:,r].T which is different
                # Actually outer(V[:,r], W[:,r]) is m*p x p*n matrix
                # Sum_r U[a,r] * outer(V[:,r], W[:,r]) = (U[a,:] * V) @ W.T ... let's verify
                # (U[a,:] * V)[b] = sum_r U[a,r] * V[b,r]
                # ((U[a,:] * V) @ W.T)[b,c] = sum_b' (U[a,:] * V)[b'] * W[c,b'] = sum_b' sum_r U[a,r]*V[b',r]*W[c,b']
                # That's not what we want. We want sum_r U[a,r]*V[b,r]*W[c,r]
                # = (V * U[a,:].reshape(1,-1)) @ W.T ... 
                # (V * U[a,:].reshape(1,-1))[b,r] = V[b,r]*U[a,r]
                # ((V * U[a,:].reshape(1,-1)) @ W.T)[b,c] = sum_r V[b,r]*U[a,r]*W[c,r] ✓
                
                # So T[a,:,:] = (V * U[a,:].reshape(1,-1)) @ W.T
                # Let Y = T[a,:,:] @ W  (shape m*p x p*n @ p*n x rank = m*p x rank)
                # Then Y = V * U[a,:].reshape(1,-1)
                # So U[a,:] = (Y * V).sum(axis=0) / (V*V).sum(axis=0)  (least squares)
                
                # More carefully: we want to minimize ||(V * U[a,:].reshape(1,-1)) @ W.T - T[a,:,:]||^2
                # Let A = V * U[a,:].reshape(1,-1), then A @ W.T = T[a,:,:]
                # This is equivalent to: for each b, sum_r A[b,r]*W[c,r] = T[a,b,c] for all c
                # So A @ W.T = T[a,:,:], meaning A = T[a,:,:] @ W^+ (pseudo-inverse of W.T)
                # But W is p*n x rank, so W.T is rank x p*n. If rank < p*n, W.T has full row rank (generically)
                # Then W.T @ W = I (if W has orthonormal columns), but generally:
                # A @ W.T = T[a,:,:]  =>  A = T[a,:,:] @ (W.T)^+ 
                # (W.T)^+ = (W.T).T @ (W.T @ W.T)^-1 = W @ (W.T @ W.T)^-1  if W.T @ W.T is invertible
                
                # Simpler: just do least squares
                # We want A such that A @ W.T = T[a,:,:]
                # This is a linear system in A: for each b, A[b,:] @ W.T = T[a,b,:]
                # So A[b,:] = T[a,b,:] @ (W.T)^+
                # (W.T)^+ = (W.T).T @ inv(W.T @ W.T) = W @ inv(W.T @ W.T)
                
                Wt = W.T  # rank x p*n
                WtWt = Wt @ Wt.T  # rank x rank
                try:
                    Wt_pinv = W @ np.linalg.inv(WtWt)  # p*n x rank
                except:
                    Wt_pinv = np.linalg.pinv(Wt).T
                
                for a in range(n*m):
                    # A = T[a,:,:] @ Wt_pinv  (m*p x rank)
                    A = T[a,:,:] @ Wt_pinv
                    # A = V * U[a,:].reshape(1,-1)
                    # U[a,:] = (A * V).sum(axis=0) / (V*V).sum(axis=0)
                    denom = (V*V).sum(axis=0)
                    denom[denom < 1e-12] = 1e-12
                    U[a,:] = (A * V).sum(axis=0) / denom
                
                # Update V similarly
                # T[a,:,:] = sum_r U[a,r] * outer(V[:,r], W[:,r])
                # For fixed b: T[:,b,:] = sum_r V[b,r] * outer(U[:,r], W[:,r])
                # = (U * V[b,:].reshape(1,-1)) @ W.T
                # Let A = T[:,b,:] @ Wt_pinv (n*m x rank)
                # A = U * V[b,:].reshape(1,-1)
                # V[b,:] = (A * U).sum(axis=0) / (U*U).sum(axis=0)
                
                for b in range(m*p):
                    A = T[:,b,:] @ Wt_pinv
                    denom = (U*U).sum(axis=0)
                    denom[denom < 1e-12] = 1e-12
                    V[b,:] = (A * U).sum(axis=0) / denom
                
                # Update W similarly
                # T[a,:,:] = sum_r U[a,r] * outer(V[:,r], W[:,r])
                # For fixed c: T[:,:,c] = sum_r W[c,r] * outer(U[:,r], V[:,r])
                # = (U * W[c,:].reshape(1,-1)) @ V.T
                # Let A = T[:,:,c] @ Vt_pinv (n*m x rank)
                # A = U * W[c,:].reshape(1,-1)
                # W[c,:] = (A * U).sum(axis=0) / (U*U).sum(axis=0)
                
                Vt = V.T  # rank x m*p
                VtVt = Vt @ Vt.T
                try:
                    Vt_pinv = V @ np.linalg.inv(VtVt)
                except:
                    Vt_pinv = np.linalg.pinv(Vt).T
                
                for c in range(p*n):
                    A = T[:,:,c] @ Vt_pinv
                    denom = (U*U).sum(axis=0)
                    denom[denom < 1e-12] = 1e-12
                    W[c,:] = (A * U).sum(axis=0) / denom
                
                # Check convergence
                if it % 50 == 0:
                    # Compute residual
                    res = 0
                    for r in range(target_rank):
                        res += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                    err = np.linalg.norm(res - T)
                    if err < 1e-8:
                        found = True
                        break
            
            if found:
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
        
        if found:
            break
    
    # Convert to list
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

