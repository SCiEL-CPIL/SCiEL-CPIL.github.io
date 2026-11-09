"""Rewritten program for 2x4x5 matrix multiplication tensor decomposition."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    # Schoolbook rank is n*m*p = 40
    # We need to find a rank-minimal decomposition.
    # Known results: For 2x4x5, the rank is at most 2*4*5 = 40 (schoolbook).
    # But we can do better. Let's think about the structure.
    
    # The multiplication tensor T has entries T[i,j,k] = 1 if the output element (i,k) 
    # depends on input elements. Specifically, C[i,k] = sum_j A[i,j]*B[j,k].
    # So the tensor is T[i,j,k] where i in [0,n), j in [0,m), k in [0,p).
    # T[i,j,k] = 1 for all valid i,j,k (it's the full multiplication tensor).
    
    # Rank of matrix multiplication tensor M_{n,m,p} is known for small cases.
    # For 2x2x2, rank is 7 (Strassen).
    # For 2x4x5, let's try to find a good decomposition.
    
    # Strategy: Use the fact that we can decompose the 4x5 part using Strassen-like ideas.
    # Actually, let's just try to find a low-rank decomposition using optimization.
    
    # The tensor T is a 2x4x5 tensor. We want to write it as sum_{r=1}^{rank} u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^2, v_r in R^4, w_r in R^5.
    
    # This is equivalent to finding matrices U (2xrank), V (4xrank), W (5xrank) such that
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    
    # Let's use a randomized optimization approach.
    
    deadline = time.time() + 140  # Leave 10s margin
    
    def tensor_value(U, V, W):
        """Compute the reconstructed tensor from U, V, W."""
        # U: (n, rank), V: (m, rank), W: (p, rank)
        # T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
        result = np.zeros((n, m, p))
        for r in range(U.shape[1]):
            result += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        return result
    
    def error(U, V, W):
        """Compute the Frobenius norm of the error."""
        T = np.ones((n, m, p))  # The target tensor
        T_recon = tensor_value(U, V, W)
        return np.linalg.norm(T - T_recon)
    
    # Target tensor
    T_target = np.ones((n, m, p))
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from low to high
    for rank in range(1, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_iters = 5000
        found = False
        
        for trial in range(10):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n, rank)
            V = np.random.randn(m, rank)
            W = np.random.randn(p, rank)
            
            # Gradient-free optimization using coordinate descent
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Update U
                for i in range(n):
                    # T[i,:,:] = sum_r U[i,r] * V[:,r] * W[:,r]
                    # For fixed i, we want to minimize ||T[i,:,:] - sum_r U[i,r] * (V[:,r]*W[:,r])||^2
                    # This is a linear least squares problem in U[i,:]
                    A = np.zeros((m*p, rank))
                    for r in range(rank):
                        A[:,r] = (V[:,r] * W[:,r]).flatten()
                    b = T_target[i,:,:].flatten()
                    # Solve A * U[i,:] = b
                    try:
                        U[i,:] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
                
                # Update V
                for j in range(m):
                    A = np.zeros((n*p, rank))
                    for r in range(rank):
                        A[:,r] = (U[:,r] * W[:,r]).flatten()
                    b = T_target[:,j,:].flatten()
                    try:
                        V[j,:] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
                
                # Update W
                for k in range(p):
                    A = np.zeros((n*m, rank))
                    for r in range(rank):
                        A[:,r] = (U[:,r] * V[:,r]).flatten()
                    b = T_target[:,:,k].flatten()
                    try:
                        W[k,:] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
                
                # Check convergence
                err = error(U, V, W)
                if err < 1e-8:
                    found = True
                    break
                
                if it % 100 == 0 and err > 1e-2:
                    # Stagnation, try reinitializing
                    if trial < 5:
                        break
            
            if found:
                break
        
        if found:
            best_rank = rank
            best_U = U
            best_V = V
            best_W = W
            break
    
    # If we didn't find a low-rank decomposition, fall back to schoolbook
    if best_U is None:
        rank = n * m * p
        U = np.zeros((n, rank))
        V = np.zeros((m, rank))
        W = np.zeros((p, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i, col] = 1
                    V[j, col] = 1
                    W[k, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank
    
    # Now we need to construct the full U, V, W matrices for the interface
    # U_full: (n*m, rank), V_full: (m*p, rank), W_full: (p*n, rank)
    # The tensor is T[i,j,k] = sum_r U_full[i*m+j, r] * V_full[j*p+k, r] * W_full[k*n+i, r]
    
    # Wait, I need to be careful about the indexing convention.
    # The task says: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # And the output factor follows W[k*n+i, r]
    
    # So the decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # But my optimization was for T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # These are different! I need to adjust.
    
    # Let me redo the optimization with the correct indexing.
    
    best_rank = n * m * p
    best_U_full = None
    best_V_full = None
    best_W_full = None
    
    for rank in range(1, best_rank + 1):
        if time.time() > deadline:
            break
        
        found = False
        for trial in range(10):
            if time.time() > deadline:
                break
            
            U_full = np.random.randn(n*m, rank)
            V_full = np.random.randn(m*p, rank)
            W_full = np.random.randn(p*n, rank)
            
            for it in range(2000):
                if time.time() > deadline:
                    break
                
                # Update each row of U_full
                for idx in range(n*m):
                    i, j = divmod(idx, m)
                    # T[i,j,k] = sum_r U_full[idx,r] * V_full[j*p+k,r] * W_full[k*n+i,r]
                    # For fixed i,j, we want to fit T[i,j,:] as a function of k
                    A = np.zeros((p, rank))
                    for r in range(rank):
                        for k in range(p):
                            A[k,r] = V_full[j*p+k, r] * W_full[k*n+i, r]
                    b = np.ones(p)  # T[i,j,k] = 1 for all k
                    try:
                        U_full[idx,:] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
                
                # Update each row of V_full
                for idx in range(m*p):
                    j, k = divmod(idx, p)
                    A = np.zeros((n, rank))
                    for r in range(rank):
                        for i in range(n):
                            A[i,r] = U_full[i*m+j, r] * W_full[k*n+i, r]
                    b = np.ones(n)
                    try:
                        V_full[idx,:] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
                
                # Update each row of W_full
                for idx in range(p*n):
                    k, i = divmod(idx, n)
                    A = np.zeros((m, rank))
                    for r in range(rank):
                        for j in range(m):
                            A[j,r] = U_full[i*m+j, r] * V_full[j*p+k, r]
                    b = np.ones(m)
                    try:
                        W_full[idx,:] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
                
                # Check error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            recon = sum(U_full[i*m+j, r] * V_full[j*p+k, r] * W_full[k*n+i, r] for r in range(rank))
                            err += (1 - recon)**2
                err = np.sqrt(err)
                
                if err < 1e-8:
                    found = True
                    break
                
                if it % 50 == 0 and err > 1e-2:
                    break
            
            if found:
                break
        
        if found:
            best_rank = rank
            best_U_full = U_full
            best_V_full = V_full
            best_W_full = W_full
            break
    
    if best_U_full is None:
        # Schoolbook fallback
        rank = n * m * p
        U_full = np.zeros((n*m, rank))
        V_full = np.zeros((m*p, rank))
        W_full = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_full[i*m+j, col] = 1
                    V_full[j*p+k, col] = 1
                    W_full[k*n+i, col] = 1
                    col += 1
        best_U_full, best_V_full, best_W_full = U_full, V_full, W_full
    
    # Convert to lists for the interface
    U_list = best_U_full.tolist()
    V_list = best_V_full.tolist()
    W_list = best_W_full.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
