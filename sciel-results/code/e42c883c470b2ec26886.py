import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0

    # Build the tensor T[i,j,k] = 1 for all i,j,k
    # We want to find rank-r decomposition:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Start with schoolbook rank 30
    rank_school = n * m * p  # 30
    U30 = np.zeros((n * m, rank_school))
    V30 = np.zeros((m * p, rank_school))
    W30 = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U30[i*m+j, col] = 1
                V30[j*p+k, col] = 1
                W30[k*n+i, col] = 1
                col += 1
    
    def verify(U, V, W, atol=1e-9):
        """Verify that the decomposition exactly reconstructs the all-ones tensor."""
        res = np.zeros((n, m, p))
        for c in range(U.shape[1]):
            Uc = U[:, c].reshape(n, m)
            Vc = V[:, c].reshape(m, p)
            Wc = W[:, c].reshape(p, n)
            # T[i,j,k] += U[i,j,c] * V[j,k,c] * W[k,i,c]
            res += Uc[:, :, np.newaxis] * Vc[np.newaxis, :, :] * Wc.T[:, np.newaxis, :]
        target = np.ones((n, m, p))
        return np.allclose(res, target, atol=atol)
    
    # Verify schoolbook
    if not verify(U30, V30, W30):
        # Should never happen, but return something valid
        return {"U": U30.tolist(), "V": V30.tolist(), "W": W30.tolist()}
    
    best_U, best_V, best_W = U30, V30, W30
    best_rank = rank_school
    
    # Try to reduce rank by removing terms one at a time
    # For each term, check if removing it still gives a valid decomposition
    # This is a greedy approach - try all single removals
    
    def try_remove_term(U, V, W, remove_idx):
        """Try removing term remove_idx and see if we can fix it with a small perturbation."""
        U_test = np.delete(U, remove_idx, axis=1)
        V_test = np.delete(V, remove_idx, axis=1)
        W_test = np.delete(W, remove_idx, axis=1)
        
        # Check if the remaining terms still span the target
        # Compute residual
        res = np.zeros((n, m, p))
        for c in range(U_test.shape[1]):
            Uc = U_test[:, c].reshape(n, m)
            Vc = V_test[:, c].reshape(m, p)
            Wc = W_test[:, c].reshape(p, n)
            res += Uc[:, :, np.newaxis] * Vc[np.newaxis, :, :] * Wc.T[:, np.newaxis, :]
        
        residual = np.ones((n, m, p)) - res
        
        # If residual is zero, we're good
        if np.allclose(residual, 0, atol=1e-9):
            return True, U_test, V_test, W_test
        
        # Check if residual can be expressed as a single rank-1 term
        # i.e., residual[i,j,k] = a[i,j] * b[j,k] * c[k,i]
        # This is a hard problem, so we'll use a heuristic:
        # Try to find a, b, c such that a[i,j]*b[j,k]*c[k,i] ≈ residual[i,j,k]
        
        # For a 2x3x5 tensor, we can try to factor the residual
        # residual[i,j,k] = a[i,j] * b[j,k] * c[k,i]
        
        # Take logarithms? No, residual can be negative or zero.
        # Let's try a different approach: for each (i,j), the slice residual[i,j,:] 
        # should be b[j,:] * c[:,i] (element-wise product)
        
        # Actually, let's just check if the residual has rank 1 in the tensor sense
        # by checking if it can be written as outer product of three vectors
        # in the appropriate modes.
        
        # For now, return False if residual is not zero
        return False, U_test, V_test, W_test
    
    # Try removing each term
    for remove_idx in range(best_rank):
        if time.time() > deadline - 15:
            break
        
        success, U_test, V_test, W_test = try_remove_term(best_U, best_V, best_W, remove_idx)
        if success:
            best_U, best_V, best_W = U_test, V_test, W_test
            best_rank -= 1
            # Reset loop to try removing more terms
            break
    
    # Now try a more aggressive approach: use optimization to find a lower-rank decomposition
    # We'll use a gradient-based approach to minimize the rank
    
    # First, let's try to find if rank 29 is possible by solving a system of equations
    # For rank r, we have r terms, each with 3*6=18 parameters (for 2x3x5)
    # Total parameters: r * 18
    # Constraints: 30 equations (one for each (i,j,k))
    # For r=29: 522 parameters, 30 constraints - should be solvable
    
    # But the problem is that the equations are nonlinear (trilinear)
    # Let's try a simple approach: start from the rank-30 solution and try to
    # merge two terms into one
    
    def try_merge_terms(U, V, W, idx1, idx2):
        """Try to merge terms idx1 and idx2 into a single term."""
        if idx1 == idx2:
            return None
        
        # The two terms contribute:
        # U1[i,j]*V1[j,k]*W1[k,i] + U2[i,j]*V2[j,k]*W2[k,i]
        # We want to find U3, V3, W3 such that this equals U3[i,j]*V3[j,k]*W3[k,i]
        
        # This is equivalent to finding a rank-1 approximation to the sum of two rank-1 tensors
        # which is generally not possible unless the two tensors are proportional
        
        # Check if the two terms are proportional
        U1 = U[:, idx1].reshape(n, m)
        U2 = U[:, idx2].reshape(n, m)
        V1 = V[:, idx1].reshape(m, p)
        V2 = V[:, idx2].reshape(m, p)
        W1 = W[:, idx1].reshape(p, n)
        W2 = W[:, idx2].reshape(p, n)
        
        # Check if U1 = alpha * U2, V1 = alpha * V2, W1 = alpha * W2 for some alpha
        # This is a strong condition, so it's unlikely to hold
        
        # Instead, let's try a numerical optimization approach
        # Minimize ||U1*V1*W1 + U2*V2*W2 - U3*V3*W3||^2 over U3, V3, W3
        
        # This is a non-convex problem, so we'll use a simple iterative approach
        # Alternate minimization: fix two, optimize the third
        
        # Initialize U3, V3, W3 with the sum of the two terms
        U3 = (U1 + U2) / 2
        V3 = (V1 + V2) / 2
        W3 = (W1 + W2) / 2
        
        # Normalize to avoid scale ambiguity
        # The product U3*V3*W3 is invariant to scaling: (a*U3)*(b*V3)*(c*W3) with abc=1
        
        # Use alternating least squares
        for iteration in range(50):
            # Optimize U3: minimize ||U1*V1*W1 + U2*V2*W2 - U3*(V3*W3)||^2
            # For each (i,j), we want U3[i,j] * (V3[j,:]*W3[:,i])_k ≈ residual_k
            # This is a linear least squares problem
            
            # Build the matrix for U3
            # residual[i,j,k] = U1[i,j]*V1[j,k]*W1[k,i] + U2[i,j]*V2[j,k]*W2[k,i]
            residual = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        residual[i,j,k] = U1[i,j]*V1[j,k]*W1[k,i] + U2[i,j]*V2[j,k]*W2[k,i]
            
            # For each (i,j), solve for U3[i,j]
            # residual[i,j,k] = U3[i,j] * (V3[j,k]*W3[k,i])
            # Let w[i,j,k] = V3[j,k]*W3[k,i]
            # Then U3[i,j] = sum_k residual[i,j,k]*w[i,j,k] / sum_k w[i,j,k]^2
            
            for i in range(n):
                for j in range(m):
                    w = V3[j, :] * W3[:, i]
                    if np.sum(w**2) > 1e-12:
                        U3[i,j] = np.sum(residual[i,j,:] * w) / np.sum(w**2)
                    else:
                        U3[i,j] = 0
            
            # Similarly for V3 and W3
            for j in range(m):
                for k in range(p):
                    w = U3[:, j] * W3[k, :]  # w[i] = U3[i,j]*W3[k,i]
                    if np.sum(w**2) > 1e-12:
                        V3[j,k] = np.sum(residual[:,j,k] * w) / np.sum(w**2)
                    else:
                        V3[j,k] = 0
            
            for k in range(p):
                for i in range(n):
                    w = U3[i, :] * V3[:, k]  # w[j] = U3[i,j]*V3[j,k]
                    if np.sum(w**2) > 1e-12:
                        W3[k,i] = np.sum(residual[i,:,k] * w) / np.sum(w**2)
                    else:
                        W3[k,i] = 0
            
            # Check convergence
            err = np.sum((residual - U3[:,:,np.newaxis]*V3[np.newaxis,:,:]*W3.T[:,np.newaxis,:])**2)
            if err < 1e-12:
                break
        
        # Check if the merge is successful
        final_err = np.sum((residual - U3[:,:,np.newaxis]*V3[np.newaxis,:,:]*W3.T[:,np.newaxis,:])**2)
        if final_err < 1e-9:
            # Build the new U, V, W
            U_new = np.delete(U, [idx1, idx2], axis=1)
            V_new = np.delete(V, [idx1, idx2], axis=1)
            W_new = np.delete(W, [idx1, idx2], axis=1)
            
            U_new = np.column_stack([U_new, U3.flatten()])
            V_new = np.column_stack([V_new, V3.flatten()])
            W_new = np.column_stack([W_new, W3.flatten()])
            
            return U_new, V_new, W_new
        
        return None
    
    # Try merging pairs of terms
    merged = False
    for idx1 in range(best_rank):
        if time.time() > deadline - 10:
            break
        for idx2 in range(idx1 + 1, best_rank):
            if time.time() > deadline - 5:
                break
            
            result = try_merge_terms(best_U, best_V, best_W, idx1, idx2)
            if result is not None:
                U_new, V_new, W_new = result
                if verify(U_new, V_new, W_new):
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank -= 1
                    merged = True
                    break
        if merged:
            break
    
    # Return the best result found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
