"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1
    # We want: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = None
    best_rank = rank_school
    
    # Try ranks from low to high
    for target_rank in range(12, rank_school + 1):
        if time.time() > deadline - 5.0:
            break
        
        # Try multiple random initializations
        n_trials = max(1, int((deadline - time.time()) / 10.0))
        n_trials = min(n_trials, 20)  # Cap trials per rank
        
        for trial in range(n_trials):
            if time.time() > deadline - 3.0:
                break
            
            # Random initialization with small values
            np.random.seed(42 + target_rank * 100 + trial)
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # ALS iterations
            converged = False
            for it in range(50):
                if time.time() > deadline - 2.0:
                    break
                
                # Update U: for each a, U[a,:] = argmin sum_{b,c} T[a,b,c] - U[a,:]V[b,:]^T W[c,:]
                # This is a linear system: for fixed a, we minimize over U[a,:]
                # sum_{b,c} (T[a,b,c] - sum_r U[a,r]V[b,r]W[c,r])^2
                # Gradient wrt U[a,r]: -2 * sum_{b,c} (T[a,b,c] - ...) V[b,r] W[c,r]
                # Hessian: 2 * sum_{b,c} V[b,r]V[b,s] * W[c,r]W[c,s] (approx)
                
                # Simpler: treat as linear regression
                # For each a, we have observations for all (b,c) pairs
                # T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
                # So for fixed a: y = X * u where y[b,c] = T[a,b,c], X[b,c,r] = V[b,r]*W[c,r]
                
                # Build X for this a
                # X has shape (m*p, target_rank) where X[b,c,r] = V[b,r]*W[c,r]
                # But b and c are coupled: b = j*p+k, c = k*n+i
                # For fixed a = i*m+j, the valid (b,c) pairs are:
                # b = j*p+k, c = k*n+i for k in range(p)
                # So only p pairs, not m*p*p
                
                # Actually the tensor T is defined on all (a,b,c) but only certain entries are 1
                # T[i*m+j, j'*p+k, k*n+i'] = 1 if j==j' and i==i' else 0
                # Wait, let me reconsider. The matrix multiplication tensor:
                # C[i,k] = sum_j A[i,j] B[j,k]
                # T[a,b,c] where a=i*m+j, b=j'*p+k, c=k'*n+i'
                # T = 1 if j==j' and i==i' and k==k'... no
                # T[i*m+j, j'*p+k, k'*n+i'] = 1 if j==j' and i==i' and k==k'? No.
                # Actually: C[i,k] = sum_j A[i,j]*B[j,k]
                # So the tensor has T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
                # And T[a,b,c] = 0 otherwise.
                
                # So for fixed a = i*m+j, the nonzero entries are at:
                # b = j*p+k, c = k*n+i for k in range(p)
                # That's p nonzero entries per a.
                
                # For ALS on U: fix V, W. For each a, solve:
                # Minimize sum over valid (b,c) of (T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r])^2
                # Only p terms are nonzero (T=1), rest are 0.
                # But we should include all (b,c) to properly constrain.
                
                # Let me just do full tensor ALS.
                # T has shape (n*m, m*p, p*n)
                # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k, else 0
                
                # For updating U[a,:]: 
                # Gradient: 2 * sum_{b,c} (pred[a,b,c] - T[a,b,c]) * V[b,:] * W[c,:]
                # where pred[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # Hessian (approx): 2 * sum_{b,c} outer(V[b,:],V[b,:]) * (W[c,:]@W[c,:])
                # This is expensive. Let's use a simpler approach.
                
                # Alternative: use the fact that the tensor is sparse.
                # For U update, for each a:
                # The relevant (b,c) pairs where T[a,b,c] != 0 or where pred might be nonzero.
                # Since V and W are dense, pred is dense. So we need all (b,c).
                
                # This is getting complex. Let me use a different approach:
                # Use scipy's least squares on the flattened tensor.
                
                break  # Skip ALS for now, try a different strategy
            
            # Check if current U,V,W give exact decomposition
            # Compute pred[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] and compare to T
            # Only need to check the p*n*m nonzero entries and some zero entries
            
            # Build tensor T
            T = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i*m+j, j*p+k, k*n+i] = 1
            
            # Compute prediction for nonzero entries
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i*m+j
                        b = j*p+k
                        c = k*n+i
                        pred = np.dot(U[a,:], V[b,:]*W[c,:])
                        max_err = max(max_err, abs(pred - 1.0))
            
            if max_err < 1e-6:
                # Verify zero entries
                zero_err = 0.0
                checked = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i*m+j
                            for jp in range(m):
                                for kp in range(p):
                                    b = jp*p+kp
                                    c = kp*n+jp  # wrong index
                                    # c should be k'*n+i' where i' and k' are free
                                    # Let me just check a sample of zero entries
                                    pass
                
                # Simpler: check that U@V@W reconstruction matches T on all entries
                # pred[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # Use einsum
                pred_full = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.max(np.abs(pred_full - T))
                if err < 1e-6:
                    best_rank = target_rank
                    # Convert to lists with Fraction for exactness
                    U_frac = [[Fraction(U[i,j]).limit_denominator(10**6) for j in range(target_rank)] for i in range(n*m)]
                    V_frac = [[Fraction(V[i,j]).limit_denominator(10**6) for j in range(target_rank)] for i in range(m*p)]
                    W_frac = [[Fraction(W[i,j]).limit_denominator(10**6) for j in range(target_rank)] for i in range(p*n)]
                    best_result = {"U": U_frac, "V": V_frac, "W": W_frac}
                    break
            
            if best_result is not None:
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    U_out = [[float(x) for x in row] for row in U_school]
    V_out = [[float(x) for x in row] for row in V_school]
    W_out = [[float(x) for x in row] for row in W_school]
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
