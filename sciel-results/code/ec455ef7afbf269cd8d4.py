import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0
    
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the multiplication tensor T[a,b,c] where:
    # a = i*m+j (row of A * col of B index)
    # b = j*p+k (row of B * col of C index)  
    # c = k*n+i (row of C * col of A index)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all (i,j,k)
    
    # The decomposition: C = A * B where C[k,i] = sum_j A[i,j]*B[j,k]
    # Using factors U, V, W:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The equation is: sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r] for the output
    # Actually: C[k*n+i] = sum_r W[k*n+i, r] * (sum_j U[i*m+j,r] * V[j*p+k,r])... 
    # No, let me re-read the convention.
    
    # The standard tensor decomposition for matrix mult:
    # (A*B)[k,i] = sum_j A[i,j] * B[j,k]
    # = sum_r W[k*n+i, r] * sum_j U[i*m+j, r] * V[j*p+k, r]
    # This doesn't look right either.
    
    # Let me think again. The multiplication tensor M has entries:
    # M[i,j,k] = 1 meaning C[k,i] += A[i,j]*B[j,k]
    # In the trace convention W[k*n+i, r], the output index is k*n+i.
    
    # The decomposition satisfies:
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # For all other (a,b,c) combinations that don't correspond to a valid (i,j,k): = 0
    
    # Actually the standard way: the tensor T is defined on indices (a, b, c)
    # where a indexes A's entries (i*m+j), b indexes B's entries (j*p+k), 
    # c indexes C's entries (k*n+i).
    # T[a,b,c] = 1 iff there exist (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i
    # The decomposition is: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Schoolbook: U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for c=(i,j,k)
    # This gives rank 60.
    
    # Let me try to find if a lower rank exists using SVD on flattenings
    # and then verify with exact arithmetic.
    
    # Build tensor as 3D array
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Check flattenings for rank lower bounds
    # Flattening 1: (nm) x (mp*pn) - matrix of size 10 x 180
    F1 = T.reshape(nm, mp*pn)
    r1 = np.linalg.matrix_rank(F1, tol=1e-10)
    
    # Flattening 2: (mp) x (nm*pn) - matrix of size 30 x 120  
    F2 = T.reshape(mp, nm*pn)
    r2 = np.linalg.matrix_rank(F2, tol=1e-10)
    
    # Flattening 3: (pn) x (nm*mp) - matrix of size 12 x 300
    F3 = T.reshape(pn, nm*mp)
    r3 = np.linalg.matrix_rank(F3, tol=1e-10)
    
    lower_bound = max(r1, r2, r3)
    
    # Try to find a decomposition with rank = lower_bound
    # Use alternating least squares or direct approach
    
    # For small ranks, we can try to solve the system directly
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # This is a system of nm*mp*pn equations in R*(nm+mp+pn) unknowns
    # But it's nonlinear. Let's try a different approach.
    
    # Try rank from lower_bound up to 60
    best_rank = n * m * p
    best_U = np.zeros((nm, best_rank))
    best_V = np.zeros((mp, best_rank))
    best_W = np.zeros((pn, best_rank))
    
    # Schoolbook as fallback
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j, idx] = 1
                best_V[j*p+k, idx] = 1
                best_W[k*n+i, idx] = 1
                idx += 1
    
    # Try to find lower rank using ALS
    for target_rank in range(lower_bound, min(lower_bound + 20, best_rank)):
        if time.time() > deadline - 30:
            break
            
        # Try multiple random initializations
        found = False
        for trial in range(50):
            if time.time() > deadline - 20:
                break
                
            # Random initialization
            U = np.random.randn(nm, target_rank) * 0.1
            V = np.random.randn(mp, target_rank) * 0.1
            W = np.random.randn(pn, target_rank) * 0.1
            
            # Alternating least squares
            converged = False
            for it in range(200):
                # Update U: for each a, solve min sum_{b,c} (T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r])^2
                # This is: U[a,:] * (V ⊙ W)^T = T[a,:,:].reshape(-1)
                # where V ⊙ W is the Khatri-Rao product
                
                # Build Khatri-Rao product of V and W: (mp*pn, rank)
                VW = np.zeros((mp*pn, target_rank))
                for b in range(mp):
                    for c in range(pn):
                        VW[b*pn+c, :] = V[b, :] * W[c, :]
                
                # Solve for U
                rhs_U = T.reshape(nm, -1)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(VW, rhs_U.T, rcond=None)
                    U = U_new.T
                except:
                    break
                
                # Update V
                UW = np.zeros((nm*pn, target_rank))
                for a in range(nm):
                    for c in range(pn):
                        UW[a*pn+c, :] = U[a, :] * W[c, :]
                rhs_V = T.reshape(mp, nm*pn)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(UW, rhs_V.T, rcond=None)
                    V = V_new.T
                except:
                    break
                
                # Update W
                UV = np.zeros((nm*mp, target_rank))
                for a in range(nm):
                    for b in range(mp):
                        UV[a*mp+b, :] = U[a, :] * V[b, :]
                rhs_W = T.reshape(pn, nm*mp)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(UV, rhs_W.T, rcond=None)
                    W = W_new.T
                except:
                    break
                
                # Check convergence
                recon = np.zeros((nm, mp, pn))
                for r in range(target_rank):
                    recon += np.outer(np.outer(U[:,r], V[:,r]), W[:,r])
                err = np.linalg.norm(recon - T)
                if err < 1e-10:
                    converged = True
                    break
            
            if converged:
                # Verify exactly
                recon = np.zeros((nm, mp, pn))
                for r in range(target_rank):
                    recon += np.outer(np.outer(U[:,r], V[:,r]), W[:,r])
                if np.max(np.abs(recon - T)) < 1e-10:
                    best_rank = target_rank
                    best_U = U
                    best_V = V
                    best_W = W
                    found = True
                    break
        
        if found:
            break
    
    # Convert to lists with Fraction for exactness
    # Check if entries are close to integers or half-integers
    def to_fraction_list(arr):
        result = []
        for row in arr:
            r = []
            for val in row:
                # Try to find nearest half-integer
                frac = Fraction(val).limit_denominator(100)
                r.append(frac)
            result.append(r)
        return result
    
    U_list = to_fraction_list(best_U)
    V_list = to_fraction_list(best_V)
    W_list = to_fraction_list(best_W)
    
    return {
        "U": U_list,
        "V": V_list,
        "W": W_list
    }
