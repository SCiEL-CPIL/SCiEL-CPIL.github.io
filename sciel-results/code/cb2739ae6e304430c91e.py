"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 32
    pn = p * n  # 16
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Try to find lower rank via linear system approach
    # For given U (nm x r) and V (mp x r), solve for W (pn x r)
    # The constraint: for all (i,j,j',k',i',k''), sum_c U[i*m+j,c]*V[j'*p+k',c]*W[k'*n+i',c] = delta
    
    # Build the linear system for W
    # Variables: W[k*n+i, c] for k in [0,p), i in [0,n), c in [0,r)
    # Total variables: p*n*r
    
    # Equations: for each (i,j,j',k',i',k'') with the delta constraint
    # But many (i,j,j',k',i',k'') give zero on RHS
    
    # Actually, let's think of it differently.
    # The tensor T[i,j,j',k',i',k''] = delta_{j,j'}*delta_{i,i'}*delta_{k',k''}
    # We need: sum_c U[i*m+j,c] * V[j'*p+k',c] * W[k'*n+i',c] = T[i,j,j',k',i',k'']
    
    # This is equivalent to: the tensor T must be in the span of the "rank-1" tensors
    # formed by outer products of rows of U, V, W (with appropriate index mapping)
    
    # Let's try a different approach: use the fact that for fixed (i,j,k'), the constraint
    # on W is linear. Group equations by (i,j,k').
    
    # For fixed (i,j,k'), the equation is:
    # sum_c U[i*m+j,c] * (sum_{j'} V[j'*p+k',c] * delta_{j,j'}) * W[k'*n+i',c] 
    # Wait, I'm confusing myself.
    
    # Let me restart the formulation.
    # The bilinear multiplication tensor for C=AB is:
    # T[i,j,k] = 1 (coefficient of A[i,j]*B[j,k] in C[i,k])
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, that's not right either. The W index should match.
    
    # From the code: W[k*n+i, c]. So W is indexed by (k,i).
    # The term for column c is: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal T[i,j,k] = 1 for all i,j,k.
    
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # This IS the all-ones tensor decomposition! But with the constraint that the indices
    # are coupled: U depends on (i,j), V on (j,k), W on (k,i).
    
    # So the problem is: find min r such that there exist U (8xr), V (32xr), W (16xr)
    # with sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i in [0,2), j in [0,4), k in [0,8)
    
    # This is a tensor decomposition problem with coupled indices.
    
    # For the all-ones tensor with free indices, rank 1 works. But here the indices are
    # coupled in a specific way, making it harder.
    
    # Let's try r=56 and see if we can find a solution.
    
    # Approach: random initialization + iterative refinement
    # Or: use the structure to construct a solution.
    
    # Idea: use the schoolbook but merge terms where possible.
    # In schoolbook, each (i,j,k) gets its own term. Can we share terms?
    
    # For term c, it contributes U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] to T[i,j,k].
    # For this to be 1 for all (i,j,k), we need the sum over c to be 1.
    
    # If we use r terms, each term is a "slice" of the tensor.
    
    # Let me try a constructive approach:
    # For each i, use m*p terms that cover all (j,k) for that i.
    # But that's still n*m*p = 64.
    
    # To reduce, we need to share terms across different i values.
    
    # Suppose we use terms that are non-zero for multiple i values.
    # For a term c, let S_i = {j : U[i*m+j,c] != 0} for each i.
    # Similarly for V and W.
    
    # This is getting complex. Let me try a numerical approach.
    
    # For a given r, set up the problem as: find U,V,W such that
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # This is a system of n*m*p = 64 polynomial equations in nm*r + mp*r + pn*r variables.
    
    # For r=56: 8*56 + 32*56 + 16*56 = 448 + 1792 + 896 = 3136 variables, 64 equations.
    # Underdetermined, so solutions should exist if r is large enough.
    
    # Let's try r=56 with random initialization and check if we can satisfy the constraints.
    
    # Actually, the constraints are not linear in U,V,W jointly, but if we fix two, they're linear in the third.
    
    # Strategy: 
    # 1. Initialize U, V randomly
    # 2. For each (i,j,k), the constraint is linear in W[k*n+i, :]
    #    But W[k*n+i, c] appears in constraints for multiple (i,j,k) with the same (k,i).
    
    # Let's group by (k,i). For fixed (k,i), W[k*n+i, c] appears in constraints for all j.
    # For fixed (k,i,j), the constraint is:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # This is linear in W[k*n+i, :]. Let a_{j,c} = U[i*m+j,c] * V[j*p+k,c]
    # Then: sum_c a_{j,c} * W[k*n+i,c] = 1 for each j.
    
    # So for fixed (i,k), we have a linear system: A_{i,k} * W_{i,k} = 1
    # where A_{i,k} is m x r with A_{i,k}[j,c] = U[i*m+j,c] * V[j*p+k,c]
    # and W_{i,k} is W[k*n+i, :] (r-vector)
    
    # We need A_{i,k} * W_{i,k} = 1_m (m-dimensional vector of ones) for all i,k.
    
    # This is solvable if 1_m is in the column space of A_{i,k}.
    
    # So the algorithm:
    # 1. Choose r
    # 2. Randomly initialize U (nm x r) and V (mp x r)
    # 3. For each (i,k), check if A_{i,k} has 1_m in its column space
    # 4. If yes for all (i,k), solve for W and verify the full decomposition
    # 5. If not, try again
    
    # Let's implement this.
    
    def try_rank(r, max_attempts=1000):
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            
            W = np.zeros((pn, r))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # A_{i,k}[j,c] = U[i*m+j, c] * V[j*p+k, c]
                    A_ik = np.zeros((m, r))
                    for j in range(m):
                        A_ik[j, :] = U[i*m + j, :] * V[j*p + k, :]
                    
                    # Solve A_ik * w = 1_m
                    # Use least squares
                    try:
                        w, residuals, rank_A, sv = np.linalg.lstsq(A_ik, np.ones(m), rcond=None)
                        # Check if the solution is exact
                        if np.allclose(A_ik @ w, np.ones(m), atol=1e-10):
                            W[k*n + i, :] = w
                        else:
                            valid = False
                            break
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                # Verify the full decomposition
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                            if abs(val - 1) > 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    return U, V, W, r
        
        return None
    
    # Try ranks from 56 down to some minimum
    # The theoretical minimum is at least max(n,m,p) = 8, but likely higher.
    # Try 56 first since we know it's achievable.
    
    for r in [56, 55, 54, 53, 52, 51, 50]:
        if time.time() > deadline:
            break
        
        result = try_rank(r)
        if result is not None:
            U, V, W, rank = result
            # Convert to lists
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback to schoolbook
    U, V, W, rank = schoolbook()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
