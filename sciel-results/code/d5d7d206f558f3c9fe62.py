import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    
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
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Try to find lower rank via structured search
    # The tensor constraint: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Try various ranks from low to high
    # Start with a reasonable guess based on known results
    # For 3x3x3, rank 7 is known. For 3x3x7, let's try various ranks.
    
    # Use a randomized search with local refinement
    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U.shape[1]):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try structured decompositions
    # Idea: for each j, the slice over (i,k) is a rank-1 outer product A[:,j] * B[j,:]
    # If we could make U and V have a special structure...
    
    # Attempt: use the fact that the tensor is "separable" in j in a specific way
    # T[i,j,k] = 1 = sum over some structure
    
    # Let's try a direct approach: assume U depends only on i (not j), V only on k (not j), W only on j
    # Then: sum_c U[i,c] * V[k,c] * W[j,c] = 1 for all i,j,k
    # This means U*V^T * W^T (broadcast) = 1, which requires U[i,c]*V[k,c]*W[j,c] sums to 1
    # This is only possible if the sum factors, which is very restrictive.
    
    # Better: let's try rank-1 per (i,j) pair or similar
    
    # Practical approach: randomized search for small ranks
    for target_rank in [21, 22, 23, 24, 25, 27, 28, 30, 32, 35, 40, 45, 50, 55, 60]:
        if time.time() > deadline - 10:
            break
        
        found = False
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random values
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5
            
            # Try to solve via iterative refinement
            # This is a hard non-linear problem; use a simple gradient-free approach
            
            # Actually, let's try a different strategy: build up from schoolbook and try to merge terms
            
            # Alternative: use the structure that T[i,j,k] = delta-like
            # For a given j, T[i,j,k] = 1 for all i,k (since it's the matmul tensor, every (i,j,k) triple has coefficient 1)
            
            # So for each j, we need: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,k
            # This is a (n x p) matrix equation for each j.
            
            break  # This approach is too slow; let's try something else
        
        if found:
            break
    
    # Let's try a more principled approach:
    # The tensor T[i,j,k] = 1. Consider it as a 3-tensor of size 3x3x7.
    # Its rank is what we're looking for (with the W-indexing twist).
    
    # Note: W is indexed as W[k*n+i, c], so the third index of the tensor is (k,i) not just k.
    # The tensor is T[i, j, (k,i)] but that's not standard.
    
    # Let me re-examine: the constraint is
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  for all i in [0,n), j in [0,m), k in [0,p)
    
    # So the "tensor" has entries T[i, j, k, i] but really it's a 3-way tensor where:
    # First index: i*m+j (9 values)
    # Second index: j*p+k (21 values)  
    # Third index: k*n+i (21 values)
    # And T[first, second, third] = 1 if they're consistent (i.e., same i,j,k), else... 
    # Actually no: for ANY i,j,k, the equation must hold. The tensor entry is always 1.
    
    # So we have a 9x21x21 tensor where every entry is 1? No!
    # U[i*m+j, c] is nonzero only for specific i,j. V[j*p+k,c] for specific j,k. W[k*n+i,c] for specific k,i.
    # The product U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] is the c-th term in the sum for (i,j,k).
    
    # The constraint is that this sum equals 1 for each (i,j,k). There are 3*3*7 = 63 constraints.
    
    # So we need 51r variables (9r + 21r + 21r) to satisfy 63 polynomial equations of degree 3.
    
    # For r=1: 51 vars, 63 eqs. Likely no solution.
    # For r=2: 102 vars, 63 eqs. Possible but hard.
    # ...
    
    # Let me try a constructive approach based on known matmul algorithms.
    
    # For 2x2x2, Strassen: rank 7.
    # For 3x3x3, best known is 23 (Bini et al.), but 27 is trivially achievable by 3x3 blocks.
    # Actually, 3x3x3 can be done at rank 23, but that's complex.
    
    # Simpler: 3x3x3 at rank 27 (schoolbook) or 24 (using 2x2 Strassen + 1x1).
    
    # For 3x3x7: 
    # Option 1: 3x3x7 = 3x3x(3+3+1). Do two 3x3x3 at rank 24 each + one 3x3x1 at rank 9. Total 57.
    # Option 2: Better sharing.
    
    # Let's try: 3x3x7 as 3x(3x7). Each 3x7 block... 
    # A 3x3 matrix times a 3x7 matrix. Think of it as 7 column multiplications: A * b_k for k=0..6.
    # Each A*b_k is a 3x1 vector. The tensor for A*b is 3x3x1 = 9 terms.
    # But we can share computation across k.
    
    # The key: (A*b_k)[i] = sum_j A[i,j] * b_k[j]
    # For all i=0,1,2 and k=0..6: sum_c U[i*3+j, c] * V[j*7+k, c] * W[k*3+i, c] = 1
    
    # Let me try a rank-21 decomposition: one term per (j,k) pair, with U depending on i.
    # U[i*3+j, c] where c = j*7+k... no, c ranges over 21.
    # Let c = j*7+k for j=0..2, k=0..6.
    # U[i*3+j, c] = delta[i, i] = 1 if we set U[i*3+j, j*7+k] = 1 for all i? No, that doesn't work.
    
    # Try: U[i*3+j, c] = 1 for all i, when c corresponds to (j,k).
    # V[j*7+k, c] = 1 when c = j*7+k, else 0.
    # W[k*3+i, c] = 1 when c = j*7+k? But W doesn't know j.
    
    # This doesn't factor cleanly.
    
    # Let me just try a numerical search with a time limit.
    
    # Use a simple approach: for a given rank r, set up the equations and try to solve.
    # Since it's non-linear, use random restarts with local optimization.
    
    from scipy.optimize import minimize
    
    def solve_for_rank(r, max_time=20):
        t0 = time.time()
        best_obj = float('inf')
        best_x = None
        
        for restart in range(50):
            if time.time() - t0 > max_time:
                break
            
            # Random initialization
            x0 = np.random.randn(nm*r + mp*r + pn*r) * 0.3
            
            def objective(x):
                U = x[:nm*r].reshape(nm, r)
                V = x[nm*r:nm*r+mp*r].reshape(mp, r)
                W = x[nm*r+mp*r:].reshape(pn, r)
                
                # Compute residual
                res = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            res += (val - 1.0)**2
                return res
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10})
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_x = res.x
            except:
                pass
        
        if best_x is None:
            return None
        
        if best_obj > 1e-6:
            return None
        
        U = best_x[:nm*r].reshape(nm, r)
        V = best_x[nm*r:nm*r+mp*r].reshape(mp, r)
        W = best_x[nm*r+mp*r:].reshape(pn, r)
        
        # Round to half-integers if close
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        
        # Check if rounded version works
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_round[i*m+j, :], V_round[j*p+k, :] * W_round[k*n+i, :])
                    if abs(val - 1.0) > 1e-6:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            return U_round, V_round, W_round, r
        else:
            # Use unrounded if it's close enough
            if best_obj < 1e-10:
                return U, V, W, r
            return None
    
    # Try ranks from low to high
    for r in [21, 22, 23, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 50, 54, 56, 60, 63]:
        if time.time() > deadline - 15:
            break
        
        result = solve_for_rank(r, max_time=min(20, deadline - time.time() - 10))
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                best_U, best_V, best_W, best_rank = U, V, W, rank
            break
    
    # Convert to lists
    return {
        "U": best_U.tolist() if hasattr(best_U, 'tolist') else best_U,
        "V": best_V.tolist() if hasattr(best_V, 'tolist') else best_V,
        "W": best_W.tolist() if hasattr(best_W, 'tolist') else best_W
    }
