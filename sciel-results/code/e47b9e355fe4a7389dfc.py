import numpy as np
import time
from itertools import product, combinations
from scipy.linalg import lstsq

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 10
    dim_v = m * p  # 25
    dim_w = p * n  # 10
    
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # Build target tensor
    target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0
    
    def verify_solution(U, V, W):
        """Verify exact decomposition with half-integer entries."""
        # Check entries are half-integers
        for M in [U, V, W]:
            if not np.allclose(M * 2, np.round(M * 2), atol=1e-9):
                return False
        T = np.einsum('ar,br,cr->abc', U, V, W)
        max_diff = np.max(np.abs(T - target))
        return max_diff < 1e-9
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strategy: structured search exploiting n=2
    # For each rank r, try to find sparse integer/half-integer solutions
    # Key: fix U and V patterns, solve for W linearly
    
    def try_rank_structured(r, max_time=30.0):
        """Try to find a rank-r decomposition using structured search."""
        start = time.time()
        
        # For rank r, we have r columns in U (10 x r), V (25 x r), W (10 x r)
        # The tensor equation: for each (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = target[a,b,c]
        
        # Strategy: enumerate sparse patterns for U and V, solve for W
        # Each column r has a "type" determined by which entries are nonzero
        
        # For matrix multiplication, natural structure:
        # U[i*m+j, r] is nonzero only for specific (i,j) pairs
        # V[j*p+k, r] is nonzero only for specific (j,k) pairs  
        # W[k*n+i, r] is nonzero only for specific (k,i) pairs
        
        # Try: each rank-1 component corresponds to a triple (i,j,k) or combinations
        # With r=25, we need to find 25 rank-1 tensors that sum to the target
        
        # Approach: use the linearity in W
        # Given U and V, for each c: sum_r U[a,r]*V[b,r]*W[c,r] = target[a,b,c]
        # This is a system: for fixed c, W[c,:] satisfies:
        # sum_r (U[:,r] * V[:,r])_elementwise * W[c,r] = target[:,c,:]
        
        # Actually, let me think differently.
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # For fixed c: sum_r W[c,r] * (U[:,r] ⊙ V[:,r])[a,b] = target[a,b,c]
        # where ⊙ is element-wise product of the column vectors
        
        # So for each c, we have a linear system in W[c,:]:
        # A_c * W[c,:] = target[:, :, c]
        # where A_c[a,b] = sum_r U[a,r] V[b,r] ... no wait
        
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For fixed c: sum_r (U[:,r] * V[:,r])_a_b * W[c,r] where (U[:,r]*V[:,r]) is outer product
        
        # Let M_r = U[:,r] * V[:,r]^T (outer product, shape dim_u x dim_v)
        # Then T[:,:,c] = sum_r W[c,r] * M_r
        
        # So for each c, we need: sum_r W[c,r] * M_r = target[:,:,c]
        # This is a linear system in W[c,:] with dim_u*dim_v equations and r unknowns
        
        # If r >= dim_u*dim_v = 250, it's overdetermined... wait no, r is the rank
        # We have r unknowns (W[c,0..r-1]) and 10*25=250 equations
        # This is overdetermined unless the target has low rank structure
        
        # Better approach: use the fact that target[:,:,c] has specific structure
        # target[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k'*n+i' where k'=c//n, i'=c%n
        # So target[:,:,c] is nonzero only at specific (a,b) positions
        
        # Let me try a direct approach: enumerate small integer patterns
        
        # For rank r, each column r of U has at most a few nonzero entries
        # For matrix multiplication, a "natural" rank-1 component might be:
        # U nonzero at one (i,j), V nonzero at one (j,k), W nonzero at one (k,i)
        # That gives the schoolbook with r = n*m*p = 50
        
        # To get r < 50, we need "combined" components where one rank-1 tensor
        # contributes to multiple (i,j,k) triples
        
        # Let's try: allow each U column to have up to 2 nonzeros, same for V, W
        # And search over small integer values {-1, 0, 1, 2}
        
        # Actually, let's use a smarter approach:
        # Fix the support pattern, solve for values
        
        # For now, try r from 25 to 49
        for r_try in range(r, min(r+10, 50)):
            if time.time() - start > max_time:
                return None
            if time.time() > deadline:
                return None
            
            # Try to find a solution with rank r_try
            # Use randomized initialization + local optimization
            # But with structure: initialize with schoolbook-like patterns
            
            best_loss = float('inf')
            best_params = None
            
            n_restarts = 5
            for restart in range(n_restarts):
                if time.time() > deadline:
                    break
                
                np.random.seed(42 + r_try * 100 + restart)
                
                # Initialize with structured patterns
                U = np.zeros((dim_u, r_try))
                V = np.zeros((dim_v, r_try))
                W = np.zeros((dim_w, r_try))
                
                # Each column: pick random (i,j) for U, (j,k) for V, (k,i) for W
                # with some probability of combining
                for col in range(r_try):
                    # Random assignment
                    i = np.random.randint(n)
                    j = np.random.randint(m)
                    k = np.random.randint(p)
                    
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    
                    # With some probability, add another component
                    if np.random.rand() < 0.3:
                        i2 = np.random.randint(n)
                        j2 = np.random.randint(m)
                        k2 = np.random.randint(p)
                        U[i2*m+j2, col] += np.random.choice([0.5, 1.0])
                        V[j2*p+k2, col] += np.random.choice([0.5, 1.0])
                        W[k2*n+i2, col] += np.random.choice([0.5, 1.0])
                
                # Optimize
                params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
                def loss_fn(params):
                    U_ = params[:dim_u*r_try].reshape(dim_u, r_try)
                    V_ = params[dim_u*r_try:dim_u*r_try+dim_v*r_try].reshape(dim_v, r_try)
                    W_ = params[dim_u*r_try+dim_v*r_try:].reshape(dim_w, r_try)
                    T = np.einsum('ar,br,cr->abc', U_, V_, W_)
                    return np.sum((T - target)**2)
                
                # Use scipy minimize
                from scipy.optimize import minimize
                try:
                    res = minimize(loss_fn, params, method='L-BFGS-B', 
                                   options={'maxiter': 200, 'ftol': 1e-12})
                    if res.fun < best_loss:
                        best_loss = res.fun
                        best_params = res.x
                        if res.fun < 1e-10:
                            # Found exact solution
                            U_f = best_params[:dim_u*r_try].reshape(dim_u, r_try)
                            V_f = best_params[dim_u*r_try:dim_u*r_try+dim_v*r_try].reshape(dim_v, r_try)
                            W_f = best_params[dim_u*r_try+dim_v*r_try:].reshape(dim_w, r_try)
                            # Snap to half-integers
                            U_s = np.round(U_f * 2) / 2
                            V_s = np.round(V_f * 2) / 2
                            W_s = np.round(W_f * 2) / 2
                            if verify_solution(U_s, V_s, W_s):
                                return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                except Exception:
                    pass
            
            if best_params is not None and best_loss < 1e-8:
                U_f = best_params[:dim_u*r_try].reshape(dim_u, r_try)
                V_f = best_params[dim_u*r_try:dim_u*r_try+dim_v*r_try].reshape(dim_v, r_try)
                W_f = best_params[dim_u*r_try+dim_v*r_try:].reshape(dim_w, r_try)
                U_s = np.round(U_f * 2) / 2
                V_s = np.round(V_f * 2) / 2
                W_s = np.round(W_f * 2) / 2
                if verify_solution(U_s, V_s, W_s):
                    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
        
        return None
    
    # Try structured search for various ranks
    for target_rank in [25, 30, 35, 40, 45]:
        if time.time() > deadline:
            break
        result = try_rank_structured(target_rank, max_time=25.0)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return get_schoolbook()
# EVOLVE-BLOCK-END
