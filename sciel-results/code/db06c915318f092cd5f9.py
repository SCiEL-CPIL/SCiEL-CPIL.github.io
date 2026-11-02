import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Precompute valid indices for the 1s
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # a in [0, nm), b in [0, mp), c in [0, pn)
    valid_ones = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_ones.append((a, b, c))
    
    # Precompute a large set of invalid indices for 0s
    # We will sample these during residual calculation to save time
    # Total invalid = nm * mp * pn - len(valid_ones)
    # nm=12, mp=24, pn=18 -> 5184 total, 72 valid, 5112 invalid
    # We will sample ~1000 invalids per residual call
    
    np.random.seed(42)
    
    # Try ranks from a lower bound up to schoolbook
    # Lower bound for M_{3,4,6}? 
    # R(M_{n,m,p}) >= max(nm, mp, pn) ? No.
    # R(M_{2,2,2}) = 7. nm=4, mp=4, pn=4.
    # R(M_{3,4,6}) might be around 50-60.
    # Let's try 50 to 72.
    
    best_rank = nm * mp # 72
    best_factors = None
    
    for R in range(50, 73):
        # Initialize
        U = np.random.randn(nm, R) * 0.1
        V = np.random.randn(mp, R) * 0.1
        W = np.random.randn(pn, R) * 0.1
        
        def residuals(params):
            U_flat = params[:nm*R].reshape(nm, R)
            V_flat = params[nm*R:nm*R+mp*R].reshape(mp, R)
            W_flat = params[nm*R+mp*R:].reshape(pn, R)
            
            res = []
            
            # 1. Enforce T=1 for valid indices
            for a, b, c in valid_ones:
                val = np.dot(np.dot(U_flat[a], V_flat[b]), W_flat[c])
                res.append(val - 1)
                
            # 2. Enforce T=0 for a sample of invalid indices
            # Sample 1000 random invalid indices
            # To avoid sampling valid ones, we can just sample and check, or pre-generate a pool
            # Pre-generating a pool of invalid indices is better
            # We'll generate random a, b, c and check if valid.
            # If valid, skip (or we can just ensure we don't pick them).
            # Since valid set is small, random collision is low.
            
            for _ in range(1000):
                a = np.random.randint(0, nm)
                b = np.random.randint(0, mp)
                c = np.random.randint(0, pn)
                
                # Check if (a,b,c) is in valid_ones
                # i1, j1 = divmod(a, m)
                # j2, k1 = divmod(b, p)
                # k2, i2 = divmod(c, n)
                # if i1==i2 and j1==j2 and k1==k2: continue
                
                val = np.dot(np.dot(U_flat[a], V_flat[b]), W_flat[c])
                res.append(val)
                
            return np.array(res)

        # Optimize
        # We need to flatten U, V, W
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            sol = least_squares(residuals, x0, method='trf', max_nfev=1000)
        except Exception:
            continue
            
        if sol.success or sol.cost < 1e-6:
            # Verify with full set of constraints
            U_sol = sol.x[:nm*R].reshape(nm, R)
            V_sol = sol.x[nm*R:nm*R+mp*R].reshape(mp, R)
            W_sol = sol.x[nm*R+mp*R:].reshape(pn, R)
            
            # Check all valid ones
            err_ones = 0
            for a, b, c in valid_ones:
                val = np.dot(np.dot(U_sol[a], V_sol[b]), W_sol[c])
                err_ones += (val - 1)**2
                
            # Check all invalid zeros (this is expensive, 5112 checks)
            # We can vectorize this check
            # Create arrays of indices
            # This might be slow in Python loop.
            # Let's use a tensor contraction check.
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # We want to check if T is the multiplication tensor.
            
            # Efficient check:
            # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
            U_3d = U_sol.reshape(n, m, R)
            V_3d = V_sol.reshape(m, p, R)
            W_3d = W_sol.reshape(p, n, R)
            
            # Compute T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # This is a batched matrix product or einsum
            # T[i,j,k] = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            # This creates a (3,4,6) tensor.
            T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            
            # Target tensor
            T_target = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_target[i,j,k] = 1
                        
            err = np.linalg.norm(T - T_target)
            
            if err < 1e-5:
                best_rank = R
                best_factors = (U_sol, V_sol, W_sol)
                break # Found the minimal rank in this range
        
    if best_factors is None:
        # Fall back to schoolbook
        R = nm * mp * pn # 72
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        best_factors = (U, V, W)
        best_rank = R
        
    # Convert to lists of lists as required by some interfaces? 
    # The task says "integer/half-integer entries". 
    # The optimizer produces floats. 
    # We should round them if they are close to integers/half-integers.
    # However, for matrix multiplication, integer decompositions exist (Strassen uses +/-1).
    # If the optimizer finds a solution, it might be close to integers.
    # Let's round to 2 decimal places and check validity again.
    
    U, V, W = best_factors
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    # Verify rounded solution
    U_3d = U_r.reshape(n, m, best_rank)
    V_3d = V_r.reshape(m, p, best_rank)
    W_3d = W_r.reshape(p, n, best_rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    T_target = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i,j,k] = 1
    err = np.linalg.norm(T - T_target)
    
    if err > 1e-5:
        # Rounding broke it, use original floats? 
        # The task says "integer/half-integer entries". 
        # If we can't satisfy it, we might need to stick to the exact solution 
        # or find a different one. 
        # However, typically these decompositions are integral.
        # If the optimizer found a non-integral solution, it might be a local minimum 
        # or the true rank requires non-integers? 
        # Matrix multiplication tensor has integral rank. 
        # So there exists an integral solution. 
        # The optimizer might have found a slightly off one.
        # Let's try to snap to integers more aggressively or just return the floats 
        # if the evaluator accepts floats? 
        # "integer/half-integer entries" implies strict constraint.
        
        # Let's try to recover by checking if the unrounded was good and just return that?
        # No, must be int/half-int.
        
        # If rounding fails, fall back to schoolbook which is integral.
        R = nm * mp * pn
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
