"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    deadline = time.time() + 140  # safety margin
    
    # Target tensor T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # We need U (nm x r), V (mp x r), W (pn x r) such that
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Strategy: Use a structured decomposition
    # For matrix multiplication A(nxm) * B(mx p), we can use:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # 
    # One approach: exploit the separable structure
    # Let's try rank = n*p + m*p + n*m - (n+m+p-1) type constructions
    # But for exact integer decompositions, let's use a different approach.
    
    # Key insight: We can decompose using the fact that the tensor is
    # a "slice" of matrix multiplication. Let's try to find a low-rank
    # decomposition using randomized methods.
    
    # First, let's try a structured approach:
    # The tensor T has T[a,b,c] = 1 where a=i*m+j, b=j*p+k, c=k*n+i
    # This means T[a,b,c] = 1 iff there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
    # i.e., j = a%m, i = a//m, k = b%p, j = b//p, i = c%n, k = c//n
    # So T[a,b,c] = 1 iff a%m == b//p and a//m == c%n and b%p == c//n
    # Otherwise T[a,b,c] = 0
    
    # Let's build the tensor and try to find a low-rank decomposition
    nm, mp, pn = n*m, m*p, p*n  # 15, 25, 15
    
    # Build the target tensor as a 3D array
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try various ranks from low to high
    # Start with a reasonable guess based on known bounds
    # For 3x5x5 matmul, the rank should be significantly less than 75
    
    # Use a random initialization + iterative refinement approach
    def try_rank(r, max_iter=2000, seed=42):
        if time.time() > deadline - 10:
            return None
        rng = np.random.RandomState(seed)
        
        # Initialize U, V, W randomly
        U = rng.randn(nm, r)
        V = rng.randn(mp, r)
        W = rng.randn(pn, r)
        
        # Normalize to avoid numerical issues
        for c in range(r):
            norm = max(np.linalg.norm(U[:,c]), np.linalg.norm(V[:,c]), np.linalg.norm(W[:,c]), 1e-10)
            U[:,c] /= norm
            V[:,c] /= norm
            W[:,c] /= norm
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        lr = 0.01
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Compute current tensor estimate
            # T_est[a,b,c] = sum_c U[a,c]*V[b,c]*W[c,c]
            # This is expensive for large r, so let's be smart
            
            # Compute gradient w.r.t. U[a,c]:
            # dL/da = 2 * (T_est - T)[a,*,*] * (V[:,c] * W[:,c])
            # Actually: L = sum_{a,b,c'} (sum_r U[a,r]*V[b,r]*W[c',r] - T[a,b,c'])^2
            # dL/da = 2 * sum_{b,c'} (T_est[a,b,c'] - T[a,b,c']) * V[b,c] * W[c',c]
            
            # For efficiency, compute using matrix operations
            # T_est[a,b,:] = U[a,:] @ (V[b,:][:,None] * W.T) ... no
            # T_est[a,b,c'] = sum_r U[a,r] * V[b,r] * W[c',r]
            
            # Let's compute the residual R = T_est - T
            # R[a,b,c'] = sum_r U[a,r]*V[b,r]*W[c',r] - T[a,b,c']
            
            # Gradient for U:
            # dL/da = 2 * sum_{b,c'} R[a,b,c'] * V[b,r] * W[c',r]
            # = 2 * (R @ (V[:,r]*W[:,r]))[a] ... not quite
            # = 2 * sum_b sum_c' R[a,b,c'] * V[b,r] * W[c',r]
            # Let M_r = outer(V[:,r], W[:,r])  (mp x pn matrix)
            # Then (R contracted with M_r over b,c') gives a vector of length nm
            # grad_U[:,r] = 2 * sum_b,c' R[a,b,c'] * M_r[b,c']
            
            # This is still expensive. Let's use a different approach.
            # Instead, let's use the fact that we can compute:
            # For each r, define P_r = V[:,r] (mp x 1), Q_r = W[:,r] (pn x 1)
            # Then R[a,b,c'] * P_r[b] * Q_r[c'] summed over b,c'
            # = (R * outer(P_r, Q_r)) summed over b,c' for each a
            # = sum_a' ... this is getting complex.
            
            # Alternative: Use a simpler iterative method - coordinate descent
            # or just use a smaller search space.
            
            # Let's try a different approach: use the structure of the problem.
            # The tensor T has a special form. Let's exploit it.
            
            # Actually, let me try a completely different strategy:
            # Use a parametric family of decompositions and optimize.
            
            # For now, let's just check if current U,V,W is good enough
            # and return if we find something valid.
            if it % 50 == 0:
                # Compute max error
                # T_est = np.einsum('ar,br,cr->abc', U, V, W)
                # This is too expensive for large r. Let's sample.
                # Instead, check a subset of entries
                errors = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a, b, c = i*m+j, j*p+k, k*n+i
                            val = np.dot(U[a,:], V[b,:]*W[c,:])
                            errors.append(abs(val - 1.0))
                max_err = max(errors) if errors else float('inf')
                
                if max_err < 1e-6:
                    # Check some off-diagonal entries should be 0
                    off_errors = []
                    for _ in range(100):
                        a = rng.randint(nm)
                        b = rng.randint(mp)
                        c = rng.randint(pn)
                        # Check if this is a "valid" (i,j,k) triple
                        i1, j1 = a // m, a % m
                        j2, k2 = b // p, b % p
                        k3, i3 = c // n, c % n
                        if j1 == j2 and i1 == i3 and k2 == k3:
                            continue  # This is a valid entry, skip
                        val = np.dot(U[a,:], V[b,:]*W[c,:])
                        off_errors.append(abs(val))
                    max_off_err = max(off_errors) if off_errors else float('inf')
                    
                    if max_off_err < 1e-6:
                        return U, V, W, max_err, max_off_err
                elif max_err < best_loss:
                    best_loss = max_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Update using a simple gradient step (approximate)
            # For efficiency, update one column at a time
            r_idx = it % r
            # Compute contribution of column r_idx
            # U[:,r_idx] * (V[:,r_idx] * W[:,r_idx]) outer product
            col_contrib = np.einsum('a,b,c->abc', U[:,r_idx], V[:,r_idx], W[:,r_idx])
            residual = col_contrib - T  # This is the residual from just this column
            
            # Gradient for U[:,r_idx]:
            # dL/da = 2 * sum_{b,c'} residual[a,b,c'] * V[b,r_idx] * W[c',r_idx]
            grad_U = 2 * np.einsum('abc,bc->a', residual, V[:,r_idx][:,None]*W[:,r_idx][None,:])
            
            # Update
            U[:,r_idx] -= lr * grad_U
            lr = max(lr * 0.999, 0.001)
        
        return None
    
    # Try increasing ranks
    for r in [30, 35, 40, 45, 50, 55, 60, 65, 70]:
        if time.time() > deadline - 15:
            break
        
        # Try multiple seeds
        for seed in [42, 123, 456, 789]:
            if time.time() > deadline - 15:
                break
            
            result = try_rank(r, max_iter=1000, seed=seed)
            if result is not None:
                U, V, W, err1, err2 = result
                # Convert to fractions
                U_frac = [[Fraction(U[i,c]).limit_denominator(100) for c in range(r)] for i in range(nm)]
                V_frac = [[Fraction(V[i,c]).limit_denominator(100) for c in range(r)] for i in range(mp)]
                W_frac = [[Fraction(W[i,c]).limit_denominator(100) for c in range(r)] for i in range(pn)]
                
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = sum(U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c] for c in range(r))
                            if s != 1:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    return {"U": U_frac, "V": V_frac, "W": W_frac}
        
        # If random didn't work, try a structured approach
        if time.time() > deadline - 15:
            break
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
