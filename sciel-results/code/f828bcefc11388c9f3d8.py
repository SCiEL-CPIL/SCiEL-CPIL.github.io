import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 145
    
    # Schoolbook decomposition
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    def verify(U, V, W, tol=1e-9):
        rank_val = U.shape[1]
        # Use einsum for speed
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need T[i,j,k] == 1 for all i,j,k
        
        # Reshape to 3D tensors
        # U: (n, m, R)
        # V: (m, p, R)
        # W: (p, n, R)
        U_3d = U.reshape(n, m, rank_val)
        V_3d = V.reshape(m, p, rank_val)
        W_3d = W.reshape(p, n, rank_val)
        
        # Compute T
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Note: W index is (k, i, r) in 3d form?
        # W is (pn, R) -> reshape to (p, n, R).
        # So W_3d[k, i, r] is the element.
        
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        return np.allclose(T, 1.0, atol=tol)

    def try_find_decomposition(rank_target, max_trials=100):
        """Try to find a decomposition of rank_target using random U, V and solving for W."""
        for _ in range(max_trials):
            if time.time() > deadline - 10:
                return None
            
            # Generate random U, V with entries in {-1, 0, 1}
            U_rand = np.random.choice([-1, 0, 1], size=(nm, rank_target))
            V_rand = np.random.choice([-1, 0, 1], size=(mp, rank_target))
            
            # Construct linear system A x = 1
            # x is vector of W entries, length rank_target * pn
            # A is (nmp) x (rank_target * pn)
            
            # We can build A column by column or use sparse?
            # For small dimensions, dense is fine.
            num_eqs = n * m * p
            num_vars = rank_target * pn
            A = np.zeros((num_eqs, num_vars))
            
            # Fill A
            # A[(i,j,k), (r,k',i')] = U[i,j,r] * V[j,k,r] if k==k', i==i'
            
            # To speed up, we can iterate over r, k, i
            for r in range(rank_target):
                for k in range(p):
                    for i in range(n):
                        # Variable index
                        var_idx = r * pn + k * n + i
                        
                        # This variable x_{r,k,i} appears in equations for all j
                        # Coefficient for equation (i,j,k) is U[i,j,r] * V[j,k,r]
                        # Equation index for (i,j,k) is i * m * p + j * p + k ?
                        # Let's define eq_idx = (i * m + j) * p + k ?
                        # No, the indices are i, j, k.
                        # Let's use a linear index for (i,j,k): i * m * p + j * p + k.
                        
                        for j in range(m):
                            eq_idx = i * m * p + j * p + k
                            coeff = U_rand[i * m + j, r] * V_rand[j * p + k, r]
                            A[eq_idx, var_idx] = coeff
            
            # Solve A x = 1
            b = np.ones(num_eqs)
            
            try:
                # Use least squares
                x, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            # Check if solution is close to half-integers
            # x contains W[k, i, r]
            # Reshape x to (R, p, n)
            W_sol = x.reshape(rank_target, p, n)
            
            # Check closeness to half-integers
            # W_sol * 2 should be close to integers
            W_scaled = W_sol * 2
            W_rounded = np.round(W_scaled)
            
            if np.allclose(W_scaled, W_rounded, atol=1e-6):
                # Potential success
                W_final = (W_rounded / 2).reshape(pn, rank_target)
                
                # Verify
                if verify(U_rand, V_rand, W_final, tol=1e-9):
                    return U_rand, V_rand, W_final, rank_target
            
        return None

    # Try to find lower ranks
    for r in range(best_rank - 1, 1, -1):
        if time.time() > deadline - 20:
            break
        
        result = try_find_decomposition(r, max_trials=50)
        if result:
            U_new, V_new, W_new, rank_new = result
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = rank_new
            # If we found a lower rank, we can try to go even lower
            # But the loop continues downwards anyway
    
    # Round to nearest half-integer (just in case, though we check in try_find)
    best_U = np.round(best_U * 2) / 2
    best_V = np.round(best_V * 2) / 2
    best_W = np.round(best_W * 2) / 2
    
    # Final verification
    if not verify(best_U, best_V, best_W, tol=1e-9):
        # Fallback to schoolbook
        best_U, best_V, best_W = U, V, W
        best_rank = rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
