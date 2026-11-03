import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    def compute_loss(params, r):
        size_U = n * m * r
        size_V = m * p * r
        size_W = p * n * r
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        U3 = U_flat.reshape(n, m, r)
        V3 = V_flat.reshape(m, p, r)
        W3 = W_flat.reshape(p, n, r)
        
        # S_{ij, j'k, ki} = sum_r U_{ij,r} V_{j'k,r} W_{ki,r}
        # We want S_{ij, j'k, ki} = delta_{j, j'}
        
        # Efficient computation:
        # For each r, we have U_r (n,m), V_r (m,p), W_r (p,n)
        # Contribution to S is U_r[i,j] * V_r[j',k] * W_r[k,i]
        # Sum over r.
        
        S = np.zeros((n, m, m, p, n)) # i, j, j', k, i
        
        for r_idx in range(r):
            Ur = U3[:, :, r_idx] # n, m
            Vr = V3[:, :, r_idx] # m, p
            Wr = W3[:, :, r_idx] # p, n
            
            # Ur[i,j] * Vr[j',k] * Wr[k,i]
            # Use einsum
            # 'ij,jk,ki->ijjki'
            contrib = np.einsum('ij,jk,ki->ijjki', Ur, Vr, Wr)
            S += contrib
            
        # Target: delta_{j, j'}
        # Target tensor T[i, j, j', k, i] = 1 if j==j' else 0
        # Note: i and k are free indices in the target (they just pass through)
        # So T[i, j, j', k, i] = delta_{j, j'} for all i, k.
        
        # Create target
        J = np.eye(m) # m x m
        # Target shape (n, m, m, p, n)
        # T[i, j, j', k, i] = J[j, j']
        # We can broadcast J to (1, m, m, 1, 1) and multiply by ones for n, p, n?
        # No, the value is independent of i and k.
        target = np.ones((n, m, m, p, n)) * J[np.newaxis, :, :, np.newaxis, np.newaxis]
        
        diff = S - target
        return np.sum(diff * diff)

    def verify(U_list, V_list, W_list):
        n, m, p = DIMENSIONS
        r = len(U_list[0])
        for i in range(n):
            for j in range(m):
                for j_prime in range(m):
                    for k in range(p):
                        s = 0.0
                        for r_idx in range(r):
                            s += U_list[i*m+j][r_idx] * V_list[j_prime*p+k][r_idx] * W_list[k*n+i][r_idx]
                        target = 1.0 if j == j_prime else 0.0
                        if abs(s - target) > 1e-9:
                            return False
        return True

    def try_rank(r, max_restarts=10):
        if time.time() > deadline - 10:
            return None
            
        size_U = n * m * r
        size_V = m * p * r
        size_W = p * n * r
        total_size = size_U + size_V + size_W
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize
            # Random small values
            init_vals = np.random.normal(0, 0.5, size=total_size)
            
            try:
                result = minimize(compute_loss, init_vals, args=(r,), method='L-BFGS-B', 
                                  options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
            except Exception:
                continue
            
            if result.fun < 1e-6:
                params = result.x
                U_flat = np.round(params[:size_U] * 2) / 2
                V_flat = np.round(params[size_U:size_U + size_V] * 2) / 2
                W_flat = np.round(params[size_U + size_V:] * 2) / 2
                
                U3 = U_flat.reshape(n, m, r)
                V3 = V_flat.reshape(m, p, r)
                W3 = W_flat.reshape(p, n, r)
                
                # Check exactness
                S = np.zeros((n, m, m, p, n))
                for r_idx in range(r):
                    S += np.einsum('ij,jk,ki->ijjki', U3[:, :, r_idx], V3[:, :, r_idx], W3[:, :, r_idx])
                
                J = np.eye(m)
                target = np.ones((n, m, m, p, n)) * J[np.newaxis, :, :, np.newaxis, np.newaxis]
                
                if np.allclose(S, target, atol=1e-9):
                    U_list = [[U_flat[(i*m+j)*r + rr] for rr in range(r)] for i in range(n) for j in range(m)]
                    V_list = [[V_flat[(j*p+k)*r + rr] for rr in range(r)] for j in range(m) for k in range(p)]
                    W_list = [[W_flat[(k*n+i)*r + rr] for rr in range(r)] for k in range(p) for i in range(n)]
                    
                    if verify(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}

    # Search for minimal rank
    # Hint suggests 21
    for r in range(21, 26):
        if time.time() > deadline - 10:
            break
        result = try_rank(r)
        if result:
            return result

    # Fallback to schoolbook if search fails
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
