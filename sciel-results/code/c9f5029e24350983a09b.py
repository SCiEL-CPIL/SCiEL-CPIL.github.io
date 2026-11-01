import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Schoolbook baseline
    R_sb = n * m * p
    U_sb = np.zeros((n * m, R_sb))
    V_sb = np.zeros((m * p, R_sb))
    W_sb = np.zeros((p * n, R_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    def verify(U, V, W, R, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def snap_to_half_integer(mat, tol=1e-6):
        snapped = np.round(mat * 2) / 2
        if np.max(np.abs(mat - snapped)) < tol:
            return snapped
        return mat

    def flatten(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

    def unflatten(x, R):
        u_size = n * m * R
        v_size = m * p * R
        U = x[:u_size].reshape(n * m, R)
        V = x[u_size:u_size + v_size].reshape(m * p, R)
        W = x[u_size + v_size:].reshape(p * n, R)
        return U, V, W

    # Build the tensor T[i,j,k] = 1 for all (i,j,k)
    # We need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = 1

    # Precompute index mappings for speed
    # For each r, we need to compute contributions to all (i,j,k)
    # Let's vectorize the residual computation

    def residuals_vectorized(x, R):
        U, V, W = unflatten(x, R)
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # For each r, outer product of U[:,r], V[:,r], W[:,r] gives a tensor
        # We sum over r
        
        # Reshape U, V, W to 3D for tensor operations
        # U[i*m+j, r] -> U3[i, j, r]
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        
        # For each r, we have U3[:,:,r] (n,m), V3[:,:,r] (m,p), W3[:,:,r] (p,n)
        # The contribution to (i,j,k) is U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        
        # Sum over r: sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # This is a trace-like operation. Let's compute it.
        
        # Create the result tensor C[i,j,k]
        C = np.zeros((n, m, p))
        for r in range(R):
            # U3[:,:,r] is (n,m), V3[:,:,r] is (m,p), W3[:,:,r] is (p,n)
            # We need C[i,j,k] += U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
            # This is like a tensor contraction
            # For fixed r, define A_r = U3[:,:,r] (n,m), B_r = V3[:,:,r] (m,p), C_r = W3[:,:,r] (p,n)
            # T_r[i,j,k] = A_r[i,j] * B_r[j,k] * C_r[k,i]
            # We can compute this as: for each i,j,k, multiply the three values.
            # Vectorized: 
            # T_r = A_r[:, :, None] * B_r[None, :, :] * C_r[None, None, :]
            # But C_r is (p,n), we need C_r[k,i] which is C_r.T[i,k]
            # So T_r[i,j,k] = A_r[i,j] * B_r[j,k] * C_r.T[i,k]
            # = A_r[:, :, None] * B_r[None, :, :] * C_r.T[None, :, :]
            # Wait, dimensions: A_r (n,m), B_r (m,p), C_r.T (n,p)
            # A_r[:, :, None] is (n,m,1)
            # B_r[None, :, :] is (1,m,p)
            # C_r.T[None, :, :] is (1,n,p) - this doesn't broadcast correctly with (n,m,p)
            
            # Let me think again.
            # T_r[i,j,k] = A_r[i,j] * B_r[j,k] * C_r[k,i]
            # Let's iterate over i,j,k or use einsum.
            # np.einsum('ij,jk,ki->ijk', A_r, B_r, C_r)
            # But A_r is (n,m), B_r is (m,p), C_r is (p,n)
            # einsum('ij,jk,ki->ijk', A_r, B_r, C_r) should work.
            C += np.einsum('ij,jk,ki->ijk', U3[:, :, r], V3[:, :, r], W3[:, :, r])
        
        # C should be all ones
        return (C - 1.0).flatten()

    # Try to find a low-rank decomposition
    # Start from a reasonable lower bound
    # The rank of matrix multiplication tensor for n x m x p is at least max(n*m, m*p, n*p) / something
    # For 2x3x7: n*m=6, m*p=21, n*p=14
    # Known lower bounds suggest rank >= 14 or so. Let's try from 14 up.
    
    best_result = None
    best_R = R_sb
    
    # Try ranks from 14 to R_sb - 1
    for R in range(14, min(R_sb, 30)):
        if time.time() > deadline - 10:
            break
        
        # Multiple random restarts
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            rng = np.random.default_rng(42 + R * 1000 + attempt * 7)
            
            # Initialize with random half-integers or small integers
            # Use a mix of -1, -0.5, 0, 0.5, 1
            vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            U = rng.choice(vals, size=(n * m, R))
            V = rng.choice(vals, size=(m * p, R))
            W = rng.choice(vals, size=(p * n, R))
            
            x0 = flatten(U, V, W)
            
            try:
                sol = least_squares(
                    lambda x: residuals_vectorized(x, R),
                    x0,
                    method='lm',
                    max_nfev=3000,
                    ftol=1e-12,
                    xtol=1e-12
                )
            except Exception:
                continue
            
            if sol.cost < 1e-8:
                U_sol, V_sol, W_sol = unflatten(sol.x, R)
                
                # Snap to half-integers
                U_snap = snap_to_half_integer(U_sol, tol=1e-5)
                V_snap = snap_to_half_integer(V_sol, tol=1e-5)
                W_snap = snap_to_half_integer(W_sol, tol=1e-5)
                
                if verify(U_snap, V_snap, W_snap, R, tol=1e-6):
                    best_result = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                    best_R = R
                    break  # Found a valid decomposition for this R, try lower R
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
