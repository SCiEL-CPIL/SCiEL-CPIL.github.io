import time
import sys
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute active indices
    active_indices = []
    active_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                active_indices.append((a, b, c))
                active_set.add((a, b, c))
                
    # Precompute inactive indices efficiently
    inactive_indices = []
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) not in active_set:
                    inactive_indices.append((a, b, c))
                    
    # Extract arrays for vectorization
    active_a = np.array([x[0] for x in active_indices])
    active_b = np.array([x[1] for x in active_indices])
    active_c = np.array([x[2] for x in active_indices])
    
    inactive_a = np.array([x[0] for x in inactive_indices])
    inactive_b = np.array([x[1] for x in inactive_indices])
    inactive_c = np.array([x[2] for x in inactive_indices])
    
    num_active = len(active_indices)
    num_inactive = len(inactive_indices)
    
    def get_residuals(params, R):
        U = params[:R*nm].reshape(nm, R)
        V = params[R*nm:R*nm+R*mp].reshape(mp, R)
        W = params[R*nm+R*mp:].reshape(pn, R)
        
        # Vectorized computation
        U_a = U[active_a]
        V_a = V[active_b]
        W_a = W[active_c]
        res_active = np.sum(U_a * V_a * W_a, axis=1) - 1.0
        
        U_i = U[inactive_a]
        V_i = V[inactive_b]
        W_i = W[inactive_c]
        res_inactive = np.sum(U_i * V_i * W_i, axis=1)
        
        return np.concatenate([res_active, res_inactive])

    # Trivial solution
    rank_trivial = n * m * p
    U_trivial = [[0] * rank_trivial for _ in range(nm)]
    V_trivial = [[0] * rank_trivial for _ in range(mp)]
    W_trivial = [[0] * rank_trivial for _ in range(pn)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][r] = 1
                V_trivial[j * p + k][r] = 1
                W_trivial[k * n + i][r] = 1
                r += 1

    best_result = {"U": U_trivial, "V": V_trivial, "W": W_trivial}
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Helper to generate structured initialization based on (i,j,k) blocks
    def get_structured_init(R):
        """Create initialization that respects the block structure"""
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        
        # Assign rank components to (i,j,k) blocks with some sharing
        # For lower R, we need to share components across triples
        # Strategy: group by i, then j, then k
        
        np.random.seed(42)
        
        if R >= 30:
            # Can use trivial-like structure
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if idx < R:
                            U[i*m+j, idx] = 1.0
                            V[j*p+k, idx] = 1.0
                            W[k*n+i, idx] = 1.0
                            idx += 1
        else:
            # Need to share: use a cyclic assignment
            # Each rank component covers multiple (i,j,k) triples
            for r in range(R):
                # Choose which (i,j,k) triples this component covers
                # Use a hash-like function to distribute
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Simple modular assignment
                            if (i * 7 + j * 3 + k * 11) % R == r:
                                U[i*m+j, r] += 1.0
                                V[j*p+k, r] += 1.0
                                W[k*n+i, r] += 1.0
            
            # Normalize to help with optimization
            for r in range(R):
                max_u = np.max(np.abs(U[:, r]))
                max_v = np.max(np.abs(V[:, r]))
                max_w = np.max(np.abs(W[:, r]))
                if max_u > 0:
                    U[:, r] /= max_u
                if max_v > 0:
                    V[:, r] /= max_v
                if max_w > 0:
                    W[:, r] /= max_w
        
        # Add small random perturbation to break symmetry
        U += np.random.randn(nm, R) * 0.05
        V += np.random.randn(mp, R) * 0.05
        W += np.random.randn(pn, R) * 0.05
        
        return U.flatten(), V.flatten(), W.flatten()

    # Try to find lower rank solutions
    # Start from R=20 and go up to 29
    for R in range(20, 30):
        if time.time() > deadline:
            break
            
        # Try multiple initializations
        num_attempts = 8 if R < 25 else 4
        found = False
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                    break
                    
            seed = R * 1000 + attempt
            np.random.seed(seed)
            
            if attempt == 0:
                # Structured init
                u_flat, v_flat, w_flat = get_structured_init(R)
                x0 = np.concatenate([u_flat, v_flat, w_flat])
            elif attempt == 1:
                # Different structured init with different grouping
                U = np.zeros((nm, R))
                V = np.zeros((mp, R))
                W = np.zeros((pn, R))
                for r in range(R):
                    # Alternate grouping by different dimensions
                    if r % 3 == 0:
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    if (i + j + k) % R == r:
                                        U[i*m+j, r] += 1.0
                                        V[j*p+k, r] += 1.0
                                        W[k*n+i, r] += 1.0
                    elif r % 3 == 1:
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    if (i * 2 + j * 3 + k * 5) % R == r:
                                        U[i*m+j, r] += 1.0
                                        V[j*p+k, r] += 1.0
                                        W[k*n+i, r] += 1.0
                    else:
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    if (i * 5 + j * 2 + k * 3) % R == r:
                                        U[i*m+j, r] += 1.0
                                        V[j*p+k, r] += 1.0
                                        W[k*n+i, r] += 1.0
                U += np.random.randn(nm, R) * 0.05
                V += np.random.randn(mp, R) * 0.05
                W += np.random.randn(pn, R) * 0.05
                x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            else:
                # Random init
                x0 = np.random.randn(R * (nm + mp + pn)) * 0.3
            
            try:
                # Use 'lm' method for unconstrained least squares
                result = least_squares(get_residuals, x0, args=(R,), method='lm', 
                                       max_nfev=3000, ftol=1e-14, xtol=1e-14, gtol=1e-14)
                
                if result.success or result.cost < 1e-12:
                    params = result.x
                    U = params[:R*nm].reshape(nm, R)
                    V = params[R*nm:R*nm+R*mp].reshape(mp, R)
                    W = params[R*nm+R*mp:].reshape(pn, R)
                    
                    # Snap to nearest 0.5
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    # Verify exactness
                    U_a = U_snap[active_a]
                    V_a = V_snap[active_b]
                    W_a = W_snap[active_c]
                    res_active = np.sum(U_a * V_a * W_a, axis=1) - 1.0
                    if np.max(np.abs(res_active)) > 1e-9:
                        continue
                        
                    U_i = U_snap[inactive_a]
                    V_i = V_snap[inactive_b]
                    W_i = W_snap[inactive_c]
                    res_inactive = np.sum(U_i * V_i * W_i, axis=1)
                    if np.max(np.abs(res_inactive)) > 1e-9:
                        continue
                        
                    # Valid solution found
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                        
            except Exception:
                pass

    return best_result
# EVOLVE-BLOCK-END
