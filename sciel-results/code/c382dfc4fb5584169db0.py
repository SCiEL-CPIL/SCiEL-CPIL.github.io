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
    
    # Indices for active terms (i, j, k) -> (i*m+j, j*p+k, k*n+i)
    active_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                active_indices.append((a, b, c))
                
    # Total number of constraints
    # Active: n*m*p
    # Inactive: (nm * mp * pn) - (n*m*p)
    # This is huge: 6 * 15 * 10 = 900 total, 30 active, 870 inactive.
    # We can exploit symmetry or just solve the full system.
    
    def get_residuals(params, R):
        # params: flat array of size R * (nm + mp + pn)
        # U: (nm, R), V: (mp, R), W: (pn, R)
        U = params[:R*nm].reshape(nm, R)
        V = params[R*nm:R*nm+R*mp].reshape(mp, R)
        W = params[R*nm+R*mp:].reshape(pn, R)
        
        residuals = []
        
        # Active constraints
        for a, b, c in active_indices:
            # sum_r U[a,r] V[b,r] W[c,r]
            val = np.sum(U[a] * V[b] * W[c])
            residuals.append(val - 1.0)
            
        # Inactive constraints
        # We need to iterate over all a, b, c that are NOT in active_indices
        # To be efficient, we can precompute the set of active indices
        active_set = set(active_indices)
        
        # Optimization: Only check a subset of inactive constraints if R is large?
        # No, we need exactness. But for the optimizer, we can use all.
        # 870 constraints is manageable.
        
        for a in range(nm):
            for b in range(mp):
                # Check if (a, b, c) is active for any c?
                # Active means a = i*m+j, b = j*p+k, c = k*n+i
                # Given a and b, is there a unique k and i?
                # a = i*m + j => j = a % m, i = a // m
                # b = j*p + k => k = b % p, j' = b // p
                # If j == j', then it is active with c = k*n + i
                j = a % m
                i = a // m
                j_prime = b // p
                k = b % p
                
                if j == j_prime:
                    c = k * n + i
                    # This is an active index
                    # We already handled it? 
                    # Wait, the loop above handled all active indices.
                    # So we should SKIP this in the inactive loop.
                    continue
                else:
                    # It is inactive for ANY c?
                    # No, for a fixed a, b, the term is inactive for ALL c if j != j_prime.
                    # If j == j_prime, it is active for ONE specific c, and inactive for others.
                    # So if j != j_prime, then for all c, (a,b,c) is inactive.
                    # There are pn such c's.
                    for c in range(pn):
                         val = np.sum(U[a] * V[b] * W[c])
                         residuals.append(val)
                # If j == j_prime, we skipped the active c.
                # We need to add constraints for the OTHER c's where (a,b,c) is inactive.
                if j == j_prime:
                    c_active = k * n + i
                    for c in range(pn):
                        if c == c_active:
                            continue
                        val = np.sum(U[a] * V[b] * W[c])
                        residuals.append(val)

        return np.array(residuals)

    # Try to find a solution for R from 15 to 25
    # We use a deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 110s limit
    
    # Trivial solution as fallback
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
    best_rank = rank_trivial

    # Search for lower rank
    # Start from R=15 up to 20
    for R in range(15, 21):
        if time.time() > deadline:
            break
            
        # Initialize with random small values
        np.random.seed(R) # Reproducible
        x0 = np.random.randn(R * (nm + mp + pn)) * 0.1
        
        # Use least_squares
        try:
            result = least_squares(get_residuals, x0, args=(R,), method='lm', max_nfev=1000)
            if result.success or result.cost < 1e-8:
                # Check if solution is close to integers/half-integers
                params = result.x
                U = params[:R*nm].reshape(nm, R)
                V = params[R*nm:R*nm+R*mp].reshape(mp, R)
                W = params[R*nm+R*mp:].reshape(pn, R)
                
                # Snap to nearest 0.5
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify exactness with snapped values
                # Convert to lists for return format later, but verify with numpy
                valid = True
                for a, b, c in active_indices:
                    val = np.sum(U_snap[a] * V_snap[b] * W_snap[c])
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if valid:
                    # Check inactive
                    for a in range(nm):
                        for b in range(mp):
                            j = a % m
                            i = a // m
                            j_prime = b // p
                            k = b % p
                            if j == j_prime:
                                c_active = k * n + i
                                for c in range(pn):
                                    if c == c_active: continue
                                    val = np.sum(U_snap[a] * V_snap[b] * W_snap[c])
                                    if abs(val) > 1e-9:
                                        valid = False
                                        break
                        if not valid: break
                    if valid:
                        # Found a valid solution!
                        # Convert to lists of ints/floats
                        U_list = U_snap.tolist()
                        V_list = V_snap.tolist()
                        W_list = W_snap.tolist()
                        return {"U": U_list, "V": V_list, "W": W_list}
                        
        except Exception as e:
            pass

    return best_result
# EVOLVE-BLOCK-END
