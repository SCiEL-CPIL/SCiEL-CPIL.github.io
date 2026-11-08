import time
import sys
import numpy as np
from scipy.optimize import least_squares
from itertools import product

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute active indices: (a, b, c) where a=i*m+j, b=j*p+k, c=k*n+i
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
                
    # Trivial solution as fallback (rank 30)
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

    # Function to verify a candidate decomposition
    def verify(U, V, W):
        # U, V, W are numpy arrays
        # Check active constraints
        for a, b, c in active_indices:
            val = np.dot(U[a], np.dot(V[b], W[c]))
            # Wait, the tensor is sum_r U[a,r] V[b,r] W[c,r]
            val = np.sum(U[a] * V[b] * W[c])
            if abs(val - 1.0) > 1e-9:
                return False
        
        # Check inactive constraints
        for a in range(nm):
            for b in range(mp):
                # Determine if (a,b) corresponds to a valid i,j,k combination for some c
                # a = i*m + j => j = a % m, i = a // m
                # b = j'*p + k => j' = b // p, k = b % p
                j = a % m
                i = a // m
                j_prime = b // p
                k = b % p
                
                if j == j_prime:
                    # There is one active c: c_active = k*n + i
                    c_active = k * n + i
                    for c in range(pn):
                        if c == c_active:
                            continue
                        val = np.sum(U[a] * V[b] * W[c])
                        if abs(val) > 1e-9:
                            return False
                else:
                    # No active c for this (a,b)
                    for c in range(pn):
                        val = np.sum(U[a] * V[b] * W[c])
                        if abs(val) > 1e-9:
                            return False
        return True

    def get_residuals(params, R):
        # params: flat array of size R * (nm + mp + pn)
        # U: (nm, R), V: (mp, R), W: (pn, R)
        U = params[:R*nm].reshape(nm, R)
        V = params[R*nm:R*nm+R*mp].reshape(mp, R)
        W = params[R*nm+R*mp:R*nm+R*mp+R*pn].reshape(pn, R)
        
        residuals = []
        
        # Active constraints: sum_r U[a,r] V[b,r] W[c,r] = 1
        for a, b, c in active_indices:
            val = np.sum(U[a] * V[b] * W[c])
            residuals.append(val - 1.0)
            
        # Inactive constraints: sum_r U[a,r] V[b,r] W[c,r] = 0
        # To speed up, we only iterate over unique (a,b) pairs and handle c appropriately
        for a in range(nm):
            for b in range(mp):
                j = a % m
                i = a // m
                j_prime = b // p
                k = b % p
                
                if j == j_prime:
                    c_active = k * n + i
                    for c in range(pn):
                        if c == c_active:
                            continue
                        val = np.sum(U[a] * V[b] * W[c])
                        residuals.append(val)
                else:
                    for c in range(pn):
                        val = np.sum(U[a] * V[b] * W[c])
                        residuals.append(val)

        return np.array(residuals)

    # Search for lower rank
    # Start from R=15 up to 25
    # We use a deadline guard
    start_time = time.time()
    deadline = start_time + 115  # 115s limit, leaving 5s buffer for safety

    for R in range(15, 26):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations for each R
        for attempt in range(3):
            if time.time() > deadline:
                break
                
            np.random.seed(R * 100 + attempt)
            x0 = np.random.randn(R * (nm + mp + pn)) * 0.5
            
            try:
                # Use least_squares with Levenberg-Marquardt
                # max_nfev is critical for speed
                result = least_squares(
                    get_residuals, 
                    x0, 
                    args=(R,), 
                    method='lm', 
                    max_nfev=500,
                    ftol=1e-12,
                    xtol=1e-12
                )
                
                if result.cost < 1e-10:
                    params = result.x
                    U = params[:R*nm].reshape(nm, R)
                    V = params[R*nm:R*nm+R*mp].reshape(mp, R)
                    W = params[R*nm+R*mp:R*nm+R*mp+R*pn].reshape(pn, R)
                    
                    # Snap to nearest 0.5
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    # Verify exactness with snapped values
                    if verify(U_snap, V_snap, W_snap):
                        # Found a valid solution!
                        # Convert to lists of ints/floats
                        U_list = U_snap.tolist()
                        V_list = V_snap.tolist()
                        W_list = W_snap.tolist()
                        
                        # Clean up floats that are effectively integers
                        def clean_list(lst):
                            cleaned = []
                            for row in lst:
                                new_row = []
                                for val in row:
                                    if abs(val - round(val)) < 1e-9:
                                        new_row.append(int(round(val)))
                                    else:
                                        new_row.append(val)
                                cleaned.append(new_row)
                            return cleaned

                        return {"U": clean_list(U_list), "V": clean_list(V_list), "W": clean_list(W_list)}
                        
            except Exception:
                pass

    return best_result
# EVOLVE-BLOCK-END
