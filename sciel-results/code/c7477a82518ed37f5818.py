"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

def construct():
    """
    Find a low-rank exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns dict with U, V, W matrices.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 72
    
    # Build schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = schoolbook_rank
    
    # Precompute target tensor T
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T is 1 if indices match the multiplication pattern, else 0
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # Reshape T for linear system:
    # We want sum_r U[a, r] V[b, r] W[c, r] = T[a, b, c]
    # For fixed c, we have:
    # sum_r W[c, r] * (U[:, r] * V[:, r])_a,b = T[a, b, c]
    # Let M_c be a matrix of size (n*m * m*p) x R
    # M_c[ (a,b), r ] = U[a, r] * V[b, r]
    # Then M_c @ W_c = T_c where T_c is T[:, :, c] flattened
    
    T_flat_by_c = np.zeros((p * n, n * m * m * p))
    for c in range(p * n):
        T_flat_by_c[c] = T[:, :, c].flatten()
        
    def try_rank(R, max_attempts=100):
        nonlocal best_U, best_V, best_W, best_rank
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return False
            
            # Generate random U, V with small integer entries
            # Using -1, 0, 1 to keep numbers small
            U = np.random.choice([-1, 0, 1], size=(n * m, R))
            V = np.random.choice([-1, 0, 1], size=(m * p, R))
            
            # Check if U, V have enough rank/structure
            # Construct M_c for all c
            # We can do this efficiently
            # M_c[:, r] = kron(U[:, r], V[:, r])
            
            # Precompute kron vectors for each r
            # kron_vecs[r] is size n*m*m*p
            kron_vecs = np.zeros((R, n * m * m * p))
            for r in range(R):
                kron_vecs[r] = np.kron(U[:, r], V[:, r])
                
            # Now for each c, solve kron_vecs.T @ W_c = T_flat_by_c[c]
            # kron_vecs.T is (n*m*m*p) x R
            # We can use lstsq
            # Since we want exact integer/half-integer, we check residual
            
            W = np.zeros((p * n, R))
            valid = True
            for c in range(p * n):
                # Solve A x = b
                # A = kron_vecs.T
                # b = T_flat_by_c[c]
                try:
                    W_c, residuals, rank, sv = np.linalg.lstsq(kron_vecs.T, T_flat_by_c[c], rcond=None)
                    W[c] = W_c
                    if len(residuals) > 0 and residuals[0] > 1e-9:
                        valid = False
                        break
                except:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # If consistent, verify exact decomposition
            # Reconstruct T_recon
            T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            if np.allclose(T_recon, T, atol=1e-9, rtol=1e-9):
                # Check if entries are close to integers or halves
                # If so, snap them
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                T_recon_snap = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)
                if np.allclose(T_recon_snap, T, atol=1e-9, rtol=1e-9):
                    best_U = U_snap
                    best_V = V_snap
                    best_W = W_snap
                    best_rank = R
                    return True
            else:
                # Maybe the solution is not snap-able but is valid float?
                # The task requires integer/half-integer.
                # If we found a float solution, we might try to rationalize it, 
                # but for now, we only accept if snap works.
                pass
                
        return False

    # Search for lower rank
    # Lower bound is max(n*m, m*p, p*n) = 24
    for R in range(24, schoolbook_rank):
        if time.time() > deadline - 20:
            break
        
        # Increase attempts for lower ranks as they are harder/rarer
        attempts = 200 if R < 40 else 50
        
        if try_rank(R, attempts):
            break
            
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
