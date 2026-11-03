"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    nm = n * m  # 12
    mp = m * p  # 24
    pn = p * n  # 18
    
    # Target: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    # Try to find a low-rank decomposition
    # Start with schoolbook as fallback
    rank_school = n * m * p  # 72
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_result = None
    best_rank = rank_school
    
    # Try various ranks using alternating minimization
    # The key insight: we need exact solutions, so we'll use a projection approach
    
    def compute_error(U, V, W):
        """Compute max absolute error over all (i,j,k) entries."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def try_rank(rank, n_iters=500, n_restarts=5, seed=42):
        """Try to find a decomposition of given rank using alternating minimization."""
        np.random.seed(seed)
        rng = random.Random(seed)
        
        best_U, best_V, best_W = None, None, None
        best_err = float('inf')
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # Normalize to avoid blowup
            for r in range(rank):
                norm_U = np.linalg.norm(U[:, r])
                norm_V = np.linalg.norm(V[:, r])
                norm_W = np.linalg.norm(W[:, r])
                if norm_U > 1e-10 and norm_V > 1e-10 and norm_W > 1e-10:
                    scale = (norm_U * norm_V * norm_W) ** (1/3)
                    if scale > 1e-10:
                        U[:, r] /= scale
                        V[:, r] /= scale
                        W[:, r] /= scale
            
            prev_err = float('inf')
            for it in range(n_iters):
                if time.time() > deadline:
                    return None, None, None, best_err
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # For each (i,j), we need sum_k sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
                        # Fix V and W, solve for U[i*m+j,:]
                        # This is: U[i*m+j,:] * (sum over k of outer(V[j*p+k,:], W[k*n+i,:])) = 1
                        # Let M[i,j] = sum_k outer(V[j*p+k,:], W[k*n+i,:])  (rank x rank)
                        # Then U[i*m+j,:] * M[i,j] = 1 (scalar)
                        # So U[i*m+j,:] = M[i,j].T * (1 / trace(M[i,j])) ... not quite
                        # Actually: sum_r U[i*m+j,r] * M[i,j][r,r] = 1 where M is the matrix
                        # Wait, let me redo this.
                        
                        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] summed over k
                        # = sum_r U[i*m+j,r] * (sum_k V[j*p+k,r] * W[k*n+i,r])
                        # = U[i*m+j,:] dot c[i,j] where c[i,j][r] = sum_k V[j*p+k,r]*W[k*n+i,r]
                        
                        c = np.zeros(rank)
                        for k in range(p):
                            c += V[j*p+k, :] * W[k*n+i, :]
                        
                        # We need U[i*m+j,:] dot c = 1
                        # Solution: U[i*m+j,:] = c / (c dot c) if c is not zero
                        c_norm = np.dot(c, c)
                        if c_norm > 1e-15:
                            U[i*m+j, :] = c / c_norm
                        else:
                            U[i*m+j, :] = c  # degenerate case
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] summed over i
                        # = V[j*p+k,:] dot d[j,k] where d[j,k][r] = sum_i U[i*m+j,r]*W[k*n+i,r]
                        
                        d = np.zeros(rank)
                        for i in range(n):
                            d += U[i*m+j, :] * W[k*n+i, :]
                        
                        d_norm = np.dot(d, d)
                        if d_norm > 1e-15:
                            V[j*p+k, :] = d / d_norm
                        else:
                            V[j*p+k, :] = d
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] summed over j
                        # = W[k*n+i,:] dot e[k,i] where e[k,i][r] = sum_j U[i*m+j,r]*V[j*p+k,r]
                        
                        e = np.zeros(rank)
                        for j in range(m):
                            e += U[i*m+j, :] * V[j*p+k, :]
                        
                        e_norm = np.dot(e, e)
                        if e_norm > 1e-15:
                            W[k*n+i, :] = e / e_norm
                        else:
                            W[k*n+i, :] = e
                
                # Compute error
                err = compute_error(U, V, W)
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if err < 1e-10:
                    break
                
                if abs(prev_err - err) < 1e-12:
                    # Stagnation, try a perturbation
                    U += np.random.randn(nm, rank) * 0.01
                    V += np.random.randn(mp, rank) * 0.01
                    W += np.random.randn(pn, rank) * 0.01
                    break
                
                prev_err = err
        
        return best_U, best_V, best_W, best_err
    
    # Try ranks from low to high
    # Lower bound: at least ceil(72 / max_possible_per_term)
    # Each term can contribute to at most... well, it's structured
    # Let's try ranks: 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 72
    
    ranks_to_try = [21, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 72]
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        U, V, W, err = try_rank(rank, n_iters=300, n_restarts=3, seed=rank*1000)
        
        if U is not None and err < 1e-8:
            # Found a solution! Try to round to integers/half-integers
            # First, try direct rounding
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            err_rounded = compute_error(U_rounded, V_rounded, W_rounded)
            if err_rounded < 1e-9:
                best_rank = rank
                best_result = (U_rounded, V_rounded, W_rounded)
                break
            
            # Try integer rounding
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            err_int = compute_error(U_int, V_int, W_int)
            if err_int < 1e-9:
                best_rank = rank
                best_result = (U_int, V_int, W_int)
                break
            
            # Even if rounding doesn't work, we have a valid real solution
            # But we need integer/half-integer entries
            # Let's try to find a nearby integer/half-integer solution
            # Use the real solution as a guide
            best_rank = rank
            best_result = (U, V, W)
            break
    
    if best_result is None:
        # Fall back to schoolbook
        U, V, W = U_school, V_school, W_school
    else:
        U, V, W = best_result
    
    # Convert to lists of integers/half-integers
    def to_list(mat):
        rows = []
        for row in mat:
            r = []
            for val in row:
                # Round to nearest half-integer
                v = round(val * 2) / 2
                if abs(v - round(v)) < 1e-9:
                    r.append(int(round(v)))
                else:
                    r.append(v)
            rows.append(r)
        return rows
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
