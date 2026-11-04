"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import random
from itertools import product
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Build the target tensor
    # T[(i,j), (j,k), (i,k)] = 1
    # Using trace convention: W[k*n+i, r]
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    schoolbook_rank = n * m * p  # 45
    
    # Lower bound considerations
    lower_bound = max(n*m, m*p, p*n)  # 15
    
    # Best result starts as schoolbook
    U_sch = np.zeros((n*m, schoolbook_rank))
    V_sch = np.zeros((m*p, schoolbook_rank))
    W_sch = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i*m+j, col] = 1
                V_sch[j*p+k, col] = 1
                W_sch[k*n+i, col] = 1
                col += 1
    
    best_result = (U_sch, V_sch, W_sch)
    best_rank = schoolbook_rank
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that the decomposition exactly reconstructs the tensor."""
        rank = U.shape[1]
        approx = np.zeros_like(T)
        for r in range(rank):
            approx += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        return np.allclose(T, approx, atol=tol)
    
    def is_half_integer(arr, tol=1e-8):
        """Check if array entries are close to half-integers."""
        doubled = arr * 2
        return np.allclose(doubled, np.round(doubled), atol=tol)
    
    def round_to_half_integer(arr):
        """Round array entries to nearest half-integer."""
        return np.round(arr * 2) / 2
    
    def try_coordinate_descent(U_init, V_init, W_init, target_rank, max_iters=2000, tol=1e-12):
        """
        Coordinate descent to minimize ||T - sum_r U_r ⊗ V_r ⊗ W_r||^2
        with U, V, W fixed except one block at a time.
        """
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        # Precompute for efficiency
        # For updating U[:,r]: minimize ||T[:,j,k] - sum_s U_s[:,r] * V_s[:,j] * W_s[:,k]||^2
        # This is a linear least squares problem
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U
            for r in range(target_rank):
                # Build RHS: T - contributions from other ranks
                residual = T.copy()
                for s in range(target_rank):
                    if s != r:
                        residual -= np.einsum('a,b,c->abc', U[:,s], V[:,s], W[:,s])
                
                # We need: sum_a U[a,r] * V[b,r] * W[c,r] = residual[a,b,c]
                # For fixed V[:,r] and W[:,r], this is linear in U[:,r]
                # residual[a,b,c] = U[a,r] * V[b,r] * W[c,r]
                # So U[a,r] = residual[a,b,c] / (V[b,r] * W[c,r]) for any (b,c) with V[b,r]*W[c,r] != 0
                # But we want least squares over all (b,c)
                
                # Reshape: for each a, we have residual[a,:,:] and need to solve
                # U[a,r] * (V[:,r] ⊗ W[:,r]) ≈ residual[a,:,:]
                # This is: U[a,r] * vec(V[:,r] ⊗ W[:,r]) ≈ vec(residual[a,:,:])
                
                v_w = np.outer(V[:,r], W[:,r])  # (m*p, p*n)
                v_w_flat = v_w.flatten()
                
                if np.linalg.norm(v_w_flat) < 1e-15:
                    continue
                
                # For each a, solve: U[a,r] * v_w_flat ≈ residual[a,:,:].flatten()
                # Least squares: U[a,r] = dot(residual[a,:,:].flatten(), v_w_flat) / dot(v_w_flat, v_w_flat)
                denom = np.dot(v_w_flat, v_w_flat)
                for a in range(n*m):
                    rhs = residual[a,:,:].flatten()
                    U[a,r] = np.dot(rhs, v_w_flat) / denom
            
            # Update V
            for r in range(target_rank):
                residual = T.copy()
                for s in range(target_rank):
                    if s != r:
                        residual -= np.einsum('a,b,c->abc', U[:,s], V[:,s], W[:,s])
                
                u_w = np.outer(U[:,r], W[:,r])  # (n*m, p*n)
                u_w_flat = u_w.flatten()
                
                if np.linalg.norm(u_w_flat) < 1e-15:
                    continue
                
                denom = np.dot(u_w_flat, u_w_flat)
                for b in range(m*p):
                    rhs = residual[:,b,:].flatten()
                    V[b,r] = np.dot(rhs, u_w_flat) / denom
            
            # Update W
            for r in range(target_rank):
                residual = T.copy()
                for s in range(target_rank):
                    if s != r:
                        residual -= np.einsum('a,b,c->abc', U[:,s], V[:,s], W[:,s])
                
                u_v = np.outer(U[:,r], V[:,r])  # (n*m, m*p)
                u_v_flat = u_v.flatten()
                
                if np.linalg.norm(u_v_flat) < 1e-15:
                    continue
                
                denom = np.dot(u_v_flat, u_v_flat)
                for c in range(p*n):
                    rhs = residual[:,:,c].flatten()
                    W[c,r] = np.dot(rhs, u_v_flat) / denom
            
            # Check convergence
            approx = np.zeros_like(T)
            for r in range(target_rank):
                approx += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            error = np.linalg.norm(T - approx)
            if error < tol:
                break
        
        return U, V, W
    
    # Search for lower rank decompositions
    # Try ranks from lower_bound up to schoolbook_rank - 1
    for target_rank in range(lower_bound, schoolbook_rank):
        if time.time() > deadline - 10:
            break
        
        found = False
        
        # Multiple initialization strategies
        n_restarts = 8
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            # Different initialization strategies
            if restart == 0:
                # Sparse random initialization with small integers
                U_init = np.zeros((n*m, target_rank))
                V_init = np.zeros((m*p, target_rank))
                W_init = np.zeros((p*n, target_rank))
                for r in range(target_rank):
                    # Randomly select a few entries
                    for _ in range(random.randint(3, 8)):
                        a = random.randint(0, n*m-1)
                        U_init[a, r] = random.choice([-1, 1, -0.5, 0.5])
                    for _ in range(random.randint(3, 8)):
                        b = random.randint(0, m*p-1)
                        V_init[b, r] = random.choice([-1, 1, -0.5, 0.5])
                    for _ in range(random.randint(3, 8)):
                        c = random.randint(0, p*n-1)
                        W_init[c, r] = random.choice([-1, 1, -0.5, 0.5])
                
            elif restart == 1:
                # Structured initialization: try to exploit the tensor structure
                # Each term should contribute to a specific (i,j,k) pattern
                U_init = np.zeros((n*m, target_rank))
                V_init = np.zeros((m*p, target_rank))
                W_init = np.zeros((p*n, target_rank))
                
                # Assign each rank to a random (i,j,k) combination
                indices = list(product(range(n), range(m), range(p)))
                random.shuffle(indices)
                
                for r in range(min(target_rank, len(indices))):
                    i, j, k = indices[r]
                    U_init[i*m+j, r] = 1
                    V_init[j*p+k, r] = 1
                    W_init[k*n+i, r] = 1
                
            elif restart == 2:
                # Random Gaussian initialization
                U_init = np.random.randn(n*m, target_rank) * 0.3
                V_init = np.random.randn(m*p, target_rank) * 0.3
                W_init = np.random.randn(p*n, target_rank) * 0.3
                
            elif restart == 3:
                # Binary initialization
                U_init = np.random.choice([0, 1], size=(n*m, target_rank)).astype(float)
                V_init = np.random.choice([0, 1], size=(m*p, target_rank)).astype(float)
                W_init = np.random.choice([0, 1], size=(p*n, target_rank)).astype(float)
                
            elif restart == 4:
                # Half-integer random
                U_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, target_rank))
                V_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, target_rank))
                W_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, target_rank))
                
            elif restart == 5:
                # Block diagonal structure
                U_init = np.zeros((n*m, target_rank))
                V_init = np.zeros((m*p, target_rank))
                W_init = np.zeros((p*n, target_rank))
                block_size = max(1, target_rank // 5)
                for r in range(target_rank):
                    block = r // block_size
                    # Assign to a specific block of indices
                    a_start = block * (n*m // 5)
                    a_end = min((block+1) * (n*m // 5), n*m)
                    if a_end > a_start:
                        a = random.randint(a_start, a_end-1)
                        U_init[a, r] = random.choice([-1, 1])
                    
                    b_start = block * (m*p // 5)
                    b_end = min((block+1) * (m*p // 5), m*p)
                    if b_end > b_start:
                        b = random.randint(b_start, b_end-1)
                        V_init[b, r] = random.choice([-1, 1])
                    
                    c_start = block * (p*n // 5)
                    c_end = min((block+1) * (p*n // 5), p*n)
                    if c_end > c_start:
                        c = random.randint(c_start, c_end-1)
                        W_init[c, r] = random.choice([-1, 1])
                        
            elif restart == 6:
                # Try to build from partial schoolbook
                U_init = np.zeros((n*m, target_rank))
                V_init = np.zeros((m*p, target_rank))
                W_init = np.zeros((p*n, target_rank))
                
                # Use first target_rank schoolbook terms
                for r in range(target_rank):
                    i = r // (m*p)
                    rem = r % (m*p)
                    j = rem // p
                    k = rem % p
                    if i < n and j < m and k < p:
                        U_init[i*m+j, r] = 1
                        V_init[j*p+k, r] = 1
                        W_init[k*n+i, r] = 1
                        
            else:
                # Mixed initialization
                U_init = np.random.randn(n*m, target_rank) * 0.2
                V_init = np.random.randn(m*p, target_rank) * 0.2
                W_init = np.random.randn(p*n, target_rank) * 0.2
                # Add some structure
                for r in range(min(3, target_rank)):
                    U_init[0, r] = 1
                    V_init[0, r] = 1
                    W_init[0, r] = 1
            
            # Run coordinate descent
            U_cd, V_cd, W_cd = try_coordinate_descent(U_init, V_init, W_cd if False else W_init, target_rank, max_iters=1500)
            
            # Check if we got a good solution
            approx = np.zeros_like(T)
            for r in range(target_rank):
                approx += np.einsum('a,b,c->abc', U_cd[:,r], V_cd[:,r], W_cd[:,r])
            error = np.linalg.norm(T - approx)
            
            if error < 1e-8:
                # Try to round to half-integers
                U_rounded = round_to_half_integer(U_cd)
                V_rounded = round_to_half_integer(V_cd)
                W_rounded = round_to_half_integer(W_cd)
                
                # Verify rounded version
                approx_rounded = np.zeros_like(T)
                for r in range(target_rank):
                    approx_rounded += np.einsum('a,b,c->abc', U_rounded[:,r], V_rounded[:,r], W_rounded[:,r])
                
                if np.allclose(T, approx_rounded, atol=1e-10):
                    best_result = (U_rounded, V_rounded, W_rounded)
                    best_rank = target_rank
                    found = True
                    break
                
                # If rounding didn't work, try a few nearby half-integer candidates
                # for the entries that are close but not exact
                for attempt in range(3):
                    U_try = U_rounded.copy()
                    V_try = V_rounded.copy()
                    W_try = W_rounded.copy()
                    
                    # Perturb some entries by ±0.5
                    for _ in range(random.randint(1, 5)):
                        if random.random() < 0.33:
                            a, r = random.randint(0, n*m-1), random.randint(0, target_rank-1)
                            U_try[a, r] += random.choice([-0.5, 0.5])
                        elif random.random() < 0.5:
                            b, r = random.randint(0, m*p-1), random.randint(0, target_rank-1)
                            V_try[b, r] += random.choice([-0.5, 0.5])
                        else:
                            c, r = random.randint(0, p*n-1), random.randint(0, target_rank-1)
                            W_try[c, r] += random.choice([-0.5, 0.5])
                    
                    # Verify
                    approx_try = np.zeros_like(T)
                    for r in range(target_rank):
                        approx_try += np.einsum('a,b,c->abc', U_try[:,r], V_try[:,r], W_try[:,r])
                    
                    if np.allclose(T, approx_try, atol=1e-10):
                        best_result = (U_try, V_try, W_try)
                        best_rank = target_rank
                        found = True
                        break
                
                if found:
                    break
        
        if found:
            # Found a working rank, continue to try lower
            pass
    
    # Also try a direct search for very low ranks with structured patterns
    for target_rank in range(lower_bound, min(schoolbook_rank, 25)):
        if time.time() > deadline - 10:
            break
        if target_rank >= best_rank:
            continue
        
        # Try specific algebraic patterns
        # Pattern: each term handles a subset of (i,j,k) tuples
        patterns = [
            # Pattern 1: Diagonal dominance
            lambda r, i, j, k: (1 if i == j == k else 0),
            # Pattern 2: Cyclic
            lambda r, i, j, k: (1 if (i + j + k) % target_rank == r else 0),
        ]
        
        for pattern in patterns:
            if time.time() > deadline - 5:
                break
            
            U_pat = np.zeros((n*m, target_rank))
            V_pat = np.zeros((m*p, target_rank))
            W_pat = np.zeros((p*n, target_rank))
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(target_rank):
                            val = pattern(r, i, j, k)
                            U_pat[i*m+j, r] += val
                            V_pat[j*p+k, r] += val
                            W_pat[k*n+i, r] += val
            
            # Verify
            approx = np.zeros_like(T)
            for r in range(target_rank):
                approx += np.einsum('a,b,c->abc', U_pat[:,r], V_pat[:,r], W_pat[:,r])
            
            if np.allclose(T, approx, atol=1e-10):
                best_result = (U_pat, V_pat, W_pat)
                best_rank = target_rank
                break
    
    # Convert to lists
    if isinstance(best_result[0], np.ndarray):
        U_final = best_result[0].tolist()
        V_final = best_result[1].tolist()
        W_final = best_result[2].tolist()
    else:
        U_final = best_result[0]
        V_final = best_result[1]
        W_final = best_result[2]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
