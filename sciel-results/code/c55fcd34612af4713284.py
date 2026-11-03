import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m      # 8
    dim_v = m * p      # 28
    dim_w = p * n      # 14
    
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill

    # Build target tensor: T[u,v,w] = 1 iff there exist i,j,k with
    # u = i*m+j, v = j*p+k, w = k*n+i
    # We'll build this as a sparse set of (u,v,w) tuples
    target_entries = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_entries.add((u, v, w))

    # Schoolbook fallback: rank = n*m*p = 56
    rank_school = n * m * p
    U_school = np.zeros((dim_u, rank_school))
    V_school = np.zeros((dim_v, rank_school))
    W_school = np.zeros((dim_w, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U_school[u, col] = 1
                V_school[v, col] = 1
                W_school[w, col] = 1
                col += 1

    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()

    def verify(U, V, W):
        """Check if sum_r U[u,r]*V[v,r]*W[w,r] = T[u,v,w] for all u,v,w"""
        for u in range(dim_u):
            for v in range(dim_v):
                for w in range(dim_w):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[u, r] * V[v, r] * W[w, r]
                    expected = 1.0 if (u, v, w) in target_entries else 0.0
                    if abs(val - expected) > 1e-8:
                        return False
        return True

    def verify_fast(U, V, W):
        """Vectorized verification"""
        rank = U.shape[1]
        for u in range(dim_u):
            for v in range(dim_v):
                # For this (u,v), the tensor values across w are:
                # T[u,v,w] = 1 if (u,v,w) in target_entries
                # We need sum_r U[u,r]*V[v,r]*W[w,r] for all w
                UV = U[u, :] * V[v, :]  # shape (rank,)
                for w in range(dim_w):
                    val = np.dot(UV, W[w, :])
                    expected = 1.0 if (u, v, w) in target_entries else 0.0
                    if abs(val - expected) > 1e-8:
                        return False
        return True

    # Strategy 1: Try deterministic structured decompositions
    # The 2x4x7 problem can be decomposed using block structure.
    # A(2x4) * B(4x7) = C(2x7)
    # Split B into 4x2 + 4x2 + 4x2 + 4x1 blocks along columns
    # For each 4x2 block: use Strassen-like or schoolbook
    # Actually, let's think about this differently.
    
    # Key insight: 2x4x7 tensor. We can try to find rank by:
    # 1. Using the fact that n=2 is small
    # 2. Trying to compress the 4-dimensional middle dimension
    
    # Approach: Treat the problem as 2x4 matrix times 4x7 matrix.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # u = i*m+j, v = j*p+k, w = k*n+i
    
    # Let's try a direct rank minimization using the structure.
    # For rank r, we need:
    # C[i,k] = sum_{r_idx=0}^{r-1} U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
    #         = sum_j sum_{r_idx} U[i,j,r_idx] * V[j,k,r_idx] * W[k,i,r_idx]
    # (where U[i,j,r] = U[i*m+j, r], etc.)
    
    # This must equal sum_j A[i,j] * B[j,k]
    # So we need: for all i,k: sum_j sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = sum_j A[i,j]*B[j,k]
    # This means: the bilinear form in (A,B) must match.
    # Equivalently: for each r, define U_r (2x4), V_r (4x7), W_r (7x2)
    # Then sum_r (U_r * V_r * W_r elementwise contracted) must give identity on the multiplication map.
    # More precisely: the tensor sum_r U_r \otimes V_r \otimes W_r (with appropriate index mapping)
    # must equal the multiplication tensor.
    
    # Let's try small ranks with randomized optimization.
    # Use the fact that the problem has a lot of structure.
    
    # Try ranks from 30 to 55
    for target_rank in range(30, rank_school):
        if time.time() - start_time > deadline - 30:
            break
        
        # Multiple random attempts
        num_attempts = max(3, int(60 / (target_rank - 29)))
        
        for attempt in range(num_attempts):
            if time.time() - start_time > deadline - 20:
                break
            
            # Initialize with small random values
            scale = 0.5
            U = np.random.uniform(-scale, scale, size=(dim_u, target_rank))
            V = np.random.uniform(-scale, scale, size=(dim_v, target_rank))
            W = np.random.uniform(-scale, scale, size=(dim_w, target_rank))
            
            # Use iterative refinement:
            # Fix two of {U, V, W}, solve for the third linearly.
            # The system is linear in each factor given the other two.
            
            converged = False
            for iteration in range(50):
                if time.time() - start_time > deadline - 10:
                    break
                
                # Solve for W given U, V
                # For each w: sum_r U[u,r]*V[v,r]*W[w,r] = T[u,v,w] for all u,v
                # This is: A_w * W[w,:] = b_w where A_w[u*v_idx, r] = U[u,r]*V[v,r]
                # But we only need to satisfy the equations for (u,v,w) in target_entries
                # and T[u,v,w]=0 for others.
                
                # Actually, we need ALL entries to match, not just nonzero ones.
                # T[u,v,w] is 1 for (u,v,w) in target_entries, 0 otherwise.
                
                new_W = np.zeros((dim_w, target_rank))
                residual = 0.0
                
                for w in range(dim_w):
                    # Build system: for each (u,v), sum_r U[u,r]*V[v,r]*W[w,r] = T[u,v,w]
                    # Number of (u,v) pairs: dim_u * dim_v = 8*28 = 224
                    # Unknowns: target_rank
                    # This is overdetermined for small target_rank.
                    
                    A_w = np.zeros((dim_u * dim_v, target_rank))
                    b_w = np.zeros(dim_u * dim_v)
                    
                    idx = 0
                    for u in range(dim_u):
                        for v in range(dim_v):
                            A_w[idx, :] = U[u, :] * V[v, :]
                            b_w[idx] = 1.0 if (u, v, w) in target_entries else 0.0
                            idx += 1
                    
                    # Least squares
                    try:
                        sol, residuals, rank_val, sv = np.linalg.lstsq(A_w, b_w, rcond=None)
                        new_W[w, :] = sol
                        if residuals.size > 0:
                            residual += residuals[0]
                    except:
                        break
                
                W = new_W
                
                # Check convergence
                if residual < 1e-6:
                    converged = True
                    break
                
                # Similarly solve for U
                new_U = np.zeros((dim_u, target_rank))
                residual_u = 0.0
                for u in range(dim_u):
                    A_u = np.zeros((dim_v * dim_w, target_rank))
                    b_u = np.zeros(dim_v * dim_w)
                    idx = 0
                    for v in range(dim_v):
                        for w in range(dim_w):
                            A_u[idx, :] = V[v, :] * W[w, :]
                            b_u[idx] = 1.0 if (u, v, w) in target_entries else 0.0
                            idx += 1
                    try:
                        sol, residuals, rank_val, sv = np.linalg.lstsq(A_u, b_u, rcond=None)
                        new_U[u, :] = sol
                        if residuals.size > 0:
                            residual_u += residuals[0]
                    except:
                        break
                U = new_U
                
                # Solve for V
                new_V = np.zeros((dim_v, target_rank))
                residual_v = 0.0
                for v in range(dim_v):
                    A_v = np.zeros((dim_u * dim_w, target_rank))
                    b_v = np.zeros(dim_u * dim_w)
                    idx = 0
                    for u in range(dim_u):
                        for w in range(dim_w):
                            A_v[idx, :] = U[u, :] * W[w, :]
                            b_v[idx] = 1.0 if (u, v, w) in target_entries else 0.0
                            idx += 1
                    try:
                        sol, residuals, rank_val, sv = np.linalg.lstsq(A_v, b_v, rcond=None)
                        new_V[v, :] = sol
                        if residuals.size > 0:
                            residual_v += residuals[0]
                    except:
                        break
                V = new_V
            
            if converged:
                # Final verification
                if verify_fast(U, V, W):
                    best_rank = target_rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    break
            elif residual < 1e-3:
                # Close but not quite - try to polish
                # More iterations
                for iteration in range(100):
                    if time.time() - start_time > deadline - 10:
                        break
                    new_W = np.zeros((dim_w, target_rank))
                    residual = 0.0
                    for w in range(dim_w):
                        A_w = np.zeros((dim_u * dim_v, target_rank))
                        b_w = np.zeros(dim_u * dim_v)
                        idx = 0
                        for u in range(dim_u):
                            for v in range(dim_v):
                                A_w[idx, :] = U[u, :] * V[v, :]
                                b_w[idx] = 1.0 if (u, v, w) in target_entries else 0.0
                                idx += 1
                        try:
                            sol, residuals, rank_val, sv = np.linalg.lstsq(A_w, b_w, rcond=None)
                            new_W[w, :] = sol
                            if residuals.size > 0:
                                residual += residuals[0]
                        except:
                            break
                    W = new_W
                    if residual < 1e-10:
                        break
                
                if verify_fast(U, V, W):
                    best_rank = target_rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    break
        
        if best_rank < rank_school:
            break

    # Strategy 2: Deterministic structured approach using block decomposition
    if best_rank == rank_school:
        # Try Kronecker-style construction
        # 2x4x7: think of it as combining 2x2x2 blocks
        # The 4 in the middle can be split into 2+2
        # The 7 in the output can be split into 2+2+2+1
        
        # Let's build a structured decomposition.
        # For the 2x2x2 core, Strassen gives rank 7.
        # But we need to handle the full 2x4x7.
        
        # Alternative: use the fact that we can share computation across blocks.
        
        # Let's try a more direct approach: enumerate possible rank-1 terms
        # and use integer linear programming or greedy selection.
        
        # For now, try a few more random seeds with different scales
        for target_rank in [35, 40, 45]:
            if time.time() - start_time > deadline - 15:
                break
            if target_rank >= best_rank:
                continue
            
            for attempt in range(10):
                if time.time() - start_time > deadline - 10:
                    break
                
                # Use a smarter initialization: start from schoolbook and compress
                U = np.zeros((dim_u, target_rank))
                V = np.zeros((dim_v, target_rank))
                W = np.zeros((dim_w, target_rank))
                
                # Initialize with random projections of schoolbook
                P_U = np.random.randn(rank_school, target_rank)
                P_V = np.random.randn(rank_school, target_rank)
                P_W = np.random.randn(rank_school, target_rank)
                
                # Normalize
                P_U = P_U / np.linalg.norm(P_U, axis=0, keepdims=True)
                P_V = P_V / np.linalg.norm(P_V, axis=0, keepdims=True)
                P_W = P_W / np.linalg.norm(P_W, axis=0, keepdims=True)
                
                U = U_school @ P_U
                V = V_school @ P_V
                W = W_school @ P_W
                
                # Refine
                for iteration in range(80):
                    if time.time() - start_time > deadline - 10:
                        break
                    
                    # Solve for W
                    new_W = np.zeros((dim_w, target_rank))
                    residual = 0.0
                    for w in range(dim_w):
                        A_w = np.zeros((dim_u * dim_v, target_rank))
                        b_w = np.zeros(dim_u * dim_v)
                        idx = 0
                        for u in range(dim_u):
                            for v in range(dim_v):
                                A_w[idx, :] = U[u, :] * V[v, :]
                                b_w[idx] = 1.0 if (u, v, w) in target_entries else 0.0
                                idx += 1
                        try:
                            sol, residuals, rank_val, sv = np.linalg.lstsq(A_w, b_w, rcond=None)
                            new_W[w, :] = sol
                            if residuals.size > 0:
                                residual += residuals[0]
                        except:
                            break
                    W = new_W
                    
                    if residual < 1e-8:
                        break
                    
                    # Solve for U
                    new_U = np.zeros((dim_u, target_rank))
                    for u in range(dim_u):
                        A_u = np.zeros((dim_v * dim_w, target_rank))
                        b_u = np.zeros(dim_v * dim_w)
                        idx = 0
                        for v in range(dim_v):
                            for w in range(dim_w):
                                A_u[idx, :] = V[v, :] * W[w, :]
                                b_u[idx] = 1.0 if (u, v, w) in target_entries else 0.0
                                idx += 1
                        try:
                            sol, residuals, rank_val, sv = np.linalg.lstsq(A_u, b_u, rcond=None)
                            new_U[u, :] = sol
                        except:
                            break
                    U = new_U
                    
                    # Solve for V
                    new_V = np.zeros((dim_v, target_rank))
                    for v in range(dim_v):
                        A_v = np.zeros((dim_u * dim_w, target_rank))
                        b_v = np.zeros(dim_u * dim_w)
                        idx = 0
                        for u in range(dim_u):
                            for w in range(dim_w):
                                A_v[idx, :] = U[u, :] * W[w, :]
                                b_v[idx] = 1.0 if (u, v, w) in target_entries else 0.0
                                idx += 1
                        try:
                            sol, residuals, rank_val, sv = np.linalg.lstsq(A_v, b_v, rcond=None)
                            new_V[v, :] = sol
                        except:
                            break
                    V = new_V
                
                if verify_fast(U, V, W):
                    best_rank = target_rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    break
        
        if best_rank < rank_school:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
