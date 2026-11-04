"""Matrix multiplication 3x3x4 rank minimization."""
import time
import numpy as np
import random

DIMENSIONS = [3, 3, 4]

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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
    return U, V, W

def _verify_fast(U, V, W, n, m, p):
    """
    Fast verification using vectorized NumPy operations.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Target T is all 1s.
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # For each rank component r, compute the outer product contribution
    # T[i,j,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    T = np.zeros((n, m, p), dtype=np.float64)
    
    for r in range(rank):
        u_r = U_np[:, r]  # shape (n*m,)
        v_r = V_np[:, r]  # shape (m*p,)
        w_r = W_np[:, r]  # shape (p*n,)
        
        # Reshape to 3D
        u_3d = u_r.reshape(n, m)  # (n, m)
        v_3d = v_r.reshape(m, p)  # (m, p)
        w_3d = w_r.reshape(p, n)  # (p, n)
        
        # For each (i,j,k), contribution is u_3d[i,j] * v_3d[j,k] * w_3d[k,i]
        # We can compute this as: for each i, sum over j,k
        # T[i,j,k] += u_3d[i,j] * v_3d[j,k] * w_3d[k,i]
        
        # Efficient: create a 3D array where T[i,j,k] gets the contribution
        # This is like a "trilinear" form
        for i in range(n):
            for j in range(m):
                u_val = u_3d[i, j]
                if u_val == 0:
                    continue
                # For this (i,j), we need to add u_val * v_3d[j,k] * w_3d[k,i] for all k
                # v_3d[j,:] has shape (p,), w_3d[:,i] has shape (p,)
                # Element-wise product: v_3d[j,k] * w_3d[k,i] for k in range(p)
                contrib = u_val * (v_3d[j, :] * w_3d[:, i])
                T[i, j, :] += contrib
    
    # Check if T is all 1s (within floating point tolerance)
    if not np.allclose(T, 1.0, atol=1e-10):
        return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """
    Exact verification using Python integers to avoid floating point issues.
    """
    rank = len(U[0])
    
    T = [[[0 for _ in range(p)] for _ in range(m)] for _ in range(n)]
    
    for r in range(rank):
        u_col = [U[i * m + j][r] for i in range(n) for j in range(m)]
        v_col = [V[j * p + k][r] for j in range(m) for k in range(p)]
        w_col = [W[k * n + i][r] for k in range(p) for i in range(n)]
        
        for i in range(n):
            for j in range(m):
                u_val = u_col[i * m + j]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = v_col[j * p + k]
                    if v_val == 0:
                        continue
                    w_val = w_col[k * n + i]
                    T[i][j][k] += u_val * v_val * w_val
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i][j][k] != 1:
                    return False
    return True

def _random_search(rank, iterations, coeff_range, n, m, p, deadline):
    """
    Try to find a valid decomposition of given rank with random coefficients.
    """
    for _ in range(iterations):
        if time.time() > deadline:
            return None
            
        # Generate random U, V, W
        U = [[random.choice(coeff_range) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice(coeff_range) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice(coeff_range) for _ in range(rank)] for _ in range(p * n)]
        
        if _verify_fast(U, V, W, n, m, p):
            # Double-check with exact verification
            if _verify_exact(U, V, W, n, m, p):
                return U, V, W
            
    return None

def _structured_search(rank, n, m, p, deadline):
    """
    Try structured patterns that are more likely to work.
    """
    # Pattern 1: Try to build from known smaller decompositions
    # For 3x3x4, we can think of it as combining 3x3x3 and 3x3x1
    
    # Try patterns where some entries are fixed to create structure
    coeff_options = [-1, 0, 1, 2, -2]
    
    for attempt in range(5000):
        if time.time() > deadline:
            return None
        
        # Create base pattern with some structure
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Randomly decide which rank components to use
        used_components = set()
        
        for r in range(rank):
            # Decide if this rank component is used
            if random.random() < 0.7:
                used_components.add(r)
        
        if len(used_components) == 0:
            continue
            
        for r in used_components:
            # Choose a pattern for this rank component
            pattern_type = random.choice([0, 1, 2, 3])
            
            if pattern_type == 0:
                # Simple: one entry in each
                i = random.randint(0, n - 1)
                j = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
                U[i * m + j][r] = random.choice(coeff_options)
                V[j * p + k][r] = random.choice(coeff_options)
                W[k * n + i][r] = random.choice(coeff_options)
            elif pattern_type == 1:
                # Two entries in one matrix
                i = random.randint(0, n - 1)
                j1 = random.randint(0, m - 1)
                j2 = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
                U[i * m + j1][r] = random.choice(coeff_options)
                if j2 != j1:
                    U[i * m + j2][r] = random.choice(coeff_options)
                V[j1 * p + k][r] = random.choice(coeff_options)
                if j2 != j1:
                    V[j2 * p + k][r] = random.choice(coeff_options)
                W[k * n + i][r] = random.choice(coeff_options)
            elif pattern_type == 2:
                # Sparse random
                num_u = random.randint(1, 3)
                num_v = random.randint(1, 3)
                num_w = random.randint(1, 3)
                
                for _ in range(num_u):
                    idx = random.randint(0, n * m - 1)
                    U[idx][r] = random.choice(coeff_options)
                for _ in range(num_v):
                    idx = random.randint(0, m * p - 1)
                    V[idx][r] = random.choice(coeff_options)
                for _ in range(num_w):
                    idx = random.randint(0, p * n - 1)
                    W[idx][r] = random.choice(coeff_options)
            else:
                # Dense random with small values
                for idx in range(n * m):
                    if random.random() < 0.3:
                        U[idx][r] = random.choice(coeff_options)
                for idx in range(m * p):
                    if random.random() < 0.3:
                        V[idx][r] = random.choice(coeff_options)
                for idx in range(p * n):
                    if random.random() < 0.3:
                        W[idx][r] = random.choice(coeff_options)
        
        if _verify_fast(U, V, W, n, m, p):
            if _verify_exact(U, V, W, n, m, p):
                return U, V, W
        
    return None

def _hybrid_search(rank, n, m, p, deadline):
    """
    Combine random and structured search.
    """
    iterations = 3000
    
    for i in range(iterations):
        if time.time() > deadline:
            return None
        
        # Alternate between random and structured
        if i % 3 == 0:
            # Pure random
            coeff_range = [-1, 0, 1]
            U = [[random.choice(coeff_range) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.choice(coeff_range) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.choice(coeff_range) for _ in range(rank)] for _ in range(p * n)]
        elif i % 3 == 1:
            # Random with larger range
            coeff_range = [-2, -1, 0, 1, 2]
            U = [[random.choice(coeff_range) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.choice(coeff_range) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.choice(coeff_range) for _ in range(rank)] for _ in range(p * n)]
        else:
            # Structured pattern
            coeff_options = [-1, 0, 1]
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            for r in range(rank):
                if random.random() < 0.8:
                    # Simple pattern
                    i = random.randint(0, n - 1)
                    j = random.randint(0, m - 1)
                    k = random.randint(0, p - 1)
                    U[i * m + j][r] = random.choice(coeff_options)
                    V[j * p + k][r] = random.choice(coeff_options)
                    W[k * n + i][r] = random.choice(coeff_options)
        
        if _verify_fast(U, V, W, n, m, p):
            if _verify_exact(U, V, W, n, m, p):
                return U, V, W
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook()
    
    # Try to find a lower rank decomposition
    # Known lower bound is around 27 for 3x3x4
    
    # First try structured search for lower ranks
    for target_rank in [27, 28, 29, 30, 31, 32, 33, 34, 35]:
        if time.time() > deadline - 15:
            break
        
        # Try structured search first (more targeted)
        result = _structured_search(target_rank, n, m, p, deadline - 10)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        # Then try hybrid search
        if time.time() > deadline - 10:
            break
        result = _hybrid_search(target_rank, n, m, p, deadline - 5)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        # Finally try pure random
        if time.time() > deadline - 5:
            break
        result = _random_search(target_rank, 2000, [-1, 0, 1], n, m, p, deadline)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
