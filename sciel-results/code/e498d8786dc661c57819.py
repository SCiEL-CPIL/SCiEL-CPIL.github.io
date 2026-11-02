import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _build_schoolbook(n, m, p):
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    return U, V, W

def _verify_full(U, V, W, n, m, p, tol=1e-9):
    """Full verification of the tensor decomposition."""
    R = U.shape[1]
    a_dim = n * m
    b_dim = m * p
    c_dim = p * n
    
    # Precompute all non-zero (i,j,k) triples
    nonzero = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero.append((a, b, c))
    
    # Check all non-zero entries
    for a, b, c in nonzero:
        s = np.sum(U[a, :] * V[b, :] * W[c, :])
        if abs(s - 1.0) > tol:
            return False
    
    # Check all zero entries (full check for small dimensions)
    # Total entries = a_dim * b_dim * c_dim = 9 * 18 * 18 = 2916
    # Non-zero = 54, so zero entries = 2862
    # This is feasible to check fully
    nonzero_set = set(nonzero)
    for a in range(a_dim):
        for b in range(b_dim):
            for c in range(c_dim):
                if (a, b, c) not in nonzero_set:
                    s = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(s) > tol:
                        return False
    
    return True

def _verify_sampled(U, V, W, n, m, p, tol=1e-9, seed=42, max_checks=5000):
    """Sampled verification for speed during search."""
    R = U.shape[1]
    rng = np.random.RandomState(seed)
    a_dim = n * m
    b_dim = m * p
    c_dim = p * n
    
    # Check all non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(s - 1.0) > tol:
                    return False
    
    # Sample zero entries
    checked = 0
    while checked < max_checks:
        i = rng.randint(0, n)
        j = rng.randint(0, m)
        k = rng.randint(0, p)
        a = i * m + j
        
        j2 = rng.randint(0, m)
        k2 = rng.randint(0, p)
        b = j2 * p + k2
        
        i2 = rng.randint(0, n)
        k3 = rng.randint(0, p)
        c = k3 * n + i2
        
        if j2 == j and i2 == i and k3 == k:
            continue
            
        s = np.sum(U[a, :] * V[b, :] * W[c, :])
        if abs(s) > tol:
            return False
        checked += 1
    
    return True

def _compute_loss(U, V, W, n, m, p):
    """Compute the loss for the tensor decomposition."""
    R = U.shape[1]
    total = 0.0
    
    # Non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.sum(U[a, :] * V[b, :] * W[c, :])
                total += (s - 1.0) ** 2
    
    # Sample zero entries
    rng = np.random.RandomState(0)
    for _ in range(300):
        i = rng.randint(0, n)
        j = rng.randint(0, m)
        k = rng.randint(0, p)
        a = i * m + j
        j2 = rng.randint(0, m)
        k2 = rng.randint(0, p)
        b = j2 * p + k2
        i2 = rng.randint(0, n)
        k3 = rng.randint(0, p)
        c = k3 * n + i2
        if j2 == j and i2 == i and k3 == k:
            continue
        s = np.sum(U[a, :] * V[b, :] * W[c, :])
        total += s ** 2
    
    return total

def _try_rank_reduction(n, m, p, target_rank, deadline, seed=0):
    """Attempt to find a rank-target_rank decomposition using numerical optimization."""
    rng = np.random.RandomState(seed)
    R = n * m * p
    a_dim = n * m
    b_dim = m * p
    c_dim = p * n
    
    # Strategy: Start from schoolbook and try to merge columns
    # The schoolbook has R columns. We want target_rank columns.
    # Idea: Initialize U, V, W with random values scaled appropriately,
    # then use gradient descent to minimize the loss.
    
    # Better initialization: use the schoolbook structure but with random linear combinations
    # to create dependencies
    
    # Approach: Random initialization with scaling
    scale = 1.0 / np.sqrt(target_rank)
    U = rng.randn(a_dim, target_rank) * scale
    V = rng.randn(b_dim, target_rank) * scale
    W = rng.randn(c_dim, target_rank) * scale
    
    # Also try a structured initialization based on schoolbook
    # Take random linear combinations of schoolbook columns
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    
    # Create random mixing matrix
    M = rng.randn(R, target_rank)
    U_struct = U_school @ M
    V_struct = V_school @ M
    W_struct = W_school @ M
    
    # Try both initializations
    best_loss = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for init_U, init_V, init_W in [(U, V, W), (U_struct, V_struct, W_struct)]:
        cur_U = init_U.copy()
        cur_V = init_V.copy()
        cur_W = init_W.copy()
        
        # Gradient descent with adaptive learning rate
        lr = 0.01
        patience = 50
        no_improve = 0
        cur_loss = _compute_loss(cur_U, cur_V, cur_W, n, m, p)
        
        for step in range(500):
            if time.time() > deadline - 5:
                break
            
            # Compute gradients using finite differences (expensive but accurate)
            # Use a more efficient approach: analytical gradients
            # d/dU[a,r] = sum over (b,c) of V[b,r]*W[c,r]*T[a,b,c]
            # But T is sparse, so we only need to consider non-zero and sampled zero entries
            
            # Let's use a simpler approach: coordinate descent
            # Update one element at a time
            
            improved = False
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                
                # Choose which matrix to update
                mat_choice = rng.randint(0, 3)
                
                if mat_choice == 0:
                    row = rng.randint(0, a_dim)
                    col = rng.randint(0, target_rank)
                    old = cur_U[row, col]
                    
                    # Try a few step sizes
                    for step_size in [lr, lr * 0.5, lr * 2, -lr]:
                        cur_U[row, col] = old + step_size
                        new_loss = _compute_loss(cur_U, cur_V, cur_W, n, m, p)
                        if new_loss < cur_loss:
                            cur_loss = new_loss
                            improved = True
                            break
                    cur_U[row, col] = old
                    
                elif mat_choice == 1:
                    row = rng.randint(0, b_dim)
                    col = rng.randint(0, target_rank)
                    old = cur_V[row, col]
                    
                    for step_size in [lr, lr * 0.5, lr * 2, -lr]:
                        cur_V[row, col] = old + step_size
                        new_loss = _compute_loss(cur_U, cur_V, cur_W, n, m, p)
                        if new_loss < cur_loss:
                            cur_loss = new_loss
                            improved = True
                            break
                    cur_V[row, col] = old
                    
                else:
                    row = rng.randint(0, c_dim)
                    col = rng.randint(0, target_rank)
                    old = cur_W[row, col]
                    
                    for step_size in [lr, lr * 0.5, lr * 2, -lr]:
                        cur_W[row, col] = old + step_size
                        new_loss = _compute_loss(cur_U, cur_V, cur_W, n, m, p)
                        if new_loss < cur_loss:
                            cur_loss = new_loss
                            improved = True
                            break
                    cur_W[row, col] = old
            
            if improved:
                no_improve = 0
                # Decay learning rate slowly
                lr = max(0.001, lr * 0.99)
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_U, best_V, best_W = cur_U.copy(), cur_V.copy(), cur_W.copy()
    
    # Check if we found a valid solution
    if best_loss < 1e-6 and best_U is not None:
        if _verify_sampled(best_U, best_V, best_W, n, m, p, tol=1e-6):
            return best_U, best_V, best_W
    
    return None

def _try_block_structure(n, m, p, deadline):
    """Try to exploit block structure: split 3x3*3x6 into two 3x3*3x3 problems.
    
    If we can find a rank-r decomposition for 3x3*3x3, we can get rank 2r for 3x3*3x6.
    The schoolbook for 3x3*3x3 is rank 27, so 2*27=54 (same as direct schoolbook).
    But if we can reduce the 3x3*3x3 rank, we win.
    """
    # For now, this is a placeholder for future work
    # The key insight: C = A * [B1 | B2] = [A*B1 | A*B2]
    # Each A*B_i is a 3x3*3x3 multiplication
    # If we have a rank-r decomposition for 3x3*3x3, total rank is 2r
    
    # Known results: minimal rank for 3x3*3x3 is between 19 and 23
    # We don't have a known exact decomposition, so we skip this for now
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard: return best found before wall-clock limit
    deadline = time.time() + 120 - 10  # 110s safety margin
    
    # Schoolbook baseline: rank 54
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    
    best_rank = n * m * p
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try a few candidate ranks
    # Start with ranks just below schoolbook and work down
    candidate_ranks = [53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 30:
            break
        
        result = _try_rank_reduction(n, m, p, target_rank, deadline, seed=target_rank)
        if result is not None:
            U_try, V_try, W_try = result
            # Full verification before accepting
            if _verify_full(U_try, V_try, W_try, n, m, p, tol=1e-8):
                best_rank = target_rank
                best_U, best_V, best_W = U_try, V_try, W_try
                break
    
    # Return the best result found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
