"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def _make_rank17():
    """Return the known rank-17 decomposition."""
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    terms = [
        ([0, 4], [1, 1], [1, 1], [1, 1], [0, 3], [1, 1]),
        ([3, 4], [1, 1], [0], [1], [1, 3], [1, -1]),
        ([0], [1], [1, 4], [1, -1], [2, 3], [1, 1]),
        ([4], [1], [3, 0], [1, -1], [0, 1], [1, 1]),
        ([0, 1], [1, 1], [4], [1], [0, 2], [-1, 1]),
        ([3, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        ([1, 4], [1, -1], [3, 4], [1, 1], [0], [1]),
        ([0], [1], [2], [1], [4], [1]),
        ([1], [1], [5], [1], [4], [1]),
        ([3], [1], [2], [1], [5], [1]),
        ([4], [1], [5], [1], [5], [1]),
        ([2], [1], [6], [1], [0], [1]),
        ([2], [1], [7], [1], [2], [1]),
        ([2], [1], [8], [1], [4], [1]),
        ([5], [1], [6], [1], [1], [1]),
        ([5], [1], [7], [1], [3], [1]),
        ([5], [1], [8], [1], [5], [1]),
    ]
    
    for r, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms):
        for a, val in zip(u_idx, u_val):
            U[a, r] = val
        for b, val in zip(v_idx, v_val):
            V[b, r] = val
        for c, val in zip(w_idx, w_val):
            W[c, r] = val
    return U, V, W

def _try_rank16_fast(deadline):
    """
    Fast local search for rank 16 using incremental error updates.
    """
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    T = _build_tensor(n, m, p)
    rank = 16
    
    # Precompute T norm squared
    T_sq = np.sum(T**2)
    
    best_E = float('inf')
    best_result = None
    
    for restart in range(2000):
        if time.time() > deadline - 5:
            break
        
        # Initialize from rank 17, drop a random column
        U, V, W = _make_rank17()
        drop_col = random.randint(0, 16)
        U = np.delete(U, drop_col, axis=1)
        V = np.delete(V, drop_col, axis=1)
        W = np.delete(W, drop_col, axis=1)
        
        # Compute initial S and E
        S = np.zeros((n_a, n_b, n_c))
        for r in range(rank):
            S += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        E = np.sum((S - T)**2)
        
        if E < 1e-9:
            return U, V, W
        
        if E < best_E:
            best_E = E
            best_result = (U.copy(), V.copy(), W.copy())
            
        # Local search with multiple move types
        D = S - T
        
        # Precompute pairwise products for each rank component
        # For efficiency, we'll compute vectors on-the-fly
        
        max_iters = 5000
        no_improve_count = 0
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            if E < 1e-9:
                return U, V, W
            
            # Try multiple move strategies
            move_type = random.random()
            
            if move_type < 0.4:
                # Single element change to {-1, 0, 1}
                which = random.randint(0, 2)
                if which == 0:
                    a, r = random.randint(0, n_a-1), random.randint(0, rank-1)
                    old_val = U[a, r]
                    vec = np.einsum('b,c->bc', V[:,r], W[:,r])
                    dot_D = np.sum(D[a, :, :] * vec)
                    norm_sq = np.sum(vec**2)
                    new_val = random.choice([-1, 0, 1])
                    if new_val == old_val:
                        new_val = (new_val + 1) % 3 - 1
                    delta = new_val - old_val
                    if delta == 0:
                        continue
                    E_new = E + 2 * delta * dot_D + (delta**2) * norm_sq
                    if E_new < E:
                        U[a, r] = new_val
                        D[a, :, :] += delta * vec
                        E = E_new
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                elif which == 1:
                    b, r = random.randint(0, n_b-1), random.randint(0, rank-1)
                    old_val = V[b, r]
                    vec = np.einsum('a,c->ac', U[:,r], W[:,r])
                    dot_D = np.sum(D[:, b, :] * vec)
                    norm_sq = np.sum(vec**2)
                    new_val = random.choice([-1, 0, 1])
                    if new_val == old_val:
                        new_val = (new_val + 1) % 3 - 1
                    delta = new_val - old_val
                    if delta == 0:
                        continue
                    E_new = E + 2 * delta * dot_D + (delta**2) * norm_sq
                    if E_new < E:
                        V[b, r] = new_val
                        D[:, b, :] += delta * vec
                        E = E_new
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    c, r = random.randint(0, n_c-1), random.randint(0, rank-1)
                    old_val = W[c, r]
                    vec = np.einsum('a,b->ab', U[:,r], V[:,r])
                    dot_D = np.sum(D[:, :, c] * vec)
                    norm_sq = np.sum(vec**2)
                    new_val = random.choice([-1, 0, 1])
                    if new_val == old_val:
                        new_val = (new_val + 1) % 3 - 1
                    delta = new_val - old_val
                    if delta == 0:
                        continue
                    E_new = E + 2 * delta * dot_D + (delta**2) * norm_sq
                    if E_new < E:
                        W[c, r] = new_val
                        D[:, :, c] += delta * vec
                        E = E_new
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
            elif move_type < 0.7:
                # Try changing two elements in the same rank component
                r = random.randint(0, rank-1)
                which1 = random.randint(0, 2)
                which2 = random.randint(0, 2)
                if which1 == which2:
                    which2 = (which1 + 1) % 3
                
                if which1 == 0:
                    a1 = random.randint(0, n_a-1)
                    old1 = U[a1, r]
                    new1 = random.choice([-1, 0, 1])
                    if new1 == old1:
                        new1 = (new1 + 1) % 3 - 1
                    delta1 = new1 - old1
                elif which1 == 1:
                    b1 = random.randint(0, n_b-1)
                    old1 = V[b1, r]
                    new1 = random.choice([-1, 0, 1])
                    if new1 == old1:
                        new1 = (new1 + 1) % 3 - 1
                    delta1 = new1 - old1
                else:
                    c1 = random.randint(0, n_c-1)
                    old1 = W[c1, r]
                    new1 = random.choice([-1, 0, 1])
                    if new1 == old1:
                        new1 = (new1 + 1) % 3 - 1
                    delta1 = new1 - old1
                
                if which2 == 0:
                    a2 = random.randint(0, n_a-1)
                    old2 = U[a2, r]
                    new2 = random.choice([-1, 0, 1])
                    if new2 == old2:
                        new2 = (new2 + 1) % 3 - 1
                    delta2 = new2 - old2
                elif which2 == 1:
                    b2 = random.randint(0, n_b-1)
                    old2 = V[b2, r]
                    new2 = random.choice([-1, 0, 1])
                    if new2 == old2:
                        new2 = (new2 + 1) % 3 - 1
                    delta2 = new2 - old2
                else:
                    c2 = random.randint(0, n_c-1)
                    old2 = W[c2, r]
                    new2 = random.choice([-1, 0, 1])
                    if new2 == old2:
                        new2 = (new2 + 1) % 3 - 1
                    delta2 = new2 - old2
                
                if delta1 == 0 and delta2 == 0:
                    continue
                
                # Compute the change in S
                # This is complex, so we'll compute directly
                S_old = np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                U_new = U[:,r].copy()
                V_new = V[:,r].copy()
                W_new = W[:,r].copy()
                if which1 == 0:
                    U_new[a1] = new1
                elif which1 == 1:
                    V_new[b1] = new1
                else:
                    W_new[c1] = new1
                if which2 == 0:
                    U_new[a2] = new2
                elif which2 == 1:
                    V_new[b2] = new2
                else:
                    W_new[c2] = new2
                S_new = np.einsum('a,b,c->abc', U_new, V_new, W_new)
                delta_S = S_new - S_old
                
                # Compute E_new
                D_new = D + delta_S
                E_new = np.sum(D_new**2)
                
                if E_new < E:
                    # Apply changes
                    if which1 == 0:
                        U[a1, r] = new1
                    elif which1 == 1:
                        V[b1, r] = new1
                    else:
                        W[c1, r] = new1
                    if which2 == 0:
                        U[a2, r] = new2
                    elif which2 == 1:
                        V[b2, r] = new2
                    else:
                        W[c2, r] = new2
                    D += delta_S
                    E = E_new
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Try a random perturbation of a whole column
                r = random.randint(0, rank-1)
                which = random.randint(0, 2)
                if which == 0:
                    # Perturb U[:,r]
                    U_old = U[:,r].copy()
                    for a in range(n_a):
                        if random.random() < 0.3:
                            U[a, r] = random.choice([-1, 0, 1])
                    # Check if any change
                    if np.array_equal(U[:,r], U_old):
                        continue
                    # Recompute S and E
                    S = np.zeros((n_a, n_b, n_c))
                    for rr in range(rank):
                        S += np.einsum('a,b,c->abc', U[:,rr], V[:,rr], W[:,rr])
                    E_new = np.sum((S - T)**2)
                    if E_new < E:
                        D = S - T
                        E = E_new
                        no_improve_count = 0
                    else:
                        U[:,r] = U_old
                        no_improve_count += 1
                elif which == 1:
                    # Perturb V[:,r]
                    V_old = V[:,r].copy()
                    for b in range(n_b):
                        if random.random() < 0.3:
                            V[b, r] = random.choice([-1, 0, 1])
                    if np.array_equal(V[:,r], V_old):
                        continue
                    S = np.zeros((n_a, n_b, n_c))
                    for rr in range(rank):
                        S += np.einsum('a,b,c->abc', U[:,rr], V[:,rr], W[:,rr])
                    E_new = np.sum((S - T)**2)
                    if E_new < E:
                        D = S - T
                        E = E_new
                        no_improve_count = 0
                    else:
                        V[:,r] = V_old
                        no_improve_count += 1
                else:
                    # Perturb W[:,r]
                    W_old = W[:,r].copy()
                    for c in range(n_c):
                        if random.random() < 0.3:
                            W[c, r] = random.choice([-1, 0, 1])
                    if np.array_equal(W[:,r], W_old):
                        continue
                    S = np.zeros((n_a, n_b, n_c))
                    for rr in range(rank):
                        S += np.einsum('a,b,c->abc', U[:,rr], V[:,rr], W[:,rr])
                    E_new = np.sum((S - T)**2)
                    if E_new < E:
                        D = S - T
                        E = E_new
                        no_improve_count = 0
                    else:
                        W[:,r] = W_old
                        no_improve_count += 1
            
            if E < best_E:
                best_E = E
                best_result = (U.copy(), V.copy(), W.copy())
                
            # If stuck for too long, break early
            if no_improve_count > 500:
                break
            
    # Polish pass: try to find exact solution from best result
    if best_result is not None and time.time() < deadline - 2:
        U_p, V_p, W_p = best_result
        S_p = np.zeros((n_a, n_b, n_c))
        for r in range(rank):
            S_p += np.einsum('a,b,c->abc', U_p[:,r], V_p[:,r], W_p[:,r])
        E_p = np.sum((S_p - T)**2)
        if E_p < 1e-9:
            return U_p, V_p, W_p
        
        # Final polish: try single-element changes with stricter criteria
        D_p = S_p - T
        for it in range(2000):
            if time.time() > deadline - 1:
                break
            if E_p < 1e-9:
                return U_p, V_p, W_p
            
            which = random.randint(0, 2)
            if which == 0:
                a, r = random.randint(0, n_a-1), random.randint(0, rank-1)
                old_val = U_p[a, r]
                vec = np.einsum('b,c->bc', V_p[:,r], W_p[:,r])
                dot_D = np.sum(D_p[a, :, :] * vec)
                norm_sq = np.sum(vec**2)
                new_val = random.choice([-1, 0, 1])
                if new_val == old_val:
                    new_val = (new_val + 1) % 3 - 1
                delta = new_val - old_val
                if delta == 0:
                    continue
                E_new = E_p + 2 * delta * dot_D + (delta**2) * norm_sq
                if E_new < E_p:
                    U_p[a, r] = new_val
                    D_p[a, :, :] += delta * vec
                    E_p = E_new
            elif which == 1:
                b, r = random.randint(0, n_b-1), random.randint(0, rank-1)
                old_val = V_p[b, r]
                vec = np.einsum('a,c->ac', U_p[:,r], W_p[:,r])
                dot_D = np.sum(D_p[:, b, :] * vec)
                norm_sq = np.sum(vec**2)
                new_val = random.choice([-1, 0, 1])
                if new_val == old_val:
                    new_val = (new_val + 1) % 3 - 1
                delta = new_val - old_val
                if delta == 0:
                    continue
                E_new = E_p + 2 * delta * dot_D + (delta**2) * norm_sq
                if E_new < E_p:
                    V_p[b, r] = new_val
                    D_p[:, b, :] += delta * vec
                    E_p = E_new
            else:
                c, r = random.randint(0, n_c-1), random.randint(0, rank-1)
                old_val = W_p[c, r]
                vec = np.einsum('a,b->ab', U_p[:,r], V_p[:,r])
                dot_D = np.sum(D_p[:, :, c] * vec)
                norm_sq = np.sum(vec**2)
                new_val = random.choice([-1, 0, 1])
                if new_val == old_val:
                    new_val = (new_val + 1) % 3 - 1
                delta = new_val - old_val
                if delta == 0:
                    continue
                E_new = E_p + 2 * delta * dot_D + (delta**2) * norm_sq
                if E_new < E_p:
                    W_p[c, r] = new_val
                    D_p[:, :, c] += delta * vec
                    E_p = E_new
        
        if E_p < 1e-9:
            return U_p, V_p, W_p
    
    if best_result is not None:
        return best_result
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    
    T = _build_tensor(n, m, p)
    
    # Try to find rank 16 with fast local search
    result_16 = _try_rank16_fast(deadline)
    if result_16 is not None:
        U, V, W = result_16
        if _verify_decomposition(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 17
    U, V, W = _make_rank17()
    if _verify_decomposition(U, V, W, T):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Final fallback: schoolbook
    rank_fb = n * m * p
    U_fb = np.zeros((n_a, rank_fb))
    V_fb = np.zeros((n_b, rank_fb))
    W_fb = np.zeros((n_c, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
