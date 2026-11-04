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
    
    for restart in range(1000):
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
            
        # Local search
        # We want to minimize E = ||S - T||^2
        # S = sum_r U_r V_r W_r
        # Change U[a, r] by delta:
        # S_new = S + delta * (e_a * V_r * W_r)
        # E_new = || S - T + delta * v_r w_r ||^2
        # Let D = S - T.
        # E_new = || D + delta * v_r w_r ||^2
        # = ||D||^2 + 2 delta <D, v_r w_r> + delta^2 ||v_r w_r||^2
        # = E + 2 delta dot(D, v_r w_r) + delta^2 sum(v_r^2 w_r^2)
        
        D = S - T
        max_iters = 2000
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            if E < 1e-9:
                return U, V, W
                
            # Pick random move
            which = random.randint(0, 2)
            if which == 0:
                a, r = random.randint(0, n_a-1), random.randint(0, rank-1)
                old_val = U[a, r]
                vec = np.einsum('b,c->bc', V[:,r], W[:,r])
                # vec is shape (n_b, n_c)
                # We need dot(D, e_a * vec) = sum_{b,c} D[a,b,c] * vec[b,c]
                dot_D = np.sum(D[a, :, :] * vec)
                norm_sq = np.sum(vec**2)
            elif which == 1:
                b, r = random.randint(0, n_b-1), random.randint(0, rank-1)
                old_val = V[b, r]
                vec = np.einsum('a,c->ac', U[:,r], W[:,r])
                dot_D = np.sum(D[:, b, :] * vec)
                norm_sq = np.sum(vec**2)
            else:
                c, r = random.randint(0, n_c-1), random.randint(0, rank-1)
                old_val = W[c, r]
                vec = np.einsum('a,b->ab', U[:,r], V[:,r])
                dot_D = np.sum(D[:, :, c] * vec)
                norm_sq = np.sum(vec**2)
                
            # Try changing to a different value
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val:
                new_val = (new_val + 1) % 3 - 1
            
            delta = new_val - old_val
            if delta == 0:
                continue
                
            # Calculate E_new
            # E_new = E + 2 * delta * dot_D + delta^2 * norm_sq
            E_new = E + 2 * delta * dot_D + (delta**2) * norm_sq
            
            if E_new < E:
                # Accept
                if which == 0:
                    U[a, r] = new_val
                    D[a, :, :] += delta * vec
                elif which == 1:
                    V[b, r] = new_val
                    D[:, b, :] += delta * vec
                else:
                    W[c, r] = new_val
                    D[:, :, c] += delta * vec
                E = E_new
            # else: reject (stay same)
            
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
