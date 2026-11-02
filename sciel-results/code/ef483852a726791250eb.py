"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    # Build the rank-21 Strassen decomposition (3 blocks x 7 steps)
    rank = 21
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Strassen coefficients for 2x2 multiplication
    # u: [A00, A01, A10, A11]
    # v: [B00, B01, B10, B11]
    # w: [C00, C01, C10, C11]
    strassen_steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]
    
    # Helper to map local indices to global indices
    def map_a(local_idx):
        return local_idx
        
    def map_b(local_idx, g):
        k0 = 2 * g
        k1 = 2 * g + 1
        if local_idx == 0:
            return 0 * 6 + k0
        elif local_idx == 1:
            return 0 * 6 + k1
        elif local_idx == 2:
            return 1 * 6 + k0
        else:
            return 1 * 6 + k1

    def map_c(local_idx, g):
        k0 = 2 * g
        k1 = 2 * g + 1
        if local_idx == 0:
            return k0 * 2 + 0
        elif local_idx == 1:
            return k1 * 2 + 0
        elif local_idx == 2:
            return k0 * 2 + 1
        else:
            return k1 * 2 + 1

    # Fill U, V, W
    for g in range(3):
        for s in range(7):
            r = 7 * g + s
            step = strassen_steps[s]
            
            for local_idx in range(4):
                u_val = step['u'][local_idx]
                if u_val != 0:
                    a_global = map_a(local_idx)
                    U[a_global, r] = u_val
                    
                v_val = step['v'][local_idx]
                if v_val != 0:
                    b_global = map_b(local_idx, g)
                    V[b_global, r] = v_val
                    
                w_val = step['w'][local_idx]
                if w_val != 0:
                    c_global = map_c(local_idx, g)
                    W[c_global, r] = w_val

    # Verify the decomposition
    T_check = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Expected tensor
    T_exp = np.zeros((4, 12, 12))
    for i in range(2):
        for j in range(2):
            for k in range(6):
                a = i * 2 + j
                b = j * 6 + k
                c = k * 2 + i
                T_exp[a, b, c] = 1.0
                
    if not np.allclose(T_check, T_exp, atol=1e-10):
        # Fallback to schoolbook if construction failed
        rank_school = 24
        U_s = np.zeros((4, rank_school))
        V_s = np.zeros((12, rank_school))
        W_s = np.zeros((12, rank_school))
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(6):
                    a = i * 2 + j
                    b = j * 6 + k
                    c = k * 2 + i
                    U_s[a, col] = 1
                    V_s[b, col] = 1
                    W_s[c, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    # Check if we have time to attempt lower rank
    elapsed = time.time() - start_time
    if elapsed > 100:  # Not enough time for search
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a lower rank decomposition (rank < 21)
    # This is a hard problem, so we'll do a limited search
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = 21
    
    # Attempt rank 20, 19, etc. with limited iterations
    for target_rank in [20, 19, 18]:
        if time.time() > deadline - 10:
            break
        
        # Use a simple random initialization + gradient-free approach
        # This is unlikely to succeed but we try
        success = False
        max_iters = 5000
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
                
            U_rand = np.random.randn(nm, target_rank)
            V_rand = np.random.randn(mp, target_rank)
            W_rand = np.random.randn(pn, target_rank)
            
            # Simple iterative refinement
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                    
                # Compute current tensor
                T_cur = np.einsum('ar,br,cr->abc', U_rand, V_rand, W_rand)
                diff = T_cur - T_exp
                if np.max(np.abs(diff)) < 1e-8:
                    success = True
                    break
                    
                # Simple gradient step (this is very naive)
                lr = 0.01
                dU = np.einsum('abcr,br->ar', diff, W_rand)
                dV = np.einsum('abcr,ar->br', diff, U_rand)
                dW = np.einsum('abcr,ar->cr', diff, U_rand)
                
                U_rand -= lr * dU
                V_rand -= lr * dV
                W_rand -= lr * dW
                
                if it % 1000 == 0:
                    pass  # No print to save time
            
            if success:
                # Round to nearest half-integer
                U_rounded = np.round(U_rand * 2) / 2
                V_rounded = np.round(V_rand * 2) / 2
                W_rounded = np.round(W_rand * 2) / 2
                
                # Verify rounded solution
                T_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                if np.allclose(T_rounded, T_exp, atol=1e-8):
                    best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                    best_rank = target_rank
                    break
    
    return best_result
# EVOLVE-BLOCK-END
