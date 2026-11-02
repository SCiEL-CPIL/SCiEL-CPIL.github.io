import numpy as np
import time
import random

DIMENSIONS = [2, 3, 3]

def construct():
        n, m, p = DIMENSIONS
        start_time = time.time()
        timeout = 110
        
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        fallback = {"U": U_fb, "V": V_fb, "W": W_fb}

        # Target Rank 15
        R = 15
        nm, mp, pn = n * m, m * p, p * n
        
        # Target Tensor
        T = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0

        # Search for Rank 15
        # We use a randomized local search.
        # We restrict coefficients to a small set to help convergence to exact solutions.
        # Candidates for coefficients: {-1, -0.5, 0, 0.5, 1}
        
        best_rank = rank_fb
        best_solution = fallback
        
        rng = np.random.default_rng(42)
        
        def calculate_error(U, V, W):
            # U: (nm, R), V: (mp, R), W: (pn, R)
            # T_calc[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            # Efficient calculation using einsum
            # T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            # This might be slow for many iterations.
            # Alternative: iterate over r?
            # Or use matrix multiplication tricks?
            # For R=15, nm=6, mp=9, pn=6.
            # einsum is fine.
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            return np.linalg.norm(T_calc - T)

        def random_vector(size, density=0.3):
            v = np.zeros(size)
            for idx in range(size):
                if rng.random() < density:
                    v[idx] = rng.choice([-1, -0.5, 0.5, 1])
            return v

        # Try multiple restarts
        max_restarts = 100
        max_iters = 500
        
        for restart in range(max_restarts):
            if time.time() - start_time > timeout:
                break
                
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            W = np.zeros((pn, R))
            
            for r in range(R):
                U[:, r] = random_vector(nm, density=0.5)
                V[:, r] = random_vector(mp, density=0.5)
                W[:, r] = random_vector(pn, density=0.5)
            
            current_err = calculate_error(U, V, W)
            
            for it in range(max_iters):
                if time.time() - start_time > timeout:
                    break
                    
                if current_err < 1e-9:
                    best_rank = R
                    # Convert to lists
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    best_solution = {"U": U_list, "V": V_list, "W": W_list}
                    break
                
                # Pick a random rank r
                r = rng.integers(0, R)
                
                # Perturb one of U[:, r], V[:, r], W[:, r]
                choice = rng.integers(0, 3)
                if choice == 0:
                    idx = rng.integers(0, nm)
                    old_val = U[idx, r]
                    U[idx, r] = rng.choice([-1, -0.5, 0, 0.5, 1])
                    if U[idx, r] == old_val:
                        continue
                    new_err = calculate_error(U, V, W)
                    if new_err < current_err:
                        current_err = new_err
                    else:
                        U[idx, r] = old_val
                elif choice == 1:
                    idx = rng.integers(0, mp)
                    old_val = V[idx, r]
                    V[idx, r] = rng.choice([-1, -0.5, 0, 0.5, 1])
                    if V[idx, r] == old_val:
                        continue
                    new_err = calculate_error(U, V, W)
                    if new_err < current_err:
                        current_err = new_err
                    else:
                        V[idx, r] = old_val
                else:
                    idx = rng.integers(0, pn)
                    old_val = W[idx, r]
                    W[idx, r] = rng.choice([-1, -0.5, 0, 0.5, 1])
                    if W[idx, r] == old_val:
                        continue
                    new_err = calculate_error(U, V, W)
                    if new_err < current_err:
                        current_err = new_err
                    else:
                        W[idx, r] = old_val
            
            if best_rank == R:
                break

        if best_rank < rank_fb:
            return best_solution
        else:
            return fallback

    # EVOLVE-BLOCK-END
