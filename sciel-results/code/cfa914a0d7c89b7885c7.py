import numpy as np
import time
import random

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_sb = [[0] * rank_school for _ in range(n * m)]
    V_sb = [[0] * rank_school for _ in range(m * p)]
    W_sb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
        
    best_U, best_V, best_W = U_sb, V_sb, V_sb
    best_rank = rank_school

    # Target ranks to try, starting from theoretical lower bound
    # Lower bound is max(n*m, m*p, p*n) = 6
    # Known rank for 2x2x3 is 10
    target_ranks = [10, 11, 12]

    def get_target_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T_target = get_target_tensor()

    def compute_error(U, V, W):
        """
        Computes the sum of squared errors between the constructed tensor and target.
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
        """
        rank = U.shape[1]
        T_prime = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            u_r = U[:, r]
            v_r = V[:, r]
            w_r = W[:, r]
            T_prime += np.einsum('i,j,k->ijk', u_r, v_r, w_r)
        diff = T_prime - T_target
        return np.sum(diff**2)

    def try_find_decomposition_discrete(target_rank):
        if time.time() > deadline - 10.0:
            return None

        # Define allowed values: integers and half-integers in a reasonable range
        # Matrix multiplication tensors often involve small integers.
        # Let's try values in {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2}
        allowed_values = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))

        # Initialize with zeros
        current_error = compute_error(U, V, W)
        
        best_error = current_error
        best_U_copy = U.copy()
        best_V_copy = V.copy()
        best_W_copy = W.copy()

        # Simulated Annealing Parameters
        T_init = 10.0
        T_end = 0.01
        alpha = 0.999 # Cooling rate
        max_iters = 50000
        
        # Time check inside loop
        iters_done = 0
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Pick a random matrix and a random element to change
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                mat = U
                rows = n*m
            elif mat_idx == 1:
                mat = V
                rows = m*p
            else:
                mat = W
                rows = p*n
            
            r_idx = random.randint(0, rows - 1)
            c_idx = random.randint(0, target_rank - 1)
            
            old_val = mat[r_idx, c_idx]
            
            # Choose a new value different from old
            possible_vals = [v for v in allowed_values if v != old_val]
            if not possible_vals:
                continue
                
            new_val = random.choice(possible_vals)
            
            # Apply change
            mat[r_idx, c_idx] = new_val
            
            # Calculate new error
            new_error = compute_error(U, V, W)
            
            # Acceptance probability
            if new_error < best_error:
                best_error = new_error
                best_U_copy = U.copy()
                best_V_copy = V.copy()
                best_W_copy = W.copy()
                
                # Check for exact solution
                if new_error < 1e-9:
                    # Round to nearest 0.5 to ensure clean output
                    U_rnd = np.round(U * 2) / 2
                    V_rnd = np.round(V * 2) / 2
                    W_rnd = np.round(W * 2) / 2
                    # Verify exactness again after rounding
                    if compute_error(U_rnd, V_rnd, W_rnd) < 1e-9:
                        return U_rnd, V_rnd, W_rnd
            else:
                # Calculate acceptance probability based on current temperature
                # We need a current temperature variable
                # Let's estimate T based on iteration count
                T_current = T_init * (alpha ** it)
                if T_current > T_end:
                    delta = new_error - current_error
                    if delta < 0 or random.random() < np.exp(-delta / T_current):
                        current_error = new_error
                    else:
                        # Reject
                        mat[r_idx, c_idx] = old_val
                else:
                    # Reject
                    mat[r_idx, c_idx] = old_val

        # If we didn't find an exact solution, check if best is close enough to round
        if best_error < 1e-6:
             U_rnd = np.round(best_U_copy * 2) / 2
             V_rnd = np.round(best_V_copy * 2) / 2
             W_rnd = np.round(best_W_copy * 2) / 2
             if compute_error(U_rnd, V_rnd, W_rnd) < 1e-9:
                 return U_rnd, V_rnd, W_rnd

        return None

    for rank in target_ranks:
        if time.time() > deadline - 10.0:
            break
        result = try_find_decomposition_discrete(rank)
        if result is not None:
            U, V, W = result
            best_U = U.tolist()
            best_V = V.tolist()
            best_W = W.tolist()
            best_rank = rank
            break

    return {"U": best_U, "V": best_V, "W": best_W}
