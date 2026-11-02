import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 145.0  # 5 seconds safety margin
    
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Precompute indices for the tensor
    # Equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # Number of equations: n * m * p
    num_eqs = n * m * p
    
    # Try to find a rank lower than schoolbook
    # We will try a few ranks below 84
    # Note: The search for exact integer/half-integer solutions is stochastic.
    
    best_result = None
    best_rank = rank_school
    
    # Generate schoolbook solution as fallback
    U_sch = [[0] * rank_school for _ in range(n * m)]
    V_sch = [[0] * rank_school for _ in range(m * p)]
    W_sch = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i * m + j][col] = 1
                V_sch[j * p + k][col] = 1
                W_sch[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_sch, "V": V_sch, "W": W_sch}
    
    # Search for lower rank
    # We iterate r from rank_school - 1 down to, say, rank_school - 10
    # If we find one, we return it immediately?
    # The task asks for "rank-minimal". So we should try to go as low as possible.
    # But "rank-minimal" implies the global minimum.
    # If we can't find the global minimum, we should find the lowest we can.
    # However, if we find rank 83, is it minimal? Maybe 82 is possible.
    # Given the time, we can't exhaustively search down to 1.
    # We will try to find *a* valid decomposition with rank < 84.
    # If we find rank 83, we return it. If we find 82, we return it.
    # We should iterate r from 83 downwards and stop at the first valid one found?
    # No, that might miss a lower one. But finding *any* lower one is an improvement.
    # To be safe and fast, we can try a few random restarts for r=83, 82, etc.
    
    def check_validity(U, V, W, r):
        # U, V, W are lists of lists
        # Convert to numpy for speed? Or just loops.
        # n*m*p is 84, r is ~80. 84*80 = 6720 ops. Very fast.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def is_half_integer(x):
        return abs(x * 2 - round(x * 2)) < 1e-9

    def check_half_integers(U, V, W):
        for row in U:
            for val in row:
                if not is_half_integer(val):
                    return False
        for row in V:
            for val in row:
                if not is_half_integer(val):
                    return False
        for row in W:
            for val in row:
                if not is_half_integer(val):
                    return False
        return True

    # Search loop
    # We try ranks 83, 82, ... down to 75
    # For each rank, we try a few random initializations.
    
    import random
    random.seed(42)
    
    found = False
    for r in range(rank_school - 1, 70, -1):
        if time.time() > deadline - 10:
            break
            
        # Try a few restarts for this rank
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W with random small values
            # To help convergence, maybe initialize with sparse random?
            U_init = np.zeros((n * m, r))
            V_init = np.zeros((m * p, r))
            W_init = np.zeros((p * n, r))
            
            # Random sparse initialization
            # Each column has a few non-zeros
            for c in range(r):
                # Pick random indices
                i_u = random.randint(0, n * m - 1)
                i_v = random.randint(0, m * p - 1)
                i_w = random.randint(0, p * n - 1)
                
                U_init[i_u, c] = random.choice([1, -1, 0.5, -0.5])
                V_init[i_v, c] = random.choice([1, -1, 0.5, -0.5])
                W_init[i_w, c] = random.choice([1, -1, 0.5, -0.5])
                
                # Maybe add a few more
                if random.random() > 0.5:
                    i_u2 = random.randint(0, n * m - 1)
                    U_init[i_u2, c] = random.choice([1, -1, 0.5, -0.5])
                    
            # Define the residual function
            def residual(params):
                # params is a flat vector of length r * (nm + mp + pn)
                size_u = n * m * r
                size_v = m * p * r
                size_w = p * n * r
                
                U_flat = params[:size_u]
                V_flat = params[size_u:size_u + size_v]
                W_flat = params[size_u + size_v:]
                
                U_mat = U_flat.reshape(n * m, r)
                V_mat = V_flat.reshape(m * p, r)
                W_mat = W_flat.reshape(p * n, r)
                
                res = np.zeros(num_eqs)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            u_row = U_mat[i * m + j]
                            v_row = V_mat[j * p + k]
                            w_row = W_mat[k * n + i]
                            val = np.dot(u_row, v_row * w_row)
                            res[idx] = val - 1.0
                            idx += 1
                return res
            
            p0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                from scipy.optimize import least_squares
                sol = least_squares(residual, p0, method='lm', max_nfev=1000)
                
                if sol.success or sol.cost < 1e-10:
                    U_sol = sol.x[:size_u].reshape(n * m, r)
                    V_sol = sol.x[size_u:size_u + size_v].reshape(m * p, r)
                    W_sol = sol.x[size_u + size_v:].reshape(p * n, r)
                    
                    # Round to half-integers
                    U_round = np.round(U_sol * 2) / 2
                    V_round = np.round(V_sol * 2) / 2
                    W_round = np.round(W_sol * 2) / 2
                    
                    # Check if rounded solution is exact
                    # Convert to lists for check_validity
                    U_list = U_round.tolist()
                    V_list = V_round.tolist()
                    W_list = W_round.tolist()
                    
                    if check_validity(U_list, V_list, W_list, r):
                        if check_half_integers(U_list, V_list, W_list):
                            return {"U": U_list, "V": V_list, "W": W_list}
                            
            except Exception as e:
                pass
                
    return fallback
