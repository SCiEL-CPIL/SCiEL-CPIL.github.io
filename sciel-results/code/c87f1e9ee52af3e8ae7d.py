import time
import random
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m  # 15
    dim_V = m * p  # 25
    dim_W = p * n  # 15
    
    # Build target tensor
    Target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                Target[alpha, beta, gamma] = 1.0

    def verify(U, V, W, tol=1e-9):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        T_calc = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        return np.allclose(T_calc, Target, atol=tol)

    def snap_to_half_int(x):
        return np.round(2 * x) / 2.0

    def solve_for_W(U, V):
        """Given U and V, solve for W using least squares."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        R = U_arr.shape[1]
        
        W = np.zeros((dim_W, R))
        residual = 0.0
        
        # M[alpha, beta, r] = U[alpha, r] * V[beta, r]
        M = np.einsum('ar,br->abr', U_arr, V_arr)
        
        for gamma in range(dim_W):
            A = M.reshape(dim_U * dim_V, R)
            b = Target[:, :, gamma].reshape(dim_U * dim_V)
            # Use lstsq to find W[gamma, :]
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            W[gamma, :] = x
            r = b - A @ x
            residual += np.sum(r**2)
        
        return W, residual

    def compute_residual(U, V):
        """Compute the residual without solving for W."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        R = U_arr.shape[1]
        
        M = np.einsum('ar,br->abr', U_arr, V_arr)
        residual = 0.0
        
        for gamma in range(dim_W):
            A = M.reshape(dim_U * dim_V, R)
            b = Target[:, :, gamma].reshape(dim_U * dim_V)
            # Project b onto column space of A
            if R > 0:
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                r = b - A @ x
                residual += np.sum(r**2)
            else:
                residual += np.sum(b**2)
        
        return residual

    def polish_solution(U, V, W, max_iter=100):
        """Polish a near-solution by refining entries."""
        U_curr = np.array(U, dtype=float).copy()
        V_curr = np.array(V, dtype=float).copy()
        W_curr = np.array(W, dtype=float).copy()
        R = U_curr.shape[1]
        
        for _ in range(max_iter):
            # Solve for optimal W given U, V
            W_opt, res = solve_for_W(U_curr, V_curr)
            W_snap = snap_to_half_int(W_opt)
            
            if verify(U_curr, V_curr, W_snap):
                return U_curr, V_curr, W_snap
            
            # Try small perturbations
            improved = False
            for _ in range(20):
                # Pick random entry in U or V
                if random.random() < 0.5:
                    idx_u = random.randint(0, dim_U - 1)
                    idx_r = random.randint(0, R - 1)
                    old_val = U_curr[idx_u, idx_r]
                    # Try nearby half-integers
                    candidates = set()
                    for delta in [-1, -0.5, 0, 0.5, 1]:
                        candidates.add(snap_to_half_int(old_val + delta))
                    new_val = random.choice(list(candidates))
                    if new_val == old_val:
                        continue
                    U_curr[idx_u, idx_r] = new_val
                    W_new, res_new = solve_for_W(U_curr, V_curr)
                    W_new_snap = snap_to_half_int(W_new)
                    if verify(U_curr, V_curr, W_new_snap):
                        U_curr[idx_u, idx_r] = new_val
                        return U_curr, V_curr, W_new_snap
                    elif res_new < res:
                        res = res_new
                        W_curr = W_new_snap
                        improved = True
                    else:
                        U_curr[idx_u, idx_r] = old_val
                else:
                    idx_v = random.randint(0, dim_V - 1)
                    idx_r = random.randint(0, R - 1)
                    old_val = V_curr[idx_v, idx_r]
                    candidates = set()
                    for delta in [-1, -0.5, 0, 0.5, 1]:
                        candidates.add(snap_to_half_int(old_val + delta))
                    new_val = random.choice(list(candidates))
                    if new_val == old_val:
                        continue
                    V_curr[idx_v, idx_r] = new_val
                    W_new, res_new = solve_for_W(U_curr, V_curr)
                    W_new_snap = snap_to_half_int(W_new)
                    if verify(U_curr, V_curr, W_new_snap):
                        V_curr[idx_v, idx_r] = new_val
                        return U_curr, V_curr, W_new_snap
                    elif res_new < res:
                        res = res_new
                        W_curr = W_new_snap
                        improved = True
                    else:
                        V_curr[idx_v, idx_r] = old_val
            
            if not improved:
                break
        
        return U_curr, V_curr, W_curr

    start_time = time.time()
    time_limit = 115.0  # Safety margin

    # Schoolbook solution (rank 75)
    best_result = None
    best_rank = n * m * p  # 75
    
    U_sb = np.zeros((dim_U, best_rank))
    V_sb = np.zeros((dim_V, best_rank))
    W_sb = np.zeros((dim_W, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    if verify(U_sb, V_sb, W_sb):
        best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    # Search for lower ranks
    # Lower bound is max(n*m, m*p, p*n) = 25
    # Upper bound is current best
    for R in range(25, best_rank):
        if time.time() - start_time > time_limit:
            break
            
        found = False
        # Number of restarts decreases as R increases (higher R is easier)
        n_restarts = 15 if R < 40 else 8
        
        for restart in range(n_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Structured initialization: exploit tensor structure
            U_init = np.zeros((dim_U, R))
            V_init = np.zeros((dim_V, R))
            
            # Strategy 1: Block-structured initialization
            # For each "column" r, assign it to handle a subset of (i,j,k) triples
            if restart % 3 == 0:
                # Dense initialization with small values
                for r in range(R):
                    # Each rank-1 component handles some interactions
                    # Assign U and V entries with probability
                    for alpha in range(dim_U):
                        if random.random() < 0.4:
                            U_init[alpha, r] = random.choice([-1, -0.5, 0.5, 1])
                    for beta in range(dim_V):
                        if random.random() < 0.4:
                            V_init[beta, r] = random.choice([-1, -0.5, 0.5, 1])
            elif restart % 3 == 1:
                # Sparse with larger values
                for r in range(R):
                    # Fewer non-zero entries but larger magnitude
                    nz_count = random.randint(3, 6)
                    alpha_indices = random.sample(range(dim_U), nz_count)
                    for alpha in alpha_indices:
                        U_init[alpha, r] = random.choice([-1, 1])
                    
                    nz_count_v = random.randint(3, 6)
                    beta_indices = random.sample(range(dim_V), nz_count_v)
                    for beta in beta_indices:
                        V_init[beta, r] = random.choice([-1, 1])
            else:
                # Very sparse
                for r in range(R):
                    if random.random() < 0.7:
                        alpha = random.randint(0, dim_U - 1)
                        U_init[alpha, r] = random.choice([-1, 1])
                    
                    if random.random() < 0.7:
                        beta = random.randint(0, dim_V - 1)
                        V_init[beta, r] = random.choice([-1, 1])
            
            # Initial solve
            W_check, res_check = solve_for_W(U_init, V_init)
            W_snap = snap_to_half_int(W_check)
            
            if verify(U_init, V_init, W_snap):
                best_result = {"U": U_init.tolist(), "V": V_init.tolist(), "W": W_snap.tolist()}
                best_rank = R
                found = True
                break
            
            # Local search with polish
            current_res = res_check
            U_curr = U_init.copy()
            V_curr = V_init.copy()
            
            max_steps = 1500
            for step in range(max_steps):
                if time.time() - start_time > time_limit:
                    break
                    
                # Pick a random variable to change
                # 50% chance U, 50% chance V
                is_u_move = random.random() < 0.5
                if is_u_move:
                    idx_u = random.randint(0, dim_U - 1)
                    idx_r = random.randint(0, R - 1)
                    old_val = U_curr[idx_u, idx_r]
                    # Generate new value from neighborhood
                    neighbors = set()
                    for delta in [-1, -0.5, 0, 0.5, 1]:
                        neighbors.add(snap_to_half_int(old_val + delta))
                    new_val = random.choice(list(neighbors))
                    
                    U_curr[idx_u, idx_r] = new_val
                else:
                    idx_v = random.randint(0, dim_V - 1)
                    idx_r = random.randint(0, R - 1)
                    old_val = V_curr[idx_v, idx_r]
                    neighbors = set()
                    for delta in [-1, -0.5, 0, 0.5, 1]:
                        neighbors.add(snap_to_half_int(old_val + delta))
                    new_val = random.choice(list(neighbors))
                    
                    V_curr[idx_v, idx_r] = new_val

                # Solve for W
                W_new, res_new = solve_for_W(U_curr, V_curr)
                W_new_snap = snap_to_half_int(W_new)
                
                # Check if we found a solution
                if verify(U_curr, V_curr, W_new_snap):
                    best_result = {"U": U_curr.tolist(), "V": V_curr.tolist(), "W": W_new_snap.tolist()}
                    best_rank = R
                    found = True
                    break
                
                # Accept if residual decreased or with small probability (simulated annealing)
                if res_new < current_res:
                    current_res = res_new
                elif random.random() < 0.05:
                    # Accept worse move occasionally to escape local minima
                    current_res = res_new
                else:
                    # Revert the specific variable that was changed
                    if is_u_move:
                        U_curr[idx_u, idx_r] = old_val
                    else:
                        V_curr[idx_v, idx_r] = old_val

            if found:
                break

        if found:
            break

    # Final polish attempt on best result
    if best_result is not None and time.time() - start_time < time_limit - 5:
        U_best = np.array(best_result['U'], dtype=float)
        V_best = np.array(best_result['V'], dtype=float)
        W_best = np.array(best_result['W'], dtype=float)
        U_pol, V_pol, W_pol = polish_solution(U_best, V_best, W_best, max_iter=50)
        if verify(U_pol, V_pol, W_pol):
            best_result = {"U": U_pol.tolist(), "V": V_pol.tolist(), "W": W_pol.tolist()}

    if best_result is None:
        best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return best_result
# EVOLVE-BLOCK-END
