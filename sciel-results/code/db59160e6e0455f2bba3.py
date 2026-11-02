"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_school = n * m * p  # 60

    # Build target tensor T[i, j, k] where i=(i1,j1), j=(j2,k2), k=(k3,i3)
    T = np.zeros((nm, mp, pn))
    for i1 in range(n):
        for j1 in range(m):
            for k2 in range(p):
                T[i1 * m + j1, j1 * p + k2, k2 * n + i1] = 1.0

    # Schoolbook decomposition
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i1 in range(n):
        for j1 in range(m):
            for k2 in range(p):
                U_school[i1 * m + j1, col] = 1.0
                V_school[j1 * p + k2, col] = 1.0
                W_school[k2 * n + i1, col] = 1.0
                col += 1

    best_rank = rank_school
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()

    deadline = time.time() + 130  # 20s safety margin before 150s kill

    # Helper: perform ALS iterations with better convergence
    def als_step(U_c, V_c, W_c, r, T, max_iters=50, tol=1e-12):
        for it in range(max_iters):
            # Solve for U: T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
            M = np.einsum('jr,kr->jkr', V_c, W_c).reshape(mp * pn, r)
            M_pinv = np.linalg.pinv(M.T)
            U_new = T.reshape(nm, mp * pn) @ M_pinv
            
            # Solve for V
            N = np.einsum('ir,kr->ikr', U_new, W_c).reshape(nm * pn, r)
            N_pinv = np.linalg.pinv(N.T)
            V_new = T.reshape(mp, nm * pn) @ N_pinv
            
            # Solve for W
            P = np.einsum('ir,jr->ijr', U_new, V_new).reshape(nm * mp, r)
            P_pinv = np.linalg.pinv(P.T)
            W_new = T.reshape(pn, nm * mp) @ P_pinv
            
            # Check convergence
            if np.max(np.abs(U_new - U_c)) < tol and \
               np.max(np.abs(V_new - V_c)) < tol and \
               np.max(np.abs(W_new - W_c)) < tol:
                return U_new, V_new, W_new
            
            U_c, V_c, W_c = U_new, V_new, W_new
        
        return U_c, V_c, W_c

    # Helper: check if a decomposition is valid and try to round
    def check_and_round(U_c, V_c, W_c, T, r):
        T_check = np.einsum('ir,jr,kr->ijk', U_c, V_c, W_c)
        error = np.max(np.abs(T_check - T))
        
        if error < 1e-8:
            # Try to round to half-integers
            U_r = np.round(U_c * 2) / 2
            V_r = np.round(V_c * 2) / 2
            W_r = np.round(W_c * 2) / 2
            
            T_check2 = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
            if np.max(np.abs(T_check2 - T)) < 1e-10:
                return True, U_r, V_r, W_r
            
            # If rounding fails, try integer rounding
            U_i = np.round(U_c)
            V_i = np.round(V_c)
            W_i = np.round(W_c)
            
            T_check3 = np.einsum('ir,jr,kr->ijk', U_i, V_i, W_i)
            if np.max(np.abs(T_check3 - T)) < 1e-10:
                return True, U_i, V_i, W_i
            
            # If still fails, keep the floating point solution if it's very accurate
            if error < 1e-12:
                return True, U_c, V_c, W_c
        
        return False, None, None, None

    # Try to find lower rank decompositions
    # We'll try a few specific ranks that are likely candidates for improvement
    # and use multiple initialization strategies.
    
    candidate_ranks = list(range(rank_school - 1, max(nm, mp, pn) - 1, -1))
    
    for r in candidate_ranks:
        if time.time() > deadline:
            break

        found = False
        # Try multiple random initializations
        for seed in range(20):
            if time.time() > deadline:
                break
            
            np.random.seed(seed)

            # Different initialization strategies
            if seed < 5:
                # Start from schoolbook truncated
                U_c = U_school[:, :r].copy() if r <= rank_school else np.random.randn(nm, r)
                V_c = V_school[:, :r].copy() if r <= rank_school else np.random.randn(mp, r)
                W_c = W_school[:, :r].copy() if r <= rank_school else np.random.randn(pn, r)
            elif seed < 10:
                # Random initialization
                U_c = np.random.randn(nm, r)
                V_c = np.random.randn(mp, r)
                W_c = np.random.randn(pn, r)
            elif seed < 15:
                # Sparse random initialization
                U_c = np.random.randn(nm, r)
                V_c = np.random.randn(mp, r)
                W_c = np.random.randn(pn, r)
                mask_u = np.random.rand(nm, r) < 0.3
                mask_v = np.random.rand(mp, r) < 0.3
                mask_w = np.random.rand(pn, r) < 0.3
                U_c[mask_u] = 0
                V_c[mask_v] = 0
                W_c[mask_w] = 0
            else:
                # Structured initialization based on tensor structure
                U_c = np.zeros((nm, r))
                V_c = np.zeros((mp, r))
                W_c = np.zeros((pn, r))
                for col in range(r):
                    i1 = col % n
                    j1 = (col // n) % m
                    k2 = (col // (n * m)) % p
                    U_c[i1 * m + j1, col] = 1.0
                    V_c[j1 * p + k2, col] = 1.0
                    W_c[k2 * n + i1, col] = 1.0

            # Perform ALS
            U_c, V_c, W_c = als_step(U_c, V_c, W_c, r, T, max_iters=30)
            
            # Check if we found a valid decomposition
            valid, U_r, V_r, W_r = check_and_round(U_c, V_c, W_c, T, r)
            
            if valid:
                best_rank = r
                best_U, best_V, best_W = U_r, V_r, W_r
                found = True
                break

        if found:
            # If we found a lower rank, we can stop searching for even lower ranks
            # if we've reached a reasonable bound, but let's try a bit more if time permits
            if r <= 40:  # Heuristic: if we're well below schoolbook, stop early
                break

    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
