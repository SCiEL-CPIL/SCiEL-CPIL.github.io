import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U_school, V_school, W_school = schoolbook()
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p

    # Target tensor T
    # T is sparse. We can represent it as a function or a dense array for small sizes.
    # Size: 9 x 12 x 12 = 1296 elements. Dense is fine.
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Function to check if a decomposition is valid (exact integer/half-integer match)
    def is_valid(U, V, W):
        # Check if entries are close to integers or half-integers
        for M in [U, V, W]:
            # Multiply by 2 to check for integers
            M2 = M * 2
            if not np.allclose(M2, np.round(M2), atol=1e-9):
                return False
        # Check if the tensor sum matches T
        # Sum over rank: U[:, r] outer V[:, r] outer W[:, r]
        # This is a tensor contraction.
        # We can do this by looping over rank or using einsum.
        # np.einsum('ir,jr,kr->ijk', U, V, W)
        try:
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            if np.allclose(T_recon, T, atol=1e-9):
                return True
        except Exception:
            return False
        return False

    # Try to find lower rank decompositions
    # We try ranks from best_rank - 1 down to some lower bound
    # Lower bound for 3x3x4 is likely around 25-30.
    # We will try a few ranks with limited time.
    
    max_time = 110.0 # Leave 10s margin
    
    # We will try ranks: 35, 34, 33, 32, 31, 30
    # For each rank, we do a few random restarts with numerical optimization.
    
    import scipy.optimize as opt
    
    def objective(params, R):
        # params is a flat array of size R * (n*m + m*p + p*n)
        U = params[:R * n * m].reshape(R, n * m).T
        V = params[R * n * m : R * n * m + R * m * p].reshape(R, m * p).T
        W = params[R * n * m + R * m * p :].reshape(R, p * n).T
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.sum((T_recon - T) ** 2)

    # We'll try a few ranks
    ranks_to_try = [35, 34, 33, 32, 31, 30]
    
    for R in ranks_to_try:
        if time.time() - start_time > max_time:
            break
            
        # Try a few random initializations
        for attempt in range(5):
            if time.time() - start_time > max_time:
                break
                
            # Initialize with random integers in [-1, 1]
            np.random.seed(attempt + R) # Reproducible-ish
            U_init = np.random.randint(-1, 2, (R, n * m))
            V_init = np.random.randint(-1, 2, (R, m * p))
            W_init = np.random.randint(-1, 2, (R, p * n))
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use L-BFGS-B or similar
            try:
                res = opt.minimize(objective, params_init, args=(R,), method='L-BFGS-B', 
                                   options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-10})
                if res.fun < 1e-8:
                    params_sol = res.x
                    U_sol = params_sol[:R * n * m].reshape(R, n * m).T
                    V_sol = params_sol[R * n * m : R * n * m + R * m * p].reshape(R, m * p).T
                    W_sol = params_sol[R * n * m + R * m * p :].reshape(R, p * n).T
                    
                    # Round to nearest half-integer
                    U_round = np.round(U_sol * 2) / 2.0
                    V_round = np.round(V_sol * 2) / 2.0
                    W_round = np.round(W_sol * 2) / 2.0
                    
                    if is_valid(U_round, V_round, W_round):
                        best_U, best_V, best_W = U_round, V_round, W_round
                        best_rank = R
                        break # Found valid for this R, try lower R
            except Exception:
                pass

    # Convert to lists of lists as required by interface
    # The interface expects {'U': U, 'V': V, 'W': W}
    # U, V, W are numpy arrays. We can return them as lists or arrays?
    # "factor matrices of shapes ... with integer/half-integer entries"
    # Usually, returning numpy arrays is fine, or lists.
    # Let's return lists to be safe, or just the arrays if the evaluator accepts them.
    # The seed program returns lists of lists.
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
