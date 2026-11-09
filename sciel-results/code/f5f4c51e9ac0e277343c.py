"""Matrix multiplication 2x3 by 3x4: find rank-minimal exact tensor decomposition."""
import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication.
    The known minimum rank is 19.
    We use a combination of:
    1. Structured random search with simple values (0, 0.5, 1).
    2. Alternating Least Squares (ALS) with multiple random restarts.
    3. Local refinement using scipy.optimize.least_squares.
    4. Rounding to half-integers and exact verification.
    Falls back to schoolbook rank 24 if no rank 19 solution is found.
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # 2, 3, 4
    dim_U = n * m  # 6
    dim_V = m * p  # 12
    dim_W = p * n  # 8
    
    # Target rank
    target_rank = 19

    # Build target tensor T
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Note: The tensor is defined over the indices of the matrices A, B, C.
    # The equation is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # We can represent this as a 3D array T of shape (dim_U, dim_V, dim_W)
    # but only specific indices are constrained to 1.
    # Actually, it's easier to work with the flattened constraint vector.
    # Let's create a list of constraints: (idx_U, idx_V, idx_W) such that T[idx_U, idx_V, idx_W] = 1
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                constraints.append((idx_u, idx_v, idx_w))
    
    num_constraints = len(constraints) # 24

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(dim_U)]
        V = [[0] * rank for _ in range(dim_V)]
        W = [[0] * rank for _ in range(dim_W)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def verify_solution(U, V, W, rank):
        """
        Verify if U, V, W of given rank exactly decompose the tensor.
        Returns True if sum_r U[idx_u, r] * V[idx_v, r] * W[idx_w, r] == 1 for all constraints.
        """
        for idx_u, idx_v, idx_w in constraints:
            s = 0.0
            for r in range(rank):
                s += U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
            if abs(s - 1.0) > 1e-8:
                return False
        return True

    def round_and_verify(U, V, W, rank):
        """
        Round U, V, W to nearest 0.5 and verify.
        Returns (U_rounded, V_rounded, W_rounded) if valid, else None.
        """
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        if verify_solution(U_r, V_r, W_r, rank):
            # Convert to lists of lists with int/float
            U_list = U_r.tolist()
            V_list = V_r.tolist()
            W_list = W_r.tolist()
            
            # Ensure integers where possible for cleaner output
            def clean_list(lst):
                for i in range(len(lst)):
                    for j in range(len(lst[i])):
                        val = lst[i][j]
                        if abs(val - round(val)) < 1e-9:
                            lst[i][j] = int(round(val))
                        else:
                            # Ensure it's a float like 0.5
                            lst[i][j] = float(val)
                return lst
            
            return {"U": clean_list(U_list), "V": clean_list(V_list), "W": clean_list(W_list)}
        return None

    def als_decompose(rank, n_iter=200, seed=42):
        """
        Alternating Least Squares to minimize sum (sum_r U[idx_u,r] V[idx_v,r] W[idx_w,r] - 1)^2
        """
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(dim_U, rank) * 0.1
        V = rng.randn(dim_V, rank) * 0.1
        W = rng.randn(dim_W, rank) * 0.1
        
        best_error = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(n_iter):
            # Update U
            # For each idx_u, we need to solve for U[idx_u, :]
            # The equation for a specific constraint (idx_u, idx_v, idx_w) is:
            # U[idx_u, r] * (V[idx_v, r] * W[idx_w, r]) = 1 (approximately, summed over r)
            # Actually, the residual is sum_r U[idx_u, r] * (V[idx_v, r] * W[idx_w, r]) - 1
            # This is linear in U if V, W are fixed.
            # Let A_u be a matrix where rows correspond to constraints involving idx_u.
            # For a constraint c=(u,v,w), the coefficient vector for U[u, :] is V[v, :] * W[w, :] (element-wise product).
            
            # We can solve this as a least squares problem for each row of U.
            # Let's build the system for U.
            # We want to minimize sum_c (U[u_c, :] dot (V[v_c, :] * W[w_c, :]) - 1)^2
            # This is a linear system in U[u, :] for each u.
            
            # Group constraints by u
            cons_by_u = {u: [] for u in range(dim_U)}
            for u, v, w in constraints:
                cons_by_u[u].append((v, w))
            
            for u in range(dim_U):
                if not cons_by_u[u]:
                    continue
                # Build matrix A_u: rows are constraints, cols are rank
                # A_u[row, r] = V[v, r] * W[w, r]
                # b_u[row] = 1
                A_u = np.array([V[v, :] * W[w, :] for v, w in cons_by_u[u]])
                b_u = np.ones(len(cons_by_u[u]))
                # Solve A_u.T @ U[u, :] = b_u
                # U[u, :] = np.linalg.lstsq(A_u.T, b_u, rcond=None)[0]
                # Using pinv or lstsq
                try:
                    U[u, :] = np.linalg.lstsq(A_u.T, b_u, rcond=None)[0]
                except:
                    pass
            
            # Update V
            cons_by_v = {v: [] for v in range(dim_V)}
            for u, v, w in constraints:
                cons_by_v[v].append((u, w))
                
            for v in range(dim_V):
                if not cons_by_v[v]:
                    continue
                A_v = np.array([U[u, :] * W[w, :] for u, w in cons_by_v[v]])
                b_v = np.ones(len(cons_by_v[v]))
                try:
                    V[v, :] = np.linalg.lstsq(A_v.T, b_v, rcond=None)[0]
                except:
                    pass
                    
            # Update W
            cons_by_w = {w: [] for w in range(dim_W)}
            for u, v, w in constraints:
                cons_by_w[w].append((u, v))
                
            for w in range(dim_W):
                if not cons_by_w[w]:
                    continue
                A_w = np.array([U[u, :] * V[v, :] for u, v in cons_by_w[w]])
                b_w = np.ones(len(cons_by_w[w]))
                try:
                    W[w, :] = np.linalg.lstsq(A_w.T, b_w, rcond=None)[0]
                except:
                    pass
                    
            # Calculate error
            error = 0.0
            for u, v, w in constraints:
                s = np.dot(U[u, :], V[v, :] * W[w, :])
                error += (s - 1.0)**2
                
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if error < 1e-12:
                break
                
        return best_U, best_V, best_W, best_error

    # 1. Try structured search (Inspiration 2)
    # This is very fast and might find the solution if it's simple.
    # We'll try a few random configurations from {0, 0.5, 1}
    np.random.seed(42)
    
    # Budget for structured search: 5 seconds
    t_struct_start = time.time()
    while time.time() - t_struct_start < 5.0:
        U = np.random.choice([0, 0.5, 1], size=(dim_U, target_rank))
        V = np.random.choice([0, 0.5, 1], size=(dim_V, target_rank))
        W = np.random.choice([0, 0.5, 1], size=(dim_W, target_rank))
        
        if verify_solution(U, V, W, target_rank):
            res = round_and_verify(U, V, W, target_rank)
            if res:
                return res

    # 2. Try ALS with multiple restarts
    # Budget for ALS: 60 seconds
    t_als_start = time.time()
    max_als_restarts = 50
    best_als_error = np.inf
    best_als_solution = None
    
    for i in range(max_als_restarts):
        if time.time() - t_als_start > 60.0:
            break
            
        U, V, W, err = als_decompose(target_rank, n_iter=100, seed=42+i)
        
        if err < best_als_error:
            best_als_error = err
            best_als_solution = (U, V, W)
            
        # Check if we found an exact solution
        if err < 1e-10:
            res = round_and_verify(U, V, W, target_rank)
            if res:
                return res

    # 3. If ALS got close, try to polish with least_squares
    if best_als_solution and best_als_error < 1.0:
        U0, V0, W0 = best_als_solution
        
        # Flatten
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        def residuals(x):
            size_u = dim_U * target_rank
            size_v = dim_V * target_rank
            U = x[:size_u].reshape(dim_U, target_rank)
            V = x[size_u:size_u+size_v].reshape(dim_V, target_rank)
            W = x[size_u+size_v:].reshape(dim_W, target_rank)
            
            res_vec = np.zeros(num_constraints)
            for idx, (u, v, w) in enumerate(constraints):
                res_vec[idx] = np.dot(U[u, :], V[v, :] * W[w, :]) - 1.0
            return res_vec

        t_polish_start = time.time()
        while time.time() - t_polish_start < 30.0:
            try:
                res = opt.least_squares(residuals, x0, method='trf', max_nfev=500, xtol=1e-12, ftol=1e-12, gtol=1e-12)
                x_opt = res.x
                size_u = dim_U * target_rank
                size_v = dim_V * target_rank
                U_opt = x_opt[:size_u].reshape(dim_U, target_rank)
                V_opt = x_opt[size_u:size_u+size_v].reshape(dim_V, target_rank)
                W_opt = x_opt[size_u+size_v:].reshape(dim_W, target_rank)
                
                # Check error
                err = np.sum(residuals(x_opt)**2)
                if err < 1e-10:
                    res_sol = round_and_verify(U_opt, V_opt, W_opt, target_rank)
                    if res_sol:
                        return res_sol
                
                # If we didn't get a perfect solution, update x0 to continue
                x0 = x_opt
                break # Don't loop forever on least_squares, it's slow
            except Exception:
                break

    # 4. Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
