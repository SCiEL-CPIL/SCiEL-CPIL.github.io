import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    """
    Constructs a low-rank matrix multiplication decomposition for 2x3 * 3x8.
    Returns a dictionary with matrices U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    # Strategy: Split the middle dimension m=3 into m1=2 and m2=1.
    # Part A: 2x2 * 2x8 (rows 0,1 of A; rows 0,1 of B)
    # Part B: 2x1 * 1x8 (row 2 of A; row 2 of B)
    #
    # For Part A: Decompose 2x2*2x8 as 4 independent 2x2*2x2 multiplications.
    # Each uses Strassen (rank 7). Total: 4*7 = 28.
    # For Part B: 2x1*1x8 needs rank 2*8 = 16 (schoolbook).
    # Total rank: 28 + 16 = 44.
    #
    # We construct this rank-44 solution. Then we attempt to find a lower rank
    # solution (e.g., 43) using numerical optimization initialized near the known
    # solution or randomly, snapping to half-integers.

    def build_rank_44():
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0

        # Part 1: A[0:2, 0:2] * B[0:2, 0:8] -> C[0:2, 0:8]
        for blk in range(4):
            bo = 2 * blk
            ve = bo
            vf = bo + 1
            vg = 8 + bo
            vh = 8 + bo + 1

            w00 = 2 * bo
            w01 = 2 * bo + 2
            w10 = 2 * bo + 1
            w11 = 2 * bo + 3

            # P1 = (a+d)(e+h)
            U[0][col] = 1
            U[4][col] = 1
            V[ve][col] = 1
            V[vh][col] = 1
            W[w00][col] = 1
            W[w11][col] = 1
            col += 1

            # P2 = (c+d)e
            U[3][col] = 1
            U[4][col] = 1
            V[ve][col] = 1
            W[w10][col] = 1
            W[w11][col] = -1
            col += 1

            # P3 = a(f-h)
            U[0][col] = 1
            V[vf][col] = 1
            V[vh][col] = -1
            W[w01][col] = 1
            W[w11][col] = 1
            col += 1

            # P4 = d(g-e)
            U[4][col] = 1
            V[vg][col] = 1
            V[ve][col] = -1
            W[w00][col] = 1
            W[w10][col] = 1
            col += 1

            # P5 = (a+b)h
            U[0][col] = 1
            U[1][col] = 1
            V[vh][col] = 1
            W[w00][col] = -1
            W[w01][col] = 1
            col += 1

            # P6 = (c-a)(e+f)
            U[3][col] = 1
            U[0][col] = -1
            V[ve][col] = 1
            V[vf][col] = 1
            W[w11][col] = 1
            col += 1

            # P7 = (b-d)(g+h)
            U[1][col] = 1
            U[4][col] = -1
            V[vg][col] = 1
            V[vh][col] = 1
            W[w00][col] = 1
            col += 1

        # Part 2: A[0:2, 2] * B[2, 0:8] -> C[0:2, 0:8]
        for k in range(8):
            for i in range(2):
                U[i * 3 + 2][col] = 1
                V[16 + k][col] = 1
                W[2 * k + i][col] = 1
                col += 1

        if col != rank:
            raise RuntimeError("Construction error")
        
        return {"U": U, "V": V, "W": W}

    # First, get the known valid solution
    base_solution = build_rank_44()

    # Helper to verify a solution
    def verify_solution(sol):
        U = np.array(sol["U"], dtype=float)
        V = np.array(sol["V"], dtype=float)
        W = np.array(sol["W"], dtype=float)
        
        # Reshape for tensor product
        # U: (n*m, R) -> (R, n, m)
        # V: (m*p, R) -> (R, m, p)
        # W: (p*n, R) -> (R, p, n)
        
        R = U.shape[1]
        U_t = U.T.reshape(R, n, m)
        V_t = V.T.reshape(R, m, p)
        W_t = W.T.reshape(R, p, n)
        
        # Compute T[i,j,k] = sum_r U[r,i,j] * V[r,j,k] * W[r,k,i]
        # Note: indices in W are k*n + i, so W_t[r, k, i]
        
        # Efficient check:
        # For each i, j:
        #   prod_j = sum_r U[r,i,j] * V[r,j,:]  -> shape (R, p) is not right.
        #   Let's do: for each r, term_r = U[r,i,j] * V[r,j,:] * W[r,:,i]
        #   Sum over r.
        
        # Actually, simpler:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # Tensor M[i,j,k] = 1 if j==k else 0.
        
        max_err = 0.0
        for i in range(n):
            for k in range(p):
                # Target is 1 if i's row index in A matches k's col index in B?
                # No, target tensor for matrix mult is T[i,j,k] = delta_{j,k}.
                # The decomposition is:
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{j,k}
                
                # Let's compute the LHS for all j
                # LHS[j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                
                w_row = W_t[:, k, i] # shape (R,)
                for j in range(m):
                    u_val = U_t[:, i, j] # shape (R,)
                    v_val = V_t[:, j, k] # shape (R,)
                    
                    val = np.sum(u_val * v_val * w_row)
                    target = 1.0 if j == k else 0.0
                    err = abs(val - target)
                    if err > max_err:
                        max_err = err
                        
        return max_err

    # Check base solution just in case
    if verify_solution(base_solution) > 1e-9:
        # Should not happen, but fallback to schoolbook
        rank_fallback = n * m * p
        U = [[0] * rank_fallback for _ in range(n * m)]
        V = [[0] * rank_fallback for _ in range(m * p)]
        W = [[0] * rank_fallback for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    # Try to find a lower rank solution using optimization
    # We will try rank 43.
    # We can use scipy.optimize.least_squares or similar.
    # Since we have a good starting point (the rank 44 solution), we can try to
    # merge two terms or nullify one term's contribution by adjusting others.
    # However, a direct numerical search for rank 43 is complex.
    # Given the constraints and time, we will attempt a simple local search
    # or just return the rank 44 solution if no quick improvement is found.
    
    # Let's try a simple heuristic: Can we combine two of the Part B terms?
    # Part B is 2x1 * 1x8. Rank is 16. This is minimal for this sub-problem.
    # Part A is 4 * Strassen(2x2*2x2). Rank is 28.
    # Is there a better decomposition for 2x2*2x8 than 4*Strassen?
    # 2x2*2x8 is equivalent to 4 independent 2x2*2x2.
    # The rank of 2x2*2x2 is 7 (Strassen).
    # So 4*7 = 28 is likely optimal for that block structure.
    # Could we exploit the structure of 2x3*3x8 more globally?
    # e.g. 2x3*3x8 rank is at least max(2*3, 3*8, 2*8) = 24? No, lower bounds are weak.
    # Known bounds: R(2,3,8) is between 32 and 44?
    # Actually, R(2,3,8) is not trivially known to be < 44.
    # If we cannot find a better one quickly, we return 44.

    # Attempt to use scipy to find rank 43
    try:
        from scipy.optimize import least_squares
        
        target_rank = 43
        size_U = n * m
        size_V = m * p
        size_W = p * n
        total_size = target_rank * (size_U + size_V + size_W)
        
        # Initialize with random values
        # Or better, initialize with the rank 44 solution and drop one column,
        # then adjust? That's hard to map directly to a vector.
        # Let's just do random restarts with a tight time budget.
        
        def error_func(params):
            U = params[0 : target_rank*size_U].reshape(target_rank, size_U)
            V = params[target_rank*size_U : target_rank*size_U + target_rank*size_V].reshape(target_rank, size_V)
            W = params[target_rank*size_U + target_rank*size_V : total_size].reshape(target_rank, size_W)
            
            residuals = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        w_col = W[:, k*n + i]
                        u_row = U[:, i*m + j]
                        v_row = V[:, j*p + k]
                        val = np.sum(w_col * u_row * v_row)
                        target = 1.0 if j == k else 0.0
                        residuals.append(val - target)
            return np.array(residuals)
        
        # We'll do a few quick attempts
        found = False
        best_sol = base_solution
        
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(123 + restart)
            params = np.random.randn(total_size) * 0.1
            
            try:
                res = least_squares(error_func, params, method='lm', max_nfev=2000)
                if res.cost < 1e-10:
                    # Snap to half-integers
                    U_snapped = np.round(res.x[0 : target_rank*size_U].reshape(target_rank, size_U) * 2) / 2
                    V_snapped = np.round(res.x[target_rank*size_U : target_rank*size_U + target_rank*size_V].reshape(target_rank, size_V) * 2) / 2
                    W_snapped = np.round(res.x[target_rank*size_U + target_rank*size_V : total_size].reshape(target_rank, size_W) * 2) / 2
                    
                    # Verify snapped
                    sol = {
                        "U": np.array(U_snapped).T.tolist(),
                        "V": np.array(V_snapped).T.tolist(),
                        "W": np.array(W_snapped).T.tolist()
                    }
                    if verify_solution(sol) < 1e-9:
                        best_sol = sol
                        found = True
                        break
            except Exception:
                continue
                
        if found:
            return best_sol
            
    except ImportError:
        pass

    return base_solution
