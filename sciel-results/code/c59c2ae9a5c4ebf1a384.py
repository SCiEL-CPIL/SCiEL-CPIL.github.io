import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 16
    pn = p * n  # 8
    
    start_time = time.time()
    deadline = start_time + 100.0  # 50s margin before 150s kill

    def check_identity(U, V, W, R):
        """
        Check if the factors satisfy the tensor identity.
        U: (nm, R)
        V: (mp, R)
        W: (pn, R)
        
        Identity:
        sum_r U[i'*m+j', r] V[j''*p+k', r] W[k*n+i, r] = delta_{i', i} delta_{k', k} delta_{j', j''}
        """
        # We can vectorize this check significantly.
        # Let's form the tensor T of size (n, p, n, p, m, m) ? No, that's huge.
        # Better: For each r, compute the outer product of U_r (vector of length nm), V_r (length mp), W_r (length pn).
        # This is a rank-1 tensor in the space of (nm, mp, pn).
        # We want the sum of these rank-1 tensors to equal the identity tensor E.
        # E has non-zero entries only when indices correspond to valid matrix mult.
        
        # Let's construct the target tensor E as a matrix of shape (nm*mp, pn) ?
        # Indexing:
        # U index a = i*m + j
        # V index b = j'*p + k'
        # W index c = k*n + i'
        
        # The condition is:
        # Sum_r U[a, r] V[b, r] W[c, r] = 1 if (a, b, c) is a valid triple for identity, 0 otherwise.
        
        # Valid triple for identity (delta_{i', i} delta_{k', k} delta_{j', j''}):
        # a = i*m + j'  (Note: U uses i' and j'. Let's stick to the prompt's derivation)
        # Prompt derivation:
        # sum_r U[i'*m+j', r] V[j''*p+k', r] W[k*n+i, r] = delta_{i', i} delta_{k', k} delta_{j', j''}
        
        # So, let's iterate over all possible (i, k, i', k', j', j'') and check.
        # There are n*p*n*p*m*m = 2*4*2*4*4*4 = 1024 checks.
        # For each check, we sum over R.
        # Total ops: 1024 * R. If R=27, ~27k ops. Very fast.
        
        max_err = 0.0
        # Pre-extract arrays for speed
        U_arr = np.asarray(U)
        V_arr = np.asarray(V)
        W_arr = np.asarray(W)
        
        for i in range(n):
            for k in range(p):
                w_idx = k * n + i
                for i_prime in range(n):
                    for k_prime in range(p):
                        for j_prime in range(m):
                            u_idx = i_prime * m + j_prime
                            for j_double_prime in range(m):
                                v_idx = j_double_prime * p + k_prime
                                target = 1.0 if (i == i_prime and k == k_prime and j_prime == j_double_prime) else 0.0
                                # val = sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                                # Vectorized dot product: (U[u_idx] * V[v_idx]) dot W[w_idx]
                                val = np.dot(U_arr[u_idx] * V_arr[v_idx], W_arr[w_idx])
                                err = abs(val - target)
                                if err > max_err:
                                    max_err = err
        return max_err

    def strassen_rank_28():
        """
        Constructs the known rank-28 Strassen-based solution.
        Returns U, V, W as numpy arrays.
        """
        rank = 28
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Strassen components for 2x2 * 2x2
        # P (2x2), Q (2x2) -> R (2x2)
        # Indices: P[i,j], Q[i,j], R[i,j]
        # Global mapping:
        # Term t has offsets: a_co (A col), b_ro (B row), b_co (B col), c_co (C col)
        # A global idx: i * m + (a_co + j)
        # B global idx: (b_ro + i) * p + (b_co + j)
        # C global idx: (c_co + j) * n + i  (W[k*n+i] where k=c_co+j, i=i)
        
        # Strassen intermediates M1..M7
        # M1 = (P00+P11)(Q00+Q11) -> R00 += M1, R11 += M1
        # M2 = (P10+P11)Q00       -> R10 += M2, R11 -= M2
        # M3 = P00(Q01-Q11)       -> R01 += M3, R11 += M3
        # M4 = P11(Q10-Q00)       -> R00 += M4, R10 += M4
        # M5 = (P00+P01)Q11       -> R00 -= M5, R01 += M5
        # M6 = (P10-P00)(Q00+Q01) -> R11 += M6
        # M7 = (P01-P11)(Q10+Q11) -> R00 += M7
        
        # We define each intermediate by its U, V, W contributions
        # U_contrib: list of (local_i, local_j, val)
        # V_contrib: list of (local_i, local_j, val)
        # W_contrib: list of (local_i, local_j, val)
        
        components = [
            # M1
            ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ]),
            # M2
            ([ (1,0,1), (1,1,1) ], [ (0,0,1) ], [ (1,0,1), (1,1,-1) ]),
            # M3
            ([ (0,0,1) ], [ (0,1,1), (1,1,-1) ], [ (0,1,1), (1,1,1) ]),
            # M4
            ([ (1,1,1) ], [ (1,0,1), (0,0,-1) ], [ (0,0,1), (1,0,1) ]),
            # M5
            ([ (0,0,1), (0,1,1) ], [ (1,1,1) ], [ (0,0,-1), (0,1,1) ]),
            # M6
            ([ (1,0,1), (0,0,-1) ], [ (0,0,1), (0,1,1) ], [ (1,1,1) ]),
            # M7
            ([ (0,1,1), (1,1,-1) ], [ (1,0,1), (1,1,1) ], [ (0,0,1) ]),
        ]
        
        # The 4 block terms
        # A0*B00 -> C0, A1*B10 -> C0, A0*B01 -> C1, A1*B11 -> C1
        # A0: cols 0,1 (offset 0). A1: cols 2,3 (offset 2)
        # B00: rows 0,1 (off 0), cols 0,1 (off 0). B10: rows 2,3 (off 2), cols 0,1 (off 0)
        # B01: rows 0,1 (off 0), cols 2,3 (off 2). B11: rows 2,3 (off 2), cols 2,3 (off 2)
        # C0: cols 0,1 (off 0). C1: cols 2,3 (off 2)
        
        terms = [
            (0, 0, 0, 0), # A0, B00, C0
            (2, 2, 0, 0), # A1, B10, C0
            (0, 0, 2, 2), # A0, B01, C1
            (2, 2, 2, 2), # A1, B11, C1
        ]
        
        col = 0
        for (a_co, b_ro, b_co, c_co) in terms:
            for (u_c, v_c, w_c) in components:
                for (li, lj, val) in u_c:
                    gi = li
                    gj = a_co + lj
                    U[gi * m + gj, col] += val
                for (li, lj, val) in v_c:
                    gi = b_ro + li
                    gj = b_co + lj
                    V[gi * p + gj, col] += val
                for (li, lj, val) in w_c:
                    gi = li
                    gj = c_co + lj
                    # W index: k*n + i. Here k is col index of C, i is row index of C.
                    # In our local 2x2 C block, li is row, lj is col.
                    # Global row i = li (since n=2, rows are 0,1)
                    # Global col k = c_co + lj
                    W[(c_co + lj) * n + li, col] += val
                col += 1
        
        return U, V, W

    def try_rank_27_randomized():
        """
        Attempts to find a rank-27 decomposition.
        Strategy:
        1. Start from a rank-28 solution.
        2. Try to "merge" two rank-1 components into one, or find a dependency.
        3. Alternatively, use a randomized algebraic solver.
        
        Given the difficulty, we will try a simple perturbation + rounding approach first.
        If that fails, we will try to solve for a specific structure.
        
        Actually, a more robust approach for "randomized algebraic solver":
        Assume the solution exists with integer/half-integer entries.
        The space of solutions is an algebraic variety.
        We can try to find a point in this variety with small coordinates.
        
        Let's try a different tactic:
        Use the rank-28 solution.
        Check if any column is a linear combination of others.
        If U[:, r1] = sum c_i U[:, r_i], then V[:, r1] must be related, etc.
        If we find such a dependency, we can potentially eliminate a column.
        
        However, for a minimal decomposition, no such dependency exists.
        For rank 27, we need a different structure.
        
        Let's try a heuristic search:
        Randomly initialize U, V, W with small integers/half-integers.
        Use gradient descent to minimize the error.
        If we find a solution with error < 1e-6, round and verify.
        """
        R = 27
        best_err = float('inf')
        best_params = None
        
        # We will use scipy.optimize.minimize if available, or a simple gradient step.
        # Given time constraints, let's try a few random restarts with a simple optimizer.
        
        from scipy.optimize import minimize
        
        def objective(params):
            U_flat = params[:R*nm].reshape(R, nm)
            V_flat = params[R*nm:R*nm+R*mp].reshape(R, mp)
            W_flat = params[R*nm+R*mp:R*nm+R*mp+R*pn].reshape(R, pn)
            # Note: check_identity expects (nm, R), (mp, R), (pn, R)
            return check_identity(U_flat.T, V_flat.T, W_flat.T, R)
        
        num_restarts = 5
        for _ in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Initialize with small random values
            # Maybe initialize near the rank-28 solution?
            # We can't easily map rank-28 to rank-27.
            # Let's try random init scaled down.
            U0 = np.random.randn(R, nm) * 0.5
            V0 = np.random.randn(R, mp) * 0.5
            W0 = np.random.randn(R, pn) * 0.5
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                # Use L-BFGS-B
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'maxfun': 200, 'ftol': 1e-12})
                if res.fun < best_err:
                    best_err = res.fun
                    best_params = res.x
                if best_err < 1e-8:
                    break
            except Exception:
                pass
        
        if best_params is not None and best_err < 1e-6:
            U_opt = best_params[:R*nm].reshape(R, nm).T
            V_opt = best_params[R*nm:R*nm+R*mp].reshape(R, mp).T
            W_opt = best_params[R*nm+R*mp:R*nm+R*mp+R*pn].reshape(R, pn).T
            
            # Round to half-integers
            U_r = np.round(U_opt * 2) / 2.0
            V_r = np.round(V_opt * 2) / 2.0
            W_r = np.round(W_opt * 2) / 2.0
            
            err = check_identity(U_r, V_r, W_r, R)
            if err < 1e-9:
                return U_r, V_r, W_r
        
        return None

    # 1. Try to find rank 27
    result_27 = try_rank_27_randomized()
    if result_27 is not None:
        U, V, W = result_27
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    # 2. Fallback to rank 28 (Strassen)
    if time.time() > deadline - 5:
        # Just return schoolbook if we are really out of time? No, Strassen is fast.
        pass

    U, V, W = strassen_rank_28()
    
    # Verify Strassen solution just in case
    err = check_identity(U, V, W, 28)
    if err > 1e-9:
        # This shouldn't happen, but fallback to schoolbook
        R = n * m * p
        U_s = np.zeros((nm, R))
        V_s = np.zeros((mp, R))
        W_s = np.zeros((pn, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m + j, col] = 1
                    V_s[j*p + k, col] = 1
                    W_s[k*n + i, col] = 1
                    col += 1
        return {
            "U": U_s.tolist(),
            "V": V_s.tolist(),
            "W": W_s.tolist()
        }

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
