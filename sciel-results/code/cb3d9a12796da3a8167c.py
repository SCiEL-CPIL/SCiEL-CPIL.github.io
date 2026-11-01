<<<<<<< SEARCH
def _try_rank54_exploit_n2(n, m, p, time_budget, seed=42):
    """
    Try to find a rank-54 decomposition by exploiting the n=2 structure.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    
    base = _strassen_rank55(n, m, p)
    U0 = np.array(base["U"], dtype=float)
    V0 = np.array(base["V"], dtype=float)
    W0 = np.array(base["W"], dtype=float)
    
    r54 = 54
    
    best = None
    
    for attempt in range(200):
        if time.time() > deadline - 10:
            break
        
        U = np.zeros((n*m, r54))
        V = np.zeros((m*p, r54))
        
        strat = attempt % 4
        
        if strat == 0:
            for row in range(n*m):
                nnz = rng.randint(2, min(6, r54))
                cols = rng.choice(r54, size=nnz, replace=False)
                U[row, cols] = rng.choice([-1, 0, 1], size=nnz)
            for row in range(m*p):
                nnz = rng.randint(2, min(6, r54))
                cols = rng.choice(r54, size=nnz, replace=False)
                V[row, cols] = rng.choice([-1, 0, 1], size=nnz)
                
        elif strat == 1:
            U[:, :r54] = U0[:, :r54] * rng.randint(0, 2, size=(n*m, r54))
            V[:, :r54] = V0[:, :r54] * rng.randint(0, 2, size=(m*p, r54))
            U += rng.randint(-1, 2, size=(n*m, r54)) * 0.1
            V += rng.randint(-1, 2, size=(m*p, r54)) * 0.1
            
        elif strat == 2:
            U = rng.randint(0, 2, size=(n*m, r54)) * 2 - 1
            V = rng.randint(0, 2, size=(m*p, r54)) * 2 - 1
            U[rng.random((n*m, r54)) < 0.3] = 0
            V[rng.random((m*p, r54)) < 0.3] = 0
            
        else:
            block_size = r54 // 5
            for g in range(5):
                start = g * block_size
                end = start + block_size if g < 4 else r54
                if end > start:
                    U[:, start:end] = rng.randint(0, 2, size=(n*m, end-start)) * 2 - 1
                    V[:, start:end] = rng.randint(0, 2, size=(m*p, end-start)) * 2 - 1
                    U[:, start:end][rng.random((n*m, end-start)) < 0.4] = 0
                    V[:, start:end][rng.random((m*p, end-start)) < 0.4] = 0
        
        W = np.zeros((p*n, r54))
        feasible = True
        
        for i in range(n):
            for k in range(p):
                A = np.zeros((m, r54))
                b = np.ones(m)
                for j in range(m):
                    A[j, :] = U[i*m+j] * V[j*p+k]
                
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    if not np.allclose(A @ sol, b, atol=1e-10):
                        feasible = False
                        break
                    W[k*n+i, :] = sol
                except:
                    feasible = False
                    break
            if not feasible:
                break
        
        if feasible:
            ok = True
            for i in range(n):
                if not ok: break
                for j in range(m):
                    if not ok: break
                    for k in range(p):
                        prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                        s = np.sum(prod)
                        if abs(s - 1.0) > 1e-10:
                            ok = False
                            break
            if ok:
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
                
    return best
=======
def _try_rank54_exploit_n2(n, m, p, time_budget, seed=42):
    """
    Try to find a rank-54 decomposition by exploiting the n=2 structure.
    Uses tensor flattening and exact linear algebra over rationals.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    
    base = _strassen_rank55(n, m, p)
    U0 = np.array(base["U"], dtype=float)
    V0 = np.array(base["V"], dtype=float)
    W0 = np.array(base["W"], dtype=float)
    
    r54 = 54
    best = None
    
    # First, compute a lower bound using tensor flattening
    # The multiplication tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1
    # For rank R, we need R such that for all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Flattening approach: fix W, then the system in U and V is bilinear.
    # Instead, use the following: for each column c, define vectors u_c, v_c, w_c.
    # The constraint is: for all (i,j,k), sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1
    
    # Key insight: with n=2, we have 2*10=20 W-rows. If we fix U and V,
    # each W[k*n+i] must satisfy: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all j.
    # This gives m=3 linear equations in r54 unknowns for each of the 20 W-rows.
    # So the system is underdetermined (20*3=60 equations, 20*54=1080 unknowns).
    
    # Better strategy: use the Strassen structure as a starting point and try
    # to merge two of the 55 rank-1 terms into a single term with half-integer coefficients.
    
    # Try: take the rank-55 decomposition and see if we can reduce it by 1.
    # The 55 terms come from 5 Strassen blocks (7 each = 35) + 20 schoolbook terms.
    # Try to find a linear dependency among the 55 rank-1 tensors.
    
    # Build the 55x(6*30*20) matrix representation is too large.
    # Instead, work with the flattening: for each term c, the tensor is
    # u_c (size 6) outer v_c (size 30) outer w_c (size 20).
    # The sum of all 55 such outer products equals the multiplication tensor.
    
    # To find rank-54, we need to find coefficients alpha_0,...,alpha_54 (not all zero)
    # such that sum_c alpha_c * (u_c outer v_c outer w_c) = 0, which would let us
    # eliminate one term. But this is a 3-tensor, not a matrix, so this is hard.
    
    # Alternative: directly search for rank-54 by random U,V with small integer entries,
    # then solve for W exactly.
    
    for attempt in range(300):
        if time.time() > deadline - 10:
            break
        
        U = np.zeros((n*m, r54))
        V = np.zeros((m*p, r54))
        
        strat = attempt % 6
        
        if strat == 0:
            # Sparse random with {-1, 0, 1}
            for row in range(n*m):
                nnz = rng.randint(2, min(5, r54))
                cols = rng.choice(r54, size=nnz, replace=False)
                U[row, cols] = rng.choice([-1, 0, 1], size=nnz)
            for row in range(m*p):
                nnz = rng.randint(2, min(5, r54))
                cols = rng.choice(r54, size=nnz, replace=False)
                V[row, cols] = rng.choice([-1, 0, 1], size=nnz)
                
        elif strat == 1:
            # Perturb Strassen: drop one term and redistribute
            # Take first 54 terms of Strassen and try to fix
            U[:n*m, :54] = U0[:n*m, :54]
            V[:m*p, :54] = V0[:m*p, :54]
            # Add small random perturbation to allow W to adjust
            mask = rng.random((n*m, 54)) < 0.1
            U[mask] += rng.choice([-1, 1], size=np.sum(mask))
            mask = rng.random((m*p, 54)) < 0.1
            V[mask] += rng.choice([-1, 1], size=np.sum(mask))
            
        elif strat == 2:
            # Block-structured: 5 blocks of ~11 columns each
            block_size = r54 // 5
            for g in range(5):
                start = g * block_size
                end = start + block_size if g < 4 else r54
                if end > start:
                    # Mimic Strassen pattern within each block
                    U[:, start:end] = rng.randint(0, 2, size=(n*m, end-start)) * 2 - 1
                    V[:, start:end] = rng.randint(0, 2, size=(m*p, end-start)) * 2 - 1
                    U[:, start:end][rng.random((n*m, end-start)) < 0.5] = 0
                    V[:, start:end][rng.random((m*p, end-start)) < 0.5] = 0
                    
        elif strat == 3:
            # Random with {-1, 0, 1} and higher density
            U = rng.randint(0, 2, size=(n*m, r54)) * 2 - 1
            V = rng.randint(0, 2, size=(m*p, r54)) * 2 - 1
            U[rng.random((n*m, r54)) < 0.4] = 0
            V[rng.random((m*p, r54)) < 0.4] = 0
            
        elif strat == 4:
            # Use Fraction-based exact arithmetic for a few random trials
            # Generate sparse U, V with small integers
            U_int = np.zeros((n*m, r54), dtype=int)
            V_int = np.zeros((m*p, r54), dtype=int)
            for row in range(n*m):
                nnz = rng.randint(2, 4)
                cols = rng.choice(r54, size=nnz, replace=False)
                U_int[row, cols] = rng.randint(-2, 3, size=nnz)
            for row in range(m*p):
                nnz = rng.randint(2, 4)
                cols = rng.choice(r54, size=nnz, replace=False)
                V_int[row, cols] = rng.randint(-2, 3, size=nnz)
            U = U_int.astype(float)
            V = V_int.astype(float)
            
        else:
            # Hybrid: mix Strassen blocks with random
            U[:, :35] = U0[:, :35]
            V[:, :35] = V0[:, :35]
            U[:, 35:] = rng.randint(0, 2, size=(n*m, 19)) * 2 - 1
            V[:, 35:] = rng.randint(0, 2, size=(m*p, 19)) * 2 - 1
            U[:, 35:][rng.random((n*m, 19)) < 0.4] = 0
            V[:, 35:][rng.random((m*p, 19)) < 0.4] = 0
        
        # Solve for W exactly using rational arithmetic
        W = np.zeros((p*n, r54))
        feasible = True
        
        for i in range(n):
            for k in range(p):
                # Build system: for each j, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                A = np.zeros((m, r54))
                for j in range(m):
                    A[j, :] = U[i*m+j] * V[j*p+k]
                
                # Check if system is consistent: rank(A) must equal rank([A|b])
                b = np.ones(m)
                
                # Use exact fraction arithmetic for reliability
                A_frac = [[Fraction(A[j, c]) for c in range(r54)] for j in range(m)]
                b_frac = [Fraction(1)] * m
                
                # Gaussian elimination over Q
                n_rows = m
                n_cols = r54
                augmented = [row[:] + [b_frac[i]] for i, row in enumerate(A_frac)]
                
                rank = 0
                for col in range(n_cols):
                    # Find pivot
                    pivot = -1
                    for row in range(rank, n_rows):
                        if augmented[row][col] != 0:
                            pivot = row
                            break
                    if pivot == -1:
                        continue
                    
                    # Swap
                    augmented[rank], augmented[pivot] = augmented[pivot], augmented[rank]
                    
                    # Eliminate
                    pivot_val = augmented[rank][col]
                    for row in range(n_rows):
                        if row != rank and augmented[row][col] != 0:
                            factor = augmented[row][col] / pivot_val
                            for c in range(n_cols + 1):
                                augmented[row][c] -= factor * augmented[rank][c]
                    rank += 1
                    if rank == n_rows:
                        break
                
                # Check consistency
                consistent = True
                for row in range(rank, n_rows):
                    if any(augmented[row][c] != 0 for c in range(n_cols)):
                        consistent = False
                        break
                
                if not consistent:
                    feasible = False
                    break
                
                # Find a particular solution (set free variables to 0)
                sol = [Fraction(0)] * n_cols
                for row in range(rank):
                    # Find the leading column of this row
                    lead_col = -1
                    for c in range(n_cols):
                        if augmented[row][c] != 0:
                            lead_col = c
                            break
                    if lead_col != -1:
                        sol[lead_col] = augmented[row][n_cols] / augmented[row][lead_col]
                
                W[k*n+i, :] = [float(s) for s in sol]
            
            if not feasible:
                break
        
        if feasible:
            # Verify with exact arithmetic
            ok = True
            for i in range(n):
                if not ok: break
                for j in range(m):
                    if not ok: break
                    for k in range(p):
                        s = Fraction(0)
                        for c in range(r54):
                            s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                        if s != 1:
                            ok = False
                            break
            if ok:
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
                
    return best
>>>>>>> REPLACE

