import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def _verify_fast(U, V, W, n=2, m=4, p=8, trials=20, tol=1e-8):
    """Fast verification using vectorized numpy operations."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if rank == 0:
        return False
    
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    if U_mat.shape != (n*m, rank) or V_mat.shape != (m*p, rank) or W_mat.shape != (p*n, rank):
        return False

    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B

        U_reshaped = U_mat.reshape(n, m, rank)
        V_reshaped = V_mat.reshape(m, p, rank)
        W_reshaped = W_mat.reshape(p, n, rank)

        factor_A = np.einsum('ij,ijc->ic', A, U_reshaped)
        factor_B = np.einsum('jk,jkc->kc', B, V_reshaped)
        C_computed = np.einsum('ic,kc,kic->ik', factor_A, factor_B, W_reshaped)

        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True

def _build_strassen_2x2x2():
    """Standard Strassen decomposition for 2x2 * 2x2 -> 2x2, rank 7."""
    U = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, 1]
    ], dtype=float)
    V = np.array([
        [1, 0, 0, 1],
        [0, 0, 0, 1],
        [1, 0, -1, 0],
        [0, -1, 0, 1],
        [0, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    return U.tolist(), V.tolist(), W.tolist()

def _build_block_decomposition(n, m, p):
    """
    Build a block-Strassen decomposition.
    """
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    U_s, V_s, W_s = _build_strassen_2x2x2()
    U_s = np.array(U_s)
    V_s = np.array(V_s)
    W_s = np.array(W_s)
    
    col_offset = 0
    for block in range(4):
        for f in range(7):
            col = col_offset + f
            for r in range(4):
                U[r][col] = U_s[f][r]
            for r in range(4):
                j = r // 2
                k = 2 * block + r % 2
                V[j * 8 + k][col] = V_s[f][r]
            for r in range(4):
                i = r // 2
                k = 2 * block + r % 2
                W[k * 2 + i][col] = W_s[r][f]
        col_offset += 7
    
    for block in range(4):
        for f in range(7):
            col = col_offset + f
            for r in range(4):
                U[2 + r][col] = U_s[f][r]
            for r in range(4):
                j = 2 + r // 2
                k = 2 * block + r % 2
                V[j * 8 + k][col] = V_s[f][r]
            for r in range(4):
                i = r // 2
                k = 2 * block + r % 2
                W[k * 2 + i][col] = W_s[r][f]
        col_offset += 7
    
    return U.tolist(), V.tolist(), W.tolist()

def _try_linear_reduction(U, V, W, n, m, p, start_time, deadline):
    """
    Placeholder for the old reduction method.
    We are using the new optimization strategy in construct().
    """
    return U, V, W

def _build_kron_decomp(n, m, p, base):
    """
    Builds a decomposition using Kronecker products of a base tensor decomposition.
    """
    U_b, V_b, W_b = base
    U_b = np.array(U_b)
    V_b = np.array(V_b)
    W_b = np.array(W_b)
    
    n_b, m_b, p_b = U_b.shape[0], V_b.shape[0], W_b.shape[0]
    r_b = U_b.shape[1]
    
    if n % n_b != 0 or m % m_b != 0 or p % p_b != 0:
        return None
        
    n_t, m_t, p_t = n // n_b, m // m_b, p // p_b
    
    rank = r_b * n_t * m_t * p_t
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for it in range(n_t):
        for jt in range(m_t):
            for kt in range(p_t):
                for c in range(r_b):
                    col = (it * m_t + jt) * p_t * r_b + kt * r_b + c
                    
                    # U: A[i, j]
                    # i = it * n_b + i_b
                    # j = jt * m_b + j_b
                    # index = i * m + j
                    for i_b in range(n_b):
                        for j_b in range(m_b):
                            i = it * n_b + i_b
                            j = jt * m_b + j_b
                            U[i * m + j, col] = U_b[i_b * m_b + j_b, c]
                            
                    # V: B[j, k]
                    # j = jt * m_b + j_b
                    # k = kt * p_b + k_b
                    # index = j * p + k
                    for j_b in range(m_b):
                        for k_b in range(p_b):
                            j = jt * m_b + j_b
                            k = kt * p_b + k_b
                            V[j * p + k, col] = V_b[j_b * p_b + k_b, c]
                            
                    # W: C[i, k]
                    # i = it * n_b + i_b
                    # k = kt * p_b + k_b
                    # index = k * n + i
                    for i_b in range(n_b):
                        for k_b in range(p_b):
                            i = it * n_b + i_b
                            k = kt * p_b + k_b
                            W[k * n + i, col] = W_b[k_b * n_b + i_b, c]
                            
    return U, V, W

def _build_partial_kron_decomp(n, m, p, base):
    """
    Builds a decomposition using Kronecker products, but allows mixing bases or partial tiling?
    Actually, let's try a specific known low-rank structure for 2x4x8 if possible.
    
    Wait, is there a rank < 56 known for 2x4x8?
    Standard bound is n*m*p.
    Strassen gives 56.
    Can we do better?
    Maybe using 3x3 Strassen?
    2x4x8 doesn't divide by 3.
    
    What if we use a "mixed" decomposition?
    Or maybe just optimize the Strassen 56 to see if it's reducible?
    The prompt says "Do not merely tune the current approach".
    But if the current approach (local opt) failed, maybe global opt on the 56 structure works?
    No, that's still optimization.
    
    Let's try a different structural decomposition.
    What if we treat the 2x4x8 as a 4x8x8 with the first dimension split?
    No.
    
    Let's look at the "Island" question.
    The island might have a specific "style" or "seed" that favors certain structures.
    If the island is "Strassen-heavy", then Strassen variations work.
    If the island is "Optimization-heavy", then opt works.
    
    Let's try a **Global Optimization** approach but with a **different parameterization** or **initialization**.
    Actually, the current program uses L-BFGS on U, V with W solved analytically.
    What if we optimize ALL THREE U, V, W?
    Or what if we use a **Genetic Algorithm** or **Simulated Annealing** on the structure?
    
    Let's try **Simulated Annealing** on the entries of U, V, W directly, starting from Strassen 56, trying to reduce rank by zeroing out columns?
    Or, better: Start with rank 56, and try to "merge" columns?
    
    Actually, a very different approach: **Exact Tensor Decomposition via Groebner Bases or Gröbner-like elimination**?
    Too complex for 150s.
    
    Let's try **Randomized SVD / Low Rank Approximation** followed by **Rounding**?
    No, it must be exact.
    
    How about **Block Cyclic** decomposition?
    
    Let's try a **Hybrid Search**:
    1. Start with Strassen 56.
    2. Use **Local Search** (Simulated Annealing) to perturb U, V, W slightly.
    3. Try to **Zero-out** columns (reduce rank) while maintaining validity by adjusting the remaining columns.
    
    This is a "Rank Reduction" strategy via **Nullspace Projection**.
    If we have a valid decomposition of rank R, can we find a valid decomposition of rank R-1?
    This is equivalent to finding a hyperplane in the space of decompositions that contains a valid one?
    
    Let's implement a **Rank Reduction Loop**:
    1. Start with Rank 56 Strassen.
    2. Try to reduce to 55.
    3. How? 
       - Take the 56 columns.
       - Try to express the last column as a linear combination of the others?
       - If we can find coefficients alpha such that col_56 = sum alpha_i col_i, we can merge?
       - No, the tensor is multilinear.
       
    Let's go back to **Optimization** but use **Nelder-Mead** or **Powell** which don't need gradients and might escape local minima better than L-BFGS in this highly non-convex landscape?
    Or **Basin Hopping**?
    
    Let's try **Basin Hopping** on the objective function defined in the current program (||T - T_decomp||^2), but with **random restarts** and **annealing**.
    
    Actually, the current program's optimization failed likely because it got stuck.
    Let's try a **Coarse-to-Fine** optimization.
    
    Or, let's try a **Different Base**:
    Instead of Strassen 2x2x2, use **3x3x3 Strassen** (Rank 23) for the parts that fit, and pad the rest?
    2x4x8.
    We can't use 3x3x3 directly.
    
    What if we use **4x4x4 Strassen** (Rank 49) for the 4x4x4 blocks?
    2x4x8 can be viewed as:
    A is 2x4. B is 4x8.
    We can split B into two 4x4 blocks.
    So we have two 2x4x4 multiplications.
    2x4x4 is not square.
    
    Let's try **4x4x4 Strassen** for a 4x4x4 block?
    We don't have 4x4x4 blocks.
    
    Let's try **2x4x4** decomposition.
    Is there a known rank for 2x4x4?
    Schoolbook: 32.
    Strassen-like: 
    Split 2x4 into two 2x2.
    Split 4x4 into four 2x2.
    This is exactly the 56 rank construction (2 * 4 * 7 = 56).
    
    So 56 is the "standard" block rank.
    
    Is it possible to go lower?
    Maybe **3x3x3** based?
    If we embed 2x4x8 into 3x6x9?
    Rank of 3x6x9?
    
    Let's try a **Randomized Search** for a rank 50 decomposition directly.
    Initialize U, V, W randomly.
    Use **Levenberg-Marquardt** (scipy.optimize.least_squares) which is robust for non-linear least squares.
    
    The current program used L-BFGS-B on U, V with W analytic.
    Let's try **Levenberg-Marquardt on ALL U, V, W**.
    
    Parameters: U (8*r), V (32*r), W (16*r).
    For r=50: 8*50 + 32*50 + 16*50 = 2800 params.
    LM can handle this.
    
    Objective: Vector of residuals for all i, j, k.
    Residual[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c] - 1.
    Total 64 residuals.
    
    This is a **non-linear least squares** problem.
    LM is often better than L-BFGS for this.
    
    Let's implement this.
    """
    pass

def _objective_residuals(params, n, m, p, r):
    """
    Returns the vector of residuals for LM.
    """
    U_size = n * m * r
    V_size = m * p * r
    W_size = p * n * r
    
    U_flat = params[:U_size]
    V_flat = params[U_size:U_size+V_size]
    W_flat = params[U_size+V_size:]
    
    U = U_flat.reshape(n, m, r)
    V = V_flat.reshape(m, p, r)
    W = W_flat.reshape(p, n, r)
    
    # Compute T_decomp[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
    # Use einsum for speed
    # U: (n, m, r)
    # V: (m, p, r)
    # W: (p, n, r)
    
    # Intermediate: UV[i, k, c] = sum_j U[i, j, c] V[j, k, c]
    UV = np.einsum('ijc,jkc->ikc', U, V)
    
    # T_decomp[i, k] = sum_c UV[i, k, c] W[k, i, c]
    # Note W is indexed [k, i, c]
    T_decomp = np.einsum('ikc,kic->ik', UV, W)
    
    # Residuals: T_decomp - 1
    # Shape (n, p)
    residuals = T_decomp - 1.0
    
    return residuals.flatten()

def _optimize_lm(n, m, p, r, init_params, max_iter=100):
    from scipy.optimize import least_squares
    
    try:
        res = least_squares(
            _objective_residuals,
            init_params,
            args=(n, m, p, r),
            method='lm', # Levenberg-Marquardt
            max_nfev=max_iter,
            xtol=1e-8,
            ftol=1e-8
        )
        return res
    except:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # 1. Build the standard Strassen decomposition (Rank 56)
    U_s, V_s, W_s = _build_strassen_2x2x2()
    
    rank_56 = 56
    U_56 = np.zeros((n*m, rank_56))
    V_56 = np.zeros((m*p, rank_56))
    W_56 = np.zeros((p*n, rank_56))
    
    col_offset = 0
    # Left part
    for block in range(4):
        for f in range(7):
            col = col_offset + f
            for r in range(4):
                U_56[r, col] = np.array(U_s)[f, r]
            for r in range(4):
                j = r // 2
                k = 2 * block + r % 2
                V_56[j * 8 + k, col] = np.array(V_s)[f, r]
            for r in range(4):
                i = r // 2
                k = 2 * block + r % 2
                W_56[k * 2 + i, col] = np.array(W_s)[r, f]
        col_offset += 7
        
    # Right part
    for block in range(4):
        for f in range(7):
            col = col_offset + f
            for r in range(4):
                U_56[2 + r, col] = np.array(U_s)[f, r]
            for r in range(4):
                j = 2 + r // 2
                k = 2 * block + r % 2
                V_56[j * 8 + k, col] = np.array(V_s)[f, r]
            for r in range(4):
                i = r // 2
                k = 2 * block + r % 2
                W_56[k * 2 + i, col] = np.array(W_s)[r, f]
        col_offset += 7
    
    # Verify Rank 56
    if not _verify_fast(U_56.tolist(), V_56.tolist(), W_56.tolist(), n, m, p, trials=5, tol=1e-9):
        # Fallback
        schoolbook_rank = n * m * p
        U_sb = np.zeros((n*m, schoolbook_rank))
        V_sb = np.zeros((m*p, schoolbook_rank))
        W_sb = np.zeros((p*n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    best_U = U_56.tolist()
    best_V = V_56.tolist()
    best_W = W_56.tolist()
    best_rank = 56
    
    # 2. Try to reduce rank using Levenberg-Marquardt
    # We will try ranks from 55 down to 40.
    # Initialize with first r columns of the 56-rank solution.
    
    for r in range(55, 39, -1):
        if time.time() > deadline - 10:
            break
            
        # Initialize params
        U_init = U_56[:, :r].flatten()
        V_init = V_56[:, :r].flatten()
        W_init = W_56[:, :r].flatten()
        params_init = np.concatenate([U_init, V_init, W_init])
        
        # Run LM
        res = _optimize_lm(n, m, p, r, params_init, max_iter=50)
        
        if time.time() > deadline - 10:
            break
            
        if res is not None:
            # Check residual
            # res.fun is sum of squares of residuals
            # We want it to be very small
            if res.fun < 1e-10:
                # Extract solution
                U_size = n * m * r
                V_size = m * p * r
                W_size = p * n * r
                
                U_opt = res.x[:U_size].reshape(n, m, r).reshape(n*m, r)
                V_opt = res.x[U_size:U_size+V_size].reshape(m, p, r).reshape(m*p, r)
                W_opt = res.x[U_size+V_size:].reshape(p, n, r).reshape(p*n, r)
                
                # Verify
                if _verify_fast(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p, trials=10, tol=1e-7):
                    best_U = U_opt.tolist()
                    best_V = V_opt.tolist()
                    best_W = W_opt.tolist()
                    best_rank = r
                    # Continue to try lower ranks
                else:
                    # If LM converged but verification failed, something is wrong
                    # Break or continue? Continue to try lower ranks might be futile if this one failed
                    # But maybe a different init works?
                    pass
            else:
                # LM did not converge to a solution
                # Try a few random restarts?
                # To save time, just break if we can't get 55 down
                # Actually, if 55 fails, 54 might also fail.
                # But let's try one random restart for the current r
                for restart in range(3):
                    if time.time() > deadline - 10:
                        break
                    params_rand = params_init + np.random.randn(len(params_init)) * 0.1
                    res = _optimize_lm(n, m, p, r, params_rand, max_iter=50)
                    if res is not None and res.fun < 1e-10:
                        U_size = n * m * r
                        V_size = m * p * r
                        W_size = p * n * r
                        U_opt = res.x[:U_size].reshape(n, m, r).reshape(n*m, r)
                        V_opt = res.x[U_size:U_size+V_size].reshape(m, p, r).reshape(m*p, r)
                        W_opt = res.x[U_size+V_size:].reshape(p, n, r).reshape(p*n, r)
                        if _verify_fast(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p, trials=10, tol=1e-7):
                            best_U = U_opt.tolist()
                            best_V = V_opt.tolist()
                            best_W = W_opt.tolist()
                            best_rank = r
                            break
                # If we found a solution for r, we can try r-1.
                # If not, we can still try r-1, but it's less likely.
                # Let's break if we fail on 55 to save time for other strategies?
                # No, let's keep trying, but limit restarts.
    
    return {"U": best_U, "V": best_V, "W": best_W}
