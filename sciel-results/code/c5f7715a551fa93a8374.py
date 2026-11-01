import numpy as np
import time
import itertools

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 140.0

    size_u = n * m   # 4
    size_v = m * p   # 18
    size_w = p * n   # 18
    n_tensors = n * m * p  # 36

    def verify(U, V, W):
        """Verify exact decomposition. Returns True if valid."""
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True

    def build_schoolbook():
        """Standard schoolbook decomposition with rank n*m*p = 36."""
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def build_strassen_block(k_start, k_end, offset):
        """
        Build Strassen decomposition for a 2x2 * 2x2 block (columns k_start to k_end-1).
        Returns U_block (4, 7), V_block (4, 7), W_block (4, 7) 
        but mapped to the global indices.
        
        Global indices:
        U: [i, j] -> i*m + j
        V: [j, k] -> j*p + k
        W: [k, i] -> k*n + i
        
        Strassen terms for local (i,j,k) in {0,1}^3:
        T0: (A00+A11)(B00+B11) -> C00, C11
        T1: (A10+A11)B00 -> C10
        T2: A00(B10-B00) -> C00, C10
        T3: A11(B01-B11) -> C01, C11
        T4: (A00+A01)B11 -> C00, C01
        T5: (A10-A00)(B00+B01) -> C10, C11 (Wait, standard Strassen has 7 terms, let's use a verified set)
        
        Let's use the standard 7-term Strassen:
        m1 = (a00+a11)(b00+b11)
        m2 = (a10+a11)b00
        m3 = a00(b10-b00)
        m4 = a11(b01-b11)
        m5 = (a00+a01)b11
        m6 = (a10-a00)(b00+b01)
        m7 = (a01-a11)(b10+b11)
        
        c00 = m1 + m3 - m5 - m7
        c01 = m4 + m5
        c10 = m2 + m3
        c11 = m1 - m2 + m4 + m7
        
        Local U (4x7):
        Row 0 (A00): [1, 0, 1, 0, 1, -1, 0]
        Row 1 (A01): [0, 0, 0, 0, 1, 0, 1]
        Row 2 (A10): [0, 1, 0, 0, 0, 1, 0]
        Row 3 (A11): [1, 1, 0, 1, 0, 0, -1]
        
        Local V (4x7):
        Row 0 (B00): [1, 1, -1, 0, 0, 1, 0]
        Row 1 (B01): [0, 0, 0, 1, 0, 1, 0]
        Row 2 (B10): [0, 0, 1, 0, 0, 0, 1]
        Row 3 (B11): [1, 0, 0, -1, 1, 0, 1]
        
        Local W (4x7):
        Row 0 (C00): [1, 0, 1, 0, -1, 0, -1]
        Row 1 (C01): [0, 0, 0, 1, 1, 0, 0]
        Row 2 (C10): [0, 1, 1, 0, 0, 0, 0]
        Row 3 (C11): [1, -1, 0, 1, 0, 0, 1]
        
        Let's verify one: C00 = m1 + m3 - m5 - m7
        m1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1] -> C00 gets 1
        m3: U=[1,0,0,0], V=[0,0,1,-1], W=[1,0,1,0] -> C00 gets 1
        m5: U=[1,1,0,0], V=[0,0,0,1], W=[-1,1,0,0] -> C00 gets -1
        m7: U=[0,1,0,-1], V=[0,0,1,1], W=[-1,0,0,1] -> C00 gets -1
        Sum: 1+1-1-1 = 0? No, the tensor T[i,j,k] is 1 for all i,j,k.
        The decomposition is for the tensor T, not just the output C.
        T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        
        Let's re-verify the Strassen decomposition for the tensor T[i,j,k]=1.
        This is the standard matrix multiplication tensor.
        The rank is 7.
        """
        U_local = np.array([
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ])
        V_local = np.array([
            [1, 1, -1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, -1, 1, 0, 1]
        ])
        W_local = np.array([
            [1, 0, 1, 0, -1, 0, -1],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 0],
            [1, -1, 0, 1, 0, 0, 1]
        ])
        
        # Verify local decomposition
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    val = np.dot(U_local[i, :], V_local[j, :] * W_local[k, :])
                    if abs(val - 1.0) > 1e-8:
                        # If verification fails, return None to signal error
                        return None, None, None
        
        U_block = np.zeros((n * m, 7))
        V_block = np.zeros((m * p, 7))
        W_block = np.zeros((p * n, 7))
        
        for i in range(2):
            for j in range(2):
                for r in range(7):
                    U_block[i * m + j, r] = U_local[i, r]
                    # V and W depend on k
                    for k in range(2):
                        V_block[j * p + (k_start + k), r] = V_local[j, r]
                        W_block[(k_start + k) * n + i, r] = W_local[k, r]
                        
        return U_block, V_block, W_block

    def build_hybrid_baseline():
        """
        Build a baseline decomposition using Strassen for the first 4 columns (2 blocks of 2)
        and schoolbook for the remaining 5 columns.
        Rank: 7 + 7 + 5*4 = 35.
        Actually, let's try 4 Strassen blocks (8 columns) + 1 schoolbook (1 column) = 28 + 4 = 32.
        """
        R = 32
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        col_offset = 0
        # 4 Strassen blocks for k=0..7
        for block_idx in range(4):
            k_start = block_idx * 2
            k_end = k_start + 2
            U_b, V_b, W_b = build_strassen_block(k_start, k_end, col_offset)
            if U_b is None:
                return build_schoolbook()
            U[:, col_offset:col_offset+7] = U_b
            V[:, col_offset:col_offset+7] = V_b
            W[:, col_offset:col_offset+7] = W_b
            col_offset += 7
        
        # 1 schoolbook block for k=8
        for i in range(n):
            for j in range(m):
                for k in range(1): # just k=8
                    U[i * m + j, col_offset] = 1
                    V[j * p + (8 + k), col_offset] = 1
                    W[(8 + k) * n + i, col_offset] = 1
                    col_offset += 1
        
        if col_offset != R:
            return build_schoolbook()
            
        if verify(U, V, W):
            return U, V, W
        else:
            return build_schoolbook()

    # Start with the best known baseline (Rank 32)
    U, V, W = build_hybrid_baseline()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W

    # Attempt to reduce rank using structured optimization
    # The tensor is T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # We can fix U and V, and solve for W, or vice versa.
    # Or use a joint optimization.
    
    # Strategy: Try to reduce rank by removing columns from the factor matrices
    # and re-optimizing the remaining ones.
    
    def try_reduce_rank(U_init, V_init, W_init, target_rank):
        """
        Try to find a rank-target_rank decomposition by projecting the initial
        high-rank decomposition and optimizing.
        """
        R = target_rank
        # Random projection to get initial guess for lower rank
        np.random.seed(42)
        P = np.random.randn(U_init.shape[1], R)
        
        U_proj = U_init @ P
        V_proj = V_init @ P
        W_proj = W_init @ P
        
        # Optimize W given U and V
        # For each (i,j,k), we need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
        # This is linear in W.
        # Let's set up a linear system for W.
        # Variables: W[k*n+i, r] for all k,i,r.
        # Equations: for each i,j,k: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
        
        # This is a system of n*m*p equations with R*p*n unknowns.
        # Since R is small, we can solve this least-squares problem.
        
        # Construct A matrix for W
        # A[idx, var] where idx is (i,j,k) and var is (k,i,r)
        # A[idx, var] = U[i*m+j, r] * V[j*p+k, r]
        
        A = np.zeros((n * m * p, R * p * n))
        b = np.ones(n * m * p)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(R):
                        var_idx = (k * n + i) * R + r
                        A[idx, var_idx] = U_proj[i * m + j, r] * V_proj[j * p + k, r]
                    idx += 1
        
        # Solve for W_flat
        try:
            W_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            W_new = W_flat.reshape(p * n, R)
        except Exception:
            return None, None, None
        
        # Now we have U_proj, V_proj, W_new.
        # We can also optimize U and V given W.
        # Let's do a few iterations of alternating least squares (ALS).
        
        U_curr = U_proj
        V_curr = V_proj
        W_curr = W_new
        
        for _ in range(10):
            # Optimize U given V, W
            # Equations: for each i,j,k: sum_r U[i*m+j, r] (V[j*p+k, r] W[k*n+i, r]) = 1
            # Variables: U[i*m+j, r]
            # For fixed i,j, the equation involves k.
            # Actually, for each (i,j), we have p equations (one for each k).
            # sum_r U[i*m+j, r] * (V[j*p+k, r] W[k*n+i, r]) = 1
            # This is a linear system for U[i*m+j, :]
            
            U_new = np.zeros_like(U_curr)
            for i in range(n):
                for j in range(m):
                    # Construct A_u for this (i,j)
                    # Rows: k (p of them)
                    # Cols: r (R of them)
                    A_u = np.zeros((p, R))
                    for k in range(p):
                        for r in range(R):
                            A_u[k, r] = V_curr[j * p + k, r] * W_curr[k * n + i, r]
                    b_u = np.ones(p)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                        U_new[i * m + j, :] = sol
                    except Exception:
                        return None, None, None
            U_curr = U_new
            
            # Optimize V given U, W
            V_new = np.zeros_like(V_curr)
            for j in range(m):
                for k in range(p):
                    A_v = np.zeros((n, R))
                    for i in range(n):
                        for r in range(R):
                            A_v[i, r] = U_curr[i * m + j, r] * W_curr[k * n + i, r]
                    b_v = np.ones(n)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                        V_new[j * p + k, :] = sol
                    except Exception:
                        return None, None, None
            V_curr = V_new
            
            # Optimize W given U, V
            W_new = np.zeros_like(W_curr)
            for k in range(p):
                for i in range(n):
                    A_w = np.zeros((m, R))
                    for j in range(m):
                        for r in range(R):
                            A_w[j, r] = U_curr[i * m + j, r] * V_curr[j * p + k, r]
                    b_w = np.ones(m)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
                        W_new[k * n + i, :] = sol
                    except Exception:
                        return None, None, None
            W_curr = W_new
            
            # Check convergence
            if verify(U_curr, V_curr, W_curr):
                return U_curr, V_curr, W_curr
        
        if verify(U_curr, V_curr, W_curr):
            return U_curr, V_curr, W_curr
        
        return None, None, None

    # Try to reduce rank from 32 down to 28
    for R in range(31, 27, -1):
        if time.time() > deadline - 20:
            break
        
        U_new, V_new, W_new = try_reduce_rank(best_U, best_V, best_W, R)
        if U_new is not None and verify(U_new, V_new, W_new):
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = R
            print(f"Reduced rank to {R}")
            break # If we find one, try to go lower? Or just take it? 
            # Let's try to go lower in the next iteration of the loop if possible,
            # but the loop structure above tries 31, then 30, etc.
            # So if 31 works, we break and return 31.
            # To try lower, we should not break, but continue to 30.
            # However, if 31 works, 30 might be harder. Let's just take the first success.
            # Actually, the loop goes 31, 30, 29, 28.
            # If 31 works, we update best and break.
            # If we want to try 30, we shouldn't break.
            # Let's change logic: try all, keep the best.
            
    # Refine: Try all ranks from 31 down to 28 and keep the best
    # The above loop breaks on first success. Let's make it try all.
    # But for speed, breaking on first success is usually fine if we start high.
    # If we want the *minimal* rank, we should try the lowest first? No, higher rank is easier to solve.
    # So starting from 31 and going down is good. If 31 works, we have rank 31.
    # If we don't break, we might find 30.
    # Let's modify to not break, but update best if found.
    
    # Re-run the loop without break to find the lowest possible
    for R in range(31, 27, -1):
        if time.time() > deadline - 20:
            break
        
        U_new, V_new, W_new = try_reduce_rank(best_U, best_V, best_W, R)
        if U_new is not None and verify(U_new, V_new, W_new):
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = R
            # Do not break, try to go lower

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
