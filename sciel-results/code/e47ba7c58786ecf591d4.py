import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    def compute_error(U, V, W, R):
        """
        Compute the error for the tensor decomposition.
        U: shape (n*m, R) -> U[i*m+j, r]
        V: shape (m*p, R) -> V[j*p+k, r]
        W: shape (p*n, R) -> W[k*n+i, r]
        
        Target: T_{ijk} = 1 for all i,j,k.
        Condition: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
        """
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_r[i, j], V_r[j, k] * W_r[k, i])
                    err += (val - 1.0)**2
        return err

    def is_exact_solution(U, V, W, R):
        """
        Check if the decomposition is exact.
        """
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return False
        return True

    def round_to_half_int(M):
        return np.round(M * 2) / 2.0

    def polish(U, V, W, R):
        """
        Local search to reduce error after rounding.
        Tries flipping entries to nearby half-integers.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        current_err = compute_error(U, V, W, R)
        if current_err < 1e-9:
            return U, V, W, True

        # Candidate values for half-integers in a reasonable range
        # Values are expected to be small.
        candidates = [i/2.0 for i in range(-10, 11)]
        
        improved = True
        max_iters = 200 # Increased iterations for better convergence
        iter_count = 0
        
        while improved and iter_count < max_iters and time.time() < deadline:
            improved = False
            iter_count += 1
            
            # Try flipping in U
            for idx in range(n*m):
                for r in range(R):
                    old_val = U[idx, r]
                    # Only try candidates close to old_val to save time
                    close_candidates = [c for c in candidates if abs(c - old_val) < 2.0]
                    if not close_candidates:
                        close_candidates = [c for c in candidates if abs(c - old_val) < 5.0]
                    
                    for new_val in close_candidates:
                        if np.isclose(old_val, new_val):
                            continue
                        U[idx, r] = new_val
                        new_err = compute_error(U, V, W, R)
                        if new_err < current_err - 1e-12:
                            current_err = new_err
                            improved = True
                            break
                        else:
                            U[idx, r] = old_val
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                continue

            # Try flipping in V
            for idx in range(m*p):
                for r in range(R):
                    old_val = V[idx, r]
                    close_candidates = [c for c in candidates if abs(c - old_val) < 2.0]
                    if not close_candidates:
                        close_candidates = [c for c in candidates if abs(c - old_val) < 5.0]
                    
                    for new_val in close_candidates:
                        if np.isclose(old_val, new_val):
                            continue
                        V[idx, r] = new_val
                        new_err = compute_error(U, V, W, R)
                        if new_err < current_err - 1e-12:
                            current_err = new_err
                            improved = True
                            break
                        else:
                            V[idx, r] = old_val
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                continue

            # Try flipping in W
            for idx in range(p*n):
                for r in range(R):
                    old_val = W[idx, r]
                    close_candidates = [c for c in candidates if abs(c - old_val) < 2.0]
                    if not close_candidates:
                        close_candidates = [c for c in candidates if abs(c - old_val) < 5.0]
                    
                    for new_val in close_candidates:
                        if np.isclose(old_val, new_val):
                            continue
                        W[idx, r] = new_val
                        new_err = compute_error(U, V, W, R)
                        if new_err < current_err - 1e-12:
                            current_err = new_err
                            improved = True
                            break
                        else:
                            W[idx, r] = old_val
                    if improved:
                        break
                if improved:
                    break
                    
        return U, V, W, current_err < 1e-9

    # Strassen-style structured initialization for rank 28
    # We decompose the matrix multiplication into blocks.
    # A is 2x3, B is 3x5.
    # Let A = [A11 A12 A13; A21 A22 A23] where Aij are 1x1 (scalars)
    # Let B = [B11 B12 B13; B21 B22 B23; B31 B32 B33] where Bij are 1x1 (scalars)
    
    # We can group the 3 columns of A into two groups: G1={1,2}, G2={3}
    # And the 3 rows of B into two groups: H1={1,2}, H2={3}
    
    # This creates a 2x2 block structure if we define:
    # A11' = A11+A12, A12' = A13
    # A21' = A21+A22, A22' = A23
    # B11' = B11, B12' = B21  (Wait, B is 3x5, so rows are 1,2,3. Cols are 1..5)
    # This simple block grouping doesn't directly reduce rank to 28 without more work.
    
    # Alternative: Use known rank 28 decomposition if exists, or try to find one.
    # The known minimum rank for 2x3x5 is likely lower than 30.
    # Let's try to find a rank 28 solution using random restarts with better initialization.
    
    # We will start by trying to find a rank 28 solution.
    # If that fails, we will try rank 29, then 30 (schoolbook).
    
    best_rank = n*m*p
    best_U, best_V, best_W = None, None, None

    # Try ranks from 22 to 30
    for R in range(22, 31):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(5000):
            if time.time() > deadline:
                break
            
            # Use deterministic seed based on trial to ensure reproducibility
            np.random.seed(42 + trial * R)
            
            # Initialize with random values
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # Iterate ALS (Alternating Least Squares)
            for it in range(30):
                if time.time() > deadline:
                    break
                
                # Solve for U
                U_new = np.zeros((n*m, R))
                V_reshaped = V.reshape(m, p, R)
                W_reshaped = W.reshape(p, n, R)
                for j in range(m):
                    for i in range(n):
                        A = V_reshaped[j] * W_reshaped[:, i]
                        b = np.ones(p)
                        try:
                            U_new[i*m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                U = U_new
                
                # Solve for V
                V_new = np.zeros((m*p, R))
                U_reshaped = U.reshape(n, m, R)
                W_reshaped = W.reshape(p, n, R)
                for j in range(m):
                    for k in range(p):
                        A = U_reshaped[:, j] * W_reshaped[k]
                        b = np.ones(n)
                        try:
                            V_new[j*p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                V = V_new
                
                # Solve for W
                W_new = np.zeros((p*n, R))
                U_reshaped = U.reshape(n, m, R)
                V_reshaped = V.reshape(m, p, R)
                for k in range(p):
                    for i in range(n):
                        A = U_reshaped[i] * V_reshaped[:, k]
                        b = np.ones(m)
                        try:
                            W_new[k*n + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                W = W_new
            
            # Round and Polish
            U_rounded = round_to_half_int(U)
            V_rounded = round_to_half_int(V)
            W_rounded = round_to_half_int(W)
            
            # Only polish if the error is reasonably small
            err = compute_error(U_rounded, V_rounded, W_rounded, R)
            if err < 5.0:
                U_polished, V_polished, W_polished, is_exact = polish(U_rounded, V_rounded, W_rounded, R)
                if is_exact:
                    best_rank = R
                    best_U, best_V, best_W = U_polished, V_polished, W_polished
                    break
            else:
                # Check if it was already exact before polishing
                if is_exact_solution(U_rounded, V_rounded, W_rounded, R):
                    best_rank = R
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    break
        
        if best_rank == R:
            break
    
    if best_U is None:
        # Fall back to schoolbook
        R = n*m*p
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
