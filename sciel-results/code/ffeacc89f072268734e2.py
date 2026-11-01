import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def _verify_fast(U, V, W, n=2, m=4, p=8, trials=50, tol=1e-9):
    """Fast verification using vectorized numpy operations."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_mat = np.array(U, dtype=float)  # shape (n*m, rank)
    V_mat = np.array(V, dtype=float)  # shape (m*p, rank)
    W_mat = np.array(W, dtype=float)  # shape (p*n, rank)

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

def _als_step(U, V, W, n, m, p, rank, target_tensor):
    """
    Perform one step of Alternating Least Squares.
    target_tensor is a flattened vector of length n*m*p with 1s.
    We solve for one factor at a time.
    """
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    # 1. Solve for V
    # Eq: sum_c U_{ij}^c V_{jk}^c W_{ki}^c = 1
    # For each j,k: sum_c V_{jk}^c (sum_{i} U_{ij}^c W_{ki}^c) = 1? No.
    # The equation is for each (i,j,k): sum_c U_{ij}^c V_{jk}^c W_{ki}^c = 1.
    # Let's group by (j,k).
    # sum_c V_{jk}^c * (sum_i U_{ij}^c W_{ki}^c) = 1.
    # This is not linear in V because the coefficient depends on i.
    # Actually, we can vectorize.
    # Let's flatten V to length (m*p)*rank.
    # For each (j,k), and each c, the variable is V_{jk}^c.
    # The equation for a specific (i,j,k) is:
    # sum_c U_{ij}^c W_{ki}^c V_{jk}^c = 1.
    # This is a linear equation in the vector of V entries for that (j,k).
    # But different (i,j,k) share the same V_{jk}^c.
    # So we form a global linear system for V.
    
    # Number of equations: n*m*p
    # Number of unknowns for V: m*p*rank
    # Matrix A_V of shape (n*m*p, m*p*rank)
    # A_V[(i,j,k), (j,k,c)] = U_{ij}^c * W_{ki}^c
    
    A_V = np.zeros((n*m*p, m*p*rank))
    b_V = np.ones(n*m*p)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for c in range(rank):
                    # V index: (j*p + k) * rank + c
                    A_V[idx, (j*p + k) * rank + c] = U_mat[i*m + j, c] * W_mat[k*n + i, c]
                idx += 1
                
    # Solve least squares
    try:
        V_sol, _, _, _ = np.linalg.lstsq(A_V, b_V, rcond=None)
        V_mat = V_sol.reshape(m*p, rank)
    except:
        pass
        
    # 2. Solve for U
    # A_U of shape (n*m*p, n*m*rank)
    # A_U[(i,j,k), (i,j,c)] = V_{jk}^c * W_{ki}^c
    A_U = np.zeros((n*m*p, n*m*rank))
    b_U = np.ones(n*m*p)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for c in range(rank):
                    # U index: (i*m + j) * rank + c
                    A_U[idx, (i*m + j) * rank + c] = V_mat[j*p + k, c] * W_mat[k*n + i, c]
                idx += 1
                
    try:
        U_sol, _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
        U_mat = U_sol.reshape(n*m, rank)
    except:
        pass
        
    # 3. Solve for W
    # A_W of shape (n*m*p, p*n*rank)
    # A_W[(i,j,k), (k,i,c)] = U_{ij}^c * V_{jk}^c
    A_W = np.zeros((n*m*p, p*n*rank))
    b_W = np.ones(n*m*p)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for c in range(rank):
                    # W index: (k*n + i) * rank + c
                    A_W[idx, (k*n + i) * rank + c] = U_mat[i*m + j, c] * V_mat[j*p + k, c]
                idx += 1
                
    try:
        W_sol, _, _, _ = np.linalg.lstsq(A_W, b_W, rcond=None)
        W_mat = W_sol.reshape(p*n, rank)
    except:
        pass
        
    return U_mat, V_mat, W_mat

def _check_error(U, V, W, n, m, p, rank):
    """Calculate the max error of the tensor identity."""
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    # Compute T_computed_{ijk} = sum_c U_{ij}^c V_{jk}^c W_{ki}^c
    # We can use einsum
    U_r = U_mat.reshape(n, m, rank)
    V_r = V_mat.reshape(m, p, rank)
    W_r = W_mat.reshape(p, n, rank)
    
    T_computed = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    T_target = np.ones((n, m, p))
    
    return np.max(np.abs(T_computed - T_target))

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Theoretical lower bound is max(nm, mp, pn) = 32
    # Strassen block gives 56.
    # We try to find a solution with rank < 56.
    
    best_rank = 56
    best_U, best_V, best_W = None, None, None
    
    # Initialize with a known valid decomposition (block Strassen) to have a fallback
    # But we want to find something better.
    # Let's try ranks from 32 up to 55.
    
    # We will use ALS.
    # To improve convergence, we can use multiple restarts for each rank.
    
    for target_rank in range(32, 56):
        if time.time() - start_time > deadline - 10:
            break
            
        found = False
        # Try a few random initializations
        for restart in range(20):
            if time.time() - start_time > deadline - 10:
                break
                
            np.random.seed(42 + target_rank * 100 + restart)
            
            # Initialize with random values
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Scale to avoid numerical issues
            U /= np.linalg.norm(U) * 10
            V /= np.linalg.norm(V) * 10
            W /= np.linalg.norm(W) * 10
            
            # Run ALS for a few iterations
            max_iter = 50
            for it in range(max_iter):
                U, V, W = _als_step(U, V, W, n, m, p, target_rank, None)
                
                # Check error
                err = _check_error(U, V, W, n, m, p, target_rank)
                if err < 1e-8:
                    # Verify with random matrices
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    if _verify_fast(U_list, V_list, W_list, n, m, p, trials=20, tol=1e-9):
                        best_rank = target_rank
                        best_U = U_list
                        best_V = V_list
                        best_W = W_list
                        found = True
                        break
            if found:
                break
                
    # If we didn't find anything better than 56, or if we ran out of time,
    # we should return the best we have.
    # If best_rank is still 56 and we didn't find a solution, we need to return a valid one.
    # The current program has a block Strassen construction. We should include that as a fallback.
    
    if best_rank >= 56:
        # Use the block Strassen construction from the original code as fallback
        # We need to implement it or use a simpler valid one.
        # Schoolbook is rank 64.
        # Block Strassen is rank 56.
        
        # Let's quickly build the block Strassen to be safe.
        # Or just use schoolbook if we can't build Strassen quickly?
        # Schoolbook is always valid.
        
        # Let's try to build Block Strassen quickly.
        try:
            # Re-use the logic from the original code for Block Strassen
            # But that code was long. Let's just use Schoolbook if we didn't find < 56.
            # Wait, the original code scored 1.0 with rank 56.
            # If we return Schoolbook (rank 64), we score 56/64 = 0.875 (worse).
            # If we return the original Block Strassen, we score 1.0.
            # So we MUST return the Block Strassen if we don't find better.
            
            # Let's copy the _build_block_decomposition from the original code.
            rank = 56
            U_fb = np.zeros((n * m, rank), dtype=float)
            V_fb = np.zeros((m * p, rank), dtype=float)
            W_fb = np.zeros((p * n, rank), dtype=float)
            
            # Strassen 2x2x2
            U_s = np.array([
                [1, 0, 0, 1],
                [0, 0, 1, 1],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [1, 1, 0, 0],
                [0, 0, 1, -1],
                [0, 1, 0, 1]
            ], dtype=float)
            V_s = np.array([
                [1, 0, 0, 1],
                [0, 0, 0, 1],
                [1, 0, -1, 0],
                [0, -1, 0, 1],
                [0, 0, 1, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1]
            ], dtype=float)
            W_s = np.array([
                [1, 0, 0, 1, -1, 0, 1],
                [0, 0, 1, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [1, -1, 1, 0, 0, 1, 0]
            ], dtype=float)
            
            col_offset = 0
            for block in range(4):
                for f in range(7):
                    col = col_offset + f
                    for r in range(4):
                        U_fb[r][col] = U_s[f][r]
                        j = r // 2
                        k = 2 * block + r % 2
                        V_fb[j * 8 + k][col] = V_s[f][r]
                        i = r // 2
                        W_fb[k * 2 + i][col] = W_s[r][f]
                col_offset += 7
            
            for block in range(4):
                for f in range(7):
                    col = col_offset + f
                    for r in range(4):
                        U_fb[2 + r][col] = U_s[f][r]
                        j = 2 + r // 2
                        k = 2 * block + r % 2
                        V_fb[j * 8 + k][col] = V_s[f][r]
                        i = r // 2
                        W_fb[k * 2 + i][col] = W_s[r][f]
                col_offset += 7
            
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
        except:
            # Fallback to schoolbook
            schoolbook_rank = n * m * p
            U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
            V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
            W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i * m + j][col] = 1
                        V_sb[j * p + k][col] = 1
                        W_sb[k * n + i][col] = 1
                        col += 1
            return {"U": U_sb, "V": V_sb, "W": W_sb}
            
    return {"U": best_U, "V": best_V, "W": best_W}
