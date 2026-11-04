"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Strategy: Use Strassen-based block decomposition (rank 33) as baseline,
    then attempt numerical optimization to reduce rank.
    """
    start_time = time.time()
    deadline = start_time + 135  # Leave 5s margin for safety within 140s limit

    n, m, p = DIMENSIONS
    dim_a = n * m  # 6
    dim_b = m * p  # 18
    dim_c = p * n  # 12

    # Build the target tensor T[a, b, c]
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify_decomposition(U, V, W, tol=1e-8):
        """Verify that U, V, W exactly decompose the tensor T."""
        rank = U.shape[1]
        result = np.zeros((dim_a, dim_b, dim_c), dtype=np.float64)
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            result += np.einsum('i,j,k->ijk', uc, vc, wc)
        return np.allclose(result, T, atol=tol, rtol=0)

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = np.zeros((dim_a, rank), dtype=np.float64)
        V = np.zeros((dim_b, rank), dtype=np.float64)
        W = np.zeros((dim_c, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return U, V, W

    def strassen_block_decomposition():
        """
        Return rank-33 decomposition: 3 blocks of (Strassen 2x2x2 + Standard 2x1x2).
        Each block handles a 2x2 sub-block of the 3x6 B matrix plus the remaining column.
        """
        rank = 33
        U = np.zeros((dim_a, rank), dtype=np.float64)
        V = np.zeros((dim_b, rank), dtype=np.float64)
        W = np.zeros((dim_c, rank), dtype=np.float64)
        
        # Strassen 2x2x2 factors
        S_U = np.array([
            [[1, 0], [0, 1]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 1], [0, -1]]
        ], dtype=np.float64)
        S_V = np.array([
            [[1, 0], [0, 1]],
            [[1, 0], [0, 0]],
            [[0, 1], [0, -1]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, 0], [1, 1]]
        ], dtype=np.float64)
        S_W = np.array([
            [[1, 0], [0, 1]],
            [[0, 1], [0, -1]],
            [[0, 0], [1, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 0], [0, 0]]
        ], dtype=np.float64)
        
        # Standard 2x1x2 factors
        Std_U = np.array([
            [[1], [0]],
            [[1], [0]],
            [[0], [1]],
            [[0], [1]]
        ], dtype=np.float64)
        Std_V = np.array([
            [[1, 0]],
            [[0, 1]],
            [[1, 0]],
            [[0, 1]]
        ], dtype=np.float64)
        Std_W = np.array([
            [[1, 0], [0, 0]],
            [[0, 0], [1, 0]],
            [[0, 1], [0, 0]],
            [[0, 0], [0, 1]]
        ], dtype=np.float64)
        
        col = 0
        for b in range(3):
            # Strassen part: handles columns 2b, 2b+1 of B (and corresponding C)
            for r in range(7):
                # Map U (2x2) to global U (rows for i=0,1 and j=0,1)
                for i in range(2):
                    for j in range(2):
                        U[i * m + j, col] = S_U[r, i, j]
                
                # Map V (2x2) to global V (rows for j=0,1 and k=2b, 2b+1)
                for j in range(2):
                    for kp in range(2):
                        k_global = 2 * b + kp
                        V[j * p + k_global, col] = S_V[r, j, kp]
                
                # Map W (2x2) to global W (rows for k=2b, 2b+1 and i=0,1)
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = S_W[r, kp, i]
                
                col += 1
            
            # Standard part: handles column j=2 of A/B, columns 2b, 2b+1 of B
            for r in range(4):
                # Map U (2x1) to global U (rows for i=0,1 and j=2)
                for i in range(2):
                    U[i * m + 2, col] = Std_U[r, i, 0]
                
                # Map V (1x2) to global V (row for j=2 and k=2b, 2b+1)
                for kp in range(2):
                    k_global = 2 * b + kp
                    V[2 * p + k_global, col] = Std_V[r, 0, kp]
                
                # Map W (2x2) to global W (rows for k=2b, 2b+1 and i=0,1)
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = Std_W[r, kp, i]
                
                col += 1
        
        return U, V, W

    # Start with the known rank-33 decomposition
    best_U, best_V, best_W = strassen_block_decomposition()
    best_rank = 33
    
    # Verify the baseline
    if not verify_decomposition(best_U, best_V, best_W):
        # Fallback to schoolbook if Strassen construction is wrong
        best_U, best_V, best_W = schoolbook()
        best_rank = n * m * p

    # Attempt to reduce rank via numerical optimization
    def try_reduce_rank(target_rank, init_U, init_V, init_W, max_time=30):
        """
        Try to find a decomposition of target_rank using least_squares optimization.
        Returns (U, V, W) if successful, None otherwise.
        """
        if time.time() > deadline - 5:
            return None
        
        from scipy.optimize import least_squares
        
        # Build target tensor flattened
        T_flat = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_flat[idx] = 1.0
                    idx += 1
        
        # Precompute index mapping for speed
        indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    indices.append((a, b, c))
        
        def objective(params):
            U = params[0:dim_a*target_rank].reshape(dim_a, target_rank)
            V = params[dim_a*target_rank:dim_a*target_rank + dim_b*target_rank].reshape(dim_b, target_rank)
            W = params[dim_a*target_rank + dim_b*target_rank:].reshape(dim_c, target_rank)
            
            result = np.zeros(n*m*p)
            for idx, (a, b, c) in enumerate(indices):
                result[idx] = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
            return result - T_flat
        
        best_params = None
        best_err = float('inf')
        
        n_restarts = 10
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialization strategies
            if restart == 0:
                # Truncation of baseline
                if target_rank <= init_U.shape[1]:
                    U_init = init_U[:, :target_rank].copy()
                    V_init = init_V[:, :target_rank].copy()
                    W_init = init_W[:, :target_rank].copy()
                else:
                    U_init = init_U.copy()
                    V_init = init_V.copy()
                    W_init = init_W.copy()
            elif restart == 1:
                # Truncation + Perturbation
                if target_rank <= init_U.shape[1]:
                    U_init = init_U[:, :target_rank].copy()
                    V_init = init_V[:, :target_rank].copy()
                    W_init = init_W[:, :target_rank].copy()
                else:
                    U_init = init_U.copy()
                    V_init = init_V.copy()
                    W_init = init_W.copy()
                np.random.seed(123 + restart * 7)
                U_init += np.random.randn(*U_init.shape) * 0.1
                V_init += np.random.randn(*V_init.shape) * 0.1
                W_init += np.random.randn(*W_init.shape) * 0.1
            else:
                # Random Gaussian initialization
                np.random.seed(42 + restart * 13)
                # Scale based on baseline norms to be in a reasonable range
                scale_U = np.linalg.norm(init_U) / np.sqrt(dim_a * target_rank) if target_rank > 0 else 1.0
                scale_V = np.linalg.norm(init_V) / np.sqrt(dim_b * target_rank) if target_rank > 0 else 1.0
                scale_W = np.linalg.norm(init_W) / np.sqrt(dim_c * target_rank) if target_rank > 0 else 1.0
                
                U_init = np.random.randn(dim_a, target_rank) * scale_U
                V_init = np.random.randn(dim_b, target_rank) * scale_V
                W_init = np.random.randn(dim_c, target_rank) * scale_W
            
            params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                # Use 'trf' for robustness, 'lm' can be faster but gets stuck
                # max_nfev is the max number of function evaluations
                res = least_squares(objective, params0, method='trf', max_nfev=1000)
                err = np.linalg.norm(res.fun)
                if err < best_err:
                    best_err = err
                    best_params = res.x.copy()
            except Exception:
                continue
        
        if best_params is not None and best_err < 1e-6:
            U_opt = best_params[0:dim_a*target_rank].reshape(dim_a, target_rank)
            V_opt = best_params[dim_a*target_rank:dim_a*target_rank + dim_b*target_rank].reshape(dim_b, target_rank)
            W_opt = best_params[dim_a*target_rank + dim_b*target_rank:].reshape(dim_c, target_rank)
            
            # Attempt to round to integers or half-integers
            # The problem statement suggests integer/half-integer entries.
            U_rounded = np.round(U_opt * 2) / 2
            V_rounded = np.round(V_opt * 2) / 2
            W_rounded = np.round(W_opt * 2) / 2
            
            if verify_decomposition(U_rounded, V_rounded, W_rounded, tol=1e-8):
                return U_rounded, V_rounded, W_rounded
            
            # If rounding fails, check if the float solution is very precise
            if verify_decomposition(U_opt, V_opt, W_opt, tol=1e-8):
                return U_opt, V_opt, W_opt
        
        return None
    
    # Try to reduce from rank 33 to 32
    if time.time() < deadline - 15:
        result = try_reduce_rank(32, best_U, best_V, best_W, max_time=25)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 32
    
    # If still have time, try rank 31
    if best_rank == 33 and time.time() < deadline - 10:
        result = try_reduce_rank(31, best_U, best_V, best_W, max_time=20)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 31
    
    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
