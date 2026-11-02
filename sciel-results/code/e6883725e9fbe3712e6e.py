"""Matrix multiplication 3x4x5 tensor decomposition - rank minimization."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 5 matrix.
    The output factor follows the official trace convention W[k*n+i,r].
    
    Strategy:
    1. Start with the schoolbook decomposition (rank 60).
    2. Try to reduce rank by removing columns and solving the resulting underdetermined system.
    3. Use iterative projection to find valid lower-rank decompositions.
    4. Return the best (lowest rank) valid decomposition found.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    rank_schoolbook = n * m * p  # 60
    
    # Build schoolbook decomposition
    U = np.zeros((n * m, rank_schoolbook), dtype=float)
    V = np.zeros((m * p, rank_schoolbook), dtype=float)
    W = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j] * V[j * p + k], W[k * n + i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def round_to_half(x):
        return np.round(x * 2) / 2
    
    def try_rank_reduction(target_rank, U_init, V_init, W_init, max_iters=200, tol=1e-8):
        """
        Try to adjust U, V, W to satisfy the tensor identity at target_rank.
        Uses iterative block-coordinate descent with least-squares updates.
        """
        U = U_init.astype(float).copy()
        V = V_init.astype(float).copy()
        W = W_init.astype(float).copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute current errors
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j] * V[j * p + k], W[k * n + i])
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < tol:
                return U, V, W, True
            
            # Update W: For each (k, i), solve min ||A_{ki} - 1|| where A_{ki} = sum_j U_ij * V_jk
            # W_{ki} appears linearly: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
            # For fixed U, V: define B_{ki, r} = sum_j U[i*m+j, r] V[j*p+k, r]
            # Then sum_r B_{ki, r} W[k*n+i, r] = 1
            # This is one equation per (k,i) with R unknowns - underdetermined.
            # Use minimum norm solution: W_{ki} = B_{ki} / (B_{ki}^T B_{ki})
            for k in range(p):
                for i in range(n):
                    B = np.zeros(target_rank, dtype=float)
                    for j in range(m):
                        B += U[i * m + j] * V[j * p + k]
                    norm_sq = np.dot(B, B)
                    if norm_sq > 1e-12:
                        W[k * n + i] = B / norm_sq
            
            # Update U: For each (i, j), solve system
            # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all k
            # Let C_{k, r} = V[j*p+k, r] W[k*n+i, r]
            # Then C U_{ij}^T = 1 (vector of ones, length p)
            for i in range(n):
                for j in range(m):
                    C = np.zeros((p, target_rank), dtype=float)
                    for k in range(p):
                        C[k] = V[j * p + k] * W[k * n + i]
                    try:
                        CCT = C.dot(C.T)
                        CCT += 1e-10 * np.eye(p)
                        U[i * m + j] = np.linalg.solve(CCT, np.ones(p)).dot(C)
                    except np.linalg.LinAlgError:
                        pass
            
            # Update V: For each (j, k), solve system
            # sum_r V[j*p+k, r] U[i*m+j, r] W[k*n+i, r] = 1 for all i
            # Let D_{i, r} = U[i*m+j, r] W[k*n+i, r]
            # Then D V_{jk}^T = 1 (vector of ones, length n)
            for j in range(m):
                for k in range(p):
                    D = np.zeros((n, target_rank), dtype=float)
                    for i in range(n):
                        D[i] = U[i * m + j] * W[k * n + i]
                    try:
                        DDT = D.dot(D.T)
                        DDT += 1e-10 * np.eye(n)
                        V[j * p + k] = np.linalg.solve(DDT, np.ones(n)).dot(D)
                    except np.linalg.LinAlgError:
                        pass
        
        # Final check
        if check_decomposition(U, V, W, tol=1e-8):
            return U, V, W, True
        return U, V, W, False
    
    def try_rank_reduction_with_random(target_rank, U_init, V_init, W_init, n_trials=5, max_iters=100):
        """
        Try multiple random perturbations to escape local minima.
        """
        best_U, best_V, best_W = U_init.copy(), V_init.copy(), W_init.copy()
        best_err = float('inf')
        
        for trial in range(n_trials):
            if time.time() > deadline - 5:
                break
            
            if trial == 0:
                U_t = U_init.copy()
                V_t = V_init.copy()
                W_t = W_init.copy()
            else:
                # Add small random perturbation
                scale = 0.1 / (trial + 1)
                U_t = U_init + scale * np.random.randn(*U_init.shape)
                V_t = V_init + scale * np.random.randn(*V_init.shape)
                W_t = W_init + scale * np.random.randn(*W_init.shape)
            
            U_sol, V_sol, W_sol, success = try_rank_reduction(target_rank, U_t, V_t, W_t, max_iters)
            
            if success:
                return U_sol, V_sol, W_sol, True
            
            # Track best error
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_sol[i * m + j] * V_sol[j * p + k], W_sol[k * n + i])
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < best_err:
                best_err = max_err
                best_U, best_V, best_W = U_sol.copy(), V_sol.copy(), W_sol.copy()
        
        return best_U, best_V, best_W, False
    
    # Try to find a lower rank decomposition
    # Start from rank 59 down to 50
    for target_rank in range(59, 49, -1):
        if time.time() > deadline - 10:
            break
        
        # Initialize with first target_rank columns of schoolbook
        U_init = U[:, :target_rank].copy()
        V_init = V[:, :target_rank].copy()
        W_init = W[:, :target_rank].copy()
        
        # Try with random restarts
        U_sol, V_sol, W_sol, success = try_rank_reduction_with_random(
            target_rank, U_init, V_init, W_init, n_trials=8, max_iters=150
        )
        
        if success:
            # Try rounding to half-integers
            U_rounded = round_to_half(U_sol)
            V_rounded = round_to_half(V_sol)
            W_rounded = round_to_half(W_sol)
            
            if check_decomposition(U_rounded, V_rounded, W_rounded):
                best_rank = target_rank
                best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                break
            else:
                # Use the unrounded solution
                best_rank = target_rank
                best_U, best_V, best_W = U_sol, V_sol, W_sol
                break
    
    # Convert to lists for the required interface
    U_list = [[float(best_U[i, j]) for j in range(best_rank)] for i in range(n * m)]
    V_list = [[float(best_V[i, j]) for j in range(best_rank)] for i in range(m * p)]
    W_list = [[float(best_W[i, j]) for j in range(best_rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
