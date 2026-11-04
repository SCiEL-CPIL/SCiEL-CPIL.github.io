"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # Dimensions of the factor matrices
    dim_U = n * m  # 9
    dim_V = m * p  # 15
    dim_W = p * n  # 15
    
    # Build the target tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # Otherwise 0
    T = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook rank
    schoolbook_rank = n * m * p
    U_sch = np.zeros((dim_U, schoolbook_rank))
    V_sch = np.zeros((dim_V, schoolbook_rank))
    W_sch = np.zeros((dim_W, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sch[a, col] = 1.0
                V_sch[b, col] = 1.0
                W_sch[c, col] = 1.0
                col += 1
    
    best_result = (U_sch.tolist(), V_sch.tolist(), W_sch.tolist())
    best_rank = schoolbook_rank
    
    deadline = time.time() + 110  # Safety margin
    
    # Try to find a lower rank decomposition using ALS
    # We iterate from rank 44 downwards
    # We limit the number of restarts and iterations to stay within time
    
    for target_rank in range(schoolbook_rank - 1, 10, -1):
        if time.time() > deadline - 10:
            break
            
        # Try a few random initializations for this rank
        # More restarts for lower ranks? No, lower ranks are harder.
        # Let's try 20 restarts for rank 44, 10 for 43, etc.
        num_restarts = 20 if target_rank == schoolbook_rank - 1 else 10
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize U, V, W
            # Small random values to avoid singularities
            U = np.random.randn(dim_U, target_rank) * 0.5
            V = np.random.randn(dim_V, target_rank) * 0.5
            W = np.random.randn(dim_W, target_rank) * 0.5
            
            # Run ALS
            # We use a fixed number of iterations, e.g., 50
            # Or until convergence
            max_iter = 50
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                    
                # Update U
                # T^{(1)} is dim_U x (dim_V * dim_W)
                # Z is (dim_V * dim_W) x target_rank
                # Z[:, r] = v_r \otimes w_r
                Z = np.zeros((dim_V * dim_W, target_rank))
                for r in range(target_rank):
                    Z[:, r] = np.kron(V[:, r], W[:, r])
                
                # Solve U * Z^T = T^{(1)}
                # U = T^{(1)} * Z * (Z^T * Z)^{-1}
                # Use lstsq for stability
                T_unfold1 = T.reshape(dim_U, dim_V * dim_W)
                try:
                    U, _, _, _ = np.linalg.lstsq(Z.T, T_unfold1.T, rcond=None)
                    U = U.T
                except np.linalg.LinAlgError:
                    pass

                # Update V
                # T^{(2)} is dim_V x (dim_W * dim_U)
                # Z is (dim_W * dim_U) x target_rank
                # Z[:, r] = w_r \otimes u_r
                Z = np.zeros((dim_W * dim_U, target_rank))
                for r in range(target_rank):
                    Z[:, r] = np.kron(W[:, r], U[:, r])
                
                T_unfold2 = T.transpose(1, 2, 0).reshape(dim_V, dim_W * dim_U)
                try:
                    V, _, _, _ = np.linalg.lstsq(Z.T, T_unfold2.T, rcond=None)
                    V = V.T
                except np.linalg.LinAlgError:
                    pass

                # Update W
                # T^{(3)} is dim_W x (dim_U * dim_V)
                # Z is (dim_U * dim_V) x target_rank
                # Z[:, r] = u_r \otimes v_r
                Z = np.zeros((dim_U * dim_V, target_rank))
                for r in range(target_rank):
                    Z[:, r] = np.kron(U[:, r], V[:, r])
                
                T_unfold3 = T.transpose(2, 0, 1).reshape(dim_W, dim_U * dim_V)
                try:
                    W, _, _, _ = np.linalg.lstsq(Z.T, T_unfold3.T, rcond=None)
                    W = W.T
                except np.linalg.LinAlgError:
                    pass

            # Check error
            # Reconstruct tensor
            approx = np.zeros((dim_U, dim_V, dim_W))
            for r in range(target_rank):
                approx += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            error = np.linalg.norm(T - approx)
            
            # If error is small, try to round to half-integers
            if error < 1e-6:
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify exactness
                approx_rounded = np.zeros((dim_U, dim_V, dim_W))
                for r in range(target_rank):
                    approx_rounded += np.einsum('a,b,c->abc', U_rounded[:, r], V_rounded[:, r], W_rounded[:, r])
                
                if np.allclose(T, approx_rounded, atol=1e-9):
                    # Found a valid decomposition
                    best_result = (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
                    best_rank = target_rank
                    # Break out of restart loop, continue to next lower rank
                    break
        
        # If we found a valid decomposition for this rank, we can continue to try lower
        # But we should probably stop if we find rank 44, as going lower is very unlikely
        # and expensive. However, the goal is "rank-minimal".
        # We will continue the loop to try lower ranks, but the time limit will likely stop us.
        # If we want to be safe, we can break if we find rank 44.
        if best_rank < schoolbook_rank:
            # We found an improvement. 
            # Should we try to find an even better one?
            # Given the time constraints and difficulty, finding rank 44 is a significant win.
            # Let's try a few more lower ranks if time permits.
            pass

    # Return the best result found
    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}

# EVOLVE-BLOCK-END
