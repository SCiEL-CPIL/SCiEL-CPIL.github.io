import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define the target tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i, else 0
    # Dimensions: n*m, m*p, p*n
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    T = np.zeros((dim_U, dim_V, dim_W), dtype=float)
    
    # Populate T
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # Greedy Tensor Rank-1 Decomposition
    # We will try to decompose T into a sum of rank-1 tensors.
    
    def tensor_power_iteration(X, u_init, v_init, w_init, max_iter=100, tol=1e-12):
        """
        Find the best rank-1 approximation of tensor X using power iteration.
        Returns u, v, w such that u o v o w approx X.
        """
        u = u_init.copy()
        v = v_init.copy()
        w = w_init.copy()
        
        # Normalize initial vectors
        u = u / np.linalg.norm(u)
        v = v / np.linalg.norm(v)
        w = w / np.linalg.norm(w)
        
        for _ in range(max_iter):
            u_old = u.copy()
            
            # Update u: contract X with v and w
            # (X x_2 v) x_3 w
            tmp = np.tensordot(X, v, axes=([1], [0])) # Shape (dim_U, dim_W)
            u_new = np.tensordot(tmp, w, axes=([1], [0])) # Shape (dim_U)
            
            # Update v
            tmp = np.tensordot(X, u, axes=([0], [0])) # Shape (dim_V, dim_W)
            v_new = np.tensordot(tmp, w, axes=([1], [0])) # Shape (dim_V)
            
            # Update w
            tmp = np.tensordot(X, u, axes=([0], [0])) # Shape (dim_V, dim_W)
            tmp = np.tensordot(tmp, v, axes=([0], [0])) # Shape (dim_W)
            w_new = tmp
            
            # Normalize
            u_new = u_new / np.linalg.norm(u_new)
            v_new = v_new / np.linalg.norm(v_new)
            w_new = w_new / np.linalg.norm(w_new)
            
            # Check convergence
            if np.linalg.norm(u_new - u_old) < tol:
                u, v, w = u_new, v_new, w_new
                break
                
            u, v, w = u_new, v_new, w_new
            
        return u, v, w

    def greedy_decompose(T, max_rank=80, num_restarts=10):
        """
        Greedy decomposition with random restarts for the power iteration initialization.
        Returns U, V, W matrices and the rank.
        """
        residual = T.copy()
        U_cols = []
        V_cols = []
        W_cols = []
        
        current_rank = 0
        start_time = time.time()
        deadline = start_time + 140
        
        while np.linalg.norm(residual) > 1e-9:
            if time.time() > deadline:
                break
            if current_rank >= max_rank:
                break
                
            best_u, best_v, best_w, best_residual_norm = None, None, None, np.inf
            
            # Try multiple initializations
            for r in range(num_restarts):
                # Random initialization
                u_init = np.random.randn(dim_U)
                v_init = np.random.randn(dim_V)
                w_init = np.random.randn(dim_W)
                
                # Run power iteration
                u, v, w = tensor_power_iteration(residual, u_init, v_init, w_init, max_iter=50)
                
                # Calculate the scalar coefficient alpha
                # alpha = <residual, u o v o w> / ||u o v o w||^2
                # Since u, v, w are normalized, ||u o v o w||^2 = 1
                # <residual, u o v o w> = sum(residual * u * v * w)
                dot_product = np.einsum('ijk,i,j,k->', residual, u, v, w)
                
                # New residual
                new_residual = residual - dot_product * (np.outer(u, np.outer(v, w)).reshape(dim_U, dim_V, dim_W))
                norm_new_residual = np.linalg.norm(new_residual)
                
                if norm_new_residual < best_residual_norm:
                    best_residual_norm = norm_new_residual
                    best_u, best_v, best_w = u, v, w
                    best_dot = dot_product
                    best_new_residual = new_residual
                    
            # If no improvement or improvement is negligible, break
            if best_u is None or best_residual_norm >= np.linalg.norm(residual) - 1e-12:
                break
                
            # Add to decomposition
            U_cols.append(best_u * best_dot)
            V_cols.append(best_v)
            W_cols.append(best_w)
            
            residual = best_new_residual
            current_rank += 1
            
            # Early termination if residual is effectively zero
            if np.linalg.norm(residual) < 1e-9:
                break

        if U_cols:
            U = np.column_stack(U_cols)
            V = np.column_stack(V_cols)
            W = np.column_stack(W_cols)
            return U, V, W, current_rank
        else:
            # Fallback to schoolbook
            return None, None, None, 0

    # Run greedy decomposition
    U, V, W, rank = greedy_decompose(T, max_rank=80, num_restarts=5)
    
    if U is None:
        # Fallback to schoolbook if greedy failed completely (should not happen)
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    else:
        # Verify the decomposition
        # Check if sum U[:,r] V[:,r] W[:,r] == T
        # Reconstruct
        # For efficiency, we can check a few random points or the whole thing if rank is small
        # Since rank is likely < 80, checking the whole thing is feasible (80 * 20 * 20 * 20 ops ~ 640k)
        
        # Let's verify fully
        # T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        # This might be slow for large rank, but rank <= 80.
        # 80 * 16 * 20 * 20 = 512,000 multiplications. Very fast.
        
        T_recon = np.zeros_like(T)
        for r in range(rank):
            T_recon += np.outer(U[:, r], np.outer(V[:, r], W[:, r])).reshape(dim_U, dim_V, dim_W)
            
        error = np.linalg.norm(T_recon - T)
        
        if error > 1e-6:
            # Decomposition failed, fallback to schoolbook
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
        else:
            # Try to round to integers or half-integers if they are close
            # The problem allows integer/half-integer entries.
            # If we have floats like 0.5000001, we should convert to 0.5.
            
            def round_to_half_int(arr):
                # Multiply by 2, round, divide by 2
                arr_rounded = np.round(arr * 2) / 2
                # Check if rounding improved the error or if it was already close
                # If the original values were close to half-integers, this is safe.
                # We will verify after rounding.
                return arr_rounded
            
            U_rounded = round_to_half_int(U)
            V_rounded = round_to_half_int(V)
            W_rounded = round_to_half_int(W)
            
            # Verify rounded version
            T_recon_rounded = np.zeros_like(T)
            for r in range(rank):
                T_recon_rounded += np.outer(U_rounded[:, r], np.outer(V_rounded[:, r], W_rounded[:, r])).reshape(dim_U, dim_V, dim_W)
                
            error_rounded = np.linalg.norm(T_recon_rounded - T)
            
            if error_rounded < 1e-6:
                U = U_rounded
                V = V_rounded
                W = W_rounded

    result = {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
    return result
# EVOLVE-BLOCK-END
