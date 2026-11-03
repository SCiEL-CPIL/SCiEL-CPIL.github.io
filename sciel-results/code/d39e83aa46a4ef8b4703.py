import time
import sys
import random
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n, m, p):
    """Check if the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # and check T[i,j,k] == delta(i,j,k) i.e., 1 if we're multiplying identity... 
    # Actually the tensor for matrix mult is: T_{i,j,k} = 1 if we want C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor decomposition is: sum_r U[i,j,r] V[j,k,r] W[r, i, k] = delta... 
    # Wait, let me re-read the convention.
    # W[k*n+i, r] means W is indexed by (k, i) -> W[k*n+i, r]
    # U[i*m+j, r], V[j*p+k, r]
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix multiplication, T[i,j,k] should be 1 (the tensor that computes C[i,k] = sum_j A[i,j] B[j,k])
    # Actually the multiplication tensor has entries T[i,j,k] = 1 for all i,j,k, and the 
    # decomposition gives C[i,k] = sum_{j,r} A[i,j] B[j,k] ... 
    # Let me think again. The standard tensor for matrix multiplication M_{n,m,p} has
    # entries T[i,j,k] = 1, and the bilinear form is sum_{i,j,k,r} T[i,j,k] U[i,j,r] V[j,k,r] W[i,k,r]
    # But here W is indexed as W[k*n+i, r], so W[i,k,r] = W[k*n+i, r].
    # So the check is: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    # Reshape for efficient computation
    # U_mat shape: (n*m, rank)
    # V_mat shape: (m*p, rank)
    # W_mat shape: (p*n, rank)
    
    # For each (i,j,k), compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We can do this by: for each r, the contribution is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Build a 3D tensor T of shape (n, m, p)
    T = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        u_col = U_mat[:, r]  # (n*m,)
        v_col = V_mat[:, r]  # (m*p,)
        w_col = W_mat[:, r]  # (p*n,)
        # u_col[i*m+j], v_col[j*p+k], w_col[k*n+i]
        u_3d = u_col.reshape(n, m)
        v_3d = v_col.reshape(m, p)
        w_3d = w_col.reshape(p, n)
        # T[i,j,k] += u_3d[i,j] * v_3d[j,k] * w_3d[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_3d[i, j] * v_3d[j, k] * w_3d[k, i]
    
    # Check if T[i,j,k] == 1 for all i,j,k
    return np.allclose(T, 1.0, atol=1e-9)

def _validate_fast(U, V, W, n, m, p):
    """Faster validation using matrix operations."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Let's compute this more efficiently.
    # For fixed i,j: T[i,j,:] = sum_r U[i*m+j, r] * V[j*p:(j+1)*p, r] * W_reshaped[:, i]
    # where W_reshaped[k, i] = W_mat[k*n+i]
    
    W_reshaped = W_mat.reshape(p, n)  # W_reshaped[k, i]
    
    for i in range(n):
        for j in range(m):
            u_val = U_mat[i*m+j, :]  # (rank,)
            v_slice = V_mat[j*p:(j+1)*p, :]  # (p, rank)
            # T[i,j,k] = sum_r u_val[r] * v_slice[k, r] * W_reshaped[k, i]
            # = sum_k' ... let's compute:
            # T[i,j,:] = (u_val * v_slice) @ W_reshaped[:, i]  ... no
            # T[i,j,k] = sum_r u_val[r] * v_slice[k,r] * W_reshaped[k,i]
            # = sum_k' delta(k,k') ... 
            # Actually: T[i,j,k] = sum_r u_val[r] * v_slice[k,r] * W_reshaped[k,i]
            # Let's compute element by element but vectorized over r:
            prod = u_val[None, :] * v_slice  # (p, rank)
            T_row = prod * W_reshaped[:, i][None, :]  # (p, rank)
            T_vals = T_row.sum(axis=1)  # (p,)
            if not np.allclose(T_vals, 1.0, atol=1e-9):
                return False
    return True

def construct():
    """
    Constructs a tensor decomposition for 3x5x5 matrix multiplication.
    Starts with schoolbook (rank 75) and attempts to find lower-rank decompositions
    using randomized search and local optimization.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Start with schoolbook decomposition
    rank0 = n * m * p
    U0 = [[0] * rank0 for _ in range(n * m)]
    V0 = [[0] * rank0 for _ in range(m * p)]
    W0 = [[0] * rank0 for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j][col] = 1
                V0[j * p + k][col] = 1
                W0[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank0
    
    # Try to find lower rank decompositions
    # Strategy: random initialization + iterative refinement
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # For small ranks, try random search with local refinement
    for target_rank in range(1, rank0):
        if time.time() > deadline - 20:
            break
        
        # Try a few random initializations
        max_attempts = 5
        if target_rank > 40:
            max_attempts = 3
        elif target_rank > 20:
            max_attempts = 2
        else:
            max_attempts = 3
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with small values
            U = [[random.uniform(-0.5, 0.5) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.uniform(-0.5, 0.5) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.uniform(-0.5, 0.5) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Local refinement: gradient-free approach
            # Use a simple iterative method to reduce the error
            error = _compute_error(U, V, W, n, m, p)
            
            # Try to improve via random perturbations (simulated annealing style)
            best_local_error = error
            best_local_U, best_local_V, best_local_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
            
            for step in range(200):
                if time.time() > deadline - 5:
                    break
                    
                # Random perturbation
                scale = 0.1 * (1.0 - step / 300.0)
                if scale < 0.01:
                    scale = 0.01
                    
                U_new = [row[:] for row in best_local_U]
                V_new = [row[:] for row in best_local_V]
                W_new = [row[:] for row in best_local_W]
                
                # Perturb a few random entries
                for _ in range(3):
                    mat = random.choice(['U', 'V', 'W'])
                    if mat == 'U':
                        arr = U_new
                    elif mat == 'V':
                        arr = V_new
                    else:
                        arr = W_new
                    ri = random.randrange(len(arr))
                    ci = random.randrange(len(arr[ri]))
                    arr[ri][ci] += random.uniform(-scale, scale)
                
                new_error = _compute_error(U_new, V_new, W_new, n, m, p)
                
                if new_error < best_local_error:
                    best_local_error = new_error
                    best_local_U, best_local_V, best_local_W = U_new, V_new, W_new
                
                if best_local_error < 1e-6:
                    # Check for exact solution
                    if _validate_fast(best_local_U, best_local_V, best_local_W, n, m, p):
                        # Round to nearest half-integer
                        best_local_U = [[round(x * 2) / 2 for x in row] for row in best_local_U]
                        best_local_V = [[round(x * 2) / 2 for x in row] for row in best_local_V]
                        best_local_W = [[round(x * 2) / 2 for x in row] for row in best_local_W]
                        if _validate_fast(best_local_U, best_local_V, best_local_W, n, m, p):
                            best_U, best_V, best_W = best_local_U, best_local_V, best_local_W
                            best_rank = target_rank
                            return {"U": best_U, "V": best_V, "W": best_W}
                    break
            
            if best_local_error < 1e-6:
                if _validate_fast(best_local_U, best_local_V, best_local_W, n, m, p):
                    best_local_U = [[round(x * 2) / 2 for x in row] for row in best_local_U]
                    best_local_V = [[round(x * 2) / 2 for x in row] for row in best_local_V]
                    best_local_W = [[round(x * 2) / 2 for x in row] for row in best_local_W]
                    if _validate_fast(best_local_U, best_local_V, best_local_W, n, m, p):
                        best_U, best_V, best_W = best_local_U, best_local_V, best_local_W
                        best_rank = target_rank
                        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _compute_error(U, V, W, n, m, p):
    """Compute the error of the tensor decomposition."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return float('inf')
    
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    W_reshaped = W_mat.reshape(p, n)
    
    total_error = 0.0
    for i in range(n):
        for j in range(m):
            u_val = U_mat[i*m+j, :]
            v_slice = V_mat[j*p:(j+1)*p, :]
            prod = u_val[None, :] * v_slice
            T_row = prod * W_reshaped[:, i][None, :]
            T_vals = T_row.sum(axis=1)
            total_error += np.sum((T_vals - 1.0) ** 2)
    
    return total_error
# EVOLVE-BLOCK-END
