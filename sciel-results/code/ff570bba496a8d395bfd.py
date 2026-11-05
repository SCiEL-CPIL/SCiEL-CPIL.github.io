import numpy as np
import time
import itertools
import sys

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    # Schoolbook decomposition: rank = n*m*p = 72
    def schoolbook():
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
        return U, V, W, rank
    
    U0, V0, W0, rank0 = schoolbook()
    best_U, best_V, best_W, best_rank = U0, V0, W0, rank0
    
    def verify(U, V, W, rank, tol=1e-9):
        """Check if decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to reduce rank by finding linear dependencies in the schoolbook decomposition
    # Strategy: check if any column can be expressed as a linear combination of others
    # while preserving the tensor identity
    
    # For the multiplication tensor, each column r corresponds to a triple (i_r, j_r, k_r)
    # where U has 1 at (i_r*m + j_r, r), V has 1 at (j_r*p + k_r, r), W has 1 at (k_r*n + i_r, r)
    # The tensor entry T[i,j,k] = sum_r delta_{i,i_r} * delta_{j,j_r} * delta_{k,k_r}
    # We need this to equal 1 for all (i,j,k) in the range, which means we need exactly one
    # column for each (i,j,k) triple, OR a set of columns whose product sums to 1.
    
    # Since the tensor is all-ones, and each rank-1 term is a "corner" tensor (1 at one position),
    # we need the sum of these corner tensors to equal the all-ones tensor. This means we need
    # at least one term for each (i,j,k), so rank >= n*m*p = 72 is a lower bound IF we restrict
    # to 0/1 entries. But with general entries, we might do better.
    
    # Let's try a small optimization: see if we can merge some columns
    # by allowing fractional entries.
    
    # Try rank 71: can we express one column as a combination of others?
    # For the all-ones tensor, if we remove column (i0,j0,k0), we need to add contributions
    # to T[i0,j0,k0] from other columns. But other columns only contribute to their own (i,j,k)
    # positions. So we can't fill in T[i0,j0,k0] without a column that has support there.
    # This means for 0/1 schoolbook, rank 72 is minimal.
    
    # With general entries, we can have a column that contributes to multiple (i,j,k) positions.
    # For example, a column with U[i,j,r] = a, V[j,k,r] = b, W[k,i,r] = c contributes
    # a*b*c to T[i,j,k] for the specific (i,j,k) determined by the row indices.
    # Wait, no: each column r has specific row indices in U, V, W.
    # U has shape (n*m, rank), so row i*m+j in U.
    # V has shape (m*p, rank), so row j*p+k in V.
    # W has shape (p*n, rank), so row k*n+i in W.
    # So column r contributes U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] to T[i,j,k].
    # For a given column r, it contributes to ALL (i,j,k) where the products are nonzero.
    
    # So we can have a column that contributes to multiple (i,j,k) positions!
    # This is the key to reducing rank below 72.
    
    # Let's try to find a decomposition with rank < 72 using optimization.
    # We'll use a structured initialization and gradient descent.
    
    def try_optimize_rank(target_rank, max_iter=2000, lr=0.001, seed=42):
        """Try to find a valid decomposition with the given rank."""
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                return None
            
            # Compute loss and gradients
            loss = 0.0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        err = val - 1.0
                        loss += err * err
                        
                        # Gradients
                        for r in range(target_rank):
                            u_val = U[i*m+j, r]
                            v_val = V[j*p+k, r]
                            w_val = W[k*n+i, r]
                            grad_U[i*m+j, r] += 2 * err * v_val * w_val
                            grad_V[j*p+k, r] += 2 * err * u_val * w_val
                            grad_W[k*n+i, r] += 2 * err * u_val * v_val
            
            # Update with learning rate decay
            current_lr = lr * (0.999 ** it)
            U -= current_lr * grad_U
            V -= current_lr * grad_V
            W -= current_lr * grad_W
            
            if it % 500 == 0 and loss < 1e-8:
                break
        
        if loss < 1e-8:
            # Round to half-integers for cleanliness
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            if verify(U, V, W, target_rank, tol=1e-6):
                return U, V, W, target_rank
        return None
    
    # Try a few ranks below 72
    for target_rank in range(71, max(60, rank0 - 10), -1):
        if time.time() > deadline - 20:
            break
        
        result = try_optimize_rank(target_rank, max_iter=1000, lr=0.005, seed=42 + (72 - target_rank))
        if result is not None:
            best_U, best_V, best_W, best_rank = result
            break
    
    # Use the best result found
    U_final = best_U if best_rank < rank0 else U0
    V_final = best_V if best_rank < rank0 else V0
    W_final = best_W if best_rank < rank0 else W0
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
