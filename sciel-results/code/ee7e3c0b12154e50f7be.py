"""Program for matrix multiplication 3x3 by 3x5 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 45
    
    # Lower bound
    lower_bound = max(n * m, m * p, p * n)  # 15
    
    # Build the tensor: entries are 1 at (i,j,k)
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # Contraction: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    def make_tensor_indices():
        """Return list of (i, j, k) tuples."""
        return [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    indices = make_tensor_indices()
    num_indices = len(indices)
    
    def compute_loss(U, V, W, indices):
        """Compute sum of squared errors for the tensor decomposition."""
        loss = 0.0
        for (i, j, k) in indices:
            u = U[i * m + j]
            v = V[j * p + k]
            w = W[k * n + i]
            val = np.dot(u * w, v)
            loss += (val - 1.0) ** 2
        return loss
    
    def verify_exact(U, V, W, indices, tol=1e-8):
        """Verify that the decomposition is exact within tolerance."""
        for (i, j, k) in indices:
            u = U[i * m + j]
            v = V[j * p + k]
            w = W[k * n + i]
            val = np.dot(u * w, v)
            if abs(val - 1.0) > tol:
                return False
        return True
    
    def to_half_integer_matrix(M):
        """Round matrix entries to nearest half-integer."""
        return np.round(M * 2) / 2
    
    def is_half_integer_matrix(M, tol=1e-6):
        """Check if all entries are integer or half-integer."""
        return np.all(np.abs(M * 2 - np.round(M * 2)) < tol)
    
    # Schoolbook solution as fallback
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    for col, (i, j, k) in enumerate(indices):
        U_sb[i * m + j, col] = 1
        V_sb[j * p + k, col] = 1
        W_sb[k * n + i, col] = 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find a lower rank decomposition
    # Use a more sophisticated approach: start from schoolbook and try to merge terms
    # or use gradient descent with better initialization
    
    rng = np.random.RandomState(42)
    
    # First, try a structured approach: exploit the block structure
    # The matrix multiplication C = A @ B where A is 3x3 and B is 3x5
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let's try to find a decomposition with rank < 45
    # Strategy: use gradient descent with multiple restarts and adaptive learning rate
    
    for target_rank in range(lower_bound, rank_schoolbook):
        if time.time() > deadline - 15:
            break
        
        found = False
        n_restarts = 10 if target_rank >= 30 else 15
        
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            
            # Random initialization with different scales
            scale = 0.5 + 0.1 * restart
            U = rng.randn(n * m, target_rank) * scale
            V = rng.randn(m * p, target_rank) * scale
            W = rng.randn(p * n, target_rank) * scale
            
            # Gradient descent with adaptive learning rate
            lr = 0.05
            n_iter = 3000
            prev_loss = compute_loss(U, V, W, indices)
            best_loss_in_run = prev_loss
            best_U_run, best_V_run, best_W_run = U.copy(), V.copy(), W.copy()
            
            for it in range(n_iter):
                if time.time() > deadline - 15:
                    break
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for (i, j, k) in indices:
                    u = U[i * m + j]
                    v = V[j * p + k]
                    w = W[k * n + i]
                    val = np.dot(u * w, v)
                    err = val - 1.0
                    # d/du: w * v * err
                    # d/dv: u * w * err
                    # d/dw: u * v * err
                    dU[i * m + j] += err * w * v
                    dV[j * p + k] += err * u * w
                    dW[k * n + i] += err * u * v
                
                # Update with momentum-like behavior
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                loss = compute_loss(U, V, W, indices)
                
                if loss < best_loss_in_run:
                    best_loss_in_run = loss
                    best_U_run, best_V_run, best_W_run = U.copy(), V.copy(), W.copy()
                
                # Adaptive learning rate
                if loss < prev_loss:
                    lr = min(lr * 1.02, 0.2)
                else:
                    lr *= 0.8
                    U = best_U_run
                    V = best_V_run
                    W = best_W_run
                    loss = best_loss_in_run
                prev_loss = loss
                
                if loss < 1e-10:
                    break
            
            # Try rounding to half-integers
            U_h = to_half_integer_matrix(best_U_run)
            V_h = to_half_integer_matrix(best_V_run)
            W_h = to_half_integer_matrix(best_W_run)
            
            if verify_exact(U_h, V_h, W_h, indices):
                best_rank = target_rank
                best_U, best_V, best_W = U_h, V_h, W_h
                found = True
                break
        
        if found:
            continue
    
    # If we didn't find anything better, return schoolbook
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
