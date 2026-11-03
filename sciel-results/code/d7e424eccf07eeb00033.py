"""Program for matrix multiplication 3x3 by 3x5 tensor decomposition."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 45
    
    # Target: find rank < rank_schoolbook
    # Lower bound
    lower_bound = max(n * m, m * p, p * n)  # 15
    
    # Build the tensor: entries are 1 at (i,j,k)
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # Contraction: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    def make_tensor_indices():
        """Return list of (i, j, k) tuples."""
        return [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
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
    
    indices = make_tensor_indices()
    
    def to_half_integer_matrix(M):
        """Round matrix entries to nearest half-integer."""
        return np.round(M * 2) / 2
    
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
    
    # Search for lower rank
    # Try ranks from lower_bound to rank_schoolbook - 1
    # Use random initialization + gradient descent
    
    rng = np.random.RandomState(42)
    
    for rank in range(lower_bound, rank_schoolbook):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        n_restarts = 5 if rank >= 25 else 8
        found = False
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = rng.randn(n * m, rank) * 0.5
            V = rng.randn(m * p, rank) * 0.5
            W = rng.randn(p * n, rank) * 0.5
            
            # Gradient descent
            lr = 0.01
            n_iter = 2000
            prev_loss = compute_loss(U, V, W, indices)
            
            for it in range(n_iter):
                if time.time() > deadline - 10:
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
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                loss = compute_loss(U, V, W, indices)
                
                # Adaptive learning rate
                if loss < prev_loss:
                    lr *= 1.05
                else:
                    lr *= 0.7
                    U += lr * dU
                    V += lr * dV
                    W += lr * dW
                    loss = prev_loss
                prev_loss = loss
                
                if loss < 1e-10:
                    break
            
            # Round to half-integers and verify
            U_h = to_half_integer_matrix(U)
            V_h = to_half_integer_matrix(V)
            W_h = to_half_integer_matrix(W)
            
            if verify_exact(U_h, V_h, W_h, indices):
                best_rank = rank
                best_U, best_V, best_W = U_h, V_h, W_h
                found = True
                break
        
        if found:
            # Try to go even lower in next iteration
            continue
    
    # Return best result
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
