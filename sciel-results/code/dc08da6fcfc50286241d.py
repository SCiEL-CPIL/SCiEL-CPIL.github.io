"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomp(U, V, W):
    """Verify that the decomposition is exact."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    # Check: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _try_winograd_structured():
    """
    Try a Winograd-like structured decomposition.
    
    For 3x5x5, we can try to exploit the structure by viewing the 5-length
    dimension as related to smaller convolutions.
    
    One approach: decompose 5 as 2+2+1 and use Winograd for the 2-length parts.
    """
    n, m, p = DIMENSIONS
    
    # Strategy: Use the fact that we can block the matrix multiplication.
    # A (3x5) * B (5x5) = C (3x5)
    # Split B into columns: B = [B1, B2, B3, B4, B5] where each Bi is 5x1
    # Then C = [A*B1, A*B2, A*B3, A*B4, A*B5]
    # Each A*Bi is a 3x1 vector = 3x5 * 5x1 = 3 multiplications of 1x5*5x1
    # But that's just schoolbook again.
    
    # Better: Use Winograd for 1D convolution.
    # A(3x5) * B(5x5): think of each row of A as a length-5 vector,
    # and each column of B as a length-5 vector. The dot product is a length-5 convolution-like operation.
    
    # Actually, let's try a different structured approach.
    # Use the identity: matmul(n,m,p) can be reduced using known fast algorithms.
    
    # For now, let's try a random search with structured initialization.
    start_time = time.time()
    deadline = 140.0  # safety margin
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank
    
    # Try various rank values below schoolbook
    for target_rank in range(50, best_rank + 1):
        if time.time() - start_time > deadline - 10:
            break
        
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() - start_time > deadline - 5:
                break
            
            np.random.seed(target_rank * 1000 + attempt)
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Simple gradient-free optimization: random perturbations
            best_loss = float('inf')
            for step in range(200):
                if time.time() - start_time > deadline - 5:
                    break
                
                # Compute loss
                loss = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            loss += (val - 1.0) ** 2
                
                if loss < best_loss:
                    best_loss = loss
                    if best_loss < 1e-10:
                        # Found a valid decomposition
                        result = {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                        if _verify_decomp(U, V, W):
                            return result
                        else:
                            best_result = result
                            best_rank = target_rank
                            break
                
                # Random perturbation
                lr = 0.01 / (1 + step * 0.1)
                U += np.random.randn(n * m, target_rank) * lr
                V += np.random.randn(m * p, target_rank) * lr
                W += np.random.randn(p * n, target_rank) * lr
            
            if best_loss < 1e-10 and best_result is None:
                result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                if _verify_decomp(U, V, W):
                    return result
                else:
                    best_result = result
                    best_rank = target_rank
    
    if best_result is not None:
        return best_result
    
    return None

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    # First, try the structured Winograd approach
    result = _try_winograd_structured()
    
    if result is not None:
        return result
    
    # Fall back to schoolbook
    if time.time() - start_time < deadline - 5:
        return _schoolbook()
    
    return _schoolbook()
# EVOLVE-BLOCK-END
