"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For matrix multiplication, T[i,j,k] should be 1 for all valid i,j,k
    # because (AB)_{ik} = sum_j A_{ij} B_{jk}
    # Let's verify by checking that the decomposition reproduces the multiplication tensor
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i,j,k] = np.sum(U[i*m+j,:] * V[j*p+k,:] * W[k*n+i,:])
    # The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=1e-10)

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_rank(rank, time_budget=10):
    """Try to find a decomposition of given rank using random initialization and optimization."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Initialize with random values
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Scale to reasonable magnitudes
    U /= np.sqrt(rank)
    V /= np.sqrt(rank)
    W /= np.sqrt(rank)
    
    # Use gradient descent to minimize ||T - T_target||^2
    T_target = np.ones((n, m, p))
    
    lr = 0.1
    for step in range(5000):
        if time.time() - start_time > time_budget:
            break
        
        # Compute current tensor
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] = np.sum(U[i*m+j,:] * V[j*p+k,:] * W[k*n+i,:])
        
        # Check convergence
        err = np.linalg.norm(T - T_target)
        if err < 1e-8:
            return U.tolist(), V.tolist(), W.tolist()
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    dT = T[i,j,k] - T_target[i,j,k]
                    dU[i*m+j,:] += dT * V[j*p+k,:] * W[k*n+i,:]
                    dV[j*p+k,:] += dT * U[i*m+j,:] * W[k*n+i,:]
                    dW[k*n+i,:] += dT * U[i*m+j,:] * V[j*p+k,:]
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    # Final check
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i,j,k] = np.sum(U[i*m+j,:] * V[j*p+k,:] * W[k*n+i,:])
    if np.allclose(T, T_target, atol=1e-8):
        return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Try schoolbook first as baseline
    result = _schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known lower bounds for matrix multiplication rank
    # For 3x5x5, try ranks from known lower bounds up to schoolbook
    for rank in range(40, best_rank):
        if time.time() > deadline:
            break
        
        max_time_per_rank = min(5, max(1, (deadline - time.time()) / (best_rank - rank)))
        found = _try_rank(rank, time_budget=max_time_per_rank)
        if found is not None:
            U, V, W = found
            if _verify(U, V, W):
                return {"U": U, "V": V, "W": W}
    
    # Return schoolbook if no better found
    return result

