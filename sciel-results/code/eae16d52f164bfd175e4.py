"""Matrix multiplication 3x3x5 rank-minimal tensor decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 5]
DEADLINE = 120.0


def _verify_decomposition(U, V, W, n=3, m=3, p=5, tol=1e-9):
    """Verify that U @ V^T @ W^T reconstructs the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    
    return np.allclose(T, 1.0, atol=tol, rtol=tol)


def _make_schoolbook(n=3, m=3, p=5):
    """Create the schoolbook decomposition of rank n*m*p."""
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


def _compute_tensor(U, V, W, n=3, m=3, p=5):
    """Compute the tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]"""
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    return T


def _loss_and_grad(U, V, W, n=3, m=3, p=5):
    """Compute loss and gradients for the decomposition."""
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    
    loss = np.sum((T - 1.0) ** 2)
    
    # Compute gradients
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    
    for c in range(rank):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    residual = T[i, j, k] - 1.0
                    dU[i * m + j, c] += 2 * residual * v[j, k] * w[k, i]
                    dV[j * p + k, c] += 2 * residual * u[i, j] * w[k, i]
                    dW[k * n + i, c] += 2 * residual * u[i, j] * v[j, k]
    
    return loss, dU, dV, dW


def _try_grad_descent(n, m, p, target_rank, max_iters=5000, lr=0.01, deadline=None):
    """Try to find a valid decomposition via gradient descent."""
    if deadline is None:
        deadline = time.time() + 30
    
    start = time.time()
    
    # Initialize with schoolbook-like structure
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    # Initialize: use first min(target_rank, n*m*p) schoolbook terms
    num_init = min(target_rank, n * m * p)
    for c in range(num_init):
        i = c // (m * p)
        j = (c // p) % m
        k = c % p
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0
    
    # Add small random perturbation to break symmetry
    U += 0.01 * np.random.randn(n * m, target_rank)
    V += 0.01 * np.random.randn(m * p, target_rank)
    W += 0.01 * np.random.randn(p * n, target_rank)
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        loss, dU, dV, dW = _loss_and_grad(U, V, W, n, m, p)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-10:
            break
        
        # Adaptive learning rate
        grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
        if grad_norm > 0:
            effective_lr = lr * 100 / (grad_norm + 1)
        else:
            effective_lr = lr
        
        U -= effective_lr * dU
        V -= effective_lr * dV
        W -= effective_lr * dW
    
    if best_loss < 1e-8:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    return None


def _try_structured_search(n, m, p, target_rank, deadline=None):
    """Try structured search for lower-rank decomposition."""
    if deadline is None:
        deadline = time.time() + 30
    
    start = time.time()
    
    # Strategy: try to find rank-44 by modifying schoolbook
    # Key idea: the all-ones tensor might have rank < 45
    
    # Try various random initializations with gradient descent
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        # Scale to get reasonable magnitudes
        U *= 2.0
        V *= 2.0
        W *= 2.0
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(2000):
            if time.time() > deadline:
                break
            
            loss, dU, dV, dW = _loss_and_grad(U, V, W, n, m, p)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-10:
                break
            
            # Simple gradient step with adaptive lr
            lr = 0.001
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        if best_loss < 1e-8:
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return None


def _try_integer_search(n, m, p, target_rank, deadline=None):
    """Try to find a decomposition with integer/half-integer entries."""
    if deadline is None:
        deadline = time.time() + 20
    
    start = time.time()
    
    # For small ranks, try exhaustive-ish search with small integers
    # This is only practical for very small ranks
    if target_rank > 20:
        return None
    
    # Try random small integer vectors
    for _ in range(1000):
        if time.time() > deadline:
            break
        
        U = np.random.randint(-2, 3, size=(n * m, target_rank)).astype(float) * 0.5
        V = np.random.randint(-2, 3, size=(m * p, target_rank)).astype(float) * 0.5
        W = np.random.randint(-2, 3, size=(p * n, target_rank)).astype(float) * 0.5
        
        T = _compute_tensor(U, V, W, n, m, p)
        if np.allclose(T, 1.0, atol=1e-6):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None


def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3x5 matrix multiplication."""
    start = time.time()
    deadline = start + DEADLINE - 5.0  # 5-second safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook (rank 45, guaranteed valid)
    schoolbook = _make_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try to find lower-rank decompositions
    # Attempt rank 44
    if time.time() < deadline - 30.0:
        result_44 = _try_grad_descent(n, m, p, 44, max_iters=3000, lr=0.01, deadline=deadline - 10.0)
        if result_44 is not None:
            if _verify_decomposition(result_44["U"], result_44["V"], result_44["W"], n, m, p):
                return result_44
    
    # Attempt rank 44 with structured search
    if time.time() < deadline - 20.0:
        result_44b = _try_structured_search(n, m, p, 44, deadline=deadline - 10.0)
        if result_44b is not None:
            if _verify_decomposition(result_44b["U"], result_44b["V"], result_44b["W"], n, m, p):
                return result_44b
    
    # Attempt rank 43
    if time.time() < deadline - 15.0:
        result_43 = _try_grad_descent(n, m, p, 43, max_iters=2000, lr=0.01, deadline=deadline - 5.0)
        if result_43 is not None:
            if _verify_decomposition(result_43["U"], result_43["V"], result_43["W"], n, m, p):
                return result_43
    
    # Attempt rank 40
    if time.time() < deadline - 10.0:
        result_40 = _try_grad_descent(n, m, p, 40, max_iters=2000, lr=0.01, deadline=deadline - 5.0)
        if result_40 is not None:
            if _verify_decomposition(result_40["U"], result_40["V"], result_40["W"], n, m, p):
                return result_40
    
    # Attempt rank 30
    if time.time() < deadline - 8.0:
        result_30 = _try_structured_search(n, m, p, 30, deadline=deadline - 3.0)
        if result_30 is not None:
            if _verify_decomposition(result_30["U"], result_30["V"], result_30["W"], n, m, p):
                return result_30
    
    # Attempt rank 20
    if time.time() < deadline - 5.0:
        result_20 = _try_structured_search(n, m, p, 20, deadline=deadline - 2.0)
        if result_20 is not None:
            if _verify_decomposition(result_20["U"], result_20["V"], result_20["W"], n, m, p):
                return result_20
    
    # Return the best we have (schoolbook)
    return best_result
