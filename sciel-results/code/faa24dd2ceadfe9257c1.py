"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def _try_rank_decomposition(T, target_rank, max_iters=500, tol=1e-6):
    """Try to find a rank-target_rank decomposition using gradient descent."""
    n_a, n_b, n_c = T.shape
    np.random.seed(42)
    
    # Initialize with random matrices
    U = np.random.randn(n_a, target_rank) * 0.1
    V = np.random.randn(n_b, target_rank) * 0.1
    W = np.random.randn(n_c, target_rank) * 0.1
    
    lr = 0.01
    for it in range(max_iters):
        # Compute current decomposition
        result = np.zeros(T.shape)
        for r in range(target_rank):
            result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
        error = result - T
        err_norm = np.linalg.norm(error)
        if err_norm < tol:
            return U, V, W, err_norm
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for r in range(target_rank):
            dU[:,r] = np.einsum('abc,bc->a', error, np.outer(V[:,r], W[:,r]).reshape(n_b, n_c))
            dV[:,r] = np.einsum('abc,ac->b', error, np.outer(U[:,r], W[:,r]).reshape(n_a, n_c))
            dW[:,r] = np.einsum('abc,ab->c', error, np.outer(U[:,r], V[:,r]).reshape(n_a, n_b))
        
        # Gradient descent
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    result = np.zeros(T.shape)
    for r in range(target_rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    err_norm = np.linalg.norm(result - T)
    return U, V, W, err_norm

def _round_to_rational(U, V, W, max_denom=100):
    """Try to round entries to simple rationals."""
    U_r = [[Fraction(int(round(x*max_denom))/max_denom if abs(x*max_denom - round(x*max_denom)) < 0.01 else x) for x in row] for row in U]
    V_r = [[Fraction(int(round(x*max_denom))/max_denom if abs(x*max_denom - round(x*max_denom)) < 0.01 else x) for x in row] for row in V]
    W_r = [[Fraction(int(round(x*max_denom))/max_denom if abs(x*max_denom - round(x*max_denom)) < 0.01 else x) for x in row] for row in W]
    return U_r, V_r, W_r

def _check_exact(U, V, W, T):
    """Check if decomposition is exact with rational entries."""
    rank = len(U[0])
    result = np.zeros(T.shape)
    for r in range(rank):
        for a in range(len(U)):
            for b in range(len(V)):
                for c in range(len(W)):
                    result[a,b,c] += float(U[a][r]) * float(V[b][r]) * float(W[c][r])
    return np.allclose(result, T, atol=1e-10)

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Try ranks from 10 to 17
    best_result = None
    for target_rank in range(10, 18):
        if time.time() > deadline - 10:
            break
        
        U, V, W, err = _try_rank_decomposition(T, target_rank, max_iters=200, tol=1e-6)
        
        if err < 1e-4:
            # Try to round to rationals
            U_r, V_r, W_r = _round_to_rational(U, V, W)
            if _check_exact(U_r, V_r, W_r, T):
                # Convert to lists
                U_list = [[float(x) for x in row] for row in U_r]
                V_list = [[float(x) for x in row] for row in V_r]
                W_list = [[float(x) for x in row] for row in W_r]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook
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
# EVOLVE-BLOCK-END
