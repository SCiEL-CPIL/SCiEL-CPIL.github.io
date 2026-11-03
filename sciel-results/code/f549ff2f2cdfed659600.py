"""Tensor decomposition for 3x3x7 matrix multiplication."""
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication.
    T[i*m+j, j*p+k, k*n+i] should equal 1 for all i,j,k, and 0 otherwise."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    # Build the tensor T from the decomposition
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Check all required entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    # Check that entries which should be 0 are 0
    # We need to check all (a,b,c) where T[a,b,c] should be 0
    # This is expensive, so check a subset
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                # Determine if this is a valid multiplication index
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                if j_a == j_b and i_a == i_c and k_b == k_c:
                    continue  # This should be 1, already checked
                else:
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > 1e-9:
                        return False
    return True

def _try_rank_r(r, n, m, p, timeout=30):
    """Try to find a rank-r decomposition using random initialization and optimization."""
    start_time = time.time()
    # Random initialization
    np.random.seed(42)
    U = np.random.randn(n * m, r)
    V = np.random.randn(m * p, r)
    W = np.random.randn(p * n, r)
    
    # Build target tensor entries
    target_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_entries.append((a, b, c, 1.0))
    
    # Also add some zero entries
    zero_entries = []
    for _ in range(200):
        a = np.random.randint(0, n * m)
        b = np.random.randint(0, m * p)
        c = np.random.randint(0, p * n)
        i_a = a // m
        j_a = a % m
        j_b = b // p
        k_b = b % p
        k_c = c // n
        i_c = c % n
        if not (j_a == j_b and i_a == i_c and k_b == k_c):
            zero_entries.append((a, b, c, 0.0))
    
    all_entries = target_entries + zero_entries
    
    # Simple gradient descent
    lr = 0.01
    for iteration in range(500):
        if time.time() - start_time > timeout:
            break
        # Compute gradient
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for a, b, c, target in all_entries:
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            err = val - target
            for r_idx in range(r):
                grad_U[a, r_idx] += err * V[b, r_idx] * W[c, r_idx]
                grad_V[b, r_idx] += err * U[a, r_idx] * W[c, r_idx]
                grad_W[c, r_idx] += err * U[a, r_idx] * V[b, r_idx]
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Check convergence
        if iteration % 50 == 0:
            max_err = 0
            for a, b, c, target in target_entries:
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                max_err = max(max_err, abs(val - target))
            if max_err < 1e-6:
                break
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return U, V, W
    return None

def construct():
    n, m, p = 3, 3, 7
    deadline = time.time() + 140  # Leave 10s margin
    
    # Try rank 21
    if time.time() < deadline:
        result = _try_rank_r(21, n, m, p, timeout=60)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank 22
    if time.time() < deadline:
        result = _try_rank_r(22, n, m, p, timeout=30)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to trivial rank-63 decomposition
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
