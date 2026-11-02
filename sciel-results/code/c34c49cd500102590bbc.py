import numpy as np
import time

DIMENSIONS = [4, 4, 6]
DEADLINE_SECONDS = 140.0  # Safety margin before 150s limit


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly represents matrix multiplication."""
    rank = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Build the tensor T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Check a few random entries
    for _ in range(50):
        i = np.random.randint(n)
        j = np.random.randint(m)
        k = np.random.randint(p)
        
        # Expected: (A @ B)[i, k] = sum_j A[i,j] * B[j,k]
        # In tensor form: T[i*m+j, j*p+k, k*n+i] should be 1 if this term contributes
        # Actually, the tensor T satisfies: for each r, the term U[:,r] * V[:,r] * W[:,r]
        # contributes to T. The full tensor is sum_r U[:,r] outer V[:,r] outer W[:,r]
        
        # The multiplication tensor has T[a, b, c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i
        # for some valid i, j, k. Otherwise 0.
        
        a = i * m + j
        b = j * p + k
        c = k * n + i
        
        # Compute sum_r U[a,r] * V[b,r] * W[c,r]
        val = np.sum(U[a, :] * V[b, :] * W[c, :])
        if abs(val - 1.0) > 1e-9:
            return False
        
        # Also check that off-diagonal terms are zero
        # Pick a random (a,b,c) that doesn't correspond to valid indices
        a2 = np.random.randint(n * m)
        b2 = np.random.randint(m * p)
        c2 = np.random.randint(p * n)
        
        # Check if (a2,b2,c2) is a valid triple
        i2 = a2 // m
        j2 = a2 % m
        j3 = b2 // p
        k2 = b2 % p
        k3 = c2 // n
        i3 = c2 % n
        
        if j2 != j3 or k2 != k3 or i2 != i3:
            # Should be zero
            val2 = np.sum(U[a2, :] * V[b2, :] * W[c2, :])
            if abs(val2) > 1e-9:
                return False
    
    return True


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition: rank = n*m*p = 4*4*6 = 96
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
    
    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Decomposition verification failed")
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS:
        raise TimeoutError("Exceeded deadline")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
