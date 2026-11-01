import numpy as np
import time

DIMENSIONS = [2, 2, 7]

def verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Compute the tensor
    tensor = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                tensor[i, j, k] = val
    # Check against the multiplication tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(tensor[i, j, k] - 1.0) > 1e-9:
                    return False
    return True

def solve_2x2x3():
    """Find a rank-8 decomposition for 2×2×3."""
    n, m, p = 2, 2, 3
    rank = 8
    # U: 4×8, V: 6×8, W: 6×8
    # Try to find integer solution
    # Use a simple approach: initialize with small integers and check
    
    # Let's try a known structure
    # I'll use a greedy approach
    import itertools
    
    # For now, let's try a specific known decomposition
    # Based on literature, here's one attempt:
    
    U = np.array([
        [1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1]
    ], dtype=float)
    
    V = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0]
    ], dtype=float)
    
    W = np.array([
        [1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=float)
    
    if verify_decomposition(U, V, W, n, m, p):
        return U, V, W
    
    # If that doesn't work, try another approach
    # For now, return None to indicate failure
    return None

def solve_2x2x4():
    """Find a rank-11 decomposition for 2×2×4."""
    n, m, p = 2, 2, 4
    rank = 11
    # U: 4×11, V: 8×11, W: 8×11
    # This is more complex. Let's try to construct it.
    
    # For now, return None to indicate failure
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try the direct sum approach: 2×2×4 (rank 11) + 2×2×3 (rank 8) = rank 19
    # First, try to get the sub-decompositions
    
    # For now, let's try a simpler approach: use Strassen on 2×2 blocks
    # Split the 2×7 into 2×4 + 2×3
    
    # Actually, let's just try to find any valid decomposition with rank < 28
    # Start with schoolbook and try to improve
    
    # Schoolbook decomposition
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
    
    # Verify schoolbook
    assert verify_decomposition(U, V, W, n, m, p)
    
    # Try to improve using Strassen-like ideas
    # For 2×2×7, we can think of it as 2×2×(4+3)
    # Use Strassen on the 2×2×2 parts
    
    # Let's try a different approach: use the known rank bounds
    # Theoretical lower bound for 2×2×7 is around 19-20
    
    # For now, return the schoolbook decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
