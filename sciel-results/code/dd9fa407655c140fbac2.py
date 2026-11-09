"""Optimized search for low-rank exact decomposition of 3x4x6 matrix multiplication."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 6]

def _construct_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _build_tensor(n, m, p):
    """Build the flattened tensor T[i, j, k] = 1 if the product C[i,k] uses A[i,j]B[j,k]."""
    tensor = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i*m + j, j*p + k, k*n + i] = 1
    return tensor

def _is_exact_decomposition(U, V, W, tensor, tol=1e-10):
    """Check if U, V, W exactly decompose the tensor."""
    try:
        T = np.einsum('ac,bc,dc->abd', U, V, W)
    except Exception:
        return False
    diff = np.abs(T - tensor)
    return np.all(diff < tol)

def _snap_to_half_integer(mat):
    """Snap values to nearest half-integer."""
    return (np.round(mat * 2) / 2)

def _construct_rank24_decomposition():
    """
    Attempt to construct a rank-24 decomposition for 3x4x6.
    
    Strategy: Split the 6 columns into three groups of 2.
    For each group, compute a 3x4x2 product.
    If we can do each 3x4x2 with rank 8, total is 24.
    
    For 3x4x2: C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k], i in [0,3), k in [0,2)
    
    We use the following rank-8 scheme for 3x4x2:
    Let A be 3x4, B be 4x2, C be 3x2.
    
    Split A into A1 (3x2, cols 0-1) and A2 (3x2, cols 2-3).
    Split B into B1 (2x2, rows 0-1) and B2 (2x2, rows 2-3).
    
    C = A1*B1 + A2*B2
    
    Each 3x2x2 product can be done with rank 6 (schoolbook).
    Total rank = 6 + 6 = 12 per 3x4x2 block.
    Three blocks = 36. Not 24.
    
    Alternative: Use a different factorization.
    
    Actually, let's try a direct rank-24 construction by treating the problem
    as a whole and using a structured random search with verification.
    """
    n, m, p = DIMENSIONS
    rank = 24
    
    # Try multiple random seeds with structured initialization
    for seed in range(50):
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize with small integers
        U = np.random.choice([-1, 0, 1], size=(n*m, rank))
        V = np.random.choice([-1, 0, 1], size=(m*p, rank))
        W = np.random.choice([-1, 0, 1], size=(p*n, rank))
        
        # Snap to half-integers
        U = _snap_to_half_integer(U)
        V = _snap_to_half_integer(V)
        W = _snap_to_half_integer(W)
        
        # Check
        tensor = _build_tensor(n, m, p)
        if _is_exact_decomposition(U, V, W, tensor):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    schoolbook = _construct_schoolbook()
    best_rank = n * m * p
    best_decomp = schoolbook
    
    # Build the target tensor for verification
    tensor = _build_tensor(n, m, p)
    
    # Try rank-24 first
    if time.time() < deadline - 20:
        result = _construct_rank24_decomposition()
        if result is not None:
            best_rank = 24
            best_decomp = result
            return best_decomp
    
    # Try other ranks
    for attempt_rank in [30, 36, 48, 60, 64, 68]:
        if time.time() > deadline - 10:
            break
        
        if attempt_rank >= best_rank:
            continue
            
        found = False
        for seed in range(20):
            if time.time() > deadline - 10:
                break
            random.seed(seed + attempt_rank)
            np.random.seed(seed + attempt_rank)
            
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, attempt_rank))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, attempt_rank))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, attempt_rank))
            
            if _is_exact_decomposition(U, V, W, tensor):
                best_rank = attempt_rank
                best_decomp = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                found = True
                break
        
        if found:
            break
    
    return best_decomp

if __name__ == "__main__":
    res = construct()
    print(res["U"][0])
