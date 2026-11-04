import numpy as np
from scipy.optimize import minimize
import time
import random

DIMENSIONS = [4, 4, 9]

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly reproduces matrix multiplication."""
    r = U.shape[1]
    if U.shape != (n*m, r) or V.shape != (m*p, r) or W.shape != (p*n, r):
        return False
    # The tensor T[i,j,k] should equal sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Check a sample of entries
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(r):
                    actual += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                if abs(actual - expected) > tol:
                    return False
    # Also check that non-multiplication entries are zero
    # The tensor only has 1s at positions (i,j,k) where we're computing C[i,k]
    # Actually, the matrix multiplication tensor T has T[i,j,k]=1 for all i,j,k
    # Wait, let me re-read the problem.
    # The output factor follows W[k*n+i,r]
    # So the decomposition is: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # And M[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # So we need M[i,j,k] = 1 for all valid i,j,k
    # But there are no "invalid" entries - all (i,j,k) tuples are valid.
    # So we just need the sum to equal 1 for all i,j,k.
    return True

def _validate_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized validation."""
    r = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Reshape to 3D tensors
    U_3d = U_np.reshape(n, m, r)
    V_3d = V_np.reshape(m, p, r)
    W_3d = W_np.reshape(p, n, r)
    
    # Compute tensor product sum over r
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Use einsum
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    # Check all entries equal 1
    if not np.allclose(T, np.ones((n, m, p)), atol=tol):
        return False
    return True

def _build_block_decomposition(n, m, p):
    """Build a block-based decomposition by splitting into smaller blocks."""
    # Split the m dimension into blocks
    # For 4x4 * 4x9, we can think of it as blocks
    # Let's try to use a 2x2 Strassen-like approach on the 4x4 part
    
    # Actually, let's try a different approach: use the fact that we can
    # decompose the multiplication into independent sub-problems
    
    # Simple approach: treat as (4x4) * (4x9) = sum over j of A[:,j] * B[j,:]
    # Each term is an outer product
    
    # Let's try to find a lower rank decomposition by searching
    
    # Start with schoolbook and try to reduce
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
    
    return U, V, W, rank

def _try_reduce_rank(U, V, W, n, m, p, max_time=100, seed=42):
    """Try to reduce the rank by removing zero or near-zero columns and merging."""
    np.random.seed(seed)
    start_time = time.time()
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    r = U.shape[1]
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = r
    
    # Try to identify columns that can be removed or merged
    # A column r is "redundant" if it can be expressed as a linear combination of others
    # in the tensor decomposition sense
    
    # Simple heuristic: try to zero out entries and check validity
    for iteration in range(200):
        if time.time() - start_time > max_time:
            break
        
        # Try to modify a random column
        col = np.random.randint(0, r)
        
        # Try setting some entries to 0
        for trial in range(5):
            U_test = U.copy()
            V_test = V.copy()
            W_test = W.copy()
            
            # Randomly zero out some entries in this column
            mask_u = np.random.random(n * m) < 0.3
            mask_v = np.random.random(m * p) < 0.3
            mask_w = np.random.random(p * n) < 0.3
            
            U_test[:, col][mask_u] = 0
            V_test[:, col][mask_v] = 0
            W_test[:, col][mask_w] = 0
            
            if _validate_fast(U_test, V_test, W_test, n, m, p):
                U, V, W = U_test, V_test, W_test
                break
        
        # Check if we can remove a column
        for col in range(r - 1, -1, -1):
            if time.time() - start_time > max_time:
                break
            
            # Check if column is all zeros
            if np.all(U[:, col] == 0) and np.all(V[:, col] == 0) and np.all(W[:, col] == 0):
                U = np.delete(U, col, axis=1)
                V = np.delete(V, col, axis=1)
                W = np.delete(W, col, axis=1)
                r = U.shape[1]
                if r < best_rank:
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = r
                break
    
    return best_U, best_V, best_W, best_rank

def _local_search(U, V, W, n, m, p, max_time=100, seed=42):
    """Local search to reduce rank."""
    np.random.seed(seed)
    start_time = time.time()
    
    U = U.copy().astype(np.float64)
    V = V.copy().astype(np.float64)
    W = W.copy().astype(np.float64)
    r = U.shape[1]
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = r
    
    for iteration in range(500):
        if time.time() - start_time > max_time:
            break
        
        # Try a random perturbation
        col = np.random.randint(0, r)
        entry_u = np.random.randint(0, n * m)
        entry_v = np.random.randint(0, m * p)
        entry_w = np.random.randint(0, p * n)
        
        U_test = U.copy()
        V_test = V.copy()
        W_test = W.copy()
        
        # Perturb one entry
        U_test[entry_u, col] += np.random.choice([-1, 0, 0, 0, 1])
        V_test[entry_v, col] += np.random.choice([-1, 0, 0, 0, 1])
        W_test[entry_w, col] += np.random.choice([-1, 0, 0, 0, 1])
        
        if _validate_fast(U_test, V_test, W_test, n, m, p):
            U, V, W = U_test, V_test, W_test
            
            # Try to remove zero columns
            for c in range(r - 1, -1, -1):
                if np.all(U[:, c] == 0) and np.all(V[:, c] == 0) and np.all(W[:, c] == 0):
                    U = np.delete(U, c, axis=1)
                    V = np.delete(V, c, axis=1)
                    W = np.delete(W, c, axis=1)
                    r = U.shape[1]
                    if r < best_rank:
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_rank = r
                    break
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W, rank = _build_block_decomposition(n, m, p)
    
    # Time budget
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try local search with different seeds
    for seed in [42, 123, 456, 789]:
        if time.time() > deadline:
            break
        
        time_budget = max(10, deadline - time.time() - 5)
        U_ls, V_ls, W_ls, r_ls = _local_search(U, V, W, n, m, p, max_time=time_budget, seed=seed)
        
        if r_ls < best_rank:
            best_U, best_V, best_W = U_ls, V_ls, W_ls
            best_rank = r_ls
    
    # Convert to list format
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
