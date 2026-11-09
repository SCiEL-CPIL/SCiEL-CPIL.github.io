"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Build the schoolbook (rank = n*m*p) decomposition."""
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

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    # Check the tensor identity:
    # For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_with_random(U, V, W, n, m, p, seed=42, trials=10, tol=1e-8):
    """Verify with random matrix multiplications."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rng = np.random.RandomState(seed)
    for _ in range(trials):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        expected = A @ B
        
        # Compute using decomposition
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # Using: C[i,k] = sum_c (sum_j A[i,j]*U[i*m+j,c]) * (sum_j B[j,k]*V[j*p+k,c]) * W[k*n+i,c]
        # Wait, that's not right. The decomposition is:
        # For the tensor T where T[i,j,k] = 1, we have:
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # So C[i,k] = sum_j A[i,j] * B[j,k] * T[i,j,k] = sum_j A[i,j] * B[j,k]
        # But we need to compute it as:
        # C[i,k] = sum_c W[k*n+i, c] * (sum_j A[i,j] * U[i*m+j, c]) * (sum_j B[j,k] * V[j*p+k, c])
        
        # Actually, the correct interpretation:
        # The multiplication is: C = A @ B
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor decomposition says: for each (i,j,k), the coefficient is 1
        # We express this as: 1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # So: C[i,k] = sum_j A[i,j] * B[j,k] * (sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c])
        #       = sum_c W[k*n+i, c] * (sum_j A[i,j] * U[i*m+j, c]) * (sum_j B[j,k] * V[j*p+k, c])
        
        # Let's compute it this way
        computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    au = sum(A[i, j] * U[i * m + j, c] for j in range(m))
                    bv = sum(B[j, k] * V[j * p + k, c] for j in range(m))
                    s += W[k * n + i, c] * au * bv
                computed[i, k] = s
        
        if not np.allclose(computed, expected, atol=tol, rtol=tol):
            return False
    return True

def _try_rank_reduction(U, V, W, n, m, p, target_rank, max_iters=500, tol=1e-7):
    """
    Try to reduce the rank by finding a lower-rank approximation.
    Uses iterative refinement: start from current decomposition, try to 
    project onto lower-rank subspace.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    current_rank = U.shape[1]
    
    if current_rank <= target_rank:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build the tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # We want T to equal the matrix multiplication tensor.
    
    # The matrix multiplication tensor M has:
    # M[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k)
    # M[other] = 0
    
    # Build target tensor
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Start from current decomposition, try to reduce rank
    # Use SVD-based approach: reshape and find low-rank approximation
    
    # For now, try a simple approach: randomly perturb and optimize
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_error = float('inf')
    
    for iteration in range(max_iters):
        # Create a lower-rank candidate by zeroing out some columns
        # and adjusting the remaining ones
        
        # Pick target_rank columns to keep
        if current_rank > target_rank:
            # Randomly select which columns to keep
            keep_indices = np.random.choice(current_rank, target_rank, replace=False)
            new_U = U[:, keep_indices].copy()
            new_V = V[:, keep_indices].copy()
            new_W = W[:, keep_indices].copy()
        else:
            new_U = U.copy()
            new_V = V.copy()
            new_W = W.copy()
        
        # Compute error
        # For each (i,j,k), check if sum_c new_U[i*m+j, c] * new_V[j*p+k, c] * new_W[k*n+i, c] = 1
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(new_U.shape[1]):
                        s += new_U[i * m + j, c] * new_V[j * p + k, c] * new_W[k * n + i, c]
                    error += (s - 1.0) ** 2
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = new_U.copy(), new_V.copy(), new_W.copy()
        
        if best_error < tol:
            break
        
        # Add small random perturbation to explore
        new_U += np.random.randn(*new_U.shape) * 0.1
        new_V += np.random.randn(*new_V.shape) * 0.1
        new_W += np.random.randn(*new_W.shape) * 0.1
    
    if best_error < tol:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4×5 × 5×6 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    result = _build_schoolbook(n, m, p)
    
    # Verify it's correct
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    current_rank = len(result["U"][0])
    
    # Try to reduce rank
    # Try target ranks from current_rank-1 down to some minimum
    min_possible = n * p  # Lower bound: at least n*p multiplications needed
    for target_rank in range(current_rank - 1, max(min_possible - 1, 50), -1):
        if time.time() > deadline - 30:
            break
        
        candidate = _try_rank_reduction(
            result["U"], result["V"], result["W"],
            n, m, p, target_rank,
            max_iters=200, tol=1e-7
        )
        
        if candidate is not None:
            # Verify the candidate
            if _verify_decomposition(candidate["U"], candidate["V"], candidate["W"], n, m, p, tol=1e-8):
                if _verify_with_random(candidate["U"], candidate["V"], candidate["W"], n, m, p, seed=42, trials=5, tol=1e-6):
                    result = candidate
                    current_rank = target_rank
                    break  # Found a valid lower-rank decomposition
    
    return result
# EVOLVE-BLOCK-END
