"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

TIME_LIMIT = 145.0
START_TIME = time.time()

def time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array T[i,j,k] where T[i,j,k]=1 if we need A[i,j]*B[j,k]"""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor"""
    rank = U.shape[1]
    # Check that U @ V @ W^T reconstructs the tensor
    # For each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        # Compute contribution to tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    # This is expensive, so we only do spot checks
                    pass
    # Better: compute full tensor and compare
    reconstructed = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    reconstructed[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    target = build_tensor(n, m, p)
    return np.allclose(reconstructed, target, atol=1e-10)

def schoolbook_solution(n, m, p):
    """Return the schoolbook (naive) decomposition"""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1.0
                V[j*p + k, idx] = 1.0
                W[k*n + i, idx] = 1.0
                idx += 1
    return U, V, W

def try_rank_reduction(U, V, W, n, m, p, target_rank, max_iterations=1000, seed=42):
    """
    Try to reduce the rank of the decomposition by removing linearly dependent terms.
    Uses iterative refinement: try to express one term as a combination of others.
    """
    current_rank = U.shape[1]
    if current_rank <= target_rank:
        return U, V, W
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    rng = np.random.RandomState(seed)
    
    # Try to remove terms one by one by checking if they can be expressed as linear combinations
    for r_remove in range(current_rank - target_rank):
        if time_left() < 5:
            break
            
        # Try each term to see if it can be removed
        found_removal = False
        for r in range(current_rank):
            if time_left() < 5:
                break
            
            # Check if column r can be expressed as linear combination of others
            # Build matrix of other columns
            other_indices = [i for i in range(current_rank) if i != r]
            
            # For each position (i,j,k) in tensor, we have:
            # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = sum_s alpha_s * U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
            
            # This is a bilinear system, hard to solve directly.
            # Instead, try: can we perturb the decomposition to make column r zero?
            
            # Alternative: try to find if removing column r still gives valid tensor
            # by checking if the remaining columns can still produce the tensor
            U_test = np.delete(U, r, axis=1)
            V_test = np.delete(V, r, axis=1)
            W_test = np.delete(W, r, axis=1)
            
            if verify_decomposition(U_test, V_test, W_test, n, m, p):
                U = U_test
                V = V_test
                W = W_test
                current_rank -= 1
                found_removal = True
                break
        
        if not found_removal:
            # Try random perturbation approach
            # Add small random perturbations and see if we can reduce rank
            break
    
    return U, V, W

def random_search_for_low_rank(n, m, p, target_rank, max_trials=100, seed=42):
    """
    Random search: try random rank-1 tensors and see if they can be combined to form the multiplication tensor.
    This is very inefficient but might find small improvements.
    """
    rng = np.random.RandomState(seed)
    
    for trial in range(max_trials):
        if time_left() < 10:
            break
        
        # Generate random rank-1 tensors
        U_rand = np.random.randn(n*m, target_rank)
        V_rand = np.random.randn(m*p, target_rank)
        W_rand = np.random.randn(p*n, target_rank)
        
        # Normalize
        U_rand /= np.linalg.norm(U_rand, axis=0, keepdims=True)
        V_rand /= np.linalg.norm(V_rand, axis=0, keepdims=True)
        W_rand /= np.linalg.norm(W_rand, axis=0, keepdims=True)
        
        # Check if this random decomposition matches the tensor
        if verify_decomposition(U_rand, V_rand, W_rand, n, m, p):
            return U_rand, V_rand, W_rand
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook
    U, V, W = schoolbook_solution(n, m, p)
    
    # Try to reduce rank
    # First, try simple linear dependence removal
    U, V, W = try_rank_reduction(U, V, W, n, m, p, target_rank=schoolbook_rank - 1)
    
    current_rank = U.shape[1]
    
    # If we reduced, try to reduce further
    while current_rank > 1 and time_left() > 20:
        U, V, W = try_rank_reduction(U, V, W, n, m, p, target_rank=current_rank - 1)
        new_rank = U.shape[1]
        if new_rank == current_rank:
            break
        current_rank = new_rank
    
    # Convert to lists with integer/half-integer entries if possible
    def to_integer_or_half(arr):
        """Convert array to integer or half-integer values if close"""
        arr = arr.copy()
        # Check if values are close to integers or half-integers
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                # Check if close to integer
                nearest_int = round(val)
                if abs(val - nearest_int) < 1e-10:
                    arr[i, j] = float(nearest_int)
                else:
                    # Check if close to half-integer
                    nearest_half = round(2*val) / 2.0
                    if abs(val - nearest_half) < 1e-10:
                        arr[i, j] = nearest_half
        return arr
    
    U = to_integer_or_half(U)
    V = to_integer_or_half(V)
    W = to_integer_or_half(W)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
