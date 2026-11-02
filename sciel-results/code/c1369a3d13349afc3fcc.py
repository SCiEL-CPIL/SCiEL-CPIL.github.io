"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    schoolbook_rank = n * m * p
    
    # Start with schoolbook as fallback
    best_result = None
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Theoretical lower bounds and known results suggest trying ranks below schoolbook
    # For 2x4 * 4x6, we try ranks from a reasonable minimum up to schoolbook-1
    
    # Known: rank >= max(n*m, m*p, p*n) = max(8, 24, 12) = 24
    # But tensor rank can be lower due to structure
    # Let's try ranks from 20 up to schoolbook-1
    
    max_rank_to_try = min(schoolbook_rank - 1, 30)  # Limit search space
    min_rank_to_try = 20  # Reasonable lower bound
    
    for rank in range(min_rank_to_try, max_rank_to_try + 1):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        success = _try_find_decomposition(n, m, p, rank)
        if success is not None:
            U, V, W = success
            best_rank = rank
            best_result = {"U": U, "V": V, "W": W}
            # If we found a low rank, try to go even lower
            break
    
    if best_result is None:
        # Return schoolbook decomposition
        U = [[0] * schoolbook_rank for _ in range(nm)]
        V = [[0] * schoolbook_rank for _ in range(mp)]
        W = [[0] * schoolbook_rank for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        best_result = {"U": U, "V": V, "W": W}
        best_rank = schoolbook_rank
    
    return best_result

def _try_find_decomposition(n, m, p, rank, max_iters=5000):
    """Try to find an exact rank-`rank` decomposition using structured random search."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    # The tensor T[i,j,k] = delta_{i,k} in appropriate indexing
    # We need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = 1 if (i,j,k) valid
    
    # Use a structured approach: try random integer/half-integer matrices
    # and verify exactness
    
    # Strategy: Use bilinear structure. For matrix multiplication,
    # we can try to exploit the fact that only certain (i,j,k) tuples matter.
    
    # Let's use a more systematic approach: initialize with random values,
    # then use gradient-free optimization to minimize the error
    
    # Actually, for exact decomposition, let's try a different approach:
    # Use the fact that we can represent the multiplication as a sum of rank-1 tensors
    
    # Try multiple random initializations
    np.random.seed(42)
    
    for attempt in range(20):
        if time.time() - START_TIME > TIME_LIMIT - 3:
            return None
        
        # Initialize with small random integers
        scale = 2
        U = np.random.randint(-scale, scale + 1, (nm, rank))
        V = np.random.randint(-scale, scale + 1, (mp, rank))
        W = np.random.randint(-scale, scale + 1, (pn, rank))
        
        # Convert to half-integers: multiply by 2, work with integers, divide by 2 at end
        # Actually, let's work directly with the constraint
        
        # Check if this random guess works (very unlikely, but let's verify the method)
        if _verify_decomposition(n, m, p, U, V, W):
            return (U.tolist(), V.tolist(), W.tolist())
        
        # Try a more structured initialization based on Strassen-like patterns
        if attempt == 5:
            result = _try_structured_decomposition(n, m, p, rank)
            if result is not None:
                return result
    
    return None

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    # The tensor T[i,j,k] should equal 1 if the multiplication is valid, 0 otherwise
    # For matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i,j,k] = delta (it's 1 for all valid (i,j,k) in the sense that
    # each term contributes)
    
    # Actually, the tensor for matrix multiplication is:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Because C[i,k] = sum_j A[i,j] * B[j,k], each (i,j,k) triple contributes
    
    # Wait, let me reconsider. The tensor decomposition requires:
    # For each (i,k), C[i,k] = sum_j A[i,j]*B[j,k]
    # In tensor form: T[i,j,k] = 1 for all valid i,j,k
    # And the decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Let's compute the left side for all (i,j,k)
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is a 3D tensor
    max_error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute the sum
                u_val = U_arr[i * m + j, :]
                v_val = V_arr[j * p + k, :]
                w_val = W_arr[k * n + i, :]
                result = np.sum(u_val * v_val * w_val)
                expected = 1.0
                error = abs(result - expected)
                max_error = max(max_error, error)
    
    # Also check that the decomposition is exact (not just close)
    # Since we want integer/half-integer entries, check if values are close to half-integers
    for arr in [U, V, W]:
        for row in arr:
            for val in row:
                # Check if val is close to a half-integer
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    
    return max_error < 1e-9

def _try_structured_decomposition(n, m, p, rank):
    """Try a structured approach based on known low-rank constructions."""
    # For now, this is a placeholder for more sophisticated methods
    # We could implement Strassen-like algorithms or use known constructions
    return None
# EVOLVE-BLOCK-END
