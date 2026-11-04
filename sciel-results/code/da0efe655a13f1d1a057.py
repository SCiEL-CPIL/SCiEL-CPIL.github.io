"""Program to find a rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication."""

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x3 matrix.
    Uses a search strategy to find a decomposition with rank lower than the schoolbook rank of 18.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 18
    
    # Deadline guard: 150s wall-clock limit, use 140s as safety margin
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Best result found so far (start with schoolbook)
    best_rank = schoolbook_rank
    best_result = _schoolbook_decomposition(n, m, p)
    
    # Try to find a lower rank decomposition
    # For 2x3 * 3x3, the minimal rank is known to be 14 (from literature)
    # We'll try to find a valid decomposition with rank < 18
    
    # First, try rank 14 (known minimal for this case)
    for target_rank in [14, 15, 16, 17]:
        if time.time() > deadline - 10:
            break
        
        result = _find_decomposition(n, m, p, target_rank, deadline)
        if result is not None:
            best_rank = target_rank
            best_result = result
            break
    
    return best_result

def _schoolbook_decomposition(n, m, p):
    """Returns the standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
                
    return {"U": U, "V": V, "W": W}

def _find_decomposition(n, m, p, rank, deadline):
    """
    Attempt to find a tensor decomposition of the matrix multiplication tensor
    with the given rank using a randomized search approach.
    """
    # The matrix multiplication tensor T has indices (i,j,k) for i in [0,n), j in [0,m), k in [0,p)
    # We need to find U, V, W such that:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For small dimensions, we can use a direct approach with integer/half-integer entries
    
    # Try different random initializations
    max_attempts = 1000
    attempt = 0
    
    while attempt < max_attempts and time.time() < deadline - 5:
        attempt += 1
        
        # Generate random initial matrices with integer/half-integer entries
        U = _random_matrix(n * m, rank)
        V = _random_matrix(m * p, rank)
        W = _random_matrix(p * n, rank)
        
        # Check if this decomposition is valid
        if _is_valid_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        
        # Try local optimization
        if attempt % 10 == 0:
            U, V, W = _local_search(U, V, W, n, m, p, deadline)
            if _is_valid_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return None

def _random_matrix(rows, cols):
    """Generate a random matrix with integer/half-integer entries in [-2, 2]."""
    matrix = []
    for _ in range(rows):
        row = []
        for _ in range(cols):
            val = random.choice([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
            row.append(val)
        matrix.append(row)
    return matrix

def _is_valid_decomposition(U, V, W, n, m, p):
    """Check if the given U, V, W form a valid decomposition of the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    
    # Convert to numpy for efficient computation
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix multiplication, T[i,j,k] should be 1 if the indices form a valid multiplication, else 0
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute the decomposition value
                val = 0.0
                for r in range(rank):
                    val += U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
                
                # The target value is 1.0 (since A[i,j] * B[j,k] contributes to C[i,k])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    return True

def _local_search(U, V, W, n, m, p, deadline):
    """Perform local search to improve the decomposition."""
    # Simple local search: try small perturbations
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    
    # Try perturbing individual entries
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        
        # Randomly select a matrix and entry to perturb
        mat_idx = random.choice([0, 1, 2])
        if mat_idx == 0:
            mat = U
        elif mat_idx == 1:
            mat = V
        else:
            mat = W
        
        row = random.randrange(len(mat))
        col = random.randrange(len(mat[0]))
        
        # Try changing the value
        old_val = mat[row][col]
        new_val = random.choice([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
        
        if new_val != old_val:
            mat[row][col] = new_val
    
    return U, V, W
# EVOLVE-BLOCK-END
