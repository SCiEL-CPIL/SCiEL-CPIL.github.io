"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]
import time
import random
import itertools
import numpy as np

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For each (i, j, k), the coefficient should be 1
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _check_half_integer(U, V, W, tol=1e-9):
    """Check if all entries are integer or half-integer."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                # Check if val is close to an integer or half-integer
                if abs(val - round(val)) > tol and abs(val - (round(2*val)/2.0)) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 45
    
    # Start with schoolbook
    best_rank = rank_schoolbook
    best_U = [[0] * rank_schoolbook for _ in range(n * m)]
    best_V = [[0] * rank_schoolbook for _ in range(m * p)]
    best_W = [[0] * rank_schoolbook for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
    
    # Try to find lower rank decompositions
    # Known results: rank(3x3x3) = 21 (Landsberg), rank(3x3x5) is between 33 and 45
    # Let's try to find a decomposition with rank < 45
    
    # Strategy: Use a structured approach based on the 3x3x3 decomposition
    # The 3x3x3 tensor can be decomposed with rank 21 (Strassen-like)
    # For 3x3x5, we can think of it as 5 slices of 3x3x1
    
    # Let's try a different approach: use a basis of matrices and find a low-rank decomposition
    # of the multiplication tensor
    
    # The multiplication tensor T has entries T[i,j,k] = 1 if i,j,k correspond to A[i,j]*B[j,k]=C[i,k]
    # Actually, the tensor is T[i,j,k] where the product is C[i,k] = sum_j A[i,j]B[j,k]
    # So the tensor has indices (i, j, k) with i in [0,n), j in [0,m), k in [0,p)
    # And T[i,j,k] = 1 (the coefficient of A[i,j]*B[j,k] in C[i,k])
    
    # We need to find U, V, W such that for all i, j, k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of nonlinear equations. Let's try a heuristic search.
    
    # Approach: Start with the schoolbook and try to merge columns
    # Two columns c1 and c2 can be merged if there's a linear transformation that
    # preserves the tensor structure.
    
    # Alternative: Use a random initialization and optimize
    # This is complex, so let's try a simpler approach first.
    
    # For now, let's try to find if rank 44 or lower is possible by a simple heuristic
    # Try removing one column from schoolbook and see if we can adjust others
    
    # Actually, let's just return the schoolbook for now as a safe baseline
    # and add a time-limited search for improvements
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try a structured decomposition based on known 3x3x3 rank-21 result
    # The 3x3x3 matrix multiplication can be done in rank 21 (not 27)
    # For 3x3x5, we can potentially use a similar structure
    
    # Let's try to construct a decomposition with rank < 45
    # This is a hard problem, so we'll use a heuristic
    
    # Heuristic: Try to find a set of rank-1 tensors that sum to the multiplication tensor
    # Each rank-1 tensor is u_i * v_j * w_k
    
    # For a 3x3x5 tensor, we have 3*3*5 = 45 entries
    # We need to find rank r such that the tensor can be written as sum of r rank-1 tensors
    
    # Let's try a random search for small improvements
    # This is not guaranteed to work, but it's better than nothing
    
    # Actually, let's just return the schoolbook decomposition
    # Finding a lower rank decomposition is a research-level problem
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
