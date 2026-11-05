"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import random
import math

def _try_rank(n, m, p, target_rank, seed, time_limit=140.0):
    """Try to find a rank target_rank decomposition using structured random initialization."""
    start_time = time.time()
    rng = random.Random(seed)
    
    # For matrix multiplication, we can try a Kronecker product structure
    # C = A * B where A is n x m, B is m x p
    # The standard rank is n*m*p
    # We can potentially reduce by exploiting the structure
    
    # Strategy: Try to find a decomposition of rank r
    # The tensor T[i,j,k] = delta(i,j,k) for matrix mult is actually:
    # T[(i,j),k] contributes to output (i,k)
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 if we're multiplying, else 0
    # This is a 3-way tensor with dimensions (n*m) x m x p? No.
    
    # Let me reconsider the tensor structure.
    # For C = A * B:
    # C[i,k] = sum_{j=1}^{m} A[i,j] * B[j,k]
    # The matrix multiplication tensor T has entries T[i,j,k] = 1
    # and C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    
    # So the tensor is of size (n*m) x (m*p) x (p*n) in the standard format
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    # The tensor entry T[(i,j),(j,k),(k,i)] = 1 for valid (i,j,k)
    # Wait, the output index is (k*n + i) per the problem statement: W[k*n+i, r]
    
    # So for each (i,j,k) triple where i in [0,n), j in [0,m), k in [0,p):
    # U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] should sum to 1 for the (i,j,k) contribution
    # And be 0 for invalid combinations.
    
    # Actually, the tensor is: T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some valid (i,j,k)
    # This means the tensor is a "permutation-like" tensor where each valid (i,j,k) gives one 1.
    
    # The rank of this tensor is at least max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # And at most n*m*p = 96
    
    # Let's try to find a decomposition with rank close to the lower bound.
    
    # One approach: use the fact that this is related to matrix multiplication
    # The rank of the matrix multiplication tensor MM_{n,m,p} is a well-studied problem.
    
    # For small dimensions, we can try random search.
    
    # Initialize U, V, W randomly
    U = [[rng.uniform(-1, 1) for _ in range(target_rank)] for _ in range(n * m)]
    V = [[rng.uniform(-1, 1) for _ in range(target_rank)] for _ in range(m * p)]
    W = [[rng.uniform(-1, 1) for _ in range(target_rank)] for _ in range(p * n)]
    
    # For each (i,j,k), we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for invalid (i,j,k) combinations, the sum should be 0.
    
    # Actually, let me reconsider. The tensor for matrix multiplication is:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # And the product is: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For this to equal sum_j A[i,j]*B[j,k], we need the decomposition to be exact.
    
    # The constraint is that for each (i,k):
    # sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = A[i,j] * B[j,k] summed over j
    
    # This is a complex constraint. Let me try a different approach.
    
    # Use the known fact that for matrix multiplication, the rank is bounded
    # by the rank of the corresponding matrix.
    
    # For now, let's just return the schoolbook decomposition if we can't find better.
    
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        return None
    
    # Try a few iterations of gradient descent
    for iteration in range(100):
        if time.time() - start_time > time_limit:
            break
        
        # Compute the residual
        # For each (i,k), compute the predicted C[i,k] and compare to the "true" value
        # The "true" value is the matrix product, but since we're decomposing the tensor,
        # we need to check that the decomposition is valid.
        
        # A simpler check: for each valid (i,j,k), the contribution should be 1
        # and for invalid, 0. But the tensor is defined such that T[i,j,k] = 1 for all valid (i,j,k).
        
        # Let's just do a random restart and return the best we find.
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best = {"U": U, "V": V, "W": W, "rank": schoolbook_rank}
    
    # Try to find a better decomposition
    start_time = time.time()
    deadline = start_time + 130  # Leave 10s margin
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound is max(n*m, m*p, p*n) = 32
    lower_bound = max(n * m, m * p, p * n)
    
    for target_rank in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline:
            break
        
        found = _try_rank(n, m, p, target_rank, seed=target_rank, time_limit=min(5.0, deadline - time.time()))
        if found is not None:
            best = found
            break
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
