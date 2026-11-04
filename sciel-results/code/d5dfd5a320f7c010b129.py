"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x8 matrix.
    Returns U, V, W factor matrices with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # The matrix multiplication tensor T has indices (i, j, k) -> T[i*m+j, j*p+k, k*n+i] = 1
    # We want to decompose T as sum of rank-1 tensors: T = sum_r U[:,r] V[:,r] W[:,r]
    # where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    
    # Schoolbook rank is n*m*p = 72. We try to find lower rank decompositions.
    
    # For matrix multiplication A (n x m) * B (m x p) = C (n x p),
    # the tensor is T[i,j,k] = delta such that C[i,k] = sum_j A[i,j] B[j,k]
    
    # Let's use the standard schoolbook decomposition as baseline (rank = n*m*p = 72)
    # Then try to find improvements.
    
    # First, let's verify: for 3x3 * 3x8, the standard rank is 3*3*8 = 72.
    # But there might be lower rank decompositions.
    
    # Known results: For n x m * m x p, the minimal rank is at most n*m*p (schoolbook).
    # For small cases, we can try exhaustive or semi-exhaustive search.
    
    # Let's try a randomized approach with local search to find lower rank decompositions.
    
    def schoolbook_decomposition():
        """Standard schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    # Get schoolbook as baseline
    U_base, V_base, W_base, rank_base = schoolbook_decomposition()
    best_rank = rank_base
    best_U, best_V, best_W = U_base.copy(), V_base.copy(), W_base.copy()
    
    # Try to find lower rank decompositions using randomized initialization and local search
    # The constraint is that the decomposition must be exact (entries are 0, 1, or possibly other values)
    # and entries must be integer or half-integer.
    
    # For exact matrix multiplication tensor, we need:
    # For all (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if this is a valid term, else 0
    
    # Actually, the tensor T[i,j,k] = 1 for all i,j,k in the appropriate ranges.
    # Wait, let me reconsider. The matrix multiplication tensor is defined as:
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k, else 0
    # And we need T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    
    # Let's try small rank decompositions using random search
    
    def check_decomposition(U, V, W, rank):
        """Check if the decomposition is exact."""
        # Compute the tensor from the decomposition
        # T[i*m+j, j*p+k, k*n+i] should be 1 for all valid (i,j,k)
        # and 0 otherwise
        
        # Build the full tensor
        tensor = np.zeros((n * m, m * p, p * n))
        for r in range(rank):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            # Outer product
            for a in range(n * m):
                if u[a] != 0:
                    for b in range(m * p):
                        if v[b] != 0:
                            for c in range(p * n):
                                if w[c] != 0:
                                    tensor[a, b, c] += u[a] * v[b] * w[c]
        
        # Check against the true tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    if abs(tensor[a, b, c] - 1.0) > 1e-9:
                        return False
        # Check that all other entries are 0
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    # This entry should be 0 unless (i,j,k) is consistent
                    # i.e., i == i2, j == j2, k == k2
                    if not (i == i2 and j == j2 and k == k2):
                        if abs(tensor[a, b, c]) > 1e-9:
                            return False
        return True
    
    # Try randomized search for lower rank
    # Start with rank = rank_base - 1 and go down
    
    def random_decomposition(rank, rng):
        """Generate a random decomposition with integer/half-integer entries."""
        # Use small integer values: -1, 0, 1, or half-integers
        values = [-1, -0.5, 0, 0.5, 1]
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for r in range(rank):
            U[:, r] = rng.choice(values, size=n * m)
            V[:, r] = rng.choice(values, size=m * p)
            W[:, r] = rng.choice(values, size=p * n)
        return U, V, W
    
    # Try a few random searches for lower rank
    rng = np.random.RandomState(42)
    
    # For efficiency, let's try ranks from rank_base - 1 down to some minimum
    # but with limited attempts due to time constraints
    
    # Actually, for this problem size, exhaustive search over small ranks might be feasible
    # if we're smart about it. Let's try a structured approach.
    
    # The key insight: for matrix multiplication, the tensor has a specific structure.
    # Let's try to use the fact that we can combine terms.
    
    # For now, let's do a limited random search
    max_attempts_per_rank = 100
    
    for rank in range(rank_base - 1, max(10, rank_base - 20), -1):
        if time.time() > deadline - 5:
            break
        
        found = False
        for attempt in range(max_attempts_per_rank):
            if time.time() > deadline - 5:
                break
            
            U, V, W = random_decomposition(rank, rng)
            
            # Quick check: compute the tensor and verify
            tensor = np.zeros((n * m, m * p, p * n))
            for r in range(rank):
                u = U[:, r]
                v = V[:, r]
                w = W[:, r]
                # Use outer product efficiently
                outer_uv = np.outer(u, v)
                for c in range(p * n):
                    if w[c] != 0:
                        tensor[:, :, c] += outer_uv * w[c]
            
            # Check
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        if abs(tensor[a, b, c] - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Check zeros
            for a in range(n * m):
                for b in range(m * p):
                    for c in range(p * n):
                        i = a // m
                        j = a % m
                        j2 = b // p
                        k = b % p
                        k2 = c // n
                        i2 = c % n
                        if not (i == i2 and j == j2 and k == k2):
                            if abs(tensor[a, b, c]) > 1e-9:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Found a valid decomposition!
            best_rank = rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            found = True
            break
        
        if found:
            # Try to go even lower
            continue
    
    # If we found a better decomposition, use it; otherwise use schoolbook
    if best_rank < rank_base:
        U_out = best_U
        V_out = best_V
        W_out = best_W
        final_rank = best_rank
    else:
        U_out = U_base
        V_out = V_base
        W_out = W_base
        final_rank = rank_base
    
    # Convert to lists of lists with integer/half-integer values
    def to_list(mat):
        return [[mat[i][j] for j in range(mat.shape[1])] for i in range(mat.shape[0])]
    
    return {"U": to_list(U_out), "V": to_list(V_out), "W": to_list(W_out)}
# EVOLVE-BLOCK-END
