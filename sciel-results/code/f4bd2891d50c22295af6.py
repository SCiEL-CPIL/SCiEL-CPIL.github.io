"""Matrix multiplication 2x3x4 rank-minimal exact tensor decomposition search."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 4]

# Deadline guard: evaluator kills at 150s; use 140s as hard stop with safety margin.
DEADLINE = 140.0
_START_TIME = time.time()


def _time_left():
    return DEADLINE - (time.time() - _START_TIME)


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3x4 matrix multiplication.
    
    The tensor T[i,j,k] (where i in {0,1}, j in {0,1,2}, k in {0,1,2,3}) is the
    matrix multiplication tensor: T[i,j,k] = 1 for all valid (i,j,k) in the
    2x3x4 grid. We seek U (6 x r), V (12 x r), W (8 x r) such that
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k).
    
    We use a randomized search with local refinement to find low-rank decompositions.
    """
    n, m, p = DIMENSIONS
    # Indices: U rows = i*m + j (6 rows), V rows = j*p + k (12 rows), W rows = k*n + i (8 rows)
    
    # The target tensor has all ones: for each (i,j,k), the product sum must equal 1.
    # We can think of this as: for each c, U_c * V_c * W_c (outer product) contributes,
    # and we need sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] = 1.
    
    # Start with the schoolbook decomposition (rank 24) as fallback.
    rank_schoolbook = n * m * p  # 24
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    # Try to find lower rank decompositions using randomized algebraic methods.
    # Key insight: we can use the fact that the tensor is structured.
    # A rank-r decomposition means: T = sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c
    # where u_c is in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}.
    
    # Strategy: Use a greedy/local search approach.
    # 1. Try small ranks first (e.g., 8, 9, 10, ...)
    # 2. For each rank r, try random initializations and refine via alternating least squares
    #    or direct solving.
    
    # Actually, for exact decomposition with integer/half-integer entries, we need
    # rational solutions. Let's use a different approach:
    
    # The matrix multiplication tensor for 2x3x4 has a known structure.
    # Let's try to find a decomposition by exploiting the separability.
    
    # Approach: We can write the multiplication as:
    # (A * B)[i,k] = sum_j A[i,j] * B[j,k]
    # This is a bilinear form. A rank-r decomposition corresponds to r bilinear forms.
    
    # Let's try a systematic search over small ranks using randomized methods.
    
    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(U.shape[1]):
                        total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    def try_rank(r, max_iters=200):
        """Try to find a rank-r decomposition using randomized local search."""
        if _time_left() < 5:
            return None
        
        n, m, p = DIMENSIONS
        U_dim = n * m  # 6
        V_dim = m * p  # 12
        W_dim = p * n  # 8
        
        # We need: for all (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # This is a system of 24 equations with 3*r*r variables (but with structure).
        
        # Use a randomized approach: initialize with random values, then refine.
        # The constraint is that the tensor of products must equal the all-ones tensor.
        
        # Let's use a simpler approach: try to find a decomposition where
        # each "fiber" is separable.
        
        # Actually, let's use a known result: for m x n x p matrix multiplication,
        # the rank is at most m*n*p (schoolbook) and at least max(m*n, m*p, n*p).
        # For 2x3x4: lower bound = max(6, 12, 8) = 12.
        
        # Let's try ranks from 12 upward.
        
        # Use a greedy algorithm:
        # 1. Start with a random rank-r decomposition.
        # 2. Compute the error tensor.
        # 3. Try to fix the error by adjusting the decomposition.
        
        # Alternative: use the fact that we can write the multiplication as
        # a sum of rank-1 tensors.
        
        # Let's try a direct approach: for each rank r, set up the equations
        # and try to solve them.
        
        # For rank r, we have variables U (6 x r), V (12 x r), W (8 x r).
        # The constraint is: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1.
        # This is a system of 24 polynomial equations in 3*r*(6+12+8)/3 = 3*r*26/3... 
        # Actually, 6*r + 12*r + 8*r = 26*r variables, and 24 equations.
        
        # For r >= 12, we have more variables than equations, so solutions may exist.
        
        # Let's use a randomized approach:
        # 1. Initialize U, V, W with random small values.
        # 2. Use gradient descent or Newton's method to minimize the error.
        
        # But we need exact solutions with integer/half-integer entries.
        # So let's try a different approach: search over rational values.
        
        # For simplicity, let's try a few specific constructions.
        
        # Construction 1: Use the fact that 2x3x4 can be decomposed using
        # the 2x3 and 3x4 multiplications separately.
        
        # Actually, let's just try random initializations and check if they work.
        
        np.random.seed(42)
        
        for attempt in range(max_iters):
            if _time_left() < 5:
                break
            
            # Initialize with random values in {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(U_dim, r))
            V = np.random.randint(-1, 2, size=(V_dim, r))
            W = np.random.randint(-1, 2, size=(W_dim, r))
            
            # Check if this works
            if verify(U, V, W):
                return U, V, W
        
        # If random didn't work, try a structured approach.
        # For rank r, we can try to make the decomposition "block diagonal"
        # or use a Kronecker product structure.
        
        # Let's try a different initialization: use the schoolbook decomposition
        # and try to merge columns.
        
        return None
    
    # Try ranks from the lower bound upward.
    # Lower bound for 2x3x4 matrix multiplication rank:
    # The rank is at least max(n*m, m*p, n*p) = max(6, 12, 8) = 12.
    
    for r in range(12, best_rank):
        if _time_left() < 10:
            break
        
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            break
    
    # Convert to lists of integers/half-integers
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
