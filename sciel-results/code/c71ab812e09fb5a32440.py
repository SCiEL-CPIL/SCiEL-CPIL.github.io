"""Program for 3x3x8 matrix multiplication tensor decomposition."""

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Start with the schoolbook decomposition (rank 72)
    # This gives a valid baseline that we can then compress via tensor rank reduction
    
    rank_school = n * m * p  # 72
    
    # Build the matrix multiplication tensor in 3D form
    # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to C[i,k]
    # In our convention:
    #   U has rows indexed by (i,j) in [0, n*m), columns by rank
    #   V has rows indexed by (j,k) in [0, m*p), columns by rank  
    #   W has rows indexed by (k,i) in [0, p*n), columns by rank
    # The decomposition satisfies: for all i,j,k:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 (always, for valid MM)
    
    # The tensor is: T_{i,j,k} where the contraction is over the middle index j
    # T is a 3-tensor of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a's (i,j), b's (j,k), c's (k,i) form a valid MM term
    
    # Let's think of this differently. The matrix multiplication tensor for
    # multiplying (n x m) by (m x p) to get (n x p) has entries indexed by
    # (i, j, k) for i in range(n), j in range(m), k in range(p).
    # The tensor value at (i,j,k) is 1 (it's a 0-1 tensor).
    # 
    # In the standard decomposition framework:
    #   U rows indexed by (i,j): n*m = 9 rows
    #   V rows indexed by (j,k): m*p = 24 rows  
    #   W rows indexed by (k,i): p*n = 24 rows
    # 
    # The tensor T has T[i,j,k] = 1 for all valid (i,j,k)
    # The decomposition: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = T[i,j,k]
    # Since T[i,j,k] = 1 for all i,j,k (in MM, every term contributes),
    # we need: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k
    
    # Actually, this isn't quite right. Let me reconsider.
    # The matrix multiplication tensor is:
    #   MM = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}  (in appropriate index spaces)
    # 
    # In the standard notation for the tensor decomposition problem:
    # We want to decompose the tensor T of shape (n*m, m*p, p*n) where
    # T[a, b, c] = 1 iff a = i*m+j, b = j'*p+k, c = k'*n+i' with j=j' and k=k' and i=i'
    # Wait, that's not right either.
    
    # Let me use the standard formulation. The matrix multiplication tensor for
    # (n x m) @ (m x p) is a 3-tensor with modes of sizes n*m, m*p, and p*n.
    # 
    # Mode 1 (size n*m=9): indexed by (i,j), i=0..n-1, j=0..m-1
    # Mode 2 (size m*p=24): indexed by (j,k), j=0..m-1, k=0..p-1  
    # Mode 3 (size p*n=24): indexed by (k,i), k=0..p-1, i=0..n-1
    # 
    # T[a, b, c] = 1 if the (i,j) from a, (j',k) from b, (k',i') from c satisfy
    # j = j' and k = k' and i = i'. Otherwise T[a,b,c] = 0.
    # 
    # Since for each (i,j,k) there's exactly one valid (a,b,c) = (i*m+j, j*p+k, k*n+i),
    # and T is 1 at those points and 0 elsewhere.
    # 
    # Actually for MM, T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise.
    # There are n*m*p = 72 such non-zero entries.
    
    # Build the full tensor
    T = [[[0.0] * (p * n) for _ in range(m * p)] for _ in range(n * m)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a][b][c] = 1.0
    
    # Now try to find a low-rank decomposition using tensor rank algorithms.
    # We'll use a randomized approach: try random rank-r decompositions and check
    # if they can be fitted, or use the Wiedemann-like algorithm.
    
    # Alternative: use the known result that for small matrices, we can sometimes
    # find decompositions via structured approaches.
    
    # For now, let's implement a basic rank reduction via:
    # 1. Start with schoolbook (rank 72)
    # 2. Try to merge terms using the structure
    
    # The key insight: in MM, the tensor has a very specific structure.
    # For 3x3x8, we can potentially exploit the small size.
    
    # Let's try a different approach: use the fact that the tensor is 0-1 and
    # try to find a decomposition with half-integer coefficients.
    
    # Actually, let me implement a proper tensor rank search.
    # We'll use the approach of trying random rank-r decompositions and checking
    # if the residual can be made zero.
    
    import numpy as np
    import random
    
    random.seed(42)
    np.random.seed(42)
    
    # Convert T to numpy
    T_np = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_np[a, b, c] = 1.0
    
    # Try ranks from low to high
    # For each rank r, try to find a decomposition
    # We use a simple iterative approach: randomly initialize and refine
    
    best_rank = rank_school
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook solution
    U_sb = np.zeros((n * m, rank_school))
    V_sb = np.zeros((m * p, rank_school))
    W_sb = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a, col] = 1.0
                V_sb[b, col] = 1.0
                W_sb[c, col] = 1.0
                col += 1
    
    # Store schoolbook as baseline
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # Use a simple approach: try random initializations and do gradient descent
    # or use the tensor train decomposition idea
    
    # For small tensors, we can try exhaustive or semi-exhaustive search
    # over structured decompositions.
    
    # Approach: try rank r from 40 to 71, and for each, try multiple random
    # initializations and do a few steps of optimization.
    
    # Actually, for a proper implementation within the time limit, let's use
    # a more structured approach based on the geometry.
    
    # Key observation: The MM tensor for (n x m) @ (m x p) can be viewed as
    # a sum over j of terms that couple (i, j) in A with (j, k) in B and (k, i) in C.
    # 
    # For each fixed j, we have n*p = 24 terms. The "j-slice" is a rank-1
    # (in a certain sense) structure.
    
    # Let's try a block-structured decomposition.
    # Group the 72 terms by j (3 groups of 24).
    
    # Alternative: use the fact that for small sizes, known decompositions exist.
    # For 3x3x8, the minimal rank is believed to be around 50-60.
    
    # Let me implement a basic randomized tensor rank search.
    
    def try_rank(r, num_trials=50):
        """Try to find a rank-r decomposition."""
        for trial in range(num_trials):
            # Random initialization with small values
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            # Do a few iterations of alternating least squares
            # or simply check if we can solve the linear system
            
            # The condition is: for all (a, b, c), 
            # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
            
            # This is equivalent to: the tensor T should be in the span of
            # rank-1 tensors formed by the columns of U, V, W.
            
            # A simpler check: build the "flattened" system
            # For each (a,b,c) with T[a,b,c] != 0, we need the sum to equal 1
            # For each (a,b,c) with T[a,b,c] == 0, we need the sum to equal 0
            
            # This gives (n*m)*(m*p)*(p*n) equations, but many are redundant.
            # The independent constraints are at most rank^2 * something.
            
            # Let's use a different approach: check the tensor norm
            # The residual tensor R = T - sum_r u_r ⊗ v_r ⊗ w_r
            # We want ||R||_F = 0
            
            # For a quick check, sample some entries
            residual = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = 0.0
                        for r in range(r):
                            val += U[a, r] * V[b, r] * W[c, r]
                        if abs(val - 1.0) > 1e-6:
                            residual += (val - 1.0) ** 2
                        # Check a few zero entries
                        if residual > 0.1:
                            break
                if residual > 0.1:
                    break
            if residual > 0.1:
                continue
            
            # Check some zero entries
            for _ in range(200):
                a = random.randint(0, n * m - 1)
                b = random.randint(0, m * p - 1)
                c = random.randint(0, p * n - 1)
                # Skip if this is a non-zero entry
                found = False
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if i * m + j == a and j * p + k == b and k * n + i == c:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    continue
                val = 0.0
                for r in range(r):
                    val += U[a, r] * V[b, r] * W[c, r]
                if abs(val) > 1e-6:
                    break
            else:
                # Possible valid decomposition
                return U, V, W, True
        
        return None, None, None, False
    
    # Try ranks from 50 to 71
    for r in range(50, 72):
        U, V, W, success = try_rank(r, num_trials=10)
        if success:
            best_rank = r
            best_U = U
            best_V = V
            best_W = W
            break
    
    # Convert to list of lists with rounded values
    def to_list(mat):
        return [[round(float(x), 10) for x in row] for row in mat]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
