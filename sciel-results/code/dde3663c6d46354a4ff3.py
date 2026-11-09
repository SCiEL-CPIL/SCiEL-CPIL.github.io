<<<<<<< SEARCH
def _greedy_sparse_decomp(n, m, p, max_rank=None, deadline=None):
    """
    Greedy sparse basis reduction for matrix multiplication tensor.
    
    We seek rank-r matrices U (n*m x r), V (m*p x r), W (p*n x r) such that
    for all (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    Strategy: Build up the decomposition term by term. Each term is a rank-1
    pattern defined by sparse vectors. We use a greedy approach to minimize rank.
    """
    if max_rank is None:
        max_rank = n * m * p
    
    # Precompute all (i,j,k) coordinates
    coords = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_coords = len(coords)
    
    # Residual: what each coordinate still needs
    residual = np.ones(num_coords)
    
    # Store terms: list of (U_vec, V_vec, W_vec)
    terms = []
    
    # For efficiency, track which coordinates are "satisfied" (residual close to 0)
    # and which still need work
    
    start_time = time.time()
    
    for rank_attempt in range(1, max_rank + 1):
        if deadline and time.time() > deadline:
            break
        
        # Try to build a rank-rank_attempt decomposition
        U = np.zeros((n * m, rank_attempt))
        V = np.zeros((m * p, rank_attempt))
        W = np.zeros((p * n, rank_attempt))
        
        # Current accumulated tensor
        current = np.zeros(num_coords)
        
        for c in range(rank_attempt):
            if deadline and time.time() > deadline:
                break
            
            # Find the best rank-1 pattern to add
            # A rank-1 pattern is (u, v, w) where u[i*m+j]*v[j*p+k]*w[k*n+i] contributes to (i,j,k)
            
            # Heuristic: try to cover as many unsatisfied coordinates as possible
            # with a structured pattern
            
            # For this problem, let's try patterns where u, v, w have structured sparsity
            # Pattern type: u non-zero on a set of (i,j) pairs, v on (j,k) pairs, w on (k,i) pairs
            
            # Simple structured patterns to try:
            # 1. Full row patterns: u non-zero on all j for fixed i, v non-zero on all k for fixed j, w non-zero on all i for fixed k
            # 2. Block patterns
            
            # Let's try a greedy approach: find the pattern that minimizes the
            # remaining residual in a least-squares sense
            
            # For now, use a simple heuristic: try all combinations of "active" indices
            # This is expensive, so we limit it
            
            # Alternative: use random restarts with structured patterns
            
            # Let's try: for each c, pick random subsets and solve for values
            # Actually, let's use a more systematic approach.
            
            # The key insight: if we fix the support (which entries are non-zero),
            # we can solve for the values linearly.
            
            # Try a few random support patterns and check if they work
            found = False
            
            for trial in range(20):  # Try a few random patterns per rank level
                if deadline and time.time() > deadline:
                    break
                
                # Random support pattern
                # u has some non-zero entries, v has some, w has some
                # The product u[i*m+j]*v[j*p+k]*w[k*n+i] is non-zero only if
                # all three are non-zero at the appropriate indices
                
                # Generate random supports
                u_support = set()
                v_support = set()
                w_support = set()
                
                # Randomly select entries
                for idx in range(n * m):
                    if random.random() < 0.3:
                        u_support.add(idx)
                for idx in range(m * p):
                    if random.random() < 0.3:
                        v_support.add(idx)
                for idx in range(p * n):
                    if random.random() < 0.3:
                        w_support.add(idx)
                
                if not u_support or not v_support or not w_support:
                    continue
                
                # Build the contribution matrix: for each (i,j,k), what's the coefficient
                # of the product u[i*m+j]*v[j*p+k]*w[k*n+i]
                # If any of the three is zero (not in support), contribution is 0
                
                # Collect the unknowns: values of u, v, w at their support entries
                # This is a bilinear/trilinear system, not linear!
                
                # For a single rank-1 term, we need:
                # sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = residual[coord]
                # for all (i,j,k)
                
                # This is a system of trilinear equations. Hard to solve directly.
                
                # Alternative approach: fix two of the vectors, solve for the third.
                # Or use iterative methods.
                
                # Let's try a different approach: use the fact that for a rank-1 term,
                # the tensor T[i,j,k] = u[i,j] * v[j,k] * w[k,i] is a rank-1 tensor
                # in the 3-way sense.
                
                # For now, let's just try the schoolbook-like patterns with some merging.
                pass
            
            # Fallback: use a single coordinate pattern (schoolbook-like)
            # Find the coordinate with the largest residual
            max_idx = np.argmax(np.abs(residual))
            i, j, k = coords[max_idx]
            
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
            
            # Update current tensor
            for coord_idx, (ci, cj, ck) in enumerate(coords):
                if ci == i and cj == j and ck == k:
                    current[coord_idx] += 1.0
                    residual[coord_idx] -= 1.0
        
        # Check if we achieved rank rank_attempt
        if _verify(U, V, W, n, m, p):
            return U, V, W
    
    return None
=======
def _greedy_sparse_decomp(n, m, p, max_rank=None, deadline=None):
    """
    Try to find a low-rank decomposition using numerical optimization.
    
    For a given rank r, we have 52*r unknowns and 60 equations.
    We use random initialization and gradient descent to minimize the residual.
    """
    if max_rank is None:
        max_rank = n * m * p
    
    coords = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_coords = len(coords)
    
    # Precompute index mappings for efficiency
    coord_indices = []
    for i, j, k in coords:
        coord_indices.append((i * m + j, j * p + k, k * n + i))
    
    for r in range(1, max_rank + 1):
        if deadline and time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(5):
            if deadline and time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Scale to reasonable magnitude
            U *= 1.0 / np.sqrt(r)
            V *= 1.0 / np.sqrt(r)
            W *= 1.0 / np.sqrt(r)
            
            # Gradient descent
            lr = 0.1
            for step in range(1000):
                if deadline and time.time() > deadline:
                    break
                
                # Compute current tensor values
                residuals = np.zeros(num_coords)
                for idx, (ci, cj, ck) in enumerate(coord_indices):
                    s = 0.0
                    for c in range(r):
                        s += U[ci, c] * V[cj, c] * W[ck, c]
                    residuals[idx] = s - 1.0  # We want sum = 1, so residual = sum - 1
                
                # Check convergence
                if np.max(np.abs(residuals)) < 1e-10:
                    return U, V, W
                
                # Compute gradients
                # d/dU[ci,c] of sum_k (U[ci,c]*V[cj,c]*W[ck,c] - 1)^2 
                # = 2 * residuals[idx] * V[cj,c] * W[ck,c]
                # But each U[ci,c] appears in multiple equations
                
                U_grad = np.zeros_like(U)
                V_grad = np.zeros_like(V)
                W_grad = np.zeros_like(W)
                
                for idx, (ci, cj, ck) in enumerate(coord_indices):
                    res = residuals[idx]
                    for c in range(r):
                        U_grad[ci, c] += 2 * res * V[cj, c] * W[ck, c]
                        V_grad[cj, c] += 2 * res * U[ci, c] * W[ck, c]
                        W_grad[ck, c] += 2 * res * U[ci, c] * V[cj, c]
                
                # Update
                U -= lr * U_grad
                V -= lr * V_grad
                W -= lr * W_grad
                
                # Adaptive learning rate
                if step % 100 == 0 and step > 0:
                    lr *= 0.9
        
        # If we get close, try to polish
        if deadline and time.time() > deadline:
            break
    
    return None
>>>>>>> REPLACE

Actually, gradient descent on a non-convex trilinear problem will likely get stuck in local minima. Let me try a different approach: exploit the specific structure of the 2x5x6 problem.

For n=2, we can think of the multiplication as:
C[0,k] = sum_j A[0,j] * B[j,k]
C[1,k] = sum_j A[1,j] * B[j,k]

Each output entry requires 5 multiplications. Total 12*5 = 60 scalar multiplications.

To reduce rank, we need to find linear combinations. One idea: if we can express some products as combinations of others, we save multiplications.

For example, consider:
A[0,0]*B[0,0] + A[0,1]*B[1,0] + ... + A[0,4]*B[4,0] = C[0,0]
A[1,0]*B[0,0] + A[1,1]*B[1,0] + ... + A[1,4]*B[4,0] = C[1,0]

If we compute S0 = (A[0,0]+A[1,0])*B[0,0], S1 = (A[0,1]+A[1,1])*B[1,0], etc., we get:
S0+S1+...+S4 = C[0,0] + C[1,0]

But we still need 5 multiplications for the sum, and we need to separate C[0,0] and C[1,0], which requires additional work.

This doesn't obviously help. Let me try a computational approach: search for rank-r decompositions using numerical methods with better initialization.

