<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    schoolbook_rank = n * m * p
    lower_bound = max(nm, mp, pn)
    
    # Start with schoolbook as fallback
    fallback = _make_schoolbook(n, m, p)
    best_result = fallback
    best_rank = schoolbook_rank
    
    rng = random.Random(42)
    
    # Try to find lower rank decompositions
    # Search from lower bound upward
    search_start = lower_bound
    search_end = min(schoolbook_rank, lower_bound + 20)  # Limit search space
    
    for rank in range(search_start, search_end + 1):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        # Try random search for this rank
        result = _try_random_decomposition(n, m, p, rank, rng, max_iters=1000)
        if result is not None:
            best_rank = rank
            best_result = result
            break
    
    # Try to improve with greedy column removal
    if best_rank < schoolbook_rank:
        improved = _greedy_improve(
            [row[:] for row in best_result["U"]],
            [row[:] for row in best_result["V"]],
            [row[:] for row in best_result["W"]],
            n, m, p
        )
        imp_rank = _rank_of(improved["U"], improved["V"], improved["W"])
        if imp_rank < best_rank:
            best_rank = imp_rank
            best_result = improved
    
    return best_result

def _make_schoolbook(n, m, p):
    """Create the schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank(n, m, p, rank, rng, max_iters=200):
    """Try to find a rank-r decomposition using random search with structured initialization."""
    nm, mp, pn = n*m, m*p, p*n
    
    for iteration in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            return None
        
        # Strategy 1: Use structured patterns based on matrix multiplication structure
        # For each (i,j,k), we need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
        
        # Try different initialization strategies
        strategy = iteration % 3
        
        if strategy == 0:
            # Random sparse initialization
            vals = [-1, -0.5, 0, 0.5, 1]
            U = [[rng.choice(vals) for _ in range(rank)] for _ in range(nm)]
            V = [[rng.choice(vals) for _ in range(rank)] for _ in range(mp)]
            W = [[rng.choice(vals) for _ in range(rank)] for _ in range(pn)]
        elif strategy == 1:
            # Structured: assign each (i,j,k) to a specific column
            U = [[0]*rank for _ in range(nm)]
            V = [[0]*rank for _ in range(mp)]
            W = [[0]*rank for _ in range(pn)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < rank:
                            U[i*m+j][col] = 1
                            V[j*p+k][col] = 1
                            W[k*n+i][col] = 1
                            col += 1
            # Add some random perturbations
            for c in range(rank):
                if rng.random() < 0.3:
                    for idx in range(nm):
                        if rng.random() < 0.2:
                            U[idx][c] += rng.choice([-0.5, 0.5])
                    for idx in range(mp):
                        if rng.random() < 0.2:
                            V[idx][c] += rng.choice([-0.5, 0.5])
                    for idx in range(pn):
                        if rng.random() < 0.2:
                            W[idx][c] += rng.choice([-0.5, 0.5])
        else:
            # Block-structured: exploit p=6 factorization
            # Split k dimension into two blocks of 3
            U = [[0]*rank for _ in range(nm)]
            V = [[0]*rank for _ in range(mp)]
            W = [[0]*rank for _ in range(pn)]
            
            # For each (i,j), assign columns for k=0,1,2 and k=3,4,5
            col = 0
            for i in range(n):
                for j in range(m):
                    # Block 1: k=0,1,2
                    if col < rank:
                        for k in range(3):
                            U[i*m+j][col] = 1
                            V[j*p+k][col] = 1
                            W[k*n+i][col] = 1
                        col += 1
                    # Block 2: k=3,4,5
                    if col < rank:
                        for k in range(3, 6):
                            U[i*m+j][col] = 1
                            V[j*p+k][col] = 1
                            W[k*n+i][col] = 1
                        col += 1
            
            # Add random structure
            for c in range(rank):
                if rng.random() < 0.4:
                    for idx in range(nm):
                        if rng.random() < 0.3:
                            U[idx][c] += rng.choice([-1, -0.5, 0.5, 1])
                    for idx in range(mp):
                        if rng.random() < 0.3:
                            V[idx][c] += rng.choice([-1, -0.5, 0.5, 1])
                    for idx in range(pn):
                        if rng.random() < 0.3:
                            W[idx][c] += rng.choice([-1, -0.5, 0.5, 1])
        
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def _schoolbook(n, m, p):
    """Create the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _structured_search(n, m, p, start_rank, end_rank, rng):
    """Search for minimum rank using structured approaches."""
    best_result = None
    best_rank = float('inf')
    
    for rank in range(start_rank, end_rank + 1):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        result = _try_rank(n, m, p, rank, rng, max_iters=300)
        if result is not None:
            best_rank = rank
            best_result = result
            break
    
    return best_result, best_rank

def _gradient_descent(U, V, W, n, m, p, max_iters=1000, lr=0.1):
    """Try to minimize the error using gradient descent, then round to half-integers."""
    rank = len(U[0])
    nm, mp, pn = n*m, m*p, p*n
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    def error():
        total_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    total_err += (val - 1.0) ** 2
        return total_err
    
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        if error() < 1e-12:
            break
        
        # Compute gradients
        U_grad = np.zeros_like(U_arr)
        V_grad = np.zeros_like(V_arr)
        W_grad = np.zeros_like(W_arr)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    err = val - 1.0
                    for c in range(rank):
                        U_grad[i*m+j, c] += 2 * err * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                        V_grad[j*p+k, c] += 2 * err * U_arr[i*m+j, c] * W_arr[k*n+i, c]
                        W_grad[k*n+i, c] += 2 * err * U_arr[i*m+j, c] * V_arr[j*p+k, c]
        
        # Update
        U_arr -= lr * U_grad
        V_arr -= lr * V_grad
        W_arr -= lr * W_grad
    
    # Round to nearest half-integer
    U_rounded = [[round(x * 2) / 2 for x in row] for row in U_arr]
    V_rounded = [[round(x * 2) / 2 for x in row] for row in V_arr]
    W_rounded = [[round(x * 2) / 2 for x in row] for row in W_arr]
    
    if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
        return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    schoolbook_rank = n * m * p
    lower_bound = max(nm, mp, pn)
    
    # Start with schoolbook as fallback
    fallback = _schoolbook(n, m, p)
    best_result = fallback
    best_rank = schoolbook_rank
    
    rng = random.Random(42)
    
    # Try structured search from lower bound
    search_start = lower_bound
    search_end = min(schoolbook_rank, lower_bound + 25)
    
    result, rank = _structured_search(n, m, p, search_start, search_end, rng)
    if result is not None:
        best_rank = rank
        best_result = result
    
    # Try gradient descent refinement if we found something
    if best_rank < schoolbook_rank:
        gd_result = _gradient_descent(
            [row[:] for row in best_result["U"]],
            [row[:] for row in best_result["V"]],
            [row[:] for row in best_result["W"]],
            n, m, p
        )
        if gd_result is not None:
            best_result = gd_result
            best_rank = len(gd_result["U"][0])
    
    return best_result
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
