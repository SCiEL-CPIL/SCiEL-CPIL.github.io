# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition correctly computes C = A*B for random inputs."""
    rank = len(U[0])
    rng = np.random.default_rng(42)
    for trial in range(8):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_true = A @ B
        C_comp = np.zeros((n, p))
        for c in range(rank):
            a_form = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    a_form[i] += U[i * m + j][c] * A[i, j]
            b_form = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    b_form[k] += V[j * p + k][c] * B[j, k]
            for i in range(n):
                for k in range(p):
                    C_comp[i, k] += W[k * n + i][c] * a_form[i] * b_form[k]
        if not np.allclose(C_true, C_comp, atol=tol, rtol=tol):
            return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _check_tensor_equation(U, V, W, n, m, p, tol=1e-8):
    """
    Check if the tensor equation holds:
    sum_c W[k*n+i, c] * U[i*m+j, c] * V[l*p+k, c] = delta(j,l)
    for all i, j, k, l.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(m):
                    val = 0.0
                    for c in range(rank):
                        val += W_arr[k * n + i, c] * U_arr[i * m + j, c] * V_arr[l * p + k, c]
                    expected = 1.0 if j == l else 0.0
                    if abs(val - expected) > tol:
                        return False
    return True

def construct():
    start = time.time()
    deadline = start + 110.0  # Leave 10s safety margin
    n, m, p = DIMENSIONS
    
    # Schoolbook is the guaranteed valid solution with rank n*m*p = 63
    candidate = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Verify schoolbook
    if not _verify(candidate["U"], candidate["V"], candidate["W"], n, m, p):
        # Should not happen, but just in case
        return candidate
    
    # Try to find lower rank decompositions using random sparse patterns
    # We will try ranks from 25 up to best_rank - 1
    # We use a wider set of values: {-2, -1, 0, 1, 2}
    # And we increase the number of attempts per rank
    
    for rank in range(25, best_rank):
        if time.time() > deadline:
            break
        
        found_at_this_rank = False
        
        # Try multiple random patterns at this rank
        # Increase attempts to 200 to cover more of the space
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            # Generate random patterns with varying densities
            rng = np.random.default_rng(seed=rank * 1000 + attempt)
            
            # Use integer values {-1, 0, 1} or {-2, -1, 0, 1, 2}
            # Let's mix densities and value sets
            if attempt % 3 == 0:
                density = 0.1
                vals = [-1, 0, 1]
            elif attempt % 3 == 1:
                density = 0.2
                vals = [-1, 0, 1]
            else:
                density = 0.15
                vals = [-2, -1, 0, 1, 2]
                
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Fill U
            for _ in range(int(n*m*rank*density)):
                r = rng.integers(0, n*m)
                c = rng.integers(0, rank)
                U[r, c] = vals[rng.integers(0, len(vals))]
            # Fill V
            for _ in range(int(m*p*rank*density)):
                r = rng.integers(0, m*p)
                c = rng.integers(0, rank)
                V[r, c] = vals[rng.integers(0, len(vals))]
            # Fill W
            for _ in range(int(p*n*rank*density)):
                r = rng.integers(0, p*n)
                c = rng.integers(0, rank)
                W[r, c] = vals[rng.integers(0, len(vals))]
            
            # Check tensor equation
            if _check_tensor_equation(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                # Found valid pattern, verify with actual computation
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                if _verify(U_list, V_list, W_list, n, m, p):
                    new_candidate = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    candidate = new_candidate
                    found_at_this_rank = True
                    break
        
        if found_at_this_rank:
            continue  # Try lower ranks
            
    return candidate
# EVOLVE-BLOCK-END
