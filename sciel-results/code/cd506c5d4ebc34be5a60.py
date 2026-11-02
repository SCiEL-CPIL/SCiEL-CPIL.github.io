# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (or at least valid) exact tensor decomposition
    for multiplying a 3x4 matrix by a 4x7 matrix.
    
    Uses Alternating Least Squares to find a low-rank decomposition.
    """
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(nm)]
    V_school = [[0] * rank_schoolbook for _ in range(mp)]
    W_school = [[0] * rank_schoolbook for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j][col] = 1
                V_school[j*p + k][col] = 1
                W_school[k*n + i][col] = 1
                col += 1
    
    # Try to find a lower rank decomposition using ALS
    best_result = None
    best_rank = rank_schoolbook
    
    def round_to_half(x):
        """Round to nearest 0.5 if close, else keep as float."""
        r = round(x * 2) / 2
        if abs(x - r) < 1e-6:
            return r
        return x
    
    # Try ranks from 30 to 83
    for rank in range(30, rank_schoolbook):
        if time.time() > deadline - 15:
            break
        
        # Multiple random initializations
        for seed in range(rank, rank + 5):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(seed)
            U = np.random.randn(nm, rank) * 0.1
            V = np.random.randn(mp, rank) * 0.1
            W = np.random.randn(pn, rank) * 0.1
            
            # ALS iterations
            for it in range(100):
                if time.time() > deadline - 15:
                    break
                
                # Solve for U
                U_new = np.zeros((nm, rank))
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, rank))
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        b = np.ones(p)
                        try:
                            x, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                            U_new[i*m+j, :] = x
                        except:
                            U_new[i*m+j, :] = U[i*m+j, :]
                U = U_new
                
                # Solve for V
                V_new = np.zeros((mp, rank))
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, rank))
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        b = np.ones(n)
                        try:
                            x, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                            V_new[j*p+k, :] = x
                        except:
                            V_new[j*p+k, :] = V[j*p+k, :]
                V = V_new
                
                # Solve for W
                W_new = np.zeros((pn, rank))
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, rank))
                        for j in range(m):
                            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        b = np.ones(m)
                        try:
                            x, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                            W_new[k*n+i, :] = x
                        except:
                            W_new[k*n+i, :] = W[k*n+i, :]
                W = W_new
                
                # Check convergence
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(rank):
                                s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            error += abs(s - 1)
                
                if error < 1e-8:
                    break
            
            # Final verification
            valid = True
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err = abs(s - 1)
                        max_err = max(max_err, err)
                        if err > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to list of lists with rounding to half-integers
                U_list = [[round_to_half(U[i,j]) for j in range(rank)] for i in range(nm)]
                V_list = [[round_to_half(V[i,j]) for j in range(rank)] for i in range(mp)]
                W_list = [[round_to_half(W[i,j]) for j in range(rank)] for i in range(pn)]
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = rank
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    else:
        return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
