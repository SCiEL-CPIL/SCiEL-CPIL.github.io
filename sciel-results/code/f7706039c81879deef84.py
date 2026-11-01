"""Matrix multiplication 2,5,6 tensor decomposition using block Strassen with optimal partition search and numerical refinement."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def _strassen_terms():
    """Return the 7 Strassen terms for 2x2x2 multiplication."""
    return [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
        ([0,0,0,1], [0,0,1,-1], [-1,1,0,0]),
        ([1,1,0,0], [0,0,0,1], [1,0,1,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,-1]),
        ([0,1,0,-1], [0,0,1,1], [-1,0,0,1]),
    ]

def _verify_strassen(terms):
    for i in range(2):
        for j in range(2):
            for k in range(2):
                total = 0
                for U, V, W in terms:
                    total += U[i] * V[j] * W[k]
                if total != 1:
                    return False
    return True

def _generate_partitions(n):
    """Generate all partitions of range(n) into blocks of size 1 or 2."""
    if n == 0:
        yield []
        return
    for rest in _generate_partitions(n - 1):
        yield [[0]] + [[x + 1 for x in block] for block in rest]
    if n >= 2:
        for rest in _generate_partitions(n - 2):
            yield [[0, 1]] + [[x + 2 for x in block] for block in rest]

def _block_cost(j_size, k_size, n):
    """Cost (rank contribution) for a sub-block of size j_size x k_size."""
    if j_size == 2 and k_size == 2:
        return 7
    elif j_size == 2 and k_size == 1:
        return 2
    elif j_size == 1 and k_size == 2:
        return 2
    elif j_size == 1 and k_size == 1:
        return n
    else:
        return j_size * k_size * n

def _find_best_partition(n, m, p, deadline):
    """Find the partition of j and k dimensions that minimizes total rank."""
    best_rank = n * m * p
    best_j_part = None
    best_k_part = None
    
    j_partitions = list(_generate_partitions(m))
    k_partitions = list(_generate_partitions(p))
    
    for jp in j_partitions:
        for kp in k_partitions:
            rank = 0
            for j_block in jp:
                for k_block in kp:
                    rank += _block_cost(len(j_block), len(k_block), n)
            if rank < best_rank:
                best_rank = rank
                best_j_part = jp
                best_k_part = kp
            if time.time() > deadline:
                return best_rank, best_j_part, best_k_part
    
    return best_rank, best_j_part, best_k_part

def _build_block_strassen(n, m, p, j_part, k_part):
    """Build the decomposition using the given partitions."""
    strassen = _strassen_terms()
    all_terms = []
    
    for j_block in j_part:
        for k_block in k_part:
            j_size = len(j_block)
            k_size = len(k_block)
            
            if j_size == 2 and k_size == 2:
                j0, j1 = j_block[0], j_block[1]
                k0, k1 = k_block[0], k_block[1]
                for U_flat, V_flat, W_flat in strassen:
                    U_sparse = {}
                    V_sparse = {}
                    W_sparse = {}
                    for idx, val in enumerate(U_flat):
                        if val != 0:
                            i = idx // 2
                            j_local = idx % 2
                            j = j0 if j_local == 0 else j1
                            U_sparse[i * m + j] = val
                    for idx, val in enumerate(V_flat):
                        if val != 0:
                            j_local = idx // 2
                            k_local = idx % 2
                            j = j0 if j_local == 0 else j1
                            k = k0 if k_local == 0 else k1
                            V_sparse[j * p + k] = val
                    for idx, val in enumerate(W_flat):
                        if val != 0:
                            k_local = idx // 2
                            i = idx % 2
                            k = k0 if k_local == 0 else k1
                            W_sparse[k * n + i] = val
                    all_terms.append((U_sparse, V_sparse, W_sparse))
            elif j_size == 2 and k_size == 1:
                j0, j1 = j_block[0], j_block[1]
                k0 = k_block[0]
                for j in [j0, j1]:
                    U_sparse = {i * m + j: 1 for i in range(n)}
                    V_sparse = {j * p + k0: 1}
                    W_sparse = {k0 * n + i: 1 for i in range(n)}
                    all_terms.append((U_sparse, V_sparse, W_sparse))
            elif j_size == 1 and k_size == 2:
                j0 = j_block[0]
                k0, k1 = k_block[0], k_block[1]
                for k in [k0, k1]:
                    U_sparse = {i * m + j0: 1 for i in range(n)}
                    V_sparse = {j0 * p + k: 1}
                    W_sparse = {k * n + i: 1 for i in range(n)}
                    all_terms.append((U_sparse, V_sparse, W_sparse))
            else:
                j0 = j_block[0]
                k0 = k_block[0]
                U_sparse = {i * m + j0: 1 for i in range(n)}
                V_sparse = {j0 * p + k0: 1}
                W_sparse = {k0 * n + i: 1 for i in range(n)}
                all_terms.append((U_sparse, V_sparse, W_sparse))
    
    return all_terms

def _sparse_to_dense(terms, n, m, p):
    """Convert sparse terms to dense U, V, W matrices."""
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (U_sparse, V_sparse, W_sparse) in enumerate(terms):
        for idx, val in U_sparse.items():
            U[idx][col] = val
        for idx, val in V_sparse.items():
            V[idx][col] = val
        for idx, val in W_sparse.items():
            W[idx][col] = val
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for k in range(p):
            expected = 1  # C[i,k] = sum_j A[i,j]*B[j,k], each term should contribute 1
            # Actually, the tensor T[i,j,k] = 1 for all valid (i,j,k)
            # We check: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # Check all (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(total - 1) > 1e-9:
                    return False
    return True

def _try_numerical_rank(n, m, p, target_rank, deadline, seed=42):
    """Try to find a decomposition of target_rank using iterative linear algebra."""
    if time.time() > deadline:
        return None
    
    np.random.seed(seed)
    
    # Initialize with random half-integer values
    U = np.random.randint(-2, 3, size=(n*m, target_rank)) * 0.5
    V = np.random.randint(-2, 3, size=(m*p, target_rank)) * 0.5
    W = np.random.randint(-2, 3, size=(p*n, target_rank)) * 0.5
    
    # Iterative refinement: fix two, solve for third
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(200):
        if time.time() > deadline - 5:
            break
        
        # Fix V and W, solve for U
        # For each (i,j): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
        # This is linear in U for fixed V,W
        # Actually, let's set up the full system
        
        # Build the constraint matrix
        # Variables: U (n*m*target_rank), V (m*p*target_rank), W (p*n*target_rank)
        # Constraints: for each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        
        # This is nonlinear. Let's use a simpler iterative approach.
        
        # Fix V and W, solve for U via least squares
        # For each (i,j), the constraint involves all k:
        # sum_k [sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]] = m (sum over k of 1)
        # But we need each individual (i,j,k) to equal 1.
        
        # Let's build a linear system for U given V,W:
        # For each (i,j,k): sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1
        # This is: for each (i,j,k), a linear equation in U[i*m+j, :]
        
        # Group by (i,j): for fixed (i,j), we have p equations in target_rank unknowns
        # A[i,j] * U[i*m+j,:] = b[i,j] where A[i,j][k,r] = V[j*p+k,r]*W[k*n+i,r]
        
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # Build A matrix: p x target_rank
                A_mat = np.zeros((p, target_rank))
                b_vec = np.ones(p)
                for k in range(p):
                    for r in range(target_rank):
                        A_mat[k, r] = V[j*p+k, r] * W[k*n+i, r]
                # Solve A_mat @ U[i*m+j, :] = b_vec
                try:
                    sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U_new[i*m+j, :] = sol
                except:
                    U_new[i*m+j, :] = U[i*m+j, :]
        
        U = U_new
        
        # Fix U and W, solve for V
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                A_mat = np.zeros((n, target_rank))
                b_vec = np.ones(n)
                for i in range(n):
                    for r in range(target_rank):
                        A_mat[i, r] = U[i*m+j, r] * W[k*n+i, r]
                try:
                    sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    V_new[j*p+k, :] = sol
                except:
                    V_new[j*p+k, :] = V[j*p+k, :]
        V = V_new
        
        # Fix U and V, solve for W
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                A_mat = np.zeros((m, target_rank))
                b_vec = np.ones(m)
                for j in range(m):
                    for r in range(target_rank):
                        A_mat[j, r] = U[i*m+j, r] * V[j*p+k, r]
                try:
                    sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W_new[k*n+i, :] = sol
                except:
                    W_new[k*n+i, :] = W[k*n+i, :]
        W = W_new
        
        # Compute error
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(target_rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    error += (total - 1) ** 2
        error = np.sqrt(error)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < 1e-8:
            # Snap to half-integers
            best_U = np.round(best_U * 2) / 2
            best_V = np.round(best_V * 2) / 2
            best_W = np.round(best_W * 2) / 2
            # Verify after snapping
            if _verify_decomposition(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p):
                return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def construct():
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    strassen = _strassen_terms()
    if not _verify_strassen(strassen):
        # Fallback to schoolbook
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
    
    # Find best partition
    best_rank, best_j_part, best_k_part = _find_best_partition(n, m, p, DEADLINE - 60)
    
    # If no valid partition found, use schoolbook
    if best_j_part is None or best_k_part is None:
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
    
    # Build decomposition using best partition
    all_terms = _build_block_strassen(n, m, p, best_j_part, best_k_part)
    
    # Check time
    if time.time() - start_time > DEADLINE - 30:
        U, V, W = _sparse_to_dense(all_terms, n, m, p)
        return {"U": U, "V": W, "W": W}
    
    # Try to improve with numerical search for ranks below best_rank
    current_best_rank = len(all_terms)
    current_best_U, current_best_V, current_best_W = _sparse_to_dense(all_terms, n, m, p)
    
    # Try numerical search for lower ranks
    for target_rank in range(current_best_rank - 1, 29, -1):
        if time.time() > DEADLINE - 15:
            break
        result = _try_numerical_rank(n, m, p, target_rank, DEADLINE - 15)
        if result is not None:
            U, V, W = result
            current_best_rank = target_rank
            current_best_U, current_best_V, current_best_W = U, V, W
            break
    
    # Final verification
    if _verify_decomposition(current_best_U, current_best_V, current_best_W, n, m, p):
        return {"U": current_best_U, "V": current_best_V, "W": current_best_W}
    
    # Fallback to block Strassen if numerical didn't verify
    U, V, W = _sparse_to_dense(all_terms, n, m, p)
    return {"U": U, "V": V, "W": W}
