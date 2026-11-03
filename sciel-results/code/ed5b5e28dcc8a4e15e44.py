"""Rewritten program: systematic merge search + random restarts to find low-rank decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _max_err(U, V, W, n, m, p):
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    max_e = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                e = abs(s - 1.0)
                if e > max_e:
                    max_e = e
    return max_e

def _strassen_2x2x2(ou, ov, ow, U, V, W, c):
    U[ou+0][c]=1; U[ou+3][c]=1; V[ov+0][c]=1; V[ov+3][c]=1; W[ow+0][c]=1; W[ow+3][c]=1; c+=1
    U[ou+2][c]=1; U[ou+3][c]=1; V[ov+0][c]=1; W[ow+2][c]=1; W[ow+3][c]=-1; c+=1
    U[ou+0][c]=1; V[ov+1][c]=1; V[ov+3][c]=-1; W[ow+0][c]=1; W[ow+3][c]=1; c+=1
    U[ou+3][c]=1; V[ov+2][c]=1; V[ov+0][c]=-1; W[ow+0][c]=1; W[ow+2][c]=1; c+=1
    U[ou+0][c]=1; U[ou+1][c]=1; V[ov+3][c]=1; W[ow+0][c]=-1; W[ow+1][c]=1; c+=1
    U[ou+2][c]=1; U[ou+0][c]=-1; V[ov+0][c]=1; V[ov+1][c]=1; W[ow+3][c]=1; c+=1
    U[ou+1][c]=1; U[ou+3][c]=-1; V[ov+2][c]=1; V[ov+3][c]=1; W[ow+0][c]=1; c+=1
    return c

def _construct_hybrid_rank36():
    n, m, p = 2, 4, 5
    R = 36
    U = [[0.0]*R for _ in range(n*m)]
    V = [[0.0]*R for _ in range(m*p)]
    W = [[0.0]*R for _ in range(p*n)]
    c = 0
    c = _strassen_2x2x2(0, 0, 0, U, V, W, c)
    U[0][c]=1; V[2][c]=1; W[4][c]=1; c+=1
    U[1][c]=1; V[3][c]=1; W[4][c]=1; c+=1
    U[2][c]=1; V[2][c]=1; W[5][c]=1; c+=1
    U[3][c]=1; V[3][c]=1; W[5][c]=1; c+=1
    c = _strassen_2x2x2(0, 6, 6, U, V, W, c)
    c = _strassen_2x2x2(4, 10, 0, U, V, W, c)
    U[4][c]=1; V[12][c]=1; W[4][c]=1; c+=1
    U[5][c]=1; V[13][c]=1; W[4][c]=1; c+=1
    U[6][c]=1; V[12][c]=1; W[5][c]=1; c+=1
    U[7][c]=1; V[13][c]=1; W[5][c]=1; c+=1
    c = _strassen_2x2x2(4, 16, 6, U, V, W, c)
    return {"U": U, "V": V, "W": W}

def _construct_schoolbook():
    n, m, p = 2, 4, 5
    R = n*m*p
    U = [[0.0]*R for _ in range(n*m)]
    V = [[0.0]*R for _ in range(m*p)]
    W = [[0.0]*R for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1.0
                V[j*p+k][c] = 1.0
                W[k*n+i][c] = 1.0
                c += 1
    return {"U": U, "V": V, "W": W}

def _random_binary_decomposition(n, m, p, rank, nnz_range=(1,3)):
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    for i in range(n*m):
        nnz = random.randint(nnz_range[0], nnz_range[1])
        cols = random.sample(range(rank), min(nnz, rank))
        for c in cols:
            U[i, c] = random.choice([-1.0, 1.0])
    for j in range(m*p):
        nnz = random.randint(nnz_range[0], nnz_range[1])
        cols = random.sample(range(rank), min(nnz, rank))
        for c in cols:
            V[j, c] = random.choice([-1.0, 1.0])
    for k in range(p*n):
        nnz = random.randint(nnz_range[0], nnz_range[1])
        cols = random.sample(range(rank), min(nnz, rank))
        for c in cols:
            W[k, c] = random.choice([-1.0, 1.0])
    return U, V, W

def _try_merge(U, V, W, n, m, p, rank, col_to_remove, tol=1e-10):
    """Try to remove col_to_remove by solving for coefficients on remaining columns."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    new_rank = rank - 1
    if new_rank <= 0:
        return None
    
    mask = [c for c in range(rank) if c != col_to_remove]
    U_new = U[:, mask].copy()
    V_new = V[:, mask].copy()
    W_new = W[:, mask].copy()
    
    # Residual from removed column
    U_r = U[:, col_to_remove]
    V_r = V[:, col_to_remove]
    W_r = W[:, col_to_remove]
    
    # Build linear system: for each (i,j,k), sum_{c' in mask} alpha_{c'} * U_new[i,c']*V_new[j,c']*W_new[k,c'] = U_r[i]*V_r[j]*W_r[k]
    A_mat = np.zeros((n*m*p, new_rank))
    b_vec = np.zeros(n*m*p)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                b_vec[idx] = U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
                for nc, oc in enumerate(mask):
                    A_mat[idx, nc] = U_new[i*m+j, nc] * V_new[j*p+k, nc] * W_new[k*n+i, nc]
                idx += 1
    
    # Solve least squares
    try:
        alpha, residuals, rank_sv, sv = np.linalg.lstsq(A_mat, b_vec, rcond=1e-12)
    except np.linalg.LinAlgError:
        return None
    
    # Check residual
    recon = A_mat @ alpha
    max_err = np.max(np.abs(recon - b_vec))
    
    if max_err > tol:
        return None
    
    # Verify the full decomposition
    U_check = U_new.copy()
    V_check = V_new.copy()
    W_check = W_new.copy()
    # The coefficients alpha modify the effective contribution; but in tensor decomposition,
    # we can't just multiply columns by alpha. Instead, we need to check if the original
    # sum equals the new sum.
    # Actually, the correct approach: the new decomposition uses the SAME U_new, V_new, W_new
    # but the tensor sum should equal the original. Let's verify directly.
    
    # Original tensor: sum over all rank columns
    # New tensor: sum over new_rank columns
    # They should be equal if the merge is valid.
    
    # Verify by computing the full tensor error
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s_new = 0.0
                for nc in range(new_rank):
                    s_new += U_new[i*m+j, nc] * V_new[j*p+k, nc] * W_new[k*n+i, nc]
                # s_new should equal 1.0 (the target)
                e = abs(s_new - 1.0)
                if e > err:
                    err = e
    
    if err > 1e-9:
        return None
    
    return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}

def _systematic_merge_search(U, V, W, n, m, p, deadline, max_rank_reduction=10):
    """Systematically try to remove columns one at a time."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    current_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    reductions = 0
    while reductions < max_rank_reduction and time.time() < deadline:
        found = False
        for col in range(best_rank):
            if time.time() > deadline:
                break
            result = _try_merge(best_U.tolist(), best_V.tolist(), best_W.tolist(),
                               n, m, p, best_rank, col)
            if result is not None:
                best_U = np.array(result["U"], dtype=np.float64)
                best_V = np.array(result["V"], dtype=np.float64)
                best_W = np.array(result["W"], dtype=np.float64)
                best_rank -= 1
                reductions += 1
                found = True
                break
        if not found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _random_search_for_rank(n, m, p, target_rank, deadline, num_trials=5000):
    """Random search: generate random sparse binary decompositions and verify."""
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        U, V, W = _random_binary_decomposition(n, m, p, target_rank, nnz_range=(1, 3))
        err = _max_err(U.tolist(), V.tolist(), W.tolist(), n, m, p)
        if err < 1e-9:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def _perturb_and_merge(U, V, W, n, m, p, deadline, num_perturb_trials=200):
    """Perturb the decomposition slightly and try to merge columns."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    for trial in range(num_perturb_trials):
        if time.time() > deadline:
            break
        
        U_p = U.copy()
        V_p = V.copy()
        W_p = W.copy()
        
        # Perturb a few entries
        num_pert = random.randint(2, 5)
        for _ in range(num_pert):
            mat = random.choice([U_p, V_p, W_p])
            row = random.randint(0, mat.shape[0]-1)
            col = random.randint(0, mat.shape[1]-1)
            mat[row, col] += random.choice([-0.5, 0.5, -1.0, 1.0])
            mat[row, col] = round(mat[row, col] * 2) / 2
        
        # Verify the perturbed decomposition is still valid
        err = _max_err(U_p.tolist(), V_p.tolist(), W_p.tolist(), n, m, p)
        if err > 1e-9:
            continue
        
        # Try to merge columns
        for col in range(rank):
            if time.time() > deadline:
                break
            result = _try_merge(U_p.tolist(), V_p.tolist(), W_p.tolist(),
                               n, m, p, rank, col)
            if result is not None:
                best_U = np.array(result["U"], dtype=np.float64)
                best_V = np.array(result["V"], dtype=np.float64)
                best_W = np.array(result["W"], dtype=np.float64)
                best_rank = rank - 1
                break
        
        if best_rank < rank:
            U, V, W = best_U, best_V, best_W
            rank = best_rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with rank-36 constructive
    result = _construct_hybrid_rank36()
    U = np.array(result["U"], dtype=np.float64)
    V = np.array(result["V"], dtype=np.float64)
    W = np.array(result["W"], dtype=np.float64)
    best_rank = U.shape[1]
    
    # Verify
    if _max_err(result["U"], result["V"], result["W"], n, m, p) > 1e-9:
        # Fallback to schoolbook
        result = _construct_schoolbook()
        U = np.array(result["U"], dtype=np.float64)
        V = np.array(result["V"], dtype=np.float64)
        W = np.array(result["W"], dtype=np.float64)
        best_rank = U.shape[1]
    
    # Systematic merge search
    remaining = deadline - time.time()
    if remaining > 20:
        merge_result = _systematic_merge_search(
            result["U"], result["V"], result["W"],
            n, m, p, deadline, max_rank_reduction=8
        )
        merge_rank = len(merge_result["U"][0])
        if merge_rank < best_rank:
            result = merge_result
            best_rank = merge_rank
    
    # Perturb and merge
    remaining = deadline - time.time()
    if remaining > 20:
        perturb_result = _perturb_and_merge(
            result["U"], result["V"], result["W"],
            n, m, p, deadline, num_perturb_trials=300
        )
        perturb_rank = len(perturb_result["U"][0])
        if perturb_rank < best_rank:
            result = perturb_result
            best_rank = perturb_rank
    
    # Random search for lower ranks
    remaining = deadline - time.time()
    if remaining > 15:
        for target_rank in range(best_rank - 1, 24, -1):
            if time.time() > deadline:
                break
            random_result = _random_search_for_rank(n, m, p, target_rank, deadline, num_trials=2000)
            if random_result is not None:
                result = random_result
                best_rank = target_rank
                break
    
    return result

def construct():
    return _construct()
# EVOLVE-BLOCK-END
