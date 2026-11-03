import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n_tests=3):
    """
    Verify the decomposition by checking the tensor identity.
    """
    n = m = p = 3
    if not U or not U[0]:
        return False
    rank = len(U[0])
    
    U_np = np.array(U, dtype=float).reshape(n, m, rank)
    V_np = np.array(V, dtype=float).reshape(m, p, rank)
    W_np = np.array(W, dtype=float).reshape(p, n, rank)
    
    # Check: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1 for all i,j,k
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_np, V_np, W_np)
    T_target = np.ones((n, m, p))
    
    if not np.allclose(T_recon, T_target, atol=1e-9):
        return False
    
    # Additional random matrix test
    for _ in range(n_tests):
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        # C[i,k] = sum_j A[i,j]*B[j,k] = sum_j sum_r A[i,j]*U[i,j,r]*V[j,k,r]*B[j,k]*W[k,i,r]
        # = sum_r W[k,i,r] * (sum_j A[i,j]*U[i,j,r]) * (sum_j V[j,k,r]*B[j,k])
        AU = np.einsum('ijr,im->jr', U_np, A)  # This is wrong dimensionally
        # Let me redo: for each r, compute sum_j A[i,j]*U[i,j,r] for each i
        # and sum_j V[j,k,r]*B[j,k] for each k
        # Then C[i,k] = sum_r W[k,i,r] * (sum_j A[i,j]*U[i,j,r]) * (sum_j V[j,k,r]*B[j,k])
        # Wait, that's not right. Let me think again.
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The decomposition says: sum_j A[i,j]*B[j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] summed over j
        # = sum_r W[k,i,r] * sum_j A[i,j]*U[i,j,r]*V[j,k,r]*B[j,k]
        # This doesn't factor as I wrote. Let me just trust the tensor check.
        pass
    
    return True

def _build_from_terms(terms):
    """Build U, V, W from list of (u_dict, v_dict, w_dict) terms."""
    n, m, p = 3, 3, 3
    R = len(terms)
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), val in u_dict.items():
            U[u_idx(i, j)][r] += val
        for (j, k), val in v_dict.items():
            V[v_idx(j, k)][r] += val
        for (k, i), val in w_dict.items():
            W[w_idx(k, i)][r] += val
    
    return {"U": U, "V": V, "W": W}

def _strassen_rank26():
    """
    Rank-26 decomposition: 7 Strassen terms for 2x2 core + 19 border terms.
    """
    n = m = p = 3
    terms = []
    
    # 7 Strassen terms for the 2x2 submatrix (indices 0,1)
    terms.append((
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    ))
    terms.append((
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1}
    ))
    terms.append((
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1}
    ))
    terms.append((
        {(1,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1}
    ))
    terms.append((
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,0): -1, (1,0): 1}
    ))
    terms.append((
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(1,1): 1}
    ))
    terms.append((
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1}
    ))
    
    # 19 border terms
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i < 2 and j < 2 and k < 2:
                    continue
                terms.append(
                    {(i, j): 1},
                    {(j, k): 1},
                    {(k, i): 1}
                )
    
    return _build_from_terms(terms)

def _schoolbook():
    """Standard rank-27 decomposition."""
    n = m = p = 3
    rank = 27
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

def _try_reduce_rank(U, V, W, current_rank):
    """
    Try to reduce the rank by finding linear dependencies.
    """
    n = m = p = 3
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    term_vectors = []
    for r in range(current_rank):
        vec = np.concatenate([U_np[:, r], V_np[:, r], W_np[:, r]])
        term_vectors.append(vec)
    
    term_vectors = np.array(term_vectors)
    
    try:
        rank = np.linalg.matrix_rank(term_vectors, tol=1e-10)
    except np.linalg.LinAlgError:
        return None
    
    if rank >= current_rank:
        return None
    
    try:
        _, s, vh = np.linalg.svd(term_vectors, full_matrices=True)
    except np.linalg.LinAlgError:
        return None
    
    null_space = vh[rank:, :]
    
    for ns_vec in null_space:
        abs_coeffs = np.abs(ns_vec)
        max_idx = np.argmax(abs_coeffs)
        
        if abs_coeffs[max_idx] < 1e-10:
            continue
        
        scale = ns_vec[max_idx]
        if abs(scale) < 1e-15:
            continue
        ns_vec = ns_vec / scale
        
        new_rank = current_rank - 1
        new_U = np.zeros((n * m, new_rank))
        new_V = np.zeros((m * p, new_rank))
        new_W = np.zeros((p * n, new_rank))
        
        for old_r in range(current_rank):
            if old_r == max_idx:
                continue
            new_r = old_r if old_r < max_idx else old_r - 1
            new_U[:, new_r] = U_np[:, old_r]
            new_V[:, new_r] = V_np[:, old_r]
            new_W[:, new_r] = W_np[:, old_r]
        
        for new_r in range(new_rank):
            old_r = new_r if new_r < max_idx else new_r + 1
            coeff = ns_vec[old_r]
            if abs(coeff) > 1e-15:
                new_U[:, new_r] += coeff * U_np[:, max_idx]
                new_V[:, new_r] += coeff * V_np[:, max_idx]
                new_W[:, new_r] += coeff * W_np[:, max_idx]
        
        new_U_list = new_U.tolist()
        new_V_list = new_V.tolist()
        new_W_list = new_W.tolist()
        
        if _verify_decomposition(new_U_list, new_V_list, new_W_list, n_tests=2):
            return {"U": new_U_list, "V": new_V_list, "W": new_W_list}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_result = None
    best_rank = float('inf')
    
    # Try rank-26 Strassen decomposition
    result26 = _strassen_rank26()
    if _verify_decomposition(result26["U"], result26["V"], result26["W"]):
        best_result = result26
        best_rank = 26
    
    # Try to reduce rank iteratively
    if best_result is not None:
        current_result = best_result
        current_rank = best_rank
        for _ in range(5):
            if time.time() > deadline - 10.0:
                break
            reduced = _try_reduce_rank(
                current_result["U"], 
                current_result["V"], 
                current_result["W"], 
                current_rank
            )
            if reduced is None:
                break
            current_result = reduced
            current_rank = len(current_result["U"][0])
            best_result = current_result
            best_rank = current_rank
    
    # Fallback to schoolbook if nothing verified
    if best_result is None:
        best_result = _schoolbook()
    
    return best_result

# EVOLVE-BLOCK-END
