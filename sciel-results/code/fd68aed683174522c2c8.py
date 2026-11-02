import numpy as np
import time

DIMENSIONS = [2, 2, 11]

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Build the tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute the decomposition
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=1e-10)

def _svd_reduce(U, V, W, tol=1e-9):
    """Use SVD to detect and remove linearly dependent terms.
    
    If term r can be expressed as a linear combination of other terms,
    we can eliminate it. We check this by seeing if the tensor contribution
    of term r is in the span of the other terms' contributions.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # Try to remove terms one by one
    removed = []
    for r in range(rank):
        if time.time() > 145:
            break
        if r in removed:
            continue
        
        # Check if term r is linearly dependent on the others
        # Build the tensor contribution of each term
        # Term r contributes: U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # We need to check if this is in the span of all other terms
        
        # Create matrix where each column is a term's tensor (flattened)
        # But that's too expensive. Instead, use a smarter approach:
        # Check if U[:,r] is in span of {U[:,r'] for r' != r} AND
        # V[:,r] is in span of {V[:,r'] for r' != r} AND
        # W[:,r] is in span of {W[:,r'] for r' != r}
        # This is sufficient but not necessary.
        
        # Better: check if the outer product U[:,r]⊗V[:,r] is in the span
        # of {U[:,r']⊗V[:,r'] for r' != r}
        # This is a necessary condition for the tensor term to be removable.
        
        # Actually, the cleanest approach: 
        # The full tensor is sum_r U[:,r]⊗V[:,r]⊗W[:,r]
        # Term r is removable if U[:,r]⊗V[:,r]⊗W[:,r] is in the span
        # of {U[:,r']⊗V[:,r']⊗W[:,r'] for r' != r}
        
        # This is hard to check directly. Let's use a different approach:
        # Check if there exist coefficients c_r' such that
        # sum_{r' != r} c_r' * U[:,r']⊗V[:,r']⊗W[:,r'] = U[:,r]⊗V[:,r]⊗W[:,r]
        
        # This is equivalent to: for all (a,b,c),
        # sum_{r' != r} c_r' * U[a,r']*V[b,r']*W[c,r'] = U[a,r]*V[b,r]*W[c,r]
        
        # Let's set up a linear system. For each (a,b,c), we have:
        # sum_{r' != r} c_r' * (U[a,r']*V[b,r']*W[c,r']) = U[a,r]*V[b,r]*W[c,r]
        
        # This is a system of (n*m)*(m*p)*(p*n) equations in (rank-1) unknowns.
        # That's 4*22*22 = 1936 equations in 38 unknowns. Overdetermined.
        
        # Let's use least squares and check if the residual is small.
        
        other_terms = [r_prime for r_prime in range(rank) if r_prime != r]
        if len(other_terms) == 0:
            continue
            
        # Build the linear system
        # A has shape (n*m*m*p*p*n, rank-1)
        # b has shape (n*m*m*p*p*n,)
        
        # This is too large. Let's sample or use a smarter method.
        
        # Alternative: check if U[:,r] is in the column span of U with column r removed,
        # AND V[:,r] is in the column span of V with column r removed,
        # AND W[:,r] is in the column span of W with column r removed.
        # If all three hold, then the term is definitely removable.
        
        U_others = np.delete(U, r, axis=1)
        V_others = np.delete(V, r, axis=1)
        W_others = np.delete(W, r, axis=1)
        
        # Check if U[:,r] is in span of U_others
        # Use SVD to check
        U_svd = np.linalg.svd(U_others, full_matrices=False)
        U_rank = np.sum(U_svd[0] > tol)
        if U_rank < U_others.shape[1]:
            # Use only the significant singular vectors
            U_basis = U_svd[0][:, :U_rank] @ np.diag(U_svd[1][:U_rank]) @ U_svd[2][:U_rank, :].T
        else:
            U_basis = U_others
        
        # Project U[:,r] onto span of U_others
        U_proj, U_residual = np.linalg.lstsq(U_others, U[:, r], rcond=None)
        U_err = np.linalg.norm(U[:, r] - U_others @ U_proj)
        
        V_proj, V_residual = np.linalg.lstsq(V_others, V[:, r], rcond=None)
        V_err = np.linalg.norm(V[:, r] - V_others @ V_proj)
        
        W_proj, W_residual = np.linalg.lstsq(W_others, W[:, r], rcond=None)
        W_err = np.linalg.norm(W[:, r] - W_others @ W_proj)
        
        if U_err < tol and V_err < tol and W_err < tol:
            # Term r is removable
            removed.append(r)
    
    if not removed:
        return U, V, W, rank
    
    # Remove the identified terms
    keep = [r for r in range(rank) if r not in removed]
    U_new = U[:, keep]
    V_new = V[:, keep]
    W_new = W[:, keep]
    new_rank = len(keep)
    
    return U_new, V_new, W_new, new_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Safety margin

    # Target Rank 39 Construction
    # 5 blocks of 2x2x2 (Strassen, rank 7) + 1 block of 2x2x1 (Schoolbook, rank 4)
    rank = 39
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    r = 0

    # Helper to add a Strassen block for columns k0, k1
    def add_strassen_block(k0, k1):
        nonlocal r
        # B indices: B[j, k] -> V[j*p + k]
        v_e = k0       # B[0, k0]
        v_f = k1       # B[0, k1]
        v_g = 11 + k0  # B[1, k0]
        v_h = 11 + k1  # B[1, k1]

        # W indices: C[i, k] -> W[k*n + i]
        w_c00 = 2 * k0     # C[0, k0]
        w_c10 = 2 * k0 + 1 # C[1, k0]
        w_c01 = 2 * k1     # C[0, k1]
        w_c11 = 2 * k1 + 1 # C[1, k1]

        # Strassen terms for 2x2 * 2x2
        # m1 = (a+d)(e+h)
        U[0, r] += 1; U[3, r] += 1
        V[v_e, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1; W[w_c11, r] += 1
        r += 1

        # m2 = (c+d)e
        U[2, r] += 1; U[3, r] += 1
        V[v_e, r] += 1
        W[w_c10, r] += 1; W[w_c11, r] += -1
        r += 1

        # m3 = a(f-h)
        U[0, r] += 1
        V[v_f, r] += 1; V[v_h, r] += -1
        W[w_c01, r] += 1; W[w_c11, r] += 1
        r += 1

        # m4 = d(g-e)
        U[3, r] += 1
        V[v_g, r] += 1; V[v_e, r] += -1
        W[w_c00, r] += 1; W[w_c10, r] += 1
        r += 1

        # m5 = (a+b)h
        U[0, r] += 1; U[1, r] += 1
        V[v_h, r] += 1
        W[w_c00, r] += -1; W[w_c01, r] += 1
        r += 1

        # m6 = (c-a)(e+f)
        U[2, r] += 1; U[0, r] += -1
        V[v_e, r] += 1; V[v_f, r] += 1
        W[w_c11, r] += 1
        r += 1

        # m7 = (b-d)(g+h)
        U[1, r] += 1; U[3, r] += -1
        V[v_g, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1
        r += 1

    # Add 5 Strassen blocks for column pairs (0,1), (2,3), (4,5), (6,7), (8,9)
    for block in range(5):
        if time.time() > deadline - 5:
            break
        k0 = block * 2
        k1 = block * 2 + 1
        add_strassen_block(k0, k1)

    # Last column k=10: Schoolbook
    if r < rank:
        k2 = 10
        v_e2 = k2      # B[0, 10]
        v_g2 = 11 + k2 # B[1, 10]
        w_c02 = 2 * k2      # C[0, 10]
        w_c12 = 2 * k2 + 1  # C[1, 10]

        # a * B[0, 10]
        U[0, r] += 1
        V[v_e2, r] += 1
        W[w_c02, r] += 1
        r += 1

        # b * B[1, 10]
        U[1, r] += 1
        V[v_g2, r] += 1
        W[w_c02, r] += 1
        r += 1

        # c * B[0, 10]
        U[2, r] += 1
        V[v_e2, r] += 1
        W[w_c12, r] += 1
        r += 1

        # d * B[1, 10]
        U[3, r] += 1
        V[v_g2, r] += 1
        W[w_c12, r] += 1
        r += 1

    # Verify the construction
    if time.time() < deadline - 2:
        if _verify(U, V, W):
            # Try to reduce rank using SVD-based detection
            if time.time() < deadline - 10:
                U_red, V_red, W_red, rank_red = _svd_reduce(U, V, W)
                if rank_red < rank and _verify(U_red, V_red, W_red):
                    return {
                        "U": U_red.tolist(),
                        "V": V_red.tolist(),
                        "W": W_red.tolist()
                    }
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    # Fallback: Schoolbook (Rank 44)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
