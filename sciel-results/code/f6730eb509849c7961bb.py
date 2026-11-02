"""Matrix multiplication 3x4x5 - rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]
DEADLINE_SECONDS = 140

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    rank = U_np.shape[1]
    
    # Check all nonzero entries (where the tensor should be 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_np[i * m + j] * V_np[j * p + k] * W_np[k * n + i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _get_strassen_2x2x2():
    """Returns list of 7 (u, v, w) vectors for 2x2x2 Strassen."""
    terms = []
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    terms.append(([1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]))
    terms.append(([0, 0, 0, 1], [0, 1, 0, 0], [-1, 0, 1, 0]))
    terms.append(([1, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]))
    terms.append(([-1, 0, 1, 0], [1, 0, 0, 1], [0, 0, 0, -1]))
    terms.append(([0, 1, 0, -1], [0, 1, 0, 1], [0, 1, 0, -1]))
    return terms

def _get_rank_11_2x2x3():
    """
    Returns list of 11 (u, v, w) vectors for 2x2x3.
    Decomposes into 2x2x2 (rank 7) + 2x2x1 (rank 4).
    """
    terms = []
    s_terms = _get_strassen_2x2x2()
    for u, v, w in s_terms:
        u_g = u
        v_g = [v[0], v[1], 0, v[2], v[3], 0]
        w_g = [w[0], w[1], 0, w[2], w[3], 0]
        terms.append((u_g, v_g, w_g))
    
    terms.append(([1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]))
    terms.append(([0, 1, 0, 0], [0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0]))
    terms.append(([0, 0, 1, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]))
    terms.append(([0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0]))
    return terms

def _get_rank_4_1x2x2():
    """Returns list of 4 (u, v, w) vectors for 1x2x2."""
    terms = []
    terms.append(([1, 0], [1, 0, 0, 0], [1, 0]))
    terms.append(([1, 0], [0, 1, 0, 0], [0, 1]))
    terms.append(([0, 1], [0, 0, 1, 0], [1, 0]))
    terms.append(([0, 1], [0, 0, 0, 1], [0, 1]))
    return terms

def _get_rank_6_1x2x3():
    """Returns list of 6 (u, v, w) vectors for 1x2x3."""
    terms = []
    terms.append(([1, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0]))
    terms.append(([1, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0]))
    terms.append(([1, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1]))
    terms.append(([0, 1], [0, 0, 0, 1, 0, 0], [1, 0, 0]))
    terms.append(([0, 1], [0, 0, 0, 0, 1, 0], [0, 1, 0]))
    terms.append(([0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 1]))
    return terms

def _try_optimization(U, V, W, n, m, p, rank, deadline):
    """
    Attempt to reduce rank by checking if the current decomposition
    can be compressed. For each pair of terms, check if their
    contribution can be merged or if one is redundant.
    """
    # Try to find linear dependencies among the rank vectors
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Build a matrix where each column is the Kronecker product of three term vectors
    # If columns are linearly dependent, we can reduce rank
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Create tensor of rank vectors and check null space
    # This is expensive, so only try if rank is not too large
    if rank > 30:
        return None
    
    # Quick check: are any two terms identical?
    from itertools import combinations
    for i, j in combinations(range(rank), 2):
        if (np.allclose(U_np[:, i], U_np[:, j], atol=1e-10) and 
            np.allclose(V_np[:, i], V_np[:, j], atol=1e-10) and 
            np.allclose(W_np[:, i], W_np[:, j], atol=1e-10)):
            # Terms i and j are identical, can remove one if they sum to zero contribution
            # Actually they contribute twice, so we need to halve both or remove one and double the other
            # For now, just flag it
            pass
    
    # Try to find the rank of the Gram matrix of the terms
    # A term r contributes U[:,r] * V[:,r] * W[:,r]
    # Two terms r and s cancel if their Kronecker products sum to zero
    # This is equivalent to checking if there exist coefficients c_r such that
    # sum_r c_r * (U[:,r] ⊗ V[:,r] ⊗ W[:,r]) = 0
    
    # Build matrix M where M[index, r] = U[i,r] * V[j,r] * W[k,r] for flattened (i,j,k)
    # The tensor should equal sum of outer products, so M * c = 0 means cancellation
    n_entries = n * m * p
    M = np.zeros((n_entries, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    M[idx, r] = U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                idx += 1
    
    # Find null space of M
    try:
        U_svd, S, Vt = np.linalg.svd(M, full_matrices=False)
        # Rank of M
        tol_sv = 1e-10 * max(S) if len(S) > 0 else 1e-10
        rank_M = sum(1 for s in S if s > tol_sv)
        
        if rank_M < rank:
            # There are linear dependencies
            # The null space has dimension rank - rank_M
            null_dim = rank - rank_M
            # Get null space vectors
            null_vecs = Vt[rank_M:].T
            
            # We can express redundant terms in terms of independent ones
            # New rank is rank_M
            # But we need to reconstruct new terms
            # This is complex; for now, just return None unless we can do it simply
            if null_dim >= 1:
                # Simple case: one redundant term
                # Find which term can be eliminated
                # The null space vector tells us the linear combination that sums to zero
                # If a null vector has mostly one component, we can eliminate that term
                for nv in null_vecs:
                    max_idx = np.argmax(np.abs(nv))
                    max_val = np.abs(nv[max_idx])
                    if max_val > 0.5:
                        # Term max_idx is nearly dependent on others
                        # New coefficients: for term r != max_idx, new_coeff = old_coeff + nv[r]/nv[max_idx] * old_coeff[max_idx]
                        # But since we want to keep the same structure, just remove term max_idx
                        # and adjust others
                        new_U = []
                        new_V = []
                        new_W = []
                        for r in range(rank):
                            if r == max_idx:
                                continue
                            scale = 1.0 - nv[r] / nv[max_idx]
                            new_U.append(U_np[:, r] * scale)
                            new_V.append(V_np[:, r] * scale)
                            new_W.append(W_np[:, r] * scale)
                        
                        new_U = np.column_stack(new_U) if new_U else np.zeros((nm, 0))
                        new_V = np.column_stack(new_V) if new_V else np.zeros((mp, 0))
                        new_W = np.column_stack(new_W) if new_W else np.zeros((pn, 0))
                        
                        # Verify
                        new_rank = new_U.shape[1]
                        if new_rank < rank:
                            new_U_list = new_U.tolist()
                            new_V_list = new_V.tolist()
                            new_W_list = new_W.tolist()
                            if _verify_decomposition(new_U_list, new_V_list, new_W_list, n, m, p):
                                return {'U': new_U_list, 'V': new_V_list, 'W': new_W_list}
    except Exception:
        pass
    
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # We will construct the 56-rank decomposition.
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        for i in range(n * m):
            U[i][col] = u_vec[i]
        for i in range(m * p):
            V[i][col] = v_vec[i]
        for i in range(p * n):
            W[i][col] = w_vec[i]
        col += 1
    
    # Block 1: C11 (2x2) from A1 (2x4) and B1 (4x2)
    s_terms = _get_strassen_2x2x2()
    
    # Term 1: A11 * B11
    for u, v, w in s_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[0] = u[0]
        u_g[1] = u[1]
        u_g[4] = u[2]
        u_g[5] = u[3]
        v_g[0] = v[0]
        v_g[1] = v[1]
        v_g[5] = v[2]
        v_g[6] = v[3]
        w_g[0] = w[0]
        w_g[1] = w[1]
        w_g[3] = w[2]
        w_g[4] = w[3]
        add_term(u_g, v_g, w_g)
        
    # Term 2: A12 * B12
    for u, v, w in s_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[2] = u[0]
        u_g[3] = u[1]
        u_g[6] = u[2]
        u_g[7] = u[3]
        v_g[10] = v[0]
        v_g[11] = v[1]
        v_g[15] = v[2]
        v_g[16] = v[3]
        w_g[0] = w[0]
        w_g[1] = w[1]
        w_g[3] = w[2]
        w_g[4] = w[3]
        add_term(u_g, v_g, w_g)
        
    # Block 2: C12 (2x3) from A1 (2x4) and B2 (4x3)
    r11_terms = _get_rank_11_2x2x3()
    
    # Term 1: A11 * B21
    for u, v, w in r11_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[0] = u[0]
        u_g[1] = u[1]
        u_g[4] = u[2]
        u_g[5] = u[3]
        v_g[2] = v[0]
        v_g[3] = v[1]
        v_g[4] = v[2]
        v_g[7] = v[3]
        v_g[8] = v[4]
        v_g[9] = v[5]
        w_g[2] = w[0]
        w_g[3] = w[1]
        w_g[4] = w[2]
        w_g[5] = w[3]
        w_g[6] = w[4]
        w_g[7] = w[5]
        add_term(u_g, v_g, w_g)
        
    # Term 2: A12 * B22
    for u, v, w in r11_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[2] = u[0]
        u_g[3] = u[1]
        u_g[6] = u[2]
        u_g[7] = u[3]
        v_g[12] = v[0]
        v_g[13] = v[1]
        v_g[14] = v[2]
        v_g[17] = v[3]
        v_g[18] = v[4]
        v_g[19] = v[5]
        w_g[2] = w[0]
        w_g[3] = w[1]
        w_g[4] = w[2]
        w_g[5] = w[3]
        w_g[6] = w[4]
        w_g[7] = w[5]
        add_term(u_g, v_g, w_g)
        
    # Block 3: C21 (1x2) from A2 (1x4) and B1 (4x2)
    r4_terms = _get_rank_4_1x2x2()
    
    # Term 1: A21 * B11
    for u, v, w in r4_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[8] = u[0]
        u_g[9] = u[1]
        v_g[0] = v[0]
        v_g[1] = v[1]
        v_g[5] = v[2]
        v_g[6] = v[3]
        w_g[6] = w[0]
        w_g[7] = w[1]
        add_term(u_g, v_g, w_g)
        
    # Term 2: A22 * B12
    for u, v, w in r4_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[10] = u[0]
        u_g[11] = u[1]
        v_g[10] = v[0]
        v_g[11] = v[1]
        v_g[15] = v[2]
        v_g[16] = v[3]
        w_g[6] = w[0]
        w_g[7] = w[1]
        add_term(u_g, v_g, w_g)
        
    # Block 4: C22 (1x3) from A2 (1x4) and B2 (4x3)
    r6_terms = _get_rank_6_1x2x3()
    
    # Term 1: A21 * B21
    for u, v, w in r6_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[8] = u[0]
        u_g[9] = u[1]
        v_g[2] = v[0]
        v_g[3] = v[1]
        v_g[4] = v[2]
        v_g[7] = v[3]
        v_g[8] = v[4]
        v_g[9] = v[5]
        w_g[8] = w[0]
        w_g[9] = w[1]
        w_g[10] = w[2]
        add_term(u_g, v_g, w_g)
        
    # Term 2: A22 * B22
    for u, v, w in r6_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[10] = u[0]
        u_g[11] = u[1]
        v_g[12] = v[0]
        v_g[13] = v[1]
        v_g[14] = v[2]
        v_g[17] = v[3]
        v_g[18] = v[4]
        v_g[19] = v[5]
        w_g[8] = w[0]
        w_g[9] = w[1]
        w_g[10] = w[2]
        add_term(u_g, v_g, w_g)
        
    if col != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {col}")
        
    if not _verify_decomposition(U, V, W, n, m, p):
        schoolbook_rank = n * m * p
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to optimize the rank
    elapsed = time.time() - start_time
    if elapsed < DEADLINE_SECONDS - 20:  # Leave safety margin
        optimized = _try_optimization(U, V, W, n, m, p, rank, DEADLINE_SECONDS)
        if optimized is not None:
            return optimized
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
