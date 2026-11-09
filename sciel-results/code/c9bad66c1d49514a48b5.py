"""Matrix multiplication 2,4,6 – rank-minimal decomposition via block decomposition."""
DIMENSIONS = [2, 4, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Standard O(nmp) decomposition for multiplying (n x m) by (m x p).
    Returns U (n*m x nmp), V (m*p x nmp), W (p*n x nmp)."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen)."""
    n_s, m_s, p_s = 2, 2, 2
    rank_s = 7
    U = np.zeros((n_s * m_s, rank_s), dtype=float)
    V = np.zeros((m_s * p_s, rank_s), dtype=float)
    W = np.zeros((p_s * n_s, rank_s), dtype=float)
    
    U[0, 0] = 1; U[3, 0] = 1
    U[2, 1] = 1; U[3, 1] = 1
    U[0, 2] = 1
    U[3, 3] = 1
    U[0, 4] = 1; U[2, 4] = 1
    U[2, 5] = -1; U[3, 5] = 1
    U[1, 6] = 1; U[3, 6] = -1
    
    V[0, 0] = 1; V[3, 0] = 1
    V[0, 1] = 1
    V[1, 2] = 1; V[3, 2] = -1
    V[0, 3] = -1; V[2, 3] = 1
    V[3, 4] = 1
    V[0, 5] = 1; V[2, 5] = 1
    V[1, 6] = 1; V[3, 6] = 1
    
    W[0, 0] = 1; W[3, 0] = 1
    W[1, 1] = 1; W[3, 1] = -1
    W[2, 2] = 1; W[3, 2] = 1
    W[0, 3] = 1; W[1, 3] = 1
    W[0, 4] = -1; W[2, 4] = 1
    W[3, 5] = 1
    W[0, 6] = 1
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(s - 1) > 1e-9:
                    return False
    return True


def _build_2x4x2_from_strassen(U_s, V_s, W_s):
    """Build 2x4x2 decomposition (rank 14) from two 2x2x2 Strassen blocks."""
    n, m, p = 2, 4, 2
    rank = 14
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r in range(7):
        for i in range(2):
            for j_local in range(2):
                U[i * 4 + j_local, r] = U_s[i * 2 + j_local, r]
                for k in range(2):
                    V[j_local * 2 + k, r] = V_s[j_local * 2 + k, r]
                for k in range(2):
                    W[k * 2 + i, r] = W_s[k * 2 + i, r]
    
    for r in range(7):
        for i in range(2):
            for j_local in range(2):
                U[i * 4 + (j_local + 2), r + 7] = U_s[i * 2 + j_local, r]
                for k in range(2):
                    V[(j_local + 2) * 2 + k, r + 7] = V_s[j_local * 2 + k, r]
                for k in range(2):
                    W[k * 2 + i, r + 7] = W_s[k * 2 + i, r]
    
    return U, V, W


def _build_2x4x4_from_2x4x2(U1, V1, W1):
    """Build 2x4x4 decomposition (rank 28) from two 2x4x2 blocks."""
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r in range(14):
        for i in range(2):
            for j in range(4):
                U[i * 4 + j, r] = U1[i * 4 + j, r]
            for j in range(4):
                for k_local in range(2):
                    V[j * 4 + k_local, r] = V1[j * 2 + k_local, r]
            for k_local in range(2):
                for i in range(2):
                    W[k_local * 2 + i, r] = W1[k_local * 2 + i, r]
    
    for r in range(14):
        for i in range(2):
            for j in range(4):
                U[i * 4 + j, r + 14] = U1[i * 4 + j, r]
            for j in range(4):
                for k_local in range(2):
                    V[j * 4 + (k_local + 2), r + 14] = V1[j * 2 + k_local, r]
            for k_local in range(2):
                for i in range(2):
                    W[(k_local + 2) * 2 + i, r + 14] = W1[k_local * 2 + i, r]
    
    return U, V, W


def _build_2x4x6_from_blocks(U4, V4, W4, U2, V2, W2):
    """Build 2x4x6 decomposition (rank 42) from 2x4x4 and 2x4x2 blocks."""
    n, m, p = 2, 4, 6
    total_rank = 42
    U_total = np.zeros((n * m, total_rank), dtype=float)
    V_total = np.zeros((m * p, total_rank), dtype=float)
    W_total = np.zeros((p * n, total_rank), dtype=float)
    
    U_total[:, :28] = U4
    for j in range(4):
        for k in range(4):
            V_total[j * 6 + k, :28] = V4[j * 4 + k, :]
    for k in range(4):
        for i in range(2):
            W_total[k * 2 + i, :28] = W4[k * 2 + i, :]
    
    U_total[:, 28:] = U2
    for j in range(4):
        for k_local in range(2):
            V_total[j * 6 + (k_local + 4), 28:] = V2[j * 2 + k_local, :]
    for k_local in range(2):
        for i in range(2):
            W_total[(k_local + 4) * 2 + i, 28:] = W2[k_local * 2 + i, :]
    
    return U_total, V_total, W_total


def _get_tensor_matrix(U, V, W, n, m, p):
    """Form the matrix M where M[entry, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]."""
    num_entries = n * m * p
    rank = U.shape[1]
    M = np.zeros((num_entries, rank), dtype=float)
    entry_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    M[entry_idx, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                entry_idx += 1
    return M


def _try_eliminate_term(U, V, W, n, m, p, term_idx):
    """Try to eliminate term_idx by finding coefficients for other terms.
    
    We want to find c_r for r != term_idx such that:
    sum_{r != term_idx} c_r * U_r[i,j] * V_r[j,k] * W_r[k,i] = -U_term[i,j] * V_term[j,k] * W_term[k,i]
    
    for all (i,j,k). This is a linear system in the c_r's.
    """
    rank = U.shape[1]
    if rank <= 1:
        return None
    
    # Build the linear system
    num_entries = n * m * p
    other_terms = [r for r in range(rank) if r != term_idx]
    num_unknowns = len(other_terms)
    
    A = np.zeros((num_entries, num_unknowns), dtype=float)
    b = np.zeros(num_entries, dtype=float)
    
    entry_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # RHS: negative of the term we want to eliminate
                b[entry_idx] = -U[i*m+j, term_idx] * V[j*p+k, term_idx] * W[k*n+i, term_idx]
                
                # LHS: contributions from other terms
                for idx, r in enumerate(other_terms):
                    A[entry_idx, idx] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                entry_idx += 1
    
    # Solve the linear system
    try:
        coeffs, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=1e-10)
    except:
        return None
    
    # Check if the solution is good
    if residuals is not None and len(residuals) > 0:
        if residuals[0] > 1e-8 * (np.linalg.norm(b)**2 + 1e-15):
            return None
    
    # Verify the solution
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for idx, r in enumerate(other_terms):
                    s += coeffs[idx] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                expected = -U[i*m+j, term_idx] * V[j*p+k, term_idx] * W[k*n+i, term_idx]
                if abs(s - expected) > 1e-8:
                    return None
    
    # Build new U, V, W without term_idx
    new_rank = rank - 1
    U_new = np.zeros((n*m, new_rank), dtype=float)
    V_new = np.zeros((m*p, new_rank), dtype=float)
    W_new = np.zeros((p*n, new_rank), dtype=float)
    
    col = 0
    for idx, r in enumerate(other_terms):
        U_new[:, col] = U[:, r] * coeffs[idx]
        V_new[:, col] = V[:, r]
        W_new[:, col] = W[:, r]
        col += 1
    
    return U_new, V_new, W_new


def _rank_reduce(U, V, W, n, m, p, max_eliminations=10):
    """Iteratively try to eliminate terms from the decomposition."""
    for _ in range(max_eliminations):
        rank = U.shape[1]
        if rank <= 1:
            break
        
        # Try to eliminate each term
        eliminated = False
        for term_idx in range(rank):
            result = _try_eliminate_term(U, V, W, n, m, p, term_idx)
            if result is not None:
                U, V, W = result
                eliminated = True
                break
        
        if not eliminated:
            break
    
    return U, V, W


def _random_projection_reduce(U, V, W, n, m, p, target_rank, num_trials=20):
    """Try to find a lower-rank decomposition via random projections.
    
    If we have a rank-R decomposition, we can try to find a rank-r < r decomposition
    by projecting onto a random r-dimensional subspace and checking if the tensor
    is still preserved.
    """
    current_rank = U.shape[1]
    if current_rank <= target_rank:
        return U, V, W
    
    best_U, best_V, best_W = U, V, W
    best_rank = current_rank
    
    for trial in range(num_trials):
        # Generate a random projection matrix
        P = np.random.randn(current_rank, target_rank)
        
        # Apply projection: new_U = U @ P, new_V = V @ P, new_W = W @ P
        # But this doesn't preserve the tensor in general.
        # Instead, we need to find U', V', W' of rank target_rank such that
        # sum_r' U'[i*m+j, r'] * V'[j*p+k, r'] * W'[k*n+i, r'] = 1
        
        # This is hard. Let's try a different approach:
        # Check if the current decomposition has a null space.
        
        M = _get_tensor_matrix(U, V, W, n, m, p)
        try:
            rank_M = np.linalg.matrix_rank(M, tol=1e-10)
        except:
            continue
        
        if rank_M < current_rank:
            # There's a null space
            try:
                null_space = np.linalg.null_space(M)
            except:
                continue
            
            if null_space.shape[1] > 0:
                # Find which term to eliminate
                for term_idx in range(current_rank):
                    result = _try_eliminate_term(U, V, W, n, m, p, term_idx)
                    if result is not None:
                        U, V, W = result
                        best_U, best_V, best_W = U, V, W
                        best_rank = U.shape[1]
                        if best_rank <= target_rank:
                            return best_U, best_V, best_W
                        break
    
    return best_U, best_V, best_W


def construct():
    """Construct rank-minimal exact tensor decomposition for 2x4 * 4x6 multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Step 1: Get Strassen 2x2x2 decomposition
    U_s, V_s, W_s = _strassen_2x2x2()
    
    if not _verify(U_s, V_s, W_s, 2, 2, 2):
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Step 2: Build 2x4x2 from two 2x2x2 Strassen blocks (rank 14)
    U1, V1, W1 = _build_2x4x2_from_strassen(U_s, V_s, W_s)
    
    if not _verify(U1, V1, W1, n, m, 2):
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if time.time() > deadline:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Step 3: Build 2x4x4 from two 2x4x2 blocks (rank 28)
    U2, V2, W2 = _build_2x4x4_from_2x4x2(U1, V1, W1)
    
    if not _verify(U2, V2, W2, n, m, 4):
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if time.time() > deadline:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Step 4: Build full 2x4x6 from 2x4x4 + 2x4x2 (rank 42)
    U_total, V_total, W_total = _build_2x4x6_from_blocks(U2, V2, W2, U1, V1, W1)
    
    if not _verify(U_total, V_total, W_total, n, m, p):
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if time.time() > deadline:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Step 5: Try to reduce the rank
    if time.time() < deadline - 5.0:
        U_total, V_total, W_total = _rank_reduce(U_total, V_total, W_total, n, m, p, max_eliminations=15)
        
        if not _verify(U_total, V_total, W_total, n, m, p):
            U_total, V_total, W_total = _build_2x4x6_from_blocks(U2, V2, W2, U1, V1, W1)
    
    if time.time() > deadline:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

# EVOLVE-BLOCK-END
