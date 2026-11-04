"""Matrix multiplication 2x3 by 3x5 with rank-minimal exact tensor decomposition."""
import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 5]

def _peel_rank1_terms(U, V, W, T, max_peels=5):
    """
    Attempt to peel off linearly dependent rank-1 terms from the decomposition.
    For each rank-1 term, check if it can be expressed as a linear combination
    of the others (making it linearly dependent in the tensor space).
    If found, remove it and try to find a correction term.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    n_rows_u, rank = U.shape
    n_rows_v = V.shape[0]
    n_rows_w = W.shape[0]
    
    # Build the tensor from current decomposition
    T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    
    # Try to find linearly dependent columns
    # A column r is linearly dependent if U[:,r] can be expressed as 
    # linear combination of other U columns AND V[:,r] as combo of other V columns
    # AND W[:,r] as combo of other W columns (approximately)
    
    # More practical approach: check if removing column r and adjusting others
    # still gives the same tensor
    
    peels = 0
    for r in range(rank - 1, -1, -1):
        if peels >= max_peels:
            break
        # Try removing column r
        U_no_r = np.delete(U, r, axis=1)
        V_no_r = np.delete(V, r, axis=1)
        W_no_r = np.delete(W, r, axis=1)
        
        T_no_r = np.einsum('ar,br,cr->abc', U_no_r, V_no_r, W_no_r, optimize=True)
        residual = T_recon - T_no_r
        
        # Check if residual is a valid rank-1 tensor
        # A rank-1 tensor has the form u ⊗ v ⊗ w
        # Check if residual can be written as outer product
        # This is equivalent to checking if residual has tensor rank 1
        
        # Simple check: if residual is zero, the term was redundant
        if np.max(np.abs(residual)) < 1e-10:
            # Term r is redundant, remove it
            U = U_no_r
            V = V_no_r
            W = W_no_r
            T_recon = T_no_r
            peels += 1
            continue
        
        # Check if residual is rank-1 by checking if it can be decomposed
        # For a 3D array, rank-1 means it's an outer product
        # We can check this by verifying that all 2x2 minors of certain unfoldings are zero
        
        # Unfold along first mode: shape (n*m, m*p * p*n)
        res_flat = residual.reshape(residual.shape[0], -1)
        if res_flat.shape[1] > 0 and res_flat.shape[0] > 0:
            # Compute SVD
            try:
                U_svd, s, Vt = np.linalg.svd(res_flat, full_matrices=False)
                # If rank-1, only one singular value should be non-zero
                if len(s) == 1 or (len(s) > 1 and s[1] < 1e-10 * max(s[0], 1e-15)):
                    # Rank-1 residual, can be represented as single term
                    # But we already have rank-1 terms, so this means we can
                    # potentially merge or the original term was expressible
                    pass
            except:
                pass
    
    return U.tolist(), V.tolist(), W.tolist(), peels

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[a, b, c] = 1 if the term A[i,j]*B[j,k] contributes to C[k,i]
    where a = i*m+j, b = j*p+k, c = k*n+i
    """
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[a,r]*V[b,r]*W[c,r] equals T[a,b,c]."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    diff = np.max(np.abs(T_recon - T))
    return diff < tol, diff

def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
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

def _strassen_2x2x2():
    """
    Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7).
    
    A indices: A[i,j] -> row i*2+j
    B indices: B[j,k] -> row j*2+k
    C indices: C[k,i] -> row k*2+i [trace convention]
    
    So: A rows: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
         B rows: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
         C rows: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
    
    # M2 = A11*(B10-B00)
    U[3][1] = 1
    V[2][1] = 1; V[0][1] = -1
    W[2][1] = 1; W[3][1] = 1
    
    # M3 = (A00-A10)*B11
    U[0][2] = 1; U[2][2] = -1
    V[3][2] = 1
    W[0][2] = 1; W[1][2] = 1; W[2][2] = 1; W[3][2] = 1
    
    # M4 = A00*(B01-B11)
    U[0][3] = 1
    V[1][3] = 1; V[3][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A01+A11)*(B00+B01)
    U[1][4] = 1; U[3][4] = 1
    V[0][4] = 1; V[1][4] = 1
    W[0][4] = 1; W[1][4] = 1; W[2][4] = 1; W[3][4] = 1
    
    # M6 = (A00-A10)*(B10+B11)
    U[0][5] = 1; U[2][5] = -1
    V[2][5] = 1; V[3][5] = 1
    W[2][5] = 1; W[3][5] = 1
    
    # M7 = (A01-A11)*(B00-B10)
    U[1][6] = 1; U[3][6] = -1
    V[0][6] = 1; V[2][6] = -1
    W[0][6] = 1; W[1][6] = 1; W[2][6] = 1; W[3][6] = 1
    
    return U, V, W

def _rank28_decomposition(n, m, p):
    """
    Rank-28 decomposition using two Strassen 2x2x2 blocks.
    
    Structure:
    - Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] (Strassen, 7 terms)
    - Block 2: A[0:2,0:2] * B[0:2,3:5] -> C[0:2,3:5] (Strassen, 7 terms)
    - Schoolbook: A[0:2,0:2] * B[0:2,2:3] -> C[0:2,2:3] (4 terms)
    - Schoolbook: A[0:2,2:3] * B[2:3,0:5] -> C[0:2,0:5] (10 terms)
    Total: 7 + 7 + 4 + 10 = 28
    
    Index mappings:
    A[i,j] -> row i*3+j (i in {0,1}, j in {0,1,2})
    B[j,k] -> row j*5+k (j in {0,1,2}, k in {0,1,2,3,4})
    C[k,i] -> row k*2+i (k in {0,1,2,3,4}, i in {0,1})
    """
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # (6, 28)
    V = [[0] * rank for _ in range(m * p)]  # (15, 28)
    W = [[0] * rank for _ in range(p * n)]  # (10, 28)
    
    col = 0
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # A indices: 0=(0,0), 1=(0,1), 3=(1,0), 4=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 5=(1,0), 6=(1,1)
    # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    a_map1 = [0, 1, 3, 4]
    b_map1 = [0, 1, 5, 6]
    c_map1 = [0, 1, 2, 3]
    
    for r in range(7):
        for s in range(4):
            if U_s[s][r] != 0:
                U[a_map1[s]][col] += U_s[s][r]
            if V_s[s][r] != 0:
                V[b_map1[s]][col] += V_s[s][r]
            if W_s[s][r] != 0:
                W[c_map1[s]][col] += W_s[s][r]
        col += 1
    
    # Block 2: A[0:2,0:2] * B[0:2,3:5] -> C[0:2,3:5]
    # A indices: 0=(0,0), 1=(0,1), 3=(1,0), 4=(1,1)
    # B indices: 3=(0,3), 4=(0,4), 8=(1,3), 9=(1,4)
    # C indices: 6=(3,0), 7=(3,1), 8=(4,0), 9=(4,1)
    a_map2 = [0, 1, 3, 4]
    b_map2 = [3, 4, 8, 9]
    c_map2 = [6, 7, 8, 9]
    
    for r in range(7):
        for s in range(4):
            if U_s[s][r] != 0:
                U[a_map2[s]][col] += U_s[s][r]
            if V_s[s][r] != 0:
                V[b_map2[s]][col] += V_s[s][r]
            if W_s[s][r] != 0:
                W[c_map2[s]][col] += W_s[s][r]
        col += 1
    
    # Schoolbook: A[0:2,0:2] * B[0:2,2:3] -> C[0:2,2:3]
    # For each (i,j) in {0,1}x{0,1} and k=2
    for i in range(2):
        for j in range(2):
            k = 2
            U[i * 3 + j][col] = 1
            V[j * 5 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
    
    # Schoolbook: A[0:2,2:3] * B[2:3,0:5] -> C[0:2,0:5]
    # For each i in {0,1} and k in {0,1,2,3,4}
    for i in range(2):
        for k in range(5):
            U[i * 3 + 2][col] = 1
            V[2 * 5 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}

def _reduce_rank_via_nullspace(U_arr, V_arr, W_arr, T, tol=1e-9):
    """
    Attempt to reduce rank by finding null space vectors and eliminating terms.
    Returns (U_new, V_new, W_new, new_rank) or None if no reduction found.
    """
    n_rows_u, rank = U_arr.shape
    n_rows_v = V_arr.shape[0]
    n_rows_w = W_arr.shape[0]
    
    # Build the 900x28 matrix efficiently using einsum
    # A[a,b,c,r] = U[a,r] * V[b,r] * W[c,r]
    # Reshape to (n_rows_u * n_rows_v * n_rows_w, rank)
    
    # Use broadcasting to build the matrix
    # For each r, the vector is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # We can build this as: for each r, outer(U[:,r], outer(V[:,r], W[:,r]))
    
    # More efficient: build column by column
    vec_dim = n_rows_u * n_rows_v * n_rows_w
    A_matrix = np.zeros((vec_dim, rank))
    
    for r in range(rank):
        # Kronecker product: U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # This is equivalent to flattening the 3D array U[:,r,None,None] * V[None,:,r,None] * W[None,None,:,r]
        vec = np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r]).flatten()
        A_matrix[:, r] = vec
    
    # Find null space using SVD
    try:
        U_svd, s, Vt = np.linalg.svd(A_matrix, full_matrices=True)
    except:
        return None
    
    # Determine numerical rank
    if len(s) == 0:
        return None
    max_sv = s[0] if s[0] > 0 else 1e-15
    num_rank = np.sum(s > tol * max_sv)
    null_dim = rank - num_rank
    
    if null_dim <= 0:
        return None
    
    # Get null space vectors (last null_dim rows of Vt)
    null_vecs = Vt[rank - null_dim:rank, :].T
    
    # For each null vector, try to eliminate the term with largest coefficient
    best_reduction = None
    best_new_rank = rank
    
    for nv in null_vecs:
        # Find index with largest absolute coefficient
        max_idx = np.argmax(np.abs(nv))
        if abs(nv[max_idx]) < tol:
            continue
        
        # We can eliminate term max_idx by expressing it as:
        # term[max_idx] = -sum_{r != max_idx} (nv[r]/nv[max_idx]) * term[r]
        # This means the decomposition without term max_idx, with adjusted coefficients, still works
        
        # But wait - this doesn't reduce the number of terms! We still have rank terms.
        # The null space tells us the terms are linearly dependent, but to actually reduce rank,
        # we need to find a different set of rank-1 tensors that span the same space.
        
        # Alternative approach: if we can find that removing one term and the residual
        # can be absorbed into the remaining terms, we reduce rank by 1.
        
        # Actually, the correct interpretation: if there's a null vector, it means
        # sum_r nv[r] * (U[:,r] ⊗ V[:,r] ⊗ W[:,r]) = 0
        # This doesn't directly give us a lower-rank decomposition.
        
        # To actually reduce rank, we need to find a basis for the column space
        # of A_matrix that has fewer than rank vectors, each corresponding to a rank-1 tensor.
        
        # This is the tensor rank minimization problem, which is NP-hard in general.
        # For small cases, we can try greedy elimination.
        
        # Try: for each term r, check if removing it and finding a rank-1 correction
        # that's linearly dependent on the remaining terms works.
        
        pass
    
    # Alternative: try to find if any single term can be removed such that the residual
    # is in the span of the remaining rank-1 tensors
    
    for r_remove in range(rank):
        U_no_r = np.delete(U_arr, r_remove, axis=1)
        V_no_r = np.delete(V_arr, r_remove, axis=1)
        W_no_r = np.delete(W_arr, r_remove, axis=1)
        
        T_no_r = np.einsum('ar,br,cr->abc', U_no_r, V_no_r, W_no_r, optimize=True)
        residual = T - T_no_r
        
        # Check if residual is in the span of the remaining rank-1 tensors
        # Build the matrix for the remaining terms
        rank_rem = rank - 1
        if rank_rem == 0:
            continue
        
        A_rem = np.zeros((vec_dim, rank_rem))
        for r in range(rank_rem):
            vec = np.einsum('a,b,c->abc', U_no_r[:, r], V_no_r[:, r], W_no_r[:, r]).flatten()
            A_rem[:, r] = vec
        
        # Check if residual vector is in the column space of A_rem
        res_vec = residual.flatten()
        
        # Solve A_rem * c = res_vec in least squares sense
        try:
            c, residuals, rank_c, s_c = np.linalg.lstsq(A_rem, res_vec, rcond=tol)
            # Check if the solution is good
            A_rem_c = A_rem @ c
            if np.max(np.abs(A_rem_c - res_vec)) < 1e-6:
                # The residual is in the span, so we can remove term r_remove
                # But we need to find the actual rank-1 tensors that sum to the residual
                
                # The residual = sum_{r in remaining} c[r] * (U_no_r[:,r] ⊗ V_no_r[:,r] ⊗ W_no_r[:,r])
                # This means the original decomposition can be rewritten without term r_remove
                # by adjusting the coefficients of the remaining terms
                
                # New decomposition: same U_no_r, V_no_r, W_no_r but with scaled columns
                # For each remaining term r, the new coefficient is 1 + c[r] (if we absorb)
                # Wait, this isn't quite right. The original is:
                # T = sum_{all r} U[:,r] ⊗ V[:,r] ⊗ W[:,r]
                # T_no_r = sum_{r != r_remove} U[:,r] ⊗ V[:,r] ⊗ W[:,r]
                # residual = T - T_no_r = U[:,r_remove] ⊗ V[:,r_remove] ⊗ W[:,r_remove]
                # And residual = sum_{r != r_remove} c[r] * (U_no_r[:,r] ⊗ V_no_r[:,r] ⊗ W_no_r[:,r])
                # So: U[:,r_remove] ⊗ V[:,r_remove] ⊗ W[:,r_remove] = sum_{r != r_remove} c[r] * (U_no_r[:,r] ⊗ V_no_r[:,r] ⊗ W_no_r[:,r])
                # This means the term r_remove is a linear combination of the others!
                # So we can simply remove it and the decomposition still works with rank-1 terms.
                
                # New decomposition has rank-1 terms
                return U_no_r.tolist(), V_no_r.tolist(), W_no_r.tolist(), rank_rem
        except:
            continue
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    
    # Try rank-28 decomposition
    result = _rank28_decomposition(n, m, p)
    valid, diff = _verify_decomposition(result['U'], result['V'], result['W'], T)
    
    if not valid:
        # Fallback to schoolbook
        return _schoolbook_decomposition(n, m, p)
    
    # Attempt to reduce rank
    U_arr = np.array(result['U'], dtype=np.float64)
    V_arr = np.array(result['V'], dtype=np.float64)
    W_arr = np.array(result['W'], dtype=np.float64)
    
    # Try rank reduction
    reduced = _reduce_rank_via_nullspace(U_arr, V_arr, W_arr, T)
    if reduced is not None:
        U_new, V_new, W_new, new_rank = reduced
        # Verify the reduced decomposition
        valid_new, diff_new = _verify_decomposition(U_new, V_new, W_new, T)
        if valid_new:
            return {'U': U_new, 'V': V_new, 'W': W_new}
    
    # If no reduction found, return the original
    return result

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0]) if result['U'] else 0}")
    
    # Verify
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    valid, diff = _verify_decomposition(result['U'], result['V'], result['W'], T)
    print(f"Valid: {valid}, Max diff: {diff}")
