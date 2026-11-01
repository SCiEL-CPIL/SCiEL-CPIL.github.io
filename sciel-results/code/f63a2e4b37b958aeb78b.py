import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 8]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the multiplication tensor T[i, j, k] where:
    # T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    # For matrix mult: (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i, j, k] should be 1 if this term appears, 0 otherwise
    # Actually, the tensor is: T[i, j, k] = 1 for all valid (i,j,k) in the multiplication
    # The decomposition must satisfy: sum_r U[i,r]*V[j,r]*W[k,r] = 1 for all i,j,k
    # Wait, that's not right either. Let me think more carefully.
    
    # The matrix multiplication tensor M has entries M[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # The decomposition requires: sum_r U[i,r] * V[j,r] * W[k,r] = M[i,j,k] = 1
    # No wait, that's also not right. The tensor for matrix multiplication is:
    # T[i, j, k] where i indexes rows of A, j indexes columns of A / rows of B, k indexes columns of B
    # T[i,j,k] = 1 for all valid indices
    # And the decomposition: sum_r U[i,r] * V[j,r] * W[k,r] = T[i,j,k]
    
    # But actually, looking at the W convention: W[k*n+i, r]
    # So W is indexed by (k, i) where k is the column of B and i is the row of A
    # This means W combines the output index. Let me re-examine.
    
    # U shape: (n*m, rank) - indexed by (i, j) where i is row of A, j is col of A
    # V shape: (m*p, rank) - indexed by (j, k) where j is row of B, k is col of B
    # W shape: (p*n, rank) - indexed by (k, i) where k is col of B, i is row of A
    
    # The tensor equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U_arr[u_idx, :], V_arr[v_idx, :] * W_arr[w_idx, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_222(U, V, W, n, m, p, col_offset, k0, k1):
    """Add Strassen 2x2x2 terms starting at col_offset for column pair (k0, k1)."""
    col = col_offset
    # M1 = (A00 + A11) * (B[0,k0] + B[1,k1])
    U[0][col] += 1
    U[3][col] += 1
    V[0 * p + k0][col] += 1
    V[1 * p + k1][col] += 1
    W[k0 * n + 0][col] += 1
    W[k1 * n + 1][col] += 1
    col += 1
    
    # M2 = (A10 + A11) * B[0,k0]
    U[2][col] += 1
    U[3][col] += 1
    V[0 * p + k0][col] += 1
    W[k0 * n + 1][col] += 1
    W[k1 * n + 1][col] += -1
    col += 1
    
    # M3 = A00 * (B[0,k1] - B[1,k1])
    U[0][col] += 1
    V[0 * p + k1][col] += 1
    V[1 * p + k1][col] += -1
    W[k1 * n + 0][col] += 1
    W[k1 * n + 1][col] += 1
    col += 1
    
    # M4 = A11 * (B[1,k0] - B[0,k0])
    U[3][col] += 1
    V[1 * p + k0][col] += 1
    V[0 * p + k0][col] += -1
    W[k0 * n + 0][col] += 1
    W[k0 * n + 1][col] += 1
    col += 1
    
    # M5 = (A00 + A01) * B[1,k1]
    U[0][col] += 1
    U[1][col] += 1
    V[1 * p + k1][col] += 1
    W[k0 * n + 0][col] += -1
    W[k1 * n + 0][col] += 1
    col += 1
    
    # M6 = (A10 - A00) * (B[0,k0] + B[0,k1])
    U[2][col] += 1
    U[0][col] += -1
    V[0 * p + k0][col] += 1
    V[0 * p + k1][col] += 1
    W[k1 * n + 1][col] += 1
    col += 1
    
    # M7 = (A01 - A11) * (B[1,k0] + B[1,k1])
    U[1][col] += 1
    U[3][col] += -1
    V[1 * p + k0][col] += 1
    V[1 * p + k1][col] += 1
    W[k0 * n + 0][col] += 1
    col += 1
    
    return col

def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """Try to find linear dependencies among the rank-1 terms to reduce rank.
    
    We look for a subset of columns that can be eliminated by expressing them
    as linear combinations of others while preserving the tensor identity.
    
    Approach: Check if any column is a linear combination of others in the
    "flattened" tensor space. The tensor is determined by the 28 rank-1 terms.
    If we can find a relation sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0, we can
    eliminate one term.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The tensor T has dimensions (n*m) x (m*p) x (p*n) = 4 x 16 x 16
    # Each rank-1 term contributes u_r ⊗ v_r ⊗ w_r
    # Flatten each term to a vector of size 4*16*16 = 1024
    # Check if these 28 vectors are linearly dependent
    
    dim = (n * m) * (m * p) * (p * n)
    terms = np.zeros((rank, dim))
    for r in range(rank):
        u = U_arr[:, r]
        v = V_arr[:, r]
        w = W_arr[:, r]
        outer = np.einsum('i,j,k->ijk', u, v, w)
        terms[r, :] = outer.flatten()
    
    # Find the rank of this 28 x 1024 matrix
    # If rank < 28, there are linear dependencies
    tol = 1e-10
    qr = np.linalg.qr(terms.T)  # 1024 x 28
    R = qr[1]
    # Diagonal of R gives pivots
    diag = np.abs(np.diag(R))
    # Count significant pivots
    significant = diag > tol
    matrix_rank = np.sum(significant)
    
    if matrix_rank >= rank:
        # No linear dependencies found, can't reduce
        return None
    
    # Try to find a specific null vector
    # Use SVD to find the null space
    U_svd, s_vals, Vt_svd = np.linalg.svd(terms, full_matrices=False)
    # The smallest singular values correspond to null space directions
    # Find vectors in the null space
    null_mask = s_vals < tol
    if not np.any(null_mask):
        return None
    
    # Get null space vectors
    null_vecs = Vt_svd[null_mask, :]
    
    # For each null vector, try to eliminate one term
    # A null vector c means sum_r c_r * term_r = 0
    # We can eliminate term r if c_r != 0 by expressing term_r = -sum_{s!=r} (c_s/c_r) * term_s
    # But we need the resulting U, V, W to have valid entries (integers or half-integers)
    
    best_result = None
    for nv_idx in range(null_vecs.shape[0]):
        c = null_vecs[nv_idx, :]
        # Find the largest component to eliminate
        max_idx = np.argmax(np.abs(c))
        if abs(c[max_idx]) < tol:
            continue
        
        # Express term max_idx in terms of others
        # New terms: for r != max_idx, new_coeff[r] = c[r] / c[max_idx]
        # We add new terms: for r != max_idx, add (c[r]/c[max_idx]) * term_r
        # And remove term max_idx
        
        new_rank = rank - 1
        new_U = [row[:] for row in U]
        new_V = [row[:] for row in V]
        new_W = [row[:] for row in W]
        
        # Remove column max_idx by shifting
        for r in range(max_idx, rank - 1):
            new_U[r] = new_U[r + 1]
            new_V[r] = new_V[r + 1]
            new_W[r] = new_W[r + 1]
        # Clear last row
        new_U[rank-1] = [0] * rank
        new_V[rank-1] = [0] * rank
        new_W[rank-1] = [0] * rank
        
        # Now add the linear combination
        # For each r != max_idx (in original indexing), add (c[r]/c[max_idx]) * term_r
        # But we need to be careful: the terms are rank-1, and we're adding scaled versions
        # The issue: (c[r]/c[max_idx]) * (u_r ⊗ v_r ⊗ w_r) is still rank-1
        # So we can just scale the existing terms
        
        # Actually, this approach doesn't directly work because we need the sum to equal the tensor.
        # Let me reconsider.
        
        # Original: sum_{r=0}^{rank-1} term_r = T
        # Null relation: sum_{r=0}^{rank-1} c_r * term_r = 0
        # So: term_{max} = -sum_{r != max} (c_r / c_max) * term_r
        # Therefore: T = sum_{r != max} term_r + term_max
        #            = sum_{r != max} term_r - sum_{r != max} (c_r / c_max) * term_r
        #            = sum_{r != max} (1 - c_r/c_max) * term_r
        # Wait, that's not right either. Let me redo.
        
        # T = sum_{r=0}^{rank-1} term_r
        # 0 = sum_{r=0}^{rank-1} c_r * term_r => term_max = -sum_{r != max} (c_r/c_max) * term_r
        # T = sum_{r != max} term_r + term_max
        #   = sum_{r != max} term_r - sum_{r != max} (c_r/c_max) * term_r
        #   = sum_{r != max} (1 - c_r/c_max) * term_r
        
        # So the new decomposition has rank-1 terms: (1 - c_r/c_max) * term_r for r != max
        # Each is still rank-1 (scalar multiple of a rank-1 tensor)
        
        new_U2 = [[0] * rank for _ in range(n * m)]
        new_V2 = [[0] * rank for _ in range(m * p)]
        new_W2 = [[0] * rank for _ in range(p * n)]
        
        new_col = 0
        for r in range(rank):
            if r == max_idx:
                continue
            scale = 1.0 - c[r] / c[max_idx]
            # Check if scale is close to 0 (then this term vanishes)
            if abs(scale) < tol:
                continue
            for i in range(n * m):
                new_U2[i][new_col] = U_arr[i, r] * scale
            for j in range(m * p):
                new_V2[j][new_col] = V_arr[j, r] * scale
            for k in range(p * n):
                new_W2[k][new_col] = W_arr[k, r] * scale
            new_col += 1
        
        if new_col == 0:
            continue
        
        # Trim
        new_U3 = [row[:new_col] for row in new_U2]
        new_V3 = [row[:new_col] for row in new_V2]
        new_W3 = [row[:new_col] for row in new_W2]
        
        # Verify
        if _verify_decomposition(new_U3, new_V3, new_W3, n, m, p):
            if best_result is None or new_col < best_result[0]:
                best_result = (new_col, new_U3, new_V3, new_W3)
        
        if time.time() > deadline:
            break
    
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n*m = 4, m*p = 16, p*n = 16
    
    # Strassen-based decomposition: 4 blocks of 2x2, each with 7 terms = 28 rank
    # This is optimal as M_2,2,8 is a direct sum of two M_2,2,4 tensors,
    # and M_2,2,4 is a direct sum of two M_2,2,2 tensors (rank 7).
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(4):  # 4 blocks
        k0 = 2 * b      # first column of this block
        k1 = 2 * b + 1  # second column of this block
        
        col = _strassen_222(U, V, W, n, m, p, col, k0, k1)
    
    # Verify the base decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback: this shouldn't happen, but just in case
        pass
    
    # Try to reduce rank via linear dependency elimination
    if time.time() < deadline - 10:
        result = _try_reduce_rank(U, V, W, n, m, p, deadline - 5)
        if result is not None:
            new_rank, new_U, new_V, new_W = result
            if new_rank < rank:
                U, V, W = new_U, new_V, new_W
                rank = new_rank
    
    return {"U": U, "V": V, "W": W}
