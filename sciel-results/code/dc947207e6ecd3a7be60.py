"""Matrix multiplication 2x4 x 4x4 tensor decomposition (rank 28 via Strassen on 2x2 blocks)."""

import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def _build_strassen_rank28():
    """Build the rank-28 Strassen decomposition deterministically.
    
    A (2x4) = [A1 | A2] where A1, A2 are 2x2
    B (4x4) = [[B11, B12], [B21, B22]] where each Bi is 2x2
    C (2x4) = [C1 | C2] where C1 = A1*B11 + A2*B21, C2 = A1*B12 + A2*B22
    
    Each 2x2*2x2 product uses Strassen (rank 7), so total rank = 4*7 = 28.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)  # 8 x 28
    V = np.zeros((m * p, rank), dtype=float)  # 16 x 28
    W = np.zeros((p * n, rank), dtype=float)  # 8 x 28

    # Strassen 2x2x2 decomposition:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    # c11 = m1+m4-m5+m7
    # c12 = m3+m5
    # c21 = m2+m4
    # c22 = m1-m2+m3+m6
    
    col = 0
    for b_col_blk in range(2):  # B column block: 0 -> cols 0,1; 1 -> cols 2,3
        for a_blk in range(2):  # A column block / B row block: 0 -> cols/rows 0,1; 1 -> cols/rows 2,3
            for s in range(7):
                c = col
                j0 = 2 * a_blk
                j1 = 2 * a_blk + 1
                k0 = 2 * b_col_blk
                k1 = 2 * b_col_blk + 1
                
                # U contributions: A[i, j] where i in {0,1}, j in {j0, j1}
                # U index: i * 4 + j
                U_contrib = {
                    0: [(0, j0, 1), (1, j1, 1)],       # a11+a22
                    1: [(1, j0, 1), (1, j1, 1)],       # a21+a22
                    2: [(0, j0, 1)],                    # a11
                    3: [(1, j1, 1)],                    # a22
                    4: [(0, j0, 1), (0, j1, 1)],       # a11+a12
                    5: [(1, j0, 1), (0, j0, -1)],      # a21-a11
                    6: [(0, j1, 1), (1, j1, -1)],      # a12-a22
                }
                for (i, j, coeff) in U_contrib[s]:
                    U[i * 4 + j, c] += coeff

                # V contributions: B[j, k] where j in {j0, j1}, k in {k0, k1}
                # V index: j * 4 + k
                V_contrib = {
                    0: [(j0, k0, 1), (j1, k1, 1)],     # b11+b22
                    1: [(j0, k0, 1)],                   # b11
                    2: [(j0, k1, 1), (j1, k1, -1)],    # b12-b22
                    3: [(j1, k0, 1), (j0, k0, -1)],    # b21-b11
                    4: [(j1, k1, 1)],                   # b22
                    5: [(j0, k0, 1), (j0, k1, 1)],     # b11+b12
                    6: [(j1, k0, 1), (j1, k1, 1)],     # b21+b22
                }
                for (j, k, coeff) in V_contrib[s]:
                    V[j * 4 + k, c] += coeff

                # W contributions: C[i, k] where i in {0,1}, k in {k0, k1}
                # W index: k * 2 + i (trace convention W[k*n+i, r])
                W_contrib = {
                    0: {(0, k0): 1, (1, k1): 1},       # c11 gets m1, c22 gets m1
                    1: {(1, k0): 1, (1, k1): -1},      # c21 gets m2, c22 gets -m2
                    2: {(0, k1): 1, (1, k1): 1},       # c12 gets m3, c22 gets m3
                    3: {(0, k0): 1, (1, k0): 1},       # c11 gets m4, c21 gets m4
                    4: {(0, k0): -1, (0, k1): 1},      # c11 gets -m5, c12 gets m5
                    5: {(1, k1): 1},                   # c22 gets m6
                    6: {(0, k0): 1},                   # c11 gets m7
                }
                for (i_loc, k_loc), coeff in W_contrib[s].items():
                    W[k_loc * 2 + i_loc, c] += coeff

                col += 1

    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Check if U, V, W correctly decompose the matmul tensor.
    
    T[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] should equal 1 for all (i,k,j).
    """
    rank = U.shape[1]
    for i in range(n):
        for k in range(m):
            for j in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _attempt_rank_reduction(U, V, W, n, m, p):
    """
    Attempt to reduce the rank of the decomposition by checking for linear dependencies
    among the rank-1 tensor components.
    
    Returns (U_reduced, V_reduced, W_reduced) if rank is reduced, else original.
    """
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W

    # Flatten each rank-1 tensor u_r \otimes v_r \otimes w_r into a vector of size n*m*p
    # Note: The tensor T is defined such that T[i, k, j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # So the flattened vector for term r is:
    # vec_r[index] = U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # where index = i*(m*p) + k*p + j
    
    size = n * m * p
    tensors = np.zeros((rank, size))
    
    for r in range(rank):
        # Vectorize the outer product
        # We can do this efficiently using broadcasting or loops
        # Given small sizes, loops are fine
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    idx = i * (m * p) + k * p + j
                    tensors[r, idx] = U[i * m + k, r] * V[k * p + j, r] * W[j * n + i, r]

    # Check for linear dependencies
    # We want to see if any column r is in the span of the others.
    # This is equivalent to checking if the rank of the matrix of all tensors is < rank.
    # If rank(tensors) < rank, there is a dependency.
    
    # Use SVD to find the rank and potentially a dependency vector
    try:
        U_svd, s, Vt = np.linalg.svd(tensors, full_matrices=False)
        # Count non-zero singular values
        tol = 1e-10
        num_rank = np.sum(s > tol)
        
        if num_rank < rank:
            # There is a dependency.
            # We can find a null vector in the rowspace of 'tensors'
            # The null space of tensors.T (or the right singular vectors corresponding to zero singular values)
            # gives coefficients c such that sum_r c_r * tensors[r] = 0.
            
            # The null space is spanned by the last (rank - num_rank) rows of Vt
            null_space = Vt[num_rank:]
            
            # We want to find a column r to remove.
            # We look for a null vector c where one component c_r is non-zero (preferably large)
            # and we can eliminate that column.
            
            best_col_to_remove = -1
            best_coeff = 0.0
            best_null_vec = None
            
            for vec in null_space:
                # Find the index with the largest absolute value
                max_idx = np.argmax(np.abs(vec))
                max_val = np.abs(vec[max_idx])
                if max_val > best_coeff:
                    best_coeff = max_val
                    best_col_to_remove = max_idx
                    best_null_vec = vec
            
            if best_col_to_remove != -1:
                # We have a dependency: sum_r best_null_vec[r] * T_r = 0
                # So T_best = - (1/best_null_vec[best]) * sum_{r != best} best_null_vec[r] * T_r
                # We can remove column 'best_col_to_remove' and adjust the others?
                # Actually, the decomposition is sum_r T_r = T_total.
                # If sum_r c_r T_r = 0, then sum_r T_r = sum_r (T_r - c_r * k) ? No.
                # The decomposition is just a set of terms.
                # If T_best = sum_{r != best} c'_r T_r, then the sum of all 28 terms is equal to the sum of the other 27 terms (with modified coefficients).
                # Wait, the decomposition is T = sum_{r=0}^{R-1} T_r.
                # If there is a linear dependency sum_{r} c_r T_r = 0, it means the vectors are dependent.
                # It does NOT mean one T_r is a linear combination of the others *in the set of terms*.
                # It means the set of vectors spans a lower dimensional space.
                # However, if T_best = sum_{r != best} k_r T_r, then T = T_best + sum_{r != best} T_r = sum_{r != best} (1 + k_r) T_r.
                # So yes, we can represent the same tensor with R-1 terms.
                
                # Let's verify this logic.
                # We found c such that sum_r c_r T_r = 0.
                # Let r0 = best_col_to_remove.
                # c_r0 T_r0 = - sum_{r != r0} c_r T_r
                # T_r0 = - (1/c_r0) sum_{r != r0} c_r T_r
                # So T_total = T_r0 + sum_{r != r0} T_r = sum_{r != r0} (1 - c_r/c_r0) T_r.
                # This is a valid decomposition with rank R-1.
                
                # We need to update U, V, W to reflect the new coefficients.
                # The new term for r (r != r0) is (1 - c_r/c_r0) * T_r.
                # Since T_r is rank 1, scaling it just scales U[:, r], V[:, r], W[:, r] by the same factor.
                
                new_U = np.delete(U, best_col_to_remove, axis=1)
                new_V = np.delete(V, best_col_to_remove, axis=1)
                new_W = np.delete(W, best_col_to_remove, axis=1)
                
                # Scale the remaining columns
                # c is the null vector. c[r0] is the pivot.
                # factor_r = 1 - c_r / c_r0
                for r_new in range(new_U.shape[1]):
                    r_old = r_new if r_new < best_col_to_remove else r_new + 1
                    factor = 1.0 - best_null_vec[r_old] / best_null_vec[best_col_to_remove]
                    new_U[:, r_new] *= factor
                    new_V[:, r_new] *= factor
                    new_W[:, r_new] *= factor
                
                return new_U, new_V, new_W
                
    except np.linalg.LinAlgError:
        pass
        
    return U, V, W

def construct():
    """
    Return a rank-minimal exact decomposition of the 2x4 * 4x4 matrix multiplication tensor.
    Uses Strassen's algorithm on 2x2 blocks (rank 28) and attempts to reduce rank.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    start_time = time.time()
    deadline = start_time + 100  # 100s limit, keep margin before 120s
    
    # Build the base rank-28 Strassen decomposition
    U, V, W = _build_strassen_rank28()
    
    # Verify it's correct
    assert _verify_decomposition(U, V, W, n, m, p), "Base rank-28 decomposition failed verification"
    
    # Attempt to reduce rank
    U, V, W = _attempt_rank_reduction(U, V, W, n, m, p)
    
    # Verify the reduced decomposition (if it changed)
    # Note: _verify_decomposition is slow, so we only run it if rank actually changed or as a final check
    # For now, we trust the linear algebra reduction, but a final check is good for safety.
    # If it fails, we return the original rank 28.
    
    original_U, original_V, original_W = _build_strassen_rank28()
    
    # Only re-verify if we think we reduced it
    if U.shape[1] < original_U.shape[1]:
        if not _verify_decomposition(U, V, W, n, m, p):
            # Reduction failed, revert
            U, V, W = original_U, original_V, original_W
    
    # Return the result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
