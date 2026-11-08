import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def _build_rank17():
    """Build the known rank-17 decomposition (Strassen 2x2 core + schoolbook cross-terms)."""
    n, m, p = 2, 3, 3
    rank = 17
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Strassen block for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # r=0: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # r=1: (A10+A11)B00
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    # r=2: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    # r=3: A11(B10-B00)
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    # r=4: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    # r=5: (A10-A00)(B00+B01)
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    # r=6: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1

    # Schoolbook: A[0:2,2] * B[2,0:2] -> C[0:2,0:2]
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1

    # Schoolbook: A[0:2,0:3] * B[0:3,2] -> C[0:2,2]
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    U[2, 13] = 1; V[8, 13] = 1; W[4, 13] = 1
    U[3, 14] = 1; V[2, 14] = 1; W[5, 14] = 1
    U[4, 15] = 1; V[5, 15] = 1; W[5, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1

    return U, V, W


def _verify(U, V, W, n=2, m=3, p=3):
    """Check that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1 for all (i,j,k)."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0
                for r in range(rank):
                    val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                if val != 1:
                    return False
    return True


def _try_reduce_rank(U, V, W, n=2, m=3, p=3, target_rank=16, deadline=None):
    """
    Attempt to find a linear dependency among the rank-17 vectors that allows
    elimination of one term while preserving exact decomposition.
    
    We look for coefficients c_0,...,c_16 (not all zero) such that for every
    (i,j,k): sum_r c_r * U[i,j,r]*V[j,k,r]*W[k,i,r] = 0.
    If such a relation exists, we can use it to eliminate one variable from
    the system, potentially reducing the rank.
    
    A simpler approach: check if any single rank term can be expressed as a
    linear combination of others in the tensor sense.
    
    Alternative: try all pairs of terms and see if one can be absorbed.
    """
    if deadline is None:
        deadline = time.time() + 100.0
    
    rank = U.shape[1]
    
    # Build the full tensor T of shape (n*m, m*p, p*n)
    # T[u_idx, v_idx, w_idx] = sum_r U[u_idx,r]*V[v_idx,r]*W[w_idx,r]
    # We want T == 1 for all valid (i,j,k) entries.
    
    # Actually, let's try a different approach:
    # Check if the 17 rank-1 tensors are linearly dependent in the tensor space.
    # The tensor space has dimension n*m*p*n = 2*3*3*2 = 36.
    # But we only care about the 18 entries (i,j,k) with i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    # Each rank-1 term contributes a vector of length 18 (one per (i,j,k) triple).
    
    # Build matrix M of shape (18, 17) where M[(i,j,k), r] = U[i,j,r]*V[j,k,r]*W[k,i,r]
    # Then check if columns are linearly dependent.
    
    n, m, p = 2, 3, 3
    num_entries = n * m * p  # 18
    M = np.zeros((num_entries, rank), dtype=float)
    
    entry_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for r in range(rank):
                    M[entry_idx, r] = U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                entry_idx += 1
    
    # Check rank of M
    rank_M = np.linalg.matrix_rank(M, tol=1e-10)
    
    if rank_M < rank:
        # Linear dependency exists! Find it.
        # Use SVD to find null space
        U_svd, s_vals, Vt = np.linalg.svd(M, full_matrices=True)
        # Null space vectors are in the last (rank - rank_M) rows of Vt
        null_dim = rank - rank_M
        null_vectors = Vt[-null_dim:]  # shape (null_dim, rank)
        
        # Try to use the null space to eliminate one term
        # For each null vector, check if we can set one coefficient to 1 and solve
        # for the rest, then verify the reduced decomposition.
        
        # Actually, the null space tells us that sum_r c_r * (rank-1 tensor r) = 0
        # in the tensor space. This means we can replace one term by a combination
        # of others.
        
        # Try each null vector: for the term with largest |c_r|, try to eliminate it.
        best_result = None
        for nv in null_vectors:
            # Find the index with largest absolute coefficient
            max_idx = np.argmax(np.abs(nv))
            max_coeff = nv[max_idx]
            
            if abs(max_coeff) < 1e-10:
                continue
            
            # The relation is: sum_r nv[r] * T_r = 0
            # So T_max_idx = -sum_{r != max_idx} (nv[r]/nv[max_idx]) * T_r
            # This means we can remove term max_idx if the remaining terms
            # (with adjusted coefficients) still give the correct result.
            
            # But wait - this only works if the null vector has rational/half-integer
            # entries that allow exact reconstruction. Let's check if we can find
            # integer coefficients.
            
            # Scale to get integers
            # Find LCM of denominators... this is tricky. Let's try a simpler approach.
            
            # Try to find if any single term is a linear combination of others
            # with small integer coefficients.
            
            pass
        
        # Alternative approach: brute force check if any subset of 16 terms
        # can be scaled to give the correct tensor.
        # This is expensive, so let's try a smarter approach.
        
        # Check if removing each single term and adjusting others works.
        # For each term t to remove, solve for coefficients of remaining 16 terms.
        
        for remove_idx in range(rank):
            if time.time() > deadline:
                break
            
            # Build system: find c_0,...,c_16 (excluding remove_idx) such that
            # sum_{r != remove_idx} c_r * T_r = T_total - c_remove * T_remove
            # But we don't know c_remove. Instead, we want:
            # sum_{r != remove_idx} c_r * T_r = T_target (the all-ones tensor)
            
            # Build matrix of remaining terms
            remaining = [r for r in range(rank) if r != remove_idx]
            M_rem = M[:, remaining]  # shape (18, 16)
            
            # Target: the all-ones vector of length 18
            target = np.ones(num_entries)
            
            # Solve M_rem @ c = target (least squares, then check exactness)
            try:
                c, residuals, rank_sol, sv = np.linalg.lstsq(M_rem, target, rcond=1e-10)
            except:
                continue
            
            # Check if the solution is exact
            M_rem_c = M_rem @ c
            if np.allclose(M_rem_c, target, atol=1e-10):
                # Check if coefficients are "nice" (close to integers or half-integers)
                # Try to round
                c_rounded = np.round(c * 2) / 2  # round to nearest half-integer
                
                if np.allclose(M_rem @ c_rounded, target, atol=1e-10):
                    # Found a valid rank-16 decomposition!
                    new_U = U[:, remaining].copy()
                    new_V = V[:, remaining].copy()
                    new_W = W[:, remaining].copy()
                    
                    # Scale by coefficients
                    for idx, r in enumerate(remaining):
                        new_U[:, idx] *= c_rounded[idx]
                        new_V[:, idx] *= c_rounded[idx]
                        new_W[:, idx] *= c_rounded[idx]
                    
                    # Verify
                    if _verify(new_U, new_V, new_W):
                        return new_U, new_V, new_W
        
        return None
    
    return None


def _build_rank16_strassen_cross():
    """
    Try an alternative rank-16 construction using Strassen on overlapping 2x2 blocks.
    
    Idea: Use Strassen for the 2x2 submatrix (rows 0-1 of A, cols 0-1 of B) giving 7 terms.
    Then handle the remaining terms more cleverly.
    
    A is 2x3, B is 3x3.
    
    Let's partition differently:
    - Strassen on A[0:2, 0:2] * B[0:2, 0:2] -> 7 terms for C[0:2, 0:2] partial
    - Then we need to add contributions to C[0:2, 0:2] from A[0:2, 2]*B[2, 0:2] (4 terms)
    - And C[0:2, 2] from A[0:2, 0:3]*B[0:3, 2] (6 terms)
    Total: 7 + 4 + 6 = 17.
    
    To get 16, we need to save one term. Let's try using Strassen-style tricks
    on the cross terms.
    
    For C[0:2, 2]: C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2] + A[i,2]*B[2,2]
    This is a 2x3 times 3x1 product. Can we do it in 5 terms instead of 6?
    The rank of 2x3x1 matrix multiplication is at least 6 (it's just 6 scalar mults
    that can't be combined since each output depends on different inputs).
    Actually, C[0,2] = A00*B02 + A01*B12 + A02*B22
    C[1,2] = A10*B02 + A11*B12 + A12*B22
    
    Can we combine? Let M1 = (A00+A10)*B02, M2 = (A01+A11)*B12, M3 = (A02+A12)*B22
    Then C[0,2] = M1 + M2 + M3 - (A10*B02 + A11*B12 + A12*B22)
    That doesn't help directly.
    
    What about: 
    M1 = (A00+A01+A02)*(B02+B12+B22) - ... no, this gets complicated.
    
    The rank of [2x3] x [3x1] is actually 6 (it's a sum of 6 rank-1 tensors
    that are linearly independent in the tensor space).
    
    Similarly, A[0:2,2]*B[2,0:2] is 2x1 times 1x2, which has rank 4.
    
    So 7 + 4 + 6 = 17 seems tight.
    
    Let me try a different partition. What if we use Strassen on a different 2x2 block?
    """
    n, m, p = 2, 3, 3
    rank = 16
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Try: Strassen on A[0:2,0:2] * B[0:2,0:2] (7 terms)
    # Plus 9 terms for the rest, trying to be clever.
    
    # Actually, let's just return None and let the caller decide.
    return None


def construct():
    start_time = time.time()
    deadline = 120.0

    n, m, p = DIMENSIONS

    # Start with the known rank-17 decomposition
    U, V, W = _build_rank17()
    
    # Verify it's correct
    if not _verify(U, V, W):
        # Fallback to schoolbook rank 18
        rank = 18
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, r] = 1
                    V[v_idx, r] = 1
                    W[w_idx, r] = 1
                    r += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce to rank 16
    if time.time() < deadline - 10:
        result = _try_reduce_rank(U, V, W, n=n, m=m, p=p, target_rank=16, deadline=deadline - 5)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try alternative rank-16 construction
    if time.time() < deadline - 10:
        result16 = _build_rank16_strassen_cross()
        if result16 is not None:
            U16, V16, W16 = result16
            if _verify(U16, V16, W16):
                return {"U": U16.tolist(), "V": V16.tolist(), "W": W16.tolist()}
    
    # Return the rank-17 decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
