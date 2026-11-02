"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W):
    """Check that the decomposition exactly represents matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    if not np.allclose(result, 1.0, atol=1e-10):
        return False
    return True

def _strassen_2x2_factors():
    """
    Return the 7 rank-1 factors for 2x2*2x2 matrix multiplication.
    Each factor is (u_vec, v_vec, w_vec) where:
      u_vec: 4-element vector for A (row-major: A00, A01, A10, A11)
      v_vec: 4-element vector for B (row-major: B00, B01, B10, B11)
      w_vec: 4-element vector for C (row-major: C00, C01, C10, C11)
    """
    # Standard Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M3 - M4 + M6
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    factors = []

    # M1 = (A00+A11)(B00+B11)
    u = [1, 0, 0, 1]
    v = [1, 0, 0, 1]
    w = [1, 0, 0, 1]
    factors.append((u, v, w))

    # M2 = (A10+A11) * B00
    u = [0, 0, 1, 1]
    v = [1, 0, 0, 0]
    w = [0, 0, 1, -1]
    factors.append((u, v, w))

    # M3 = A00 * (B01-B11)
    u = [1, 0, 0, 0]
    v = [0, 1, 0, -1]
    w = [0, 1, 0, 1]
    factors.append((u, v, w))

    # M4 = A11 * (B10-B00)
    u = [0, 0, 0, 1]
    v = [-1, 0, 1, 0]
    w = [1, 0, 1, 0]
    factors.append((u, v, w))

    # M5 = (A00+A01) * B11
    u = [1, 1, 0, 0]
    v = [0, 0, 0, 1]
    w = [-1, 1, 0, 0]
    factors.append((u, v, w))

    # M6 = (A10-A00) * (B00+B01)
    u = [-1, 0, 1, 0]
    v = [1, 1, 0, 0]
    w = [0, 0, 0, 1]
    factors.append((u, v, w))

    # M7 = (A01-A11) * (B10+B11)
    u = [0, 1, 0, -1]
    v = [0, 0, 1, 1]
    w = [1, 0, 0, 0]
    factors.append((u, v, w))

    return factors

def _try_strassen_4x4_4x8():
    """
    Construct a rank-98 decomposition for 4x4 * 4x8 using Strassen.
    
    Strategy: Split the 4x8 matrix B into two 4x4 blocks B1 and B2.
    Compute C1 = A*B1 and C2 = A*B2, each using Strassen (rank 49).
    Total rank = 98.
    
    Each 4x4*4x4 Strassen decomposition uses 7 top-level Strassen products,
    each of which is a 2x2*2x2 multiplication, itself decomposed with 7 scalar
    Strassen products. Total: 7 * 7 = 49 per subproblem.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Get the 2x2 Strassen factors
    s2_factors = _strassen_2x2_factors()
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # (16, 98)
    V = [[0] * rank for _ in range(m * p)]  # (32, 98)
    W = [[0] * rank for _ in range(p * n)]  # (32, 98)
    
    # For each of the two subproblems (t=0 for B1, t=1 for B2):
    for t in range(2):
        # For each of the 7 top-level Strassen products (s=0..6):
        for s in range(7):
            # For each of the 7 scalar Strassen products (r=0..6):
            for r in range(7):
                col = t * 49 + s * 7 + r
                
                # The top-level Strassen product s involves certain A blocks and B blocks.
                # We need to determine the U and V coefficients at the block level,
                # then combine with the scalar-level coefficients.
                
                # Top-level Strassen for 4x4*4x4 (same structure as 2x2*2x2 but with 2x2 blocks):
                # The 7 products are:
                # s=0: (A11+A22)(B11+B22)
                # s=1: (A21+A22)B11
                # s=2: A11(B12-B22)
                # s=3: A22(B21-B11)
                # s=4: (A11+A12)B22
                # s=5: (A21-A11)(B11+B12)
                # s=6: (A12-A22)(B21+B22)
                
                # For product s, the A blocks involved and their coefficients:
                # Using the same pattern as the 2x2 Strassen U factors:
                s2_u_block = s2_factors[s][0]  # [coeff for A00, A01, A10, A11] in block terms
                s2_v_block = s2_factors[s][1]  # [coeff for B00, B01, B10, B11] in block terms
                
                # For the scalar Strassen product r:
                s2_u_scalar = s2_factors[r][0]  # [coeff for element (0,0), (0,1), (1,0), (1,1)]
                s2_v_scalar = s2_factors[r][1]
                s2_w_scalar = s2_factors[r][2]
                
                # U contributions:
                # A block (bi, bj) where bi=0,1 (row block), bj=0,1 (col block)
                # Block (bi, bj) corresponds to index bi*2+bj in the 4-element vector
                for bi in range(2):
                    for bj in range(2):
                        block_idx = bi * 2 + bj
                        c_block = s2_u_block[block_idx]
                        if c_block == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                scalar_idx = ei * 2 + ej
                                c_scalar = s2_u_scalar[scalar_idx]
                                if c_scalar == 0:
                                    continue
                                i = 2 * bi + ei
                                j = 2 * bj + ej
                                U[i * m + j][col] += c_block * c_scalar
                
                # V contributions:
                # B block (bi, bj) for subproblem t
                for bi in range(2):
                    for bj in range(2):
                        block_idx = bi * 2 + bj
                        c_block = s2_v_block[block_idx]
                        if c_block == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                scalar_idx = ei * 2 + ej
                                c_scalar = s2_v_scalar[scalar_idx]
                                if c_scalar == 0:
                                    continue
                                j = 2 * bi + ei  # B row
                                k = 4 * t + 2 * bj + ej  # B col (offset by 4*t)
                                V[j * p + k][col] += c_block * c_scalar
                
                # W contributions:
                # The output C for subproblem t. The top-level Strassen product s
                # contributes to certain C blocks with certain coefficients.
                # The scalar Strassen product r determines which elements of those
                # C blocks are affected.
                
                # We need to determine which C blocks are affected by product s
                # and with what coefficients. This comes from the W vector of the
                # top-level Strassen factor s, but applied at the block level.
                
                # Actually, the W factor for the top-level Strassen product s tells us
                # which C blocks are affected. But we're using the same s2_factors for
                # both levels. The W vector of s2_factors[s] gives the block-level
                # C coefficients: [C00_coeff, C01_coeff, C10_coeff, C11_coeff].
                
                s2_w_block = s2_factors[s][2]  # [coeff for C00, C01, C10, C11] in block terms
                
                for ci in range(2):
                    for cj in range(2):
                        block_idx = ci * 2 + cj
                        c_block = s2_w_block[block_idx]
                        if c_block == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                scalar_idx = ei * 2 + ej
                                c_scalar = s2_w_scalar[scalar_idx]
                                if c_scalar == 0:
                                    continue
                                i = 2 * ci + ei  # C row
                                k = 4 * t + 2 * cj + ej  # C col (offset by 4*t)
                                W[k * n + i][col] += c_block * c_scalar
    
    return {"U": U, "V": V, "W": W}

def _schoolbook():
    """Return the trivial rank n*m*p decomposition."""
    n, m, p = DIMENSIONS
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

def _reduce_rank(U, V, W, deadline=None):
    """
    Try to reduce the rank by eliminating linearly dependent rank-1 factors.
    A factor r is redundant if there exist scalars a_r (r in subset) such that
    for all (i,j,k): sum_{r in subset} a_r * U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 0
    and a_r != 0 for at least one r in subset.
    
    We check: for each pair of columns (r1, r2), check if column r2 is in the
    span of the other columns in a tensor sense. More practically:
    - Form the matrix of all rank-1 tensors flattened
    - Find the rank of this matrix
    - If rank < number of columns, some columns are redundant
    
    Simpler approach: check if removing column r still gives a valid decomposition.
    If yes, column r is redundant.
    
    For a more aggressive reduction: try to find linear combinations.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    # Flatten each rank-1 factor into a vector of size n*m*p
    # Factor r contributes: U[:,:,r] * V[:,:,r] * W[:,:,r] (outer product)
    # The full tensor is sum_r of these outer products
    # We need sum_r u_r ⊗ v_r ⊗ w_r = all-ones tensor
    
    # Check: is column r redundant?
    # If we remove column r, does the remaining sum still equal the target?
    # Target = sum_all factors
    # Without r: sum_all - factor_r
    # So factor_r must be zero for r to be removable... unless other columns compensate.
    
    # Better: find the actual rank of the set of rank-1 tensors.
    # Build matrix M where M[:, r] = flatten(U[:,:,r] * V[:,:,r] * W[:,:,r])
    # Then find rank(M). If rank(M) < rank, we can reduce.
    
    # But we need to find explicit linear dependencies to actually reduce.
    
    # Approach: 
    # 1. Build the n*m*p x rank matrix of flattened factors
    # 2. Find its rank via SVD
    # 3. Identify null space vectors
    # 4. Use null space to eliminate columns
    
    flat_factors = np.zeros((n*m*p, rank))
    for r in range(rank):
        flat_factors[:, r] = (U3[:,:,r] * V3[:,:,r] * W3[:,:,r]).flatten()
    
    # Compute SVD to find rank
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(flat_factors, full_matrices=False)
        # Rank is number of significant singular values
        tol = 1e-10
        num_rank = np.sum(s_vals > tol)
    except Exception:
        return U, V, W, rank
    
    if num_rank >= rank:
        # No reduction possible
        return U, V, W, rank
    
    # Find null space: columns of Vt_svd where s_vals is small
    null_indices = [i for i in range(len(s_vals)) if s_vals[i] <= tol]
    if not null_indices:
        return U, V, W, rank
    
    # For each null space vector, we have a linear combination of columns that sums to zero
    # v = Vt_svd[null_idx, :] means sum_r v[r] * factor_r = 0
    # We can use this to eliminate one column.
    
    # Strategy: greedily eliminate columns using null space vectors
    U_new = U_arr.copy()
    V_new = V_arr.copy()
    W_new = W_arr.copy()
    
    # Track which columns are still active
    active = list(range(rank))
    removed = set()
    
    # For each null space vector, try to eliminate a column
    for ni in null_indices:
        if deadline is not None and time.time() > deadline:
            break
        v = Vt_svd[ni, :]
        # Find the column with largest absolute coefficient in v
        # We'll express that column as a combination of others
        abs_v = np.abs(v)
        abs_v[list(removed)] = 0
        if np.max(abs_v) < 1e-15:
            continue
        target_col = int(np.argmax(abs_v))
        if target_col in removed:
            continue
        
        # v[target_col] * factor[target_col] = -sum_{r != target_col} v[r] * factor[r]
        # So factor[target_col] = -sum_{r != target_col} (v[r]/v[target_col]) * factor[r]
        # This means we can replace column target_col with a linear combination of other columns.
        # But in our representation, each column is a rank-1 tensor U[:,r]*V[:,r]*W[:,r].
        # A linear combination of rank-1 tensors is NOT generally rank-1.
        # So we can't simply replace one column with a combination.
        
        # Instead, we need a different approach:
        # The null space tells us the factors are linearly dependent.
        # We can perform Gaussian elimination on the factor matrix.
        
        # Actually, the key insight: if factors are linearly dependent, we can
        # re-express the decomposition with fewer terms by combining them.
        # But combining two rank-1 terms generally gives a rank-2 term.
        # So linear dependence of the outer products doesn't directly help.
        
        # Let me reconsider. The condition for a valid decomposition is:
        # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all (i,j,k)
        # This is equivalent to: the tensor T = sum_r u_r ⊗ v_r ⊗ w_r equals the all-ones tensor.
        # If the set {u_r ⊗ v_r ⊗ w_r} is linearly dependent, there exist a_r not all zero
        # such that sum_r a_r * (u_r ⊗ v_r ⊗ w_r) = 0.
        # Then sum_r (a_r) * (u_r ⊗ v_r ⊗ w_r) + sum_r (1) * (u_r ⊗ v_r ⊗ w_r) = T
        # But this doesn't reduce the number of terms unless we can absorb the a_r into the coefficients.
        
        # Wait - we CAN'T absorb them because the tensor is a sum of OUTER products,
        # not a linear combination with scalar coefficients on the outside.
        
        # So linear dependence of the outer product tensors doesn't help directly.
        # The rank of the tensor is the minimum number of rank-1 tensors needed.
        # If our current decomposition has rank R, the tensor rank is at most R.
        # To reduce, we need to find a different set of rank-1 tensors.
        
        # This is much harder. Let's try a different approach:
        # Check if any single column can be removed (i.e., the tensor without that column still works).
        # This would mean that column's contribution is zero, which is unlikely.
        
        # Actually, let's just check: for each column r, compute the tensor without column r
        # and see if it still equals the all-ones tensor. If yes, column r is redundant.
        pass
    
    # Simpler check: for each column, see if removing it still gives a valid result
    U_list = [list(row) for row in U_arr]
    V_list = [list(row) for row in V_arr]
    W_list = [list(row) for row in W_arr]
    
    for r in range(rank):
        if deadline is not None and time.time() > deadline:
            break
        # Check if column r is all zeros
        if np.allclose(U_arr[:, r], 0) and np.allclose(V_arr[:, r], 0) and np.allclose(W_arr[:, r], 0):
            # This column is zero, remove it
            U_new = np.delete(U_new, r, axis=1)
            V_new = np.delete(V_new, r, axis=1)
            W_new = np.delete(W_new, r, axis=1)
            # Need to reindex... this is getting complicated
            # Let's use a different approach with lists
            pass
    
    # Let's try a cleaner approach: build the decomposition from scratch with only
    # linearly independent factors by checking each factor's contribution.
    
    # Actually, the most practical approach: try removing each column one at a time
    # and verify. If valid, keep it removed.
    
    U_rows = [list(U_arr[i]) for i in range(len(U_arr))]
    V_rows = [list(V_arr[i]) for i in range(len(V_arr))]
    W_rows = [list(W_arr[i]) for i in range(len(W_arr))]
    
    new_rank = rank
    columns_to_keep = list(range(rank))
    
    for r in range(rank):
        if deadline is not None and time.time() > deadline:
            break
        # Try removing column r
        test_U = [row[:r] + row[r+1:] for row in U_rows]
        test_V = [row[:r] + row[r+1:] for row in V_rows]
        test_W = [row[:r] + row[r+1:] for row in W_rows]
        
        if _verify(test_U, test_V, test_W):
            # Column r is redundant
            U_rows = test_U
            V_rows = test_V
            W_rows = test_W
            new_rank -= 1
    
    # Try removing pairs of columns that are linearly dependent
    # Check if for some pair (r1, r2), the two factors are proportional
    # (i.e., U[:,r1]*V[:,r1]*W[:,r1] = c * U[:,r2]*V[:,r2]*W[:,r2])
    # In that case, they can be combined into one factor.
    # Actually, if two rank-1 tensors are proportional, their sum is also rank-1.
    
    # More generally, try to find if any two columns can be merged.
    # Two factors r1 and r2 can be merged if there exist scalars such that
    # the sum of the two outer products is itself a rank-1 tensor.
    # This is a rare condition. Let's check a simpler case:
    # If U[:,r1] is proportional to U[:,r2], V[:,r1] proportional to V[:,r2],
    # and W[:,r1] proportional to W[:,r2], then the two factors are proportional.
    
    current_rank = new_rank
    for r1 in range(current_rank):
        if deadline is not None and time.time() > deadline:
            break
        for r2 in range(r1+1, current_rank):
            if deadline is not None and time.time() > deadline:
                break
            # Check if factors r1 and r2 are proportional
            # Compare U columns
            if not np.allclose(U_arr[:, r1], 0) and not np.allclose(U_arr[:, r2], 0):
                ratio_u = U_arr[:, r1] / U_arr[:, r2]
                # Check if ratio is constant
                nonzero = np.abs(U_arr[:, r2]) > 1e-15
                if np.allclose(ratio_u[nonzero], ratio_u[nonzero][0], atol=1e-10):
                    # Check V
                    if not np.allclose(V_arr[:, r1], 0) and not np.allclose(V_arr[:, r2], 0):
                        ratio_v = V_arr[:, r1] / V_arr[:, r2]
                        nonzero_v = np.abs(V_arr[:, r2]) > 1e-15
                        if np.allclose(ratio_v[nonzero_v], ratio_v[nonzero_v][0], atol=1e-10):
                            # Check W
                            if not np.allclose(W_arr[:, r1], 0) and not np.allclose(W_arr[:, r2], 0):
                                ratio_w = W_arr[:, r1] / W_arr[:, r2]
                                nonzero_w = np.abs(W_arr[:, r2]) > 1e-15
                                if np.allclose(ratio_w[nonzero_w], ratio_w[nonzero_w][0], atol=1e-10):
                                    # Factors are proportional, can merge
                                    # Combine: new_factor = factor_r1 + factor_r2
                                    # But we need to express as a single rank-1 factor
                                    # U_new = U_r1 + U_r2 (scaled appropriately)
                                    # This is complex; skip for now
                                    pass
    
    if new_rank < rank:
        return U_rows, V_rows, W_rows, new_rank
    
    return U, V, W, rank

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Tries Strassen-based decomposition (rank 98) first, then tries to reduce rank,
    falls back to schoolbook (rank 128).
    """
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s safety margin
    
    best_result = None
    best_rank = float('inf')
    
    # Try Strassen-based decomposition
    try:
        result = _try_strassen_4x4_4x8()
        if _verify(result["U"], result["V"], result["W"]):
            best_result = result
            best_rank = len(result["U"][0])
            
            # Try to reduce rank
            if time.time() < deadline:
                U_red, V_red, W_red, new_rank = _reduce_rank(
                    result["U"], result["V"], result["W"], deadline=deadline
                )
                if new_rank < best_rank:
                    if _verify(U_red, V_red, W_red):
                        best_result = {"U": U_red, "V": V_red, "W": W_red}
                        best_rank = new_rank
    except Exception:
        pass
    
    if best_result is not None:
        return best_result
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
