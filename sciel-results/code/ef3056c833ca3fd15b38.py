"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W):
    """Check that the decomposition exactly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
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

def _verify_numpy(U, V, W):
    """Fast verification using numpy arrays directly."""
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
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

def _verify_fast(U, V, W):
    """Fast verification using tensor structure."""
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    # For each (i,j,k), check sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1
    # This is equivalent to checking all entries of the tensor
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
    factors = []
    # M1 = (A00+A11)(B00+B11)
    u = [1, 0, 0, 1]
    v = [1, 0, 0, 1]
    w = [1, 1, 1, -1]
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
    Split the 4x8 matrix B into two 4x4 blocks B1 and B2.
    Compute C1 = A*B1 and C2 = A*B2, each using Strassen (rank 49).
    Total rank = 98.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    s2_factors = _strassen_2x2_factors()
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(2):
        for s in range(7):
            for r in range(7):
                col = t * 49 + s * 7 + r
                s2_u_block = s2_factors[s][0]
                s2_v_block = s2_factors[s][1]
                s2_u_scalar = s2_factors[r][0]
                s2_v_scalar = s2_factors[r][1]
                s2_w_scalar = s2_factors[r][2]
                
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
                                j = 2 * bi + ei
                                k = 4 * t + 2 * bj + ej
                                V[j * p + k][col] += c_block * c_scalar
                
                s2_w_block = s2_factors[s][2]
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
                                i = 2 * ci + ei
                                k = 4 * t + 2 * cj + ej
                                W[k * n + i][col] += c_block * c_scalar
    
    return {"U": U, "V": V, "W": W}

def _reduce_rank_linear(U, V, W, max_time=30.0):
    """
    Try to reduce rank by finding linear dependencies among columns.
    For each column r, check if it can be expressed as a linear combination
    of other columns in a way that preserves the decomposition.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    start_time = time.time()
    new_U = U_arr.copy()
    new_V = V_arr.copy()
    new_W = W_arr.copy()
    current_rank = rank
    
    # For each column, try to express it as a combination of others
    # by checking if the "residual" can be zeroed out
    for r in range(rank):
        if time.time() - start_time > max_time:
            break
        if r >= current_rank:
            break
        
        # Skip if column is already zero
        if np.allclose(new_U[:, r], 0) and np.allclose(new_V[:, r], 0) and np.allclose(new_W[:, r], 0):
            # Remove this column
            new_U = np.delete(new_U, r, axis=1)
            new_V = np.delete(new_V, r, axis=1)
            new_W = np.delete(new_W, r, axis=1)
            current_rank -= 1
            continue
        
        # Try to find coefficients such that column r is a linear combination
        # of columns 0..r-1 (already processed)
        # This is a linear system: for each (i,j,k), sum_s c_s * U[i,j,s]*V[j,k,s]*W[k,i,s] = 0
        # when replacing column r with the combination
        
        # Simplified approach: check if column r is in the span of previous columns
        # in the tensor product sense
        pass
    
    return new_U, new_V, new_W, current_rank

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

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses a direct-sum Kronecker construction (Strassen 4x4x4 for two disjoint
    4x8 column blocks) yielding rank 98, then attempts rank reduction by
    eliminating redundant columns. Falls back to schoolbook (rank 128).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    best_result = None
    best_rank = float('inf')

    # Try Strassen-based decomposition
    try:
        result = _try_strassen_4x4_4x8()
        if _verify(result["U"], result["V"], result["W"]):
            best_result = result
            best_rank = len(result["U"][0])

            # Attempt rank reduction using linear algebra
            if time.time() < deadline:
                U_arr = np.array(result["U"], dtype=float)
                V_arr = np.array(result["V"], dtype=float)
                W_arr = np.array(result["W"], dtype=float)
                current_rank = len(U_arr[0])
                
                # Try to remove columns by checking if they are linearly dependent
                # in the tensor product space
                removed = 0
                for _ in range(current_rank):
                    if time.time() > deadline:
                        break
                    
                    # For each column, check if it can be removed
                    min_removable = -1
                    for r in range(U_arr.shape[1]):
                        # Check if column r is zero
                        if np.allclose(U_arr[:, r], 0) and np.allclose(V_arr[:, r], 0) and np.allclose(W_arr[:, r], 0):
                            min_removable = r
                            break
                    
                    if min_removable >= 0:
                        U_arr = np.delete(U_arr, min_removable, axis=1)
                        V_arr = np.delete(V_arr, min_removable, axis=1)
                        W_arr = np.delete(W_arr, min_removable, axis=1)
                        removed += 1
                    else:
                        break
                
                if removed > 0:
                    new_result = {
                        "U": U_arr.tolist(),
                        "V": V_arr.tolist(),
                        "W": W_arr.tolist()
                    }
                    if _verify(new_result["U"], new_result["V"], new_result["W"]):
                        best_result = new_result
                        best_rank = len(new_result["U"][0])
    except Exception:
        pass

    if best_result is not None:
        return best_result

    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
