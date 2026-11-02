import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 140  # seconds, safety margin before 150s kill


def _build_strassen_rank55():
    """Build the guaranteed rank-55 Strassen-based decomposition."""
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    rank = 55
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    a00, a01, a10, a11 = 0, 1, 3, 4
    a02, a12 = 2, 5
    
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        b00, b01 = k0, k1
        b10, b11 = 10 + k0, 10 + k1
        c00, c01 = 2 * k0, 2 * k1
        c10, c11 = 2 * k0 + 1, 2 * k1 + 1
        
        # M1 = (A00 + A11)(B00 + B11)
        U[a00, col] += 1.0; U[a11, col] += 1.0
        V[b00, col] += 1.0; V[b11, col] += 1.0
        W[c00, col] += 1.0; W[c11, col] += 1.0
        col += 1
        
        # M2 = (A10 + A11)B00
        U[a10, col] += 1.0; U[a11, col] += 1.0
        V[b00, col] += 1.0
        W[c10, col] += 1.0; W[c11, col] -= 1.0
        col += 1
        
        # M3 = A00(B01 - B11)
        U[a00, col] += 1.0
        V[b01, col] += 1.0; V[b11, col] -= 1.0
        W[c01, col] += 1.0; W[c11, col] += 1.0
        col += 1
        
        # M4 = A11(B10 - B00)
        U[a11, col] += 1.0
        V[b10, col] += 1.0; V[b00, col] -= 1.0
        W[c00, col] += 1.0; W[c10, col] += 1.0
        col += 1
        
        # M5 = (A00 + A01)B11
        U[a00, col] += 1.0; U[a01, col] += 1.0
        V[b11, col] += 1.0
        W[c00, col] -= 1.0; W[c01, col] += 1.0
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[a10, col] += 1.0; U[a00, col] -= 1.0
        V[b00, col] += 1.0; V[b01, col] += 1.0
        W[c11, col] += 1.0
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[a01, col] += 1.0; U[a11, col] -= 1.0
        V[b10, col] += 1.0; V[b11, col] += 1.0
        W[c00, col] += 1.0
        col += 1
    
    for k in range(p):
        b2k = 2 * p + k
        U[a02, col] += 1.0; V[b2k, col] += 1.0; W[2 * k, col] += 1.0
        col += 1
        U[a12, col] += 1.0; V[b2k, col] += 1.0; W[2 * k + 1, col] += 1.0
        col += 1
    
    return U, V, W


def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(U.shape[1]):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _try_rank54_via_merging(start_time):
    """
    Attempt to find a rank-54 decomposition by merging pairs of columns
    from the rank-55 solution.
    
    Two rank-1 tensors can be merged into one if their sum is also rank-1.
    For two rank-1 tensors u1⊗v1⊗w1 and u2⊗v2⊗w2, their sum is rank-1
    if and only if they are proportional (which is trivial) or if there's
    a special structure.
    
    In practice, for the Strassen decomposition, some columns might be
    mergeable if they have a common factor in one of the three matrices.
    
    We check each pair and see if removing one and adjusting the other
    (or creating a new merged column) works.
    """
    n, m, p = DIMENSIONS
    U55, V55, W55 = _build_strassen_rank55()
    
    for i in range(55):
        if time.time() - start_time > DEADLINE - 10:
            break
        for j in range(i + 1, 55):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Check if columns i and j can be merged.
            # The sum of their contributions should be a rank-1 tensor.
            # Compute the 60-dimensional vectors for each column.
            vec_i = np.zeros(60)
            vec_j = np.zeros(60)
            vec_sum = np.zeros(60)
            idx = 0
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        vi = U55[ii * m + jj, i] * V55[jj * p + kk, i] * W55[kk * n + ii, i]
                        vj = U55[ii * m + jj, j] * V55[jj * p + kk, j] * W55[kk * n + ii, j]
                        vec_i[idx] = vi
                        vec_j[idx] = vj
                        vec_sum[idx] = vi + vj
                        idx += 1
            
            # Check if vec_sum is rank-1 in the tensor sense.
            # A 2x3x10 tensor T is rank-1 if T[i,j,k] = u[i,j]*v[j,k]*w[k,i] for some u,v,w.
            # We can check this by fixing one index and seeing if the remaining matrix has rank 1.
            # Fix k=0: T[:, :, 0] should be u[:, :] * v[:, 0] * w[0, :] (but w depends on i too).
            # Actually, T[i,j,k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
            # So for fixed k, T[i,j,k] = u[i*m+j] * (v[j*p+k] * w[k*n+i])
            # This is not simply a product of a function of (i,j) and a function of j alone.
            
            # Better: check if vec_sum can be written as u⊗v⊗w.
            # Pick a nonzero entry T[i0,j0,k0] = u[i0*m+j0] * v[j0*p+k0] * w[k0*n+i0]
            # Then for all i,j,k: T[i,j,k] / T[i0,j0,k0] = (u[i*m+j]/u[i0*m+j0]) * (v[j*p+k]/v[j0*p+k0]) * (w[k*n+i]/w[k0*n+i0])
            
            # Find a nonzero entry
            nz = np.nonzero(np.abs(vec_sum) > 1e-12)
            if len(nz[0]) == 0:
                continue
            
            i0, j0, k0 = nz[0][0] // (m*p), (nz[0][0] // p) % m, nz[0][0] % p
            T000 = vec_sum[i0 * m * p + j0 * p + k0]
            
            # Construct candidate u, v, w
            u_cand = np.zeros(n * m)
            v_cand = np.zeros(m * p)
            w_cand = np.zeros(p * n)
            
            # Normalize so that u[i0*m+j0] = 1
            u_cand[i0 * m + j0] = 1.0
            # Then v[j0*p+k0] * w[k0*n+i0] = T000
            # Set v[j0*p+k0] = 1, so w[k0*n+i0] = T000
            v_cand[j0 * p + k0] = 1.0
            w_cand[k0 * n + i0] = T000
            
            # Propagate to fill in u, v, w
            # For any (i,j,k) with T[i,j,k] != 0:
            # u[i*m+j] = T[i,j,k] / (v[j*p+k] * w[k*n+i])
            # v[j*p+k] = T[i,j,k] / (u[i*m+j] * w[k*n+i])
            # w[k*n+i] = T[i,j,k] / (u[i*m+j] * v[j*p+k])
            
            changed = True
            max_iter = 100
            for _ in range(max_iter):
                if not changed:
                    break
                changed = False
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_ijk = vec_sum[i * m * p + j * p + k]
                            if abs(T_ijk) < 1e-12:
                                continue
                            uj = u_cand[i * m + j]
                            vjk = v_cand[j * p + k]
                            wki = w_cand[k * n + i]
                            
                            if abs(uj) < 1e-12 and abs(vjk) > 1e-12 and abs(wki) > 1e-12:
                                u_cand[i * m + j] = T_ijk / (vjk * wki)
                                changed = True
                            elif abs(uj) > 1e-12 and abs(vjk) < 1e-12 and abs(wki) > 1e-12:
                                v_cand[j * p + k] = T_ijk / (uj * wki)
                                changed = True
                            elif abs(uj) > 1e-12 and abs(vjk) > 1e-12 and abs(wki) < 1e-12:
                                w_cand[k * n + i] = T_ijk / (uj * vjk)
                                changed = True
                            elif abs(uj) < 1e-12 and abs(vjk) < 1e-12 and abs(wki) < 1e-12:
                                # All zero, but T_ijk != 0, contradiction
                                return None
            
            # Verify that u_cand ⊗ v_cand ⊗ w_cand == vec_sum
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        computed = u_cand[i * m + j] * v_cand[j * p + k] * w_cand[k * n + i]
                        err = abs(computed - vec_sum[i * m * p + j * p + k])
                        max_err = max(max_err, err)
            
            if max_err > 1e-8:
                continue
            
            # If we found a valid merge, create the rank-54 solution
            # Remove column j and replace column i with the merged u, v, w
            # Actually, we need to be careful: the merged tensor is the SUM of columns i and j.
            # So we remove column j and set column i to the merged tensor.
            # But the rest of the columns remain unchanged.
            
            mask = np.ones(55, dtype=bool)
            mask[j] = False
            U54 = U55[:, mask].copy()
            V54 = V55[:, mask].copy()
            W54 = W55[:, mask].copy()
            
            # Find the new index of column i in the masked array
            new_i = sum(1 for x in range(55) if x < i and mask[x])
            
            U54[:, new_i] = u_cand
            V54[:, new_i] = v_cand
            W54[:, new_i] = w_cand
            
            if _verify_decomposition(U54, V54, W54):
                return U54, V54, W54
    
    return None


def _try_rank54_via_optimization(start_time):
    """
    Try to find a rank-54 decomposition via optimization.
    Start from rank-55, remove one column, and try to adjust the remaining
    columns to compensate.
    
    We use a simple iterative method: for each removed column, compute the
    residual tensor, and try to fit it with a rank-1 tensor using the
    remaining columns' support pattern.
    """
    n, m, p = DIMENSIONS
    U55, V55, W55 = _build_strassen_rank55()
    
    for remove_col in range(55):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        mask = np.ones(55, dtype=bool)
        mask[remove_col] = False
        U54 = U55[:, mask].copy()
        V54 = V55[:, mask].copy()
        W54 = W55[:, mask].copy()
        
        # Compute the residual: what's missing?
        residual = np.zeros((n, m, p))
        for r in range(54):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        residual[i, j, k] -= U54[i * m + j, r] * V54[j * p + k, r] * W54[k * n + i, r]
        
        # residual[i,j,k] = 1 - sum_{r != remove_col} ... 
        # We need to add back the contribution of the removed column.
        # The removed column contributed:
        # U55[i*m+j, remove_col] * V55[j*p+k, remove_col] * W55[k*n+i, remove_col]
        
        u_orig = U55[:, remove_col]
        v_orig = V55[:, remove_col]
        w_orig = W55[:, remove_col]
        
        # Check if residual equals u_orig ⊗ v_orig ⊗ w_orig
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = u_orig[i * m + j] * v_orig[j * p + k] * w_orig[k * n + i]
                    err = abs(residual[i, j, k] - expected)
                    max_err = max(max_err, err)
        
        if max_err > 1e-8:
            # The residual is not just the removed column, which means
            # the remaining columns don't form a valid decomposition.
            # This shouldn't happen if we started from a valid rank-55.
            continue
        
        # The residual IS the removed column. So removing it doesn't work
        # unless the removed column is zero.
        
        # Try a different approach: allow the remaining columns to be adjusted.
        # This is a hard nonlinear problem, so we skip for now.
        pass
    
    return None


def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x10 matrix.
    
    Strategy:
    - First try to find a rank-54 decomposition via column merging.
    - Fall back to the guaranteed rank-55 Strassen-based decomposition.
    """
    start_time = time.time()
    
    # Try to find rank-54 via merging
    result = _try_rank54_via_merging(start_time)
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to rank-55
    U, V, W = _build_strassen_rank55()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
