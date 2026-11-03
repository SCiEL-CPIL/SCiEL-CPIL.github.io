import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # 4 Strassen blocks + schoolbook
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    total_rank = 4 * 7 + (n * m * p - 4 * 8)
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        c = col
        U[i0*m + j0, c] += 1
        U[(i0+1)*m + (j0+1), c] += 1
        V[j0*p + k0, c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        W[k0*n + i0, c] += 1
        W[(k0+1)*n + (i0+1), c] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        c = col
        U[(i0+1)*m + j0, c] += 1
        U[(i0+1)*m + (j0+1), c] += 1
        V[j0*p + k0, c] += 1
        W[k0*n + (i0+1), c] += 1
        W[(k0+1)*n + (i0+1), c] += -1
        col += 1
        
        # M3 = A00(B01-B11)
        c = col
        U[i0*m + j0, c] += 1
        V[j0*p + (k0+1), c] += 1
        V[(j0+1)*p + (k0+1), c] += -1
        W[(k0+1)*n + i0, c] += 1
        W[(k0+1)*n + (i0+1), c] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        c = col
        U[(i0+1)*m + (j0+1), c] += 1
        V[(j0+1)*p + k0, c] += 1
        V[j0*p + k0, c] += -1
        W[k0*n + i0, c] += 1
        W[k0*n + (i0+1), c] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        c = col
        U[i0*m + j0, c] += 1
        U[i0*m + (j0+1), c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        W[k0*n + i0, c] += -1
        W[(k0+1)*n + i0, c] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        c = col
        U[(i0+1)*m + j0, c] += 1
        U[i0*m + j0, c] += -1
        V[j0*p + k0, c] += 1
        V[j0*p + (k0+1), c] += 1
        W[(k0+1)*n + (i0+1), c] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        c = col
        U[i0*m + (j0+1), c] += 1
        U[(i0+1)*m + (j0+1), c] += -1
        V[(j0+1)*p + k0, c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        W[k0*n + i0, c] += 1
        col += 1
    
    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0)
    
    # Schoolbook for remaining
    covered = np.zeros((n, m, p), dtype=bool)
    for i0, j0, k0 in blocks:
        covered[i0:i0+2, j0:j0+2, k0:k0+2] = True
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i, j, k]:
                    U[i*m + j, col] += 1
                    V[j*p + k, col] += 1
                    W[k*n + i, col] += 1
                    col += 1
    
    assert col == total_rank
    
    # Attempt rank reduction via merging
    # A set of rank-1 terms can be merged if their sum is rank 1
    # We'll try to find pairs (or small groups) that can be merged
    
    def is_rank1(tensor, tol=1e-10):
        """Check if a 3D tensor is rank 1."""
        # Find a non-zero entry
        nz = np.nonzero(np.abs(tensor) > tol)
        if len(nz[0]) == 0:
            return True, None  # Zero tensor is trivially rank 1
        
        i0, j0, k0 = nz[0][0], nz[1][0], nz[2][0]
        
        # Extract vectors
        a = tensor[:, j0, k0]  # a[i] = T[i, j0, k0]
        b = tensor[i0, :, k0]  # b[j] = T[i0, j, k0]
        c = tensor[i0, j0, :]  # c[k] = T[i0, j0, k]
        
        # Check: a[i]*b[j]*c[k] == T[i,j,k] for all (i,j,k)
        reconstructed = np.einsum('i,j,k->ijk', a, b, c)
        return np.allclose(tensor, reconstructed, atol=tol, rtol=tol), (a, b, c)
    
    def try_merge(U_mat, V_mat, W_mat, c1, c2):
        """Try to merge terms c1 and c2. Returns (success, new_u, new_v, new_w)."""
        # Form sum tensor
        u1, v1, w1 = U_mat[:, c1], V_mat[:, c1], W_mat[:, c1]
        u2, v2, w2 = U_mat[:, c2], V_mat[:, c2], W_mat[:, c2]
        
        # T[i,j,k] = u1[i]*v1[j]*w1[k] + u2[i]*v2[j]*w2[k]
        T = np.einsum('i,j,k->ijk', u1, v1, w1) + np.einsum('i,j,k->ijk', u2, v2, w2)
        
        is_r1, (a, b, c_vec) = is_rank1(T)
        if is_r1:
            return True, a, b, c_vec
        return False, None, None, None
    
    # Try all pairs
    merged_cols = set()
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    active_cols = list(range(total_rank))
    
    max_passes = 10
    for _ in range(max_passes):
        if time.time() > deadline - 20:
            break
        found = False
        new_active = []
        i = 0
        while i < len(active_cols):
            c1 = active_cols[i]
            if c1 in merged_cols:
                i += 1
                continue
            merged_with = -1
            for j in range(i+1, len(active_cols)):
                c2 = active_cols[j]
                if c2 in merged_cols:
                    continue
                success, new_u, new_v, new_w = try_merge(current_U, current_V, current_W, c1, c2)
                if success:
                    # Replace c1 with the merged term, mark c2 as merged
                    current_U[:, c1] = new_u
                    current_V[:, c1] = new_v
                    current_W[:, c1] = new_w
                    merged_cols.add(c2)
                    merged_with = c2
                    found = True
                    break
            if merged_with == -1:
                new_active.append(c1)
            i += 1
        active_cols = new_active
        if not found:
            break
    
    if merged_cols:
        keep = [c for c in range(total_rank) if c not in merged_cols]
        U_final = current_U[:, keep]
        V_final = current_V[:, keep]
        W_final = current_W[:, keep]
    else:
        U_final = U
        V_final = V
        W_final = W
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
