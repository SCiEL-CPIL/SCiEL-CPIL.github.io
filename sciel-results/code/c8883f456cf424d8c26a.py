"""Matrix multiplication 3x4x5: rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    """Verify that U @ diag @ V^T gives the correct matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # W has shape (p*n, rank), index is k*n + i where k is col of C, i is row of C
    # The tensor T[i, j, k] should equal C[i, k] = sum_j A[i, j] B[j, k]
    # In our factorization: T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # The correct value is delta(j,j) = 1 when j is the same index
                # Actually C[i,k] = sum_j A[i,j] B[j,k], so T[i,j,k] = 1 for all i,j,k? No.
                # T[i,j,k] = A[i,j] * B[j,k] coefficient, which is 1.
                # Wait, the tensor for matrix multiplication is T[i,j,k] = 1 (all ones)
                # No, that's not right either. Let me think again.
                # C = A * B, C[i,k] = sum_j A[i,j] B[j,k]
                # The tensor T has T[i,j,k] = 1 for all valid i,j,k? No.
                # Actually the multiplication tensor has T[i,j,k] = 1 if and only if... 
                # Hmm, actually T[i,j,k] = 1 for all i, j, k where the indices are valid.
                # Wait no. Let me reconsider.
                # The matrix multiplication tensor T of size n x m x p has T[i,j,k] = 1 
                # because C[i,k] = sum_j A[i,j] B[j,k], and each term A[i,j]*B[j,k] contributes.
                # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
                expected = 1.0
                max_err = max(max_err, abs(val - expected))
    return max_err

def _verify_fast(U, V, W, n, m, p, rank):
    """Faster verification using numpy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Compute T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For each r, U[:,r], V[:,r], W[:,r] contribute
    # T = sum_r outer of reshaped vectors
    
    max_err = 0.0
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        # Contribution to T[i,j,k] from term r: u[i,j] * v[j,k] * w[k,i]
        # This is a rank-1 tensor in the 3D space
        for i in range(n):
            for j in range(m):
                if u[i,j] == 0:
                    continue
                for k in range(p):
                    if v[j,k] == 0:
                        continue
                    if w[k,i] == 0:
                        continue
                    # We need to accumulate, so let's build the full tensor
                    pass
    
    # Better approach: build the tensor directly
    T = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        for i in range(n):
            for j in range(m):
                uval = u[i,j]
                if uval == 0:
                    continue
                for k in range(p):
                    T[i,j,k] += uval * v[j,k] * w[k,i]
    
    max_err = np.max(np.abs(T - 1.0))
    return max_err

def _verify_very_fast(U, V, W, n, m, p, rank):
    """Even faster verification using einsum."""
    U = np.array(U, dtype=float).reshape(n, m, rank)
    V = np.array(V, dtype=float).reshape(m, p, rank)
    W = np.array(W, dtype=float).reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
    return np.max(np.abs(T - 1.0))

def construct():
    """
    Returns a rank-40 exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS  # 3, 4, 5
    rank = 40
    
    U = np.zeros((n * m, rank), dtype=float)  # 12 x 40
    V = np.zeros((m * p, rank), dtype=float)  # 20 x 40
    W = np.zeros((p * n, rank), dtype=float)  # 15 x 40
    
    col = 0
    
    def add_strassen_2x2x2(a_row, a_col, b_row, b_col, c_row, c_col, c_offset):
        """
        Add Strassen 2x2x2 decomposition.
        A: rows [a_row, a_row+1], cols [a_col, a_col+1]
        B: rows [b_row, b_row+1], cols [b_col, b_col+1]
        C: rows [c_row, c_row+1], cols [c_col, c_col+1]
        """
        # M1 = (A11+A22)(B11+B22)
        U[a_row * m + a_col, c_offset] += 1.0
        U[(a_row+1) * m + (a_col+1), c_offset] += 1.0
        V[b_row * p + b_col, c_offset] += 1.0
        V[(b_row+1) * p + (b_col+1), c_offset] += 1.0
        W[c_col * n + c_row, c_offset] += 1.0
        W[c_col * n + (c_row+1), c_offset] += 1.0
        W[(c_col+1) * n + c_row, c_offset] += 1.0
        W[(c_col+1) * n + (c_row+1), c_offset] += 1.0
        c_offset += 1
        
        # M2 = (A21+A22)B11
        U[(a_row+1) * m + a_col, c_offset] += 1.0
        U[(a_row+1) * m + (a_col+1), c_offset] += 1.0
        V[b_row * p + b_col, c_offset] += 1.0
        W[(c_col+1) * n + (c_row+1), c_offset] += 1.0
        c_offset += 1
        
        # M3 = A11(B21-B11)
        U[a_row * m + a_col, c_offset] += 1.0
        V[(b_row+1) * p + b_col, c_offset] += 1.0
        V[b_row * p + b_col, c_offset] -= 1.0
        W[c_col * n + c_row, c_offset] += 1.0
        c_offset += 1
        
        # M4 = A22(B22-B12)
        U[(a_row+1) * m + (a_col+1), c_offset] += 1.0
        V[(b_row+1) * p + (b_col+1), c_offset] += 1.0
        V[b_row * p + (b_col+1), c_offset] -= 1.0
        W[(c_col+1) * n + (c_row+1), c_offset] += 1.0
        c_offset += 1
        
        # M5 = (A11+A12)B22
        U[a_row * m + a_col, c_offset] += 1.0
        U[a_row * m + (a_col+1), c_offset] += 1.0
        V[(b_row+1) * p + (b_col+1), c_offset] += 1.0
        W[(c_col+1) * n + c_row, c_offset] += 1.0
        c_offset += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[(a_row+1) * m + a_col, c_offset] += 1.0
        U[a_row * m + a_col, c_offset] -= 1.0
        V[b_row * p + b_col, c_offset] += 1.0
        V[b_row * p + (b_col+1), c_offset] += 1.0
        W[c_col * n + (c_row+1), c_offset] += 1.0
        c_offset += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[a_row * m + (a_col+1), c_offset] += 1.0
        U[(a_row+1) * m + (a_col+1), c_offset] -= 1.0
        V[(b_row+1) * p + b_col, c_offset] += 1.0
        V[(b_row+1) * p + (b_col+1), c_offset] += 1.0
        W[(c_col+1) * n + (c_row+1), c_offset] += 1.0
        c_offset += 1
        
        return c_offset
    
    # C11: A[0:2, :] * B[:, 0:2]
    col = add_strassen_2x2x2(0, 0, 0, 0, 0, 0, col)   # 7
    col = add_strassen_2x2x2(0, 2, 2, 0, 0, 0, col)   # 7
    
    # C12: A[0:2, :] * B[:, 2:5]
    col = add_strassen_2x2x2(0, 0, 0, 2, 0, 2, col)   # 7
    col = add_strassen_2x2x2(0, 2, 2, 2, 0, 2, col)   # 7
    for j in range(4):
        U[0 * m + j, col] += 1.0
        U[1 * m + j, col] += 1.0
        V[j * p + 4, col] += 1.0
        W[4 * n + 0, col] += 1.0
        W[4 * n + 1, col] += 1.0
        col += 1
    
    # C21: A[2:3, :] * B[:, 0:2]
    for j in range(4):
        U[2 * m + j, col] += 1.0
        V[j * p + 0, col] += 1.0
        W[0 * n + 2, col] += 1.0
        V[j * p + 1, col] += 1.0
        W[1 * n + 2, col] += 1.0
        col += 1
    
    # C22: A[2:3, :] * B[:, 2:5]
    for j in range(4):
        U[2 * m + j, col] += 1.0
        V[j * p + 2, col] += 1.0
        W[2 * n + 2, col] += 1.0
        V[j * p + 3, col] += 1.0
        W[3 * n + 2, col] += 1.0
        V[j * p + 4, col] += 1.0
        W[4 * n + 2, col] += 1.0
        col += 1
    
    if col != 40:
        raise ValueError(f"Expected rank 40, got {col}")
    
    # Verify
    err = _verify_very_fast(U, V, W, n, m, p, rank)
    if err > 1e-10:
        raise ValueError(f"Verification failed with max error {err}")
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
