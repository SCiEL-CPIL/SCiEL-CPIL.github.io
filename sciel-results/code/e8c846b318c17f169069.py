"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 130.0  # safety margin before 150s kill

def schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, rank

def strassen_2x2_block(U, V, W, i0, i1, j0, j1, k0, k1, offset):
    """Apply Strassen to a 2x2x2 sub-block.
    
    U rows indexed by (i,j), V rows by (j,k), W rows by (k,i).
    For the 2x2x2 block with i in {i0,i1}, j in {j0,j1}, k in {k0,k1}:
    
    The 8 multiplications are replaced by 7:
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11)B00
    M3 = A00(B10 - B00)
    M4 = A11(B01 - B11)
    M5 = (A00 + A01)B11
    M6 = (A10 - A00)(B00 + B10)
    M7 = (A01 - A11)(B01 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    n, m, p = DIMENSIONS
    # Row indices in U: i*m + j
    # Row indices in V: j*p + k
    # Row indices in W: k*n + i
    
    # For the 2x2x2 block:
    # U rows: (i0,j0), (i0,j1), (i1,j0), (i1,j1)
    # V rows: (j0,k0), (j0,k1), (j1,k0), (j1,k1)
    # W rows: (k0,i0), (k0,i1), (k1,i0), (k1,i1)
    
    # The 8 schoolbook columns in this block are at:
    # (i0,j0,k0), (i0,j0,k1), (i0,j1,k0), (i0,j1,k1),
    # (i1,j0,k0), (i1,j0,k1), (i1,j1,k0), (i1,j1,k1)
    
    # Columns in the full matrices:
    col_base = (i0 * m + j0) * p + k0 if False else None  # Not a simple formula
    
    # Actually, let's think about this differently.
    # In the schoolbook decomposition, each column c corresponds to (i,j,k).
    # The mapping from (i,j,k) to column index is:
    # col = i*m*p + j*p + k
    
    cols = []
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                i = i0 + di
                j = j0 + dj
                k = k0 + dk
                col = i * m * p + j * p + k
                cols.append(col)
    
    # cols has 8 entries
    # M1 = (A00+A11)(B00+B11): U has A00+A11, V has B00+B11
    # We need to express each M in terms of the original U,V,W entries.
    
    # Let me define the 8 original tensors:
    # T0 = (i0,j0,k0), T1 = (i0,j0,k1), T2 = (i0,j1,k0), T3 = (i0,j1,k1)
    # T4 = (i1,j0,k0), T5 = (i1,j0,k1), T6 = (i1,j1,k0), T7 = (i1,j1,k1)
    
    # U entries: U[i0*m+j0, c], U[i0*m+j1, c], U[i1*m+j0, c], U[i1*m+j1, c]
    # V entries: V[j0*p+k0, c], V[j0*p+k1, c], V[j1*p+k0, c], V[j1*p+k1, c]
    # W entries: W[k0*n+i0, c], W[k0*n+i1, c], W[k1*n+i0, c], W[k1*n+i1, c]
    
    # For Strassen, we create 7 new tensors replacing the 8 original ones.
    # The new U,V,W for these 7 tensors are linear combinations of the old ones.
    
    # M1: U has A00+A11 = rows (i0,j0) and (i1,j1) both = 1
    #       V has B00+B11 = rows (j0,k0) and (j1,k1) both = 1
    #       W: all four C outputs need M1 with appropriate signs
    #       C00 gets +M1, C11 gets +M1
    #       W[k0*n+i0] += 1, W[k1*n+i1] += 1
    
    # M2: U has A10+A11 = rows (i1,j0) and (i1,j1) = 1
    #       V has B00 = row (j0,k0) = 1
    #       C10 gets +M2, C11 gets -M2
    #       W[k0*n+i1] += 1, W[k1*n+i1] -= 1
    
    # M3: U has A00 = row (i0,j0) = 1
    #       V has B10-B00 = rows (j1,k0) = 1, (j0,k0) = -1
    #       C01 gets +M3, C11 gets +M3
    #       W[k1*n+i0] += 1, W[k1*n+i1] += 1
    
    # M4: U has A11 = row (i1,j1) = 1
    #       V has B01-B11 = rows (j0,k1) = 1, (j1,k1) = -1
    #       C00 gets +M4, C10 gets +M4
    #       W[k0*n+i0] += 1, W[k0*n+i1] += 1
    
    # M5: U has A00+A01 = rows (i0,j0) and (i0,j1) = 1
    #       V has B11 = row (j1,k1) = 1
    #       C00 gets -M5, C01 gets +M5
    #       W[k0*n+i0] -= 1, W[k1*n+i0] += 1
    
    # M6: U has A10-A00 = rows (i1,j0) = 1, (i0,j0) = -1
    #       V has B00+B10 = rows (j0,k0) = 1, (j1,k0) = 1
    #       C11 gets +M6
    #       W[k1*n+i1] += 1
    
    # M7: U has A01-A11 = rows (i0,j1) = 1, (i1,j1) = -1
    #       V has B01+B11 = rows (j0,k1) = 1, (j1,k1) = 1
    #       C00 gets +M7
    #       W[k0*n+i0] += 1
    
    # Now, the 8 original columns are at indices cols[0..7].
    # We'll zero them out and create 7 new columns.
    
    # First, zero out the 8 original columns
    for c in cols:
        U[:, c] = 0
        V[:, c] = 0
        W[:, c] = 0
    
    # Now create 7 new columns. We need to find 7 unused column indices.
    # Since we're replacing 8 columns with 7, we free up 1 column.
    # Let's use the last of the 8 original columns as the "free" one,
    # and use 7 of them for the new Strassen tensors.
    
    # Actually, it's simpler to just use 7 of the 8 columns for the new tensors
    # and leave the 8th as zero (which effectively removes it).
    
    new_cols = cols[:7]
    free_col = cols[7]  # This will be all zeros
    
    # M1: new_cols[0]
    c = new_cols[0]
    U[i0*m+j0, c] += 1
    U[i1*m+j1, c] += 1
    V[j0*p+k0, c] += 1
    V[j1*p+k1, c] += 1
    W[k0*n+i0, c] += 1
    W[k1*n+i1, c] += 1
    
    # M2: new_cols[1]
    c = new_cols[1]
    U[i1*m+j0, c] += 1
    U[i1*m+j1, c] += 1
    V[j0*p+k0, c] += 1
    W[k0*n+i1, c] += 1
    W[k1*n+i1, c] -= 1
    
    # M3: new_cols[2]
    c = new_cols[2]
    U[i0*m+j0, c] += 1
    V[j1*p+k0, c] += 1
    V[j0*p+k0, c] -= 1
    W[k1*n+i0, c] += 1
    W[k1*n+i1, c] += 1
    
    # M4: new_cols[3]
    c = new_cols[3]
    U[i1*m+j1, c] += 1
    V[j0*p+k1, c] += 1
    V[j1*p+k1, c] -= 1
    W[k0*n+i0, c] += 1
    W[k0*n+i1, c] += 1
    
    # M5: new_cols[4]
    c = new_cols[4]
    U[i0*m+j0, c] += 1
    U[i0*m+j1, c] += 1
    V[j1*p+k1, c] += 1
    W[k0*n+i0, c] -= 1
    W[k1*n+i0, c] += 1
    
    # M6: new_cols[5]
    c = new_cols[5]
    U[i1*m+j0, c] += 1
    U[i0*m+j0, c] -= 1
    V[j0*p+k0, c] += 1
    V[j1*p+k0, c] += 1
    W[k1*n+i1, c] += 1
    
    # M7: new_cols[6]
    c = new_cols[6]
    U[i0*m+j1, c] += 1
    U[i1*m+j1, c] -= 1
    V[j0*p+k1, c] += 1
    V[j1*p+k1, c] += 1
    W[k0*n+i0, c] += 1
    
    # free_col (cols[7]) is already zeroed out
    
    return U, V, W

def apply_strassen_to_3x4x8():
    """Apply Strassen to a 2x2x2 block within the 3x4x8 multiplication."""
    U, V, W, rank = schoolbook()
    
    # Choose a 2x2x2 block. Let's use i in {0,1}, j in {0,1}, k in {0,1}
    i0, i1 = 0, 1
    j0, j1 = 0, 1
    k0, k1 = 0, 1
    
    U, V, W = strassen_2x2_block(U, V, W, i0, i1, j0, j1, k0, k1, 0)
    
    # Now rank is effectively 95 (96 - 1)
    # The last column (cols[7]) is all zeros, so we can trim it
    # But it's simpler to just return with rank=95 by removing the zero column
    
    # Find which column is the free one
    # It's the 8th in the 2x2x2 block
    free_col = 1 * 4 * 8 + 1 * 8 + 1  # (i1,j1,k1) = 1*32+1*8+1 = 41
    # Wait, let me recalculate: col = i*m*p + j*p + k = 1*4*8 + 1*8 + 1 = 32+8+1 = 41
    
    # Remove column 41
    mask = np.ones(rank, dtype=bool)
    mask[free_col] = False
    U = U[:, mask]
    V = V[:, mask]
    W = W[:, mask]
    
    new_rank = rank - 1
    return U, V, W, new_rank

def verify_decomposition(U, V, W, rank):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n, m, p = DIMENSIONS
    # Test with random matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute C from the decomposition: C[i,k] = sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c]
    # But we need to be careful: the decomposition is
    # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    
    # Actually, the standard form is:
    # C = sum_c W_c * (U_c^T * A) * (V_c^T * B) ... no, that's not right either.
    
    # Let me think more carefully. In the tensor decomposition:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    # 
    # And matrix multiplication is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The tensor for matrix multiplication is:
    # T[i,j,k] = 1 (it's the identity in the j index, coupling i to j and j to k)
    # 
    # Wait, actually the matrix multiplication tensor is:
    # T[i,j,k] = delta_{j,j} ... no.
    # 
    # The correct statement is: C[i,k] = sum_{j} A[i,j] B[j,k]
    # This is a trilinear form: sum_{i,j,k} C[i,k] * A[i,j] * B[j,k] = sum_{i,j,k} (sum_j' A[i,j'] B[j',k]) * A[i,j] * B[j,k]
    # 
    # Hmm, let me just directly verify by checking that for all (i,k):
    # sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k]) = C[i,k]
    
    C_computed = np.zeros((n, p))
    for c in range(rank):
        u_col = U[:, c]  # shape (n*m,)
        v_col = V[:, c]  # shape (m*p,)
        w_val = W[:, c]  # shape (p*n,)
        
        # Reshape u_col to (n, m)
        u_mat = u_col.reshape(n, m)
        # Reshape v_col to (m, p)
        v_mat = v_col.reshape(m, p)
        
        # For each (i,k), the contribution is w_val[k*n+i] * (u_mat * A.T)[i,:] ... 
        # Actually: sum_j U[i*m+j,c] * A[i,j] is the dot product of u_mat[i,:] and A[i,:]
        # And sum_j V[j*p+k,c] * B[j,k] is the dot product of v_mat[:,k] and B[:,k]
        
        for i in range(n):
            for k in range(p):
                u_dot = np.dot(u_mat[i, :], A[i, :])
                v_dot = np.dot(v_mat[:, k], B[:, k])
                C_computed[i, k] += w_val[k * n + i] * u_dot * v_dot
    
    error = np.max(np.abs(C_computed - C_expected))
    return error

def construct():
    """Construct the matrix multiplication decomposition."""
    # Start with schoolbook
    U, V, W, rank = schoolbook()
    
    # Apply Strassen to a 2x2x2 block
    try:
        U, V, W, rank = apply_strassen_to_3x4x8()
        # Verify
        error = verify_decomposition(U, V, W, rank)
        if error > 1e-9:
            # Strassen failed verification, fall back to schoolbook
            U, V, W, rank = schoolbook()
    except Exception:
        U, V, W, rank = schoolbook()
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
