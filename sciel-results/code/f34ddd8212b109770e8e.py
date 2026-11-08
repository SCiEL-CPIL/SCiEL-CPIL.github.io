import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    deadline = time.time() + 100  # safety margin
    
    # Try block-Strassen composition
    result = _block_strassen(n, m, p)
    if result is not None:
        return result
    
    # Fallback: trivial decomposition
    return _trivial_decomposition(n, m, p)

def _block_strassen(n, m, p):
    """
    Block-Strassen composition:
    - A is 4x4, split into 2x2 blocks of 2x2 submatrices
    - B is 4x8, split into 2x4 blocks of 2x2 submatrices
    - For each pair of A-blocks and B-blocks, use 7-term Strassen
    - Total: 2*4 = 8 block multiplications * 7 = 56 terms
    """
    # Strassen 7-term decomposition for 2x2 matrices
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Strassen term is a product of two linear combinations
    # of 2x2 blocks. We need to express each as a rank-1 tensor
    # (outer product of vectors).
    
    # For a 2x2 matrix multiplication C = A*B where A,B are 2x2:
    # The standard decomposition has rank 8 (trivial) or 7 (Strassen).
    # Each Strassen term M_s = (linear combo of A entries) * (linear combo of B entries)
    # which is a rank-1 tensor in the (A-entry, B-entry, C-entry) space.
    
    # Let's build the full decomposition.
    
    # A is 4x4: indices (i, k) where i in 0..3, k in 0..3
    # B is 4x8: indices (k, j) where k in 0..3, j in 0..7
    # C is 4x8: indices (i, j) where i in 0..3, j in 0..7
    
    # Block structure:
    # A blocks: A_{ab} where a in {0,1} (row block), b in {0,1} (col block)
    #   A_{00} = A[0:2, 0:2], A_{01} = A[0:2, 2:4]
    #   A_{10} = A[2:4, 0:2], A_{11} = A[2:4, 2:4]
    # B blocks: B_{bc} where b in {0,1} (row block), c in {0,1,2,3} (col block)
    #   B_{b0} = B[2b:2b+2, 0:2], B_{b1} = B[2b:2b+2, 2:4]
    #   B_{b2} = B[2b:2b+2, 4:6], B_{b3} = B[2b:2b+2, 6:8]
    # C blocks: C_{ac} where a in {0,1} (row block), c in {0,1,2,3} (col block)
    
    # For each (a, c) pair, we compute C_{a,c} = sum_b A_{a,b} * B_{b,c}
    # Each A_{a,b} * B_{b,c} is a 2x2 * 2x2 = 2x2 multiplication
    # Using Strassen, each such multiplication takes 7 terms.
    # Total terms: 2 * 4 * 7 = 56
    
    # Let's define the Strassen terms for 2x2 matrices.
    # For 2x2 matrices X, Y with entries:
    # X = [[x11, x12], [x21, x22]]
    # Y = [[y11, y12], [y21, y22]]
    # Z = X * Y
    # z11 = x11*y11 + x12*y21
    # z12 = x11*y12 + x12*y22
    # z21 = x21*y11 + x22*y21
    # z22 = x21*y12 + x22*y22
    
    # Strassen:
    # M1 = (x11+x22)*(y11+y22)
    # M2 = (x21+x22)*y11
    # M3 = x11*(y12-y22)
    # M4 = x22*(y21-y11)
    # M5 = (x11+x12)*y22
    # M6 = (x21-x11)*(y11+y12)
    # M7 = (x12-x22)*(y21+y22)
    # z11 = M1 + M4 - M5 + M7
    # z12 = M3 + M5
    # z21 = M2 + M4
    # z22 = M1 - M2 + M3 + M6
    
    # Each M_s is a product of a linear combo of X entries and a linear combo of Y entries.
    # This is a rank-1 tensor in the (X-entry, Y-entry, Z-entry) space.
    
    # Define the Strassen terms as (coeff_X, coeff_Y, coeff_Z) where:
    # M_s = (sum_i coeff_X[i] * X[i]) * (sum_j coeff_Y[j] * Y[j])
    # and contributes coeff_Z[z] * M_s to Z[z]
    
    # X entries indexed as: 0=x11, 1=x12, 2=x21, 3=x22
    # Y entries indexed as: 0=y11, 1=y12, 2=y21, 3=y22
    # Z entries indexed as: 0=z11, 1=z12, 2=z21, 3=z22
    
    strassen_terms = [
        # (coeff_X, coeff_Y, coeff_Z)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),  # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 0, 1]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),   # M5
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, 1]),  # M7
    ]
    
    # Now build the full decomposition.
    # U has shape (n*m, rank) = (16, rank)
    # V has shape (m*p, rank) = (32, rank)
    # W has shape (p*n, rank) = (32, rank)
    
    # Index conventions:
    # U[i*m + k, t] = coefficient for A[i,k] in term t
    # V[k*p + j, t] = coefficient for B[k,j] in term t
    # W[j*n + i, t] = coefficient for C[i,j] in term t
    
    terms = []  # list of (U_vec, V_vec, W_vec)
    
    for a in range(2):  # row block of A and C
        for b in range(2):  # col block of A, row block of B
            for c in range(4):  # col block of B and C
                # A_{a,b} is at rows 2a:2a+2, cols 2b:2b+2
                # B_{b,c} is at rows 2b:2b+2, cols 2c:2c+2
                # C_{a,c} is at rows 2a:2a+2, cols 2c:2c+2
                
                for s in range(7):
                    coeff_X, coeff_Y, coeff_Z = strassen_terms[s]
                    
                    U_vec = np.zeros(n*m)
                    V_vec = np.zeros(m*p)
                    W_vec = np.zeros(p*n)
                    
                    # Fill U_vec for A_{a,b}
                    for i_local in range(2):
                        for k_local in range(2):
                            i_global = 2*a + i_local
                            k_global = 2*b + k_local
                            U_idx = i_global * m + k_global
                            U_vec[U_idx] = coeff_X[i_local * 2 + k_local]
                    
                    # Fill V_vec for B_{b,c}
                    for k_local in range(2):
                        for j_local in range(2):
                            k_global = 2*b + k_local
                            j_global = 2*c + j_local
                            V_idx = k_global * p + j_global
                            V_vec[V_idx] = coeff_Y[k_local * 2 + j_local]
                    
                    # Fill W_vec for C_{a,c}
                    for i_local in range(2):
                        for j_local in range(2):
                            i_global = 2*a + i_local
                            j_global = 2*c + j_local
                            W_idx = j_global * n + i_global
                            W_vec[W_idx] = coeff_Z[i_local * 2 + j_local]
                    
                    terms.append((U_vec, V_vec, W_vec))
    
    rank = len(terms)
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for t, (U_vec, V_vec, W_vec) in enumerate(terms):
        U[:, t] = U_vec
        V[:, t] = V_vec
        W[:, t] = W_vec
    
    # Verify correctness
    if _verify(n, m, p, U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return None

def _verify(n, m, p, U, V, W):
    """Verify that the decomposition correctly computes matrix multiplication."""
    # Generate random test matrices
    np.random.seed(42)
    A = np.random.randint(-2, 3, size=(n, m))
    B = np.random.randint(-2, 3, size=(m, p))
    C_expected = A @ B
    
    # Compute using the decomposition
    U_mat = np.array(U)
    V_mat = np.array(V)
    W_mat = np.array(W)
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U_mat.reshape(n, m, -1)
    V_reshaped = V_mat.reshape(m, p, -1)
    W_reshaped = W_mat.reshape(p, n, -1)
    
    # C[i,j] = sum_t U_reshaped[i, :, t] @ V_reshaped[:, j, t] ... 
    # Actually: C[i,j] = sum_t sum_k sum_l U[i,k,t] * V[k,l,t] * W[l,i,t]
    # Wait, let me think about this more carefully.
    # The tensor decomposition says:
    # T[i,k,j] = sum_t U[i*m+k, t] * V[k*p+j, t] * W[j*n+i, t]
    # where T[i,k,j] is the matrix multiplication tensor.
    # C[i,j] = sum_k A[i,k] * B[k,j] = sum_k sum_l A[i,k] * B[k,j] * delta... 
    # Actually the tensor is T[i,k,j] = A[i,k]*B[k,j] which equals C[i,j] when summed over k.
    # So C[i,j] = sum_k T[i,k,j] = sum_k sum_t U[i*m+k,t]*V[k*p+j,t]*W[j*n+i,t]
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            s = 0.0
            for k in range(m):
                for t in range(U_mat.shape[1]):
                    s += U_mat[i*m+k, t] * V_mat[k*p+j, t] * W_mat[j*n+i, t]
            C_computed[i, j] = s
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        return False
    return True

def _trivial_decomposition(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    idx = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k, idx] = 1
                V[k * p + j, idx] = 1
                W[j * n + i, idx] = 1
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
