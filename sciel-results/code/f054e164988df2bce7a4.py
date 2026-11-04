"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _validate(U, V, W, n, m, p, seed=42, num_tests=3):
    """
    Validate that the decomposition correctly computes matrix multiplication.
    The decomposition satisfies:
      C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    Wait, that's not right. Let me think about this more carefully.
    
    The tensor decomposition means:
    For the matrix multiplication tensor T with T[i,j,k] = δ (identity in j),
    we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    But actually, the correct interpretation is:
    The bilinear map computes C[i,k] = sum_j A[i,j] * B[j,k]
    Using the decomposition:
    C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    
    Wait, that's not right either. Let me reconsider.
    
    Actually, the standard tensor decomposition for matrix multiplication is:
    C[i,k] = sum_c alpha_c * (sum_j U_c[i,j] * A[i,j]) * (sum_j V_c[j,k] * B[j,k])
    
    But the format here is different. Let me look at the schoolbook case:
    U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for each (i,j,k)
    
    So: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    = sum_c 1 * 1 * 1 (when c corresponds to (i,j,k))
    = 1 (for the specific (i,j,k) triple)
    
    But this gives T[i,j,k] = 1 for all i,j,k, which is not the matrix multiplication tensor.
    
    I think the correct interpretation is that the decomposition defines a bilinear form:
    C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    
    No wait, that would have W depending on both i and k, which is weird.
    
    Let me re-read: "U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)"
    
    So U is indexed by (i,j) pairs, V by (j,k) pairs, W by (k,i) pairs.
    
    The decomposition should satisfy:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    where T is the matrix multiplication tensor. For matrix multiplication,
    T[i,j,k] = 1 if and only if... hmm.
    
    Actually, I think the tensor is defined such that:
    C[i,k] = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k]
    
    For standard matrix multiplication, C[i,k] = sum_j A[i,j] * B[j,k]
    So T[i,j,k] = 1 if the indices are consistent, i.e., always 1.
    
    But that's trivial. Let me think differently.
    
    Maybe the convention is:
    C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    
    For schoolbook: U[i*m+j, c] = 1 for c=(i,j,k), so sum_j U[i*m+j, c] * A[i,j] = A[i,j]
    Similarly for V: sum_j V[j*p+k, c] * B[j,k] = B[j,k]
    And W[k*n+i, c] = 1
    
    So C[i,k] = sum_{c=(i',j',k')} A[i,j'] * B[j',k'] * δ_{i',i} * δ_{k',k}
    = A[i,j] * B[j,k] summed over j... 
    
    Hmm, this is getting confusing. Let me just test with the actual formula.
    
    I think the correct formula is:
    C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    
    Let me verify with schoolbook:
    U[i*m+j, c] = δ_{c,(i,j,k')} for some k'
    V[j'*p+k, c] = δ_{c,(i',j,k)} for some i'  
    W[k'*n+i', c] = δ_{c,(i,j,k')} for some k'
    
    So for a fixed c = (i0, j0, k0):
    sum_j U[i*m+j, c] * A[i,j] = A[i, j0] if i = i0, else 0
    sum_j V[j*p+k, c] * B[j,k] = B[j0, k] if k = k0, else 0
    W[k0*n+i0, c] = 1
    
    So C[i,k] = sum_c A[i,j_c] * B[j_c,k] * W[k*n+i, c]
    
    For schoolbook, W[k*n+i, c] = 1 when c = (i,j,k), so:
    C[i,k] = sum_j A[i,j] * B[j,k] * 1 = sum_j A[i,j] * B[j,k]
    
    Great, that works! So the formula is:
    C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    
    Let me verify this is what we should check.
    """
    np.random.seed(seed)
    for _ in range(num_tests):
        A = np.random.randint(0, 3, size=(n, m)).astype(float)
        B = np.random.randint(0, 3, size=(m, p)).astype(float)
        C_expected = A @ B
        
        # Compute using decomposition
        C_computed = np.zeros((n, p), dtype=float)
        for c in range(U.shape[1]):
            # Compute intermediate products
            left = np.zeros((n, m), dtype=float)
            right = np.zeros((m, p), dtype=float)
            for i in range(n):
                for j in range(m):
                    left[i, j] = U[i * m + j, c] * A[i, j]
            for j in range(m):
                for k in range(p):
                    right[j, k] = V[j * p + k, c] * B[j, k]
            
            # Now C[i,k] += (sum_j left[i,j]) * (sum_j right[j,k]) * W[k*n+i, c]
            # Wait, that's not right. Let me re-derive.
            
            # Actually, I think the formula should be:
            # C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
            
            # But that doesn't make sense dimensionally. Let me think again.
            
            # For schoolbook with c = (i0, j0, k0):
            # sum_j U[i*m+j, c] * A[i,j] = A[i, j0] if i = i0
            # sum_j V[j*p+k, c] * B[j,k] = B[j0, k] if k = k0
            # W[k*n+i, c] = 1 if (k,i) = (k0,i0)
            
            # So the product is A[i0,j0] * B[j0,k0] when i=i0, k=k0
            # And we sum over all c = (i0,j0,k0)
            # C[i,k] = sum_{i0,j0,k0} A[i0,j0]*B[j0,k0] * δ_{i,i0} * δ_{k,k0}
            # = A[i,j0]*B[j0,k] summed over j0
            # = sum_j A[i,j]*B[j,k]
            
            # OK so the formula is correct. Let me implement it properly.
            
            for i in range(n):
                for k in range(p):
                    if W[k * n + i, c] != 0:
                        left_sum = 0.0
                        for j in range(m):
                            left_sum += U[i * m + j, c] * A[i, j]
                        right_sum = 0.0
                        for j in range(m):
                            right_sum += V[j * p + k, c] * B[j, k]
                        C_computed[i, k] += left_sum * right_sum * W[k * n + i, c]
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True


def _to_list(matrix):
    """Convert numpy array to list of lists with integer/half-integer entries."""
    return [[int(round(val * 2)) / 2 for val in row] for row in matrix]


def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _strassen_2x2x2():
    """
    Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) for multiplying 2x2 by 2x2.
    
    The 7 Strassen products:
    M1 = (a11+a22)(b11+b22)
    M2 = (a21+a22)b11
    M3 = a11(b12-b22)
    M4 = a22(b21-b11)
    M5 = (a11+a12)b22
    M6 = (a21-a11)(b11+b12)
    M7 = (a12-a22)(b21+b22)
    
    C11 = M1 - M4 + M5 - M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Using the tensor decomposition convention:
    U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] summed over c gives the tensor entry.
    
    For each Strassen product Mc, we need to express it as a product of three vectors.
    """
    rank = 7
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((4, rank), dtype=float)
    W = np.zeros((4, rank), dtype=float)
    
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # U[0*2+0, 0] = 1 (A[0,0])
    # U[1*2+1, 0] = 1 (A[1,1])
    # V[0*2+0, 0] = 1 (B[0,0])
    # V[1*2+1, 0] = 1 (B[1,1])
    # W[0*2+0, 0] = 1 (for C[0,0])
    # W[1*2+1, 0] = 1 (for C[1,1])
    # But wait, W[k*2+i, c] should be such that when summed, it gives the right C entries.
    # For M1, it contributes to C[0,0] and C[1,1].
    # C[0,0] += M1, so W[0*2+0, 0] = 1
    # C[1,1] += M1, so W[1*2+1, 0] = 1
    
    U[0, 0] = 1  # A[0,0]
    U[3, 0] = 1  # A[1,1]
    V[0, 0] = 1  # B[0,0]
    V[3, 0] = 1  # B[1,1]
    W[0, 0] = 1  # C[0,0]
    W[3, 0] = 1  # C[1,1]
    
    # M2 = (A[1,0] + A[1,1]) * B[0,0]
    U[2, 1] = 1  # A[1,0]
    U[3, 1] = 1  # A[1,1]
    V[0, 1] = 1  # B[0,0]
    W[2, 1] = 1  # C[1,0]
    
    # M3 = A[0,0] * (B[0,1] - B[1,1])
    U[0, 2] = 1  # A[0,0]
    V[1, 2] = 1  # B[0,1]
    V[3, 2] = -1  # B[1,1]
    W[1, 2] = 1  # C[0,1]
    
    # M4 = A[1,1] * (B[1,0] - B[0,0])
    U[3, 3] = 1  # A[1,1]
    V[2, 3] = 1  # B[1,0]
    V[0, 3] = -1  # B[0,0]
    W[3, 3] = -1  # C[0,0] (with negative sign in C11 = M1 - M4 + ...)
    W[2, 3] = 1  # C[1,0]
    
    # M5 = (A[0,0] + A[0,1]) * B[1,1]
    U[0, 4] = 1  # A[0,0]
    U[1, 4] = 1  # A[0,1]
    V[3, 4] = 1  # B[1,1]
    W[0, 4] = 1  # C[0,0]
    W[1, 4] = 1  # C[0,1]
    
    # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    U[2, 5] = 1  # A[1,0]
    U[0, 5] = -1  # A[0,0]
    V[0, 5] = 1  # B[0,0]
    V[1, 5] = 1  # B[0,1]
    W[3, 5] = 1  # C[1,1]
    
    # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    U[1, 6] = 1  # A[0,1]
    U[3, 6] = -1  # A[1,1]
    V[2, 6] = 1  # B[1,0]
    V[3, 6] = 1  # B[1,1]
    W[0, 6] = -1  # C[0,0]
    
    return U, V, W


def _winograd_4x4x4():
    """
    Winograd frequency-domain algorithm for 4x4x4 matrix multiplication.
    This uses the F(2x2, 2) Winograd algorithm applied to 4x4 matrices.
    
    The key idea: use convolution theorem. C = A*B can be computed as:
    C = F^{-1} * (F(A) * F(B)) where F is a transform.
    
    For 4x4, we can use a 2x2 block approach or a direct Winograd transform.
    
    Actually, let me use a simpler approach: decompose 4x4x4 as a combination
    of smaller blocks.
    
    4x4x4 can be seen as 2x2 blocks of 2x2 matrices.
    C = A*B where A, B, C are 4x4.
    Write A = [[A11, A12], [A21, A22]] where each Ai,j is 2x2.
    Similarly for B and C.
    
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    Each of these is a sum of two 2x2x2 products.
    Using Strassen for each 2x2x2, we get 7 products per 2x2x2.
    But we can share some products.
    
    Actually, the total number of 2x2x2 multiplications is 8 (one for each block product).
    Using Strassen, each 2x2x2 takes 7 scalar multiplications.
    Total: 8 * 7 = 56 scalar multiplications.
    
    But the rank of the tensor decomposition is different from the number of scalar multiplications.
    
    Let me think about this differently. The rank of the 4x4x4 matrix multiplication tensor
    is known to be at most 49 (using Winograd) and at least 49 (lower bound).
    
    Actually, the known upper bound for 4x4x4 is 49 using the Winograd algorithm.
    
    Let me implement the Winograd F(2x2, 2) algorithm for 4x4.
    
    The F(2x2, 2) Winograd algorithm computes 2x2 matrix products using 5 scalar multiplications
    instead of 4 (schoolbook) or 7 (Strassen). Wait, that's not right.
    
    Actually, F(2x2, 2) uses 5 multiplications for a 2x2 product, but that's for a specific
    structured case. For general 2x2, Strassen's 7 is better.
    
    Let me reconsider. For 4x4x4, the best known algorithm is based on:
    - Treating 4x4 as 2x2 blocks of 2x2
    - Using Strassen for the 2x2x2 subproblems
    
    This gives rank = 8 * 7 = 56 for the 4x4x4 tensor.
    
    But there are better algorithms. The Winograd algorithm for 4x4x4 has rank 49.
    
    Let me implement the Winograd F(4x4, 2) algorithm or a similar approach.
    
    Actually, I'll use a simpler approach: combine the 2x2x2 Strassen decomposition
    with appropriate block structure.
    
    For 4x4x4, write the matrices as 2x2 blocks:
    A = [[A11, A12], [A21, A22]]
    B = [[B11, B12], [B21, B22]]
    C = [[C11, C12], [C21, C22]]
    
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    Each Ci,j is a sum of two 2x2x2 products.
    Using Strassen for each, we need 7 multiplications per 2x2x2.
    Total: 8 * 7 = 56.
    
    But we can do better by sharing. Let me use the full Strassen recursion:
    4x4x4 = 2x(2x2x2) at the block level, then each 2x2x2 uses Strassen.
    
    Actually, the standard way to get 4x4x4 from 2x2x2 Strassen is:
    Use the Kronecker product of two 2x2x2 Strassen decompositions.
    This gives rank 7*7 = 49.
    
    Let me implement this using the Kronecker product.
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    # Kronecker product for 4x4x4
    # U has shape (n*m, rank) = (16, 49)
    # V has shape (m*p, rank) = (16, 49)
    # W has shape (p*n, rank) = (16, 49)
    
    n1, m1, p1 = 2, 2, 2
    n2, m2, p2 = 2, 2, 2
    
    U_kron = np.kron(U2, U2)
    V_kron = np.kron(V2, V2)
    W_kron = np.kron(W2, W2)
    
    # Permute rows to match the correct index ordering.
    # For U: row index should be (i1*n2+i2)*(m1*m2) + (j1*m2+j2)
    # np.kron gives: (i1*m1+j1)*(m1*m2) + (i2*m2+j2)
    
    perm_U = np.zeros(n1*n2 * m1*m2, dtype=int)
    for i1 in range(n1):
        for j1 in range(m1):
            for i2 in range(n2):
                for j2 in range(m2):
                    kron_row = (i1*m1 + j1) * (m1*m2) + (i2*m2 + j2)
                    correct_row = (i1*n2 + i2) * (m1*m2) + (j1*m2 + j2)
                    perm_U[kron_row] = correct_row
    
    U_444 = U_kron[perm_U, :]
    
    # For V: row index should be (j1*m2+j2)*(p1*p2) + (k1*p2+k2)
    # np.kron gives: (j1*p1+k1)*(p1*p2) + (j2*p2+k2)
    
    perm_V = np.zeros(m1*m2 * p1*p2, dtype=int)
    for j1 in range(m1):
        for k1 in range(p1):
            for j2 in range(m2):
                for k2 in range(p2):
                    kron_row = (j1*p1 + k1) * (p1*p2) + (j2*p2 + k2)
                    correct_row = (j1*m2 + j2) * (p1*p2) + (k1*p2 + k2)
                    perm_V[kron_row] = correct_row
    
    V_444 = V_kron[perm_V, :]
    
    # For W: row index should be (k1*p2+k2)*(n1*n2) + (i1*n2+i2)
    # np.kron gives: (k1*n1+i1)*(n1*n2) + (k2*n2+i2)
    
    perm_W = np.zeros(p1*p2 * n1*n2, dtype=int)
    for k1 in range(p1):
        for i1 in range(n1):
            for k2 in range(p2):
                for i2 in range(n2):
                    kron_row = (k1*n1 + i1) * (n1*n2) + (k2*n2 + i2)
                    correct_row = (k1*p2 + k2) * (n1*n2) + (i1*n2 + i2)
                    perm_W[kron_row] = correct_row
    
    W_444 = W_kron[perm_W, :]
    
    return U_444, V_444, W_444


def _expand_4x4x4_to_4x4x9(U444, V444, W444):
    """
    Expand a 4x4x4 decomposition to 4x4x9.
    
    For 4x4x9, we multiply a 4x4 matrix by a 4x9 matrix.
    We can view the 4x9 matrix as three 4x3 matrices, or use a different approach.
    
    Actually, the simplest approach is to use the schoolbook method for the extra columns.
    But that would give rank 49 + 4*4*5 = 49 + 80 = 129, which is worse than schoolbook (4*4*9=144).
    
    A better approach: use the 4x4x4 decomposition for the first 4 columns,
    and schoolbook for the remaining 5 columns.
    
    Rank = 49 + 4*4*5 = 49 + 80 = 129.
    
    Or we can use a more clever approach. Let me think...
    
    Actually, for 4x4x9, we can use:
    - 4x4x4 Strassen (rank 49) for the first 4 columns
    - 4x4x5 schoolbook (rank 80) for the last 5 columns
    Total rank = 129.
    
    But schoolbook for 4x4x9 is rank 144, so 129 is an improvement.
    
    Let me implement this.
    """
    n, m, p = 4, 4, 9
    p1, p2 = 4, 5
    
    # 4x4x4 decomposition (Strassen)
    U1, V1, W1 = U444, V444, W444
    r1 = U1.shape[1]
    
    # 4x4x5 schoolbook
    r2 = n * m * p2  # 4*4*5 = 80
    U2 = np.zeros((n * m, r2), dtype=float)
    V2 = np.zeros((m * p2, r2), dtype=float)
    W2 = np.zeros((p2 * n, r2), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p2):
                U2[i * m + j, col] = 1
                V2[j * p2 + k, col] = 1
                W2[k * n + i, col] = 1
                col += 1
    
    # Combine: U has shape (n*m, r1+r2) = (16, 129)
    # V has shape (m*p, r1+r2) = (36, 129)
    # W has shape (p*n, r1+r2) = (36, 129)
    
    U = np.hstack([U1, U2])
    
    # V needs to be expanded from (m*p1, r1) to (m*p, r1)
    # V1 has shape (4*4, 49) = (16, 49)
    # We need V to have shape (4*9, 129) = (36, 129)
    # For the first 4 columns (p1=4), use V1
    # For the last 5 columns (p2=5), use V2
    
    V = np.zeros((m * p, r1 + r2), dtype=float)
    V[0:m*p1, 0:r1] = V1  # First 16 rows, first 49 columns
    V[m*p1:m*p, r1:r1+r2] = V2  # Last 20 rows, last 80 columns
    
    # W needs to be expanded from (p1*n, r1) to (p*n, r1)
    # W1 has shape (4*4, 49) = (16, 49)
    # We need W to have shape (9*4, 129) = (36, 129)
    # For the first 4 k-values (p1=4), use W1
    # For the last 5 k-values (p2=5), use W2
    
    W = np.zeros((p * n, r1 + r2), dtype=float)
    # W is indexed by k*n + i
    # For k in [0,4), use W1[k*n+i, :]
    # For k in [4,9), use W2[(k-4)*n+i, r1:]
    for k in range(p1):
        for i in range(n):
            W[k * n + i, 0:r1] = W1[k * n + i, :]
    for k in range(p2):
        for i in range(n):
            W[(k + p1) * n + i, r1:r1+r2] = W2[k * n + i, :]
    
    return U, V, W


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4x9 matrix multiplication.
    """
    start_time = time.time()
    deadline = 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    
    try:
        # Get 4x4x4 Strassen decomposition
        U444, V444, W444 = _winograd_4x4x4()
        
        if time.time() > deadline:
            U, V, W = _schoolbook(n, m, p)
            return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
        
        # Expand to 4x4x9
        U, V, W = _expand_4x4x4_to_4x4x9(U444, V444, W444)
        
        if time.time() > deadline:
            U, V, W = _schoolbook(n, m, p)
            return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
        
        # Verify dimensions
        if U.shape != (n * m, U.shape[1]) or V.shape != (m * p, V.shape[1]) or W.shape != (p * n, W.shape[1]):
            U, V, W = _schoolbook(n, m, p)
            return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
        
        # Quick verification
        if time.time() < deadline - 30:
            if not _validate(U, V, W, n, m, p, seed=42, num_tests=2):
                # Fallback to schoolbook if validation fails
                U, V, W = _schoolbook(n, m, p)
        
        return {
            "U": _to_list(U),
            "V": _to_list(V),
            "W": _to_list(W)
        }
        
    except Exception:
        # Fallback to schoolbook on any error
        U, V, W = _schoolbook(n, m, p)
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

# EVOLVE-BLOCK-END
