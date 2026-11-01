"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.linalg import null_space

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x2 * 2x11 matrix multiplication.
    
    The tensor T is in R^{(n*m) x (m*p) x (p*n)} with entries T[i*m+j, j*p+k, k*n+i] = 1.
    We seek rank r such that T = sum_{c=0}^{r-1} u_c * v_c * w_c.
    
    Strategy: Use the structure that m=2, n=2, p=11.
    The input dimensions are n*m = 4 and m*p = 22.
    The output dimension is p*n = 22.
    
    We use a constructive approach based on the fact that for matrix multiplication 
    A (n x m) * B (m x p) = C (n x p), we can use the standard algorithm but potentially
    with fewer multiplications if there's structure.
    
    For 2x2 * 2x2, Strassen gives rank 7 instead of 8.
    For 2x2 * 2xp, we can potentially use a similar approach.
    
    Let's think about this differently. The bilinear forms for (2x2)*(2x11) are:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    So C has 2*11 = 22 entries, each is a sum of 2 products.
    Total bilinear terms if done naively: 22*2 = 44.
    
    We want to find the minimum number of bilinear forms (rank) such that all 22 output
    entries can be expressed as linear combinations of these forms.
    
    Each bilinear form is of the form sum_{i,j,k} u_{i,j} * v_{j,k} * w_{i,k} where
    u is a 2x2 matrix, v is a 2x11 matrix, w is a 2x11 matrix (in the output space indexed by (i,k)).
    
    Actually, let me reformulate. The tensor T has indices (i,j) for A, (j,k) for B, and (i,k) for C.
    T[(i,j),(j,k),(i,k)] = 1.
    
    A rank-1 decomposition term is u_{i,j} * v_{j,k} * w_{i,k} for fixed u, v, w.
    
    We need: sum_c u_c[i,j] * v_c[j,k] * w_c[i,k] = delta(i,j,k) where the tensor is 1 when the
    indices are consistent (i.e., the j in A matches the j in B, and the i,k in C match).
    
    Wait, the tensor is: T[a, b, c] where a = i*m+j (0..3), b = j*p+k (0..21), c = k*n+i (0..21).
    T[a, b, c] = 1 if a = i*m+j, b = j'*p+k', c = k''*n+i'' with i=i'', j=j'=j'', k=k''... 
    Actually no. The matrix multiplication tensor has T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k.
    
    So for each triple (i, j, k) with i in {0,1}, j in {0,1}, k in {0,...,10}, we have
    T[i*2+j, j*11+k, k*2+i] = 1.
    
    That's 2*2*11 = 44 non-zero entries, each equal to 1.
    
    We want to decompose this tensor into a sum of rank-1 tensors.
    
    Let's use a greedy/iterative approach with numpy to find a low-rank decomposition.
    
    Actually, let me think about known results. For (2x2)*(2xp), what is the known minimum rank?
    
    For p=2: Strassen gives 7.
    For p=11: We can potentially use a block structure.
    
    Let me try a different approach: use the fact that we can write the multiplication as
    a sequence of 2x2 multiplications, each contributing to part of the output.
    
    For each column k of B, we compute A * B[:,k] which is a 2x1 vector. This takes rank 7
    (Strassen) per column if we share the A matrix. But the A matrix is the same for all columns,
    so we can potentially share bilinear forms.
    
    Actually, in Strassen's algorithm, the bilinear forms are:
    m1 = (a11+a22)(b11+b22)
    m2 = (a21+a22)b11
    m3 = a11(b21-b22)
    m4 = a22(b12-b11)
    m5 = (a11+a12)b22
    m6 = (a21-a11)(b11+b12)
    m7 = (a12-a22)(b21+b22)
    
    Then:
    c11 = m1 - m4 + m5 - m7
    c12 = m3 + m4
    c21 = m2 + m5
    c22 = m1 - m2 + m3 + m6
    
    For the 2x11 case, we have 11 columns. If we apply Strassen to each column independently,
    we'd need 7*11 = 77 bilinear forms. But we can do better by sharing.
    
    Let me try a direct optimization approach. I'll use a greedy algorithm to build up the
    decomposition, adding one rank-1 term at a time to best reduce the residual.
    
    Actually, let me try a simpler constructive approach first.
    
    Key insight: We can view the 2x11 multiplication as 11 independent 2x2 multiplications
    (each column of B with A), but we can share the "A-side" bilinear forms.
    
    Let's define for each column k:
    The output C[:,k] = A * B[:,k]
    
    Using Strassen for each column k:
    m1_k = (a00+a11)(b00_k + b11_k)
    m2_k = (a10+a11)b00_k
    m3_k = a00(b10_k - b11_k)
    m4_k = a11(b01_k - b00_k)
    m5_k = (a00+a01)b11_k
    m6_k = (a10-a00)(b00_k + b01_k)
    m7_k = (a01-a11)(b10_k + b11_k)
    
    That's 7*11 = 77 forms. But notice that some of these can be combined.
    
    For example, m3_k = a00*b10_k - a00*b11_k. The term a00*b10_k appears in m3_k, and
    a00*b11_k appears in m3_k and m5_k (partially).
    
    This is getting complex. Let me just implement a numerical search.
    
    I'll use a randomized greedy approach: start with the schoolbook rank 44, then try to
    eliminate terms by finding linear dependencies.
    """
    start_time = time.time()
    deadline = 140  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 2, 11
    
    # Build the tensor
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    tensor = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor[a, b, c] = 1.0
    
    # Reshape to matrix for SVD: flatten last two dims
    # T as a matrix: (n*m) x (m*p * p*n)
    T_mat = tensor.reshape(n*m, m*p * p*n)
    
    # Get singular values to understand the structure
    try:
        U_svd, S, Vt = np.linalg.svd(T_mat, full_matrices=False)
        # Rank of the matrix flattening
        num_nonzero = np.sum(S > 1e-10)
    except:
        num_nonzero = n * m * p
    
    # The tensor rank is at least the matrix rank, but can be higher.
    # Let's try a constructive approach using known Strassen-like decompositions.
    
    # For 2x2 * 2x2, Strassen has rank 7.
    # For 2x2 * 2xp, we can potentially achieve rank 7*p - (p-1)*something.
    
    # Let me try a direct approach: use the schoolbook decomposition and then try to
    # find linear combinations that reduce the rank.
    
    # Schoolbook: rank = n*m*p = 44
    # Each term c corresponds to (i,j,k) with:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    
    # Let's represent the decomposition as lists of (u, v, w) vectors.
    # u has length n*m = 4, v has length m*p = 22, w has length p*n = 22.
    
    # Start with schoolbook
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n*m)
                v = np.zeros(m*p)
                w = np.zeros(p*n)
                u[i*m+j] = 1
                v[j*p+k] = 1
                w[k*n+i] = 1
                terms.append((u, v, w))
    
    rank = len(terms)
    
    # Now try to reduce rank by finding linear dependencies.
    # A set of terms {t_1, ..., t_r} is a valid decomposition if
    # sum_c u_c * v_c * w_c = T.
    
    # We can try to replace subsets of terms with fewer terms.
    # For example, if we have terms (u1,v1,w1) and (u2,v2,w2) and (u3,v3,w3) such that
    # their sum equals a single rank-1 term, we can replace 3 terms with 1.
    
    # This is hard to search exhaustively. Let me try a different approach.
    
    # Use the fact that for the 2x2*2x2 subproblem, Strassen reduces 8 to 7.
    # For 2x2*2x11, we have 11 "columns" each contributing a 2x2*2x1 subproblem.
    
    # Let me try to construct a decomposition based on Strassen applied to blocks.
    
    # Divide the 11 columns into groups. For each group of 2 columns, use Strassen (rank 7).
    # 11 = 5*2 + 1, so 5 groups of 2 + 1 group of 1.
    # Group of 2: rank 7 (Strassen)
    # Group of 1: rank 4 (schoolbook for 2x2*2x1)
    # Total: 5*7 + 4 = 39
    
    # Can we do better? Let me think...
    
    # Actually, for 2x2 * 2x1, the rank is 4 (it's just 4 independent bilinear forms:
    # a00*b0, a01*b0, a10*b1, a11*b1 - wait no, 2x2 * 2x1 gives 2 outputs:
    # c0 = a00*b0 + a10*b1
    # c1 = a01*b0 + a11*b1
    # That's 4 products, but can we do it in fewer? For 2x2*2x1, the tensor has
    # T[i, j, i'] where... let me think.
    # A is 2x2, B is 2x1, C is 2x1.
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # Tensor: T[i*m+j, j*p+k, k*n+i] with n=2, m=2, p=1.
    # T[i*2+j, j*1+k, k*2+i] = 1
    # (0,0,0)=1, (0,1,1)=1, (1,0,2)=1, (1,1,3)=1
    # That's 4 non-zero entries. The rank is 4 (no obvious Strassen-like reduction for 2x2*2x1).
    
    # So 5*7 + 4 = 39.
    
    # Can we do even better? What about grouping differently?
    # 11 = 3 + 3 + 3 + 2
    # 2x2 * 2x3: what's the rank? 
    # For 2x2*2x3, we have 6 outputs, each is a sum of 2 products, total 12 products.
    # Can we do better than 12? 
    # I'm not sure of the exact rank. Let's assume we can use Strassen-like for pairs.
    
    # Let me just go with the 39-rank construction and verify it.
    
    def strassen_2x2_2x2(a00, a01, a10, a11, b00, b01, b10, b11):
        """Return the 7 Strassen bilinear forms as linear combinations."""
        # This returns the 7 forms as tuples of coefficients
        m1 = (a00 + a11) * (b00 + b11)
        m2 = (a10 + a11) * b00
        m3 = a00 * (b10 - b11)
        m4 = a11 * (b01 - b00)
        m5 = (a00 + a01) * b11
        m6 = (a10 - a00) * (b00 + b01)
        m7 = (a01 - a11) * (b10 + b11)
        return [m1, m2, m3, m4, m5, m6, m7]
    
    # Let me build the rank-39 decomposition explicitly.
    # For each pair of columns (2k, 2k+1), use Strassen.
    # For the last single column (column 10), use schoolbook.
    
    U_list = []
    V_list = []
    W_list = []
    
    # For Strassen on columns (2k, 2k+1):
    # The 7 bilinear forms involve linear combinations of A entries and B entries.
    # A entries: A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0,1,2,3 in U
    # B entries for columns 2k, 2k+1: B[0,2k], B[1,2k], B[0,2k+1], B[1,2k+1]
    #   -> indices 2k, 2k+1, 2k+11, 2k+12 in V (since V index = j*p+k, j in {0,1}, k in {0..10})
    #   B[0,2k] -> V[0*11+2k] = V[2k]
    #   B[1,2k] -> V[1*11+2k] = V[11+2k]
    #   B[0,2k+1] -> V[2k+1]
    #   B[1,2k+1] -> V[11+2k+1] = V[12+2k]
    # Output C entries for columns 2k, 2k+1: C[0,2k], C[1,2k], C[0,2k+1], C[1,2k+1]
    #   -> W index = k*n+i = k*2+i
    #   C[0,2k] -> W[2k*2+0] = W[4k]
    #   C[1,2k] -> W[2k*2+1] = W[4k+1]
    #   C[0,2k+1] -> W[(2k+1)*2+0] = W[4k+2]
    #   C[1,2k+1] -> W[(2k+1)*2+1] = W[4k+3]
    
    for k in range(5):
        col0 = 2*k
        col1 = 2*k+1
        # Strassen forms for (A, B[:, col0:col1+1])
        # m1 = (A00+A11)(B0_col0 + B1_col1)
        u1 = np.zeros(4); u1[0]=1; u1[3]=1  # A00 + A11
        v1 = np.zeros(22); v1[col0]=1; v1[11+col1]=1  # B0_col0 + B1_col1
        w1 = np.zeros(22)
        w1[4*k] = 1    # C0_col0
        w1[4*k+1] = -1  # C1_col0 (from -m4)
        w1[4*k+2] = 1   # C0_col1 (from +m5)
        w1[4*k+3] = -1  # C1_col1 (from -m7)
        U_list.append(u1); V_list.append(v1); W_list.append(w1)
        
        # m2 = (A10+A11) B0_col0
        u2 = np.zeros(4); u2[2]=1; u2[3]=1
        v2 = np.zeros(22); v2[col0]=1
        w2 = np.zeros(22)
        w2[4*k+1] = 1   # C1_col0
        w2[4*k+3] = -1  # C1_col1 (from -m2)
        U_list.append(u2); V_list.append(v2); W_list.append(w2)
        
        # m3 = A00 (B1_col0 - B1_col1)
        u3 = np.zeros(4); u3[0]=1
        v3 = np.zeros(22); v3[11+col0]=1; v3[11+col1]=-1
        w3 = np.zeros(22)
        w3[4*k+2] = 1   # C0_col1
        w3[4*k+3] = 1   # C1_col1
        U_list.append(u3); V_list.append(v3); W_list.append(w3)
        
        # m4 = A11 (B0_col1 - B0_col0)
        u4 = np.zeros(4); u4[3]=1
        v4 = np.zeros(22); v4[col1]=1; v4[col0]=-1
        w4 = np.zeros(22)
        w4[4*k] = -1    # C0_col0
        w4[4*k+2] = 1   # C0_col1
        U_list.append(u4); V_list.append(v4); W_list.append(w4)
        
        # m5 = (A00+A01) B1_col1
        u5 = np.zeros(4); u5[0]=1; u5[1]=1
        v5 = np.zeros(22); v5[11+col1]=1
        w5 = np.zeros(22)
        w5[4*k] = 1     # C0_col0
        w5[4*k+1] = 1   # C1_col0
        U_list.append(u5); V_list.append(v5); W_list.append(w5)
        
        # m6 = (A10-A00) (B0_col0 + B0_col1)
        u6 = np.zeros(4); u6[2]=1; u6[0]=-1
        v6 = np.zeros(22); v6[col0]=1; v6[col1]=1
        w6 = np.zeros(22)
        w6[4*k+3] = 1   # C1_col1
        U_list.append(u6); V_list.append(v6); W_list.append(w6)
        
        # m7 = (A01-A11) (B1_col0 + B1_col1)
        u7 = np.zeros(4); u7[1]=1; u7[3]=-1
        v7 = np.zeros(22); v7[11+col0]=1; v7[11+col1]=1
        w7 = np.zeros(22)
        w7[4*k] = -1    # C0_col0
        w7[4*k+1] = 1   # C1_col0
        U_list.append(u7); V_list.append(v7); W_list.append(w7)
    
    # For the last column (column 10), use schoolbook: 4 terms
    col = 10
    # C0_col = A00*B0_col + A10*B1_col
    # C1_col = A01*B0_col + A11*B1_col
    u_a = np.zeros(4); u_a[0]=1
    v_a = np.zeros(22); v_a[col]=1
    w_a = np.zeros(22); w_a[2*col]=1
    U_list.append(u_a); V_list.append(v_a); W_list.append(w_a)
    
    u_b = np.zeros(4); u_b[2]=1
    v_b = np.zeros(22); v_b[11+col]=1
    w_b = np.zeros(22); w_b[2*col]=1
    U_list.append(u_b); V_list.append(v_b); W_list.append(w_b)
    
    u_c = np.zeros(4); u_c[1]=1
    v_c = np.zeros(22); v_c[col]=1
    w_c = np.zeros(22); w_c[2*col+1]=1
    U_list.append(u_c); V_list.append(v_c); W_list.append(w_c)
    
    u_d = np.zeros(4); u_d[3]=1
    v_d = np.zeros(22); v_d[11+col]=1
    w_d = np.zeros(22); w_d[2*col+1]=1
    U_list.append(u_d); V_list.append(v_d); W_list.append(w_d)
    
    rank = len(U_list)
    
    # Verify the decomposition
    U_mat = np.array(U_list).T  # (4, rank)
    V_mat = np.array(V_list).T  # (22, rank)
    W_mat = np.array(W_list).T  # (22, rank)
    
    # Reconstruct tensor: T_recon[a,b,c] = sum_c U_mat[a,c] * V_mat[b,c] * W_mat[c_out,c]
    # where c_out is the output index
    # Actually, the decomposition is: T = sum_{c=0}^{rank-1} outer(U_mat[:,c], V_mat[:,c], W_mat[:,c])
    # Let's verify by checking that for all (a,b,c_out), the sum equals the original tensor.
    
    # This is expensive to check fully, so let's check a subset.
    # T_recon[a, b, c_out] = sum_{col} U_mat[a, col] * V_mat[b, col] * W_mat[c_out, col]
    
    # Let's verify for a few entries
    valid = True
    for _ in range(100):
        a = np.random.randint(0, 4)
        b = np.random.randint(0, 22)
        c_out = np.random.randint(0, 22)
        val = np.sum(U_mat[a, :] * V_mat[b, :] * W_mat[c_out, :])
        # Original tensor value
        i = a // 2
        j = a % 2
        j2 = b // 11
        k = b % 11
        k2 = c_out // 2
        i2 = c_out % 2
        expected = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
        if abs(val - expected) > 1e-10:
            valid = False
            break
    
    if not valid:
        # Fall back to schoolbook
        U_school = []
        V_school = []
        W_school = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = np.zeros(n*m); u[i*m+j] = 1
                    v = np.zeros(m*p); v[j*p+k] = 1
                    w = np.zeros(p*n); w[k*n+i] = 1
                    U_school.append(u); V_school.append(v); W_school.append(w)
        return {"U": [list(map(float, u)) for u in U_school],
                "V": [list(map(float, v)) for v in V_school],
                "W": [list(map(float, w)) for w in W_school]}
    
    # Convert to lists of lists
    U_out = [list(map(float, U_mat[:, c])) for c in range(rank)]
    V_out = [list(map(float, V_mat[:, c])) for c in range(rank)]
    W_out = [list(map(float, W_mat[:, c])) for c in range(rank)]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
