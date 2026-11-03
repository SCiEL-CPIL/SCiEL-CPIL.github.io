import numpy as np
import time

def construct():
    start_time = time.time()
    
    n, m, p = 3, 5, 7
    nm = n * m
    mp = m * p
    pn = p * n
    
    # We will build the decomposition.
    # U: (3*5, rank)
    # V: (5*7, rank)
    # W: (7*3, rank)
    
    # We partition the output C (3x7) into blocks.
    # Block 1: Rows 0,1; Cols 0,1. (2x2 output). Middle dim 5.
    # Block 2: Rows 0,1; Cols 2,3. (2x2 output). Middle dim 5.
    # Block 3: Rows 0,1; Cols 4,5. (2x2 output). Middle dim 5.
    # Block 4: Rows 0,1; Col 6. (2x1 output). Middle dim 5.
    # Block 5: Row 2; Cols 0,1,2,3,4,5,6. (1x7 output). Middle dim 5.
    
    # For Blocks 1-3 (2x2 output, 2x5 input A, 5x2 input B):
    # Split middle dim j=0..4 into {0,1}, {2,3}, {4}.
    # j in {0,1}: 2x2 by 2x2. Use Strassen (rank 7).
    # j in {2,3}: 2x2 by 2x2. Use Strassen (rank 7).
    # j in {4}: 2x1 by 1x2. Use Schoolbook (rank 4).
    # Total rank per block: 7 + 7 + 4 = 18.
    # 3 blocks * 18 = 54.
    
    # For Block 4 (2x1 output, 2x5 input A, 5x1 input B):
    # Schoolbook: 2*5*1 = 10 terms.
    
    # For Block 5 (1x7 output, 1x5 input A, 5x7 input B):
    # Schoolbook: 1*5*7 = 35 terms.
    
    # Total Rank: 54 + 10 + 35 = 99.
    
    rank_terms = []
    
    # Helper to add a schoolbook term (i, j, k)
    def add_schoolbook(i, j, k):
        U = np.zeros(nm)
        V = np.zeros(mp)
        W = np.zeros(pn)
        U[i*m + j] = 1.0
        V[j*p + k] = 1.0
        W[k*n + i] = 1.0
        rank_terms.append((U, V, W))
        
    # Helper to add Strassen terms for a 2x2 block
    # i0, i1: rows of A (and C)
    # k0, k1: cols of B (and C)
    # j0, j1: cols of A / rows of B
    def add_strassen_block(i0, i1, k0, k1, j0, j1):
        # A elements: A[i0,j0], A[i0,j1], A[i1,j0], A[i1,j1]
        # B elements: B[j0,k0], B[j0,k1], B[j1,k0], B[j1,k1]
        # C elements: C[i0,k0], C[i0,k1], C[i1,k0], C[i1,k1]
        
        # Indices in flattened vectors
        A00 = i0*m + j0
        A01 = i0*m + j1
        A10 = i1*m + j0
        A11 = i1*m + j1
        
        B00 = j0*p + k0
        B01 = j0*p + k1
        B10 = j1*p + k0
        B11 = j1*p + k1
        
        C00 = k0*n + i0
        C01 = k1*n + i0
        C10 = k0*n + i1
        C11 = k1*n + i1
        
        def make_U(indices_values):
            vec = np.zeros(nm)
            for idx, val in indices_values:
                vec[idx] += val
            return vec
            
        def make_V(indices_values):
            vec = np.zeros(mp)
            for idx, val in indices_values:
                vec[idx] += val
            return vec
            
        def make_W(indices_values):
            vec = np.zeros(pn)
            for idx, val in indices_values:
                vec[idx] += val
            return vec
            
        # M1 = (A00 + A11) * (B00 + B11)
        U1 = make_U([(A00, 1), (A11, 1)])
        V1 = make_V([(B00, 1), (B11, 1)])
        W1 = make_W([(C00, 1), (C11, 1)])
        rank_terms.append((U1, V1, W1))
        
        # M2 = (A10 + A11) * B00
        U2 = make_U([(A10, 1), (A11, 1)])
        V2 = make_V([(B00, 1)])
        W2 = make_W([(C10, 1), (C11, -1)])
        rank_terms.append((U2, V2, W2))
        
        # M3 = A00 * (B10 - B11)
        U3 = make_U([(A00, 1)])
        V3 = make_V([(B10, 1), (B11, -1)])
        W3 = make_W([(C01, 1), (C11, 1)])
        rank_terms.append((U3, V3, W3))
        
        # M4 = A11 * (B01 - B00)
        U4 = make_U([(A11, 1)])
        V4 = make_V([(B01, 1), (B00, -1)])
        W4 = make_W([(C00, 1), (C10, 1)])
        rank_terms.append((U4, V4, W4))
        
        # M5 = (A00 + A01) * B11
        U5 = make_U([(A00, 1), (A01, 1)])
        V5 = make_V([(B11, 1)])
        W5 = make_W([(C01, 1), (C00, -1)])
        rank_terms.append((U5, V5, W5))
        
        # M6 = (A10 - A00) * (B00 + B01)
        U6 = make_U([(A10, 1), (A00, -1)])
        V6 = make_V([(B00, 1), (B01, 1)])
        W6 = make_W([(C11, 1)])
        rank_terms.append((U6, V6, W6))
        
        # M7 = (A01 - A11) * (B10 + B11)
        U7 = make_U([(A01, 1), (A11, -1)])
        V7 = make_V([(B10, 1), (B11, 1)])
        W7 = make_W([(C00, 1)])
        rank_terms.append((U7, V7, W7))

    # --- Block 1: i in {0,1}, k in {0,1} ---
    add_strassen_block(0, 1, 0, 1, 0, 1)
    add_strassen_block(0, 1, 0, 1, 2, 3)
    # j=4 schoolbook
    add_schoolbook(0, 4, 0)
    add_schoolbook(0, 4, 1)
    add_schoolbook(1, 4, 0)
    add_schoolbook(1, 4, 1)

    # --- Block 2: i in {0,1}, k in {2,3} ---
    add_strassen_block(0, 1, 2, 3, 0, 1)
    add_strassen_block(0, 1, 2, 3, 2, 3)
    # j=4 schoolbook
    add_schoolbook(0, 4, 2)
    add_schoolbook(0, 4, 3)
    add_schoolbook(1, 4, 2)
    add_schoolbook(1, 4, 3)

    # --- Block 3: i in {0,1}, k in {4,5} ---
    add_strassen_block(0, 1, 4, 5, 0, 1)
    add_strassen_block(0, 1, 4, 5, 2, 3)
    # j=4 schoolbook
    add_schoolbook(0, 4, 4)
    add_schoolbook(0, 4, 5)
    add_schoolbook(1, 4, 4)
    add_schoolbook(1, 4, 5)

    # --- Block 4: i in {0,1}, k in {6} ---
    # Schoolbook for j=0..4
    for j in range(5):
        add_schoolbook(0, j, 6)
        add_schoolbook(1, j, 6)

    # --- Block 5: i in {2}, k in {0..6} ---
    # Schoolbook for j=0..4, k=0..6
    for j in range(5):
        for k in range(7):
            add_schoolbook(2, j, k)

    R = len(rank_terms)
    
    U_mat = np.zeros((nm, R))
    V_mat = np.zeros((mp, R))
    W_mat = np.zeros((pn, R))
    
    for r, (U, V, W) in enumerate(rank_terms):
        U_mat[:, r] = U
        V_mat[:, r] = V
        W_mat[:, r] = W

    # Verification
    def verify(U, V, W, R):
        # Reshape to (n, m, R), (m, p, R), (p, n, R)
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        
        # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        # Note: W is indexed by (k, i) in the tensor product space for C?
        # The task says W[k*n+i, r].
        # So W_3d[k, i, r] is the entry.
        # The tensor T is sum over r of U[i,j,r] V[j,k,r] W[k,i,r].
        # We want T[i,j,k] to be 1 if it's the identity multiplication tensor?
        # The multiplication tensor M_{n,m,p} has entries 1 at (i, j, k) corresponding to A[i,j] B[j,k] -> C[i,k].
        # So yes, T[i,j,k] should be 1 for all i,j,k?
        # Wait. The tensor is T = sum_{i,j,k} E_{ij}^A \otimes E_{jk}^B \otimes E_{ik}^C.
        # The contraction sum_r U[i,j,r] V[j,k,r] W[k,i,r] should yield 1 for all i,j,k?
        # No.
        # If U, V, W are rank-1 terms u \otimes v \otimes w.
        # The sum is the tensor itself.
        # The entries of the tensor M_{n,m,p} are 1 if the indices match the structure of matrix multiplication.
        # Specifically, M[i,j,k] = 1.
        # So yes, the resulting array should be all 1s.
        
        try:
            T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            return np.allclose(T, 1.0, atol=1e-9)
        except:
            return False

    if verify(U_mat, V_mat, W_mat, R):
        return {
            "U": U_mat.tolist(),
            "V": V_mat.tolist(),
            "W": W_mat.tolist()
        }
    else:
        # Fallback to schoolbook rank 105
        U_school = np.zeros((nm, 105))
        V_school = np.zeros((mp, 105))
        W_school = np.zeros((pn, 105))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col] = 1.0
                    V_school[j * p + k, col] = 1.0
                    W_school[k * n + i, col] = 1.0
                    col += 1
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
