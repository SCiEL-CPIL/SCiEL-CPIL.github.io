"""Program searching for rank-minimal exact tensor decomposition for 5x5x5 matmul."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Convention: W[k*n + i, c]
    Target: T[i, j, k] = 1 if k == j (i.e., C_{k,i} = sum_j A_{i,j} B_{j,k})
    Tensor component (i, k, j) should equal 1 if k == j, else 0.
    Sum_c U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c] == T[i,k,j]
    """
    rank = U[0].__len__() if len(U) > 0 and len(U[0]) > 0 else V[0].__len__() if len(V) > 0 and len(V[0]) > 0 else W[0].__len__()
    # Convert to arrays for faster verification
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    
    # Use batch operations for verification
    # T[i, k, j] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # Reshape for tensor contraction
    # U[i*m+j, c] -> U_t[i, j, c]
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    
    # Compute product tensor P[i, j, k] = sum_c U_t[i, j, c] V_t[j, k, c] W_t[k, i, c]
    # This is a contraction over c, with i, j, k, but W has indices k, i, c
    # So P[i, j, k] = sum_c U_t[i, j, c] * V_t[j, k, c] * W_t[k, i, c]
    # Let's use einsum: 'ijc,jkc,kic->ijk'
    try:
        P = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    except Exception:
        # Fallback to loops if einsum fails (should not)
        P = np.zeros((n, m, p), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    P[i, j, k] = s
    
    # Target tensor T[i, j, k] = 1 if j == k, else 0
    T_target = (np.arange(m) == np.arange(p)).reshape(n, m, p).astype(np.int64)
    
    return np.all(P == T_target)

def _strassen_4x4():
    """
    Return Strassen's 7-term decomposition for 4x4 matrix multiplication.
    Uses the standard Strassen algorithm applied to 2x2 blocks of 2x2 matrices.
    Returns (U, V, W) each of shape (16, 7) for the 4x4x4 tensor.
    """
    # Strassen's 7 multiplications for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 4x4, we split into 2x2 blocks of 2x2 matrices.
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    # B = [[B11, B12], [B21, B22]] where each Bij is 2x2
    # C = [[C11, C12], [C21, C22]]
    
    # The tensor indices for 4x4: A[i,j], B[j,k], C[k,i]
    # i,j,k in {0,1,2,3}
    # Block structure: 
    #   i in {0,1}: block row 1, i in {2,3}: block row 2
    #   j in {0,1}: block col 1, j in {2,3}: block col 2
    #   k in {0,1}: block col 1 for B, k in {2,3}: block col 2 for B
    
    # We'll construct U, V, W of shape (16, 7)
    U = [[0]*7 for _ in range(16)]  # A indices: i*4+j
    V = [[0]*7 for _ in range(16)]  # B indices: j*4+k
    W = [[0]*7 for _ in range(16)]  # C indices: k*4+i
    
    # Define the 7 Strassen terms for the block level
    # Each term contributes to specific (i,j,k) positions
    
    # M1 = (A11+A22)(B11+B22)
    # A11: i in {0,1}, j in {0,1}
    # A22: i in {2,3}, j in {2,3}
    # B11: j in {0,1}, k in {0,1}
    # B22: j in {2,3}, k in {2,3}
    # C contribution: C11 gets +M1, C22 gets +M1
    # C11: k in {0,1}, i in {0,1}
    # C22: k in {2,3}, i in {2,3}
    for i in range(4):
        for j in range(4):
            # A[i,j] contributes to M1 if (i,j) in A11 or A22
            if (i < 2 and j < 2) or (i >= 2 and j >= 2):
                U[i*4+j][0] = 1
    for j in range(4):
        for k in range(4):
            # B[j,k] contributes to M1 if (j,k) in B11 or B22
            if (j < 2 and k < 2) or (j >= 2 and k >= 2):
                V[j*4+k][0] = 1
    for k in range(4):
        for i in range(4):
            # C[k,i] gets +M1 if (k,i) in C11 or C22
            if (k < 2 and i < 2) or (k >= 2 and i >= 2):
                W[k*4+i][0] = 1
    
    # M2 = (A21+A22)B11
    # A21: i in {2,3}, j in {0,1}
    # A22: i in {2,3}, j in {2,3}
    # B11: j in {0,1}, k in {0,1}
    # C contribution: C11 gets -M2, C21 gets +M2
    # C11: k in {0,1}, i in {0,1}
    # C21: k in {2,3}, i in {0,1}
    for i in range(4):
        for j in range(4):
            if (i >= 2 and j < 2) or (i >= 2 and j >= 2):
                U[i*4+j][1] = 1
    for j in range(4):
        for k in range(4):
            if (j < 2 and k < 2):
                V[j*4+k][1] = 1
    for k in range(4):
        for i in range(4):
            if (k < 2 and i < 2):
                W[k*4+i][1] = -1
            elif (k >= 2 and i < 2):
                W[k*4+i][1] = 1
    
    # M3 = A11(B21-B22)
    # A11: i in {0,1}, j in {0,1}
    # B21: j in {2,3}, k in {0,1}
    # B22: j in {2,3}, k in {2,3}
    # C contribution: C12 gets +M3, C22 gets +M3
    # C12: k in {0,1}, i in {2,3}
    # C22: k in {2,3}, i in {2,3}
    for i in range(4):
        for j in range(4):
            if (i < 2 and j < 2):
                U[i*4+j][2] = 1
    for j in range(4):
        for k in range(4):
            if (j >= 2 and k < 2):
                V[j*4+k][2] = 1
            elif (j >= 2 and k >= 2):
                V[j*4+k][2] = -1
    for k in range(4):
        for i in range(4):
            if (k < 2 and i >= 2):
                W[k*4+i][2] = 1
            elif (k >= 2 and i >= 2):
                W[k*4+i][2] = 1
    
    # M4 = A22(B12-B11)
    # A22: i in {2,3}, j in {2,3}
    # B12: j in {0,1}, k in {2,3}
    # B11: j in {0,1}, k in {0,1}
    # C contribution: C11 gets +M4, C21 gets +M4
    # C11: k in {0,1}, i in {0,1}
    # C21: k in {2,3}, i in {0,1}
    for i in range(4):
        for j in range(4):
            if (i >= 2 and j >= 2):
                U[i*4+j][3] = 1
    for j in range(4):
        for k in range(4):
            if (j < 2 and k >= 2):
                V[j*4+k][3] = 1
            elif (j < 2 and k < 2):
                V[j*4+k][3] = -1
    for k in range(4):
        for i in range(4):
            if (k < 2 and i < 2):
                W[k*4+i][3] = 1
            elif (k >= 2 and i < 2):
                W[k*4+i][3] = 1
    
    # M5 = (A11+A12)B22
    # A11: i in {0,1}, j in {0,1}
    # A12: i in {0,1}, j in {2,3}
    # B22: j in {2,3}, k in {2,3}
    # C contribution: C11 gets -M5, C12 gets +M5
    # C11: k in {0,1}, i in {0,1}
    # C12: k in {0,1}, i in {2,3}
    for i in range(4):
        for j in range(4):
            if (i < 2 and j < 2) or (i < 2 and j >= 2):
                U[i*4+j][4] = 1
    for j in range(4):
        for k in range(4):
            if (j >= 2 and k >= 2):
                V[j*4+k][4] = 1
    for k in range(4):
        for i in range(4):
            if (k < 2 and i < 2):
                W[k*4+i][4] = -1
            elif (k < 2 and i >= 2):
                W[k*4+i][4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    # A21: i in {2,3}, j in {0,1}
    # A11: i in {0,1}, j in {0,1}
    # B11: j in {0,1}, k in {0,1}
    # B12: j in {0,1}, k in {2,3}
    # C contribution: C22 gets +M6
    # C22: k in {2,3}, i in {2,3}
    for i in range(4):
        for j in range(4):
            if (i >= 2 and j < 2):
                U[i*4+j][5] = 1
            elif (i < 2 and j < 2):
                U[i*4+j][5] = -1
    for j in range(4):
        for k in range(4):
            if (j < 2 and k < 2) or (j < 2 and k >= 2):
                V[j*4+k][5] = 1
    for k in range(4):
        for i in range(4):
            if (k >= 2 and i >= 2):
                W[k*4+i][5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    # A12: i in {0,1}, j in {2,3}
    # A22: i in {2,3}, j in {2,3}
    # B21: j in {2,3}, k in {0,1}
    # B22: j in {2,3}, k in {2,3}
    # C contribution: C11 gets +M7
    # C11: k in {0,1}, i in {0,1}
    for i in range(4):
        for j in range(4):
            if (i < 2 and j >= 2):
                U[i*4+j][6] = 1
            elif (i >= 2 and j >= 2):
                U[i*4+j][6] = -1
    for j in range(4):
        for k in range(4):
            if (j >= 2 and k < 2) or (j >= 2 and k >= 2):
                V[j*4+k][6] = 1
    for k in range(4):
        for i in range(4):
            if (k < 2 and i < 2):
                W[k*4+i][6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to build a hybrid decomposition using Strassen for the 4x4 block
    # Split: A = [A44, A41; A14, A11], B = [B44, B41; B14, B11]
    # where A44 is 4x4, A41 is 4x1, A14 is 1x4, A11 is 1x1
    
    # C44 = A44*B44 + A41*B14
    # C41 = A44*B41 + A41*B11
    # C14 = A14*B44 + A11*B14
    # C11 = A14*B41 + A11*B11
    
    # Rank contributions:
    # A44*B44: 49 (Strassen)
    # A41*B14: 1 (outer product of 4x1 and 1x4)
    # A44*B41: 8 (4x1 matmul, each row needs 4 mults... wait, A44 is 4x4, B41 is 4x1)
    #   Actually A44*B41 is 4x1 result. Each element is dot product of 4-dim vectors.
    #   This requires 4*1 = 4 multiplications? No, it's 4 separate dot products of length 4.
    #   Each dot product can be done in rank 4 (schoolbook for 1x4x4).
    #   So total rank = 4*4 = 16? Or can we do better?
    #   Actually, A44*B41: C[i,4] = sum_j A[i,j] B[j,4] for i=0..3
    #   This is 4 independent 4-term dot products, each rank 4, total rank 16.
    #   But wait, we can share some structure? Probably not easily.
    #   Let's use rank 16 for A44*B41.
    # A41*B11: 4 (A41 is 4x1, B11 is 1x1, so C[i,4] += A[i,4]*B[4,4])
    #   Each element is a scalar multiply, rank 1 each, total rank 4.
    # A14*B44: 16 (similar to A44*B41, 4 independent dot products)
    # A11*B14: 4 (A11 is 1x1, B14 is 1x4, C[4,k] += A[4,4]*B[4,k])
    # A14*B41: 4 (dot product of two 4-vectors, rank 4)
    # A11*B11: 1 (scalar multiply)
    
    # Total: 49 + 1 + 16 + 4 + 16 + 4 + 4 + 1 = 95
    
    # Let's construct this
    U_hybrid = []
    V_hybrid = []
    W_hybrid = []
    
    # Helper to add a term
    def add_term(u_vals, v_vals, w_vals):
        """Add a rank-1 term to the decomposition.
        u_vals: dict mapping (i,j) -> coefficient
        v_vals: dict mapping (j,k) -> coefficient
        w_vals: dict mapping (k,i) -> coefficient
        """
        max_coeff = max(abs(v) for v in u_vals.values() + v_vals.values() + w_vals.values()) if u_vals else 1
        # Use integer coefficients
        col_idx = len(U_hybrid[0]) if U_hybrid else 0
        
        for i in range(n):
            for j in range(m):
                U_hybrid[i * m + j].append(u_vals.get((i, j), 0))
        for j in range(m):
            for k in range(p):
                V_hybrid[j * p + k].append(v_vals.get((j, k), 0))
        for k in range(p):
            for i in range(n):
                W_hybrid[k * n + i].append(w_vals.get((k, i), 0))
    
    # Initialize empty lists
    for i in range(n * m):
        U_hybrid.append([])
    for j in range(m * p):
        V_hybrid.append([])
    for k in range(p * n):
        W_hybrid.append([])
    
    # Term 1: A44*B44 using Strassen (rank 49)
    U_s, V_s, W_s = _strassen_4x4()
    for c in range(7):
        for i in range(4):
            for j in range(4):
                U_hybrid[i * m + j].append(U_s[i * 4 + j][c])
        for j in range(4):
            for k in range(4):
                V_hybrid[j * p + k].append(V_s[j * 4 + k][c])
        for k in range(4):
            for i in range(4):
                W_hybrid[k * n + i].append(W_s[k * 4 + i][c])
    
    # Term 2: A41*B14 (rank 1, outer product)
    # C[i,k] += A[i,4] * B[4,k] for i=0..3, k=0..3
    u_vals = {}
    v_vals = {}
    w_vals = {}
    for i in range(4):
        u_vals[(i, 4)] = 1
    for k in range(4):
        v_vals[(4, k)] = 1
    for i in range(4):
        for k in range(4):
            w_vals[(k, i)] = 1
    add_term(u_vals, v_vals, w_vals)
    
    # Term 3: A44*B41 (rank 16)
    # C[i,4] = sum_{j=0}^{3} A[i,j] * B[j,4] for i=0..3
    for i in range(4):
        for j in range(4):
            u_vals = {(i, j): 1}
            v_vals = {(j, 4): 1}
            w_vals = {(4, i): 1}
            add_term(u_vals, v_vals, w_vals)
    
    # Term 4: A41*B11 (rank 4)
    # C[i,4] += A[i,4] * B[4,4] for i=0..3
    for i in range(4):
        u_vals = {(i, 4): 1}
        v_vals = {(4, 4): 1}
        w_vals = {(4, i): 1}
        add_term(u_vals, v_vals, w_vals)
    
    # Term 5: A14*B44 (rank 16)
    # C[4,k] = sum_{j=0}^{3} A[4,j] * B[j,k] for k=0..3
    for k in range(4):
        for j in range(4):
            u_vals = {(4, j): 1}
            v_vals = {(j, k): 1}
            w_vals = {(k, 4): 1}
            add_term(u_vals, v_vals, w_vals)
    
    # Term 6: A11*B14 (rank 4)
    # C[4,k] += A[4,4] * B[4,k] for k=0..3
    for k in range(4):
        u_vals = {(4, 4): 1}
        v_vals = {(4, k): 1}
        w_vals = {(k, 4): 1}
        add_term(u_vals, v_vals, w_vals)
    
    # Term 7: A14*B41 (rank 4)
    # C[4,4] += sum_{j=0}^{3} A[4,j] * B[j,4]
    for j in range(4):
        u_vals = {(4, j): 1}
        v_vals = {(j, 4): 1}
        w_vals = {(4, 4): 1}
        add_term(u_vals, v_vals, w_vals)
    
    # Term 8: A11*B11 (rank 1)
    # C[4,4] += A[4,4] * B[4,4]
    u_vals = {(4, 4): 1}
    v_vals = {(4, 4): 1}
    w_vals = {(4, 4): 1}
    add_term(u_vals, v_vals, w_vals)
    
    # Check if hybrid is valid
    if time.time() < deadline:
        if _verify_decomposition(U_hybrid, V_hybrid, W_hybrid, n, m, p):
            return {"U": U_hybrid, "V": V_hybrid, "W": W_hybrid}
    
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
