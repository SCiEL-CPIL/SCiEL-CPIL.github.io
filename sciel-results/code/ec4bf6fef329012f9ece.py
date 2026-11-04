import numpy as np

def construct():
    n, m, p = 3, 5, 6
    
    # Helper to create a single schoolbook term
    def schoolbook_term(n, m, p, i, j, k):
        # Returns U, V, W for a single multiplication A[i,j] * B[j,k]
        # U has shape (n*m, 1), V (m*p, 1), W (p*n, 1)
        U = np.zeros((n * m, 1))
        V = np.zeros((m * p, 1))
        W = np.zeros((p * n, 1))
        U[i * m + j, 0] = 1
        V[j * p + k, 0] = 1
        W[k * n + i, 0] = 1
        return U, V, W

    # Helper to combine two decompositions T1 + T2
    # T1 is for (n, m1, p), T2 is for (n, m2, p) where m = m1 + m2
    # The indices for T1 use j in [0, m1), T2 use j in [m1, m)
    def combine_split_m(n, m1, m2, p, U1, V1, W1, U2, V2, W2):
        rank1 = U1.shape[1]
        rank2 = U2.shape[1]
        total_rank = rank1 + rank2
        
        U_total = np.zeros((n * (m1 + m2), total_rank))
        V_total = np.zeros(((m1 + m2) * p, total_rank))
        W_total = np.zeros((p * n, total_rank))
        
        # Place T1
        # U1 rows correspond to i, j (j < m1). Global row index: i * (m1+m2) + j
        # We need to map local row index (i * m1 + j) to global (i * (m1+m2) + j)
        for i in range(n):
            for j in range(m1):
                local_idx = i * m1 + j
                global_idx = i * (m1 + m2) + j
                U_total[global_idx, :rank1] = U1[local_idx, :]
                
        # V1 rows correspond to j, k (j < m1). Global row index: j * p + k
        # Local row index: j * p + k. Same!
        V_total[:m1 * p, :rank1] = V1
        
        # W1 rows correspond to k, i. Global row index: k * n + i. Same!
        W_total[:, :rank1] = W1
        
        # Place T2
        # U2 rows correspond to i, j_local (j_local < m2). Global j = j_local + m1.
        # Global row index: i * (m1+m2) + (j_local + m1)
        for i in range(n):
            for j_local in range(m2):
                local_idx = i * m2 + j_local
                global_j = j_local + m1
                global_idx = i * (m1 + m2) + global_j
                U_total[global_idx, rank1:] = U2[local_idx, :]
                
        # V2 rows correspond to j_local, k. Global j = j_local + m1.
        # Global row index: (j_local + m1) * p + k
        for j_local in range(m2):
            for k in range(p):
                local_idx = j_local * p + k
                global_j = j_local + m1
                global_idx = global_j * p + k
                V_total[global_idx, rank1:] = V2[local_idx, :]
                
        # W2 rows correspond to k, i. Same as W1.
        W_total[:, rank1:] = W2
        
        return U_total, V_total, W_total

    def decompose(n, m, p):
        # Base case: 1x1x1
        if n == 1 and m == 1 and p == 1:
            return schoolbook_term(1, 1, 1, 0, 0, 0)
            
        # Base case: 2x2x2 Strassen
        if n == 2 and m == 2 and p == 2:
            # Strassen decomposition for 2x2x2
            # M1 = (A11+A22)(B11+B22)
            # M2 = (A21+A22)B11
            # M3 = A11(B21-B22)
            # M4 = A22(B12-B11)
            # M5 = (A11+A12)B22
            # M6 = (A21-A11)(B11+B12)
            # M7 = (A12-A22)(B21+B22)
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # We need to construct U, V, W such that sum U[i,j,r] V[j,k,r] W[k,i,r] = C[i,k]
            # The "rank" is 7.
            # U has shape (4, 7), V (4, 7), W (4, 7).
            # Indices: i,j in 0..1. k in 0..1.
            # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
            
            U = np.zeros((4, 7))
            V = np.zeros((4, 7))
            W = np.zeros((4, 7))
            
            # M1: (A11+A22)(B11+B22)
            # A11 is i=0,j=0. A22 is i=1,j=1.
            # B11 is j=0,k=0. B22 is j=1,k=1.
            # U[0,0]=1, U[3,0]=1 (A11, A22)
            # V[0,0]=1, V[3,0]=1 (B11, B22)
            # W[0,0]=1 (C11 term), W[3,0]=1 (C22 term)
            U[0, 0] = 1; U[3, 0] = 1
            V[0, 0] = 1; V[3, 0] = 1
            W[0, 0] = 1; W[3, 0] = 1
            
            # M2: (A21+A22)B11
            # A21 is i=1,j=0. A22 is i=1,j=1.
            # B11 is j=0,k=0.
            U[2, 1] = 1; U[3, 1] = 1
            V[0, 1] = 1
            W[2, 1] = 1 # C21 term
            
            # M3: A11(B21-B22)
            # A11 is i=0,j=0.
            # B21 is j=1,k=0. B22 is j=1,k=1.
            U[0, 2] = 1
            V[2, 2] = 1 # B21
            V[3, 2] = -1 # B22
            W[1, 2] = 1 # C12 term
            
            # M4: A22(B12-B11)
            # A22 is i=1,j=1.
            # B12 is j=0,k=1. B11 is j=0,k=0.
            U[3, 3] = 1
            V[1, 3] = 1 # B12
            V[0, 3] = -1 # B11
            W[0, 3] = 1 # C11 term
            W[2, 3] = 1 # C21 term
            
            # M5: (A11+A12)B22
            # A11 is i=0,j=0. A12 is i=0,j=1.
            # B22 is j=1,k=1.
            U[0, 4] = 1; U[1, 4] = 1
            V[3, 4] = 1
            W[0, 4] = -1 # C11 term
            W[1, 4] = 1 # C12 term
            
            # M6: (A21-A11)(B11+B12)
            # A21 is i=1,j=0. A11 is i=0,j=0.
            # B11 is j=0,k=0. B12 is j=0,k=1.
            U[2, 5] = 1; U[0, 5] = -1
            V[0, 5] = 1; V[1, 5] = 1
            W[3, 5] = 1 # C22 term
            
            # M7: (A12-A22)(B21+B22)
            # A12 is i=0,j=1. A22 is i=1,j=1.
            # B21 is j=1,k=0. B22 is j=1,k=1.
            U[1, 6] = 1; U[3, 6] = -1
            V[2, 6] = 1; V[3, 6] = 1
            W[0, 6] = 1 # C11 term
            W[3, 6] = 1 # C22 term
            
            return U, V, W

        # Recursive case
        # Find the largest dimension to split
        if m >= n and m >= p:
            # Split m
            m1 = m // 2
            m2 = m - m1
            U1, V1, W1 = decompose(n, m1, p)
            U2, V2, W2 = decompose(n, m2, p)
            return combine_split_m(n, m1, m2, p, U1, V1, W1, U2, V2, W2)
        elif n >= m and n >= p:
            # Split n
            # This is trickier to combine because U and W indices change.
            # C = A B. A is n x m. Split A into A1 (n1 x m) and A2 (n2 x m).
            # C1 = A1 B (n1 x p). C2 = A2 B (n2 x p).
            # We need to combine U, V, W.
            # U1 shape (n1*m, r1). U2 shape (n2*m, r2).
            # Global U shape (n*m, r1+r2).
            # Rows of U1: i_local in [0, n1). Global i = i_local.
            # Rows of U2: i_local in [0, n2). Global i = i_local + n1.
            # V1, V2 are same shape (m*p, r).
            # W1 shape (p*n1, r1). W2 shape (p*n2, r2).
            # Global W shape (p*n, r1+r2).
            # Rows of W1: k*n1 + i_local. Global k*n + i_local.
            # Rows of W2: k*n2 + i_local. Global k*n + (i_local + n1).
            
            n1 = n // 2
            n2 = n - n1
            U1, V1, W1 = decompose(n1, m, p)
            U2, V2, W2 = decompose(n2, m, p)
            
            r1 = U1.shape[1]
            r2 = U2.shape[1]
            rt = r1 + r2
            
            U_total = np.zeros((n * m, rt))
            V_total = np.zeros((m * p, rt))
            W_total = np.zeros((p * n, rt))
            
            # Place U1
            for i in range(n1):
                U_total[i * m : (i + 1) * m, :r1] = U1[i * m : (i + 1) * m, :]
            # Place U2
            for i in range(n2):
                U_total[(i + n1) * m : (i + n1 + 1) * m, r1:] = U2[i * m : (i + 1) * m, :]
                
            # Place V
            V_total[:, :r1] = V1
            V_total[:, r1:] = V2
            
            # Place W1
            for k in range(p):
                W_total[k * n : k * n + n1, :r1] = W1[k * n1 : k * n1 + n1, :]
            # Place W2
            for k in range(p):
                W_total[k * n + n1 : k * n + n, r1:] = W2[k * n2 : k * n2 + n2, r1:] # Wait, W2 is (p*n2, r2)
                # W2 rows are k*n2 + i_local.
                # We need to map to global k*n + i_local + n1.
                for i_local in range(n2):
                    W_total[k * n + n1 + i_local, r1:] = W2[k * n2 + i_local, :]
            
            return U_total, V_total, W_total
        else:
            # Split p
            # C = A B. Split B into B1 (m x p1) and B2 (m x p2).
            # C1 = A B1 (n x p1). C2 = A B2 (n x p2).
            # U1 shape (n*m, r1). U2 shape (n*m, r2).
            # V1 shape (m*p1, r1). V2 shape (m*p2, r2).
            # W1 shape (p1*n, r1). W2 shape (p2*n, r2).
            # Global U shape (n*m, r1+r2).
            # Global V shape (m*p, r1+r2).
            # Global W shape (p*n, r1+r2).
            
            p1 = p // 2
            p2 = p - p1
            U1, V1, W1 = decompose(n, m, p1)
            U2, V2, W2 = decompose(n, m, p2)
            
            r1 = U1.shape[1]
            r2 = U2.shape[1]
            rt = r1 + r2
            
            U_total = np.zeros((n * m, rt))
            V_total = np.zeros((m * p, rt))
            W_total = np.zeros((p * n, rt))
            
            # Place U
            U_total[:, :r1] = U1
            U_total[:, r1:] = U2
            
            # Place V1
            for j in range(m):
                V_total[j * p : j * p + p1, :r1] = V1[j * p1 : (j + 1) * p1, :]
            # Place V2
            for j in range(m):
                V_total[j * p + p1 : j * p + p, r1:] = V2[j * p2 : (j + 1) * p2, :]
                
            # Place W1
            for k in range(p1):
                W_total[k * n : (k + 1) * n, :r1] = W1[k * n : (k + 1) * n, :]
            # Place W2
            for k in range(p2):
                W_total[(k + p1) * n : (k + p1 + 1) * n, r1:] = W2[k * n : (k + 1) * n, :]
                
            return U_total, V_total, W_total

    U, V, W = decompose(n, m, p)
    
    # Convert to lists of lists as required by interface?
    # The interface says "integer/half-integer entries". NumPy arrays are fine if they are converted or if the evaluator accepts arrays.
    # The seed program returns lists of lists. Let's stick to lists to be safe.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
