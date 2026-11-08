import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_4x4x4_terms():
    # Returns list of (M, P) where M is 4x4 linear form of A, P is 4x4 linear form of B.
    # M[i,j] corresponds to A_{i,j} coefficient.
    # The product M*P contributes to C = A*B via C = sum terms.
    # We represent linear forms as 4x4 integer matrices.
    terms = []
    
    # M1 = A11 + A22
    M1 = [[1,0,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,0]]
    P1 = [[0,1,0,0],[0,0,0,0],[1,0,0,0],[0,0,0,0]]
    terms.append((M1, P1))
    
    # M2 = A21 + A22
    M2 = [[0,0,0,0],[0,1,0,0],[0,1,0,0],[0,0,0,0]]
    P2 = [[0,0,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,0]]
    terms.append((M2, P2))
    
    # M3 = A11
    M3 = [[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    P3 = [[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    terms.append((M3, P3))
    
    # M4 = A22 - A11
    M4 = [[-1,0,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,0]]
    P4 = [[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    terms.append((M4, P4))
    
    # M5 = A11 + A12
    M5 = [[1,1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    P5 = [[0,0,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,0]]
    terms.append((M5, P5))
    
    # M6 = A21 - A11
    M6 = [[-1,0,0,0],[0,0,0,0],[1,-1,0,0],[0,0,0,0]]
    P6 = [[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    terms.append((M6, P6))
    
    # M7 = A12 + A22
    M7 = [[0,1,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,0]]
    P7 = [[0,0,0,0],[0,0,0,0],[1,0,0,0],[0,0,0,0]]
    terms.append((M7, P7))
    
    return terms

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that sum_c U[:,c] outer W[:,c] * V[:,c] equals matmul tensor.
    U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    Tensor T[i,j,k] = sum_ab A[i,a] B[b,k] * delta_{j,a} * delta_{b,j} ... no.
    Actually T[i,j,k] = A[i,j] * B[j,k].
    Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    import numpy as np
    rank = len(U[0])
    T = np.zeros((n, m, p))
    for c in range(rank):
        uc = np.array(U[:, c])
        vc = np.array(V[:, c])
        wc = np.array(W[:, c])
        uc = uc.reshape(n, m)
        vc = vc.reshape(m, p)
        wc = wc.reshape(p, n)
        # contribution to T[i,j,k] = uc[i,j] * vc[j,k] * wc[k,i]
        # This is a rank-1 tensor in (i,j,k) space.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += uc[i,j] * vc[j,k] * wc[k,i]
    
    # Target tensor: T[i,j,k] = 1 if we think of A[i,j]=1, B[j,k]=1? No.
    # The bilinear map is (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T is defined by T[i,j,k] = 1 for all i,j,k? No.
    # The multiplication tensor T_matmul has components T[i,j,k] = 1 if the term A[i,j]B[j,k] appears.
    # Actually, the standard tensor for matrix multiplication of n x m by m x p has:
    # T[i, j, k] = 1 for all i in [n], j in [m], k in [p].
    # Because C[i,k] = sum_j A[i,j] B[j,k]. Each term is A[i,j] B[j,k].
    # So the tensor is the all-ones tensor? No, that would be rank 1 if all ones.
    # Wait. The tensor is indexed by (i, j, k). The value is 1 for all (i,j,k).
    # Let's check: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # If this equals 1 for all i,j,k, then it's the all-ones tensor.
    # But the schoolbook decomposition has rank n*m*p.
    # Let's check schoolbook:
    # For each (i,j,k), there is a term c(i,j,k) where U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.
    # All other entries for this c are 0.
    # So sum_c ... = 1 for each (i,j,k).
    # Yes, the matrix multiplication tensor for n x m by m x p is the all-ones tensor of size n x m x p.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i,j,k] != 1.0:
                    return False
    return True

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    
    # We will build the decomposition using Strassen for the 4x4x4 block
    # and schoolbook for the rest.
    
    # Split A (4x5) into A_left (4x4) and A_right (4x1)
    # Split B (5x6) into B_left (4x6) and B_right (1x6)
    # C = A_left * B_left + A_right * B_right
    # A_left * B_left is 4x4 * 4x6
    # A_right * B_right is 4x1 * 1x6
    
    # Let's handle A_left * B_left first.
    # B_left is 4x6. Split into B_left1 (4x4) and B_left2 (4x2).
    # A_left * B_left = A_left * B_left1 + A_left * B_left2
    
    # A_left * B_left1: 4x4 * 4x4 -> Strassen (rank 7)
    # A_left * B_left2: 4x4 * 4x2 -> Schoolbook (rank 32) or something better?
    # Actually, 4x4 * 4x2 can be seen as 4x4 * [B2a, B2b] where B2a, B2b are 4x1.
    # So it's 2 * (4x4 * 4x1).
    # 4x4 * 4x1 is matrix-vector product. Rank is 16 (schoolbook).
    # So 4x4 * 4x2 is rank 32.
    # Total for A_left * B_left: 7 + 32 = 39.
    
    # A_right * B_right: 4x1 * 1x6.
    # C[i,k] = A[4,k] * B[5,k]? No.
    # A_right is row 4 of A (indices 0..3 in the 5th column, so index 4).
    # A_right[i] = A[i, 4].
    # B_right[k] = B[4, k].
    # C[i,k] += A[i,4] * B[4,k].
    # This is outer product of A_right (vector of size 4) and B_right (vector of size 6).
    # Rank of outer product of size 4 and 6 is min(4,6) = 4?
    # u v^T = sum_{i=1}^4 u_i e_i v^T.
    # So rank is 4.
    
    # Total rank: 39 + 4 = 43.
    
    # Let's implement this.
    
    # We'll accumulate the columns of U, V, W.
    U_rows = n * m  # 20
    V_rows = m * p  # 30
    W_rows = p * n  # 24
    
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        # u_vec is list of length U_rows
        # v_vec is list of length V_rows
        # w_vec is list of length W_rows
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
    
    # --- Part 1: A_left * B_left1 (4x4 * 4x4) using Strassen ---
    # This contributes to C[i,k] for i in 0..3, k in 0..3.
    # The terms are linear forms M_c(A_left) * P_c(B_left1).
    # M_c is a 4x4 matrix, P_c is a 4x4 matrix.
    # We need to express M_c(A_left) and P_c(B_left1) as linear forms in the original A and B.
    # A_left[i,j] = A[i,j] for i,j in 0..3.
    # B_left1[j,k] = B[j,k] for j in 0..3, k in 0..3.
    
    strassen_terms = _strassen_4x4x4_terms()
    for M, P in strassen_terms:
        u_vec = [0] * U_rows
        v_vec = [0] * V_rows
        w_vec = [0] * W_rows
        
        # The product is sum_{i,j} M[i,j] A[i,j] * sum_{j,k} P[j,k] B[j,k]
        # = sum_{i,j,k} M[i,j] P[j,k] A[i,j] B[j,k]
        # This contributes to C[i,k] with coefficient M[i,j] P[j,k] summed over j.
        # But the decomposition requires:
        # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i,j,k.
        # Wait, the tensor is T[i,j,k] = A[i,j] B[j,k].
        # So for each Strassen term, we have a product of two linear forms.
        # L_A = sum_{i,j} M[i,j] A[i,j]
        # L_B = sum_{j,k} P[j,k] B[j,k]
        # Product L_A * L_B contributes to C.
        # But in the tensor decomposition, we need to write this as a sum of rank-1 tensors.
        # L_A * L_B = (sum_{i,j} M[i,j] A[i,j]) * (sum_{j',k} P[j',k] B[j',k])
        # = sum_{i,j,j',k} M[i,j] P[j',k] A[i,j] B[j',k]
        # The term A[i,j] B[j',k] is part of the tensor if j == j'.
        # But L_A * L_B is a scalar. It doesn't directly give the tensor components.
        
        # I think I confused the algorithm with the tensor decomposition.
        # Strassen gives an algorithm to compute C from A and B using 7 multiplications of linear forms.
        # The tensor decomposition of the 4x4x4 matmul is rank 7.
        # The 7 terms are rank-1 tensors.
        # Let's find the explicit rank-1 decomposition for 4x4x4.
        
        # It's easier to just use schoolbook for all parts for now to ensure correctness,
        # then optimize if time permits. But 120 is the score.
        # Let's try to get a better rank with a simple approach.
        
        # Actually, let's just use the schoolbook decomposition but check if we can merge any terms.
        # No, schoolbook is minimal for the "diagonal" terms?
        
        # Let's try the split approach but with schoolbook for the sub-blocks.
        # Rank of 4x4x4 schoolbook is 16.
        # Rank of 4x4x2 schoolbook is 32.
        # Rank of 4x1x6 is 24? No, 4*1*6 = 24.
        # Total: 16 + 32 + 24 = 72.
        
        # This is better than 120.
        
        pass # We will restart the construction logic below.

    # Let's restart the construction with a clear plan.
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
    
    # Block 1: A[0:4, 0:4] * B[0:4, 0:4] (4x4x4)
    # Schoolbook: 16 terms.
    for i in range(4):
        for j in range(4):
            for k in range(4):
                u_vec = [0] * U_rows
                v_vec = [0] * V_rows
                w_vec = [0] * W_rows
                u_vec[i*m + j] = 1
                v_vec[j*p + k] = 1
                w_vec[k*n + i] = 1
                add_term(u_vec, v_vec, w_vec)
    
    # Block 2: A[0:4, 0:4] * B[0:4, 4:6] (4x4x2)
    # Schoolbook: 4*4*2 = 32 terms.
    for i in range(4):
        for j in range(4):
            for k in range(2):
                k_orig = 4 + k
                u_vec = [0] * U_rows
                v_vec = [0] * V_rows
                w_vec = [0] * W_rows
                u_vec[i*m + j] = 1
                v_vec[j*p + k_orig] = 1
                w_vec[k_orig*n + i] = 1
                add_term(u_vec, v_vec, w_vec)
    
    # Block 3: A[0:4, 4:5] * B[4:5, 0:6] (4x1x6)
    # Schoolbook: 4*1*6 = 24 terms.
    for i in range(4):
        for k in range(6):
            u_vec = [0] * U_rows
            v_vec = [0] * V_rows
            w_vec = [0] * W_rows
            u_vec[i*m + 4] = 1
            v_vec[4*p + k] = 1
            w_vec[k*n + i] = 1
            add_term(u_vec, v_vec, w_vec)
    
    rank = len(U)
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        U = [[0] * (n*m*p) for _ in range(n*m)]
        V = [[0] * (n*m*p) for _ in range(m*p)]
        W = [[0] * (n*m*p) for _ in range(p*n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][column] = 1
                    V[j*p+k][column] = 1
                    W[k*n+i][column] = 1
                    column += 1
        rank = n*m*p
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
