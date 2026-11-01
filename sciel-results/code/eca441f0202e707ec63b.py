import time
import random
import numpy as np
from numpy import linalg as LA

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # A is n x m (2x2), B is m x p (2x7), C is n x p (2x7)
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # Decomposition: sum_c U[i*m+k, c] * V[k*p+j, c] * W[j*n+i, c] = 1 for all i,k,j
    
    deadline = time.time() + 145.0
    
    def schoolbook():
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i*m + k][col] = 1
                    V[k*p + j][col] = 1
                    W[j*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    
    def verify(U_mat, V_mat, W_mat):
        """Check if sum_c U[i*m+k,c]*V[k*p+j,c]*W[j*n+i,c] == 1 for all i,k,j"""
        r = U_mat.shape[1]
        u_r = U_mat.T.reshape(n, m, r)
        v_r = V_mat.T.reshape(m, p, r)
        w_r = W_mat.T.reshape(p, n, r)
        T = np.einsum('ikc,kjc,jic->ijk', u_r, v_r, w_r)
        return np.allclose(T, 1.0, atol=1e-8)
    
    def is_half_integer(mat):
        return np.allclose(mat * 2, np.round(mat * 2), atol=1e-8)
    
    def mat_to_list(mat):
        """Convert matrix to list of lists of Python floats/ints"""
        return mat.tolist()
    
    def try_strassen_extension():
        """
        Strategy: Construct a rank-25 decomposition using Strassen's algorithm 
        for the first 6 columns and a simple expansion for the last column.
        
        Strassen for 2x2x2 (rank 7) decomposes A(2x2) * B(2x2) -> C(2x2).
        For B(2x7), we can view B as [B_1 (2x2), B_2 (2x2), B_3 (2x2), B_4 (2x1)].
        
        If we use independent Strassen decompositions for B_1, B_2, B_3, 
        the rank would be 7*3 + 2 = 23. 
        However, Strassen terms for different columns of B are independent 
        in the V factor (since V depends on k and j). 
        The U factor (depending on i, k) is the same for all j.
        The W factor (depending on j, i) is independent for different j.
        
        So, we can simply concatenate the rank-7 Strassen decomposition 
        for each 2x2 block of B, and handle the last 2x1 block separately.
        
        Rank 7 Strassen terms for A(2x2) * B(2x2):
        Let A = [[A00, A01], [A10, A11]] and B = [[B00, B01], [B10, B11]].
        C00 = A00*B00 + A01*B10
        C01 = A00*B01 + A01*B11
        C10 = A10*B00 + A11*B10
        C11 = A10*B01 + A11*B11
        
        Strassen computes 7 products:
        M0 = (A00+A11)(B00+B11)
        M1 = (A10+A11)B00
        M2 = A00(B10-B11)
        M3 = A11(B01-B00)
        M4 = (A00+A10)B11
        M5 = (A10-A00)(B00+B01)
        M6 = (A01-A11)(B10+B11)
        
        Combinations:
        C00 = M0 - M3 - M5 + M6
        C01 = M4 + M5
        C10 = M1 + M2
        C11 = M0 - M2 - M4 + M3
        
        We need to map these to U, V, W factors.
        U[i*m+k, c] corresponds to coefficient of A[i,k] in term c.
        V[k*p+j, c] corresponds to coefficient of B[k,j] in term c.
        W[j*n+i, c] corresponds to coefficient of C[i,j] in term c.
        
        Let's define the 7 terms for a generic 2x2 block of B (columns j0, j1):
        
        Term 0 (M0):
          U: A00(1), A11(1) -> U[0,0]=1, U[1,1]=1 (indices: i*m+k)
          V: B00(1), B11(1) -> V[0, j0]=1, V[1, j1]=1 (indices: k*p+j)
          W: C00(1), C11(1) -> W[j0, 0]=1, W[j1, 1]=1 (indices: j*n+i)
          
        Term 1 (M1):
          U: A10(1), A11(1) -> U[1,0]=1, U[1,1]=1
          V: B00(1) -> V[0, j0]=1
          W: C10(1) -> W[j0, 1]=1
          
        Term 2 (M2):
          U: A00(1) -> U[0,0]=1
          V: B10(1), B11(-1) -> V[1, j0]=1, V[1, j1]=-1
          W: C10(1) -> W[j0, 1]=1
          
        Term 3 (M3):
          U: A11(1) -> U[1,1]=1
          V: B01(1), B00(-1) -> V[0, j1]=1, V[0, j0]=-1
          W: C11(1) -> W[j1, 1]=1
          
        Term 4 (M4):
          U: A00(1), A10(1) -> U[0,0]=1, U[1,0]=1
          V: B11(1) -> V[1, j1]=1
          W: C01(1) -> W[j1, 0]=1
          
        Term 5 (M5):
          U: A10(1), A00(-1) -> U[1,0]=1, U[0,0]=-1
          V: B00(1), B01(1) -> V[0, j0]=1, V[0, j1]=1
          W: C10(1), C11(1) -> W[j0, 1]=1, W[j1, 1]=1
          
        Term 6 (M6):
          U: A01(1), A11(-1) -> U[0,1]=1, U[1,1]=-1
          V: B10(1), B11(1) -> V[1, j0]=1, V[1, j1]=1
          W: C00(1), C01(1) -> W[j0, 0]=1, W[j1, 0]=1
          
        Wait, I need to check the W combinations again.
        C00 = M0 - M3 - M5 + M6
        C01 = M4 + M5
        C10 = M1 + M2
        C11 = M0 - M2 - M4 + M3
        
        So:
        W for C00 (index j0*n+0):
          T0: +1
          T3: -1
          T5: -1
          T6: +1
          Others: 0
          
        W for C01 (index j1*n+0):
          T4: +1
          T5: +1
          T6: +1  <-- Wait, M6 is (A01-A11)(B10+B11). 
          
          Let's re-verify Strassen formulas.
          Standard Strassen:
          M1 = (A11 + A22)(B11 + B22)
          M2 = (A21 + A22)B11
          M3 = A11(B21 - B22)
          M4 = A22(B12 - B11)
          M5 = (A11 + A12)B22
          M6 = (A21 - A11)(B11 + B12)
          M7 = (A12 - A22)(B21 + B22)
          
          C11 = M1 - M4 - M5 + M7
          C12 = M5 + M6
          C21 = M2 + M3
          C22 = M1 - M3 + M4 - M7
          
          Mapping to my indices (1-based in standard, 0-based in code):
          A00=A11, A01=A12, A10=A21, A11=A22
          B00=B11, B01=B12, B10=B21, B11=B22
          C00=C11, C01=C12, C10=C21, C11=C22
          
          M0 (std M1): (A00+A11)(B00+B11)
             U: A00(1), A11(1)
             V: B00(1), B11(1)
             W: C00(1), C11(1)
             
          M1 (std M2): (A10+A11)B00
             U: A10(1), A11(1)
             V: B00(1)
             W: C10(1)
             
          M2 (std M3): A00(B10-B11)
             U: A00(1)
             V: B10(1), B11(-1)
             W: C10(1)
             
          M3 (std M4): A11(B01-B00)
             U: A11(1)
             V: B01(1), B00(-1)
             W: C11(1)
             
          M4 (std M5): (A00+A01)B11
             U: A00(1), A01(1)
             V: B11(1)
             W: C01(1)
             
          M5 (std M6): (A10-A00)(B00+B01)
             U: A10(1), A00(-1)
             V: B00(1), B01(1)
             W: C10(1), C11(1)
             
          M6 (std M7): (A01-A11)(B10+B11)
             U: A01(1), A11(-1)
             V: B10(1), B11(1)
             W: C00(1), C01(1)
          
          Let's check C00 = M0 - M3 - M5 + M6
          C00 = (A00+A11)(B00+B11) - A11(B01-B00) - (A10-A00)(B00+B01) + (A01-A11)(B10+B11)
          
          Expand:
          A00B00 + A00B11 + A11B00 + A11B11
          - A11B01 + A11B00
          - A10B00 - A10B01 + A00B00 + A00B01
          + A01B10 + A01B11 - A11B10 - A11B11
          
          Target C00 = A00B00 + A01B10.
          
          Coeffs:
          A00B00: 1 + 1 = 2? No, wait.
          
          Let's re-evaluate standard Strassen C11 formula.
          C11 = M1 - M4 - M5 + M7
          M1 = (A11+A22)(B11+B22)
          M4 = A22(B12-B11)
          M5 = (A11+A12)B22
          M7 = (A12-A22)(B21+B22)
          
          C11 = (A11+A22)(B11+B22) - A22(B12-B11) - (A11+A12)B22 + (A12-A22)(B21+B22)
          
          Expand:
          A11B11 + A11B22 + A22B11 + A22B22
          - A22B12 + A22B11
          - A11B22 - A12B22
          + A12B21 + A12B22 - A22B21 - A22B22
          
          Sum:
          A11B11: 1
          A11B22: 1 - 1 = 0
          A22B11: 1 + 1 = 2? No.
          
          There is a sign error in my manual expansion or the standard formula recall.
          
          Standard Strassen:
          C11 = M1 - M4 - M5 + M7
          M1 = (A11 + A22)(B11 + B22)
          M2 = (A21 + A22)B11
          M3 = A11(B21 - B22)
          M4 = A22(B12 - B11)
          M5 = (A11 + A12)B22
          M6 = (A21 - A11)(B11 + B12)
          M7 = (A12 - A22)(B21 + B22)
          
          C11 = M1 - M4 - M5 + M7
          = (A11 + A22)(B11 + B22) - A22(B12 - B11) - (A11 + A12)B22 + (A12 - A22)(B21 + B22)
          
          Let's check A22B11 term:
          From M1: A22B11
          From -M4: -A22(-B11) = +A22B11
          Total 2 A22B11. This is wrong.
          
          Ah, M4 = A22(B12 - B11). So -M4 = -A22B12 + A22B11.
          Yes, so A22B11 appears twice.
          
          Wait, C11 should be A11B11 + A12B21.
          
          Let's check A11B11:
          From M1: A11B11
          Total 1. Correct.
          
          Let's check A12B21:
          From M7: A12B21
          Total 1. Correct.
          
          Let's check cross terms (should be 0):
          A11B22: M1(1) - M5(1) = 0.
          A22B22: M1(1) - M7(1) = 0.
          A22B12: -M4(-1) = 1? No, M4 = A22B12 - A22B11. -M4 = -A22B12 + A22B11.
          So A22B12 coeff is -1.
          A12B22: -M5(-1) = 1? M5 = A11B22 + A12B22. -M5 = -A11B22 - A12B22.
          M7 = A12B21 + A12B22 - A22B21 - A22B22.
          A12B22 coeff: -1 (from -M5) + 1 (from M7) = 0.
          
          So where does A22B12 come from? It should be 0.
          My expansion of -M4 gave -A22B12.
          Is there another term?
          
          Let's re-read M4. M4 = A22(B12 - B11).
          So -M4 contributes -A22B12.
          This means C11 has a term -A22B12.
          This implies the standard formula I recalled is for a different combination or I am misremembering.
          
          Actually, Strassen is:
          C11 = M1 - M4 - M5 + M7
          C12 = M5 + M6
          C21 = M2 + M3
          C22 = M1 - M3 + M4 - M7
          
          Let's check C22 = M1 - M3 + M4 - M7
          M1 = (A11+A22)(B11+B22)
          M3 = A11(B21-B22)
          M4 = A22(B12-B11)
          M7 = (A12-A22)(B21+B22)
          
          Target C22 = A21B12 + A22B22.
          
          A22B22:
          M1: 1
          -M3: 0
          M4: 0
          -M7: 0 (M7 has -A22B22, so -M7 has +A22B22)
          Total 2?
          
          There is a known issue with my manual verification. I will rely on the fact that Strassen works.
          I will construct the matrices based on the standard definitions and verify numerically in the code.
          
          If verification fails, I will fall back to schoolbook.
        """
        
        # Define Strassen terms for a block of B with columns j0, j1
        # Returns U_block (2x2 x 7), V_block (2x2 x 7), W_block (2x2 x 7)
        # U is indexed by i, k. V by k, j. W by j, i.
        
        def get_strassen_block(j0, j1):
            U = np.zeros((4, 7))
            V = np.zeros((14, 7)) # m*p = 2*7 = 14
            W = np.zeros((14, 7)) # p*n = 7*2 = 14
            
            # Term 0: M1 = (A00+A11)(B00+B11)
            U[0, 0] = 1; U[3, 0] = 1 # A00, A11
            V[0, 0] = 1; V[1, 0] = 1 # B00, B11 (indices k*p+j)
            # V[0, 0] is k=0, j=j0. V[1, 0] is k=1, j=j1? No.
            # V index is k*p + j.
            # B00 is k=0, j=j0. Index 0*7 + j0 = j0.
            # B11 is k=1, j=j1. Index 1*7 + j1 = 7 + j1.
            V[j0, 0] = 1
            V[7 + j1, 0] = 1
            
            W[j0, 0] = 1; W[2 + j1, 0] = 1 # C00, C11
            # W index is j*n + i.
            # C00 is j=j0, i=0. Index j0*2 + 0 = 2*j0.
            # C11 is j=j1, i=1. Index j1*2 + 1 = 2*j1 + 1.
            W[2*j0, 0] = 1
            W[2*j1 + 1, 0] = 1
            
            # Term 1: M2 = (A10+A11)B00
            U[2, 1] = 1; U[3, 1] = 1 # A10, A11
            V[j0, 1] = 1 # B00
            W[2*j0 + 1, 1] = 1 # C10 (j=j0, i=1)
            
            # Term 2: M3 = A00(B10-B11)
            U[0, 2] = 1 # A00
            V[7 + j0, 2] = 1 # B10 (k=1, j=j0)
            V[7 + j1, 2] = -1 # B11 (k=1, j=j1)
            W[2*j0 + 1, 2] = 1 # C10
            
            # Term 3: M4 = A11(B01-B00)
            U[3, 3] = 1 # A11
            V[j1, 3] = 1 # B01 (k=0, j=j1)
            V[j0, 3] = -1 # B00 (k=0, j=j0)
            W[2*j1 + 1, 3] = 1 # C11
            
            # Term 4: M5 = (A00+A01)B11
            U[0, 4] = 1; U[1, 4] = 1 # A00, A01
            V[7 + j1, 4] = 1 # B11
            W[2*j1, 4] = 1 # C01 (j=j1, i=0)
            
            # Term 5: M6 = (A10-A00)(B00+B01)
            U[2, 5] = 1; U[0, 5] = -1 # A10, A00
            V[j0, 5] = 1 # B00
            V[j1, 5] = 1 # B01
            W[2*j0 + 1, 5] = 1 # C10
            W[2*j1 + 1, 5] = 1 # C11
            
            # Term 6: M7 = (A01-A11)(B10+B11)
            U[1, 6] = 1; U[3, 6] = -1 # A01, A11
            V[7 + j0, 6] = 1 # B10
            V[7 + j1, 6] = 1 # B11
            W[2*j0, 6] = 1 # C00
            W[2*j1, 6] = 1 # C01
            
            return U, V, W

        # Construct full U, V, W for rank 25
        # 3 blocks of Strassen (7 terms each) = 21 terms
        # 1 block for last column (2x1). Rank 2 decomposition? 
        # A(2x2) * B(2x1) -> C(2x1).
        # C0 = A00B0 + A01B1
        # C1 = A10B0 + A11B1
        # Rank 4 is trivial. Can we do better?
        # It's just 2 dot products of length 2.
        # We can use 4 terms:
        # T21: A00B0 -> C0
        # T22: A01B1 -> C0
        # T23: A10B0 -> C1
        # T24: A11B1 -> C1
        # This adds 4 terms. Total 21 + 4 = 25.
        
        U_mat = np.zeros((n*m, 25))
        V_mat = np.zeros((m*p, 25))
        W_mat = np.zeros((p*n, 25))
        
        # Block 1: cols 0, 1
        U1, V1, W1 = get_strassen_block(0, 1)
        U_mat[:, 0:7] = U1
        V_mat[:, 0:7] = V1
        W_mat[:, 0:7] = W1
        
        # Block 2: cols 2, 3
        U2, V2, W2 = get_strassen_block(2, 3)
        U_mat[:, 7:14] = U2
        V_mat[:, 7:14] = V2
        W_mat[:, 7:14] = W2
        
        # Block 3: cols 4, 5
        U3, V3, W3 = get_strassen_block(4, 5)
        U_mat[:, 14:21] = U3
        V_mat[:, 14:21] = V3
        W_mat[:, 14:21] = W3
        
        # Block 4: col 6
        # T21: A00 * B[0,6] -> C[0,6]
        U_mat[0, 21] = 1
        V_mat[6, 21] = 1 # k=0, j=6
        W_mat[12, 21] = 1 # j=6, i=0 -> 6*2+0=12
        
        # T22: A01 * B[1,6] -> C[0,6]
        U_mat[1, 22] = 1
        V_mat[13, 22] = 1 # k=1, j=6
        W_mat[12, 22] = 1
        
        # T23: A10 * B[0,6] -> C[1,6]
        U_mat[2, 23] = 1
        V_mat[6, 23] = 1
        W_mat[13, 23] = 1 # j=6, i=1 -> 6*2+1=13
        
        # T24: A11 * B[1,6] -> C[1,6]
        U_mat[3, 24] = 1
        V_mat[13, 24] = 1
        W_mat[13, 24] = 1
        
        if verify(U_mat, V_mat, W_mat):
            if is_half_integer(U_mat) and is_half_integer(V_mat) and is_half_integer(W_mat):
                return {
                    "U": mat_to_list(U_mat),
                    "V": mat_to_list(V_mat),
                    "W": mat_to_list(W_mat)
                }
        return None

    sol = try_strassen_extension()
    if sol is not None:
        return sol

    # Fallback to schoolbook
    return best
# EVOLVE-BLOCK-END
