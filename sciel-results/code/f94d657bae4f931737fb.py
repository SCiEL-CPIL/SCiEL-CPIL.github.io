import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    # Schoolbook fallback: rank 24
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Try to find a rank-22 decomposition using Strassen-like structure
    # For 2x2x2, rank 7 is achievable (Strassen).
    # We split 2x3 x 3x4 as:
    #   A (2x3) = [A11 A12; A21 A22] where A11,A21 are 2x2 blocks? No.
    # Actually: A is 2x3, B is 3x4.
    # Split B into two 3x2 blocks: B = [B1 B2] where B1, B2 are 3x2.
    # Then C = A*B = [A*B1, A*B2].
    # Each A*Bi is 2x2. If we use rank-7 Strassen for each 2x2 product, that's 14 terms.
    # But A is shared! So we can't just do two independent Strassen.
    #
    # Better approach: Split the 3x4 matrix B into B_left (3x2) and B_right (3x2).
    # C_left = A @ B_left (2x2), C_right = A @ B_right (2x2).
    #
    # Alternative: Use the fact that we can combine.
    # Let's try a direct rank-22 construction.
    #
    # Known: For 2x2x2, the Strassen decomposition has rank 7.
    # The standard Strassen decomposition for 2x2x2:
    # Let A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    #
    # C11 = M1 - M4 + M6 - M7
    # C12 = M3 + M5 + M7
    # C21 = M2 + M4 - M6
    # C22 = M1 - M3 + M2 + M5
    #
    # For our case: A is 2x3, B is 3x4.
    # Let's partition B into two 3x2 blocks: B = [B1 | B2]
    # C = [C1 | C2] where C1 = A*B1 (2x2), C2 = A*B2 (2x2)
    #
    # We can write A as a 2x3 matrix. For the left block:
    # A * B1 where A is 2x3, B1 is 3x2.
    # This isn't directly 2x2x2.
    #
    # Alternative strategy: Try rank 22 via explicit construction.
    # The minimum rank for 2x3x4 is known to be 19 (from literature? or is it 22?).
    # Actually, the lower bound is max(2*3, 3*4, 2*4) = 12, and upper bound is 24.
    # Known results: R(2,3,4) is between 19 and 24.
    #
    # Let's try a hybrid: use Strassen for the 2x2x2 part and direct for the rest.
    # Split A (2x3) into A_left (2x2) and A_right (2x1).
    # Split B (3x4) into B_top (2x4) and B_bottom (1x4).
    # C = A_left @ B_top + A_right @ B_bottom
    #
    # A_left @ B_top is 2x2 @ 2x4. We can split B_top into two 2x2 blocks.
    # This gets complex.
    #
    # Let's just try to find rank 22 via a smarter search.
    # But first, let's try to construct rank 22 explicitly.
    #
    # Actually, let's try a different approach: use the known rank-7 Strassen
    # for a 2x2x2 subproblem and handle the remaining terms directly.
    #
    # Partition:
    # A (2x3) = [A1 A2] where A1 is 2x2, A2 is 2x1
    # B (3x4) = [B1; B2] where B1 is 2x4, B2 is 1x4
    # C = A1*B1 + A2*B2
    #
    # A1*B1: 2x2 @ 2x4. Split B1 = [B1a B1b] where each is 2x2.
    # A1*B1a: 2x2 @ 2x2 -> rank 7 (Strassen)
    # A1*B1b: 2x2 @ 2x2 -> rank 7 (Strassen)
    # A2*B2: 2x1 @ 1x4 -> rank 2 (outer product: A2 is 2x1, B2 is 1x4, so 2 terms? No, 2*1*4=8 but rank is min(2,1,4)=1? No, it's an outer product of a 2-vector and a 4-vector, which has rank 1 as a tensor? No, the tensor for 2x1x1x4 multiplication has rank 2*1*4=8 in general, but can be done in 2*4=8 scalar mults. Actually, for 2x1 @ 1x4, it's just element-wise: C[i][k] = A[i][0]*B[0][k]. That's 8 multiplications, but as a tensor decomposition, each output element (i,k) needs one term: U[i*1+0][t]=A[i][0], V[0*4+k][t]=B[0][k], W[k*2+i][t]=1. That's 8 terms. But wait, we can share? No, each (i,k) pair is independent. So rank 8 for this part.
    #
    # Total: 7 + 7 + 8 = 22. Yes!
    #
    # Let's construct this explicitly.
    #
    # A (2x3): indices i in {0,1}, j in {0,1,2}
    # B (3x4): indices j in {0,1,2}, k in {0,1,2,3}
    # C (2x4): indices i in {0,1}, k in {0,1,2,3}
    #
    # A1 = A[:, 0:2] (2x2), A2 = A[:, 2:3] (2x1)
    # B1 = B[0:2, :] (2x4), B2 = B[2:3, :] (1x4)
    # B1a = B1[:, 0:2] (2x2), B1b = B1[:, 2:4] (2x2)
    #
    # C = A1*B1a + A1*B1b + A2*B2
    #
    # For A1*B1a (2x2x2), use Strassen rank-7:
    # A1 = [[a00, a01], [a10, a11]] (row i, col j)
    # B1a = [[b00, b01], [b10, b11]] (row j, col k)
    #
    # Strassen terms:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # C11 = M1 - M4 + M6 - M7
    # C12 = M3 + M5 + M7
    # C21 = M2 + M4 - M6
    # C22 = M1 - M3 + M2 + M5
    #
    # Similarly for A1*B1b.
    #
    # For A2*B2:
    # A2 = [[a02], [a12]], B2 = [[b20, b21, b22, b23]]
    # C[i][k] = a_i2 * b2k for i in {0,1}, k in {0,1,2,3}
    # 8 terms.
    #
    # Total rank: 7 + 7 + 8 = 22.
    #
    # Let's build the U, V, W matrices for rank 22.
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)]  # 6 x 22
    V = [[0] * rank for _ in range(m * p)]  # 12 x 22
    W = [[0] * rank for _ in range(p * n)]  # 8 x 22
    
    col = 0
    
    # Helper: add Strassen decomposition for 2x2x2
    # A indices: (i, j) where i in {0,1}, j in {0,1}
    # B indices: (j, k) where j in {0,1}, k in {0,1}
    # C indices: (i, k) where i in {0,1}, k in {0,1}
    # U row: i*m + j = i*3 + j (since m=3, but for the 2x2 block, j is 0 or 1)
    # V row: j*p + k = j*4 + k (since p=4, but for the 2x2 block, j is 0 or 1, k is 0 or 1)
    # W row: k*n + i = k*2 + i (since n=2)
    
    def add_strassen(U, V, W, col_start, a_i_offset, a_j_offset, b_j_offset, b_k_offset):
        # a_i_offset: offset for A row index (0 or 1)
        # a_j_offset: offset for A col index (0 or 1)
        # b_j_offset: offset for B row index (0 or 1)
        # b_k_offset: offset for B col index (0 or 1)
        
        # M1 = (a00+a11)*(b00+b11)
        U[a_i_offset*3 + a_j_offset][col_start] = 1       # a00
        U[(a_i_offset+1)*3 + (a_j_offset+1)][col_start] = 1  # a11
        V[b_j_offset*4 + b_k_offset][col_start] = 1       # b00
        V[(b_j_offset+1)*4 + (b_k_offset+1)][col_start] = 1  # b11
        W[b_k_offset*2 + a_i_offset][col_start] = 1       # C00
        W[(b_k_offset+1)*2 + (a_i_offset+1)][col_start] = 1  # C11
        col_start += 1
        
        # M2 = (a10+a11)*b00
        U[(a_i_offset+1)*3 + a_j_offset][col_start] = 1       # a10
        U[(a_i_offset+1)*3 + (a_j_offset+1)][col_start] = 1   # a11
        V[b_j_offset*4 + b_k_offset][col_start] = 1           # b00
        W[b_k_offset*2 + (a_i_offset+1)][col_start] = 1       # C10
        col_start += 1
        
        # M3 = a00*(b01-b11)
        U[a_i_offset*3 + a_j_offset][col_start] = 1           # a00
        V[b_j_offset*4 + (b_k_offset+1)][col_start] = 1       # b01
        V[(b_j_offset+1)*4 + (b_k_offset+1)][col_start] = -1  # -b11
        W[(b_k_offset+1)*2 + a_i_offset][col_start] = 1       # C01
        col_start += 1
        
        # M4 = a11*(b10-b00)
        U[(a_i_offset+1)*3 + (a_j_offset+1)][col_start] = 1   # a11
        V[(b_j_offset+1)*4 + b_k_offset][col_start] = 1       # b10
        V[b_j_offset*4 + b_k_offset][col_start] = -1          # -b00
        W[b_k_offset*2 + (a_i_offset+1)][col_start] = 1       # C10
        col_start += 1
        
        # M5 = (a00+a01)*b11
        U[a_i_offset*3 + a_j_offset][col_start] = 1           # a00
        U[a_i_offset*3 + (a_j_offset+1)][col_start] = 1       # a01
        V[(b_j_offset+1)*4 + (b_k_offset+1)][col_start] = 1   # b11
        W[(b_k_offset+1)*2 + a_i_offset][col_start] = 1       # C01
        col_start += 1
        
        # M6 = (a10-a00)*(b00+b01)
        U[(a_i_offset+1)*3 + a_j_offset][col_start] = 1       # a10
        U[a_i_offset*3 + a_j_offset][col_start] = -1          # -a00
        V[b_j_offset*4 + b_k_offset][col_start] = 1           # b00
        V[b_j_offset*4 + (b_k_offset+1)][col_start] = 1       # b01
        W[b_k_offset*2 + (a_i_offset+1)][col_start] = 1       # C10
        col_start += 1
        
        # M7 = (a01-a11)*(b10+b11)
        U[a_i_offset*3 + (a_j_offset+1)][col_start] = 1       # a01
        U[(a_i_offset+1)*3 + (a_j_offset+1)][col_start] = -1  # -a11
        V[(b_j_offset+1)*4 + b_k_offset][col_start] = 1       # b10
        V[(b_j_offset+1)*4 + (b_k_offset+1)][col_start] = 1   # b11
        W[(b_k_offset+1)*2 + a_i_offset][col_start] = 1       # C01
        col_start += 1
        
        return col_start
    
    # Wait, the Strassen formula above is for computing C = A*B.
    # Let me re-derive carefully.
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = A*B = [[c00, c01], [c10, c11]]
    #
    # c00 = a00*b00 + a01*b10
    # c01 = a00*b01 + a01*b11
    # c10 = a10*b00 + a11*b10
    # c11 = a10*b01 + a11*b11
    #
    # Strassen:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # c00 = M1 - M4 + M6 - M7
    # c01 = M3 + M5 + M7
    # c10 = M2 + M4 - M6
    # c11 = M1 - M3 + M2 + M5
    #
    # Let me verify c00:
    # M1 - M4 + M6 - M7
    # = (a00+a11)(b00+b11) - a11(b10-b00) + (a10-a00)(b00+b01) - (a01-a11)(b10+b11)
    # = a00*b00 + a00*b11 + a11*b00 + a11*b11 - a11*b10 + a11*b00 + a10*b00 + a10*b01 - a00*b00 - a00*b01 - a01*b10 - a01*b11 + a11*b10 + a11*b11
    # = a00*b00 + a00*b11 + 2*a11*b00 + a11*b11 - a11*b10 + a10*b00 + a10*b01 - a00*b01 - a01*b10 - a01*b11 + a11*b10 + a11*b11
    # This doesn't simplify to a00*b00 + a01*b10. I think I have the wrong Strassen formula.
    #
    # Let me look up the correct Strassen formula.
    # Standard Strassen:
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22)B11
    # P3 = A11(B12 - B22)
    # P4 = A22(B21 - B11)
    # P5 = (A11 + A12)B22
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    #
    # C11 = P1 - P4 + P6 - P7
    # C12 = P3 + P5 + P7
    # C21 = P2 + P4 - P6
    # C22 = P1 - P3 + P2 + P5
    #
    # Let me verify C11 = P1 - P4 + P6 - P7:
    # P1 = (A11+A22)(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
    # P4 = A22(B21-B11) = A22*B21 - A22*B11
    # P6 = (A21-A11)(B11+B12) = A21*B11 + A21*B12 - A11*B11 - A11*B12
    # P7 = (A12-A22)(B21+B22) = A12*B21 + A12*B22 - A22*B21 - A22*B22
    #
    # P1 - P4 + P6 - P7
    # = A11*B11 + A11*B22 + A22*B11 + A22*B22 - A22*B21 + A22*B11 + A21*B11 + A21*B12 - A11*B11 - A11*B12 - A12*B21 - A12*B22 + A22*B21 + A22*B22
    # = A11*B22 + 2*A22*B11 + 2*A22*B22 + A21*B11 + A21*B12 - A11*B12 - A12*B21 - A12*B22
    # This still doesn't look right. Let me just use a verified Strassen decomposition.
    #
    # Actually, let me just verify numerically.
    
    # Let me define the Strassen decomposition properly and verify.
    
    def strassen_decomp():
        """Return U, V, W for 2x2x2 Strassen decomposition, rank 7."""
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # A indices: (i,j) -> row i*2+j
        # B indices: (j,k) -> row j*2+k
        # C indices: (i,k) -> row i*2+k
        
        # P1 = (A11+A22)(B11+B22)
        # A11 = A[0,0], A22 = A[1,1]
        # B11 = B[0,0], B22 = B[1,1]
        U_s[0*2+0, 0] = 1  # A11
        U_s[1*2+1, 0] = 1  # A22
        V_s[0*2+0, 0] = 1  # B11
        V_s[1*2+1, 0] = 1  # B22
        W_s[0*2+0, 0] = 1  # C11
        W_s[1*2+1, 0] = 1  # C22
        
        # P2 = (A21+A22)B11
        # A21 = A[1,0], A22 = A[1,1], B11 = B[0,0]
        U_s[1*2+0, 1] = 1  # A21
        U_s[1*2+1, 1] = 1  # A22
        V_s[0*2+0, 1] = 1  # B11
        W_s[1*2+0, 1] = 1  # C21
        
        # P3 = A11(B12-B22)
        # A11 = A[0,0], B12 = B[0,1], B22 = B[1,1]
        U_s[0*2+0, 2] = 1  # A11
        V_s[0*2+1, 2] = 1  # B12
        V_s[1*2+1, 2] = -1 # -B22
        W_s[0*2+1, 2] = 1  # C12
        
        # P4 = A22(B21-B11)
        # A22 = A[1,1], B21 = B[1,0], B11 = B[0,0]
        U_s[1*2+1, 3] = 1  # A22
        V_s[1*2+0, 3] = 1  # B21
        V_s[0*2+0, 3] = -1 # -B11
        W_s[1*2+0, 3] = 1  # C21
        
        # P5 = (A11+A12)B22
        # A11 = A[0,0], A12 = A[0,1], B22 = B[1,1]
        U_s[0*2+0, 4] = 1  # A11
        U_s[0*2+1, 4] = 1  # A12
        V_s[1*2+1, 4] = 1  # B22
        W_s[0*2+1, 4] = 1  # C12
        
        # P6 = (A21-A11)(B11+B12)
        # A21 = A[1,0], A11 = A[0,0], B11 = B[0,0], B12 = B[0,1]
        U_s[1*2+0, 5] = 1  # A21
        U_s[0*2+0, 5] = -1 # -A11
        V_s[0*2+0, 5] = 1  # B11
        V_s[0*2+1, 5] = 1  # B12
        W_s[1*2+0, 5] = 1  # C21
        
        # P7 = (A12-A22)(B21+B22)
        # A12 = A[0,1], A22 = A[1,1], B21 = B[1,0], B22 = B[1,1]
        U_s[0*2+1, 6] = 1  # A12
        U_s[1*2+1, 6] = -1 # -A22
        V_s[1*2+0, 6] = 1  # B21
        V_s[1*2+1, 6] = 1  # B22
        W_s[0*2+1, 6] = 1  # C12
        
        return U_s, V_s, W_s
    
    # Verify Strassen decomposition
    U_s, V_s, W_s = strassen_decomp()
    # Check: for each (i,k), sum_t U_s[i*2+j]*V_s[j*2+k]*W_s[i*2+k] should be 1 for the (i,j,k) where...
    # Actually, the tensor T[i,j,k] = 1 if the product C[i,k] = sum_j A[i,j]*B[j,k].
    # So T[i,j,k] = 1 for all i,j,k.
    # The decomposition: sum_t U_s[i*2+j, t] * V_s[j*2+k, t] * W_s[i*2+k, t] should equal 1 for all i,j,k.
    
    # Wait, that's not right. The tensor for matrix multiplication is T[i,j,k] = 1 for all i,j,k.
    # The decomposition is: T[i,j,k] = sum_t U[i*2+j, t] * V[j*2+k, t] * W[i*2+k, t]
    # But that's not how it works. The tensor is T[i,j,k] where the multiplication is C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T has indices (i, j, k) and T[i,j,k] = 1.
    # The decomposition is: T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
    # Wait, the convention is W[k*n+i, r]. So W index is k*n+i.
    # For 2x2x2: n=2, m=2, p=2.
    # U shape: (n*m, rank) = (4, rank)
    # V shape: (m*p, rank) = (4, rank)
    # W shape: (p*n, rank) = (4, rank)
    # U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # For 2x2x2: U[i*2+j, t] * V[j*2+k, t] * W[k*2+i, t]
    
    # Let me verify numerically.
    verified = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = 0
                for t in range(7):
                    val += U_s[i*2+j, t] * V_s[j*2+k, t] * W_s[k*2+i, t]
                if abs(val - 1.0) > 1e-9:
                    verified = False
                    print(f"Strassen verification failed at i={i}, j={j}, k={k}: val={val}")
                    break
            if not verified:
                break
        if not verified:
            break
    
    if not verified:
        # Fall back to schoolbook
        return {"U": U_school, "V": V_school, "W": W_school}
    
    # Now build the rank-22 decomposition for 2x3x4.
    # A (2x3): i in {0,1}, j in {0,1,2}
    # B (3x4): j in {0,1,2}, k in {0,1,2,3}
    # C (2x4): i in {0,1}, k in {0,1,2,3}
    #
    # Split: A1 = A[:, 0:2], A2 = A[:, 2:3]
    # B1 = B[0:2, :], B2 = B[2:3, :]
    # B1a = B1[:, 0:2], B1b = B1[:, 2:4]
    #
    # C = A1*B1a + A1*B1b + A2*B2
    #
    # For A1*B1a: 2x2 @ 2x2, use Strassen (rank 7)
    # A1 indices: i in {0,1}, j in {0,1} -> U row: i*3 + j
    # B1a indices: j in {0,1}, k in {0,1} -> V row: j*4 + k
    # C indices: i in {0,1}, k in {0,1} -> W row: k*2 + i
    
    U22 = [[0] * rank for _ in range(n * m)]
    V22 = [[0] * rank for _ in range(m * p)]
    W22 = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Add Strassen for A1*B1a
    # A1: i in {0,1}, j in {0,1} -> U row i*3+j
    # B1a: j in {0,1}, k in {0,1} -> V row j*4+k
    # C: i in {0,1}, k in {0,1} -> W row k*2+i
    
    # P1 = (A11+A22)(B11+B22)
    U22[0*3+0, col] = 1  # A[0,0]
    U22[1*3+1, col] = 1  # A[1,1]
    V22[0*4+0, col] = 1  # B[0,0]
    V22[1*4+1, col] = 1  # B[1,1]
    W22[0*2+0, col] = 1  # C[0,0]
    W22[1*2+1, col] = 1  # C[1,1]
    col += 1
    
    # P2 = (A21+A22)B11
    U22[1*3+0, col] = 1  # A[1,0]
    U22[1*3+1, col] = 1  # A[1,1]
    V22[0*4+0, col] = 1  # B[0,0]
    W22[0*2+1, col] = 1  # C[1,0]
    col += 1
    
    # P3 = A11(B12-B22)
    U22[0*3+0, col] = 1  # A[0,0]
    V22[0*4+1, col] = 1  # B[0,1]
    V22[1*4+1, col] = -1 # B[1,1]
    W22[1*2+0, col] = 1  # C[0,1]
    col += 1
    
    # P4 = A22(B21-B11)
    U22[1*3+1, col] = 1  # A[1,1]
    V22[1*4+0, col] = 1  # B[1,0]
    V22[0*4+0, col] = -1 # B[0,0]
    W22[0*2+1, col] = 1  # C[1,0]
    col += 1
    
    # P5 = (A11+A12)B22
    U22[0*3+0, col] = 1  # A[0,0]
    U22[0*3+1, col] = 1  # A[0,1]
    V22[1*4+1, col] = 1  # B[1,1]
    W22[1*2+0, col] = 1  # C[0,1]
    col += 1
    
    # P6 = (A21-A11)(B11+B12)
    U22[1*3+0, col] = 1  # A[1,0]
    U22[0*3+0, col] = -1 # A[0,0]
    V22[0*4+0, col] = 1  # B[0,0]
    V22[0*4+1, col] = 1  # B[0,1]
    W22[0*2+1, col] = 1  # C[1,0]
    col += 1
    
    # P7 = (A12-A22)(B21+B22)
    U22[0*3+1, col] = 1  # A[0,1]
    U22[1*3+1, col] = -1 # A[1,1]
    V22[1*4+0, col] = 1  # B[1,0]
    V22[1*4+1, col] = 1  # B[1,1]
    W22[1*2+0, col] = 1  # C[0,1]
    col += 1
    
    # Now add Strassen for A1*B1b
    # A1: i in {0,1}, j in {0,1} -> U row i*3+j
    # B1b: j in {0,1}, k in {2,3} -> V row j*4+k where k is 2 or 3
    # C: i in {0,1}, k in {2,3} -> W row k*2+i where k is 2 or 3
    
    # P1 = (A11+A22)(B11+B22) where B11=B[0,2], B22=B[1,3]
    U22[0*3+0, col] = 1  # A[0,0]
    U22[1*3+1, col] = 1  # A[1,1]
    V22[0*4+2, col] = 1  # B[0,2]
    V22[1*4+3, col] = 1  # B[1,3]
    W22[2*2+0, col] = 1  # C[0,2]
    W22[3*2+1, col] = 1  # C[1,3]
    col += 1
    
    # P2 = (A21+A22)B11 where B11=B[0,2]
    U22[1*3+0, col] = 1  # A[1,0]
    U22[1*3+1, col] = 1  # A[1,1]
    V22[0*4+2, col] = 1  # B[0,2]
    W22[2*2+1, col] = 1  # C[1,2]
    col += 1
    
    # P3 = A11(B12-B22) where B12=B[0,3], B22=B[1,3]
    U22[0*3+0, col] = 1  # A[0,0]
    V22[0*4+3, col] = 1  # B[0,3]
    V22[1*4+3, col] = -1 # B[1,3]
    W22[3*2+0, col] = 1  # C[0,3]
    col += 1
    
    # P4 = A22(B21-B11) where B21=B[1,2], B11=B[0,2]
    U22[1*3+1, col] = 1  # A[1,1]
    V22[1*4+2, col] = 1  # B[1,2]
    V22[0*4+2, col] = -1 # B[0,2]
    W22[2*2+1, col] = 1  # C[1,2]
    col += 1
    
    # P5 = (A11+A12)B22 where B22=B[1,3]
    U22[0*3+0, col] = 1  # A[0,0]
    U22[0*3+1, col] = 1  # A[0,1]
    V22[1*4+3, col] = 1  # B[1,3]
    W22[3*2+0, col] = 1  # C[0,3]
    col += 1
    
    # P6 = (A21-A11)(B11+B12) where B11=B[0,2], B12=B[0,3]
    U22[1*3+0, col] = 1  # A[1,0]
    U22[0*3+0, col] = -1 # A[0,0]
    V22[0*4+2, col] = 1  # B[0,2]
    V22[0*4+3, col] = 1  # B[0,3]
    W22[2*2+1, col] = 1  # C[1,2]
    col += 1
    
    # P7 = (A12-A22)(B21+B22) where B21=B[1,2], B22=B[1,3]
    U22[0*3+1, col] = 1  # A[0,1]
    U22[1*3+1, col] = -1 # A[1,1]
    V22[1*4+2, col] = 1  # B[1,2]
    V22[1*4+3, col] = 1  # B[1,3]
    W22[3*2+0, col] = 1  # C[0,3]
    col += 1
    
    # Now add A2*B2: 2x1 @ 1x4
    # A2: i in {0,1}, j=2 -> U row i*3+2
    # B2: j=2, k in {0,1,2,3} -> V row 2*4+k
    # C: i in {0,1}, k in {0,1,2,3} -> W row k*2+i
    
    for i in range(2):
        for k in range(4):
            U22[i*3+2, col] = 1
            V22[2*4+k, col] = 1
            W22[k*2+i, col] = 1
            col += 1
    
    # Verify the full decomposition
    exact = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for t in range(rank):
                    val += U22[i*m+j][t] * V22[j*p+k][t] * W22[k*n+i][t]
                if abs(val - 1.0) > 1e-9:
                    exact = False
                    break
            if not exact:
                break
        if not exact:
            break
    
    if exact:
        return {"U": U22, "V": V22, "W": W22}
    
    return {"U": U_school, "V": V_school, "W": W_school}
