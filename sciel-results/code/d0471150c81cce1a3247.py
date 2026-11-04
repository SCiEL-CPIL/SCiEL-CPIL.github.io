import numpy as np
import time

n, m, p = 2, 5, 6

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin

    def check_decomposition(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def get_schoolbook():
        """Return the schoolbook decomposition with rank n*m*p = 60."""
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
        return U, V, W

    def get_strassen_2x2x2():
        """
        Return a rank-7 Strassen decomposition for 2x2 x 2x2 multiplication.
        Uses the standard Strassen formulas.
        """
        # Strassen's 7 terms for C = A * B where A, B are 2x2
        # C[0,0] = M1 - M4 + M6 + M7
        # C[0,1] = M2 + M4
        # C[1,0] = M3 + M5
        # C[1,1] = M1 - M2 + M5 + M6
        #
        # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        # M2 = (A[1,0] + A[1,1]) * B[0,0]
        # M3 = A[0,0] * (B[0,1] - B[1,1])
        # M4 = A[1,1] * (B[1,0] - B[0,0])
        # M5 = (A[0,0] + A[0,1]) * B[1,1]
        # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
        # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Term 0: M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        U[0, 0] = 1  # A[0,0]
        U[3, 0] = 1  # A[1,1]
        V[0, 0] = 1  # B[0,0]
        V[3, 0] = 1  # B[1,1]
        W[0, 0] = 1  # C[0,0]
        W[3, 0] = 1  # C[1,1]
        
        # Term 1: M2 = (A[1,0] + A[1,1]) * B[0,0]
        U[2, 1] = 1  # A[1,0]
        U[3, 1] = 1  # A[1,1]
        V[0, 1] = 1  # B[0,0]
        W[1, 1] = 1  # C[0,1]
        
        # Term 2: M3 = A[0,0] * (B[0,1] - B[1,1])
        U[0, 2] = 1  # A[0,0]
        V[1, 2] = 1  # B[0,1]
        V[3, 2] = -1 # B[1,1]
        W[2, 2] = 1  # C[1,0]
        
        # Term 3: M4 = A[1,1] * (B[1,0] - B[0,0])
        U[3, 3] = 1  # A[1,1]
        V[2, 3] = 1  # B[1,0]
        V[0, 3] = -1 # B[0,0]
        W[0, 3] = -1 # C[0,0]
        W[1, 3] = 1  # C[0,1]
        
        # Term 4: M5 = (A[0,0] + A[0,1]) * B[1,1]
        U[0, 4] = 1  # A[0,0]
        U[1, 4] = 1  # A[0,1]
        V[3, 4] = 1  # B[1,1]
        W[0, 4] = -1 # C[0,0]
        W[2, 4] = 1  # C[1,0]
        
        # Term 5: M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
        U[2, 5] = 1  # A[1,0]
        U[0, 5] = -1 # A[0,0]
        V[0, 5] = 1  # B[0,0]
        V[1, 5] = 1  # B[0,1]
        W[0, 5] = 1  # C[0,0]
        W[3, 5] = 1  # C[1,1]
        
        # Term 6: M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
        U[1, 6] = 1  # A[0,1]
        U[3, 6] = -1 # A[1,1]
        V[2, 6] = 1  # B[1,0]
        V[3, 6] = 1  # B[1,1]
        W[0, 6] = 1  # C[0,0]
        
        return U, V, W

    def combine_decompositions(U1, V1, W1, U2, V2, W2):
        """Combine two decompositions by concatenating their rank dimensions."""
        U = np.hstack([U1, U2])
        V = np.hstack([V1, V2])
        W = np.hstack([W1, W2])
        return U, V, W

    def get_hybrid_strassen_schoolbook():
        """
        Construct a hybrid decomposition using Strassen for one 2x2 block
        and schoolbook for the rest.
        
        We partition the 2x5 and 5x6 matrices into blocks.
        Let's use the top-left 2x2 block of A and B for Strassen.
        
        A is 2x5:
        A = [A11 A12]
             [A21 A22]
        where A11 is 2x2, A12 is 2x3, A21 is 2x2, A22 is 2x3.
        
        B is 5x6:
        B = [B11 B12]
            [B21 B22]
        where B11 is 2x2, B12 is 2x4, B21 is 3x2, B22 is 3x4.
        
        C = A * B = [A11*B11 + A12*B21,  A11*B12 + A12*B22]
                    [A21*B11 + A22*B21,  A21*B12 + A22*B22]
        
        We can use Strassen for A11*B11 (rank 7 instead of 8).
        The rest is schoolbook:
        A12*B21: 2x3 * 3x2 = 2x2, rank 12
        A11*B12: 2x2 * 2x4 = 2x4, rank 16
        A12*B22: 2x3 * 3x4 = 2x4, rank 24
        A21*B11: 2x2 * 2x2 = 2x2, rank 8
        A22*B21: 2x3 * 3x2 = 2x2, rank 12
        A21*B12: 2x2 * 2x4 = 2x4, rank 16
        A22*B22: 2x3 * 3x4 = 2x4, rank 24
        
        Total schoolbook rank for non-Strassen parts: 12+16+24+8+12+16+24 = 112
        But wait, this is overcounting. The total schoolbook rank for the full 2x5*5x6 is 60.
        
        Let me reconsider. The schoolbook algorithm computes each C[i,k] independently.
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        For the 2x2 block A11*B11, we have 4 outputs (C[0,0], C[0,1], C[1,0], C[1,1] within the block).
        Schoolbook uses 8 multiplications for this block.
        Strassen uses 7.
        
        So we can save 1 multiplication by replacing the 8 schoolbook terms for this block with 7 Strassen terms.
        
        The other 56 terms remain schoolbook.
        Total rank = 56 + 7 = 63. That's worse than 60.
        
        Wait, I need to think about this differently. The Strassen algorithm is for 2x2 matrices.
        If we apply it to a sub-block, we need to be careful about how the indices map.
        
        Actually, the standard Strassen algorithm gives rank 7 for 2x2x2, which is better than 8.
        But when embedded in a larger matrix multiplication, the savings might not be direct.
        
        Let me try a different approach: use Strassen for the entire 2x2x2 corner and keep the rest schoolbook.
        The issue is that Strassen computes linear combinations of A and B, which may affect other terms.
        
        Actually, the cleanest way is to just use Strassen for one 2x2x2 sub-problem and schoolbook for the rest.
        The rank would be 7 + (60 - 8) = 59.
        
        Let me construct this explicitly.
        """
        # Start with schoolbook
        U_school, V_school, W_school = get_schoolbook()
        
        # Now, replace the 8 schoolbook terms for the top-left 2x2 block with 7 Strassen terms.
        # The top-left 2x2 block involves:
        # i in {0,1}, j in {0,1}, k in {0,1}
        # These are the first 8 terms in the schoolbook decomposition (columns 0-7).
        
        # Create the Strassen decomposition for 2x2x2
        U_str, V_str, W_str = get_strassen_2x2x2()
        
        # The Strassen U, V, W are for indices:
        # U: (i, j) where i in {0,1}, j in {0,1} -> 4 rows
        # V: (j, k) where j in {0,1}, k in {0,1} -> 4 rows
        # W: (k, i) where k in {0,1}, i in {0,1} -> 4 rows
        
        # In the full problem:
        # U has 10 rows (i*m + j = i*5 + j)
        # V has 30 rows (j*p + k = j*6 + k)
        # W has 12 rows (k*n + i = k*2 + i)
        
        # For the top-left 2x2 block:
        # U rows: i*5 + j for i in {0,1}, j in {0,1} -> 0, 1, 5, 6
        # V rows: j*6 + k for j in {0,1}, k in {0,1} -> 0, 1, 6, 7
        # W rows: k*2 + i for k in {0,1}, i in {0,1} -> 0, 1, 2, 3
        
        # Create new U, V, W with rank 59
        U_new = np.zeros((10, 59))
        V_new = np.zeros((30, 59))
        W_new = np.zeros((12, 59))
        
        # Copy schoolbook terms for columns 8-59 (the non-top-left-block terms)
        # Schoolbook columns 8-59 correspond to all (i,j,k) except the first 8
        U_new[:, 8:59] = U_school[:, 8:59]
        V_new[:, 8:59] = V_school[:, 8:59]
        W_new[:, 8:59] = W_school[:, 8:59]
        
        # Now fill columns 0-6 with Strassen terms
        # Map Strassen U (4x7) to U_new
        # Strassen U row 0 -> A[0,0] -> U_new row 0*5+0 = 0
        # Strassen U row 1 -> A[0,1] -> U_new row 0*5+1 = 1
        # Strassen U row 2 -> A[1,0] -> U_new row 1*5+0 = 5
        # Strassen U row 3 -> A[1,1] -> U_new row 1*5+1 = 6
        U_new[0, 0:7] = U_str[0, :]
        U_new[1, 0:7] = U_str[1, :]
        U_new[5, 0:7] = U_str[2, :]
        U_new[6, 0:7] = U_str[3, :]
        
        # Map Strassen V (4x7) to V_new
        # Strassen V row 0 -> B[0,0] -> V_new row 0*6+0 = 0
        # Strassen V row 1 -> B[0,1] -> V_new row 0*6+1 = 1
        # Strassen V row 2 -> B[1,0] -> V_new row 1*6+0 = 6
        # Strassen V row 3 -> B[1,1] -> V_new row 1*6+1 = 7
        V_new[0, 0:7] = V_str[0, :]
        V_new[1, 0:7] = V_str[1, :]
        V_new[6, 0:7] = V_str[2, :]
        V_new[7, 0:7] = V_str[3, :]
        
        # Map Strassen W (4x7) to W_new
        # Strassen W row 0 -> C[0,0] -> W_new row 0*2+0 = 0
        # Strassen W row 1 -> C[0,1] -> W_new row 0*2+1 = 1
        # Strassen W row 2 -> C[1,0] -> W_new row 1*2+0 = 2
        # Strassen W row 3 -> C[1,1] -> W_new row 1*2+1 = 3
        W_new[0, 0:7] = W_str[0, :]
        W_new[1, 0:7] = W_str[1, :]
        W_new[2, 0:7] = W_str[2, :]
        W_new[3, 0:7] = W_str[3, :]
        
        return U_new, V_new, W_new

    # Try the hybrid Strassen-schoolbook decomposition (rank 59)
    if time.time() < deadline - 5:
        U, V, W = get_hybrid_strassen_schoolbook()
        if check_decomposition(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook (Rank 60)
    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
