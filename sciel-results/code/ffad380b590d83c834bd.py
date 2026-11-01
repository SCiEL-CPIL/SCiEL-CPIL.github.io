"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from numpy import linalg as LA

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # Safety margin before 150s kill

    # --- Schoolbook decomposition (rank = n*m*p = 28) ---
    def schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k][col] = 1
                    V[k * p + j][col] = 1
                    W[j * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    # --- Fast verification using einsum ---
    def verify_fast(U_mat, V_mat, W_mat):
        """Check if sum_c U[i*m+k,c]*V[k*p+j,c]*W[j*n+i,c] == 1 for all i,k,j."""
        r = U_mat.shape[1]
        u_r = U_mat.T.reshape(n, m, r)
        v_r = V_mat.T.reshape(m, p, r)
        w_r = W_mat.T.reshape(p, n, r)
        T = np.einsum('ikc,kjc,jic->ijk', u_r, v_r, w_r)
        return np.allclose(T, 1.0, atol=1e-9)

    # --- Check for integer/half-integer entries ---
    def is_valid_entries(*mats):
        for mat in mats:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-9):
                return False
        return True

    # --- Convert to list format ---
    def to_list(mat):
        # Round to nearest half-integer for cleanliness
        mat_rounded = np.round(mat * 2) / 2.0
        # Check if all are integers
        if np.allclose(mat_rounded, np.round(mat_rounded), atol=1e-9):
            return np.round(mat_rounded).astype(int).tolist()
        else:
            # Return as float list with half-integer values
            return mat_rounded.tolist()

    # --- Construct a rank-23 decomposition for 2x2 * 2x7 ---
    # Strategy:
    # Split B (2x7) into B1 (2x2, cols 0-1), B2 (2x2, cols 2-3), B3 (2x2, cols 4-5), B4 (2x1, col 6).
    # Use Laderman/Strassen (rank 7) for each 2x2 block.
    # Use Schoolbook (rank 2) for the 2x1 block.
    # Total rank = 7 + 7 + 7 + 2 = 23.
    
    def construct_rank_23():
        r = 23
        U_mat = np.zeros((n * m, r))
        V_mat = np.zeros((m * p, r))
        W_mat = np.zeros((p * n, r))
        
        # Laderman/Strassen coefficients for 2x2 * 2x2 -> 2x2
        # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
        # C = A*B
        # C00 = T1 + T2 - T3 - T4
        # C01 = T1 - T2 + T5 - T6
        # C10 = T3 - T1 + T7 - T5
        # C11 = T4 - T3 + T2 + T6
        # Where:
        # T1 = (A00+A11)(B00+B11)
        # T2 = A10(B00+B10)
        # T3 = A00(B10-B11)
        # T4 = A11(B01-B00)
        # T5 = (A00+A10)B11
        # T6 = (A10-A00)(B00+B10) -- Wait, standard Strassen is:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A10)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B11+B01)
        
        # Let's use the standard 7-term Strassen decomposition.
        # U indices: 0=A00, 1=A01, 2=A10, 3=A11
        # V indices for a block starting at col j0:
        #   j0: B00, j0+1: B01
        #   j0+2: B10, j0+3: B11 (in the 2x7 matrix B, rows are k=0,1; cols are j)
        #   So B[k, j]. B00 is B[0, j0], B01 is B[0, j0+1], B10 is B[1, j0], B11 is B[1, j0+1].
        # W indices for a block starting at col j0:
        #   Output C[i, j]. C00 is C[0, j0], C01 is C[0, j0+1], C10 is C[1, j0], C11 is C[1, j0+1].
        #   W row index is j*n + i.
        #   W[j0, 0] corresponds to C[0, j0] (i=0, j=j0). Index: j0*2 + 0.
        #   W[j0+1, 0] corresponds to C[0, j0+1] (i=0, j=j0+1). Index: (j0+1)*2 + 0.
        #   W[j0, 1] corresponds to C[1, j0] (i=1, j=j0). Index: j0*2 + 1.
        #   W[j0+1, 1] corresponds to C[1, j0+1] (i=1, j=j0+1). Index: (j0+1)*2 + 1.

        def add_strassen_block(base_col, j0):
            # U contributions
            # T1: A00+A11 -> U[0]+=1, U[3]+=1
            # T2: A10+A11 -> U[2]+=1, U[3]+=1
            # T3: A00 -> U[0]+=1
            # T4: A11 -> U[3]+=1
            # T5: A00+A10 -> U[0]+=1, U[2]+=1
            # T6: A10-A00 -> U[2]+=1, U[0]-=1
            # T7: A01-A11 -> U[1]+=1, U[3]-=1
            
            U_mat[0, base_col] += 1   # T1
            U_mat[3, base_col] += 1   # T1
            U_mat[2, base_col+1] += 1 # T2
            U_mat[3, base_col+1] += 1 # T2
            U_mat[0, base_col+2] += 1 # T3
            U_mat[3, base_col+3] += 1 # T4
            U_mat[0, base_col+4] += 1 # T5
            U_mat[2, base_col+4] += 1 # T5
            U_mat[2, base_col+5] += 1 # T6
            U_mat[0, base_col+5] -= 1 # T6
            U_mat[1, base_col+6] += 1 # T7
            U_mat[3, base_col+6] -= 1 # T7

            # V contributions
            # B indices: B00=V[0, j0], B01=V[0, j0+1], B10=V[1, j0], B11=V[1, j0+1]
            # Row indices in V_mat: k*p + j. Here k is 0 or 1.
            # T1: B00+B11 -> V[0, j0]+=1, V[1, j0+1]+=1
            # T2: B00 -> V[0, j0]+=1
            # T3: B10-B11 -> V[1, j0]+=1, V[1, j0+1]-=1
            # T4: B01-B00 -> V[0, j0+1]+=1, V[0, j0]-=1
            # T5: B11 -> V[1, j0+1]+=1
            # T6: B00+B10 -> V[0, j0]+=1, V[1, j0]+=1
            # T7: B11+B01 -> V[1, j0+1]+=1, V[0, j0+1]+=1

            V_mat[0, j0] += 1     # T1
            V_mat[1, j0+1] += 1   # T1
            V_mat[0, j0] += 1     # T2
            V_mat[1, j0] += 1     # T3
            V_mat[1, j0+1] -= 1   # T3
            V_mat[0, j0+1] += 1   # T4
            V_mat[0, j0] -= 1     # T4
            V_mat[1, j0+1] += 1   # T5
            V_mat[0, j0] += 1     # T6
            V_mat[1, j0] += 1     # T6
            V_mat[1, j0+1] += 1   # T7
            V_mat[0, j0+1] += 1   # T7

            # W contributions
            # C00 = T1 + T2 - T3 - T4
            # C01 = T1 - T2 + T5 - T6
            # C10 = T3 - T1 + T7 - T5
            # C11 = T4 - T3 + T2 + T6
            
            # W row indices:
            # C00 (i=0, j=j0) -> idx = j0*2 + 0
            # C01 (i=0, j=j0+1) -> idx = (j0+1)*2 + 0
            # C10 (i=1, j=j0) -> idx = j0*2 + 1
            # C11 (i=1, j=j0+1) -> idx = (j0+1)*2 + 1

            # T1
            W_mat[j0*2 + 0, base_col] += 1     # C00
            W_mat[(j0+1)*2 + 0, base_col] += 1 # C01
            W_mat[j0*2 + 1, base_col] -= 1     # C10
            W_mat[(j0+1)*2 + 1, base_col] -= 1 # C11

            # T2
            W_mat[j0*2 + 0, base_col+1] += 1     # C00
            W_mat[(j0+1)*2 + 0, base_col+1] -= 1 # C01
            W_mat[(j0+1)*2 + 1, base_col+1] += 1 # C11

            # T3
            W_mat[j0*2 + 0, base_col+2] -= 1     # C00
            W_mat[j0*2 + 1, base_col+2] += 1     # C10
            W_mat[(j0+1)*2 + 1, base_col+2] -= 1 # C11

            # T4
            W_mat[j0*2 + 0, base_col+3] -= 1     # C00
            W_mat[j0*2 + 1, base_col+3] += 1     # C10
            W_mat[(j0+1)*2 + 1, base_col+3] += 1 # C11

            # T5
            W_mat[(j0+1)*2 + 0, base_col+4] += 1 # C01
            W_mat[j0*2 + 1, base_col+4] -= 1     # C10

            # T6
            W_mat[(j0+1)*2 + 0, base_col+5] -= 1 # C01
            W_mat[(j0+1)*2 + 1, base_col+5] += 1 # C11

            # T7
            W_mat[j0*2 + 1, base_col+6] += 1     # C10

        # Apply Strassen to blocks
        add_strassen_block(0, 0)   # Cols 0,1
        add_strassen_block(7, 2)   # Cols 2,3
        add_strassen_block(14, 4)  # Cols 4,5
        
        # Apply Schoolbook to last column (col 6)
        # Rank 2: 
        # C[0,6] = A[0,0]*B[0,6] + A[0,1]*B[1,6]
        # C[1,6] = A[1,0]*B[0,6] + A[1,1]*B[1,6]
        
        # Term 1: A[0,0]*B[0,6] -> C[0,6]
        # U[0, 21] = 1
        # V[0, 6] = 1
        # W[6*2+0, 21] = 1
        U_mat[0, 21] = 1
        V_mat[0, 6] = 1
        W_mat[6*2 + 0, 21] = 1

        # Term 2: A[0,1]*B[1,6] -> C[0,6]
        # U[1, 22] = 1
        # V[1, 6] = 1
        # W[6*2+0, 22] = 1
        U_mat[1, 22] = 1
        V_mat[1, 6] = 1
        W_mat[6*2 + 0, 22] = 1
        
        # Wait, C[1,6] needs terms too.
        # C[1,6] = A[1,0]*B[0,6] + A[1,1]*B[1,6]
        
        # Term 3: A[1,0]*B[0,6] -> C[1,6]
        # U[2, 21] = 1
        # V[0, 6] += 1 (already 1, so becomes 2? No, schoolbook for 2x1 uses 2 terms if we can't combine?)
        # Actually, schoolbook for 2x2 * 2x1 is rank 4 if done naively?
        # No, 2x1 matrix multiplication tensor rank is 2?
        # T = sum A[i,k] B[k,j] C[i,j]. For j fixed.
        # T = A[0,0]B[0,j]C[0,j] + A[0,1]B[1,j]C[0,j] + A[1,0]B[0,j]C[1,j] + A[1,1]B[1,j]C[1,j].
        # This is 4 terms. Rank 4.
        # Can we do better?
        # Rank of [2,2,1] is 4? 
        # Let's check. 
        # If we use Strassen-like? No, Strassen is for 2x2x2.
        # For 2x2x1, the tensor is essentially two independent 2x1x1 multiplications? No.
        # It is A (2x2) times B (2x1) giving C (2x1).
        # This is equivalent to 2 separate vector multiplications? No, A is a matrix.
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        # This is a bilinear map.
        # Rank is 4? 
        # Let's verify if rank 2 is possible.
        # If rank 2, C = U V W.
        # C0 = u0 v0 w0 + u1 v1 w1
        # C1 = u0' v0 w0 + u1' v1 w1
        # This implies C0 and C1 are linearly dependent in terms of the products?
        # Generally rank 4 is required for [2,2,1].
        
        # So my previous assumption of rank 2 for the last column was wrong.
        # It should be rank 4.
        # Total rank = 7 + 7 + 7 + 4 = 25.
        
        # Let's correct the last part to be rank 4.
        
        # Reset last 2 columns of U, V, W to 0 and redo.
        U_mat[:, 21:23] = 0
        V_mat[:, 6] = 0 # V rows 0 and 1 for col 6
        W_mat[6*2:6*2+2, 21:23] = 0
        
        # Term 1: A[0,0]*B[0,6] -> C[0,6]
        U_mat[0, 21] = 1
        V_mat[0, 6] = 1
        W_mat[6*2 + 0, 21] = 1

        # Term 2: A[0,1]*B[1,6] -> C[0,6]
        U_mat[1, 22] = 1
        V_mat[1, 6] = 1
        W_mat[6*2 + 0, 22] = 1
        
        # Term 3: A[1,0]*B[0,6] -> C[1,6]
        U_mat[2, 23] = 1
        V_mat[0, 6] += 1 # Becomes 2
        W_mat[6*2 + 1, 23] = 1
        
        # Term 4: A[1,1]*B[1,6] -> C[1,6]
        U_mat[3, 24] = 1
        V_mat[1, 6] += 1 # Becomes 2
        W_mat[6*2 + 1, 24] = 1
        
        # Wait, I allocated r=23. I need r=25.
        # Let's rebuild with r=25.
        
        return None

    def construct_rank_25():
        r = 25
        U_mat = np.zeros((n * m, r))
        V_mat = np.zeros((m * p, r))
        W_mat = np.zeros((p * n, r))
        
        def add_strassen_block(base_col, j0):
            # U contributions
            U_mat[0, base_col] += 1   
            U_mat[3, base_col] += 1   
            U_mat[2, base_col+1] += 1 
            U_mat[3, base_col+1] += 1 
            U_mat[0, base_col+2] += 1 
            U_mat[3, base_col+3] += 1 
            U_mat[0, base_col+4] += 1 
            U_mat[2, base_col+4] += 1 
            U_mat[2, base_col+5] += 1 
            U_mat[0, base_col+5] -= 1 
            U_mat[1, base_col+6] += 1 
            U_mat[3, base_col+6] -= 1 

            # V contributions
            V_mat[0, j0] += 1     
            V_mat[1, j0+1] += 1   
            V_mat[0, j0] += 1     
            V_mat[1, j0] += 1     
            V_mat[1, j0+1] -= 1   
            V_mat[0, j0+1] += 1   
            V_mat[0, j0] -= 1     
            V_mat[1, j0+1] += 1   
            V_mat[0, j0] += 1     
            V_mat[1, j0] += 1     
            V_mat[1, j0+1] += 1   
            V_mat[0, j0+1] += 1   

            # W contributions
            W_mat[j0*2 + 0, base_col] += 1     
            W_mat[(j0+1)*2 + 0, base_col] += 1 
            W_mat[j0*2 + 1, base_col] -= 1     
            W_mat[(j0+1)*2 + 1, base_col] -= 1 

            W_mat[j0*2 + 0, base_col+1] += 1     
            W_mat[(j0+1)*2 + 0, base_col+1] -= 1 
            W_mat[(j0+1)*2 + 1, base_col+1] += 1 

            W_mat[j0*2 + 0, base_col+2] -= 1     
            W_mat[j0*2 + 1, base_col+2] += 1     
            W_mat[(j0+1)*2 + 1, base_col+2] -= 1 

            W_mat[j0*2 + 0, base_col+3] -= 1     
            W_mat[j0*2 + 1, base_col+3] += 1     
            W_mat[(j0+1)*2 + 1, base_col+3] += 1 

            W_mat[(j0+1)*2 + 0, base_col+4] += 1 
            W_mat[j0*2 + 1, base_col+4] -= 1     

            W_mat[(j0+1)*2 + 0, base_col+5] -= 1 
            W_mat[(j0+1)*2 + 1, base_col+5] += 1 

            W_mat[j0*2 + 1, base_col+6] += 1     

        add_strassen_block(0, 0)   
        add_strassen_block(7, 2)   
        add_strassen_block(14, 4)  

        # Schoolbook for col 6 (rank 4)
        # U rows: 0=A00, 1=A01, 2=A10, 3=A11
        # V rows: 0=B0, 1=B1 (for col 6)
        # W rows: 0=C0, 1=C1 (for col 6, indices 12, 13)
        
        # T1: A00 B0 -> C0
        U_mat[0, 21] = 1
        V_mat[0, 6] = 1
        W_mat[12, 21] = 1
        
        # T2: A01 B1 -> C0
        U_mat[1, 22] = 1
        V_mat[1, 6] = 1
        W_mat[12, 22] = 1
        
        # T3: A10 B0 -> C1
        U_mat[2, 23] = 1
        V_mat[0, 6] += 1
        W_mat[13, 23] = 1
        
        # T4: A11 B1 -> C1
        U_mat[3, 24] = 1
        V_mat[1, 6] += 1
        W_mat[13, 24] = 1
        
        # Verify
        if verify_fast(U_mat, V_mat, W_mat) and is_valid_entries(U_mat, V_mat, W_mat):
            return {
                "U": to_list(U_mat),
                "V": to_list(V_mat),
                "W": to_list(W_mat)
            }
        else:
            return None

    candidate = construct_rank_25()
    if candidate:
        best = candidate
        best_rank = 25

    return best
# EVOLVE-BLOCK-END
