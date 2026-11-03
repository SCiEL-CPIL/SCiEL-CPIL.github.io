import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    def validate(U, V, W):
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_arr[i*m+j]
                    v_val = V_arr[j*p+k]
                    w_val = W_arr[k*n+i]
                    s = np.dot(u_val, v_val * w_val)
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def build_rank_28():
        """
        Rank-28 decomposition using block Strassen.
        
        Structure:
        - A is 2x3: split as [A_left (2x2), A_right (2x1)]
        - B is 3x5: split as [B_top (2x5), B_bottom (1x5)]
        
        C = A*B = A_left * B_top + A_right * B_bottom
        
        For A_left * B_top (2x2 * 2x5):
          Split B_top into [B_TL (2x2), B_TR (2x3)]
          A_left * B_TL: use Strassen (rank 7)
          A_left * B_TR: use Strassen on first 2 cols + schoolbook on last col
            Actually, let's split B_top into three 2x? blocks...
            
        Let me use a cleaner decomposition:
        - Split the 3 middle dimension into {0,1} and {2}
        - Split the 5 output dimension into {0,1}, {2,3}, {4}
        
        Block 1: A[:,0:2] * B[0:2, 0:2] -> C[:, 0:2] (2x2 * 2x2 -> 2x2) : Strassen rank 7
        Block 2: A[:,0:2] * B[0:2, 2:4] -> C[:, 2:4] (2x2 * 2x2 -> 2x2) : Strassen rank 7  
        Block 3: A[:,0:2] * B[0:2, 4:5] -> C[:, 4:5] (2x2 * 2x1 -> 2x1) : Schoolbook rank 4
        Block 4: A[:,2:3] * B[2:3, 0:5] -> C[:, 0:5] (2x1 * 1x5 -> 2x5) : Schoolbook rank 10
        
        Total: 7 + 7 + 4 + 10 = 28
        """
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(rank_offset, j_offset, k_offset):
            """
            Add a Strassen decomposition for:
            A[i, j_offset:j_offset+2] * B[j_offset:j_offset+2, k_offset:k_offset+2] -> C[i, k_offset:k_offset+2]
            where i in {0,1}, j in {j_offset, j_offset+1}, k in {k_offset, k_offset+1}
            """
            r = rank_offset
            # A indices: U[i*m + j]
            # B indices: V[j*p + k]
            # C indices: W[k*n + i]
            
            # Strassen 7 terms for 2x2 * 2x2:
            # M1 = (A00 + A11)(B00 + B11) -> contributes to C00, C11
            # M2 = (A10 + A11)B00 -> contributes to C10
            # M3 = A00(B10 - B11) -> contributes to C01
            # M4 = A11(B01 - B00) -> contributes to C11, C01
            # M5 = (A00 + A01)B11 -> contributes to C01, C11
            # M6 = (A10 - A00)(B00 + B10) -> contributes to C10
            # M7 = (A01 - A11)(B01 + B11) -> contributes to C01, C11
            
            a00 = 0 * m + j_offset
            a01 = 0 * m + (j_offset + 1)
            a10 = 1 * m + j_offset
            a11 = 1 * m + (j_offset + 1)
            
            b00 = j_offset * p + k_offset
            b01 = j_offset * p + (k_offset + 1)
            b10 = (j_offset + 1) * p + k_offset
            b11 = (j_offset + 1) * p + (k_offset + 1)
            
            c00 = k_offset * n + 0
            c01 = (k_offset + 1) * n + 0
            c10 = k_offset * n + 1
            c11 = (k_offset + 1) * n + 1
            
            # M1 = (A00 + A11)(B00 + B11)
            U[a00][r] += 1
            U[a11][r] += 1
            V[b00][r] += 1
            V[b11][r] += 1
            W[c00][r] += 1
            W[c11][r] += 1
            
            # M2 = (A10 + A11)B00
            U[a10][r+1] += 1
            U[a11][r+1] += 1
            V[b00][r+1] += 1
            W[c10][r+1] += 1
            
            # M3 = A00(B10 - B11)
            U[a00][r+2] += 1
            V[b10][r+2] += 1
            V[b11][r+2] -= 1
            W[c01][r+2] += 1
            
            # M4 = A11(B01 - B00)
            U[a11][r+3] += 1
            V[b01][r+3] += 1
            V[b00][r+3] -= 1
            W[c11][r+3] += 1
            W[c01][r+3] += 1
            
            # M5 = (A00 + A01)B11
            U[a00][r+4] += 1
            U[a01][r+4] += 1
            V[b11][r+4] += 1
            W[c01][r+4] += 1
            W[c11][r+4] += 1
            
            # M6 = (A10 - A00)(B00 + B10)
            U[a10][r+5] += 1
            U[a00][r+5] -= 1
            V[b00][r+5] += 1
            V[b10][r+5] += 1
            W[c10][r+5] += 1
            
            # M7 = (A01 - A11)(B01 + B11)
            U[a01][r+6] += 1
            U[a11][r+6] -= 1
            V[b01][r+6] += 1
            V[b11][r+6] += 1
            W[c01][r+6] += 1
            W[c11][r+6] -= 1

        # Block 1: A[:,0:2] * B[0:2, 0:2] -> C[:, 0:2]
        # j_offset=0, k_offset=0
        add_strassen_block(0, 0, 0)
        
        # Block 2: A[:,0:2] * B[0:2, 2:4] -> C[:, 2:4]
        # j_offset=0, k_offset=2
        add_strassen_block(7, 0, 2)
        
        # Block 3: A[:,0:2] * B[0:2, 4:5] -> C[:, 4:5]
        # Schoolbook: 2x2 * 2x1 = 4 terms
        # A[0,0]*B[0,4] -> C[0,4]
        U[0*m+0][14] += 1
        V[0*p+4][14] += 1
        W[4*n+0][14] += 1
        
        # A[0,1]*B[1,4] -> C[0,4]
        U[0*m+1][15] += 1
        V[1*p+4][15] += 1
        W[4*n+0][15] += 1
        
        # A[1,0]*B[0,4] -> C[1,4]
        U[1*m+0][16] += 1
        V[0*p+4][16] += 1
        W[4*n+1][16] += 1
        
        # A[1,1]*B[1,4] -> C[1,4]
        U[1*m+1][17] += 1
        V[1*p+4][17] += 1
        W[4*n+1][17] += 1
        
        # Block 4: A[:,2:3] * B[2:3, 0:5] -> C[:, 0:5]
        # Schoolbook: 2x1 * 1x5 = 10 terms
        # A[0,2]*B[2,k] -> C[0,k] for k=0..4
        for k in range(5):
            col_idx = 18 + k
            U[0*m+2][col_idx] += 1
            V[2*p+k][col_idx] += 1
            W[k*n+0][col_idx] += 1
        
        # A[1,2]*B[2,k] -> C[1,k] for k=0..4
        for k in range(5):
            col_idx = 23 + k
            U[1*m+2][col_idx] += 1
            V[2*p+k][col_idx] += 1
            W[k*n+1][col_idx] += 1
        
        return {"U": U, "V": V, "W": W}

    # Try rank 28 first
    result = build_rank_28()
    if validate(result["U"], result["V"], result["W"]):
        return result

    # Fallback to schoolbook
    return get_schoolbook()
