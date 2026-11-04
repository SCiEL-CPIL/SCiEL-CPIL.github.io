import numpy as np
import time

def construct():
    n, m, p = 4, 5, 6
    # Target rank 112
    rank = 112
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a decomposition block
    # u_idx: list of (i, j) indices for U
    # v_idx: list of (j, k) indices for V
    # w_idx: list of (k, i) indices for W
    # U_block, V_block, W_block: factor matrices for the block
    # offset: current column offset in U, V, W
    def add_block(u_indices, v_indices, w_indices, U_b, V_b, W_b, offset):
        r_b = U_b.shape[1]
        for idx_u, (i, j) in enumerate(u_indices):
            U[i * m + j, offset:offset+r_b] = U_b[idx_u, :]
        for idx_v, (j, k) in enumerate(v_indices):
            V[j * p + k, offset:offset+r_b] = V_b[idx_v, :]
        for idx_w, (k, i) in enumerate(w_indices):
            W[k * n + i, offset:offset+r_b] = W_b[idx_w, :]
        return offset + r_b

    # --- Strassen 2x2x2 ---
    # C = A B
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Indices for 2x2 A, 2x2 B, 2x2 C
    # A: (0,0), (0,1), (1,0), (1,1)
    # B: (0,0), (0,1), (1,0), (1,1)
    # C: (0,0), (0,1), (1,0), (1,1) -> W indices (k, i)
    
    # We need to define U, V, W for the 7 terms.
    # Term 1: (A00+A11) * (B00+B11)
    # U: A00=1, A11=1
    # V: B00=1, B11=1
    # W: C00=1, C11=1 (with signs later? No, W is just the output coefficient)
    # The decomposition is sum U_r V_r W_r = T.
    # For Strassen, the "tensor" is the multiplication tensor.
    # We need to map the Strassen factors to the tensor factors.
    
    # Let's build the 2x2x2 decomposition explicitly.
    # U is 4x7, V is 4x7, W is 4x7.
    # Rows of U: A00, A01, A10, A11
    # Rows of V: B00, B01, B10, B11
    # Rows of W: C00, C01, C10, C11 (Note: W index is k*n+i. For 2x2, n=2.
    # C00 -> k=0, i=0 -> 0*2+0=0
    # C01 -> k=1, i=0 -> 1*2+0=2
    # C10 -> k=0, i=1 -> 0*2+1=1
    # C11 -> k=1, i=1 -> 1*2+1=3
    # So W rows order: C00, C10, C01, C11?
    # Wait, W[k*n+i].
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # So W[0]=C00, W[1]=C10, W[2]=C01, W[3]=C11.
    
    U_222 = np.zeros((4, 7))
    V_222 = np.zeros((4, 7))
    W_222 = np.zeros((4, 7))
    
    # M1
    U_222[0, 0] = 1; U_222[3, 0] = 1
    V_222[0, 0] = 1; V_222[3, 0] = 1
    W_222[0, 0] = 1; W_222[3, 0] = 1
    
    # M2
    U_222[2, 1] = 1; U_222[3, 1] = 1
    V_222[0, 1] = 1
    W_222[1, 1] = 1; W_222[3, 1] = -1 # C10 = M2+M4, C11 = M1-M2+M3+M6. 
    # Wait, C10 = M2+M4. So W for C10 is +1.
    # C11 = M1 - M2 + M3 + M6. So W for C11 is -1.
    W_222[1, 1] = 1
    W_222[3, 1] = -1
    
    # M3
    U_222[0, 2] = 1
    V_222[2, 2] = 1; V_222[3, 2] = -1
    W_222[2, 2] = 1; W_222[3, 2] = 1 # C01 = M3+M5, C11 = ... +M3
    
    # M4
    U_222[3, 3] = 1
    V_222[1, 3] = 1; V_222[2, 3] = -1
    W_222[0, 3] = 1; W_222[1, 3] = 1 # C00 = ... +M4, C10 = M2+M4
    
    # M5
    U_222[0, 4] = 1; U_222[1, 4] = 1
    V_222[3, 4] = 1
    W_222[0, 4] = -1; W_222[2, 4] = 1 # C00 = ... -M5, C01 = M3+M5
    
    # M6
    U_222[2, 5] = 1; U_222[0, 5] = -1
    V_222[0, 5] = 1; V_222[2, 5] = 1
    W_222[3, 5] = 1 # C11 = ... +M6
    
    # M7
    U_222[1, 6] = 1; U_222[3, 6] = -1
    V_222[1, 6] = 1; V_222[3, 6] = 1
    W_222[0, 6] = 1 # C00 = ... +M7
    
    # Verify 2x2x2
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # We assume the standard flattening.
    
    # --- Build 2x2x1 (Rank 4) ---
    # A (2x2), B (2x1), C (2x1)
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    U_221 = np.zeros((4, 4))
    V_221 = np.zeros((2, 4)) # B is 2x1, so 2 rows
    W_221 = np.zeros((2, 4)) # C is 2x1, so 2 rows. W index k*n+i. n=2? No, for this block, output is 2x1.
    # Wait, the global W is p*n. Here p=1, n=2. So W size 2.
    # W[k*n+i] -> W[0*2+0]=W[0], W[0*2+1]=W[1].
    
    U_221[0, 0] = 1; V_221[0, 0] = 1; W_221[0, 0] = 1
    U_221[1, 1] = 1; V_221[1, 1] = 1; W_221[0, 1] = 1
    U_221[2, 2] = 1; V_221[0, 2] = 1; W_221[1, 2] = 1
    U_221[3, 3] = 1; V_221[1, 3] = 1; W_221[1, 3] = 1
    
    # --- Build 2x2x3 (Rank 11) ---
    # A (2x2), B (2x3), C (2x3)
    # B = [B_22 (2x2) | B_21 (2x1)]
    # C = [C_22 (2x2) | C_21 (2x1)]
    # C_22 = A B_22 (Rank 7)
    # C_21 = A B_21 (Rank 4)
    
    U_223 = np.zeros((4, 11))
    V_223 = np.zeros((6, 11)) # B is 2x3 -> 6 elements
    W_223 = np.zeros((6, 11)) # C is 2x3 -> 6 elements. W index k*n+i. n=2.
    # C indices:
    # k=0 (row 0 of C): C00 (i=0), C01 (i=1) -> W[0], W[1]
    # k=1 (row 1 of C): C10 (i=0), C11 (i=1) -> W[2], W[3]
    # k=2 (row 2 of C): C20 (i=0), C21 (i=1) -> W[4], W[5]
    # Wait, C is 2x3. Rows are i=0,1. Cols are k=0,1,2.
    # W index is k*n+i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # k=2, i=0 -> 4
    # k=2, i=1 -> 5
    
    # Part 1: 2x2x2 (cols 0,1 of B, cols 0,1 of C)
    U_223[:, 0:7] = U_222
    # V_223 rows: B00, B01, B10, B11, B02, B12
    # B_22 is B00, B01, B10, B11. Indices 0,1,2,3.
    V_223[0:4, 0:7] = V_222
    # W_223 rows: C00, C01, C10, C11, C20, C21
    # C_22 is C00, C01, C10, C11. Indices 0,1,2,3.
    W_223[0:4, 0:7] = W_222
    
    # Part 2: 2x2x1 (col 2 of B, col 2 of C)
    U_223[:, 7:11] = U_221
    # V_223 rows: B02, B12. Indices 4,5.
    V_223[4:6, 7:11] = V_221
    # W_223 rows: C20, C21. Indices 4,5.
    W_223[4:6, 7:11] = W_221
    
    # --- Build 2x1x3 (Rank 6) ---
    # A (2x1), B (1x3), C (2x3)
    # C0 = A0 (B0, B1, B2)
    # C1 = A1 (B0, B1, B2)
    U_213 = np.zeros((2, 6))
    V_213 = np.zeros((3, 6))
    W_213 = np.zeros((6, 6))
    
    # A0 * B0 -> C00
    U_213[0, 0] = 1; V_213[0, 0] = 1; W_213[0, 0] = 1
    # A0 * B1 -> C01
    U_213[0, 1] = 1; V_213[1, 1] = 1; W_213[1, 1] = 1
    # A0 * B2 -> C02
    U_213[0, 2] = 1; V_213[2, 2] = 1; W_213[2, 2] = 1
    # A1 * B0 -> C10
    U_213[1, 3] = 1; V_213[0, 3] = 1; W_213[3, 3] = 1
    # A1 * B1 -> C11
    U_213[1, 4] = 1; V_213[1, 4] = 1; W_213[4, 4] = 1
    # A1 * B2 -> C12
    U_213[1, 5] = 1; V_213[2, 5] = 1; W_213[5, 5] = 1
    
    # --- Build 2x3x3 (Rank 17) ---
    # A (2x3), B (3x3), C (2x3)
    # A = [A_22 (2x2) | A_21 (2x1)]
    # B = [B_23 (2x3); B_13 (1x3)]
    # C = A_22 B_23 + A_21 B_13
    # Part 1: 2x2x3 (Rank 11)
    # Part 2: 2x1x3 (Rank 6)
    
    U_233 = np.zeros((6, 17))
    V_233 = np.zeros((9, 17))
    W_233 = np.zeros((6, 17))
    
    # Part 1: 2x2x3
    U_233[0:4, 0:11] = U_223
    # V_233 rows: B00, B01, B02, B10, B11, B12, B20, B21, B22
    # B_23 is B00, B01, B02, B10, B11, B12. Indices 0-5.
    V_233[0:6, 0:11] = V_223
    # W_233 rows: C00, C01, C02, C10, C11, C12
    # C is 2x3. Same as W_223.
    W_233[:, 0:11] = W_223
    
    # Part 2: 2x1x3
    U_233[4:6, 11:17] = U_213
    # V_233 rows: B20, B21, B22. Indices 6,7,8.
    V_233[6:9, 11:17] = V_213
    # W_233 rows: C00, C01, C02, C10, C11, C12.
    W_233[:, 11:17] = W_213
    
    # --- Build 2x5x3 (Rank 28) ---
    # A (2x5), B (5x3), C (2x3)
    # A = [A_22 (2x2) | A_23 (2x3)]
    # B = [B_23 (2x3); B_33 (3x3)]
    # C = A_22 B_23 + A_23 B_33
    # Part 1: 2x2x3 (Rank 11)
    # Part 2: 2x3x3 (Rank 17)
    
    U_253 = np.zeros((10, 28))
    V_253 = np.zeros((15, 28))
    W_253 = np.zeros((6, 28))
    
    # Part 1: 2x2x3
    U_253[0:4, 0:11] = U_223
    # V_253 rows: B00, B01, B02, B10, B11, B12, B20, B21, B22, B30, B31, B32, B40, B41, B42
    # B_23 is B00, B01, B02, B10, B11, B12. Indices 0-5.
    V_253[0:6, 0:11] = V_223
    # W_253 rows: C00, C01, C02, C10, C11, C12
    W_253[:, 0:11] = W_223
    
    # Part 2: 2x3x3
    U_253[4:10, 11:28] = U_233
    # V_253 rows: B20, B21, B22, B30, B31, B32, B40, B41, B42. Indices 6-14.
    V_253[6:15, 11:28] = V_233
    # W_253 rows: C00, C01, C02, C10, C11, C12
    W_253[:, 11:28] = W_233
    
    # --- Build 2x5x6 (Rank 56) ---
    # A (2x5), B (5x6), C (2x6)
    # B = [B_L (5x3) | B_R (5x3)]
    # C = [C_L (2x3) | C_R (2x3)]
    # C_L = A B_L (Rank 28)
    # C_R = A B_R (Rank 28)
    
    U_256 = np.zeros((10, 56))
    V_256 = np.zeros((30, 56))
    W_256 = np.zeros((12, 56))
    
    # Part 1: 2x5x3 (Left)
    U_256[:, 0:28] = U_253
    # V_256 rows: B00..B44 (30 rows).
    # B_L is B00..B42 (first 3 cols). Indices 0-14.
    V_256[0:15, 0:28] = V_253
    # W_256 rows: C00..C15 (12 rows).
    # C_L is C00, C01, C02, C10, C11, C12.
    # W index k*n+i. n=2.
    # C_L cols k=0,1,2.
    # k=0: C00 (i=0), C01 (i=1) -> W[0], W[1]
    # k=1: C10 (i=0), C11 (i=1) -> W[2], W[3]
    # k=2: C20 (i=0), C21 (i=1) -> W[4], W[5]
    # Wait, C is 2x3. Rows i=0,1. Cols k=0,1,2.
    # W_256 indices for C_L:
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # k=2, i=0 -> 4
    # k=2, i=1 -> 5
    W_256[0:6, 0:28] = W_253
    
    # Part 2: 2x5x3 (Right)
    U_256[:, 28:56] = U_253
    # V_256 rows: B_R is B03..B45 (last 3 cols). Indices 15-29.
    V_256[15:30, 28:56] = V_253
    # W_256 rows: C_R is C03, C04, C05, C13, C14, C15.
    # k=3: C03 (i=0), C04 (i=1) -> W[3*2+0]=6, W[3*2+1]=7
    # k=4: C13 (i=0), C14 (i=1) -> W[4*2+0]=8, W[4*2+1]=9
    # k=5: C23 (i=0), C24 (i=1) -> W[5*2+0]=10, W[5*2+1]=11
    W_256[6:12, 28:56] = W_253
    
    # --- Build 4x5x6 (Rank 112) ---
    # A (4x5), B (5x6), C (4x6)
    # A = [A_Top (2x5); A_Bot (2x5)]
    # C = [C_Top (2x6); C_Bot (2x6)]
    # C_Top = A_Top B (Rank 56)
    # C_Bot = A_Bot B (Rank 56)
    
    offset = 0
    # Top part
    u_idx_top = [(i, j) for i in range(2) for j in range(5)]
    v_idx = [(j, k) for j in range(5) for k in range(6)]
    w_idx_top = [(k, i) for k in range(6) for i in range(2)] # i=0,1
    
    offset = add_block(u_idx_top, v_idx, w_idx_top, U_256, V_256, W_256, offset)
    
    # Bottom part
    u_idx_bot = [(i, j) for i in range(2, 4) for j in range(5)]
    # V is same
    # W indices for bottom: i=2,3.
    # W[k*n+i] -> k*4+i.
    # k=0..5, i=2,3.
    w_idx_bot = [(k, i) for k in range(6) for i in range(2, 4)]
    
    offset = add_block(u_idx_bot, v_idx, w_idx_bot, U_256, V_256, W_256, offset)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
