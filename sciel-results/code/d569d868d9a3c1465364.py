"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import scipy
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=7
    
    # Strategy:
    # Split the middle dimension m=4 into two blocks of size 2.
    # A is 2x4, split into A1 (2x2) and A2 (2x2).
    # B is 4x7, split into B1 (2x7) and B2 (2x7).
    # C = A1 B1 + A2 B2.
    # Each term is a 2x2x7 multiplication.
    # Rank of 2x2x7 multiplication:
    #   Split p=7 into 2+2+3.
    #   2x2x2 part: Strassen rank 7.
    #   2x2x2 part: Strassen rank 7.
    #   2x2x3 part: Lower bound is max(4, 6, 6) = 6, but actual rank is 10 (Bürgisser).
    #   Total rank for 2x2x7 is 7 + 7 + 10 = 24.
    # Total rank for 2x4x7 is 2 * 24 = 48.
    # This is better than schoolbook (56).
    #
    # However, we can do better if we mix the blocks?
    # No, the blocks are independent in the standard decomposition.
    # Let's stick to the constructive 48 rank first, as it is guaranteed.
    #
    # Wait, is there a better decomposition for 2x2x3?
    # Yes, rank 10 is the known minimum.
    #
    # Let's construct the 2x2x2 Strassen decomposition.
    # U, V, W are 4x7 matrices for the 2x2x2 case.
    # Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # Strassen terms for 2x2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's define the Strassen matrices for 2x2x2.
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    # U rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    def get_strassen_222():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Term 0: M1
        # U: A11 + A22 -> U[0,0]=1, U[3,0]=1
        U[0, 0] = 1
        U[3, 0] = 1
        # V: B11 + B22 -> V[0,0]=1, V[3,0]=1
        V[0, 0] = 1
        V[3, 0] = 1
        # W: C11 += 1, C22 += 1 -> W[0,0]=1, W[3,0]=1
        W[0, 0] = 1
        W[3, 0] = 1
        
        # Term 1: M2
        # U: A21 + A22 -> U[2,1]=1, U[3,1]=1
        U[2, 1] = 1
        U[3, 1] = 1
        # V: B11 -> V[0,1]=1
        V[0, 1] = 1
        # W: C21 += 1 -> W[2,1]=1
        W[2, 1] = 1
        
        # Term 2: M3
        # U: A11 -> U[0,2]=1
        U[0, 2] = 1
        # V: B21 - B22 -> V[2,2]=1, V[3,2]=-1
        V[2, 2] = 1
        V[3, 2] = -1
        # W: C12 += 1 -> W[1,2]=1
        W[1, 2] = 1
        
        # Term 3: M4
        # U: A22 -> U[3,3]=1
        U[3, 3] = 1
        # V: B12 - B11 -> V[1,3]=1, V[0,3]=-1
        V[1, 3] = 1
        V[0, 3] = -1
        # W: C11 += 1, C21 += 1 -> W[0,3]=1, W[2,3]=1
        W[0, 3] = 1
        W[2, 3] = 1
        
        # Term 4: M5
        # U: A11 + A12 -> U[0,4]=1, U[1,4]=1
        U[0, 4] = 1
        U[1, 4] = 1
        # V: B22 -> V[3,4]=1
        V[3, 4] = 1
        # W: C11 -= 1, C12 += 1 -> W[0,4]=-1, W[1,4]=1
        W[0, 4] = -1
        W[1, 4] = 1
        
        # Term 5: M6
        # U: A21 - A11 -> U[2,5]=1, U[0,5]=-1
        U[2, 5] = 1
        U[0, 5] = -1
        # V: B11 + B12 -> V[0,5]=1, V[1,5]=1
        V[0, 5] = 1
        V[1, 5] = 1
        # W: C22 += 1 -> W[3,5]=1
        W[3, 5] = 1
        
        # Term 6: M7
        # U: A12 - A22 -> U[1,6]=1, U[3,6]=-1
        U[1, 6] = 1
        U[3, 6] = -1
        # V: B21 + B22 -> V[2,6]=1, V[3,6]=1
        V[2, 6] = 1
        V[3, 6] = 1
        # W: C11 += 1 -> W[0,6]=1
        W[0, 6] = 1
        
        return U, V, W
    
    # Now we need a rank-10 decomposition for 2x2x3.
    # This is harder to construct manually. 
    # However, we can use the fact that 2x2x3 is equivalent to 2x2x2 + 2x2x1.
    # Rank(2x2x1) is 2 (since C1 = A1 B1 + A2 B2, two terms).
    # So rank(2x2x3) <= rank(2x2x2) + rank(2x2x1) = 7 + 2 = 9? 
    # No, the lower bound is 10. So 9 is impossible.
    # Actually, rank(2x2x3) is known to be 10.
    # Can we construct it?
    # A 2x2x3 multiplication can be seen as:
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C13 = A11 B13 + A12 B23
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    # C23 = A21 B13 + A22 B23
    #
    # This is just schoolbook rank 6? No, schoolbook for 2x2x3 is 2*2*3 = 12.
    # Wait, schoolbook is sum_j A_ij B_jk.
    # For 2x2x3, j goes 0..1, k goes 0..2.
    # C_ik = A_i0 B_0k + A_i1 B_1k.
    # Terms: 
    # (i=0,k=0): A00 B00, A01 B10
    # (i=0,k=1): A00 B01, A01 B11
    # (i=0,k=2): A00 B02, A01 B12
    # (i=1,k=0): A10 B00, A11 B10
    # (i=1,k=1): A10 B01, A11 B11
    # (i=1,k=2): A10 B02, A11 B12
    # Total 12 terms.
    #
    # Is there a rank 10 decomposition? Yes.
    # But constructing it is non-trivial.
    # 
    # Alternative: Use the 2x2x2 Strassen for the first 2 columns, and 2x2x1 for the last column.
    # Rank = 7 + 2 = 9. But this is impossible.
    # Why? Because the 2x2x1 part shares the A matrix with the 2x2x2 part.
    # The "addition" of tensors is not simply additive in rank if they share dimensions.
    # Actually, the rank of the direct sum of tensors is the sum of ranks.
    # 2x2x3 is NOT the direct sum of 2x2x2 and 2x2x1.
    # It is a "slice" or "fiber" product.
    #
    # Let's assume we can get rank 10 for 2x2x3.
    # Total rank for 2x4x7 would be 2 * (7 + 10) = 34.
    #
    # If I cannot easily construct the rank-10 2x2x3 decomposition, I can fall back to schoolbook for that part.
    # Schoolbook 2x2x3 rank is 12.
    # Total rank = 2 * (7 + 12) = 38.
    # This is still much better than 56.
    #
    # Let's implement the 2x2x2 Strassen and 2x2x3 Schoolbook.
    # Total rank 38.
    
    def get_schoolbook_223():
        # 2x2x3
        # U (4, 12), V (6, 12), W (6, 12)
        # U rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # V rows: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
        # W rows: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
        
        U = np.zeros((4, 12))
        V = np.zeros((6, 12))
        W = np.zeros((6, 12))
        
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[i*2+j, col] = 1
                    V[j*3+k, col] = 1
                    W[k*2+i, col] = 1
                    col += 1
        return U, V, W
    
    # Construct the full 2x4x7 decomposition
    # Total rank = 2 * (7 + 12) = 38
    
    rank_222 = 7
    rank_223 = 12
    total_rank = 2 * (rank_222 + rank_223)
    
    U = np.zeros((n*m, total_rank)) # 8 x 38
    V = np.zeros((m*p, total_rank)) # 28 x 38
    W = np.zeros((p*n, total_rank)) # 14 x 38
    
    # We have two blocks: Block 1 (j=0,1) and Block 2 (j=2,3)
    # Block 1: A1 (2x2), B1 (2x7)
    # Block 2: A2 (2x2), B2 (2x7)
    
    # Indices for Block 1:
    # U rows: 0..3 (i=0,1; j=0,1)
    # V rows: 0..5 (j=0,1; k=0..6) -> 2*7 = 14 rows? No.
    # V is 4x7. Rows are (j,k).
    # For Block 1, j=0,1. So V rows 0..13?
    # V index = j*p + k.
    # j=0: 0..6
    # j=1: 7..13
    # So V rows 0..13 for Block 1.
    # j=2: 14..20
    # j=3: 21..27
    # So V rows 14..27 for Block 2.
    
    # W is 7x2. Rows are (k,i).
    # W index = k*n + i.
    # This is shared by both blocks.
    
    # Let's fill U, V, W.
    
    # Block 1 (j=0,1)
    # Split k into 0,1 (2x2x2) and 2,3,4,5,6 (2x2x5)?
    # No, I said 2x2x2 and 2x2x3.
    # k=0,1 for Strassen. k=2,3,4 for Schoolbook?
    # Wait, p=7.
    # k=0,1 -> 2x2x2.
    # k=2,3,4,5,6 -> 2x2x5.
    # Rank(2x2x5) = Rank(2x2x2) + Rank(2x2x3) = 7 + 10 = 17?
    # Or Schoolbook 2x2x5 = 20.
    # If I use Schoolbook for 2x2x5, rank is 20.
    # Total for Block 1: 7 (for k=0,1) + 20 (for k=2..6) = 27.
    # Total for 2 blocks: 54.
    # This is barely better than 56.
    
    # Let's try to optimize the 2x2x5 part.
    # 2x2x5 = 2x2x2 + 2x2x3.
    # If I use Strassen for 2x2x2 (rank 7) and Schoolbook for 2x2x3 (rank 12), total 19.
    # Total for Block 1: 7 (k=0,1) + 19 (k=2..6) = 26.
    # Total for 2 blocks: 52.
    
    # Can I do better?
    # What if I use Strassen for the whole 2x2x7?
    # No, Strassen is for square matrices.
    
    # Let's try a different split.
    # Split m=4 into 1+3?
    # 2x1x7 and 2x3x7.
    # Rank(2x1x7) = 2*1*7 = 14 (Schoolbook).
    # Rank(2x3x7) = ?
    # This seems worse.
    
    # Let's stick to the 52 rank solution for now.
    # It is a significant improvement over 56.
    
    # Re-evaluating the 2x2x5 part.
    # Is there a rank 17 decomposition for 2x2x5?
    # Lower bound is max(4, 10, 10) = 10.
    # Known ranks: 2x2x2=7, 2x2x3=10, 2x2x4=14 (Strassen 7+7), 2x2x5=?
    # If 2x2x4 is 14, then 2x2x5 is at least 14.
    # If we use Strassen for 2x2x4 (rank 14) and Schoolbook for 2x2x1 (rank 2), total 16.
    # So 2x2x5 rank <= 16.
    # Total for Block 1: 7 (k=0,1) + 16 (k=2..6) = 23.
    # Total for 2 blocks: 46.
    
    # Let's implement 2x2x4 Strassen (two 2x2x2 Strassen).
    # And 2x2x1 Schoolbook.
    
    def get_strassen_224():
        # 2x2x4
        # Two 2x2x2 blocks.
        # U (4, 14), V (8, 14), W (8, 14)
        # U rows: 0..3
        # V rows: 0..7 (j=0,1; k=0..3)
        # W rows: 0..7 (k=0..3; i=0,1)
        
        U1, V1, W1 = get_strassen_222()
        U2, V2, W2 = get_strassen_222()
        
        U = np.zeros((4, 14))
        V = np.zeros((8, 14))
        W = np.zeros((8, 14))
        
        # Block 1 (k=0,1)
        U[:, 0:7] = U1
        V[0:4, 0:7] = V1
        W[0:4, 0:7] = W1
        
        # Block 2 (k=2,3)
        U[:, 7:14] = U2
        # V rows for k=2,3 are 4..7
        V[4:8, 7:14] = V2
        # W rows for k=2,3 are 4..7
        W[4:8, 7:14] = W2
        
        return U, V, W
    
    def get_schoolbook_221():
        # 2x2x1
        # U (4, 2), V (2, 2), W (2, 2)
        U = np.zeros((4, 2))
        V = np.zeros((2, 2))
        W = np.zeros((2, 2))
        
        # Term 0: A11 B11
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        
        # Term 1: A21 B21? No, A is 2x2, B is 2x1.
        # C1 = A10 B0 + A11 B1
        # C2 = A20 B0 + A21 B1
        
        # Term 0: A10 B0
        U[0, 0] = 1 # A10
        V[0, 0] = 1 # B0
        W[0, 0] = 1 # C1
        
        # Term 1: A11 B1
        U[1, 1] = 1 # A11
        V[1, 1] = 1 # B1
        W[0, 1] = 1 # C1
        
        # Term 2: A20 B0
        U[2, 2] = 1 # A20
        V[0, 2] = 1 # B0
        W[1, 2] = 1 # C2
        
        # Term 3: A21 B1
        U[3, 3] = 1 # A21
        V[1, 3] = 1 # B1
        W[1, 3] = 1 # C2
        
        # Wait, rank is 4?
        # C1 = A10 B0 + A11 B1
        # C2 = A20 B0 + A21 B1
        # This is 4 terms.
        # Can we do it in 2?
        # No, lower bound is max(2, 2, 2) = 2.
        # But 2x2x1 is equivalent to two 1x2x1 multiplications?
        # No.
        # Actually, 2x2x1 is just two independent 1x2x1 multiplications?
        # C1 = A10 B0 + A11 B1
        # C2 = A20 B0 + A21 B1
        # These are independent.
        # Rank is 4.
        
        return U, V, W
    
    # So 2x2x5 rank is 14 + 4 = 18.
    # Total for Block 1: 7 + 18 = 25.
    # Total for 2 blocks: 50.
    
    # This matches the inspiration 1 score (1.12 -> rank 50).
    # Let's implement this rank 50 solution.
    
    rank_222 = 7
    rank_224 = 14
    rank_221 = 4
    rank_225 = rank_224 + rank_221 # 18
    
    # Block 1: k=0,1 (222), k=2,3 (222), k=4,5,6 (223?)
    # Wait, I used 224 and 221 for 225.
    # k=0,1: 222 (Rank 7)
    # k=2,3: 222 (Rank 7)
    # k=4,5,6: 223 (Rank 10?)
    # If I use Schoolbook for 223 (Rank 12), total 7+7+12 = 26.
    # Total 52.
    
    # If I use the 224+221 split for k=2..6:
    # k=2,3: 222 (Rank 7)
    # k=4,5: 222 (Rank 7)
    # k=6: 221 (Rank 4)
    # Total 7+7+7+4 = 25.
    # Total 50.
    
    # Let's implement the 50 rank solution.
    
    total_rank = 50
    
    U = np.zeros((n*m, total_rank)) # 8 x 50
    V = np.zeros((m*p, total_rank)) # 28 x 50
    W = np.zeros((p*n, total_rank)) # 14 x 50
    
    # Block 1 (j=0,1)
    # U rows: 0..3
    # V rows: 0..13
    # W rows: 0..13
    
    # Sub-block 1: k=0,1 (222)
    U1, V1, W1 = get_strassen_222()
    U[0:4, 0:7] = U1
    V[0:4, 0:7] = V1
    W[0:4, 0:7] = W1
    
    # Sub-block 2: k=2,3 (222)
    U2, V2, W2 = get_strassen_222()
    U[0:4, 7:14] = U2
    V[4:8, 7:14] = V2
    W[4:8, 7:14] = W2
    
    # Sub-block 3: k=4,5 (222)
    U3, V3, W3 = get_strassen_222()
    U[0:4, 14:21] = U3
    V[8:12, 14:21] = V3
    W[8:12, 14:21] = W3
    
    # Sub-block 4: k=6 (221)
    U4, V4, W4 = get_schoolbook_221()
    U[0:4, 21:25] = U4
    V[12:14, 21:25] = V4
    W[12:14, 21:25] = W4
    
    # Block 2 (j=2,3)
    # U rows: 4..7
    # V rows: 14..27
    # W rows: 0..13 (Shared W? No, W is output. Output is same for both blocks?
    # C = A1 B1 + A2 B2.
    # C is 2x7.
    # W is 7x2.
    # W rows are (k,i).
    # Both blocks contribute to the same C.
    # So W rows are shared.
    # But the columns of W are different for each block.
    # Block 1 uses columns 0..24 of W.
    # Block 2 will use columns 25..49 of W.
    
    # Sub-block 5: k=0,1 (222)
    U5, V5, W5 = get_strassen_222()
    U[4:8, 25:32] = U5
    V[14:18, 25:32] = V5
    W[0:4, 25:32] = W5
    
    # Sub-block 6: k=2,3 (222)
    U6, V6, W6 = get_strassen_222()
    U[4:8, 32:39] = U6
    V[18:22, 32:39] = V6
    W[4:8, 32:39] = W6
    
    # Sub-block 7: k=4,5 (222)
    U7, V7, W7 = get_strassen_222()
    U[4:8, 39:46] = U7
    V[22:26, 39:46] = V7
    W[8:12, 39:46] = W7
    
    # Sub-block 8: k=6 (221)
    U8, V8, W8 = get_schoolbook_221()
    U[4:8, 46:50] = U8
    V[26:28, 46:50] = V8
    W[12:14, 46:50] = W8
    
    # Final check
    # U shape: 8x50
    # V shape: 28x50
    # W shape: 14x50
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
