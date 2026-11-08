import numpy as np
import time

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 8
    rank = 56
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2 matrices (for the block level)
    # These are the coefficients for the 7 Strassen terms
    # U_S, V_S, W_S are 4x7
    U_S = np.zeros((4, 7))
    V_S = np.zeros((4, 7))
    W_S = np.zeros((4, 7))
    
    # Standard Strassen
    U_S[0, 0] = 1; U_S[3, 0] = 1
    V_S[0, 0] = 1; V_S[3, 0] = 1
    W_S[0, 0] = 1; W_S[3, 0] = 1
    
    U_S[2, 1] = 1; U_S[3, 1] = 1
    V_S[0, 1] = 1
    W_S[2, 1] = 1; W_S[3, 1] = -1
    
    U_S[0, 2] = 1
    V_S[2, 2] = 1; V_S[0, 2] = -1
    W_S[1, 2] = 1; W_S[3, 2] = 1
    
    U_S[3, 3] = 1
    V_S[1, 3] = 1; V_S[3, 3] = -1
    W_S[0, 3] = 1; W_S[2, 3] = 1
    
    U_S[0, 4] = 1; U_S[1, 4] = 1
    V_S[3, 4] = 1
    W_S[0, 4] = -1; W_S[1, 4] = 1
    
    U_S[2, 5] = 1; U_S[0, 5] = -1
    V_S[0, 5] = 1; V_S[1, 5] = 1
    W_S[3, 5] = 1
    
    U_S[1, 6] = 1; U_S[3, 6] = -1
    V_S[2, 6] = 1; V_S[3, 6] = 1
    W_S[0, 6] = 1
    
    # Trivial 2x2x2 matrices (for the element level)
    # Rank 4
    # U_T, V_T, W_T are 4x4
    U_T = np.zeros((4, 4))
    V_T = np.zeros((4, 4))
    W_T = np.zeros((4, 4))
    
    for j in range(2):
        for i in range(2):
            for k in range(2):
                t = j * 4 + i * 2 + k # Just a unique index 0..3? No, rank is 4.
                # Trivial terms for 2x2:
                # j=0: i=0,k=0; i=0,k=1; i=1,k=0; i=1,k=1
                # j=1: i=0,k=0; ...
                # Wait, rank of 2x2x2 trivial is 4?
                # No, trivial rank of 2x2x2 is 4?
                # C_00 = A_00 B_00 + A_01 B_10
                # C_01 = A_00 B_01 + A_01 B_11
                # C_10 = A_10 B_00 + A_11 B_10
                # C_11 = A_10 B_01 + A_11 B_11
                # Terms:
                # 1: A_00 B_00 -> C_00
                # 2: A_00 B_01 -> C_01
                # 3: A_10 B_00 -> C_10
                # 4: A_10 B_01 -> C_11
                # 5: A_01 B_10 -> C_00
                # 6: A_01 B_11 -> C_01
                # 7: A_11 B_10 -> C_10
                # 8: A_11 B_11 -> C_11
                # Total 8 terms.
                # So trivial rank is 8?
                # But Strassen is 7.
                # So trivial is 8.
                # My previous calculation of 28 for 4x4x4 was based on 7 * 4 = 28.
                # But if trivial 2x2 is 8, then 7 * 8 = 56.
                # And 56 for one 4x4x4.
                # Total 112.
                # This is worse than 98.
                
                # Let's re-evaluate trivial rank of 2x2x2.
                # Is it 4?
                # C = A B.
                # If we use the identity C = A B, the rank is the number of scalar multiplications.
                # For 2x2, it is 8.
                # So rank 8.
                # Then 4x4x4 using Strassen (7) on 2x2 blocks, and Trivial (8) on 2x2 matrices.
                # Rank = 7 * 8 = 56.
                # Total for 4x4x8 = 56 + 56 = 112.
                # Worse than 98.

                # What if we use Strassen for 2x2 matrices too?
                # Rank = 7 * 7 = 49.
                # Total 98.
                # This matches the current program.

                # Is there a better way?
                # Maybe use a rank 24 decomposition for 4x4x4?
                # If rank 24 exists, total 48.
                # But I don't have the factors.

                # Let's stick to the current program's approach but see if we can optimize.
                # The current program uses rank 98.
                # It uses Strassen (7) for 4x4x4 blocks.
                # It has two 4x4x4 blocks.
                # So 2 * 49 = 98.
                
                # Can we share factors between the two blocks?
                # A is the same.
                # B is different.
                # C is different.
                # The Strassen factors for A are the same for both blocks?
                # In the current program, U is constructed for both blocks.
                # U has shape (16, 98).
                # Columns 0-48 are for block 1.
                # Columns 49-97 are for block 2.
                # The U values for block 1 and block 2 are identical?
                # Yes, because A is the same.
                # So we can just copy the first 49 columns to the next 49.
                # This doesn't reduce rank, but it's efficient.

                # Is there a rank < 98?
                # Maybe 96?
                # Or use a different algorithm for 4x4x4?
                # Best known for 4x4x4 is 24? No, that's 3x3.
                # For 4x4x4, the best known is 24?
                # Actually, the "Strassen" bound for 4x4x4 is 49.
                # But there are algorithms with rank 48?
                # Or 46?
                # I don't have the factors.

                # Let's try to verify if the current program is correct.
                # It seems to be a standard construction.

                # I will output the current program as is, or make minor optimizations.
                # Since I cannot find a better rank easily, I will keep the rank 98.
                # But I can clean up the code.

                pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
