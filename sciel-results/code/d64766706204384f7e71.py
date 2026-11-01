"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    # Helper to verify decomposition
    def verify_decomposition(U, V, W, n=n, m=m, p=p):
        R = U.shape[1]
        # Tensor T[i,j,k] = 1 for all i,j,k
        # Check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(R):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Helper to build schoolbook decomposition for M_{n,m,p}
    def build_schoolbook(n, m, p):
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, R

    # Helper to build Strassen decomposition for M_{2,2,2} (Rank 7)
    # Standard Strassen formulas for 2x2 * 2x2
    def build_strassen_2x2x2():
        n, m, p = 2, 2, 2
        R = 7
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # Indices: A[i,j] -> U[i*m+j], B[j,k] -> V[j*p+k], C[i,k] -> W[k*n+i]
        # Strassen products:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # Term 0: M1
        # A00=1, A11=1; B00=1, B11=1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # Term 1: M2
        # A10=1, A11=1; B00=1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # Term 2: M3
        # A00=1; B01=1, B11=-1
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # Term 3: M4
        # A11=1; B10=1, B00=-1
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # Term 4: M5
        # A00=1, A01=1; B11=1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # Term 5: M6
        # A10=1, A00=-1; B00=1, B01=1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1
        
        # Term 6: M7
        # A01=1, A11=-1; B10=1, B11=1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[1, 6] = -1; W[3, 6] = 1
        
        return U, V, W, R

    # Helper to build a rank-4 decomposition for M_{2,2,1} (Schoolbook is 4)
    def build_m221():
        return build_schoolbook(2, 2, 1)

    # Helper to build a rank-6 decomposition for M_{2,2,2}? No, Strassen is 7.
    # Is there a rank 6 for 2x2x2? No, it's 7.
    # What about M_{2,2,3}? Lower bound max(4, 6, 6) = 6.
    # Can we do M_{2,2,3} in rank 6?
    # Actually, R(M_{2,2,3}) is known to be 6? Let's check.
    # C = A B. A is 2x2, B is 2x3.
    # C0k = A00 B0k + A01 B1k
    # C1k = A10 B0k + A11 B1k
    # This is 6 bilinear forms. Can they be done in 6 terms?
    # If yes, we can decompose 2x2x15 into five 2x2x3 blocks?
    # 5 * 6 = 30. Schoolbook is 60.
    # Let's try to construct a rank-6 decomposition for 2x2x3.
    
    def build_m223_rank6():
        n, m, p = 2, 2, 3
        R = 6
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # We need to satisfy:
        # Sum_c U[0,c]V[0,c]W[0,c] = 1 (C00)
        # Sum_c U[0,c]V[1,c]W[1,c] = 1 (C01)
        # Sum_c U[0,c]V[2,c]W[2,c] = 1 (C02)
        # Sum_c U[1,c]V[0,c]W[0,c] = 1 (C01? No, C01 is i=0,k=1)
        # Let's map indices carefully.
        # U index: i*m+j. i in 0,1. j in 0,1.
        # V index: j*p+k. j in 0,1. k in 0,1,2.
        # W index: k*n+i. k in 0,1,2. i in 0,1.
        
        # Terms:
        # We want to cover 6 entries in W: (0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2)?
        # Wait, W has size p*n = 3*2 = 6.
        # W indices:
        # k=0: W[0] (i=0), W[1] (i=1)
        # k=1: W[2] (i=0), W[3] (i=1)
        # k=2: W[4] (i=0), W[5] (i=1)
        
        # Equations:
        # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] -> U[0]V[0]W[0] + U[1]V[3]W[0] = 1
        # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] -> U[0]V[1]W[2] + U[1]V[4]W[2] = 1
        # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] -> U[0]V[2]W[4] + U[1]V[5]W[4] = 1
        # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0] -> U[2]V[0]W[1] + U[3]V[3]W[1] = 1
        # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1] -> U[2]V[1]W[3] + U[3]V[4]W[3] = 1
        # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] -> U[2]V[2]W[5] + U[3]V[5]W[5] = 1
        
        # Let's try a specific assignment.
        # Term 0: Contributes to C[0,0] via A00 B00.
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # Term 1: Contributes to C[0,0] via A01 B10.
        U[1, 1] = 1; V[3, 1] = 1; W[0, 1] = 1
        # Term 2: Contributes to C[0,1] via A00 B01.
        U[0, 2] = 1; V[1, 2] = 1; W[2, 2] = 1
        # Term 3: Contributes to C[0,1] via A01 B11.
        U[1, 3] = 1; V[4, 3] = 1; W[2, 3] = 1
        # Term 4: Contributes to C[0,2] via A00 B02.
        U[0, 4] = 1; V[2, 4] = 1; W[4, 4] = 1
        # Term 5: Contributes to C[0,2] via A01 B12.
        U[1, 5] = 1; V[5, 5] = 1; W[4, 5] = 1
        
        # Now check C[1,*].
        # C[1,0] needs U[2]V[0]W[1] + U[3]V[3]W[1] = 1.
        # Currently W[1] is 0. We need to add terms or modify.
        # The above 6 terms only cover C[0,*].
        # This simple assignment fails because we used all 6 ranks for row 0.
        # We need to share terms between row 0 and row 1.
        
        # Let's try a different structure.
        # Can we use Strassen on 2x2x2 and add 1 column?
        # Strassen for 2x2x2 (cols 0,1) uses rank 7.
        # Adding col 2 schoolbook uses rank 4. Total 11.
        # Is there a rank 6?
        # Lower bound is 6. It is an open question if R(M223)=6 or 7?
        # Actually, R(M223) is known to be 6? No, I think it's 6 for 2x2x2? No, 7.
        # For 2x2x3, R is 6?
        # Let's check the literature mentally.
        # R(M_{2,2,2}) = 7.
        # R(M_{2,2,3}) <= 7+4 = 11.
        # Can we do better?
        # If we assume R(M223) is not 6, then we can't use 5 blocks of 3.
        
        # Let's stick to known safe decompositions.
        # 2x2x2: Rank 7 (Strassen)
        # 2x2x1: Rank 4 (Schoolbook)
        
        # Decompose 2x2x15 into:
        # 7 blocks of 2x2x2 (using 14 columns) + 1 block of 2x2x1 (using 1 column).
        # Total rank = 7 * 7 + 4 = 49 + 4 = 53.
        
        # Is there a better split?
        # 2x2x4: Can we do better than 2*7=14?
        # R(M224) <= 2*R(M222) = 14.
        # Is R(M224) < 14?
        # Lower bound max(4, 8, 8) = 8.
        # Known: R(M224) is 14? Or 13?
        # Actually, R(M_{2,2,4}) is known to be 14?
        # Wait, R(M_{2,2,4}) can be computed using tensor powers.
        # M_{2,2,2}^2 has rank <= 49.
        # But M_{2,2,4} is not a tensor power.
        
        # Let's try 2x2x3 again. If R(M223)=6, then 5*6=30.
        # If R(M223)=7, then 5*7=35.
        # If R(M223)=8, then 5*8=40.
        
        # Let's try to find a rank-6 decomposition for 2x2x3 programmatically?
        # No, time is limited.
        
        # Let's use the 7*7+4 = 53 strategy first, as it is guaranteed.
        # Then try to optimize the 2x2x2 blocks or find better splits.
        
        # Strategy:
        # 1. Build 7 blocks of Strassen (2x2x2) for first 14 columns.
        # 2. Build 1 block of Schoolbook (2x2x1) for last 1 column.
        # Total Rank 53.
        
        # Can we do 2x2x3 in rank 6?
        # Let's try a specific rank-6 construction for 2x2x3.
        # If it works, 5 blocks of 3 -> Rank 30.
        
        # Attempt Rank-6 for 2x2x3:
        # We need 6 terms.
        # Let's use a solver? No, too slow.
        # Let's use a known identity if exists.
        # Actually, R(M_{2,2,3}) is 6?
        # According to some sources, R(M_{2,2,3}) = 6.
        # Let's try to construct it.
        
        # Equations for 2x2x3:
        # C00 = A00B00 + A01B10
        # C01 = A00B01 + A01B11
        # C02 = A00B02 + A01B12
        # C10 = A10B00 + A11B10
        # C11 = A10B01 + A11B11
        # C12 = A10B02 + A11B12
        
        # Try:
        # T1: A00 B00 -> C00 (U0=1, V0=1, W0=1)
        # T2: A01 B10 -> C00 (U1=1, V3=1, W0=1)
        # T3: A00 B01 -> C01 (U0=1, V1=1, W2=1)
        # T4: A01 B11 -> C01 (U1=1, V4=1, W2=1)
        # T5: A00 B02 -> C02 (U0=1, V2=1, W4=1)
        # T6: A01 B12 -> C02 (U1=1, V5=1, W4=1)
        
        # This covers C0*.
        # What about C1*?
        # C10 = A10B00 + A11B10.
        # We need terms involving U2, U3.
        # The above terms have U2=U3=0.
        # So this simple assignment doesn't work.
        
        # We need to mix.
        # Let's try:
        # T1: (A00+A10) B00 -> C00 + C10
        #    U0=1, U2=1. V0=1. W0=1, W1=1.
        #    Contribution: A00B00 to C00, A10B00 to C10.
        # T2: (A01+A11) B10 -> C00 + C10
        #    U1=1, U3=1. V3=1. W0=1, W1=1.
        #    Contribution: A01B10 to C00, A11B10 to C10.
        # Now C00 and C10 are done.
        
        # T3: (A00+A10) B01 -> C01 + C11
        #    U0=1, U2=1. V1=1. W2=1, W3=1.
        # T4: (A01+A11) B11 -> C01 + C11
        #    U1=1, U3=1. V4=1. W2=1, W3=1.
        # Now C01 and C11 are done.
        
        # T5: (A00+A10) B02 -> C02 + C12
        #    U0=1, U2=1. V2=1. W4=1, W5=1.
        # T6: (A01+A11) B12 -> C02 + C12
        #    U1=1, U3=1. V5=1. W4=1, W5=1.
        # Now C02 and C12 are done.
        
        # Let's verify this Rank-6 construction for 2x2x3.
        U = np.zeros((4, 6))
        V = np.zeros((6, 6))
        W = np.zeros((6, 6))
        
        # T1
        U[0, 0] = 1; U[2, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1; W[1, 0] = 1
        
        # T2
        U[1, 1] = 1; U[3, 1] = 1
        V[3, 1] = 1
        W[0, 1] = 1; W[1, 1] = 1
        
        # T3
        U[0, 2] = 1; U[2, 2] = 1
        V[1, 2] = 1
        W[2, 2] = 1; W[3, 2] = 1
        
        # T4
        U[1, 3] = 1; U[3, 3] = 1
        V[4, 3] = 1
        W[2, 3] = 1; W[3, 3] = 1
        
        # T5
        U[0, 4] = 1; U[2, 4] = 1
        V[2, 4] = 1
        W[4, 4] = 1; W[5, 4] = 1
        
        # T6
        U[1, 5] = 1; U[3, 5] = 1
        V[5, 5] = 1
        W[4, 5] = 1; W[5, 5] = 1
        
        if verify_decomposition(U, V, W, 2, 2, 3):
            # Great! Rank 6 for 2x2x3.
            # We can use 5 blocks of 2x2x3 to cover 15 columns.
            # Total Rank = 5 * 6 = 30.
            
            # Construct full 2x2x15 using 5 blocks of this rank-6 decomposition.
            U_full = np.zeros((4, 30))
            V_full = np.zeros((30, 30)) # m*p = 2*15 = 30
            W_full = np.zeros((30, 30)) # p*n = 15*2 = 30
            
            for block in range(5):
                offset = block * 6
                k_offset = block * 3
                
                # Copy U
                U_full[:, offset:offset+6] = U
                
                # Copy V (adjust k indices)
                # V indices are j*p+k. In block, k is 0,1,2.
                # In full, k is k_offset..k_offset+2.
                for j in range(2):
                    for k_local in range(3):
                        k_global = k_offset + k_local
                        V_full[j*15 + k_global, offset:offset+6] = V[j*3 + k_local, :]
                
                # Copy W (adjust k indices)
                # W indices are k*n+i. In block, k is 0,1,2.
                # In full, k is k_offset..k_offset+2.
                for k_local in range(3):
                    k_global = k_offset + k_local
                    for i in range(2):
                        W_full[k_global*2 + i, offset:offset+6] = W[k_local*2 + i, :]
                        
            if verify_decomposition(U_full, V_full, W_full, 2, 2, 15):
                return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
            
        # Fallback: 7 blocks of Strassen (2x2x2) + 1 Schoolbook (2x2x1)
        # Rank = 7*7 + 4 = 53
        
        U_blocks = []
        V_blocks = []
        W_blocks = []
        
        # 7 blocks of 2x2x2
        for b in range(7):
            Ub, Vb, Wb, _ = build_strassen_2x2x2()
            U_blocks.append(Ub)
            V_blocks.append(Vb)
            W_blocks.append(Wb)
            
        # 1 block of 2x2x1
        Ub, Vb, Wb, _ = build_schoolbook(2, 2, 1)
        U_blocks.append(Ub)
        V_blocks.append(Vb)
        W_blocks.append(Wb)
        
        U_full = np.hstack(U_blocks)
        
        # V needs careful index mapping
        # Strassen blocks use k=0,1. Schoolbook uses k=0.
        # Block 0: k=0,1
        # Block 1: k=2,3
        # ...
        # Block 6: k=12,13
        # Block 7 (Schoolbook): k=14
        
        V_full = np.zeros((30, 53))
        W_full = np.zeros((30, 53))
        
        col = 0
        for b in range(7):
            k_start = b * 2
            # V
            for j in range(2):
                for k_local in range(2):
                    k_global = k_start + k_local
                    V_full[j*15 + k_global, col:col+7] = V_blocks[b][j*2 + k_local, :]
            # W
            for k_local in range(2):
                k_global = k_start + k_local
                for i in range(2):
                    W_full[k_global*2 + i, col:col+7] = W_blocks[b][k_local*2 + i, :]
            col += 7
            
        # Schoolbook block
        k_start = 14
        # V
        for j in range(2):
            for k_local in range(1):
                k_global = k_start + k_local
                V_full[j*15 + k_global, col:col+4] = V_blocks[7][j*1 + k_local, :]
        # W
        for k_local in range(1):
            k_global = k_start + k_local
            for i in range(2):
                W_full[k_global*2 + i, col:col+4] = W_blocks[7][k_local*2 + i, :]
        col += 4
        
        if verify_decomposition(U_full, V_full, W_full, 2, 2, 15):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
            
        # Final fallback: Schoolbook 2x2x15 (Rank 60)
        U_s, V_s, W_s, _ = build_schoolbook(2, 2, 15)
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

# EVOLVE-BLOCK-END
