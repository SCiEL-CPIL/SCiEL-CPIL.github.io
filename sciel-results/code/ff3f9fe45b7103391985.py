import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create the base case 3x2x2 decomposition (Rank 11)
    def get_base_322():
        n_b, m_b, p_b = 3, 2, 2
        rank_b = 11
        U_b = np.zeros((n_b * m_b, rank_b))
        V_b = np.zeros((m_b * p_b, rank_b))
        W_b = np.zeros((p_b * n_b, rank_b))
        
        # Strassen for rows 0,1 (2x2x2)
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Term 0: M1
        # U: A00, A11 -> idx 0, 3 (i*m+j: 0*2+0=0, 1*2+1=3)
        U_b[0, 0] = 1; U_b[3, 0] = 1
        # V: B00, B11 -> idx 0, 3 (j*p+k: 0*2+0=0, 1*2+1=3)
        V_b[0, 0] = 1; V_b[3, 0] = 1
        # W: C00, C11 -> idx k*n+i: 0*3+0=0, 1*3+1=4
        W_b[0, 0] = 1; W_b[4, 0] = 1
        
        # Term 1: M2
        # U: A10, A11 -> idx 2, 3
        U_b[2, 1] = 1; U_b[3, 1] = 1
        # V: B00 -> idx 0
        V_b[0, 1] = 1
        # W: C10 -> idx 1*3+1=4? No C10 is i=1, k=0. idx 0*3+1=1.
        # Wait, W index is k*n + i.
        # C10: i=1, k=0. idx 0.
        # C11: i=1, k=1. idx 1*3+1=4.
        # M2 contributes to C10.
        W_b[1, 1] = 1
        
        # Term 2: M3
        # U: A00 -> idx 0
        U_b[0, 2] = 1
        # V: B01, B11 -> idx 1, 3
        V_b[1, 2] = 1; V_b[3, 2] = -1
        # W: C01 -> i=0, k=1. idx 1*3+0=3.
        W_b[3, 2] = 1
        
        # Term 3: M4
        # U: A11 -> idx 3
        U_b[3, 3] = 1
        # V: B10, B00 -> idx 2, 0
        V_b[2, 3] = 1; V_b[0, 3] = -1
        # W: C00, C10 -> i=0,k=0 (idx 0), i=1,k=0 (idx 1)
        W_b[0, 3] = 1; W_b[1, 3] = 1
        
        # Term 4: M5
        # U: A00, A01 -> idx 0, 1
        U_b[0, 4] = 1; U_b[1, 4] = 1
        # V: B11 -> idx 3
        V_b[3, 4] = 1
        # W: C00, C01 -> i=0,k=0 (idx 0), i=0,k=1 (idx 3)
        W_b[0, 4] = -1; W_b[3, 4] = 1
        
        # Term 5: M6
        # U: A10, A00 -> idx 2, 0
        U_b[2, 5] = 1; U_b[0, 5] = -1
        # V: B00, B01 -> idx 0, 1
        V_b[0, 5] = 1; V_b[1, 5] = 1
        # W: C11 -> i=1, k=1. idx 4.
        W_b[4, 5] = 1
        
        # Term 6: M7
        # U: A01, A11 -> idx 1, 3
        U_b[1, 6] = 1; U_b[3, 6] = -1
        # V: B10, B11 -> idx 2, 3
        V_b[2, 6] = 1; V_b[3, 6] = 1
        # W: C00 -> i=0, k=0. idx 0.
        W_b[0, 6] = 1
        
        # Trivial for row 2
        # Term 7: A20 B00 -> C20
        # U: A20 -> idx 2*2+0=4
        U_b[4, 7] = 1
        # V: B00 -> idx 0
        V_b[0, 7] = 1
        # W: C20 -> i=2, k=0. idx 0*3+2=2.
        W_b[2, 7] = 1
        
        # Term 8: A20 B01 -> C21
        # U: A20 -> idx 4
        U_b[4, 8] = 1
        # V: B01 -> idx 1
        V_b[1, 8] = 1
        # W: C21 -> i=2, k=1. idx 1*3+2=5.
        W_b[5, 8] = 1
        
        # Term 9: A21 B10 -> C20
        # U: A21 -> idx 5
        U_b[5, 9] = 1
        # V: B10 -> idx 2
        V_b[2, 9] = 1
        # W: C20 -> idx 2
        W_b[2, 9] = 1
        
        # Term 10: A21 B11 -> C21
        # U: A21 -> idx 5
        U_b[5, 10] = 1
        # V: B11 -> idx 3
        V_b[3, 10] = 1
        # W: C21 -> idx 5
        W_b[5, 10] = 1
        
        return U_b, V_b, W_b

    # Recursive assembly
    # We need to build U, V, W for 3x4x8
    # Split 4x8 into two 2x8? No, 3x4x8.
    # Split m=4 into 2+2.
    # A = [A1, A2] (3x2 each)
    # B = [B1; B2] (2x8 each)
    # C = A1 B1 + A2 B2
    
    # Get base 3x2x2
    U_base, V_base, W_base = get_base_322()
    rank_base = U_base.shape[1]
    
    # Build 3x2x4 (p=4)
    # Split p=4 into 2+2.
    # B = [B1, B2] (2x2 each)
    # C = [C1, C2]
    # C1 = A B1, C2 = A B2
    # Two independent 3x2x2 decompositions.
    # U is same for both? Yes, A is same.
    # V and W are different.
    
    # Actually, it's easier to just concatenate the columns.
    # For 3x2x4:
    # Rank = 2 * rank_base = 22
    U_324 = np.zeros((3 * 2, 2 * rank_base))
    V_324 = np.zeros((2 * 4, 2 * rank_base))
    W_324 = np.zeros((4 * 3, 2 * rank_base))
    
    # Part 1: B[:, 0:2]
    U_324[:, 0:rank_base] = U_base
    # V for B1 (2x2) is V_base
    # V index in 324: j*p+k. p=4.
    # V_base indices are for p=2.
    # j in 0..1, k in 0..1.
    # In 324, k is 0..1 for first block.
    # So V_324[j*4+k, t] = V_base[j*2+k, t]
    for j in range(2):
        for k in range(2):
            V_324[j*4+k, 0:rank_base] = V_base[j*2+k, :]
    # W for C1 (3x2) is W_base
    # W index in 324: k*n+i. n=3.
    # k in 0..1 for first block.
    # W_324[k*3+i, t] = W_base[k*2+i? No, W_base index is k*n+i with n=3?
    # Wait, W_base was built for 3x2x2. W index k*n+i.
    # So W_base[k*3+i, t].
    # In 324, W index is k*3+i.
    # k is 0..1.
    # So W_324[k*3+i, 0:rank_base] = W_base[k*3+i, :]
    W_324[:, 0:rank_base] = W_base # This is wrong because W_base size is 6, W_324 part is 6.
    # But W_324 total size is 4*3=12.
    # W_base covers k=0,1.
    # So W_324[0:6, 0:rank_base] = W_base
    # W_324[6:12, 0:rank_base] = 0
    
    # Part 2: B[:, 2:4]
    U_324[:, rank_base:2*rank_base] = U_base
    # V for B2 (2x2) is V_base, but k offset by 2.
    for j in range(2):
        for k in range(2):
            V_324[j*4+(k+2), rank_base:2*rank_base] = V_base[j*2+k, :]
    # W for C2 (3x2) is W_base, but k offset by 2.
    for k in range(2):
        for i in range(3):
            W_324[(k+2)*3+i, rank_base:2*rank_base] = W_base[k*3+i, :]

    # Build 3x2x8 (p=8)
    # Split p=8 into 4+4.
    # Two 3x2x4 decompositions.
    rank_324 = U_324.shape[1]
    U_328 = np.zeros((3 * 2, 2 * rank_324))
    V_328 = np.zeros((2 * 8, 2 * rank_324))
    W_328 = np.zeros((8 * 3, 2 * rank_324))
    
    # Part 1: B[:, 0:4]
    U_328[:, 0:rank_324] = U_324
    # V offset 0
    V_328[:, 0:rank_324] = V_324
    # W offset 0
    W_328[:, 0:rank_324] = W_324
    
    # Part 2: B[:, 4:8]
    U_328[:, rank_324:2*rank_324] = U_324
    # V offset 4
    V_328[:, rank_324:2*rank_324] = V_324
    # W offset 4
    W_328[:, rank_324:2*rank_324] = W_324

    # Build 3x4x8 (m=4)
    # Split m=4 into 2+2.
    # A = [A1, A2]
    # B = [B1; B2]
    # C = A1 B1 + A2 B2
    # Two 3x2x8 decompositions.
    rank_328 = U_328.shape[1]
    U_final = np.zeros((3 * 4, 2 * rank_328))
    V_final = np.zeros((4 * 8, 2 * rank_328))
    W_final = np.zeros((8 * 3, 2 * rank_328))
    
    # Part 1: A1 (rows 0:2 of A? No, cols 0:2 of A)
    # A is 3x4. A1 is 3x2 (cols 0,1).
    # U_328 is for 3x2.
    # U_final index i*m+j. m=4.
    # A1 corresponds to j=0,1.
    # So U_final[i*4+j, 0:rank_328] = U_328[i*2+j, :]
    for i in range(3):
        for j in range(2):
            U_final[i*4+j, 0:rank_328] = U_328[i*2+j, :]
    
    # B1 is 2x8 (rows 0,1 of B).
    # V_328 is for 2x8.
    # V_final index j*p+k. p=8.
    # B1 corresponds to j=0,1.
    # So V_final[j*8+k, 0:rank_328] = V_328[j*8+k, :]
    V_final[:, 0:rank_328] = V_328
    
    # W is same for both parts?
    # W_final index k*n+i.
    # W_328 is for 3x2x8.
    # W_final[k*3+i, 0:rank_328] = W_328[k*3+i, :]
    W_final[:, 0:rank_328] = W_328
    
    # Part 2: A2 (cols 2,3 of A)
    # U_final[i*4+j, rank_328:2*rank_328] = U_328[i*2+(j-2), :]
    for i in range(3):
        for j in range(2, 4):
            U_final[i*4+j, rank_328:2*rank_328] = U_328[i*2+(j-2), :]
    
    # B2 is 2x8 (rows 2,3 of B).
    # V_final[j*8+k, rank_328:2*rank_328] = V_328[(j-2)*8+k, :]
    for j in range(2, 4):
        for k in range(8):
            V_final[j*8+k, rank_328:2*rank_328] = V_328[(j-2)*8+k, :]
    
    # W is same
    W_final[:, rank_328:2*rank_328] = W_328
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
