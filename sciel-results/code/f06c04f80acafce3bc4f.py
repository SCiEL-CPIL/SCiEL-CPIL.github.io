import time
import sys

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0
    
    # We will construct a rank 44 decomposition.
    # 2x4x6 = (2x2x6) + (2x2x6)
    # 2x2x6 = (2x2x3) + (2x2x3)
    # 2x2x3 = (2x2x2 Strassen) + (2x2x1 Schoolbook)
    
    # Helper to create zero matrices
    def zero_mat(r, c):
        return [[0.0] * c for _ in range(r)]
    
    # 1. Strassen 2x2x2 (Rank 7)
    # U: 4x7, V: 4x7, W: 4x7 (indices k*n+i for 2x2 is k*2+i)
    U_s = zero_mat(4, 7)
    V_s = zero_mat(4, 7)
    W_s = zero_mat(4, 7)
    
    # Term 0
    U_s[0][0] = 1; U_s[3][0] = 1
    V_s[0][0] = 1; V_s[3][0] = 1
    W_s[0][0] = 1; W_s[3][0] = 1
    
    # Term 1
    U_s[2][1] = 1; U_s[3][1] = 1
    V_s[0][1] = 1
    W_s[1][1] = 1; W_s[3][1] = -1
    
    # Term 2
    U_s[1][2] = 1
    V_s[2][2] = 1; V_s[0][2] = -1
    W_s[2][2] = 1; W_s[3][2] = 1
    
    # Term 3
    U_s[3][3] = 1
    V_s[3][3] = 1; V_s[1][3] = -1
    W_s[0][3] = 1
    
    # Term 4
    U_s[0][4] = 1; U_s[1][4] = 1
    V_s[3][4] = 1
    W_s[2][4] = 1; W_s[3][4] = 1
    
    # Term 5
    U_s[2][5] = 1; U_s[0][5] = -1
    V_s[0][5] = 1; V_s[1][5] = 1
    W_s[0][5] = -1; W_s[1][5] = 1
    
    # Term 6
    U_s[1][6] = 1; U_s[3][6] = -1
    V_s[2][6] = 1; V_s[3][6] = 1
    W_s[0][6] = -1
    
    # 2. Schoolbook 2x2x1 (Rank 4)
    # U: 4x4, V: 2x4 (B is 2x1), W: 2x4 (C is 2x1, idx k*2+i)
    U_k = zero_mat(4, 4)
    V_k = zero_mat(2, 4) # Note: V for 2x1 is 2 rows
    W_k = zero_mat(2, 4) # Note: W for 2x1 is 2 rows (k=0)
    
    U_k[0][0] = 1
    V_k[0][0] = 1
    W_k[0][0] = 1
    
    U_k[1][1] = 1
    V_k[1][1] = 1
    W_k[0][1] = 1
    
    U_k[2][2] = 1
    V_k[0][2] = 1
    W_k[1][2] = 1
    
    U_k[3][3] = 1
    V_k[1][3] = 1
    W_k[1][3] = 1
    
    # 3. Combine to 2x2x3 (Rank 11)
    # U: 4x11
    # V: 6x11 (B is 2x3)
    # W: 6x11 (C is 2x3, idx k*2+i)
    
    U_223 = zero_mat(4, 11)
    V_223 = zero_mat(6, 11)
    W_223 = zero_mat(6, 11)
    
    # Copy Strassen (cols 0,1 of B and C)
    for r in range(7):
        for i in range(4):
            U_223[i][r] = U_s[i][r]
            V_223[i][r] = V_s[i][r] # B indices 0-3 map to 0-3 in 2x3? 
            # B is 2x3. Indices j*3+k.
            # Strassen V indices are j*2+k for j,k in 0,1.
            # j=0,k=0 -> 0. j=0,k=1 -> 1. j=1,k=0 -> 2. j=1,k=1 -> 3.
            # In 2x3: j=0,k=0 -> 0. j=0,k=1 -> 1. j=1,k=0 -> 3. j=1,k=1 -> 4.
            # Mismatch!
            
    # Let's redo the V mapping for 2x2x3.
    # Strassen V (2x2) indices:
    # 0: B00, 1: B01, 2: B10, 3: B11.
    # 2x3 V indices:
    # 0: B00, 1: B01, 2: B02, 3: B10, 4: B11, 5: B12.
    # So Strassen V[0] -> V_223[0]
    # Strassen V[1] -> V_223[1]
    # Strassen V[2] -> V_223[3]
    # Strassen V[3] -> V_223[4]
    
    v_map_s = [0, 1, 3, 4]
    for r in range(7):
        for i in range(4):
            U_223[i][r] = U_s[i][r]
            V_223[v_map_s[i]][r] = V_s[i][r]
            W_223[i][r] = W_s[i][r] # W indices k*2+i. C is 2x2. 
            # C indices in 2x3:
            # 0: C00, 1: C10, 2: C01, 3: C11.
            # 2x3 C indices:
            # 0: C00, 1: C10, 2: C01, 3: C11, 4: C02, 5: C12.
            # So W_s indices 0-3 map directly to 0-3.
            
    # Copy Schoolbook (col 2 of B and C)
    # U_k is 4x4. U_223 cols 7-10.
    for r in range(4):
        for i in range(4):
            U_223[i][7+r] = U_k[i][r]
            
    # V_k is 2x4. B indices j*1+k (k=0).
    # V_k[0] is B00 (of 2x1 block). Maps to B02 in 2x3. Index 2.
    # V_k[1] is B10 (of 2x1 block). Maps to B12 in 2x3. Index 5.
    v_map_k = [2, 5]
    for r in range(4):
        for i in range(2):
            V_223[v_map_k[i]][7+r] = V_k[i][r]
            
    # W_k is 2x4. C indices k*2+i (k=0).
    # W_k[0] is C00 (of 2x1 block). Maps to C02 in 2x3. Index 4.
    # W_k[1] is C10 (of 2x1 block). Maps to C12 in 2x3. Index 5.
    w_map_k = [4, 5]
    for r in range(4):
        for i in range(2):
            W_223[w_map_k[i]][7+r] = W_k[i][r]
            
    # 4. Combine to 2x2x6 (Rank 22)
    # U: 4x22
    # V: 12x22 (B is 2x6)
    # W: 12x22 (C is 2x6, idx k*2+i)
    
    U_226 = zero_mat(4, 22)
    V_226 = zero_mat(12, 22)
    W_226 = zero_mat(12, 22)
    
    # Block 1: cols 0-2 of B, cols 0-2 of C
    for r in range(11):
        for i in range(4):
            U_226[i][r] = U_223[i][r]
            V_226[i][r] = V_223[i][r] # B 2x3 indices 0-5 map to 0-5 in 2x6?
            # B 2x6 indices j*6+k.
            # 2x3 indices j*3+k.
            # j=0,k=0->0. j=0,k=1->1. j=0,k=2->2.
            # j=1,k=0->6. j=1,k=1->7. j=1,k=2->8.
            # So V_223 indices 0,1,2 map to 0,1,2.
            # V_223 indices 3,4,5 map to 6,7,8.
            
    v_map_23_to_26 = [0, 1, 2, 6, 7, 8]
    w_map_23_to_26 = [0, 1, 2, 3, 4, 5] # C 2x3 indices k*2+i.
    # C 2x6 indices k*2+i.
    # k=0: 0,1. k=1: 2,3. k=2: 4,5.
    # 2x3 C indices: k=0:0,1. k=1:2,3. k=2:4,5.
    # So W maps directly.
    
    for r in range(11):
        for i in range(6):
            V_226[v_map_23_to_26[i]][r] = V_223[i][r]
            W_226[i][r] = W_223[i][r]
            
    # Block 2: cols 3-5 of B, cols 3-5 of C
    for r in range(11):
        for i in range(4):
            U_226[i][11+r] = U_223[i][r]
            
    # V indices for cols 3-5.
    # j=0,k=3->3. j=0,k=4->4. j=0,k=5->5.
    # j=1,k=3->9. j=1,k=4->10. j=1,k=5->11.
    v_map_23_to_26_b = [3, 4, 5, 9, 10, 11]
    # W indices for cols 3-5.
    # k=3: 6,7. k=4: 8,9. k=5: 10,11.
    w_map_23_to_26_b = [6, 7, 8, 9, 10, 11]
    
    for r in range(11):
        for i in range(6):
            V_226[v_map_23_to_26_b[i]][11+r] = V_223[i][r]
            W_226[i+6][11+r] = W_223[i][r]
            
    # 5. Combine to 2x4x6 (Rank 44)
    # U: 8x44 (A is 2x4)
    # V: 24x44 (B is 4x6)
    # W: 12x44 (C is 2x6, idx k*2+i)
    
    U_final = zero_mat(8, 44)
    V_final = zero_mat(24, 44)
    W_final = zero_mat(12, 44)
    
    # Block 1: A rows 0-1, B rows 0-1
    for r in range(22):
        for i in range(4):
            U_final[i][r] = U_226[i][r]
            V_final[i][r] = V_226[i][r] # B 2x6 indices 0-11 map to 0-11 in 4x6?
            # B 4x6 indices j*6+k.
            # 2x6 indices j*6+k (j=0,1).
            # j=0: 0-5. j=1: 6-11.
            # So V_226 maps directly.
            
    for r in range(22):
        for i in range(12):
            W_final[i][r] = W_226[i][r]
            
    # Block 2: A rows 0-1, B rows 2-3
    for r in range(22):
        for i in range(4):
            U_final[i][22+r] = U_226[i][r]
            
    # V indices for rows 2-3.
    # j=2: 12-17. j=3: 18-23.
    v_map_26_to_46_b = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    for r in range(22):
        for i in range(12):
            V_final[v_map_26_to_46_b[i]][22+r] = V_226[i][r]
            
    for r in range(22):
        for i in range(12):
            W_final[i][22+r] = W_226[i][r]
            
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
