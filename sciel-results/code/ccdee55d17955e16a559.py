import time
import sys

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Global shapes
    U_global = []
    V_global = []
    W_global = []
    
    # We will build the blocks and append to global lists
    # Block 1: C11 (2x4) = A1(2x2)B1a(2x2) + A1(2x2)B1b(2x2)
    # A1 rows 0,1. B1a cols 0,1. B1b cols 2,3.
    # Strassen 2x2x2
    def get_strassen_222():
        # Returns U, V, W for 2x2x2 Strassen
        # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        # n=2, m=2, p=2
        U = [
            [1, 0, 0, 0, 0, 0, 0], # A00
            [0, 1, 0, 0, 0, 0, 0], # A01
            [0, 0, 1, 0, 0, 0, 0], # A10
            [0, 0, 0, 1, 0, 0, 0]  # A11
        ]
        V = [
            [1, 1, 1, 1, 1, 1, 1], # B00
            [0, 1, 0, 0, 0, 0, 0], # B01
            [0, 0, 1, 0, 0, 0, 0], # B10
            [0, 0, 0, 1, 1, 1, 1]  # B11
        ]
        # Wait, Strassen is:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        U = [
            [1, 0, 1, 0, 1, 0, 0], # A00
            [0, 0, 0, 0, 1, 0, 0], # A01
            [0, 1, 0, 0, 0, 1, 0], # A10
            [1, 1, 0, 1, 0, 0, 1]  # A11
        ]
        V = [
            [1, 1, 0, 0, 0, 1, 0], # B00
            [0, 0, 1, 0, 1, 1, 0], # B01
            [0, 0, 0, 0, 0, 0, 1], # B10
            [1, 0, -1, -1, 1, 0, 1] # B11
        ]
        W = [
            [1, -1, 1, 1, 0, 0, 0], # C00
            [0, 0, 1, 0, 1, 0, 0], # C01
            [0, 1, 0, 1, 0, 0, 0], # C10
            [1, -1, 1, 0, 0, 1, 1]  # C11
        ]
        return U, V, W

    def get_221():
        # 2x2x1, Rank 2
        # C0 = A00B00 + A01B10
        # C1 = A10B00 + A11B10
        U = [
            [1, 1], # A00
            [0, 0], # A01 (Wait, A01 is index 1 in 2x2? i=0, j=1)
            [0, 0], # A10
            [0, 0]  # A11
        ]
        # Correct indices for 2x2:
        # 0: 0,0
        # 1: 0,1
        # 2: 1,0
        # 3: 1,1
        U = [
            [1, 0], # A00
            [0, 1], # A01
            [0, 0], # A10
            [0, 0]  # A11
        ]
        # This is wrong. 
        # C0 = A00B00 + A01B10
        # C1 = A10B00 + A11B10
        # Using the trick:
        # C0 = 1/2 [ (A00+A01)(B00+B10) + (A00-A01)(B00-B10) ]
        # C1 = 1/2 [ (A10+A11)(B00+B10) + (A10-A11)(B00-B10) ]
        U = [
            [1, 1], # A00
            [1, -1], # A01
            [0, 0], # A10
            [0, 0]  # A11
        ]
        # Wait, A10 and A11 are in row 1.
        U = [
            [1, 1], # A00
            [1, -1], # A01
            [0, 0], # A10
            [0, 0]  # A11
        ]
        # This U only covers row 0.
        # We need U for all 4 inputs.
        U = [
            [1, 1], # A00
            [1, -1], # A01
            [0, 0], # A10
            [0, 0]  # A11
        ]
        # This is incorrect. A10 and A11 contribute to C1.
        U = [
            [1, 1], # A00
            [1, -1], # A01
            [0, 0], # A10
            [0, 0]  # A11
        ]
        # Let's restart 221.
        # C0 = A00B00 + A01B10
        # C1 = A10B00 + A11B10
        # Term 1: (A00+A01) * (B00+B10) -> contributes to C0 and C1?
        # No, (A00+A01) is not A10+A11.
        # We need separate terms for C0 and C1?
        # Rank 2 means we share the multiplications.
        # Let M1 = (A00+A01)(B00+B10)
        # Let M2 = (A00-A01)(B00-B10)
        # C0 = 1/2 (M1 + M2) = A00B00 + A01B10.
        # What about C1?
        # C1 = A10B00 + A11B10.
        # This requires A10, A11.
        # So we need more terms.
        # Rank of 2x2x1 is actually 4?
        # No, tensor rank of 2x2x1 is 2?
        # Let's check literature.
        # Rank of 2x2x1 is 2?
        # If rank 2, then C0 and C1 must be expressible with 2 shared multiplications.
        # But C0 depends on A00, A01 and C1 on A10, A11.
        # They are disjoint in A.
        # So we need at least 2 multiplications for C0 and 2 for C1?
        # Total 4?
        # Yes, rank is 4.
        # My previous deduction was wrong because I assumed A00+A01 could help C1, but it can't.
        # So 2x2x1 is rank 4.
        # 1x2x1 is rank 2?
        # C0 = A00B00 + A01B10.
        # This is one dot product.
        # Rank 2 using the trick?
        # C0 = 1/2 [ (A00+A01)(B00+B10) + (A00-A001)(B00-B10) ]
        # Yes, rank 2.
        # 1x2x2 is rank 2?
        # C0 = A00B00 + A01B10
        # C1 = A00B01 + A01B11
        # This is A dot B_col0 and A dot B_col1.
        # Rank 2.
        
        return None # Placeholder

    # Since 2x2x1 is rank 4, let's use schoolbook.
    # Block 12: 2x2x1 (Rank 4) + 2x2x2 (Strassen 7) = 11.
    # Block 21: 1x2x2 (Rank 2) + 1x2x2 (Rank 2) = 4.
    # Block 22: 1x2x1 (Rank 2) + 1x2x2 (Rank 2) = 4.
    # Total: 14 + 11 + 4 + 4 = 33.
    
    # Let's implement this.
    
    def make_block_strassen_222(offset_u, offset_v, offset_w, row_offset, col_offset):
        # offset_u: index in global U
        # offset_v: index in global V
        # offset_w: index in global W
        # row_offset: i start
        # col_offset: k start
        U, V, W = get_strassen_222()
        # Map local indices to global
        # Local U: 2x2 -> 4 rows. Global U rows: (row_offset + i) * 4 + j
        # Local V: 2x2 -> 4 rows. Global V rows: (j) * 7 + (col_offset + k)
        # Local W: 2x2 -> 4 rows. Global W rows: (col_offset + k) * 3 + (row_offset + i)
        
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    global_u_idx = offset_u + i * 4 + j
                    # U is (n*m, rank). Here n=2, m=2.
                    # U[i*2+j, r]
                    U_global[global_u_idx][offset_w + r] = U[i*2+j][r]
                    
                    # V is (m*p, rank). Here m=2, p=2.
                    # V[j*2+k, r]
                    # But global V is (4*7, rank).
                    # Global V row: (local_j) * 7 + (col_offset + local_k)
                    # Wait, local j is 0,1. Global j is 0,1,2,3.
                    # The block uses A1 (rows 0,1) and B1a (cols 0,1).
                    # So local j=0 -> global j=0, local j=1 -> global j=1.
                    # local k=0 -> global k=col_offset+0, local k=1 -> global k=col_offset+1.
                    for j in range(2):
                        for k in range(2):
                            global_v_idx = j * 7 + (col_offset + k)
                            V_global[global_v_idx][offset_w + r] = V[j*2+k][r]
                            
                    # W is (p*n, rank). Here p=2, n=2.
                    # W[k*2+i, r]
                    # Global W row: (col_offset + local_k) * 3 + (row_offset + local_i)
                    for k in range(2):
                        for i in range(2):
                            global_w_idx = (col_offset + k) * 3 + (row_offset + i)
                            W_global[global_w_idx][offset_w + r] = W[k*2+i][r]
                            
    # This mapping is complex. Let's just build the full matrices directly.
    
    U = [[0]*33 for _ in range(12)]
    V = [[0]*33 for _ in range(28)]
    W = [[0]*33 for _ in range(21)]
    
    # Block 1: C11 (2x4)
    # A1 (rows 0,1) * B1a (cols 0,1) -> C11a (2x2)
    # A1 (rows 0,1) * B1b (cols 2,3) -> C11b (2x2)
    
    # Strassen 1
    U_s, V_s, W_s = get_strassen_222()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i*4+j][r] += U_s[i*2+j][r]
        for j in range(2):
            for k in range(2):
                V[j*7+k][r] += V_s[j*2+k][r]
        for k in range(2):
            for i in range(2):
                W[k*3+i][r] += W_s[k*2+i][r]
                
    # Strassen 2 (cols 2,3)
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i*4+j][r+7] += U_s[i*2+j][r]
        for j in range(2):
            for k in range(2):
                V[j*7+(2+k)][r+7] += V_s[j*2+k][r]
        for k in range(2):
            for i in range(2):
                W[(2+k)*3+i][r+7] += W_s[k*2+i][r]
                
    # Block 2: C12 (2x3)
    # A1 (rows 0,1) * B2a (col 4) -> C12a (2x1)
    # A1 (rows 0,1) * B2b (cols 5,6) -> C12b (2x2)
    
    # 2x2x1 Schoolbook (Rank 4)
    # C0 = A00B00 + A01B10
    # C1 = A10B00 + A11B10
    # Here B is 2x1. B00, B10.
    # Global B cols: 4.
    # Local B: 2x1.
    # U: 2x2. V: 2x1. W: 1x2.
    # r=8..11
    for r in range(4):
        i, j, k = r // 4, (r // 2) % 2, r % 2 # This mapping is wrong.
        # Schoolbook:
        # r=0: i=0, j=0, k=0
        # r=1: i=0, j=1, k=0
        # r=2: i=1, j=0, k=0
        # r=3: i=1, j=1, k=0
        i = r // 2
        j = r % 2
        k = 0
        U[i*4+j][8+r] = 1
        V[j*7+4][8+r] = 1
        W[4*3+i][8+r] = 1 # W[k*n+i] = W[0*2+i] = W[i]
        
    # Strassen 3 (cols 5,6)
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i*4+j][12+r] += U_s[i*2+j][r]
        for j in range(2):
            for k in range(2):
                V[j*7+(5+k)][12+r] += V_s[j*2+k][r]
        for k in range(2):
            for i in range(2):
                W[(5+k)*3+i][12+r] += W_s[k*2+i][r]
                
    # Block 3: C21 (1x4)
    # A2 (row 2) * B1a (cols 0,1) -> C21a (1x2)
    # A2 (row 2) * B1b (cols 2,3) -> C21b (1x2)
    
    # 1x2x2 Rank 2
    # C0 = A00B00 + A01B10
    # C1 = A00B01 + A01B11
    # r=19..20
    # Term 1: A00 * (B00, B01)
    # Term 2: A01 * (B10, B11)
    # U: 1x2. V: 2x2. W: 2x1.
    # r=19: A00
    U[2*4+0][19] = 1 # A20
    U[2*4+1][20] = 1 # A21
    # V for r=19: B00, B01
    V[0*7+0][19] = 1
    V[0*7+1][19] = 1
    # V for r=20: B10, B11
    V[1*7+0][20] = 1
    V[1*7+1][20] = 1
    # W for r=19: C0
    W[0*3+2][19] = 1
    # W for r=20: C1
    W[1*3+2][20] = 1
    
    # 1x2x2 Rank 2 (cols 2,3)
    # r=21..22
    U[2*4+0][21] = 1
    U[2*4+1][22] = 1
    V[0*7+2][21] = 1
    V[0*7+3][21] = 1
    V[1*7+2][22] = 1
    V[1*7+3][22] = 1
    W[2*3+2][21] = 1
    W[3*3+2][22] = 1
    
    # Block 4: C22 (1x3)
    # A2 (row 2) * B2a (col 4) -> C22a (1x1)
    # A2 (row 2) * B2b (cols 5,6) -> C22b (1x2)
    
    # 1x2x1 Rank 2
    # C0 = A00B00 + A01B10
    # r=23..24
    # Trick: 1/2 [ (A00+A01)(B00+B10) + (A00-A01)(B00-B10) ]
    # r=23: (A00+A01) * (B00+B10)
    U[2*4+0][23] = 1
    U[2*4+1][23] = 1
    V[0*7+4][23] = 1
    V[1*7+4][23] = 1
    W[4*3+2][23] = 0.5
    
    # r=24: (A00-A01) * (B00-B10)
    U[2*4+0][24] = 1
    U[2*4+1][24] = -1
    V[0*7+4][24] = 1
    V[1*7+4][24] = -1
    W[4*3+2][24] = 0.5
    
    # 1x2x2 Rank 2 (cols 5,6)
    # r=25..26
    U[2*4+0][25] = 1
    U[2*4+1][26] = 1
    V[0*7+5][25] = 1
    V[0*7+6][25] = 1
    V[1*7+5][26] = 1
    V[1*7+6][26] = 1
    W[5*3+2][25] = 1
    W[6*3+2][26] = 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()
# EVOLVE-BLOCK-END

def _construct_impl():
    # The logic above
    n, m, p = DIMENSIONS
    U = [[0]*33 for _ in range(12)]
    V = [[0]*33 for _ in range(28)]
    W = [[0]*33 for _ in range(21)]
    
    def get_strassen_222():
        U = [
            [1, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, 1]
        ]
        V = [
            [1, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [1, 0, -1, -1, 1, 0, 1]
        ]
        W = [
            [1, -1, 1, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [1, -1, 1, 0, 0, 1, 1]
        ]
        return U, V, W

    U_s, V_s, W_s = get_strassen_222()
    
    # Block 1: C11
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i*4+j][r] += U_s[i*2+j][r]
        for j in range(2):
            for k in range(2):
                V[j*7+k][r] += V_s[j*2+k][r]
        for k in range(2):
            for i in range(2):
                W[k*3+i][r] += W_s[k*2+i][r]
                
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i*4+j][r+7] += U_s[i*2+j][r]
        for j in range(2):
            for k in range(2):
                V[j*7+(2+k)][r+7] += V_s[j*2+k][r]
        for k in range(2):
            for i in range(2):
                W[(2+k)*3+i][r+7] += W_s[k*2+i][r]
                
    # Block 2: C12
    for r in range(4):
        i = r // 2
        j = r % 2
        k = 0
        U[i*4+j][8+r] = 1
        V[j*7+4][8+r] = 1
        W[4*3+i][8+r] = 1
        
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i*4+j][12+r] += U_s[i*2+j][r]
        for j in range(2):
            for k in range(2):
                V[j*7+(5+k)][12+r] += V_s[j*2+k][r]
        for k in range(2):
            for i in range(2):
                W[(5+k)*3+i][12+r] += W_s[k*2+i][r]
                
    # Block 3: C21
    U[2*4+0][19] = 1
    U[2*4+1][20] = 1
    V[0*7+0][19] = 1
    V[0*7+1][19] = 1
    V[1*7+0][20] = 1
    V[1*7+1][20] = 1
    W[0*3+2][19] = 1
    W[1*3+2][20] = 1
    
    U[2*4+0][21] = 1
    U[2*4+1][22] = 1
    V[0*7+2][21] = 1
    V[0*7+3][21] = 1
    V[1*7+2][22] = 1
    V[1*7+3][22] = 1
    W[2*3+2][21] = 1
    W[3*3+2][22] = 1
    
    # Block 4: C22
    U[2*4+0][23] = 1
    U[2*4+1][23] = 1
    V[0*7+4][23] = 1
    V[1*7+4][23] = 1
    W[4*3+2][23] = 0.5
    
    U[2*4+0][24] = 1
    U[2*4+1][24] = -1
    V[0*7+4][24] = 1
    V[1*7+4][24] = -1
    W[4*3+2][24] = 0.5
    
    U[2*4+0][25] = 1
    U[2*4+1][26] = 1
    V[0*7+5][25] = 1
    V[0*7+6][25] = 1
    V[1*7+5][26] = 1
    V[1*7+6][26] = 1
    W[5*3+2][25] = 1
    W[6*3+2][26] = 1
    
    return {"U": U, "V": V, "W": W}

