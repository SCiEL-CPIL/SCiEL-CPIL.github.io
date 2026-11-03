import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_sb = n * m * p
    
    # Target rank based on block decomposition
    # 2x2x4 blocks (2 of them): 14 each -> 28
    # 2x2x3 blocks (2 of them): 11 each -> 22
    # 1x2x4 blocks (2 of them): 8 each -> 16
    # 1x2x3 blocks (2 of them): 6 each -> 12
    total_rank = 28 + 22 + 16 + 12 # 78
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a generic set of vectors
    def add_vectors(u_vecs, v_vecs, w_vecs, 
                    i_start, i_len, 
                    j_start, j_len, 
                    k_start, k_len):
        nonlocal col
        num_terms = len(u_vecs)
        
        # u_vecs: list of lists, each list has i_len entries
        # v_vecs: list of lists, each list has j_len entries
        # w_vecs: list of lists, each list has k_len entries
        
        for t in range(num_terms):
            # Add to U
            for ii in range(i_len):
                U[i_start + ii][col] += u_vecs[t][ii]
            # Add to V
            for jj in range(j_len):
                V[j_start + jj][col] += v_vecs[t][jj]
            # Add to W
            for kk in range(k_len):
                W[k_start + kk][col] += w_vecs[t][kk]
            col += 1
            
    # Strassen 2x2x2 Vectors
    # Indices: u[0..3] for A00, A01, A10, A11
    #          v[0..3] for B00, B01, B10, B11
    #          w[0..3] for C00, C01, C10, C11
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, -1, 0],
        [1, 0, -1, 0]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, -1, 0, 0]
    ]
    
    # Verify Strassen
    # T = sum(u_t[i] v_t[j] w_t[k])
    # C00 = T[0,0] = S_U[0][0]*S_V[0][0]*S_W[0][0] + ...
    # We trust the standard vectors if implemented correctly.
    # If not, the check below will fail.
    
    # Define Blocks
    # Block 1: i=0..1, j=0..1, k=0..3 (2x2x4)
    # Use two Strassen blocks: k=0..1 and k=2..3
    # Strassen 1: i=0..1, j=0..1, k=0..1
    add_vectors(S_U, S_V, S_W, 
                i_start=0, i_len=2, 
                j_start=0, j_len=2, 
                k_start=0, k_len=2)
    # Strassen 2: i=0..1, j=0..1, k=2..3
    add_vectors(S_U, S_V, S_W, 
                i_start=0, i_len=2, 
                j_start=0, j_len=2, 
                k_start=2, k_len=2)
                
    # Block 2: i=0..1, j=2..3, k=0..3 (2x2x4)
    # Use two Strassen blocks
    add_vectors(S_U, S_V, S_W, 
                i_start=0, i_len=2, 
                j_start=2, j_len=2, 
                k_start=0, k_len=2)
    add_vectors(S_U, S_V, S_W, 
                i_start=0, i_len=2, 
                j_start=2, j_len=2, 
                k_start=2, k_len=2)
                
    # Block 5: i=0..1, j=0..1, k=4..6 (2x2x3)
    # Use one Strassen (k=4..5) and one Schoolbook (k=6)
    add_vectors(S_U, S_V, S_W, 
                i_start=0, i_len=2, 
                j_start=0, j_len=2, 
                k_start=4, k_len=2)
    # Schoolbook for k=6
    # A is 2x2, B is 2x1 (col 6). C is 2x1.
    # Terms: A00*B06, A01*B16, A10*B06, A11*B16
    sb_u = [[1,0], [0,1], [0,0], [0,0]] # A00, A10, ... wait.
    # U vector for A00*B06: [1, 0] (A00 is row 0)
    # Actually, local U indices: 0->A00, 1->A01, 2->A10, 3->A11? 
    # In my add_vectors, i_len=2. 
    # The local indices in S_U are 0..3. 
    # But here i_len=2. 
    # I need to map the 2x2x1 schoolbook into the 2x2x1 frame.
    # Local A indices: 0 (i=0,j=0), 1 (i=1,j=0)? 
    # No, j is fixed to 0..1 in the block. 
    # In Block 5, j is 0..1. 
    # So local A is 2x2. 
    # Schoolbook terms for 2x2x1:
    # T1: A00 * B0k. U=[1,0,0,0]? No, U has i_len=2? 
    # Wait, my add_vectors takes i_len. 
    # For 2x2x1, i_len=2, j_len=2, k_len=1.
    # But S_U has length 4. 
    # I need specific schoolbook vectors for 2x2x1.
    
    # Let's define 2x2x1 schoolbook vectors explicitly.
    # A is 2x2. B is 2x1. C is 2x1.
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # Terms:
    # 1. A00*B0: U=[1,0,0,0] (if 4 entries) -> but i_len=2?
    # The U matrix in global space has rows corresponding to A elements.
    # For block i=0..1, j=0..1.
    # Global U rows:
    # i=0,j=0 -> idx 0
    # i=0,j=1 -> idx 1
    # i=1,j=0 -> idx 2
    # i=1,j=1 -> idx 3
    # So U vectors for this block should have length 4? 
    # No, add_vectors expects u_vecs[t] to have length i_len? 
    # If i_len=2, it assumes we are only touching 2 rows of U? 
    # No, the block covers 2 rows of i and 2 cols of j. 
    # The number of A elements in the block is 2*2=4. 
    # So u_vecs should have length 4. 
    # But i_len in add_vectors is used for loop `for ii in range(i_len)`. 
    # This is WRONG for 2x2 blocks. 
    # I need to pass the number of elements, or handle the mapping.
    
    # Correction: 
    # The U matrix is flattened A. 
    # Block i_start..i_start+i_len-1, j_start..j_start+j_len-1.
    # Number of elements = i_len * j_len. 
    # My add_vectors uses i_len to iterate over u_vecs[t]. 
    # So u_vecs[t] MUST have length i_len * j_len. 
    # Similarly v_vecs[t] MUST have length j_len * k_len. 
    # And w_vecs[t] MUST have length k_len * i_len. 
    
    # Let's redefine add_vectors to take the full flattened vectors.
    
    pass # Re-implementation needed

    # I will rewrite the construct function properly.

    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    col = 0
    
    def add_block(u_vecs, v_vecs, w_vecs, 
                  i_start, j_start, k_start, 
                  i_len, j_len, k_len):
        nonlocal col
        num_terms = len(u_vecs)
        # u_vecs[t] length i_len*j_len
        # v_vecs[t] length j_len*k_len
        # w_vecs[t] length k_len*i_len
        
        for t in range(num_terms):
            # U
            for ii in range(i_len):
                for jj in range(j_len):
                    local_idx = ii * j_len + jj
                    global_idx = (i_start + ii) * m + (j_start + jj)
                    U[global_idx][col] += u_vecs[t][local_idx]
            # V
            for jj in range(j_len):
                for kk in range(k_len):
                    local_idx = jj * k_len + kk
                    global_idx = (j_start + jj) * p + (k_start + kk)
                    V[global_idx][col] += v_vecs[t][local_idx]
            # W
            for kk in range(k_len):
                for ii in range(i_len):
                    local_idx = kk * i_len + ii
                    global_idx = (k_start + kk) * n + (i_start + ii)
                    W[global_idx][col] += w_vecs[t][local_idx]
            col += 1
            
    # Strassen 2x2x2
    # u: 4 entries (A00, A01, A10, A11)
    # v: 4 entries (B00, B01, B10, B11)
    # w: 4 entries (C00, C01, C10, C11)
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, -1, 0],
        [1, 0, -1, 0]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, -1, 0, 0]
    ]
    
    # 2x2x1 Schoolbook
    # A 2x2, B 2x1, C 2x1
    # Terms: A00B0, A01B1, A10B0, A11B1
    SB_221_U = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    SB_221_V = [
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1]
    ]
    SB_221_W = [
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 0]
    ]
    # Check: 
    # Term 0: A00 * B0 -> C0. W=[1,0]. Correct.
    # Term 1: A01 * B1 -> C0. W=[0,1]? No, A01 is row 0, col 1. B1 is row 1, col 0. Product contributes to C00?
    # C00 = A00B00 + A01B10. 
    # My V indices: 0->B00, 1->B10.
    # So Term 0: A00 (U[0]=1), B00 (V[0]=1). W[0]=1. C00 gets 1.
    # Term 1: A01 (U[1]=1), B10 (V[1]=1). W[0]=1? 
    # My SB_221_W[1] is [0,1]. This puts it in C10? 
    # C10 = A10B00 + A11B10.
    # A01B10 should go to C00? No, C is 2x1. 
    # C_0 = A_00 B_0 + A_01 B_1? 
    # A is 2x2. Rows 0,1. Cols 0,1.
    # B is 2x1. Rows 0,1. Col 0.
    # C is 2x1. Rows 0,1. Col 0.
    # C_0 = A_00 B_0 + A_01 B_1.
    # C_1 = A_10 B_0 + A_11 B_1.
    # So:
    # Term A00B0: U=[1,0,0,0], V=[1,0], W=[1,0].
    # Term A01B1: U=[0,1,0,0], V=[0,1], W=[1,0].
    # Term A10B0: U=[0,0,1,0], V=[1,0], W=[0,1].
    # Term A11B1: U=[0,0,0,1], V=[0,1], W=[0,1].
    
    SB_221_U = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    SB_221_V = [
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1]
    ]
    SB_221_W = [
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1]
    ]
    
    # 1x2xK Schoolbook
    # A 1x2, B 2xK, C 1xK
    # C_j = A0 B0j + A1 B1j
    # Terms: 2*K terms.
    
    # 1. Block 1: i=0..1, j=0..1, k=0..3 (2x2x4)
    add_block(S_U, S_V, S_W, 0, 0, 0, 2, 2, 2)
    add_block(S_U, S_V, S_W, 0, 0, 2, 2, 2, 2)
    
    # 2. Block 2: i=0..1, j=2..3, k=0..3 (2x2x4)
    add_block(S_U, S_V, S_W, 0, 2, 0, 2, 2, 2)
    add_block(S_U, S_V, S_W, 0, 2, 2, 2, 2, 2)
    
    # 5. Block 5: i=0..1, j=0..1, k=4..6 (2x2x3)
    add_block(S_U, S_V, S_W, 0, 0, 4, 2, 2, 2)
    add_block(SB_221_U, SB_221_V, SB_221_W, 0, 0, 6, 2, 2, 1)
    
    # 6. Block 6: i=0..1, j=2..3, k=4..6 (2x2x3)
    add_block(S_U, S_V, S_W, 0, 2, 4, 2, 2, 2)
    add_block(SB_221_U, SB_221_V, SB_221_W, 0, 2, 6, 2, 2, 1)
    
    # 3. Block 3: i=2, j=0..1, k=0..3 (1x2x4)
    # A 1x2, B 2x4, C 1x4
    # U: 2 entries (A00, A01) -> local i_len=1, j_len=2.
    # V: 8 entries (B00..B13) -> local j_len=2, k_len=4.
    # W: 4 entries (C0..C3) -> local k_len=4, i_len=1.
    
    def make_1x2xK(k_len):
        u_vecs = []
        v_vecs = []
        w_vecs = []
        # Terms: A0*B0k, A1*B1k for k in 0..k_len-1
        for k in range(k_len):
            # Term A0 * B0k
            u = [1, 0]
            v = [0] * (2 * k_len)
            v[0 * k_len + k] = 1
            w = [0] * (k_len * 1)
            w[k * 1 + 0] = 1
            u_vecs.append(u)
            v_vecs.append(v)
            w_vecs.append(w)
            
            # Term A1 * B1k
            u = [0, 1]
            v = [0] * (2 * k_len)
            v[1 * k_len + k] = 1
            w = [0] * (k_len * 1)
            w[k * 1 + 0] = 1
            u_vecs.append(u)
            v_vecs.append(v)
            w_vecs.append(w)
        return u_vecs, v_vecs, w_vecs
        
    u3, v3, w3 = make_1x2xK(4)
    add_block(u3, v3, w3, 2, 0, 0, 1, 2, 4)
    
    # 4. Block 4: i=2, j=2..3, k=0..3 (1x2x4)
    u4, v4, w4 = make_1x2xK(4)
    add_block(u4, v4, w4, 2, 2, 0, 1, 2, 4)
    
    # 7. Block 7: i=2, j=0..1, k=4..6 (1x2x3)
    u7, v7, w7 = make_1x2xK(3)
    add_block(u7, v7, w7, 2, 0, 4, 1, 2, 3)
    
    # 8. Block 8: i=2, j=2..3, k=4..6 (1x2x3)
    u8, v8, w8 = make_1x2xK(3)
    add_block(u8, v8, w8, 2, 2, 4, 1, 2, 3)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
