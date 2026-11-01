import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 6
    
    # Rank 42 solution based on Strassen blocks
    rank = 42
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen 2x2 @ 2x2
    # Indices: A(i,j), B(j,k) -> C(i,k)
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # We split m=4 into two blocks of size 2: j in {0,1} and j in {2,3}
    # For each block, we use Strassen.
    
    strassen_U = [
        {0: 1, 3: 1},       # P1
        {2: 1, 3: 1},       # P2
        {0: 1},             # P3
        {3: 1},             # P4
        {0: 1, 1: 1},       # P5
        {0: -1, 2: 1},      # P6
        {1: 1, 3: -1},      # P7
    ]
    strassen_V = [
        {0: 1, 3: 1},       # P1
        {0: 1},             # P2
        {1: 1, 3: -1},      # P3
        {2: 1, 0: -1},      # P4
        {3: 1},             # P5
        {0: 1, 1: 1},       # P6
        {2: 1, 3: 1},       # P7
    ]
    strassen_W = [
        {0: 1, 3: 1},       # P1
        {2: 1, 3: -1},      # P2
        {1: 1, 3: 1},       # P3
        {0: 1, 2: 1},       # P4
        {0: -1, 1: 1},      # P5
        {3: 1},             # P6
        {0: 1},             # P7
    ]
    
    col_offset = 0
    for j_block in range(2):
        # j_block 0: j=0,1
        # j_block 1: j=2,3
        
        for prod_idx in range(7):
            col = col_offset + prod_idx
            
            # Map local 2x2 indices to global indices
            # Local i in {0,1}, j in {0,1}, k in {0,5} (since p=6, but Strassen is 2x2 @ 2x2, so k in {0,1} for the block? No.)
            
            # Wait, the second matrix is 4x6.
            # If we split m=4 into 2+2, we have two 2x6 matrices B_L and B_R.
            # Strassen works on 2x2 @ 2x2.
            # We cannot directly use Strassen on 2x2 @ 2x6.
            # We must split p=6 into 2+2+2.
            
            # So for each j_block (2x2 A part), we have three 2x2 B parts (cols 0-1, 2-3, 4-5).
            # For each of these, we use Strassen (rank 7).
            # Total rank per j_block = 3 * 7 = 21.
            # Total rank = 2 * 21 = 42.
            
            pass # The loop above is incomplete in logic for the 2x6 case
            
    # Correct Implementation for 2x4x6 via 2x(2x2x2)x3
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for j_block in range(2): # j=0,1 and j=2,3
        for k_block in range(3): # k=0,1 and k=2,3 and k=4,5
            for prod_idx in range(7):
                r = col + prod_idx
                
                # U part: A[i, j]
                # i in {0,1}, j in {j_block*2, j_block*2+1}
                for local_i, sign in strassen_U[prod_idx].items():
                    # local_i 0->0, 1->1, 2->0, 3->1 ? No, Strassen U uses indices 0..3 for the 2x2 matrix.
                    # strassen_U maps local 2x2 indices to signs.
                    # local 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                    gi = local_i // 2
                    gj = local_i % 2
                    U[gi * m + (j_block * 2 + gj), r] = sign
                
                # V part: B[j, k]
                # j in {j_block*2, j_block*2+1}, k in {k_block*2, k_block*2+1}
                for local_jk, sign in strassen_V[prod_idx].items():
                    # local_jk 0..3 for the 2x2 matrix B
                    gj = local_jk // 2
                    gk = local_jk % 2
                    V[(j_block * 2 + gj) * p + (k_block * 2 + gk), r] = sign
                
                # W part: C[i, k]
                # i in {0,1}, k in {k_block*2, k_block*2+1}
                for local_ik, sign in strassen_W[prod_idx].items():
                    gi = local_ik // 2
                    gk = local_ik % 2
                    W[(k_block * 2 + gk) * n + gi, r] = sign
            
            col += 7
            
    if time.time() - start_time > 110:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
