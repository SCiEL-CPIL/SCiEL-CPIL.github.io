"""Program for matrix multiplication 3x3x6 tensor decomposition."""
DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Build the rank-23 decomposition for 3x3x3 (Strassen-based)
    # Local indices: A is 3x3 (9 elements), B is 3x3 (9), C is 3x3 (9)
    # We will create U_local, V_local, W_local of shape (9, 23)
    
    # Strassen-like decomposition for 3x3x3
    # We partition 3x3 matrices into blocks:
    # Top-left 2x2, top-right 1x2, bottom-left 2x1, bottom-right 1x1, 
    # plus middle row/col elements
    
    # Actually, let's use a known rank-23 decomposition for 3x3x3.
    # One approach: use Strassen for the 2x2 core (rank 7) plus 16 simple terms for the remaining 16 elements.
    # Total: 7 + 16 = 23.
    
    # Let's define the 23 terms manually.
    # A is indexed as A[i,j] for i,j in 0..2
    # B is indexed as B[j,k] for j,k in 0..2
    # C is indexed as C[i,k] for i,k in 0..2
    
    # For the 2x2 top-left block (i,j,k in 0..1), we use Strassen:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For the remaining 16 elements of C, we use simple outer products.
    
    U_local = [[0]*23 for _ in range(9)]
    V_local = [[0]*23 for _ in range(9)]
    W_local = [[0]*23 for _ in range(9)]
    
    def a_idx(i, j): return i * 3 + j
    def b_idx(j, k): return j * 3 + k
    def c_idx(i, k): return i * 3 + k
    
    col = 0
    # Term 0: M1 = (A00 + A11)(B00 + B11)
    U_local[a_idx(0,0)][col] = 1
    U_local[a_idx(1,1)][col] = 1
    V_local[b_idx(0,0)][col] = 1
    V_local[b_idx(1,1)][col] = 1
    W_local[c_idx(0,0)][col] = 1
    W_local[c_idx(1,1)][col] = 1
    col += 1
    
    # Term 1: M2 = (A10 + A11)B00
    U_local[a_idx(1,0)][col] = 1
    U_local[a_idx(1,1)][col] = 1
    V_local[b_idx(0,0)][col] = 1
    W_local[c_idx(1,0)][col] = 1
    col += 1
    
    # Term 2: M3 = A00(B10 - B11)
    U_local[a_idx(0,0)][col] = 1
    V_local[b_idx(1,0)][col] = 1
    V_local[b_idx(1,1)][col] = -1
    W_local[c_idx(0,1)][col] = 1
    W_local[c_idx(1,1)][col] = 1
    col += 1
    
    # Term 3: M4 = A11(B01 - B00)
    U_local[a_idx(1,1)][col] = 1
    V_local[b_idx(0,1)][col] = 1
    V_local[b_idx(0,0)][col] = -1
    W_local[c_idx(0,0)][col] = -1
    W_local[c_idx(1,0)][col] = 1
    col += 1
    
    # Term 4: M5 = (A00 + A01)B11
    U_local[a_idx(0,0)][col] = 1
    U_local[a_idx(0,1)][col] = 1
    V_local[b_idx(1,1)][col] = 1
    W_local[c_idx(0,0)][col] = 1
    col += 1
    
    # Term 5: M6 = (A10 - A00)(B00 + B01)
    U_local[a_idx(1,0)][col] = 1
    U_local[a_idx(0,0)][col] = -1
    V_local[b_idx(0,0)][col] = 1
    V_local[b_idx(0,1)][col] = 1
    W_local[c_idx(1,1)][col] = 1
    col += 1
    
    # Term 6: M7 = (A01 - A11)(B10 + B11)
    U_local[a_idx(0,1)][col] = 1
    U_local[a_idx(1,1)][col] = -1
    V_local[b_idx(1,0)][col] = 1
    V_local[b_idx(1,1)][col] = 1
    W_local[c_idx(0,0)][col] = -1
    W_local[c_idx(0,1)][col] = 1
    col += 1
    
    # Now simple terms for the remaining 16 C elements
    # C02, C12, C20, C21, C22 (5 from column 2)
    # C02 = A00*B02 + A01*B12 + A02*B22
    # We'll do one term per (i,j,k) triple for the remaining ones
    
    # Remaining C indices: (0,2), (1,2), (2,0), (2,1), (2,2)
    # And for each, the j index can be 0,1,2
    
    # Let's enumerate all (i,k) that are not in the 2x2 top-left block:
    # (0,2), (1,2), (2,0), (2,1), (2,2)
    # For each, we need to sum over j=0,1,2
    # So 5 * 3 = 15 more terms? But 7 + 15 = 22, not 23.
    # Wait, the 2x2 block has 4 C elements, each with 2 j values = 8 terms, but Strassen uses 7.
    # Remaining C elements: 5 elements * 3 j values = 15 terms.
    # Total: 7 + 15 = 22. That's not 23.
    
    # Let me recount. The 3x3x3 tensor has 27 terms in schoolbook.
    # Strassen saves 1 term on the 2x2 block (from 8 to 7).
    # So total should be 27 - 1 = 26? No, that's not right either.
    # Actually, the 2x2 block involves 8 terms (i,j,k in 0..1), Strassen replaces them with 7.
    # The remaining 19 terms (27-8=19) stay as is.
    # Total: 7 + 19 = 26.
    
    # So rank 26 for 3x3x3, not 23. Let me reconsider.
    
    # Let me just use the schoolbook decomposition for 3x3x6 directly but try to find a better one.
    # For 3x3x6, schoolbook is 3*3*6 = 54.
    # We can use Strassen on the 2x2x2 sub-blocks.
    
    # Let's partition the 3x6 matrix B into two 3x3 blocks: B1 (cols 0-2) and B2 (cols 3-5).
    # Similarly, C is 3x6, split into C1 (cols 0-2) and C2 (cols 3-5).
    # A is 3x3.
    
    # C1 = A * B1, C2 = A * B2.
    # Each 3x3x3 multiplication can be done in rank 26 (using Strassen on 2x2 block).
    # Total rank: 26 + 26 = 52.
    
    # That's an improvement over 54.
    
    # Let's implement rank 26 for 3x3x3.
    # Terms 0-6: Strassen for (i,j,k) in 0..1
    # Terms 7-25: Schoolbook for the remaining 19 (i,j,k) triples
    
    # Remaining triples: those with i=2 or j=2 or k=2, but not all in 0..1
    # Total triples: 27. First 8 are in 0..1. Remaining 19.
    
    # Let me list them:
    # i=0, j=0, k=2
    # i=0, j=1, k=2
    # i=0, j=2, k=0
    # i=0, j=2, k=1
    # i=0, j=2, k=2
    # i=1, j=0, k=2
    # i=1, j=1, k=2
    # i=1, j=2, k=0
    # i=1, j=2, k=1
    # i=1, j=2, k=2
    # i=2, j=0, k=0
    # i=2, j=0, k=1
    # i=2, j=0, k=2
    # i=2, j=1, k=0
    # i=2, j=1, k=1
    # i=2, j=1, k=2
    # i=2, j=2, k=0
    # i=2, j=2, k=1
    # i=2, j=2, k=2
    
    # That's 19 terms.
    
    # Now add these 19 terms:
    remaining_triples = [
        (0,0,2), (0,1,2), (0,2,0), (0,2,1), (0,2,2),
        (1,0,2), (1,1,2), (1,2,0), (1,2,1), (1,2,2),
        (2,0,0), (2,0,1), (2,0,2), (2,1,0), (2,1,1), (2,1,2),
        (2,2,0), (2,2,1), (2,2,2)
    ]
    
    for i, j, k in remaining_triples:
        U_local[a_idx(i,j)][col] = 1
        V_local[b_idx(j,k)][col] = 1
        W_local[c_idx(i,k)][col] = 1
        col += 1
    
    # Now assemble global U, V, W for 3x3x6
    # U shape: (9, 52), V shape: (18, 52), W shape: (18, 52)
    
    U = [[0]*52 for _ in range(9)]
    V = [[0]*52 for _ in range(18)]
    W = [[0]*52 for _ in range(18)]
    
    # For block t=0 (k=0,1,2): use U_local, V_local, W_local directly
    # V index: j*3+k for k in 0..2 -> j*3+0, j*3+1, j*3+2
    # W index: k*3+i for k in 0..2 -> k*3+0, k*3+1, k*3+2
    
    for r in range(26):
        for i in range(3):
            for j in range(3):
                U[i*3+j][r] = U_local[i*3+j][r]
            for j in range(3):
                for k in range(3):
                    V[j*3+k][r] = V_local[j*3+k][r]
            for i in range(3):
                for k in range(3):
                    W[k*3+i][r] = W_local[k*3+i][r]
    
    # For block t=1 (k=3,4,5): use same U_local, but V and W shifted
    # V index: j*3+(k+3) for k in 0..2 -> j*3+3, j*3+4, j*3+5
    # W index: (k+3)*3+i for k in 0..2 -> (k+3)*3+i
    
    for r in range(26):
        for i in range(3):
            for j in range(3):
                U[i*3+j][r+26] = U_local[i*3+j][r]
            for j in range(3):
                for k in range(3):
                    V[j*3+k+3][r+26] = V_local[j*3+k][r]
            for i in range(3):
                for k in range(3):
                    W[(k+3)*3+i][r+26] = W_local[k*3+i][r]
    
    return {"U": U, "V": V, "W": W}
