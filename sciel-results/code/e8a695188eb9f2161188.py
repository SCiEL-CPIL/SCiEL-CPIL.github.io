<<<<<<< SEARCH
def _constructive_low_rank(n, m, p):
    """
    Try to construct a low-rank decomposition using known techniques.
    For 3x4x5, we can try various block decompositions.
    """
    # Strategy: Use the fact that rank(3x4x5) <= rank(2x2x2) + rank(2x2x3) + ...
    # Let's try a systematic approach using known bounds.
    
    # One approach: decompose using Strassen-like patterns on sub-blocks
    # A is 3x4, B is 4x5
    # Split rows of A: [1, 2] and columns of B: [2, 3]
    # C = A*B where A = [A1; A2] (1x4, 2x4) and B = [B1, B2] (4x2, 4x3)
    # C11 = A1*B1 (1x2), C12 = A1*B2 (1x3)
    # C21 = A2*B1 (2x2), C22 = A2*B2 (2x3)
    
    # Rank contributions:
    # C11: 1x2x1 tensor -> rank 1 (outer product)
    # C12: 1x3x1 tensor -> rank 1
    # C21: 2x2x2 tensor -> rank 7 (Strassen)
    # C22: 2x3x2 tensor -> need to find rank
    
    # Total so far: 1 + 1 + 7 = 9, plus rank of 2x3x2
    # Known: rank(2x3x2) = rank(2x2x3) by symmetry
    # rank(2x2x3) is at most 10, possibly 9
    
    # Let's try to construct a rank-9 or rank-10 decomposition for 2x2x3
    # Then total would be 9 + 9 = 18 or 9 + 10 = 19
    
    # For now, return None and use random search
    return None
=======
def _constructive_low_rank(n, m, p):
    """
    Construct a low-rank decomposition for 3x4x5 using slicing and embedding.
    
    Strategy:
    - Split A (3x4) into A1 (1x4) and A2 (2x4)
    - C1 = A1 * B (1x5): rank 1 decomposition
    - C2 = A2 * B (2x5): construct explicit decomposition
    
    For C2 (2x4x5):
    - Split B (4x5) into B1 (4x2) and B2 (4x3)
    - C21 = A2 * B1 (2x2): use Strassen rank-7
    - C22 = A2 * B2 (2x3): construct rank-6 decomposition
    
    Total: 1 + 7 + 6 = 14? Let's verify.
    """
    if (n, m, p) != (3, 4, 5):
        return None
    
    # We'll build the full 3x4x5 decomposition directly
    # U has shape (3*4, rank) = (12, rank)
    # V has shape (4*5, rank) = (20, rank)  
    # W has shape (5*3, rank) = (15, rank)
    
    # Strategy: Build using block structure
    # Rows of A: indices 0,1,2 (n=3)
    # Cols of A / Rows of B: indices 0,1,2,3 (m=4)
    # Cols of B: indices 0,1,2,3,4 (p=5)
    
    # U row index: i*m + j = i*4 + j for i in 0..2, j in 0..3
    # V row index: j*p + k = j*5 + k for j in 0..3, k in 0..4
    # W row index: k*n + i = k*3 + i for k in 0..4, i in 0..2
    
    rank = 20
    U = [[0] * rank for _ in range(12)]
    V = [[0] * rank for _ in range(20)]
    W = [[0] * rank for _ in range(15)]
    
    col = 0
    
    # Block 1: A1 (row 0) * B (all cols) -> C row 0, all 5 cols
    # T[i=0, j, k] = 1 for j in 0..3, k in 0..4
    # This is rank 1: U[0*4+j][c]=1, V[j*5+k][c]=1, W[k*3+0][c]=1
    # But that's not right - we need sum over j of A[0,j]*B[j,k] = C[0,k]
    # The tensor is T[i,j,k] = 1 if i==0 else 0
    # Rank 1: single term with U[i*4+j] = 1 if i==0, V[j*5+k]=1, W[k*3+i]=1 if i==0
    # Actually the tensor T[i,j,k] for 1x4x5 is: T[0,j,k]=1 for all j,k
    # This factors as (vector of 1s in j) outer (vector of 1s in k) with i fixed
    # So rank 1: U[0*4+j][c] = 1 for all j, V[j*5+k][c] = 1 for all j,k... no that doesn't work
    # 
    # Wait, the tensor for matrix mult is T[i,j,k] = delta_{i fixed} * 1 * 1
    # For 1xm x px1: T[0,j,k] = 1
    # This is the outer product of vec(1_j) and vec(1_k), so rank 1
    # U[0*m+j][c] = 1 for all j (since i=0)
    # V[j*p+k][c] = 1 for all j,k -- NO, this doesn't work because V needs to be separable
    #
    # Actually for rank 1: U[i*m+j] * V[j*p+k] * W[k*n+i] = T[i,j,k]
    # For T[0,j,k] = 1: we need U[0*m+j]*V[j*p+k]*W[k*n+0] = 1
    # Set U[0*m+j] = 1, W[k*n+0] = 1, then V[j*p+k] = 1 for all j,k
    # But V[j*p+k] = 1 for all j,k means V is all 1s in this column
    # Check: sum over c of U[i*m+j]*V[j*p+k]*W[k*n+i] should equal T[i,j,k]
    # For i=0: U[0*m+j]=1, V[j*p+k]=1, W[k*n+0]=1, product=1, sum=1=T[0,j,k] ✓
    # For i=1,2: U[i*m+j]=0 (we set them to 0), so product=0, sum=0=T[i,j,k] ✓
    # Great, rank 1 works!
    
    for j in range(4):
        U[0*4+j][col] = 1
    for j in range(4):
        for k in range(5):
            V[j*5+k][col] = 1
    for k in range(5):
        W[k*3+0][col] = 1
    col += 1
    
    # Block 2: A2 (rows 1,2) * B1 (cols 0,1) -> C[1:3, 0:2] (2x2)
    # Use Strassen rank-7 for 2x2x2
    # A2 is 2x4, B1 is 4x2, C21 is 2x2
    # This is NOT 2x2x2 - it's 2x4x2
    # T[i,j,k] = 1 for i in {1,2}, j in {0,1,2,3}, k in {0,1}
    # i local: 0,1; j: 0,1,2,3; k local: 0,1
    # This is rank(2x4x2) = rank(2x2x4) by symmetry
    # 
    # For 2x4x2: we can split j into pairs
    # j = 0,1 and j = 2,3
    # C21 = A2[:,0:2]*B1[0:2,0:2] + A2[:,2:4]*B1[2:4,0:2]
    # Each is 2x2x2, rank 7 each, total 14
    # But we can do better by sharing
    #
    # Actually, let's just use the schoolbook for this block: rank 2*4*2 = 16
    # Or use a smarter decomposition
    #
    # Let's use: for each j in 0..3, k in 0..1, i in 1..2:
    # That's 2*4*2 = 16 terms (schoolbook)
    # 
    # Better: rank(2x4x2). The tensor T[i,j,k] = 1 for i in {0,1} (local), j in {0,1,2,3}, k in {0,1}
    # This is the sum over j of (e_i * e_j * (e_k0 + e_k1))... no
    # T = sum_{j=0}^{3} (vec_2(1) ⊗ e_j ⊗ vec_2(1))
    # Each term is rank 1, so rank ≤ 4
    # Let's verify: T[i,j,k] = sum_{c=0}^{3} U_c[i,j] * V_c[j,k] * W_c[k,i]
    # Set for c=0..3: U_c[i,j] = delta_{j,c} * 1 (for i in {0,1} local)
    # V_c[j,k] = delta_{j,c} * 1 (for k in {0,1})
    # W_c[k,i] = 1 (for all k,i)
    # Then T[i,j,k] = sum_c delta_{j,c}*1 * delta_{j,c}*1 * 1 = 1 if j is in {0,1,2,3} ✓
    # So rank 4!
    
    # Local indices: i_local in {0,1} maps to global i in {1,2}
    # j in {0,1,2,3}, k_local in {0,1} maps to global k in {0,1}
    for c_local in range(4):
        # U: for i_global in {1,2}, j=c_local
        for i_local in range(2):
            i_global = 1 + i_local
            U[i_global*4 + c_local][col] = 1
        # V: j=c_local, k in {0,1}
        for k_local in range(2):
            k_global = k_local
            V[c_local*5 + k_global][col] = 1
        # W: k in {0,1}, i_global in {1,2}
        for k_local in range(2):
            k_global = k_local
            for i_local in range(2):
                i_global = 1 + i_local
                W[k_global*3 + i_global][col] = 1
        col += 1
    
    # Block 3: A2 (rows 1,2) * B2 (cols 2,3,4) -> C[1:3, 2:5] (2x3)
    # 2x4x3 tensor: T[i,j,k] = 1 for i in {1,2}, j in {0,1,2,3}, k in {2,3,4}
    # Local: i in {0,1}, j in {0,1,2,3}, k in {0,1,2}
    # T = sum_{j=0}^{3} (vec_2(1) ⊗ e_j ⊗ vec_3(1))
    # Rank ≤ 4
    for c_local in range(4):
        for i_local in range(2):
            i_global = 1 + i_local
            U[i_global*4 + c_local][col] = 1
        for k_local in range(3):
            k_global = 2 + k_local
            V[c_local*5 + k_global][col] = 1
        for k_local in range(3):
            k_global = 2 + k_local
            for i_local in range(2):
                i_global = 1 + i_local
                W[k_global*3 + i_global][col] = 1
        col += 1
    
    # Total rank used: 1 + 4 + 4 = 9
    # But wait, we have rank=20 allocated. Let's check if this is correct.
    # The issue: in Block 1, V[j*5+k] = 1 for ALL j,k. In Block 2 and 3, V also has entries.
    # These are in different columns, so no conflict.
    
    # Let's verify the total rank used
    assert col == 9, f"Expected rank 9, got {col}"
    
    U_out = [row[:col] for row in U]
    V_out = [row[:col] for row in V]
    W_out = [row[:col] for row in W]
    
    if _verify(U_out, V_out, W_out, n, m, p, tol=1e-6):
        return U_out, V_out, W_out
    
    return None
>>>>>>> REPLACE

Now let me update the `construct` function to use this constructive approach first:

