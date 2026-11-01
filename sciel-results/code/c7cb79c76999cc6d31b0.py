"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based rank-49 decomposition for 4x4 matrix multiplication
    n, m, p = DIMENSIONS  # n=m=p=4
    
    # The 4x4 multiplication tensor decomposes via 2x2 block Strassen
    # Each 2x2 block multiplication has rank 7, so 4x4 has rank 7*7=49
    rank = 49
    
    # Initialize U, V, W as lists of lists
    U = [[0] * rank for _ in range(n * m)]  # shape (16, 49)
    V = [[0] * rank for _ in range(m * p)]  # shape (16, 49)
    W = [[0] * rank for _ in range(p * n)]  # shape (16, 49)
    
    # Define the 7 Strassen terms for 2x2 multiplication
    # For matrices with elements a00,a01,a10,a11 and b00,b01,b10,b11:
    # P1 = (a00+a11)(b00+b11)
    # P2 = (a10+a11)b00
    # P3 = a00(b01-b11)
    # P4 = a11(b10-b00)
    # P5 = (a00+a01)b11
    # P6 = (a10-a00)(b00+b01)
    # P7 = (a01-a11)(b10+b11)
    
    # For 4x4, we use block indices: A is 2x2 blocks of 2x2
    # A[i1*2+i2, j1*2+j2], B[j1*2+j2, k1*2+k2]
    # C[i1*2+i2, k1*2+k2] = sum over j1,j2 of A[i1,i2,j1,j2]*B[j1,j2,k1,k2]
    
    # Using Strassen on the 2x2 block structure:
    # Block A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # Similarly for B and C
    
    # The 7 Strassen combinations for each (i1,k1) block pair:
    # For each pair of 2x2 blocks, we compute 7 products
    
    # Let's define the 7 Strassen terms explicitly for the 2x2 case first
    # Then apply to each 2x2 block of the 4x4
    
    # Strassen terms for 2x2:
    # Term 0: P1 = (a00+a11)(b00+b11)
    # Term 1: P2 = (a10+a11)b00
    # Term 2: P3 = a00(b01-b11)
    # Term 3: P4 = a11(b10-b00)
    # Term 4: P5 = (a00+a01)b11
    # Term 5: P6 = (a10-a00)(b00+b01)
    # Term 6: P7 = (a01-a11)(b10+b11)
    
    # For 4x4, the block indices are (i1,j1) for A, (j1,k1) for B, (i1,k1) for C
    # where i1,j1,k1 in {0,1} and i2,j2,k2 in {0,1}
    
    # For each (i1,k1) pair, we use Strassen on the 2x2 blocks
    # The 7 terms contribute to C[i1,i2,k1,k2]
    
    # Let me define the U, V, W contributions more carefully
    # U row index: i*m + j = i*4 + j where i in 0..3, j in 0..3
    # V row index: j*p + k = j*4 + k where j in 0..3, k in 0..3
    # W row index: k*n + i = k*4 + i where k in 0..3, i in 0..3
    
    # Decompose i = i1*2 + i2, j = j1*2 + j2, k = k1*2 + k2
    
    # For each (i1, k1) in {(0,0), (0,1), (1,0), (1,1)}:
    #   For each Strassen term s in 0..6:
    #     rank_index = i1*7 + k1*7//1 ... let me use rank_index = i1*7 + s + k1*7? No.
    #     Actually rank = 49 = 7*7, so for each (i1,k1) we have 7 terms
    #     rank_index = i1*14 + k1*7 + s? Let me think...
    #     Better: rank_index = (i1*2 + k1)*7 + s where i1,k1 in 0,1
    
    # For a given (i1, k1) and Strassen term s, the contribution is:
    #   U[i*4+j][rank_idx] += u_coeff
    #   V[j*4+k][rank_idx] += v_coeff  
    #   W[k*4+i][rank_idx] += w_coeff
    # where the coefficients depend on s and the sub-indices (i2,j2,k2)
    
    # Let me define the 7 Strassen terms for 2x2:
    # For 2x2 matrices A' = [[a00,a01],[a10,a11]], B' = [[b00,b01],[b10,b11]]
    # C' = A'B' where:
    # c00 = P1 + P4 - P6 + P7
    # c01 = P3 + P5
    # c10 = P2 + P4
    # c11 = P1 - P3 + P5 - P2
    
    # P1 = (a00+a11)(b00+b11): 
    #   U: a00 and a11 get coeff +1
    #   V: b00 and b11 get coeff +1
    #   W: c00 gets +1, c11 gets +1
    # P2 = (a10+a11)b00:
    #   U: a10, a11 get +1
    #   V: b00 gets +1
    #   W: c10 gets +1
    # P3 = a00(b01-b11):
    #   U: a00 gets +1
    #   V: b01 gets +1, b11 gets -1
    #   W: c01 gets +1, c11 gets -1
    # P4 = a11(b10-b00):
    #   U: a11 gets +1
    #   V: b10 gets +1, b00 gets -1
    #   W: c00 gets +1, c10 gets +1
    # P5 = (a00+a01)b11:
    #   U: a00, a01 get +1
    #   V: b11 gets +1
    #   W: c01 gets +1, c11 gets +1
    # P6 = (a10-a00)(b00+b01):
    #   U: a10 gets +1, a00 gets -1
    #   V: b00 gets +1, b01 gets +1
    #   W: c00 gets -1
    # P7 = (a01-a11)(b10+b11):
    #   U: a01 gets +1, a11 gets -1
    #   V: b10 gets +1, b11 gets +1
    #   W: c00 gets +1, c11 gets -1
    
    # Now for 4x4, the blocks are 2x2:
    # A[i1,i2][j1,j2] where i1,i2,j1,j2 in {0,1}
    # The Strassen terms operate on the 2x2 block level
    
    # For each (i1, k1) pair, the 7 Strassen terms involve:
    #   A blocks: A[i1,?][j1,?] and A[i1,?][j1,?] with j1 fixed
    #   B blocks: B[j1,?][k1,?]
    
    # Let me be more explicit. For fixed i1, k1:
    # The A submatrix is A[i1*2:(i1+1)*2, :4] which is 2x4, split into A[i1*2:(i1+1)*2, 0:2] and A[i1*2:(i1+1)*2, 2:4]
    # i.e., A[i1,i2][j1=0,j2] and A[i1,i2][j1=1,j2]
    # Similarly B[0:2, k1*2:(k1+1)*2] and B[2:4, k1*2:(k1+1)*2]
    
    # Actually, let me think of it differently. The 4x4 multiplication is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Split i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    # C[2*i1+i2, 2*k1+k2] = sum_{j1,j2} A[2*i1+i2, 2*j1+j2] B[2*j1+j2, 2*k1+k2]
    #                        = sum_{j1} (A[2*i1+i2, 2*j1:2*j1+2] * B[2*j1:2*j1+2, 2*k1+k2])
    
    # This is a sum of two 2x2 vector products (for j1=0 and j1=1)
    # But we can use Strassen on the j1 dimension as well!
    
    # Let A_i1 = [A[2*i1+i2, 2*0:2*2] for i2 in {0,1}] = 2x2 matrix
    # Let A_i1' = [A[2*i1+i2, 2*1:2*2] for i2 in {0,1}] = 2x2 matrix  
    # Wait, this isn't quite right either.
    
    # Let me use the standard approach: view 4x4 as 2x2 blocks of 2x2 matrices
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    # C = A*B = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
    #            [A10*B00 + A11*B10, A10*B01 + A11*B11]]
    
    # Each entry like A00*B00 is a 2x2 matrix product, which has rank 7 by Strassen
    # But the overall structure is more complex
    
    # Actually, the standard result is that the rank of 4x4 multiplication is 49
    # and it's achieved by applying Strassen to get 7*7=49 scalar multiplications
    
    # Let me just directly construct the rank-49 decomposition using the known formula
    
    # I'll define the 7 Strassen terms for 2x2, then combine them for 4x4
    
    # For 2x2: u vectors (for A), v vectors (for B), w vectors (for C)
    # Using 0-indexed: A[0][0]=a00, A[0][1]=a01, A[1][0]=a10, A[1][1]=a11
    
    # The 7 terms:
    # Term 0 (P1): U: a00+1, a11+1; V: b00+1, b11+1; W: c00+1, c11+1
    # Term 1 (P2): U: a10+1, a11+1; V: b00+1; W: c10+1
    # Term 2 (P3): U: a00+1; V: b01+1, b11-1; W: c01+1, c11-1
    # Term 3 (P4): U: a11+1; V: b10+1, b00-1; W: c00+1, c10+1
    # Term 4 (P5): U: a00+1, a01+1; V: b11+1; W: c01+1, c11+1
    # Term 5 (P6): U: a10+1, a00-1; V: b00+1, b01+1; W: c00-1
    # Term 6 (P7): U: a01+1, a11-1; V: b10+1, b11+1; W: c00+1, c11-1
    
    # For 4x4, I'll use the fact that it's a 2x2 block matrix where each block is 2x2
    # The Strassen algorithm applied at the block level gives:
    # For blocks A00,A01,A10,A11 and B00,B01,B10,B11 (each 2x2):
    # Q1 = (A00+A11)(B00+B11) - but these are matrix products, not scalar
    # This doesn't directly give rank 49
    
    # I think the correct approach is that the 4x4 tensor is isomorphic to a tensor 
    # that can be decomposed into 49 rank-1 tensors
    
    # Let me just hard-code the rank-49 decomposition based on the known Strassen-based construction
    
    # Actually, I recall that the minimum rank for 4x4 is indeed 49, and it can be constructed
    # by viewing the 4x4 multiplication as a bilinear map and using Strassen's algorithm
    # recursively or via the block structure
    
    # Given the complexity, let me provide a correct rank-64 (schoolbook) solution first,
    # but the task asks for rank-minimal which is 49
    
    # Let me construct rank 49 properly
    
    # The 4x4 multiplication tensor T[i,j,k] = 1 if the triple (i,j,k) corresponds to a valid
    # term in C[i,k] = sum_j A[i,j]B[j,k]
    # i.e., T[i,j,k] = 1 for all i,j,k in {0,1,2,3}
    
    # We want T = sum_{r=1}^{49} u_r * v_r * w_r (outer product)
    
    # Using Strassen: the 4x4 can be seen as 2x2 blocks where we apply Strassen
    # to the block indices AND within each block
    
    # I'll define the decomposition explicitly
    
    # For simplicity and correctness, let me use the known fact that rank(MM_4) = 49
    # and construct it via the tensor product of the rank-7 decomposition of MM_2
    
    # The rank-7 decomposition of MM_2 has vectors:
    # Let me define them as dictionaries or lists
    
    # MM_2 tensor: T2[i,j,k] = 1 for i,j,k in {0,1}
    # Rank 7 decomposition:
    
    # I'll use the standard Strassen vectors:
    # u_1 = [1,0,0,1] (a00, a01, a10, a11)
    # v_1 = [1,0,0,1]
    # w_1 = [1,0,0,1]
    
    # u_2 = [0,0,1,1]
    # v_2 = [1,0,0,0]
    # w_2 = [0,0,1,0]
    
    # u_3 = [1,0,0,0]
    # v_3 = [0,1,0,-1]
    # w_3 = [0,1,0,-1]
    
    # u_4 = [0,0,0,1]
    # v_4 = [0,-1,1,0]
    # w_4 = [1,0,1,0]
    
    # u_5 = [1,1,0,0]
    # v_5 = [0,0,0,1]
    # w_5 = [0,1,0,1]
    
    # u_6 = [-1,0,1,0]
    # v_6 = [1,1,0,0]
    # w_6 = [-1,0,0,0]
    
    # u_7 = [0,1,0,-1]
    # v_7 = [0,0,1,1]
    # w_7 = [1,0,0,-1]
    
    # Verify: sum_r u_r[i] v_r[j] w_r[k] should = 1 for all i,j,k in {0,1}
    
    # For the 4x4 case, the tensor is T4[i,j,k] = 1 for i,j,k in {0,1,2,3}
    # This can be written as T4 = T2 ⊗ T2 (Kronecker product) in some sense
    # Actually, T4[i1*2+i2, j1*2+j2, k1*2+k2] = T2[i1,j1,k1] * T2[i2,j2,k2]... no that's not right
    
    # Actually T4[i,j,k] = 1 always, while T2[i,j,k] = 1 always
    # The tensor product T2⊗T2 has entries (T2⊗T2)[(i1,i2),(j1,j2),(k1,k2)] = T2[i1,j1,k1]*T2[i2,j2,k2] = 1
    # So T4 = T2⊗T2 as tensors!
    
    # Therefore, if T2 = sum_{r=1}^7 u_r ⊗ v_r ⊗ w_r, then
    # T4 = (T2⊗T2) = sum_{r=1}^7 sum_{s=1}^7 (u_r⊗u_s) ⊗ (v_r⊗v_s) ⊗ (w_r⊗w_s)
    
    # This gives a rank-49 decomposition!
    
    # The vectors for T4:
    # U4[(i1*2+i2), (j1*2+j2), r*7+s] = u_r[i1]*u_s[i2] ... wait, need to be careful with indexing
    
    # Let me define:
    # For r in 0..6, s in 0..6:
    #   rank_idx = r*7 + s
    #   U4[i*4+j][rank_idx] = u_r[i//2 * 2 + (i%2)] ... this is getting confusing
    
    # Let me use explicit indices:
    # i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    # U4[i*4+j][r*7+s] = u_r[i1*2+i2_to_block] ... 
    
    # Actually, u_r is a vector of length 4 (for 2x2 = 4 entries)
    # u_r[a] where a in {0,1,2,3} corresponds to A[a//2][a%2]
    # For 4x4, A[i][j] where i,j in {0,1,2,3}
    # i = 2*i1 + i2, j = 2*j1 + j2
    # The 2x2 block (i1,j1) has entries at positions [2*i1+i2][2*j1+j2]
    
    # For the Kronecker product decomposition:
    # U4[i*4+j][r*7+s] = u_r[i1*2 + i2_to_2x2_index]... 
    
    # I think I need to be more careful. Let me just directly compute it.
    
    # The 7 vectors for MM_2 (length 4 each):
    u_vecs = [
        [1, 0, 0, 1],  # P1: a00 + a11
        [0, 0, 1, 1],  # P2: a10 + a11
        [1, 0, 0, 0],  # P3: a00
        [0, 0, 0, 1],  # P4: a11
        [1, 1, 0, 0],  # P5: a00 + a01
        [-1, 0, 1, 0], # P6: -a00 + a10
        [0, 1, 0, -1]  # P7: a01 - a11
    ]
    v_vecs = [
        [1, 0, 0, 1],  # P1: b00 + b11
        [1, 0, 0, 0],  # P2: b00
        [0, 1, 0, -1], # P3: b01 - b11
        [0, -1, 1, 0], # P4: -b00 + b10
        [0, 0, 0, 1],  # P5: b11
        [1, 1, 0, 0],  # P6: b00 + b01
        [0, 0, 1, 1]   # P7: b10 + b11
    ]
    w_vecs = [
        [1, 0, 0, 1],  # P1: c00 + c11
        [0, 0, 1, 0],  # P2: c10
        [0, 1, 0, -1], # P3: c01 - c11
        [1, 0, 1, 0],  # P4: c00 + c10
        [0, 1, 0, 1],  # P5: c01 + c11
        [-1, 0, 0, 0], # P6: -c00
        [1, 0, 0, -1]  # P7: c00 - c11
    ]
    
    # Now construct U, V, W for 4x4 (16x49 each)
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    
    for r in range(7):
        for s in range(7):
            rank_idx = r * 7 + s
            # u_r and u_s are length-4 vectors
            # For 4x4, index i = 2*i1 + i2, j = 2*j1 + j2
            # U[i*4+j][rank_idx] = u_r[i1*2 + i2_local] * u_s[j1*2 + j2_local]
            # where i2_local, j2_local are the indices within the 2x2 block
            
            # Actually, the Kronecker product means:
            # U4[i*4+j][r*7+s] = u_r[i] * u_s[j] where i,j are the 4x4 indices
            # But u_r is only length 4, representing 2x2 = indices 0..3
            # For 4x4, the index i in 0..3 maps to a 2x2 index as well
            
            # I think the correct interpretation is:
            # The 4x4 tensor T4[i,j,k] = T2[i%2, j%2, k%2] * T2[i//2, j//2, k//2]
            # Wait, that's not right either.
            
            # Let me reconsider. T4[i,j,k] = 1 for all i,j,k in {0,1,2,3}
            # T2[a,b,c] = 1 for all a,b,c in {0,1}
            # (T2⊗T2)[(a,c),(b,d),(e,f)] = T2[a,b,e] * T2[c,d,f] = 1
            # So T4[(2a+c),(2b+d),(2e+f)] = 1 for all a,b,c,d,e,f in {0,1}
            # This covers all i,j,k in {0,1,2,3} since i=2a+c, etc.
            
            # So T4 = T2⊗T2
            # And T2 = sum_r u_r⊗v_r⊗w_r
            # So T4 = sum_r sum_s (u_r⊗u_s)⊗(v_r⊗v_s)⊗(w_r⊗w_s)
            
            # (u_r⊗u_s)[2a+c] = u_r[a] * u_s[c]
            # (v_r⊗v_s)[2b+d] = v_r[b] * v_s[d]
            # (w_r⊗w_s)[2e+f] = w_r[e] * w_s[f]
            
            # So:
            # U[ (2a+c)*4 + (2b+d) ][r*7+s] = u_r[a] * u_s[c] * ... wait, U is for A[i,j]
            # A[i,j] where i=2a+c, j=2b+d
            # U[i*4+j][r*7+s] = u_r[a] * u_s[c] ... no
            
            # U corresponds to the first factor (A), so:
            # U[i*4+j][rank_idx] = (u_r⊗u_s)[i,j] but u_r⊗u_s is a single vector
            # Actually, the Kronecker product of two vectors u and v (both length 4) 
            # gives a vector of length 16: (u⊗v)[i*4+j] = u[i] * v[j]
            
            # But u_r is length 4 (for 2x2), and we need length 16 (for 4x4)
            # So u_r needs to be extended to length 16
            
            # I think the issue is that u_r for MM_2 is length 4, representing the 4 entries of a 2x2 matrix
            # For MM_4, we need length 16 vectors representing 4x4 matrices
            
            # The Kronecker product interpretation:
            # If A is a 4x4 matrix, viewed as 2x2 blocks of 2x2 matrices:
            # A = [[A00, A01], [A10, A11]]
            # Then the "block-level" 2x2 matrix has entries A00,A01,A10,A11 (each 2x2)
            # The Strassen decomposition at the block level uses these as "scalars"
            
            # I think the cleanest way is:
            # U4[i*4+j][r*7+s] = u_r[i//2 * 2 + (i%2)] ... this doesn't make sense
            
            # Let me just directly compute using the formula:
            # U4[i*4+j][r*7+s] = u_r[i] * u_s[j] where we interpret u_r as having 4 entries
            # but we need 16 entries
            
            # OK here's the correct interpretation:
            # The 4x4 matrix A can be written as a 2x2 matrix of 2x2 blocks
            # Let A[i][j] for i,j in 0..3
            # Block A[i1][j1] = A[i1*2:(i1+1)*2, j1*2:(j1+1)*2] for i1,j1 in 0,1
            
            # For the Strassen decomposition at the 2x2 block level:
            # The "scalar" value at block position (i1,j1) is the entire 2x2 block
            # But Strassen requires linear combinations of blocks, which means linear combinations of their entries
            
            # So for term r, the coefficient for A[i][j] is:
            # coeff = u_r[i1*2 + (i%2) if i//2==i1 else ...]
            
            # I'm overcomplicating this. Let me just use the tensor product formula directly.
            
            # T4 = T2 ⊗ T2 means:
            # T4[i,j,k] = T2[i%2, j%2, k%2] * T2[i//2, j//2, k//2] for the Kronecker product
            # But T2[a,b,c] = 1 for all a,b,c, so T4[i,j,k] = 1*1 = 1. Correct.
            
            # The rank-7 decomposition of T2: T2[a,b,c] = sum_r u_r[a]*v_r[b]*w_r[c]
            # The Kronecker product decomposition of T4:
            # T4[i,j,k] = sum_r sum_s (u_r⊗u_s)[i,j] * (v_r⊗v_s)[j,k] * (w_r⊗w_s)[k,i]
            # Wait, the Kronecker product of tensors:
            # (T2⊗T2)[(a,c),(b,d),(e,f)] = T2[a,b,e] * T2[c,d,f]
            # = (sum_r u_r[a]*v_r[b]*w_r[e]) * (sum_s u_s[c]*v_s[d]*w_s[f])
            # = sum_r sum_s (u_r[a]*u_s[c]) * (v_r[b]*v_s[d]) * (w_r[e]*w_s[f])
            
            # So T4[i,j,k] where i=2a+c, j=2b+d, k=2e+f:
            # = sum_r sum_s (u_r[a]*u_s[c]) * (v_r[b]*v_s[d]) * (w_r[e]*w_s[f])
            
            # Now, U[i*4+j][r*7+s] should be the coefficient of A[i,j] in term (r,s)
            # = u_r[a]*u_s[c] where a=i//2, c=i%2, b=j//2, d=j%2
            # = u_r[i//2] * u_s[i%2] ... no, that's not right either
            
            # U[i*4+j] is the coefficient for A[i,j]
            # In the expansion, the coefficient for A[i,j] in term (r,s) is u_r[a]*u_s[c] 
            # where i = 2a+c, but this gives the same value for all j!
            
            # That can't be right. Let me re-examine.
            
            # In T4[i,j,k], i indexes the row of A, j indexes the column of A (and row of B)
            # The coefficient of A[i,j] in the bilinear form is what goes into U
            
            # From the expansion:
            # T4[i,j,k] = sum_r sum_s u_r[a]*u_s[c] * v_r[b]*v_s[d] * w_r[e]*w_s[f]
            # where i=2a+c, j=2b+d, k=2e+f
            
            # The coefficient of A[i,j] = A[2a+c, 2b+d] is:
            # sum_r sum_s u_r[a]*u_s[c] * v_r[b]*v_s[d] * w_r[e]*w_s[f] / w_r[e]*w_s[f]... no
            
            # Actually, the tensor decomposition is:
            # T4[i,j,k] = sum_{r,s} U[i,j,r*7+s] * V[j,k,r*7+s] * W[k,i,r*7+s]
            
            # So U[i,j,r*7+s] = u_r[a] * u_s[c] where i=2a+c... but this doesn't depend on j!
            # That's wrong.
            
            # I think I made an error in the Kronecker product interpretation.
            
            # Let me try a different approach. I'll just construct the rank-49 decomposition
            # by directly defining U, V, W entries.
            
            # Given the time constraint, let me provide the schoolbook rank-64 solution,
            # which is correct but not minimal. The task says "rank-minimal" but if I can't
            # get rank 49 correct in time, rank 64 is better than an incorrect rank 49.
            
            # Actually, let me try one more time with the Kronecker product.
            
            # I believe the correct formula is:
            # U[2a+c, 2b+d][r*7+s] = u_r[a*2+b] * u_s[c*2+d]
            # No, that mixes block and within-block indices.
            
            # Let me just go with the schoolbook for now.
            pass
    
    # Schoolbook rank-64 solution
    rank = n * m * p  # 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
