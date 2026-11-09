"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 5, 5, 5
    nm = n * m  # 25
    mp = m * p  # 25
    pn = p * n  # 25
    
    # Verify decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1
    def _verify(U, V, W, tol=1e-9):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Build decomposition using 2+2+1 block partition with Strassen on 2x2x2 blocks
    # Partition: indices 0,1 | 2,3 | 4
    # For 2x2x2 blocks, use Strassen (rank 7 instead of 8)
    # For 1x1x1 blocks, use schoolbook (rank 1)
    
    # Block structure:
    # A is 5x5, B is 5x5, C is 5x5
    # A partitioned into blocks: A[0:2,0:2], A[0:2,2:4], A[0:2,4:5],
    #                           A[2:4,0:2], A[2:4,2:4], A[2:4,4:5],
    #                           A[4:5,0:2], A[4:5,2:4], A[4:5,4:5]
    # Similarly for B and C
    
    # Strassen for 2x2x2:
    # C = A * B where A, B, C are 2x2
    # Let A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]], C = [[c00,c01],[c10,c11]]
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b10-b11)
    # M4 = a11*(b01-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b10)
    # M7 = (a01-a11)*(b01+b11)
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # For our tensor decomposition, we need to express each product a_ij * b_jk 
    # as part of the Strassen factors.
    
    # The key insight: for a 2x2x2 block, instead of 8 rank-1 terms,
    # we use 7 rank-1 terms from Strassen.
    
    # Let's build the full decomposition:
    # Blocks: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)
    # where block (i,j) refers to rows i, cols j in the 3x3 block grid
    # Block sizes: 2,2,1 for rows; 2,2,1 for cols
    
    # For block (i,j) in A (rows i, cols j) times block (j,k) in B (rows j, cols k):
    # - If both blocks are 2x2: use Strassen (7 terms)
    # - If one is 2x1 or 1x2: need special handling
    # - If both are 1x1: 1 term
    
    # Let's think about this more carefully.
    # A is partitioned as:
    #   A = [[A00, A01, A02],
    #        [A10, A11, A12],
    #        [A20, A21, A22]]
    # where A00, A01, A10, A11 are 2x2, A02, A12, A20, A21 are 2x1, A22 is 1x1
    # Similarly for B.
    
    # C = A * B = sum_k A_ik * B_kj
    # C00 = A00*B00 + A01*B10 + A02*B20
    # C01 = A00*B01 + A01*B11 + A02*B21
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C10 = A10*B00 + A11*B10 + A12*B20
    # C11 = A10*B01 + A11*B11 + A12*B21
    # C12 = A10*B02 + A11*B12 + A12*B22
    # C20 = A20*B00 + A21*B10 + A22*B20
    # C21 = A20*B01 + A21*B11 + A22*B21
    # C22 = A20*B02 + A21*B12 + A22*B22
    
    # For 2x2 * 2x2: Strassen gives 7 terms
    # For 2x2 * 2x1: This is like multiplying 2x2 by a 2x1 matrix
    #   Let A be 2x2, B be 2x1 (column vector b)
    #   C = A * b = [a00*b0 + a01*b1, a10*b0 + a11*b1]
    #   We can use a variant of Strassen or just 4 terms (schoolbook)
    #   Actually, for 2x2 * 2x1, we need 4 rank-1 terms in the tensor sense
    #   But wait, the tensor structure is different.
    
    # Let me reconsider. The matrix multiplication tensor for 5x5x5 has
    # entries T[i,j,k] = 1 if the product a_ij * b_jk contributes to c_ki.
    
    # For a block A_ik (size r_i x c_k) times B_kj (size r_k x c_j):
    # The product contributes to C_ij (size r_i x c_j)
    # Each element (a, b) in A_ik times element (b, c) in B_kj gives element (a, c) in C_ij
    # So for block A_ik of size (r_i x s_k) and B_kj of size (s_k x c_j):
    # Number of rank-1 terms = r_i * s_k * c_j (schoolbook)
    # With Strassen on 2x2x2 subblocks, we can reduce this.
    
    # Let's use a simpler approach: decompose into 2x2x2 and 1x1x1 blocks
    # and apply Strassen to each 2x2x2 block.
    
    # Actually, let me think about this differently.
    # The 5x5x5 tensor can be seen as a sum of rank-1 tensors.
    # Schoolbook: 125 terms.
    # 
    # Partition indices into {0,1}, {2,3}, {4}
    # For each triple (I, J, K) where I, J, K are blocks:
    # - If I=J=K=2x2: 8 schoolbook terms -> 7 Strassen terms (save 1)
    # - Other combinations: need to count carefully
    
    # Let me enumerate all block combinations:
    # I can be {0,1}, {2,3}, {4} with sizes 2, 2, 1
    # J can be {0,1}, {2,3}, {4} with sizes 2, 2, 1  
    # K can be {0,1}, {2,3}, {4} with sizes 2, 2, 1
    
    # Block (I,J,K) has size |I| * |J| * |K|
    # Schoolbook terms for block (I,J,K): |I| * |J| * |K|
    # Total schoolbook: sum over all I,J,K of |I|*|J|*|K| = (2+2+1)^3 = 125 ✓
    
    # For Strassen on 2x2x2 blocks: save 1 term per such block
    # Number of 2x2x2 blocks: I={0,1}, J={0,1}, K={0,1} -> 1 block
    # But we also have mixed blocks like I={0,1}, J={0,1}, K={2,3} etc.
    
    # Wait, Strassen applies when all three dimensions are 2x2.
    # Let me reconsider.
    
    # Actually, the Strassen algorithm works on the 2x2x2 sub-tensor.
    # For the full 5x5x5 tensor, we can partition it into sub-tensors
    # and apply Strassen to the 2x2x2 sub-tensors.
    
    # The 5x5x5 tensor T[i,j,k] can be partitioned into 27 sub-tensors
    # based on the block structure {0,1}|{2,3}|{4} for each index.
    
    # Sub-tensor (I,J,K) where I,J,K ∈ {{0,1},{2,3},{4}}
    # Size of sub-tensor (I,J,K): |I| x |J| x |K|
    
    # For sub-tensor ({0,1},{0,1},{0,1}): 2x2x2 -> Strassen saves 1 term (8->7)
    # For sub-tensor ({0,1},{0,1},{2,3}): 2x2x2 -> Strassen saves 1 term (8->7)
    # For sub-tensor ({0,1},{0,1},{4}): 2x2x1 -> can't use Strassen directly
    # etc.
    
    # Let me count how many 2x2x2 sub-tensors there are:
    # I ∈ {{0,1},{2,3}}, J ∈ {{0,1},{2,3}}, K ∈ {{0,1},{2,3}}
    # That's 2*2*2 = 8 sub-tensors of size 2x2x2
    # Each saves 1 term: total savings = 8
    
    # For 2x2x1 sub-tensors: I ∈ {{0,1},{2,3}}, J ∈ {{0,1},{2,3}}, K = {4}
    # That's 2*2*1 = 4 sub-tensors of size 2x2x1
    # Similarly: I ∈ {{0,1},{2,3}}, J = {4}, K ∈ {{0,1},{2,3}}: 2*1*2 = 4
    # I = {4}, J ∈ {{0,1},{2,3}}, K ∈ {{0,1},{2,3}}: 1*2*2 = 4
    # Total 2x2x1 sub-tensors: 12, each with 4 schoolbook terms
    
    # For 2x1x1 sub-tensors: 
    # I ∈ {{0,1},{2,3}}, J = {4}, K = {4}: 2*1*1 = 2
    # I = {4}, J ∈ {{0,1},{2,3}}, K = {4}: 1*2*1 = 2
    # I = {4}, J = {4}, K ∈ {{0,1},{2,3}}: 1*1*2 = 2
    # Total: 6 sub-tensors of size 2x1x1, each with 2 schoolbook terms
    
    # For 1x1x1 sub-tensor: I=J=K={4}: 1 sub-tensor, 1 schoolbook term
    
    # Total schoolbook: 8*8 + 12*4 + 6*2 + 1*1 = 64 + 48 + 12 + 1 = 125 ✓
    # With Strassen on 8 2x2x2 blocks: 8*7 + 12*4 + 6*2 + 1*1 = 56 + 48 + 12 + 1 = 117
    
    # So the rank should be 117.
    
    # Now let's construct the decomposition.
    # For each 2x2x2 sub-tensor, we use Strassen.
    # For other sub-tensors, we use schoolbook.
    
    # Let's define the block index mappings:
    # Block 0: indices {0,1}, Block 1: indices {2,3}, Block 2: index {4}
    
    def block_indices(block):
        if block == 0:
            return [0, 1]
        elif block == 1:
            return [2, 3]
        else:
            return [4]
    
    # Strassen factors for 2x2x2:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b10-b11)
    # M4 = a11*(b01-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b10)
    # M7 = (a01-a11)*(b01+b11)
    
    # In terms of the tensor decomposition:
    # For sub-tensor with A-indices {i0,i1}, B-indices {j0,j1}, C-indices {k0,k1}:
    # The 8 schoolbook terms are (i,j,k) for i∈{i0,i1}, j∈{j0,j1}, k∈{k0,k1}
    # Strassen replaces these with 7 terms.
    
    # Let's define the Strassen coefficients:
    # For a 2x2x2 sub-tensor with A-indices (i0,i1), B-indices (j0,j1), C-indices (k0,k1):
    
    # M1: U has +1 at (i0,j0), (i1,j1); V has +1 at (j0,k0), (j1,k1); W has +1 at (k0,i0), (k1,i1)
    # Wait, that's not quite right. Let me think more carefully.
    
    # In Strassen, M1 = (a00+a11)*(b00+b11)
    # This means: sum over (a00*b00 + a00*b11 + a11*b00 + a11*b11)
    # In the tensor decomposition, each product a_ij*b_jk contributes to c_ki
    # So M1 contributes to:
    #   a00*b00 -> c_00 (i=0,j=0,k=0)
    #   a00*b11 -> c_10 (i=0,j=0,k=1)
    #   a11*b00 -> c_01 (i=1,j=1,k=0)
    #   a11*b11 -> c_11 (i=1,j=1,k=1)
    
    # For the tensor decomposition, M1 is a rank-1 tensor where:
    # U[i,j] = 1 if (i,j) in {(0,0),(1,1)}
    # V[j,k] = 1 if (j,k) in {(0,0),(1,1)}
    # W[k,i] = 1 if (k,i) in {(0,0),(1,1)}
    
    # But this doesn't work because M1 = (a00+a11)*(b00+b11) is a product of sums,
    # not a sum of products. The rank-1 tensor for M1 would be:
    # U[i,j] = 1 for (i,j) in {(0,0),(1,1)}
    # V[j,k] = 1 for (j,k) in {(0,0),(1,1)}
    # W[k,i] = 1 for (k,i) in {(0,0),(1,1)}
    # This gives: sum_{(i,j)∈{(0,0),(1,1)}, (j,k)∈{(0,0),(1,1)}, (k,i)∈{(0,0),(1,1)}} a_ij*b_jk*c_ki
    # = a00*b00*c00 + a00*b11*c10 + a11*b00*c01 + a11*b11*c11
    # Wait, that's not right either. Let me reconsider.
    
    # In the tensor decomposition, we have:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # And we want T[i,j,k] = 1 for all i,j,k.
    
    # For the schoolbook decomposition, each term is a single (i,j,k) with U[i,j,c]=1, V[j,k,c]=1, W[k,i,c]=1.
    
    # For Strassen, we need to express the 2x2x2 sub-tensor as 7 rank-1 tensors.
    # The 2x2x2 sub-tensor has T[i,j,k]=1 for i,j,k ∈ {0,1} (local indices).
    
    # Strassen says: C = A*B can be computed with 7 multiplications.
    # In the tensor sense, the 2x2x2 tensor T[i,j,k]=1 can be decomposed as:
    # T = M1 + M2 + M3 + M4 + M5 + M6 + M7
    # where each Mm is a rank-1 tensor.
    
    # Let me look up the exact Strassen tensor decomposition.
    # The 2x2x2 matrix multiplication tensor has rank 7 (Strassen).
    # The decomposition is:
    # T[i,j,k] = sum_{m=1}^{7} u_m[i,j] * v_m[j,k] * w_m[k,i]
    
    # The Strassen factors are:
    # M1: u1 = e_00 + e_11, v1 = e_00 + e_11, w1 = e_00 + e_11
    # M2: u2 = e_10 + e_11, v2 = e_00, w2 = e_01
    # M3: u3 = e_00, v3 = e_10 - e_11, w3 = e_10
    # M4: u4 = e_11, v4 = e_01 - e_00, w4 = e_11
    # M5: u5 = e_00 + e_01, v5 = e_11, w5 = e_00
    # M6: u6 = e_10 - e_00, v6 = e_00 + e_10, w6 = e_11
    # M7: u7 = e_01 - e_11, v7 = e_01 + e_11, w7 = e_00
    
    # Where e_ij is the matrix unit with 1 at position (i,j).
    
    # Let me verify: T[0,0,0] should be 1.
    # M1: u1[0,0]=1, v1[0,0]=1, w1[0,0]=1 -> 1
    # M2: u2[0,0]=0 -> 0
    # M3: u3[0,0]=1, v3[0,0]=0 -> 0
    # M4: u4[0,0]=0 -> 0
    # M5: u5[0,0]=1, v5[0,0]=0 -> 0
    # M6: u6[0,0]=-1, v6[0,0]=1, w6[0,0]=0 -> 0
    # M7: u7[0,0]=0 -> 0
    # Sum: 1 ✓
    
    # T[0,0,1]:
    # M1: u1[0,0]=1, v1[0,1]=0 -> 0
    # M2: u2[0,0]=0 -> 0
    # M3: u3[0,0]=1, v3[0,1]=-1, w3[1,0]=0 -> 0
    # M4: u4[0,0]=0 -> 0
    # M5: u5[0,0]=1, v5[0,1]=0 -> 0
    # M6: u6[0,0]=-1, v6[0,1]=0 -> 0
    # M7: u7[0,0]=0 -> 0
    # Sum: 0 ✗ (should be 1)
    
    # Hmm, that's not right. Let me reconsider the Strassen decomposition.
    
    # Actually, I think I'm confusing the Strassen algorithm for matrix multiplication
    # with the tensor decomposition. Let me look at this more carefully.
    
    # The matrix multiplication tensor for 2x2 matrices is:
    # T[i,j,k] = 1 for all i,j,k ∈ {0,1}
    # This tensor has rank 7 (Strassen's result).
    
    # The Strassen decomposition expresses T as a sum of 7 rank-1 tensors.
    # Let me find the correct decomposition.
    
    # From the literature, the Strassen decomposition of the 2x2x2 tensor is:
    # T = (e00+e11)⊗(e00+e11)⊗(e00+e11)
    #   + (e10+e11)⊗e00⊗e01
    #   + e00⊗(e10-e11)⊗e10
    #   + e11⊗(e01-e00)⊗e11
    #   + (e00+e01)⊗e11⊗e00
    #   + (e10-e00)⊗(e00+e10)⊗e11
    #   + (e01-e11)⊗(e01+e11)⊗e00
    
    # Let me verify T[0,0,1] again:
    # Term 1: (e00+e11)[0,0]=1, (e00+e11)[0,1]=0, (e00+e11)[1,0]=0 -> 0
    # Term 2: (e10+e11)[0,0]=0 -> 0
    # Term 3: e00[0,0]=1, (e10-e11)[0,1]=-1, e10[1,0]=0 -> 0
    # Term 4: e11[0,0]=0 -> 0
    # Term 5: (e00+e01)[0,0]=1, e11[0,1]=0 -> 0
    # Term 6: (e10-e00)[0,0]=-1, (e00+e10)[0,1]=0 -> 0
    # Term 7: (e01-e11)[0,0]=0 -> 0
    # Sum: 0, but T[0,0,1] should be 1.
    
    # I think I have the tensor indices wrong. Let me reconsider.
    
    # In the matrix multiplication tensor T[i,j,k], the indices are:
    # i: row of A
    # j: column of A / row of B
    # k: column of B
    # And T[i,j,k] = 1 means a_ij * b_jk contributes to c_ik.
    
    # So the tensor is T[i,j,k] where i is row of A, j is col of A/row of B, k is col of B.
    # The output c_ik = sum_j a_ij * b_jk.
    
    # In the tensor decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Wait, that doesn't look right. Let me re-read the problem.
    
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k,i) where k is the col of B and i is the row of A.
    # And the decomposition is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    
    # So U[i,j,c] * V[j,k,c] * W[k,i,c] summed over c should give 1 for each (i,j,k).
    
    # For the 2x2x2 case:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for i,j,k ∈ {0,1}
    
    # Let me try the Strassen decomposition again with this convention.
    # The 8 schoolbook terms are: for each (i,j,k), U[i,j,c]=1, V[j,k,c]=1, W[k,i,c]=1.
    
    # Strassen should give 7 terms that sum to the same tensor.
    
    # I think the issue is that I need to be more careful about which indices go where.
    
    # Let me try a different approach: use the known Strassen decomposition from the literature.
    # The 2x2x2 matrix multiplication tensor has the following rank-7 decomposition:
    
    # Let me denote the rank-1 tensors as (U_m, V_m, W_m) for m=1,...,7.
    
    # From Strassen's algorithm:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b10-b11)
    # M4 = (a11)(b01-b00)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    
    # And:
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # In the tensor decomposition, each Mm is a product of a "U factor" and a "V factor",
    # and the W factor determines how Mm contributes to each c_ik.
    
    # For M1 = (a00+a11)(b00+b11):
    # U1[i,j] = 1 if (i,j) ∈ {(0,0),(1,1)}
    # V1[j,k] = 1 if (j,k) ∈ {(0,0),(1,1)}
    # W1[k,i] = ? 
    # M1 contributes to c00 with coefficient +1, c11 with coefficient +1, and c01, c10 with 0.
    # So W1[0,0] = 1, W1[1,1] = 1, W1[0,1] = 0, W1[1,0] = 0.
    # But wait, M1 = (a00+a11)(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11
    # a00*b00 -> c00 (i=0,k=0)
    # a00*b11 -> c01 (i=0,k=1)
    # a11*b00 -> c10 (i=1,k=0)
    # a11*b11 -> c11 (i=1,k=1)
    # So M1 contributes to all of c00, c01, c10, c11.
    # But in the Strassen formula, c00 = M1 + M4 - M5 + M7, c01 = M3 + M5, etc.
    # So M1 contributes +1 to c00 and +1 to c11, and 0 to c01 and c10?
    # That doesn't match.
    
    # I think the issue is that in the tensor decomposition, the W factor is not simply
    # determined by which c_ik the product contributes to. The W factor is part of the
    # rank-1 tensor decomposition, and the sum over c gives the final result.
    
    # Let me try a completely different approach: just use the schoolbook decomposition
    # but with a clever ordering that might allow for some merging.
    
    # Actually, for this task, let me just implement the 2+2+1 block partition with
    # Strassen on the 2x2x2 blocks, using the known Strassen tensor decomposition.
    
    # I'll use the following Strassen decomposition for the 2x2x2 tensor:
    # (Verified from literature)
    
    strassen_U = [
        [1, 0, 0, 0, 0, 0, 0],  # M1: a00
        [0, 0, 0, 0, 1, 0, 0],  # M1: a11
        [0, 1, 0, 0, 0, 1, 0],  # M2: a10, M6: a10
        [0, 0, 1, 0, 0, -1, 0], # M3: a00, M6: -a00
        [0, 0, 0, 1, 0, 0, -1], # M4: a11, M7: -a11
        [0, 0, 0, 0, 1, 0, 1],  # M5: a01, M7: a01
    ]
    # Wait, this isn't working. Let me just use the schoolbook for now and return 125.
    # The Strassen decomposition is tricky to get right without careful verification.
    
    # For safety, return the schoolbook decomposition with rank 125.
    rank = n * m * p  # 125
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if not _verify(result["U"], result["V"], result["W"]):
        pass
    
    return result
# EVOLVE-BLOCK-END
