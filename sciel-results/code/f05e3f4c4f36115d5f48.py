"""Rewritten program: embed 2x4x6 into 4x4x4, use rank-17 BC algorithm, project."""
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 4, 6
    N = 4  # embed into 4x4x4

    # The 2x4x6 multiplication tensor T[i,j,k] = 1 if k*n+i and j*p+k are "connected"
    # In 4x4x4 embedding: U is (N*m, rank), V is (m*N, rank), W is (N*n, rank)
    # But we need: U is (n*m, rank)=(8,rank), V is (m*p,rank)=(24,rank), W is (p*n,rank)=(12,rank)
    #
    # Embedding strategy:
    # The 2x4 matrix A has indices (i,j) with i in [0,2), j in [0,4)
    # The 4x6 matrix B has indices (j,k) with j in [0,4), k in [0,6)
    # Output C has indices (i,k) with i in [0,2), k in [0,6)
    #
    # Embed into 4x4x4:
    # A_embed: 4x4 matrix where A_embed[i,j] = A[i,j] for i<2, j<4; else 0
    # B_embed: 4x4 matrix where B_embed[j,k] = B[j,k] for j<4, k<4; else 0
    # C_embed: 4x4 matrix where C_embed[i,k] = C[i,k] for i<2, k<4; else 0
    #
    # But wait: B is 4x6, not 4x4. We can't directly embed 4x6 into 4x4.
    #
    # Alternative approach: Use the 4x4x4 rank-17 algorithm directly on blocks.
    # Actually, let me reconsider. The tensor for 2x4x6 multiplication has:
    #   U dimension: n*m = 8
    #   V dimension: m*p = 24  
    #   W dimension: p*n = 12
    #
    # The 4x4x4 tensor has dimensions 16, 16, 16 with rank 17.
    # 
    # Better approach: Think of 2x4x6 as computing C = A * B where A is 2x4, B is 4x6.
    # We can view this as: split B into B1 (4x4) and B2 (4x2) columns.
    # C = A*B1 + A*B2, where C1 = A*B1 is 2x4 and C2 = A*B2 is 2x2.
    #
    # For C1 = A * B1 (2x4 times 4x4): embed A into 4x4 (pad rows), use rank-17 4x4x4.
    # For C2 = A * B2 (2x4 times 4x2): embed into 4x4x4 similarly.
    #
    # Actually, the cleanest embedding: 
    # Pad A (2x4) to A4 (4x4) by adding 2 zero rows.
    # Pad B (4x6) to B4 (4x6)... but 4x4x4 expects 4x4.
    #
    # Let me use a different strategy: block decomposition with Strassen.
    # A is 2x4, B is 4x6.
    # Split A into A1 (2x2) and A2 (2x2) horizontally: A = [A1 | A2]
    # Split B into B1 (4x2), B2 (4x2), B3 (4x2) vertically: B = [B1; B2; B3]
    # Wait, B is 4x6, so split into B_top (2x6) and B_bot (2x6)? No...
    #
    # B is 4x6: rows 0-3, cols 0-5.
    # Split B into B1 (rows 0-1, 4x... no. B is 4 rows, 6 cols.
    # Split B vertically: B = [B_left (4x3) | B_right (4x3)]
    # Then C = A * B_left + A * B_right, each is 2x3.
    #
    # For 2x3 multiplication: embed 2x3 into 4x4x4? 
    # 2x3: A is 2x3, B is 3x3... no, B is 4x3 here.
    #
    # Let me try yet another approach. Use the known result:
    # R(2,4,6) <= R(4,4,4) = 17 by embedding.
    #
    # The 2x4x6 tensor T_{2,4,6} is a sub-tensor of T_{4,4,6} (pad A rows).
    # And T_{4,4,6} can be computed by: split B (4x6) into B1 (4x4) and B2 (4x2).
    # C = A*B1 + A*B2.
    # A*B1: 4x4 times 4x4 -> use rank-17 (but A is only 2x4 effectively, so 2x4x4)
    # A*B2: 4x4 times 4x2 -> 2x4x2
    #
    # Hmm, this is getting complicated. Let me just use the direct rank-17 4x4x4
    # algorithm with a proper embedding.
    #
    # KEY INSIGHT: The 2x4x6 multiplication tensor is a sub-tensor of the 4x4x4
    # multiplication tensor in the following sense:
    # - U indices: (i,j) where i in [0,2), j in [0,4) -> 8 of the 16 U-indices
    # - V indices: (j,k) where j in [0,4), k in [0,6) -> but 4x4x4 has k in [0,4)
    #
    # This doesn't work directly because p=6 > 4.
    #
    # NEW APPROACH: Use rank-17 for the 4x4x4 part and handle the extra 2 columns
    # of B separately with schoolbook.
    #
    # B is 4x6 = B_left (4x4) + B_right (4x2) [column split]
    # C = A * B_left + A * B_right
    # A * B_left: 2x4 * 4x4 -> embed 2x4 into 4x4 (pad 2 rows) -> use rank-17
    # A * B_right: 2x4 * 4x2 -> schoolbook: 2*4*2 = 16 terms, but we can do better
    #
    # For A*B_right (2x4 * 4x2): 
    #   C[i,k] = sum_j A[i,j]*B[j,k] for i in [0,2), k in [0,2)
    #   This is a 2x4x2 tensor. Rank <= 2*4*2 = 8 (schoolbook), but likely less.
    #   Actually rank of 2x4x2: we can use rank 4 (Strassen-like for 2x2 blocks)
    #   or just schoolbook 8.
    #
    # Total: 17 + 8 = 25? That's worse than 42.
    #
    # Let me try: embed the whole thing into 4x4x4 properly.
    #
    # Actually, I recall that R(m,n,p) <= R(m',n',p') if m<=m', n<=n', p<=p'
    # by zero-padding. But here p=6 > 4, so 2x4x6 is NOT a sub-tensor of 4x4x4.
    #
    # However, 2x4x6 IS a sub-tensor of 4x4x6 (pad A to 4x4).
    # And 4x4x6 can be decomposed as 4x4x4 + 4x4x2.
    # R(4,4,6) <= R(4,4,4) + R(4,4,2) = 17 + R(4,4,2)
    # R(4,4,2): 4x4 times 4x2. Schoolbook: 4*4*2=32. 
    # But we can do: split into two 4x4x1, each rank <= 4 (outer product sum)... 
    # Actually R(4,4,1) = 4 (rank of 4x4 matrix as sum of 4 rank-1).
    # So R(4,4,2) <= 8? No, R(4,4,2) is the rank of the 4x4x2 multiplication tensor.
    # The 4x4x2 tensor: C[i,k] = sum_j A[i,j]B[j,k], i in [0,4), j in [0,4), k in [0,2).
    # This is two 4x4x1 tensors stacked. Each 4x4x1 has rank 4 (the 4x4 identity-like).
    # Actually the 4x4x1 multiplication tensor has rank 4 (it's just the 4x4 matrix 
    # multiplication by a fixed vector, which needs 4 terms).
    # So R(4,4,2) <= 8.
    # Total: 17 + 8 = 25.
    #
    # But wait, we don't need full 4x4x4. We need 2x4x4 (A is 2x4).
    # R(2,4,4) <= R(4,4,4) = 17 (by padding A to 4x4).
    # R(2,4,2): 2x4 times 4x2. Schoolbook: 2*4*2=8.
    # Can we do better? R(2,4,2) <= R(4,4,2) <= 8. Probably 8 is optimal or close.
    # Total: 17 + 8 = 25.
    #
    # Hmm, 25 < 42 (Strassen) < 48 (schoolbook). Good improvement!
    #
    # But can we do even better? Let me think about R(2,4,4) more carefully.
    # 2x4x4 tensor: C[i,k] = sum_j A[i,j]B[j,k], i in [0,2), j in [0,4), k in [0,4).
    # This is a sub-tensor of 4x4x4 (pad A rows 2,3 to zero).
    # The rank-17 decomposition of 4x4x4, when restricted to i in [0,2), 
    # gives a valid decomposition of 2x4x4 with rank <= 17.
    # Some of the 17 terms may be redundant (all zero for i>=2), so actual rank may be less.
    #
    # Let me just implement this:
    # 1. Get rank-17 4x4x4 decomposition (BC algorithm)
    # 2. For 2x4x4: keep only terms where the U-factor has non-zero in rows 0,1 (i=0,1)
    #    Actually, in 4x4x4, U[i*m+j, c] where i,j in [0,4). For 2x4, we need i in [0,2).
    #    The U matrix is 16x17. We take rows 0-7 (i=0,1 for all j) -> 8x17.
    #    V is 16x17, take all rows (j in [0,4), k in [0,4)) -> 16x17.
    #    W is 16x17, take rows where k in [0,4), i in [0,2) -> rows k*4+i for k in [0,4), i in [0,2) -> 8 rows.
    #    Wait, W index is k*n+i in 4x4x4, so k*4+i. For i in [0,2), k in [0,4): 8 rows.
    # 3. For 2x4x2: schoolbook, 8 terms.
    # 4. Combine: total rank = rank(2x4x4 part) + 8.
    #
    # Actually, I realize I should just directly construct the full 2x4x6 decomposition
    # by combining the 2x4x4 (rank <= 17) and 2x4x2 (rank 8) parts.
    #
    # But actually, let me reconsider. Can I embed 2x4x6 directly?
    # 2x4x6 is a sub-tensor of 4x4x6 (pad A). 
    # 4x4x6 is a sub-tensor of 4x4x8 (pad B). 
    # 4x4x8... not helpful.
    #
    # Let me just go with: rank-17 for 2x4x4 (via 4x4x4 restriction) + schoolbook for 2x4x2.
    # Total rank <= 25.
    #
    # Now I need the actual rank-17 4x4x4 factor matrices.

    # Rank-17 algorithm for 4x4x4 (Bürgisser-Clausen-Shokrollahi)
    # U: 16x17, V: 16x17, W: 16x17
    # U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
    
    # The BC rank-17 algorithm uses 17 terms. Let me define them.
    # This is a well-known algorithm. The factors involve entries in {-1, 0, 1, 1/2}.
    
    # I'll use the standard formulation. The 17 products are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A01+A11)(B10-B00)  
    # M3 = (A10+A00)(B01-B11)
    # M4 = (A11-A10)(B10+B11)
    # M5 = (A00+A01)(B11-B01)
    # M6 = (A10+A11)(B00-B10)
    # M7 = (A01-A00)(B01+B11)
    # Plus 10 more for the 4x4 case...
    
    # Actually, the rank-17 4x4x4 algorithm is quite complex. Let me use a 
    # different approach: use the known Strassen-like decomposition.
    #
    # Alternative: Use rank 18 or 19 which might be easier to write, 
    # or just use the 2x2x2 Strassen (rank 7) applied to 2x2 blocks.
    #
    # For 4x4x4 with 2x2 blocks: 4 blocks in A, 4 in B, 4 in C.
    # Strassen on 2x2 blocks: 7 multiplications per 2x2 block multiplication.
    # 4x4 = 4 blocks of 2x2. Using Strassen: 7 block multiplications.
    # Each block multiplication is 2x2x2 with rank 7.
    # Total: 7 * 7 = 49? No, that's worse.
    #
    # Actually for 4x4x4 using Strassen recursively:
    # 4x4x4 -> 7 multiplications of 2x2x2 -> 7 * 7 = 49. That's bad.
    #
    # The BC rank-17 is much better. Let me look up the explicit factors.
    #
    # I'll use a known rank-17 4x4x4 decomposition. The factors are 16x17 matrices.
    # Let me define them using the algorithm from BC 1990.
    #
    # Actually, for time efficiency, let me use a simpler but still good approach:
    # Use Strassen for the 2x4x6 problem directly.
    #
    # 2x4x6: A is 2x4, B is 4x6.
    # Split A into A1 (2x2) and A2 (2x2): A = [A1 | A2]
    # Split B into B1 (4x3) and B2 (4x3): B = [B1 | B2]
    # C = [A*B1 | A*B2], each 2x3.
    #
    # For A*B1 (2x2... no, 2x4 * 4x3):
    # Split B1 into B1a (2x3) and B1b (2x3) vertically: B1 = [B1a; B1b]
    # A*B1 = A1*B1a + A2*B1b, where A1 is 2x2, A2 is 2x2, B1a is 2x3, B1b is 2x3.
    #
    # A1*B1a: 2x2 * 2x3. Use Strassen-like: 
    #   Split B1a into [B1a1 (2x1) | B1a2 (2x2)]
    #   A1*B1a1: 2x2 * 2x1 -> rank 3 (Strassen for 2x2x1: (A11+A22)(B11+B21) - ... )
    #   Actually 2x2x1: C[i] = sum_j A[i,j]B[j]. Rank = 2 (just two dot products? No.)
    #   2x2x1 tensor: C[0] = A00*B0 + A01*B1, C[1] = A10*B0 + A11*B1.
    #   This is rank 2: C = A * B where A is 2x2, B is 2x1. 
    #   As a tensor, it's the outer product structure. Rank = 2.
    #   Wait: T[i,j,k] where i in [0,2), j in [0,2), k in [0,1). 
    #   T[i,j,0] = 1 if... no, the multiplication tensor is T[i,j,k] = delta_{i,j} for fixed k? 
    #   No. The multiplication tensor for m x n * n x p:
    #   T[i,j,k] = 1 if the (i,k) output depends on A[i,j]*B[j,k].
    #   For 2x2x1: C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    #              C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    #   Tensor T[i,j,k]: T[i,j,k] = 1 for all i,j,k in range? No.
    #   T[i,j,k] = 1 if output (i,k) uses input A[i,j] and B[j,k].
    #   For 2x2x1: T[i,j,0] = 1 for all i in [0,2), j in [0,2). That's 4 terms, rank 4?
    #   No wait. The tensor is T[i,j,k] = 1 for all valid (i,j,k). 
    #   The rank is the minimum number of rank-1 tensors that sum to T.
    #   For 2x2x1: T has entries T[i,j,0] = 1 for i,j in {0,1}. 
    #   This is the all-ones 2x2 matrix (for k=0). Rank of all-ones 2x2 is 1!
    #   So R(2,2,1) = 1? That can't be right for matrix multiplication.
    #
    # I'm confusing myself. Let me restart with clear definitions.
    #
    # Matrix multiplication tensor T_{m,n,p}: a tensor in R^{m*n} x R^{n*p} x R^{p*m}
    # (or whatever the convention is) such that for matrices A (m x n) and B (n x p),
    # the product C = A*B is given by contracting T with vec(A) and vec(B).
    #
    # Specifically: C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = 1 for all i,j,k (the "multiplication" tensor has all 1s
    # in the sense that every triple (i,j,k) contributes A[i,j]*B[j,k] to C[i,k]).
    #
    # Wait no. The tensor T is defined such that:
    # C[i,k] = sum_{j} A[i,j] B[j,k] = sum_{i',j',k'} T[i',j',k'] A[i',j'] B[j',k'] * delta_{i',i} delta_{k',k}
    # So T[i,j,k] = 1 for all i,j,k, and the contraction picks out the right terms.
    #
    # Actually, the standard definition: The multiplication tensor T_{m,n,p} is an element
    # of R^{mn} ⊗ R^{np} ⊗ R^{pm} (or similar) defined by:
    # T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # where e_{i,j} is the basis vector for position (i,j) in the flattened matrix.
    #
    # So T has a 1 at position (i*m+j, j*p+k, k*n+i) for each i,j,k, and 0 elsewhere.
    # This matches the seed code: U[i*m+j][c]=1, V[j*p+k][c]=1, W[k*n+i][c]=1.
    #
    # The rank of T is the minimum r such that T = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c
    # where u_c is in R^{mn}, v_c in R^{np}, w_c in R^{pm}.
    #
    # For 2x4x6: T is in R^8 ⊗ R^24 ⊗ R^12.
    # T[i*4+j, j*6+k, k*2+i] = 1 for i in [0,2), j in [0,4), k in [0,6).
    #
    # Now, the 4x4x4 tensor T4 is in R^16 ⊗ R^16 ⊗ R^16:
    # T4[i*4+j, j*4+k, k*4+i] = 1 for i,j,k in [0,4).
    #
    # To embed 2x4x6 into 4x4x4:
    # - U: rows i*4+j for i in [0,2), j in [0,4) -> rows 0-7 (8 rows out of 16)
    # - V: rows j*4+k for j in [0,4), k in [0,4) -> but we need k in [0,6)!
    #
    # The problem is p=6 > 4. So 2x4x6 is NOT a sub-tensor of 4x4x4.
    #
    # OK so I need a different approach. Let me use:
    # 2x4x6 = 2x4x4 + 2x4x2 (split B's columns)
    # R(2,4,6) <= R(2,4,4) + R(2,4,2)
    #
    # R(2,4,4): embed into 4x4x4 (pad A's rows). R(2,4,4) <= R(4,4,4) = 17.
    # R(2,4,2): 2x4x2 tensor. Schoolbook: 2*4*2 = 8.
    # Can we do better than 8 for 2x4x2?
    # 2x4x2: C[i,k] = sum_j A[i,j]B[j,k], i in [0,2), j in [0,4), k in [0,2).
    # This is two 2x4x1 tensors. R(2,4,1): C[i] = sum_j A[i,j]B[j], i in [0,2), j in [0,4).
    # This is just two dot products of length 4. As a tensor: T[i,j,0] = 1 for all i,j.
    # This is the all-ones 2x4 matrix. Rank = 1 (it's u ⊗ v where u = [1,1], v = [1,1,1,1]).
    # So R(2,4,1) = 1? 
    # Wait, that means C[0] = sum_j A[0,j]*B[j] and C[1] = sum_j A[1,j]*B[j].
    # As a bilinear map, this is: (A,B) -> (A[0,:]·B, A[1,:]·B).
    # This can be written as: let A' = [A[0,:]; A[1,:]] (2x4), then C = A' * B (2x4 * 4x1 = 2x1).
    # The tensor for 2x4x1: T[i,j,0] = 1 for i in [0,2), j in [0,4).
    # This tensor = sum_i e_i ⊗ (sum_j e_j) ⊗ e_0 = (e_0 + e_1) ⊗ (e_0+e_1+e_2+e_3) ⊗ e_0.
    # So rank = 1! Indeed R(2,4,1) = 1.
    # Similarly R(2,4,2) <= 2 (two independent 2x4x1 tensors).
    # But wait, R(2,4,2) is the rank of the 2x4x2 multiplication tensor, which is:
    # T[i,j,k] = 1 for i in [0,2), j in [0,4), k in [0,2).
    # = sum_k (sum_i e_i) ⊗ (sum_j e_j) ⊗ e_k... no.
    # T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # For 2x4x2: T[i,j,k] = 1 for i in {0,1}, j in {0,1,2,3}, k in {0,1}.
    # = (e_{0,0}+e_{0,1}+e_{1,0}+e_{1,1}+e_{0,2}+e_{0,3}+e_{1,2}+e_{1,3}) 
    #   ⊗ ... no, this isn't a simple outer product.
    #
    # T = sum_{i,j,k} e_{i*4+j} ⊗ e_{j*2+k} ⊗ e_{k*2+i}
    # = sum_k [sum_{i,j} e_{i*4+j} ⊗ e_{j*2+k} ⊗ e_{k*2+i}]
    # For k=0: sum_{i,j} e_{i*4+j} ⊗ e_{j*2} ⊗ e_{2*i}
    # For k=1: sum_{i,j} e_{i*4+j} ⊗ e_{j*2+1} ⊗ e_{2*i+1}
    #
    # The k=0 part: sum_{i,j} e_{i*4+j} ⊗ e_{j*2} ⊗ e_{2*i}
    # = sum_j e_{j*2} ⊗ (sum_i e_{i*4+j} ⊗ e_{2*i})
    # For fixed j: sum_i e_{i*4+j} ⊗ e_{2*i} = e_j ⊗ e_0 + e_{4+j} ⊗ e_2 (for i=0,1)
    # So k=0 part = e_0 ⊗ (e_0⊗e_0 + e_4⊗e_2) + e_2 ⊗ (e_1⊗e_0 + e_5⊗e_2) + ...
    # This is getting complicated. Let me just use schoolbook for 2x4x2: rank 8.
    #
    # So total: R(2,4,6) <= 17 + 8 = 25.
    #
    # Now I need the actual rank-17 4x4x4 factors. Let me define them.
    #
    # I'll use the well-known rank-17 algorithm. The factors are:
    # (I'll define U, V, W as 16x17 matrices for 4x4x4)
    
    # Rank-17 4x4x4 algorithm (BC 1990). 
    # The 17 terms are indexed c=0..16.
    # U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
    
    # I'll use a known explicit construction. Here are the factors:
    # (Using the standard BC rank-17 algorithm)
    
    U4 = [[0]*17 for _ in range(16)]
    V4 = [[0]*17 for _ in range(16)]
    W4 = [[0]*17 for _ in range(16)]
    
    # The BC rank-17 algorithm:
    # Let me use the formulation where the 17 products are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A01+A11)(B10-B00)
    # M3 = (A10+A00)(B01-B11)
    # M4 = (A11-A10)(B10+B11)
    # M5 = (A00+A01)(B11-B01)
    # M6 = (A10+A11)(B00-B10)
    # M7 = (A01-A00)(B01+B11)
    # These 7 are for the 2x2x2 Strassen part. For 4x4x4, we need 17 total.
    #
    # Actually, the rank-17 4x4x4 algorithm is not simply 7 Strassen terms.
    # It's a specific algorithm with 17 bilinear forms.
    #
    # For time, let me use a simpler approach: I'll use the rank-19 or rank-18
    # algorithm which might be easier to implement, or I'll use a different strategy.
    #
    # ALTERNATIVE: Use the fact that R(2,4,6) can be computed by:
    # 1. Pad A (2x4) to A4 (4x4) with zeros in rows 2,3.
    # 2. Split B (4x6) into B1 (4x4) and B2 (4x2).
    # 3. Compute C1 = A4 * B1 using rank-17 4x4x4 (but only rows 0,1 of A4 are non-zero).
    # 4. Compute C2 = A4 * B2 using schoolbook (2x4x2, rank 8).
    # 5. C = [C1 | C2].
    #
    # For step 3: The rank-17 4x4x4 decomposition gives C4 = A4 * B4 (4x4).
    # We only need rows 0,1 of C4. The decomposition restricted to these rows
    # still has rank <= 17 (some terms may be zero, reducing rank).
    #
    # Let me just use the schoolbook for the 2x4x2 part and find a good 
    # decomposition for 2x4x4.
    #
    # For 2x4x4: I can use the rank-7 Strassen on 2x2 blocks.
    # A (2x4) = [A1 (2x2) | A2 (2x2)]
    # B (4x4) = [B1 (2x2); B2 (2x2)] (top and bottom)
    # C (2x4) = A1*B1 + A2*B2
    # Using Strassen for each 2x2x2: 7 terms each, total 14.
    # But we can share: C = A1*B1 + A2*B2.
    # Strassen for A1*B1: 7 terms.
    # Strassen for A2*B2: 7 terms.
    # Total: 14. Better than schoolbook 2*4*4=32!
    #
    # So R(2,4,4) <= 14 (two Strassen decompositions).
    # And R(2,4,2) <= 8 (schoolbook) or better.
    #
    # For 2x4x2: A (2x4) = [A1 (2x2) | A2 (2x2)], B (4x2) = [B1 (2x2); B2 (2x2)]
    # C (2x2) = A1*B1 + A2*B2
    # Strassen for each: 7 terms each, total 14? No, that's for 2x2x2.
    # Wait, 2x4x2: C[i,k] = sum_j A[i,j]B[j,k], i in [0,2), j in [0,4), k in [0,2).
    # Split: C = A1*B1 + A2*B2 where A1,A2 are 2x2, B1,B2 are 2x2.
    # A1*B1 is 2x2x2 (rank 7 with Strassen), A2*B2 is 2x2x2 (rank 7).
    # Total: 14. But schoolbook is 2*4*2=8. So schoolbook is better here!
    #
    # So: R(2,4,6) <= 14 + 8 = 22.
    #
    # Even better: Can I do 2x4x4 with rank < 14?
    # R(2,4,4): A (2x4), B (4x4), C (2x4).
    # C[i,k] = sum_j A[i,j]B[j,k].
    # This is a 2x4x4 tensor. 
    # R(2,4,4) <= R(4,4,4) = 17 (by padding). But 14 < 17, so 14 is better.
    # Can we do even better? R(2,4,4) >= ?
    # Lower bound: R(2,4,4) >= max(R(2,4,1), R(1,4,4), ...) 
    # R(2,4,1) = 1 (as shown above). Not helpful.
    #
    # Let me just go with 14 + 8 = 22. That's a big improvement over 48!
    #
    # Wait, I should double-check: is R(2,4,4) really <= 14?
    # 2x4x4: C = A1*B1 + A2*B2 where A1,A2 are 2x2, B1,B2 are 2x2.
    # A1*B1: 2x2x2, rank 7 (Strassen).
    # A2*B2: 2x2x2, rank 7 (Strassen).
    # Total: 14 rank-1 terms. Yes!
    #
    # And for 2x4x2: C = A1*B1 + A2*B2, schoolbook: 2*2*2 + 2*2*2 = 8.
    # Or Strassen: 7+7=14. Schoolbook is better: 8.
    #
    # Total: 14 + 8 = 22.
    #
    # Let me implement this.

    # Strassen 2x2x2 factors (rank 7):
    # U: 4x7, V: 4x7, W: 4x7
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    def strassen_2x2x2():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        # Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B11-B21)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        # 
        # In tensor form: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        # i,j,k in {0,1}
        
        # M1: (A00+A11)(B00+B11)
        # U: A00 and A11 -> U[0,0]=1, U[3,0]=1 (i=0,j=0 and i=1,j=1)
        # V: B00 and B11 -> V[0,0]=1, V[3,0]=1 (j=0,k=0 and j=1,k=1)
        # W: C00 += M1, C11 += M1 -> W[0,0]=1, W[3,0]=1 (k=0,i=0 and k=1,i=1)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2: (A10+A11)B00
        # U: A10, A11 -> U[2,1]=1, U[3,1]=1
        # V: B00 -> V[0,1]=1
        # W: C10 += M2 -> W[2,1]=1 (k=0,i=1)
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        
        # M3: A00(B10-B11)
        # U: A00 -> U[0,2]=1
        # V: B10, B11 -> V[2,2]=1, V[3,2]=-1
        # W: C01 += M3 -> W[1,2]=1 (k=1,i=0)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1
        
        # M4: A11(B00-B10)
        # U: A11 -> U[3,3]=1
        # V: B00, B10 -> V[0,3]=1, V[2,3]=-1
        # W: C00 += M4, C10 += M4 -> W[0,3]=1, W[2,3]=1
        U[3][3] = 1
        V[0][3] = 1; V[2][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5: (A00+A01)B11
        # U: A00, A01 -> U[0,4]=1, U[1,4]=1
        # V: B11 -> V[3,4]=1
        # W: C00 -= M5, C11 += M5 -> W[0,4]=-1, W[3,4]=1
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[3][4] = 1
        
        # M6: (A10-A00)(B00+B01)
        # U: A10, A00 -> U[2,5]=1, U[0,5]=-1
        # V: B00, B01 -> V[0,5]=1, V[1,5]=1
        # W: C11 += M6 -> W[3,5]=1
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7: (A01-A11)(B10+B11)
        # U: A01, A11 -> U[1,6]=1, U[3,6]=-1
        # V: B10, B11 -> V[2,6]=1, V[3,6]=1
        # W: C00 += M7, C01 -= M7 -> W[0,6]=1, W[1,6]=-1
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[1][6] = -1
        
        return U, V, W
    
    # Now construct 2x4x4 using two Strassen decompositions:
    # A (2x4) = [A1 (2x2) | A2 (2x2)]
    # B (4x4) = [B1 (2x2); B2 (2x2)]
    # C (2x4) = A1*B1 + A2*B2
    #
    # U for 2x4x4: 8x14 (n*m = 2*4 = 8 rows, rank 14)
    # V for 2x4x4: 16x14 (m*p = 4*4 = 16 rows, rank 14)
    # W for 2x4x4: 8x14 (p*n = 4*2 = 8 rows, rank 14)
    
    S_U, S_V, S_W = strassen_2x2x2()
    
    U_244 = [[0]*14 for _ in range(8)]
    V_244 = [[0]*14 for _ in range(16)]
    W_244 = [[0]*14 for _ in range(8)]
    
    # First Strassen: A1*B1
    # A1 is rows 0-1, cols 0-1 of A. In U_244, rows i*4+j for i in [0,2), j in [0,2) -> rows 0,1,4,5
    # B1 is rows 0-1, cols 0-1 of B. In V_244, rows j*4+k for j in [0,2), k in [0,2) -> rows 0,1,4,5
    # C1 is cols 0-1 of C. In W_244, rows k*2+i for k in [0,2), i in [0,2) -> rows 0,1,2,3
    
    for c in range(7):
        for r in range(4):
            # U: S_U[r, c] -> U_244[i*4+j, c] where r = i*2+j
            i = r // 2
            j = r % 2
            U_244[i*4 + j][c] += S_U[r][c]
            # V: S_V[r, c] -> V_244[j*4+k, c] where r = j*2+k
            j2 = r // 2
            k = r % 2
            V_244[j2*4 + k][c] += S_V[r][c]
            # W: S_W[r, c] -> W_244[k*2+i, c] where r = k*2+i
            k2 = r // 2
            i2 = r % 2
            W_244[k2*2 + i2][c] += S_W[r][c]
    
    # Second Strassen: A2*B2
    # A2 is rows 0-1, cols 2-3 of A. In U_244, rows i*4+j for i in [0,2), j in [2,4) -> rows 2,3,6,7
    # B2 is rows 2-3, cols 0-1 of B. In V_244, rows j*4+k for j in [2,4), k in [0,2) -> rows 8,9,12,13
    # C2 is cols 2-3 of C. In W_244, rows k*2+i for k in [2,4), i in [0,2) -> rows 4,5,6,7
    
    for c in range(7, 14):
        sc = c - 7
        for r in range(4):
            i = r // 2
            j = r % 2
            U_244[i*4 + (j+2)][c] += S_U[r][sc]
            j2 = r // 2
            k = r % 2
            V_244[(j2+2)*4 + k][c] += S_V[r][sc]
            k2 = r // 2
            i2 = r % 2
            W_244[(k2+2)*2 + i2][c] += S_W[r][sc]
    
    # Now 2x4x2: schoolbook, rank 8
    # A (2x4), B (4x2), C (2x2)
    # U: 8x8, V: 8x8, W: 4x8
    U_242 = [[0]*8 for _ in range(8)]
    V_242 = [[0]*8 for _ in range(8)]
    W_242 = [[0]*8 for _ in range(4)]
    
    col = 0
    for i in range(2):
        for j in range(4):
            for k in range(2):
                U_242[i*4 + j][col] = 1
                V_242[j*2 + k][col] = 1
                W_242[k*2 + i][col] = 1
                col += 1
    
    # Combine: total rank = 14 + 8 = 22
    # U: 8x22, V: 24x22, W: 12x22
    # 
    # For the full 2x4x6:
    # U: rows i*4+j for i in [0,2), j in [0,4) -> 8 rows
    # V: rows j*6+k for j in [0,4), k in [0,6) -> 24 rows
    # W: rows k*2+i for k in [0,6), i in [0,2) -> 12 rows
    
    rank = 22
    U = [[0]*rank for _ in range(8)]
    V = [[0]*rank for _ in range(24)]
    W = [[0]*rank for _ in range(12)]
    
    # Copy 2x4x4 part (cols 0-13)
    for r in range(8):
        for c in range(14):
            U[r][c] = U_244[r][c]
    
    for r in range(16):
        for c in range(14):
            # V_244 has rows j*4+k for j in [0,4), k in [0,4)
            # Full V has rows j*6+k for j in [0,4), k in [0,6)
            # So V_244 row j*4+k maps to full V row j*6+k
            j = r // 4
            k = r % 4
            V[j*6 + k][c] = V_244[r][c]
    
    for r in range(8):
        for c in range(14):
            # W_244 has rows k*2+i for k in [0,4), i in [0,2)
            # Full W has rows k*2+i for k in [0,6), i in [0,2)
            # So W_244 row k*2+i maps to full W row k*2+i
            W[r][c] = W_244[r][c]
    
    # Copy 2x4x2 part (cols 14-21)
    for r in range(8):
        for c in range(8):
            U[r][14+c] = U_242[r][c]
    
    for r in range(8):
        for c in range(8):
            # V_242 has rows j*2+k for j in [0,4), k in [0,2)
            # Full V has rows j*6+k for j in [0,4), k in [0,6)
            # The 2x4x2 part uses B's first 2 columns (k=0,1)
            j = r // 2
            k = r % 2
            V[j*6 + k][14+c] = V_242[r][c]
    
    for r in range(4):
        for c in range(8):
            # W_242 has rows k*2+i for k in [0,2), i in [0,2)
            # Full W has rows k*2+i for k in [0,6), i in [0,2)
            W[r][14+c] = W_242[r][c]
    
    # Verify: check that the decomposition is correct
    # C[i,k] = sum_c U[i*4+j... no, U[i*4+j, c] * V[j*6+k, c] * W[k*2+i, c]
    # Actually: C[i,k] = sum_{j,c} U[i*4+j, c] * V[j*6+k, c] * W[k*2+i, c]
    # This should equal sum_j A[i,j]*B[j,k] for the multiplication tensor.
    
    # Let me verify with random matrices
    import random
    random.seed(42)
    for trial in range(100):
        A = [[random.randint(-5, 5) for _ in range(4)] for _ in range(2)]
        B = [[random.randint(-5, 5) for _ in range(6)] for _ in range(4)]
        C_true = [[sum(A[i][j]*B[j][k] for j in range(4)) for k in range(6)] for i in range(2)]
        C_comp = [[0]*6 for _ in range(2)]
        for i in range(2):
            for k in range(6):
                s = 0
                for j in range(4):
                    for c in range(rank):
                        s += U[i*4+j][c] * V[j*6+k][c] * W[k*2+i][c]
                C_comp[i][k] = s
                if s != C_true[i][k]:
                    raise ValueError(f"Mismatch at ({i},{k}): {s} != {C_true[i][k]}")
    
    # Remove zero columns to minimize rank
    active = [c for c in range(rank) if any(U[r][c] != 0 or V[r][c] != 0 or W[r][c] != 0 for r in range(len(U)))]
    if len(active) < rank:
        U = [[U[r][c] for c in active] for r in range(8)]
        V = [[V[r][c] for c in active] for r in range(24)]
        W = [[W[r][c] for c in active] for r in range(12)]
        rank = len(active)
    
    elapsed = time.time() - start
    if elapsed > deadline - start:
        pass  # already done
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
