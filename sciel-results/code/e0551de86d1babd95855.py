"""Program to find a rank-minimal decomposition for 5x5 matrix multiplication."""
import numpy as np
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Step 1: Construct the 4x4 decomposition using Strassen's algorithm (Rank 49)
    # We will generate the 4x4 factors by applying Strassen to 2x2 blocks recursively.
    
    # Base case: 2x2 Strassen decomposition
    # Indices: A[i,j] -> U[i*2+j, r], B[j,k] -> V[j*2+k, r], C[i,k] -> W[k*2+i, r]
    # Note: The task specifies W[k*n+i, r]. For 2x2, n=2, so W[k*2+i, r].
    
    U2 = np.zeros((4, 7), dtype=object)
    V2 = np.zeros((4, 7), dtype=object)
    W2 = np.zeros((4, 7), dtype=object)
    
    # P1 = (A11+A22)(B11+B22)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1  # C11 (k=0,i=0), C22 (k=1,i=1)
    
    # P2 = (A21+A22)B11
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1 # C21 (k=1,i=0 -> idx 2), C22 (k=1,i=1 -> idx 3)
    
    # P3 = A11(B12-B22)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1  # C12 (k=0,i=1 -> idx 1), C22 (k=1,i=1 -> idx 3)
    
    # P4 = A22(B21-B11)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1  # C11 (k=0,i=0 -> idx 0), C21 (k=1,i=0 -> idx 2)
    
    # P5 = (A11+A12)B22
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[1, 4] = 1; W2[0, 4] = -1 # C12 (k=0,i=1 -> idx 1), C11 (k=0,i=0 -> idx 0)
    
    # P6 = (A21-A11)(B11+B12)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1                # C22 (k=1,i=1 -> idx 3)
    
    # P7 = (A12-A22)(B21+B22)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1                # C11 (k=0,i=0 -> idx 0)
    
    # Now expand to 4x4.
    # The 4x4 matrix multiplication tensor can be decomposed by viewing 4x4 matrices as 2x2 blocks of 2x2 matrices.
    # Let A = [[A11, A12], [A21, A22]] where each block is 2x2.
    # Then C = A*B = [[A11*B11 + A12*B21, A11*B12 + A12*B22], [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    # We apply Strassen's 7 multiplications to these blocks.
    # Each "multiplication" in the 2x2 Strassen formula is actually a 2x2 matrix multiplication.
    # Since we have a decomposition for 2x2 multiplication (U2, V2, W2), we can use it.
    
    # The rank of 4x4 multiplication via this method is 7 * 7 = 49.
    # The structure is:
    # For each of the 7 Strassen steps (r1 in 0..6), we perform a 2x2 multiplication (r2 in 0..6).
    # Total rank R = 7 * 7 = 49.
    
    R4 = 49
    U4 = np.zeros((16, R4), dtype=object)
    V4 = np.zeros((16, R4), dtype=object)
    W4 = np.zeros((16, R4), dtype=object)
    
    # Map 2x2 block indices to 4x4 indices.
    # Block (bi, bj) in 2x2 space corresponds to 4x4 indices:
    # i = 2*bi + i2, j = 2*bj + j2
    # For U (input A): index = i*4 + j
    # For V (input B): index = j*4 + k
    # For W (output C): index = k*4 + i
    
    # Strassen terms for 2x2 blocks:
    # Term 0: U blocks: (0,0), (1,1). V blocks: (0,0), (1,1). W blocks: (0,0), (1,1).
    # Term 1: U blocks: (1,0), (1,1). V blocks: (0,0). W blocks: (1,0), (1,1) with signs 1, -1.
    # ... etc.
    
    # Let's define the block-level Strassen coefficients.
    # u_block[r1, bi, bj] is the coefficient for block (bi, bj) in U for Strassen step r1.
    # v_block[r1, bj, bk] is the coefficient for block (bj, bk) in V for Strassen step r1.
    # w_block[r1, bk, bi] is the coefficient for block (bk, bi) in W for Strassen step r1.
    
    # Using the U2, V2, W2 definitions above:
    # r1=0: U: (0,0)=1, (1,1)=1. V: (0,0)=1, (1,1)=1. W: (0,0)=1, (1,1)=1.
    # r1=1: U: (1,0)=1, (1,1)=1. V: (0,0)=1. W: (1,0)=1, (1,1)=-1.
    # r1=2: U: (0,0)=1. V: (0,1)=1, (1,1)=-1. W: (0,1)=1, (1,1)=1.
    # r1=3: U: (1,1)=1. V: (1,0)=1, (0,0)=-1. W: (0,0)=1, (1,0)=1.
    # r1=4: U: (0,0)=1, (0,1)=1. V: (1,1)=1. W: (0,1)=1, (0,0)=-1.
    # r1=5: U: (1,0)=1, (0,0)=-1. V: (0,0)=1, (0,1)=1. W: (1,1)=1.
    # r1=6: U: (0,1)=1, (1,1)=-1. V: (1,0)=1, (1,1)=1. W: (0,0)=1.
    
    # We can just iterate over the non-zero entries of U2, V2, W2 to build the block coefficients.
    
    for r1 in range(7):
        # Get non-zero entries in U2[:, r1]
        # Indices in U2 are i*2+j.
        # For each non-zero (u_idx, coeff_u) in U2[:, r1]:
        #   i_block = u_idx // 2, j_block = u_idx % 2
        #   For each non-zero (v_idx, coeff_v) in V2[:, r1]:
        #     j_block_v = v_idx // 2, k_block = v_idx % 2
        #     # The block multiplication is for block (i_block, j_block) * (j_block_v, k_block)
        #     # This is only valid if j_block == j_block_v.
        #     if j_block == j_block_v:
        #        # Now we need to perform the 2x2 multiplication for this block product.
        #        # We use the base 2x2 decomposition (U2, V2, W2) for the inner product.
        #        # The inner rank is 7.
        #        for r2 in range(7):
        #            r4 = r1 * 7 + r2
        #            # The contribution to U4 comes from U2[:, r2] scaled by coeff_u
        #            # The contribution to V4 comes from V2[:, r2] scaled by coeff_v
        #            # The contribution to W4 comes from W2[:, r2] scaled by coeff_u * coeff_v * (W2 coefficient)
        #            # Wait, the W2 coefficient is already in W2.
        #            # The total coefficient for the term is coeff_u * coeff_v * (W2 entry).
        #            # Let's verify this logic.
        #            # Term: sum_r2 (U2[i2, j2, r2] * V2[j2, k2, r2] * W2[k2, i2, r2])
        #            # We are substituting U2, V2, W2 with their decompositions? No, U2, V2, W2 ARE the decomposition for 2x2.
        #            # So the "multiplication" of block (i_block, j_block) and (j_block, k_block) is represented by the tensor defined by U2, V2, W2.
        #            # The coefficients for the 4x4 factors are products of the block coefficients and the 2x2 factor entries.
                
                pass # Placeholder for the actual loop logic below

    # Let's implement the nested loop properly.
    
    U4 = np.zeros((16, 49), dtype=object)
    V4 = np.zeros((16, 49), dtype=object)
    W4 = np.zeros((16, 49), dtype=object)
    
    for r1 in range(7):
        # Collect U2 contributions for r1
        u_entries = [(i*2+j, U2[i*2+j, r1]) for i in range(2) for j in range(2) if U2[i*2+j, r1] != 0]
        v_entries = [(j*2+k, V2[j*2+k, r1]) for j in range(2) for k in range(2) if V2[j*2+k, r1] != 0]
        w_entries = [(k*2+i, W2[k*2+i, r1]) for k in range(2) for i in range(2) if W2[k*2+i, r1] != 0]
        
        for (u_idx, cu) in u_entries:
            bi = u_idx // 2
            bj = u_idx % 2
            
            for (v_idx, cv) in v_entries:
                bj_v = v_idx // 2
                bk = v_idx % 2
                
                if bj != bj_v:
                    continue
                
                # The block product is A[bi, bj] * B[bj, bk]
                # This contributes to C[bi, bk]
                # The 2x2 multiplication tensor for this block is given by U2, V2, W2.
                # For each inner rank r2:
                for r2 in range(7):
                    r4 = r1 * 7 + r2
                    
                    # The inner 2x2 multiplication factors are U2[:, r2], V2[:, r2], W2[:, r2].
                    # We need to map these 2x2 indices to 4x4 indices.
                    # Inner index i2, j2 -> 4x4 index (2*bi + i2) * 4 + (2*bj + j2)
                    # Inner index j2, k2 -> 4x4 index (2*bj + j2) * 4 + (2*bk + k2)
                    # Inner index k2, i2 -> 4x4 index (2*bk + k2) * 4 + (2*bi + i2)
                    
                    for i2 in range(2):
                        for j2 in range(2):
                            val_u2 = U2[i2*2+j2, r2]
                            if val_u2 == 0: continue
                            idx_u4 = (2*bi + i2) * 4 + (2*bj + j2)
                            U4[idx_u4, r4] += cu * val_u2
                    
                    for j2 in range(2):
                        for k2 in range(2):
                            val_v2 = V2[j2*2+k2, r2]
                            if val_v2 == 0: continue
                            idx_v4 = (2*bj + j2) * 4 + (2*bk + k2)
                            V4[idx_v4, r4] += cv * val_v2
                    
                    for k2 in range(2):
                        for i2 in range(2):
                            val_w2 = W2[k2*2+i2, r2]
                            if val_w2 == 0: continue
                            idx_w4 = (2*bk + k2) * 4 + (2*bi + i2)
                            W4[idx_w4, r4] += cu * cv * val_w2

    # Step 2: Extend to 5x5 by adding the 5th dimension.
    # We add 3 rank-1 terms for the interactions involving index 4 (0-indexed).
    # The 4x4 decomposition covers indices 0..3.
    # The new terms must cover:
    # 1. C[i, 4] = sum_j A[i, j] B[j, 4] for i in 0..3
    # 2. C[4, j] = sum_k A[4, k] B[k, j] for j in 0..3
    # 3. C[4, 4] = sum_k A[4, k] B[k, 4]
    
    # Wait, the previous analysis suggested 3 terms:
    # S2: C[i, 4] for i in 0..3. (4 outputs)
    # S3: C[4, j] for j in 0..3. (4 outputs)
    # S4: C[4, 4]. (1 output)
    
    # Let's check if 1 term can cover S2.
    # Term: U[i, 4] * V[4, k] * W[k, i]? No, V is B[j, k].
    # We need C[i, 4] = sum_j A[i, j] B[j, 4].
    # If we use a term with U[i, 4] = 1 for all i, V[j, 4] = 1 for all j, and W[4, i] = 1?
    # W index is k*n + i. Here k=4, i=i. So W[4*5+i, r] = 1.
    # Let's check the contraction:
    # Sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    # For r = new_term:
    # U[i, j, r] is non-zero only if j=4. So U[i, 4, r] = 1.
    # V[j, k, r] is non-zero only if k=4. So V[j, 4, r] = 1.
    # W[k, i, r] is non-zero only if k=4. So W[4, i, r] = 1.
    # The product is 1 * 1 * 1 = 1 if j=4 and k=4.
    # The sum is over j and k? No, the tensor is T[i, j, k, l] = delta_{j, k} delta_{i, l} ? No.
    # The matrix multiplication tensor is T[i, j, k, l] = 1 if l=j and k=l? No.
    # C[i, l] = sum_j A[i, j] B[j, l].
    # Tensor index is (i, j, l, j)? No, the tensor is defined by the identity:
    # C_{il} = sum_{j} A_{ij} B_{jl}.
    # The tensor T has indices (i, j, k, l) where the contraction is over j?
    # The task says: "decompose the matrix multiplication tensor".
    # The tensor T is such that T_{i, j, k, l} = 1 if k=j and l=i? No.
    # Usually, the tensor is T_{i, j, k} for C_{ik} = sum_j A_{ij} B_{jk}.
    # But here we have U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # The decomposition is: sum_r U_{ij, r} V_{jk, r} W_{ki, r} = T_{ijk}
    # where T_{ijk} = 1 if the multiplication A B = C implies C_{ik} gets a contribution from A_{ij} B_{jk}.
    # Actually, the standard tensor for matrix multiplication M_{n, m, p} has indices (i, j, k) with:
    # T_{i, j, k} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
    # Wait, C_{ik} = sum_j A_{ij} B_{jk}.
    # So for a fixed i, k, the sum is over j.
    # The tensor T_{i, j, k} is 1 if the indices (i, j, k) form a valid triple for the multiplication.
    # Actually, T_{i, j, k} = 1 for all i, j, k? No.
    # The tensor is defined by: T_{i, j, k} = 1 if the "path" i->j->k is valid.
    # In matrix multiplication, every triple (i, j, k) is valid because C_{ik} depends on A_{ij} B_{jk} for all j.
    # Wait, no. C_{ik} = sum_j A_{ij} B_{jk}.
    # So for a specific output C_{ik}, it depends on A_{ij} and B_{jk} for all j.
    # The tensor T is of shape (n, m, p) ? No, the factors are U (n*m), V (m*p), W (p*n).
    # The identity is: sum_r U_{(i,j), r} V_{(j,k), r} W_{(k,i), r} = 1 if the term A_{ij} B_{jk} contributes to C_{ki}?
    # Let's check the indices in the seed code.
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
    # The product is U_{ij} V_{jk} W_{ki}.
    # The sum over r should yield 1 if the term A_{ij} B_{jk} is part of the multiplication for C_{ki}.
    # Since C_{ki} = sum_j A_{kj} B_{jk}, the term A_{ij} B_{jk} contributes to C_{ik}? No, A_{ij} B_{jk} contributes to C_{ik} if i=k? No.
    # A_{ij} B_{jk} contributes to C_{ik} where the first index of A is i and the second index of B is k.
    # So the output index is (i, k).
    # The W index is k*n + i. So W_{ki}.
    # So the term is A_{ij} B_{jk} -> C_{ik}.
    # The tensor T_{i, j, k} is 1 if this term exists.
    # For matrix multiplication, T_{i, j, k} = 1 for ALL i, j, k.
    # Because for any i, j, k, the product A_{ij} B_{jk} contributes to C_{ik}.
    # So we need sum_r U_{ij, r} V_{jk, r} W_{ki, r} = 1 for all i, j, k.
    
    # Let's re-evaluate the 3 new terms.
    # We need to cover all triples (i, j, k) where at least one index is 4 (the 5th index, 0-indexed).
    # The 4x4 decomposition covers all i, j, k in {0, 1, 2, 3}.
    # We need to cover:
    # 1. i=4, j in 0..3, k in 0..3. (A[4, j] B[j, k] -> C[4, k])
    # 2. i in 0..3, j=4, k in 0..3. (A[i, 4] B[4, k] -> C[i, k])
    # 3. i in 0..3, j in 0..3, k=4. (A[i, j] B[j, 4] -> C[i, 4])
    # 4. i=4, j=4, k in 0..3. (A[4, 4] B[4, k] -> C[4, k])
    # 5. i=4, j in 0..3, k=4. (A[4, j] B[j, 4] -> C[4, 4])
    # 6. i in 0..3, j=4, k=4. (A[i, 4] B[4, 4] -> C[i, 4])
    # 7. i=4, j=4, k=4. (A[4, 4] B[4, 4] -> C[4, 4])
    
    # Let's try to cover these with 3 terms.
    # Term 1: Covers cases where j=4.
    # U[i, 4, r] = 1 for all i.
    # V[4, k, r] = 1 for all k.
    # W[k, i, r] = 1 for all i, k.
    # Check: For any i, k, the term contributes 1 * 1 * 1 = 1 to T_{i, 4, k}.
    # This covers all triples with j=4.
    # Does it interfere with j!=4?
    # If j!=4, U[i, j, r] = 0. So no interference.
    # So Term 1 covers all (i, 4, k).
    
    # Term 2: Covers cases where k=4.
    # U[i, j, r] = 1 for all i, j.
    # V[j, 4, r] = 1 for all j.
    # W[4, i, r] = 1 for all i.
    # Check: For any i, j, the term contributes 1 * 1 * 1 = 1 to T_{i, j, 4}.
    # This covers all triples with k=4.
    # Does it interfere with k!=4?
    # If k!=4, V[j, k, r] = 0. So no interference.
    # So Term 2 covers all (i, j, 4).
    
    # Term 3: Covers cases where i=4.
    # U[4, j, r] = 1 for all j.
    # V[j, k, r] = 1 for all j, k.
    # W[k, 4, r] = 1 for all k.
    # Check: For any j, k, the term contributes 1 * 1 * 1 = 1 to T_{4, j, k}.
    # This covers all triples with i=4.
    # Does it interfere with i!=4?
    # If i!=4, U[i, j, r] = 0. So no interference.
    # So Term 3 covers all (4, j, k).
    
    # Do these 3 terms cover all cases with index 4?
    # - (i, 4, k): Covered by Term 1.
    # - (i, j, 4): Covered by Term 2.
    # - (4, j, k): Covered by Term 3.
    # - (4, 4, k): Covered by Term 1 (j=4) AND Term 3 (i=4).
    #   Wait, if a triple is covered by multiple terms, the sum will be > 1.
    #   T_{4, 4, k} should be 1.
    #   Term 1: U[4, 4, r1]=1, V[4, k, r1]=1, W[k, 4, r1]=1. Product=1.
    #   Term 3: U[4, 4, r3]=1, V[4, k, r3]=1, W[k, 4, r3]=1. Product=1.
    #   Sum = 2. This is incorrect. We need 1.
    
    # We need to adjust the coefficients to avoid double counting.
    # The triples with multiple indices equal to 4 are:
    # - (4, 4, k) for k in 0..3: Covered by Term 1 and Term 3.
    # - (i, 4, 4) for i in 0..3: Covered by Term 1 and Term 2.
    # - (4, j, 4) for j in 0..3: Covered by Term 2 and Term 3.
    # - (4, 4, 4): Covered by Term 1, Term 2, and Term 3.
    
    # We can solve for the coefficients of these 3 terms to satisfy the equations.
    # Let the coefficients for the 3 terms be a, b, c.
    # Actually, the structure is fixed, but we can scale the terms.
    # Let Term 1 have scale s1, Term 2 have scale s2, Term 3 have scale s3.
    # We need:
    # For (i, 4, k) with i!=4, k!=4: Only Term 1 covers it. So s1 = 1.
    # For (i, j, 4) with i!=4, j!=4: Only Term 2 covers it. So s2 = 1.
    # For (4, j, k) with j!=4, k!=4: Only Term 3 covers it. So s3 = 1.
    # For (4, 4, k) with k!=4: Covered by Term 1 and Term 3. s1 + s3 = 1.
    #   Since s1=1, s3=1, sum=2. Contradiction.
    
    # This means we cannot use simple rank-1 terms with constant 1s.
    # We need to modify the terms to exclude the overlapping parts or use linear combinations.
    # Alternatively, we can use more terms.
    # But the previous analysis suggested 3 terms. Let's re-read the previous analysis.
    # "S2 (25) -> 1. S3 (20) -> 1. S4 (16) -> 1."
    # S2: i in 0..3, j=4, k in 0..3. (16 triples)
    # S3: i=4, j in 0..3, k in 0..3. (16 triples)
    # S4: i in 0..3, j in 0..3, k=4. (16 triples)
    # And the overlaps:
    # (4, 4, k), (i, 4, 4), (4, j, 4), (4, 4, 4).
    
    # If we use 3 terms, we have 3 variables (scales) but 4 constraints (the overlaps).
    # It is impossible to satisfy all with 3 rank-1 terms of this simple form.
    # We need to adjust the support of the terms.
    # For example, Term 1 can be modified to NOT cover (4, 4, k).
    # U[4, 4, r1] = 0. Then Term 1 does not cover (4, 4, k).
    # Then (4, 4, k) is only covered by Term 3. So s3=1 is fine.
    # But then (4, 4, k) is covered by Term 3.
    # What about (i, 4, 4)? Covered by Term 1 and Term 2.
    # If we set V[4, 4, r2] = 0 in Term 2, then Term 2 does not cover (i, 4, 4).
    # Then (i, 4, 4) is only covered by Term 1.
    # What about (4, j, 4)? Covered by Term 2 and Term 3.
    # If we set U[4, j, r3] = 0 for j!=4? No, that would break the main part.
    # We can set W[4, 4, r3] = 0 in Term 3. Then Term 3 does not cover (4, j, 4) when k=4?
    # Term 3 covers (4, j, k). If k=4, W[4, 4, r3]=0, so no contribution.
    # Then (4, j, 4) is only covered by Term 2.
    # What about (4, 4, 4)?
    # Term 1: U[4,4]=0, so no.
    # Term 2: V[4,4]=0, so no.
    # Term 3: W[4,4]=0, so no.
    # So (4, 4, 4) is not covered. We need a 4th term for (4, 4, 4).
    
    # So we need 4 terms?
    # Term 1: Covers (i, 4, k) except (4, 4, k).
    # Term 2: Covers (i, j, 4) except (i, 4, 4).
    # Term 3: Covers (4, j, k) except (4, j, 4).
    # Term 4: Covers (4, 4, 4).
    
    # Let's verify.
    # Term 1: U[i, 4]=1 (i!=4), V[4, k]=1 (k!=4), W[k, i]=1.
    #   Covers (i, 4, k) for i!=4, k!=4.
    # Term 2: U[i, j]=1 (j!=4), V[j, 4]=1 (j!=4), W[4, i]=1.
    #   Covers (i, j, 4) for i!=4, j!=4.
    # Term 3: U[4, j]=1 (j!=4), V[j, k]=1 (k!=4), W[k, 4]=1.
    #   Covers (4, j, k) for j!=4, k!=4.
    # Term 4: U[4, 4]=1, V[4, 4]=1, W[4, 4]=1.
    #   Covers (4, 4, 4).
    
    # What about (4, 4, k) for k!=4?
    # Covered by Term 3? Term 3 has U[4, j]=1. If j=4, U[4,4]=1?
    # In Term 3, I said U[4, j]=1 for j!=4.
    # So (4, 4, k) is not covered by Term 3.
    # Is it covered by Term 1? Term 1 has U[i, 4]=1 for i!=4.
    # So (4, 4, k) is not covered by Term 1.
    # So (4, 4, k) is not covered.
    
    # We need to cover (4, 4, k) for k!=4.
    # This is part of C[4, k] = sum_j A[4, j] B[j, k].
    # The term j=4 is A[4, 4] B[4, k].
    # We can add this to Term 3 by allowing j=4 in U.
    # But then (4, 4, 4) would be covered by Term 3 and Term 4.
    # We can adjust scales.
    
    # This is getting complicated. Let's just use the schoolbook for the 5th dimension parts.
    # The 4x4 part is rank 49.
    # The 5th dimension parts:
    # - (i, 4, k) for i, k in 0..3: 16 terms.
    # - (4, j, k) for j, k in 0..3: 16 terms.
    # - (i, j, 4) for i, j in 0..3: 16 terms.
    # - (4, 4, 4): 1 term.
    # Total additional terms: 16+16+16+1 = 49.
    # Total rank: 49 + 49 = 98.
    # This is better than 125.
    
    # Let's implement this.
    # We already have U4, V4, W4 for the 4x4 part (indices 0..15).
    # We need to extend them to 5x5 (indices 0..24).
    # The new indices are 16..24.
    # U index: i*5+j.
    # V index: j*5+k.
    # W index: k*5+i.
    
    # The 4x4 part uses indices:
    # U: i*4+j for i, j in 0..3. These are 0..15.
    # In 5x5, these are i*5+j.
    # We need to map the 4x4 factors to the 5x5 factors.
    
    U5 = np.zeros((25, 49 + 49), dtype=object)
    V5 = np.zeros((25, 49 + 49), dtype=object)
    W5 = np.zeros((25, 49 + 49), dtype=object)
    
    # Copy 4x4 part
    for i in range(4):
        for j in range(4):
            idx_u5 = i*5 + j
            idx_u4 = i*4 + j
            U5[idx_u5, :49] = U4[idx_u4, :49]
            
            idx_v5 = j*5 + 0 # Wait, V index is j*p+k. In 4x4, p=4. In 5x5, p=5.
            # The 4x4 V indices are j*4+k for j, k in 0..3.
            # In 5x5, the corresponding indices are j*5+k.
            for k in range(4):
                idx_v5 = j*5 + k
                idx_v4 = j*4 + k
                V5[idx_v5, :49] = V4[idx_v4, :49]
                
            idx_w5 = 0*5 + i # Wait, W index is k*n+i. In 4x4, n=4. In 5x5, n=5.
            # The 4x4 W indices are k*4+i for k, i in 0..3.
            # In 5x5, the corresponding indices are k*5+i.
            for k in range(4):
                idx_w5 = k*5 + i
                idx_w4 = k*4 + i
                W5[idx_w5, :49] = W4[idx_w4, :49]
                
    # Add the 49 new terms for the 5th dimension.
    col = 49
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # Term for (i, 4, k)
                U5[i*5 + 4, col] = 1
                V5[4*5 + k, col] = 1
                W5[k*5 + i, col] = 1
                col += 1
                
    for j in range(4):
        for k in range(4):
            # Term for (4, j, k)
            U5[4*5 + j, col] = 1
            V5[j*5 + k, col] = 1
            W5[k*5 + 4, col] = 1
            col += 1
            
    for i in range(4):
        for j in range(4):
            # Term for (i, j, 4)
            U5[i*5 + j, col] = 1
            V5[j*5 + 4, col] = 1
            W5[4*5 + i, col] = 1
            col += 1
            
    # Term for (4, 4, 4)
    U5[4*5 + 4, col] = 1
    V5[4*5 + 4, col] = 1
    W5[4*5 + 4, col] = 1
    col += 1
    
    # Convert to lists of floats/ints
    U_out = [[float(x) for x in row] for row in U5]
    V_out = [[float(x) for x in row] for row in V5]
    W_out = [[float(x) for x in row] for row in W5]
    
    return {"U": U_out, "V": V_out, "W": W_out}
