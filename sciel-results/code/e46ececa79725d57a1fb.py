"""Matrix multiplication 4,5,5: structured Strassen + schoolbook decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]
_DEADLINE = time.time() + 130  # safety margin under 150s limit


def _verify(U, V, W, n, m, p):
    """Check that sum_c U[:,c]*V[:,c]*W[:,c] equals the matmul tensor."""
    rank = len(U[0])
    # Build the target tensor T[k*n+i, j*m+k] = delta_{i,k} ... wait.
    # Actually the tensor for (n,m,p) is T[i*m+j, k*p+i... 
    # Let me re-read: output factor W[k*n+i, r]
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # The product sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal the matmul tensor.
    # For matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i, j, k] = delta_{i,j} * delta_{j,k}? No.
    # Actually the multiplication tensor has T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k].
    # The decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,j,k) 
    # where delta is 1 when the indices correspond to a valid term in the product.
    # More precisely: for each (i,j,k), the value should be 1 if j is the "connecting" index,
    # but actually it's always 1 for all (i,j,k) in the sense that...
    # No. The tensor T_{i,j,k} = 1 for all i,j,k? That can't be right.
    # 
    # Let me think again. The multiplication map is:
    # (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # As a bilinear map, this is a tensor in U^* ⊗ V^* ⊗ W^* where:
    # U has basis indexed by (i,j) for A, V by (j,k) for B, W by (i,k) for C.
    # T[i,j; j,k; i,k] = delta_{i,i'} delta_{k,k'} ... 
    # Actually T[(i,j),(j,k),(i,k)] = 1 when the first index of second factor = second index of first factor
    # and the output indices match.
    # 
    # The standard form: T_{(i,j),(j',k),(i',k')} = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # But with the constraint that j = j' (the shared index). So:
    # T_{(i,j),(j',k),(i',k')} = 1 if i=i', j=j', k=k' (with j being the shared index).
    # 
    # In our indexing: U row = i*m+j, V row = j*p+k, W row = k*n+i.
    # So T[U_row, V_row, W_row] = 1 if the indices are consistent, i.e.,
    # from U row: i, j; from V row: j', k; from W row: k', i'
    # We need i = i', j = j', k = k'.
    # Since U encodes (i,j), V encodes (j',k), W encodes (k',i'):
    # T = 1 iff i=i' AND j=j' AND k=k'.
    # 
    # For the schoolbook: each term is a rank-1 tensor with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    # for a specific (i,j,k). The sum over all (i,j,k) gives the full tensor.
    
    # Verify: build T as a 3D array of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Compute sum_r U[:,r] V[:,r] W[:,r] as a 3D tensor via outer products
    # Use einsum: result[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    Uarr = np.array(U, dtype=np.float64)
    Varr = np.array(V, dtype=np.float64)
    Warr = np.array(W, dtype=np.float64)
    
    # This could be expensive for large rank; use einsum
    result = np.einsum('ar,br,cr->abc', Uarr, Varr, Warr)
    
    diff = np.max(np.abs(result - T))
    return diff < 1e-9


def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 matrix multiplication.
    U shape (4, 7), V shape (4, 7), W shape (4, 7)
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 products for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # For each of the 7 sub-products, we need a rank-1 tensor for 2x2x2.
    # But wait, each sub-product is itself a 2x2x2 multiplication, which has rank 7 (or 8 schoolbook).
    # Actually no - in Strassen, the 7 "products" M1..M7 are each a full 2x2x2 multiplication
    # of linear combinations. So the total rank is 7 * (rank of 2x2x2) = 7*7 = 49 if using
    # schoolbook for the inner, or 7*7 = 49.
    #
    # Actually, the Strassen algorithm says: to multiply two 2x2 matrices, compute 7 products
    # of (linear combos of 2x2 blocks) * (linear combos of 2x2 blocks), where each "product"
    # is a 2x2x2 multiplication. So the total rank is 7 * 7 = 49.
    #
    # Let me implement this properly.
    #
    # The 7 linear combinations for A (2x2):
    # For M1: A' = A00 + A11, B' = B00 + B11
    # For M2: A' = A10 + A11, B' = B00
    # For M3: A' = A00, B' = B01 - B11
    # For M4: A' = A11, B' = B10 - B00
    # For M5: A' = A00 + A01, B' = B11
    # For M6: A' = A10 - A00, B' = B00 + B01
    # For M7: A' = A01 - A11, B' = B10 + B11
    #
    # Each A' is a 2x2 matrix formed by linear combination of entries.
    # Each B' similarly.
    #
    # Then C = combine the 7 resulting 2x2 matrices.
    
    # Represent each 2x2 linear combo as coefficients on (a00, a01, a10, a11)
    # A' entries:
    # A'_00 = s00*a00 + s01*a01 + s10*a10 + s11*a11
    # A'_01 = t00*a00 + t01*a01 + t10*a10 + t11*a11
    # A'_10 = u00*a00 + u01*a01 + u10*a10 + u11*a11
    # A'_11 = v00*a00 + v01*a01 + v10*a10 + v11*a11
    
    # For the 7 combos:
    # M1: A' = [[1,0],[0,1]] (A00+A11 means A'_00=A00, A'_01=A01... wait no.
    # Actually A is 2x2: A = [[A00, A01],[A10, A11]]
    # A00+A11 is not a matrix, it's confusing notation. Let me use block notation.
    # 
    # In Strassen for 2x2x2, we split the 2x2 matrices into 1x1 blocks:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # 
    # So each M_i is a product of two scalars (linear combos).
    # The rank of each scalar product is 1. Total rank = 7.
    # 
    # So for 2x2x2, Strassen gives rank 7!
    # 
    # Now for 4x4x4 using Strassen: split into 2x2 blocks.
    # A (4x4) = [[A11, A12], [A21, A22]] where each block is 2x2.
    # B (4x4) = [[B11, B12], [B21, B22]]
    # 
    # M1 = (A11+A22)(B11+B22)  -- product of 2x2 matrices, rank 7
    # M2 = (A21+A22)B11       -- product of 2x2 matrices, rank 7
    # M3 = A11(B12-B22)       -- product of 2x2 matrices, rank 7
    # M4 = A22(B21-B11)       -- product of 2x2 matrices, rank 7
    # M5 = (A11+A12)B22       -- product of 2x2 matrices, rank 7
    # M6 = (A21-A11)(B11+B12) -- product of 2x2 matrices, rank 7
    # M7 = (A12-A22)(B21+B22) -- product of 2x2 matrices, rank 7
    # 
    # Total rank = 7 * 7 = 49.
    # 
    # Each of the 7 block products is a 2x2x2 multiplication with Strassen rank 7.
    # The linear combinations of blocks just add up the corresponding U,V,W entries.
    
    # First, get the 2x2x2 Strassen decomposition (rank 7)
    # U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r] for i,j,k in {0,1}, r in {0,...,6}
    
    # For 2x2x2 Strassen:
    # The 7 products correspond to:
    # r=0: (a+d)(e+h) -> U: a and d have coeff 1; V: e and h have coeff 1; W: C00 and C11 get +1
    # Let me just construct it directly.
    
    # 2x2x2 tensor: T[a_idx, b_idx, c_idx] where a_idx = i*2+j, b_idx = j*2+k, c_idx = k*2+i
    # T is 1 when (i,j,k) is consistent.
    # 
    # Strassen rank-7 decomposition:
    # Term 0: M1 = (a+d)(e+h) contributes to C00 and C11
    #   U: a(0,0) -> (i=0,j=0): row 0; d(1,1) -> (i=1,j=1): row 3
    #   V: e(0,0) -> (j=0,k=0): row 0; h(1,1) -> (j=1,k=1): row 3
    #   W: C00 -> (k=0,i=0): row 0; C11 -> (k=1,i=1): row 3
    #   But M1 = (a+d)(e+h) = ae + ah + de + dh
    #   This is NOT a rank-1 term! It's a sum of 4 rank-1 terms.
    #   
    # I think I'm confusing myself. Let me restart.
    #
    # The multiplication tensor for 2x2x2 has 8 entries (i,j,k) in {0,1}^3.
    # T[i,j,k] = 1 for all i,j,k? No.
    # T[(i,j),(j,k),(i,k)]... let me use the convention from the problem.
    # U row = i*m+j = i*2+j, V row = j*p+k = j*2+k, W row = k*n+i = k*2+i
    # T[U_row, V_row, W_row] = 1 iff the indices are consistent.
    # U_row = i*2+j encodes (i,j), V_row = j'*2+k encodes (j',k), W_row = k'*2+i' encodes (k',i')
    # T = 1 iff i=i', j=j', k=k'.
    # 
    # So T is the identity-like tensor: T[a,b,c] = 1 iff 
    #   (a // 2, a % 2) = (c % 2, b // 2) ... this is getting complicated.
    # 
    # Let me just enumerate:
    # (i,j,k) -> U=i*2+j, V=j*2+k, W=k*2+i
    # (0,0,0): U=0, V=0, W=0
    # (0,0,1): U=0, V=1, W=2
    # (0,1,0): U=1, V=0, W=0
    # (0,1,1): U=1, V=1, W=2
    # (1,0,0): U=2, V=0, W=1
    # (1,0,1): U=2, V=1, W=3
    # (1,1,0): U=3, V=2, W=1
    # (1,1,1): U=3, V=3, W=3
    
    # So T has 1s at these 8 positions.
    # 
    # Strassen rank-7: we want 7 rank-1 tensors that sum to T.
    # This is known to exist. Let me look up the explicit form.
    #
    # Actually, for the purposes of this problem, let me just use schoolbook for 2x2x2 (rank 8)
    # and build up from there. The total rank for 4x4x4 would be 7*8=56, and then add the
    # cross terms. Or I can implement Strassen properly.
    #
    # Let me try a different approach: just do schoolbook for everything but use
    # Strassen-like structure where possible.
    #
    # Actually, the simplest improvement: use Strassen for the 4x4x4 core.
    # Total rank = 49 (Strassen 4x4x4) + 16 (4x4x1 schoolbook) + 20 (4x1x5 schoolbook) = 85.
    # Wait, that's not quite right. Let me think about the block structure.
    #
    # 4x5x5: A is 4x5, B is 5x5.
    # Split B into B = [B1 B2] where B1 is 5x3 and B2 is 5x2? No, 5 = 3+2 or 5=4+1.
    # Let's split as: A is 4x5, B is 5x5.
    # C = A * B, C is 4x5.
    # 
    # Split the 5 columns of A and rows of B: A = [A1 A2] (4x3, 4x2), B = [B1; B2] (3x5, 2x5)
    # C = A1*B1 + A2*B2
    # 
    # A1*B1: 4x3 * 3x5 -> 4x5, rank 4*3*5 = 60 schoolbook
    # A2*B2: 4x2 * 2x5 -> 4x5, rank 4*2*5 = 40 schoolbook
    # Total: 100. Same as before.
    #
    # Better: use Strassen on the 4x4x4 part.
    # A = [A1 A2] where A1 is 4x4, A2 is 4x1.
    # B = [B1; B2] where B1 is 4x5, B2 is 1x5.
    # C = A1*B1 + A2*B2
    #
    # A1*B1: 4x4 * 4x5. Split B1 = [B1a B1b] where B1a is 4x4, B1b is 4x1.
    # A1*B1a: 4x4*4x4, rank 49 (Strassen)
    # A1*B1b: 4x4*4x1, rank 16 (schoolbook)
    # A2*B2: 4x1*1x5, rank 20 (schoolbook)
    # Total: 49 + 16 + 20 = 85.
    
    # Now I need to implement the Strassen 4x4x4 decomposition.
    # This requires the 2x2x2 Strassen (rank 7) decomposition.
    #
    # Let me hardcode the 2x2x2 Strassen rank-7 decomposition.
    # After research, the 7 terms for 2x2x2 Strassen are:
    # (using 0-indexed i,j,k):
    #
    # I'll construct it numerically.
    
    if time.time() > _DEADLINE:
        return None
    
    # 2x2x2 Strassen rank-7:
    # U2: 4x7, V2: 4x7, W2: 4x7
    # From known results:
    U2 = [
        [1, 1, 1, 0, 1, 1, 1],   # row 0: (i=0,j=0)
        [0, 1, 0, 1, 0, 1, 0],   # row 1: (i=0,j=1)
        [0, 0, 1, 1, 1, 0, 1],   # row 2: (i=1,j=0)
        [1, 0, 0, 1, 0, 0, 1],   # row 3: (i=1,j=1)
    ]
    V2 = [
        [1, 1, 1, 0, 1, 1, 0],   # row 0: (j=0,k=0)
        [0, 0, 1, 1, 0, 1, 1],   # row 1: (j=0,k=1)
        [0, 1, 0, 1, 1, 0, 1],   # row 2: (j=1,k=0)
        [1, 0, 1, 0, 0, 1, 0],   # row 3: (j=1,k=1)
    ]
    W2 = [
        [1, 0, 0, 1, 0, 1, 1],   # row 0: (k=0,i=0)
        [0, 0, 1, 0, 1, 0, 0],   # row 1: (k=0,i=1)
        [0, 1, 0, 0, 0, 1, 0],   # row 2: (k=1,i=0)
        [1, 1, 1, 0, 1, 0, 1],   # row 3: (k=1,i=1)
    ]
    
    # Verify 2x2x2
    # Actually, let me not trust my memory and instead verify.
    # For now, let's just use schoolbook for 2x2x2 (rank 8) to be safe,
    # giving 4x4x4 rank 56. Total: 56+16+20 = 92.
    # Or try Strassen and verify.
    
    # Let me verify the above Strassen 2x2x2:
    T2 = np.zeros((4, 4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T2[i*2+j, j*2+k, k*2+i] = 1.0
    U2a = np.array(U2, dtype=np.float64)
    V2a = np.array(V2, dtype=np.float64)
    W2a = np.array(W2, dtype=np.float64)
    res2 = np.einsum('ar,br,cr->abc', U2a, V2a, W2a)
    if np.max(np.abs(res2 - T2)) > 1e-9:
        # Strassen 2x2x2 is wrong, use schoolbook
        U2 = [[1 if (i*2+j) == idx else 0 for idx in range(8)] for i in range(2) for j in range(2)]
        V2 = [[1 if (j*2+k) == idx else 0 for idx in range(8)] for j in range(2) for k in range(2)]
        W2 = [[1 if (k*2+i) == idx else 0 for idx in range(8)] for i in range(2) for k in range(2)]
        inner_rank = 8
    else:
        inner_rank = 7
    
    # Build 4x4x4 Strassen decomposition
    # 7 block products, each with inner_rank terms
    U4 = [[0] * (7 * inner_rank) for _ in range(16)]
    V4 = [[0] * (7 * inner_rank) for _ in range(16)]
    W4 = [[0] * (7 * inner_rank) for _ in range(16)]
    
    # The 7 Strassen combinations for 4x4:
    # Each combo specifies how to form A' and B' from the 2x2 blocks of A and B.
    # A blocks: A11(top-left), A12(top-right), A21(bottom-left), A22(bottom-right)
    # Each block is 2x2, so A is 4x4 with entries A[r,c] where r,c in 0..3.
    # Block (bi,bj) has rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
    
    # For each of the 7 products, define the linear combination coefficients
    # for A blocks and B blocks.
    # 
    # M1: A' = A11 + A22, B' = B11 + B22
    # M2: A' = A21 + A22, B' = B11
    # M3: A' = A11, B' = B12 - B22
    # M4: A' = A22, B' = B21 - B11
    # M5: A' = A11 + A12, B' = B22
    # M6: A' = A21 - A11, B' = B11 + B12
    # M7: A' = A12 - A22, B' = B21 + B22
    #
    # A' is a 4x4 matrix formed by placing the linear combos of 2x2 blocks.
    # For M1: A' = [[A11, 0], [0, A22]] + [[0, 0], [0, 0]]... no.
    # A' for M1: the top-left block is A11, top-right is 0, bottom-left is 0, bottom-right is A22.
    # Wait, that's not right either. A11+A22 as a 4x4 matrix means:
    # A' = [[A11, A12'], [A21', A22]] where A12' and A21' are zero? No.
    # 
    # In Strassen, when we say A' = A11 + A22, we mean the 4x4 matrix:
    # [[A11, 0], [0, A22]]? No, that's not standard.
    # 
    # Actually in the standard Strassen for block matrices:
    # A = [[A11, A12], [A21, A22]]
    # The linear combinations are:
    # M1 = (A11 + A22)(B11 + B22)
    # Here A11+A22 is a 2x2 matrix (if blocks are 1x1) or a 2m x 2m matrix (if blocks are mxm).
    # For our case, blocks are 2x2, so A11+A22 is a 2x2 matrix (sum of two 2x2 matrices).
    # But we need a 4x4 matrix to multiply. So A' is formed by:
    # A' = [[A11+A22, 0], [0, 0]]? No.
    # 
    # I think the correct interpretation is:
    # A' is a 4x4 matrix where:
    # - Top-left 2x2 block = A11 + A22
    # - All other blocks = 0
    # Similarly for B'.
    # Then M1 = A' * B' is a 4x4 matrix, and we extract the relevant parts.
    # 
    # Hmm, but that doesn't seem right for Strassen. Let me reconsider.
    #
    # In standard Strassen for 2x2 blocks of size m:
    # A is 2m x 2m, split into four m x m blocks.
    # M1 = (A11 + A22)(B11 + B22) is an m x m product.
    # So each M_i is an m x m matrix product.
    # For our case, m=2, so each M_i is a 2x2 product, and we compute all 7 of them.
    # The total rank is 7 * rank(2x2x2) = 7 * 7 = 49.
    #
    # So for the 4x4x4 decomposition:
    # For each of the 7 Strassen products:
    #   - Form A' (2x2) as linear combo of A's 2x2 blocks
    #   - Form B' (2x2) as linear combo of B's 2x2 blocks  
    #   - Compute M = A' * B' using the 2x2x2 decomposition (rank 7)
    #   - Add to C using the appropriate linear combination
    #
    # The U, V, W matrices for 4x4x4:
    # U has 16 rows (4x4 entries of A), V has 16 rows (4x4 entries of B), W has 16 rows (4x4 entries of C).
    # 
    # For each Strassen product i (0..6):
    #   For each of the 7 inner terms (r=0..6):
    #     column = i * 7 + r
    #     U[entry, column] = coeff_A * U2[inner_entry, r]
    #     V[entry, column] = coeff_B * V2[inner_entry, r]
    #     W[entry, column] = coeff_C * W2[inner_entry, r]
    #
    # Where coeff_A depends on which A block entry corresponds to which A' entry.
    
    # Define the 7 Strassen combinations:
    # Each is (A_blocks, B_blocks, C_combination)
    # A_blocks: dict from block_pos to coefficient
    # B_blocks: dict from block_pos to coefficient
    # C_combination: for each of the 7 inner C entries, which C block and coefficient
    
    # Actually, let me think about this differently.
    # 
    # For Strassen product i, we have:
    # A' is a 2x2 matrix. Entry A'[a,b] (a,b in {0,1}) is a linear combo of A's entries.
    # Specifically, A'[a,b] corresponds to a specific block of A.
    # 
    # For M1: A' = A11 + A22 (as a 2x2 matrix, this means A' IS the sum, so A'[0,0]=A11[0,0]+A22[0,0], etc.)
    # Wait, A11 and A22 are both 2x2 matrices, so A11+A22 is a 2x2 matrix.
    # A'[r,c] = A11[r,c] + A22[r,c] for r,c in {0,1}.
    # A11[r,c] = A[2*0+r, 2*0+c] = A[r,c] (top-left block of A)
    # A22[r,c] = A[2*1+r, 2*1+c] = A[2+r, 2+c] (bottom-right block of A)
    # So A'[r,c] = A[r,c] + A[2+r, 2+c].
    # 
    # Similarly for other combinations.
    # 
    # For the inner 2x2x2 decomposition, U2[a*2+b, r] gives the coefficient for A'[a,b] in term r.
    # So U[A_global_row, column] = sum over inner entries of U2[inner_row, r] * coeff.
    # Where A_global_row corresponds to A[global_r, global_c].
    # 
    # Let me define for each Strassen product i:
    # a_coeff[global_r, global_c, inner_r, inner_c] = coefficient
    # But this is getting complex. Let me just build it step by step.
    
    # For each of the 7 products, define:
    # A_map: mapping from (inner_row, inner_col) -> list of (global_row, global_col, coeff)
    # B_map: mapping from (inner_row, inner_col) -> list of (global_row, global_col, coeff)
    # C_map: mapping from (inner_row, inner_col) -> list of (global_row, global_col, coeff)
    
    # M1: A' = A11 + A22, B' = B11 + B22
    # A'[r,c] = A[r,c] + A[2+r, 2+c]
    # B'[r,c] = B[r,c] + B[2+r, 2+c]
    # C contribution: C11 += M1, C22 += M1 (in block terms)
    # C[2*bi+r, 2*bj+c] gets contribution from M1[r,c] for the appropriate block.
    # From Strassen: C11 = M1 + M4 - M5 + M7, C12 = M3 + M5, C21 = M2 + M4, C22 = M1 - M2 + M3 + M6
    # So C block (0,0) gets M1, C block (1,1) gets M1.
    # C[2*0+r, 2*0+c] += M1[r,c] and C[2*1+r, 2*1+c] += M1[r,c].
    
    # This is very involved. Let me just code it up carefully.
    
    if time.time() > _DEADLINE:
        return None
    
    # Define the 7 Strassen products with their A, B, C linear combinations
    # For each product, A and B are 2x2 matrices formed from 2x2 blocks of the 4x4 matrices.
    # C is a 2x2 matrix of results, which gets added to the appropriate 2x2 blocks of the 4x4 C.
    
    # A_comb[i][a, b] = list of (block_row, block_col, coeff)
    # where block_row, block_col are in {0,1} and refer to the 2x2 blocks.
    # A'[a,b] = sum over (br, bc, coeff) of coeff * A_block[br,bc][a,b]
    # A_block[br,bc][a,b] = A[2*br+a, 2*bc+b]
    
    products = []
    # M1: A' = A11 + A22, B' = B11 + B22, C: (0,0)+=M, (1,1)+=M
    products.append({
        'A': {(0,0): [(0,0,1), (1,1,1)], (0,1): [(0,0,1), (1,1,1)], (1,0): [(0,0,1), (1,1,1)], (1,1): [(0,0,1), (1,1,1)]},
        'B': {(0,0): [(0,0,1), (1,1,1)], (0,1): [(0,0,1), (1,1,1)], (1,0): [(0,0,1), (1,1,1)], (1,1): [(0,0,1), (1,1,1)]},
        'C': {(0,0): 1, (1,1): 1}
    })
    # M2: A' = A21 + A22, B' = B11, C: (1,0)+=M
    products.append({
        'A': {(0,0): [(1,0,1), (1,1,1)], (0,1): [(1,0,1), (1,1,1)], (1,0): [(1,0,1), (1,1,1)], (1,1): [(1,0,1), (1,1,1)]},
        'B': {(0,0): [(0,0,1)], (0,1): [(0,0,1)], (1,0): [(0,0,1)], (1,1): [(0,0,1)]},
        'C': {(1,0): 1}
    })
    # M3: A' = A11, B' = B12 - B22, C: (0,1)+=M
    products.append({
        'A': {(0,0): [(0,0,1)], (0,1): [(0,0,1)], (1,0): [(0,0,1)], (1,1): [(0,0,1)]},
        'B': {(0,0): [(0,1,1), (1,1,-1)], (0,1): [(0,1,1), (1,1,-1)], (1,0): [(0,1,1), (1,1,-1)], (1,1): [(0,1,1), (1,1,-1)]},
        'C': {(0,1): 1}
    })
    # M4: A' = A22, B' = B21 - B11, C: (0,0)+=M, (1,0)+=M
    products.append({
        'A': {(0,0): [(1,1,1)], (0,1): [(1,1,1)], (1,0): [(1,1,1)], (1,1): [(1,1,1)]},
        'B': {(0,0): [(1,0,1), (0,0,-1)], (0,1): [(1,0,1), (0,0,-1)], (1,0): [(1,0,1), (0,0,-1)], (1,1): [(1,0,1), (0,0,-1)]},
        'C': {(0,0): 1, (1,0): 1}
    })
    # M5: A' = A11 + A12, B' = B22, C: (0,0)-=M, (0,1)+=M
    products.append({
        'A': {(0,0): [(0,0,1), (0,1,1)], (0,1): [(0,0,1), (0,1,1)], (1,0): [(0,0,1), (0,1,1)], (1,1): [(0,0,1), (0,1,1)]},
        'B': {(0,0): [(1,1,1)], (0,1): [(1,1,1)], (1,0): [(1,1,1)], (1,1): [(1,1,1)]},
        'C': {(0,0): -1, (0,1): 1}
    })
    # M6: A' = A21 - A11, B' = B11 + B12, C: (1,1)+=M
    products.append({
        'A': {(0,0): [(1,0,1), (0,0,-1)], (0,1): [(1,0,1), (0,0,-1)], (1,0): [(1,0,1), (0,0,-1)], (1,1): [(1,0,1), (0,0,-1)]},
        'B': {(0,0): [(0,0,1), (0,1,1)], (0,1): [(0,0,1), (0,1,1)], (1,0): [(0,0,1), (0,1,1)], (1,1): [(0,0,1), (0,1,1)]},
        'C': {(1,1): 1}
    })
    # M7: A' = A12 - A22, B' = B21 + B22, C: (0,0)+=M, (1,1)-=M
    products.append({
        'A': {(0,0): [(0,1,1), (1,1,-1)], (0,1): [(0,1,1), (1,1,-1)], (1,0): [(0,1,1), (1,1,-1)], (1,1): [(0,1,1), (1,1,-1)]},
        'B': {(0,0): [(1,0,1), (1,1,1)], (0,1): [(1,0,1), (1,1,1)], (1,0): [(1,0,1), (1,1,1)], (1,1): [(1,0,1), (1,1,1)]},
        'C': {(0,0): 1, (1,1): -1}
    })
    
    U2a = np.array(U2, dtype=np.float64)
    V2a = np.array(V2, dtype=np.float64)
    W2a = np.array(W2, dtype=np.float64)
    
    total_rank = 7 * inner_rank
    U4 = [[0] * total_rank for _ in range(16)]
    V4 = [[0] * total_rank for _ in range(16)]
    W4 = [[0] * total_rank for _ in range(16)]
    
    for pi, prod in enumerate(products):
        for r in range(inner_rank):
            col = pi * inner_rank + r
            # For each inner entry of A' (a,b):
            for a in range(2):
                for b in range(2):
                    u2_val = U2a[a*2+b, r]
                    if u2_val == 0:
                        continue
                    for (br, bc, coeff) in prod['A'][(a,b)]:
                        global_r = 2*br + a
                        global_c = 2*bc + b
                        U4[global_r*4 + global_c][col] += u2_val * coeff
            
            for j in range(2):
                for k in range(2):
                    v2_val = V2a[j*2+k, r]
                    if v2_val == 0:
                        continue
                    for (br, bc, coeff) in prod['B'][(j,k)]:
                        global_r = 2*br + j
                        global_c = 2*bc + k
                        V4[global_r*4 + global_c][col] += v2_val * coeff
            
            for k in range(2):
                for i in range(2):
                    w2_val = W2a[k*2+i, r]
                    if w2_val == 0:
                        continue
                    for (bi, bj) in prod['C']:
                        coeff = prod['C'][(bi,bj)]
                        global_r = 2*bi + k
                        global_c = 2*bj + i
                        W4[global_r*4 + global_c][col] += w2_val * coeff
    
    # Verify 4x4x4
    if time.time() > _DEADLINE:
        return None
    
    T4 = np.zeros((16, 16, 16))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                T4[i*4+j, j*4+k, k*4+i] = 1.0
    U4a = np.array(U4, dtype=np.float64)
    V4a = np.array(V4, dtype=np.float64)
    W4a = np.array(W4, dtype=np.float64)
    res4 = np.einsum('ar,br,cr->abc', U4a, V4a, W4a)
    if np.max(np.abs(res4 - T4)) > 1e-9:
        # Fallback to schoolbook for 4x4x4
        U4 = [[0]*64 for _ in range(16)]
        V4 = [[0]*64 for _ in range(16)]
        W4 = [[0]*64 for _ in range(16)]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    c = i*16 + j*4 + k
                    U4[i*4+j][c] = 1
                    V4[j*4+k][c] = 1
                    W4[k*4+i][c] = 1
        core_rank = 64
    else:
        core_rank = total_rank
    
    if time.time() > _DEADLINE:
        return None
    
    # Now build the full 4x5x5 decomposition
    # A (4x5) = [A1 | A2] where A1 is 4x4, A2 is 4x1
    # B (5x5) = [[B1], [B2]] where B1 is 4x5, B2 is 1x5
    # C = A1*B1 + A2*B2
    # B1 = [B1a | B1b] where B1a is 4x4, B1b is 4x1
    # A1*B1a: 4x4*4x4, rank core_rank
    # A1*B1b: 4x4*4x1, rank 16
    # A2*B2: 4x1*1x5, rank 20
    
    total = core_rank + 16 + 20
    U = [[0] * total for _ in range(20)]  # 4*5
    V = [[0] * total for _ in range(25)]  # 5*5
    W = [[0] * total for _ in range(20)]  # 5*4
    
    col = 0
    # Part 1: A1*B1a (4x4 * 4x4)
    for i in range(4):
        for j in range(4):
            for c in range(core_rank):
                U[i*5+j][col+c] = U4[i*4+j][c]
                V[j*5+c][col+c] = V4[j*4+c][c]  # B1a[j, k] where k is col in B
                # Wait, B1a is 4x4, entry (j, k) corresponds to B[j, k] in the 5x5 B.
                # V row = j*p+k = j*5+k
                for k in range(4):
                    V[j*5+k][col+c] = V4[j*4+k][c]
                # W: C[i, k] where k is col in C (4 cols from B1a)
                for k in range(4):
                    W[k*4+i][col+c] = W4[k*4+i][c]
    col += core_rank
    
    # Part 2: A1*B1b (4x4 * 4x1)
    # A1 is 4x4, B1b is 4x1, result is 4x1 (C cols 4)
    for i in range(4):
        for j in range(4):
            c = col
            U[i*5+j][c] = 1
            V[j*5+4][c] = 1  # B1b[j, 0] = B[j, 4]
            W[4*4+i][c] = 1  # C[i, 4]
            col += 1
    
    # Part 3: A2*B2 (4x1 * 1x5)
    # A2 is 4x1 (A[i,4]), B2 is 1x5 (B[4, k]), result is 4x5
    for i in range(4):
        for k in range(5):
            c = col
            U[i*5+4][c] = 1  # A2[i, 0] = A[i, 4]
            V[4*5+k][c] = 1  # B2[0, k] = B[4, k]
            W[k*4+i][c] = 1  # C[i, k]
            col += 1
    
    if col != total:
        raise ValueError(f"Column mismatch: {col} != {total}")
    
    # Verify
    if _verify(U, V, W, 4, 5, 5):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback: full schoolbook
        rank = 4 * 5 * 5
        U = [[0] * rank for _ in range(20)]
        V = [[0] * rank for _ in range(25)]
        W = [[0] * rank for _ in range(20)]
        c = 0
        for i in range(4):
            for j in range(5):
                for k in range(5):
                    U[i*5+j][c] = 1
                    V[j*5+k][c] = 1
                    W[k*4+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}


def construct():
    return construct()
