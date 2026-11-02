<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    rank = n * m * p
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
=======
# EVOLVE-BLOCK-START
import time

def _add_term(U, V, W, col, u_idx, v_idx, w_idx, coeff_u=1, coeff_v=1, coeff_w=1):
    """Add one rank-1 term with given coefficients."""
    U[u_idx][col] += coeff_u
    V[v_idx][col] += coeff_v
    W[w_idx][col] += coeff_w

def _strassen_2x2x2(U, V, W, col, i0, j0, k0, n, m, p):
    """
    Apply Strassen 2x2x2 decomposition for a 2x2 block of A (rows i0..i0+1, cols j0..j0+1),
    2x2 block of B (rows j0..j0+1, cols k0..k0+1), contributing to 2x2 block of C (rows i0..i0+1, cols k0..k0+1).
    
    U index: i*m + j
    V index: j*p + k
    W index: k*n + i
    """
    # Strassen's 7 products for 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each "product" in Strassen corresponds to summing over the 2x2 sub-blocks.
    # For term M1: (A00+A11)*(B00+B11)
    #   U contributes: A00 at (i0,j0) and A11 at (i0+1,j0+1) -> both +1
    #   V contributes: B00 at (j0,k0) and B11 at (j0+1,k0+1) -> both +1
    #   W: C00 gets +1, C11 gets +1
    #   But wait - in tensor decomposition, each rank-1 term is a single product.
    #   Strassen's M1 = (A00+A11)*(B00+B11) is NOT a single rank-1 term in the original
    #   tensor; it's a sum of 4 rank-1 terms. The trick is that we can combine them.
    
    # Actually, let me think about this differently. The standard Strassen algorithm
    # computes 7 multiplications of 2x2 matrices. Each 2x2 multiplication can be
    # done with 4 scalar multiplications (schoolbook). So 7*4 = 28 scalar mults
    # instead of 8*4 = 32. But that's for the whole 2x2.
    
    # For the tensor decomposition approach: we need to express the 2x2x2 tensor
    # (rank 8 schoolbook) as 7 rank-1 terms... but that's not right either.
    # The 2x2x2 matrix multiplication tensor has rank 7 (Strassen), not 8.
    
    # Let me use the explicit Strassen rank-7 decomposition of the 2x2x2 tensor.
    # The 2x2x2 tensor T[i,j,k] = 1 if i,j,k in {0,1} with the matmul structure.
    # Actually T[i,j,k] = delta_{structure} - the matmul tensor has T[i,j,k] = 1
    # for all i,j,k since C[i,k] = sum_j A[i,j]*B[j,k].
    
    # The rank-7 decomposition of the 2x2x2 matmul tensor:
    # Each term is (u, v, w) where u, v, w are vectors in {0,1}^2.
    # u_i * v_j * w_k should reconstruct the tensor.
    
    # Standard Strassen decomposition (one common form):
    # Term 1: u=[1,1], v=[1,1], w=[1,-1]  -> (A00+A11)(B00+B11) contributes to C00-C11... 
    # Let me use a verified decomposition.
    
    # Verified rank-7 decomposition of 2x2x2 matmul tensor T[i,j,k]=1:
    # We need: sum_r u_r[i] * v_r[j] * w_r[k] = 1 for all i,j,k in {0,1}
    # where the tensor is T[i,j,k] = 1 (since every entry of A times every entry of B
    # with matching middle index contributes).
    
    # Wait, the matmul tensor for 2x2 is: T[i,j,k] = 1 if the indices are valid,
    # meaning C[i,k] = sum_j A[i,j] B[j,k]. The tensor entries are T[i,j,k] = 1
    # for all i,j,k in {0,1}. So we need a rank decomposition of the all-ones 2x2x2 tensor.
    # But that tensor has rank 1! That can't be right.
    
    # I'm confusing myself. The matrix multiplication tensor T_{ijk} is defined such that
    # C_{ik} = sum_j T_{ijk} A_{ij} B_{jk}. For 2x2, T_{ijk} = 1 for all i,j,k in {0,1}.
    # The rank of this tensor is 7 (Strassen's result).
    
    # A rank-7 decomposition: T_{ijk} = sum_{r=1}^{7} u_r[i] v_r[j] w_r[k]
    # where u_r, v_r, w_r are in R^2.
    
    # One standard decomposition (from Strassen 1969):
    # r=1: u=(1,1), v=(1,1), w=(1,1)
    # r=2: u=(1,0), v=(0,1), w=(1,0)
    # r=3: u=(1,0), v=(1,0), w=(0,1)
    # r=4: u=(0,1), v=(0,1), w=(0,1)
    # r=5: u=(0,1), v=(1,1), w=(1,0)
    # r=6: u=(1,1), v=(0,1), w=(0,1)
    # r=7: u=(1,0), v=(1,1), w=(1,1)
    # Let me verify: T[0,0,0] = 1*1*1 + 1*0*1 + 1*1*0 + 0*0*0 + 0*1*1 + 1*0*0 + 1*1*1 = 1+0+0+0+0+0+1 = 2 ≠ 1
    # That's wrong.
    
    # Let me look up the correct Strassen decomposition more carefully.
    # The correct rank-7 decomposition of the 2x2x2 matmul tensor:
    # Using the formulation where we need sum_r u_r[i] * v_r[j] * w_r[k] = 1 for all i,j,k.
    
    # After careful derivation, one valid decomposition is:
    # r1: u=(1,0), v=(1,0), w=(1,0)
    # r2: u=(1,0), v=(0,1), w=(0,1)
    # r3: u=(0,1), v=(1,0), w=(0,1)
    # r4: u=(0,1), v=(0,1), w=(1,0)
    # r5: u=(1,1), v=(1,1), w=(1,1)
    # r6: u=(1,-1), v=(1,1), w=(1,0)
    # r7: u=(1,1), v=(1,-1), w=(0,1)
    # Check T[0,0,0]: 1*1*1 + 1*0*0 + 0*1*0 + 0*0*1 + 1*1*1 + 1*1*1 + 1*1*0 = 1+0+0+0+1+1+0 = 3 ≠ 1
    
    # I'm making errors. Let me use a different approach: just use the known fact that
    # Strassen reduces 2x2x2 from rank 8 to rank 7, and implement it by partitioning
    # the problem into 2x2 blocks and using 7 terms per block instead of 8.
    
    # Actually, the cleanest approach: for each 2x2x2 sub-problem, use 7 rank-1 terms
    # instead of 8. The 7 terms for Strassen are well-known. Let me use this verified set:
    
    # From literature, the Strassen 7-term decomposition of the 2x2x2 tensor:
    # (u_r, v_r, w_r) for r=1..7, where each is a pair (value at index 0, value at index 1):
    terms = [
        # (u0, u1), (v0, v1), (w0, w1)
        ((1, 1), (1, 1), (1, 0)),   # r1
        ((1, 0), (0, 1), (1, 0)),   # r2
        ((0, 1), (0, 1), (1, 0)),   # r3
        ((1, 0), (1, 0), (0, 1)),   # r4
        ((0, 1), (1, 1), (0, 1)),   # r5
        ((1, 1), (1, 1), (0, 1)),   # r6
        ((0, 1), (1, 0), (1, 1)),   # r7
    ]
    
    # Verify T[0,0,0]:
    # r1: 1*1*1=1, r2: 1*0*1=0, r3: 0*0*1=0, r4: 1*1*0=0, r5: 0*1*0=0, r6: 1*1*0=0, r7: 0*1*1=0 -> sum=1 ✓
    # T[0,0,1]:
    # r1: 1*1*0=0, r2: 1*0*0=0, r3: 0*0*0=0, r4: 1*1*1=1, r5: 0*1*1=0, r6: 1*1*1=1, r7: 0*1*0=0 -> sum=2 ✗
    
    # Still wrong. Let me just use a brute-force verified decomposition.
    # I'll use the following known-correct Strassen decomposition (from standard references):
    
    # The 7 terms (u, v, w) where each component is (val_at_0, val_at_1):
    strassen_terms = [
        ((1, 1), (1, 1), (1, -1)),
        ((1, 0), (0, 1), (1, 0)),
        ((1, 0), (1, 0), (0, 1)),
        ((0, 1), (0, 1), (1, 0)),
        ((0, 1), (1, 1), (0, 1)),
        ((1, 1), (1, 1), (0, 1)),
        ((1, 0), (1, 1), (1, 1)),
    ]
    
    # T[0,0,0]: 1*1*1 + 1*0*1 + 1*1*0 + 0*0*1 + 0*1*0 + 1*1*0 + 1*1*1 = 1+0+0+0+0+0+1 = 2 ✗
    
    # OK let me just carefully derive it. We need 7 triples (u,v,w) in R^2 x R^2 x R^2
    # such that for all i,j,k in {0,1}: sum_{r=1}^7 u_r[i]*v_r[j]*w_r[k] = 1.
    
    # This is equivalent to: the 2x2x2 all-ones tensor has a rank-7 decomposition.
    # But the all-ones 2x2x2 tensor has rank 1 (it's 1*1*1)!
    
    # I see my confusion now. The matrix multiplication tensor is NOT the all-ones tensor.
    # Let me re-examine: C[i,k] = sum_j A[i,j] * B[j,k].
    # So T[i,j,k] = 1 for all valid i,j,k. For 2x2, that's T[i,j,k]=1 for all i,j,k in {0,1}.
    # That IS the all-ones tensor, which has rank 1.
    
    # But Strassen says 2x2 matmul can be done with 7 multiplications, implying rank 7.
    # The resolution: the "rank" in tensor decomposition for matmul is different.
    # In the standard tensor rank formulation for matmul, we have:
    # C[i,k] = sum_j sum_r T[i,j,k] * (A[i,j] * B[j,k])
    # where T[i,j,k] = 1 for all i,j,k. The tensor T is indeed all-ones.
    # But the "multiplication rank" counts the number of scalar multiplications,
    # which is the rank of T in a different sense... 
    
    # Actually no. The standard formulation: we want to compute C = A*B where
    # C[i,k] = sum_j A[i,j]B[j,k]. We can write this as:
    # C[i,k] = sum_r (sum_j u_r[i,j] A[i,j]) * (sum_k' v_r[j,k'] B[j,k']) * w_r[i,k]
    # No, that's not right either.
    
    # The correct tensor formulation: define T in R^{(n x m) x (m x p) x (p x n)} by
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if j=j' and i=i' and k=k', else 0.
    # Wait no. Let me be very precise.
    
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Let a_{ij} = A[i,j], b_{jk} = B[j,k], c_{ik} = C[i,k].
    # c_{ik} = sum_j a_{ij} b_{jk}
    # The tensor is T_{(i,j),(j,k),(k,i)} = 1, i.e., T is indexed by (i,j) for A,
    # (j,k) for B, (k,i) for C, and T[(i,j),(j,k),(k,i)] = 1.
    # For 2x2: A has 4 entries, B has 4, C has 4. T is a 4x4x4 tensor.
    # T[(i,j),(j',k),(k',i')] = 1 if j=j', i=i', k=k', else 0.
    # So T has 8 ones (one for each i,j,k combination), and it's the "matmul tensor".
    # Its rank is 7 (Strassen).
    
    # So the 4x4x4 tensor has 8 non-zero entries and rank 7.
    # A rank-1 decomposition term is (u, v, w) where u in R^4, v in R^4, w in R^4,
    # and T = sum_r u_r ⊗ v_r ⊗ w_r.
    
    # For the 2x2x2 case, the 8 non-zero entries are at positions:
    # (i*m+j, j*p+k, k*n+i) for i,j,k in {0,1}
    # = (0,0,0), (0,2,1), (1,0,2), (1,2,3), (2,1,0), (2,3,1), (3,1,2), (3,3,3)
    # Wait: i*m+j where m=2: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # j*p+k where p=2: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # k*n+i where n=2: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # So for (i,j,k): U_idx = i*2+j, V_idx = j*2+k, W_idx = k*2+i
    # (0,0,0): U=0, V=0, W=0
    # (0,0,1): U=0, V=1, W=1
    # (0,1,0): U=1, V=2, W=2
    # (0,1,1): U=1, V=3, W=3
    # (1,0,0): U=2, V=0, W=0
    # (1,0,1): U=2, V=1, W=1
    # (1,1,0): U=3, V=2, W=2
    # (1,1,1): U=3, V=3, W=3
    
    # We need 7 vectors u_r, v_r, w_r in R^4 such that:
    # sum_r u_r[a] * v_r[b] * w_r[c] = 1 if (a,b,c) is one of the 8 positions above, else 0.
    
    # This is a known result. Let me use a verified decomposition.
    # After research, one valid rank-7 decomposition is:
    
    # I'll use the following approach: partition into 2x2 blocks and for each 2x2x2
    # sub-block, use 7 terms. For the 3x5x5 case, I'll partition as:
    # - 2x2 blocks where possible, 1x1 for the remainder.
    
    # For now, let me just return the schoolbook to ensure correctness, and add
    # a time guard.
    n, m, p = DIMENSIONS
    rank = n * m * p
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
>>>>>>> REPLACE

Let me reconsider. The Strassen decomposition is tricky to get right by hand. Let me use a verified approach by actually computing the rank-7 decomposition numerically and verifying it.

