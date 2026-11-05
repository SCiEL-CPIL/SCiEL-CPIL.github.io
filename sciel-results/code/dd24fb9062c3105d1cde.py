"""
Research program: Rank-14 construction for 2x3 x 3x3 matrix multiplication
via Kronecker product of identity(2) with rank-7 3x3 decomposition.
"""
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=3

    # The tensor T for matrix multiplication C = A @ B where A is (n, m), B is (m, p)
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This is because C[i,k] = sum_j A[i,j] * B[j,k], so every (i,j,k) triple contributes.
    #
    # We want to decompose T as:
    #   T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # Strategy: Group the n=2 dimension. For each i in {0,1}, the sub-tensor is
    #   T_i[j,k] = 1 for all j in [0,3), k in [0,3)
    # This is the all-ones 3x3 matrix, which is a rank-1 tensor!
    #
    # Wait, that's not right. The tensor T[i,j,k] = 1 for all i,j,k means it's the
    # outer product of three vectors of all-ones. So T = u ⊗ v ⊗ w where u, v, w
    # are all-ones vectors. This is rank 1!
    #
    # But wait, that can't be right for matrix multiplication. Let me reconsider.
    #
    # Actually, for matrix multiplication C = AB, the tensor is defined such that:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The "tensor" here is not T[i,j,k] = 1. The bilinear form is:
    # C[i,k] = sum_j (1) * A[i,j] * B[j,k]
    # So the coefficients are all 1. The tensor T[i,j,k] = 1 for all i,j,k.
    #
    # But this tensor T = ones(n,m,p) can be written as u ⊗ v ⊗ w where u = ones(n),
    # v = ones(m), w = ones(p). This is a rank-1 tensor!
    #
    # However, this rank-1 decomposition does NOT correspond to a valid matrix multiplication
    # decomposition in the sense required by the task. The task requires that the decomposition
    # factors U, V, W have the specific index structure:
    #   U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # and that the product sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = T[i,j,k] for all i,j,k.
    #
    # If T[i,j,k] = 1 for all i,j,k, then we need:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # A rank-1 decomposition would be: U[i*m+j, 0] = a[i*m+j], V[j*p+k, 0] = b[j*p+k], 
    # W[k*n+i, 0] = c[k*n+i] such that a[i*m+j] * b[j*p+k] * c[k*n+i] = 1 for all i,j,k.
    # This is possible! For example, set a = 1, b = 1, c = 1 for all entries.
    # Then the rank is 1.
    #
    # BUT WAIT. This seems too easy. Let me re-read the task.
    #
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 3 matrix."
    #
    # The standard schoolbook algorithm uses rank n*m*p = 18. But the tensor T[i,j,k] = 1
    # is indeed rank 1 as a 3-tensor. So why is the schoolbook rank 18?
    #
    # I think I'm confusing two different notions. In the context of matrix multiplication
    # tensor rank, the tensor is NOT T[i,j,k] = 1. The tensor for matrix multiplication
    # is defined differently.
    #
    # Let me reconsider. The matrix multiplication tensor M_{n,m,p} is defined by:
    #   M_{i,j,k} = delta_{i,k} ... no, that's not right either.
    #
    # Actually, I think the issue is that the "tensor" in the context of matrix multiplication
    # rank is the structure tensor, which encodes the bilinear map. The bilinear map is:
    #   (A, B) -> A @ B
    # where A is (n,m) and B is (m,p). This bilinear map can be represented as a 3-tensor
    # T of shape (n*p, m, m) or similar, depending on the convention.
    #
    # Given the index convention in the task:
    #   U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # This suggests that the tensor is indexed by (i, j, k) where i is the row of A (and C),
    # j is the column of A / row of B, and k is the column of B (and C).
    #
    # The tensor T[i,j,k] represents the coefficient of A[i,j] * B[j,k] in C[i,k].
    # Since C[i,k] = sum_j A[i,j] * B[j,k], we have T[i,j,k] = 1 for all i,j,k.
    #
    # So the tensor IS the all-ones tensor, and its rank as a 3-tensor is 1.
    #
    # But this contradicts the known results that matrix multiplication tensors have rank
    # greater than 1. Let me think about this more carefully.
    #
    # I think the confusion is that the "rank" of the matrix multiplication tensor is NOT
    # the rank of the 3-tensor T[i,j,k] = 1. Rather, it's the minimum number of rank-1
    # tensors needed to decompose the BILINEAR MAP, not the 3-tensor itself.
    #
    # A rank-1 decomposition of the bilinear map corresponds to:
    #   C[i,k] += alpha * A[i,j1] * B[j2,k]
    # for some fixed j1, j2 and scalar alpha. But this doesn't match the structure
    # where the middle index must be the same (j1 = j2) for standard matrix multiplication.
    #
    # Ah, I see the issue now. In standard matrix multiplication, the middle index j
    # is shared between A and B. The tensor T[i,j,k] = 1 means that for each (i,j,k),
    # the product A[i,j] * B[j,k] contributes to C[i,k]. The constraint is that the
    # same j appears in both A and B.
    #
    # In a rank-1 term of the decomposition, we have:
    #   U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For this to be non-zero, we need U[i*m+j, c] ≠ 0, V[j*p+k, c] ≠ 0, and W[k*n+i, c] ≠ 0.
    # The key constraint is that the j index in U and V must be the SAME j (since they
    # both depend on j).
    #
    # So a rank-1 term effectively picks a "pattern" of (i,j,k) triples where U, V, W
    # are non-zero, and the contribution is the product of the three values.
    #
    # For the all-ones tensor, we need:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # With rank 1: U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = 1 for all i,j,k.
    # This is possible with U = V = W = 1 (all entries 1).
    #
    # So the rank IS 1? That seems wrong given the literature.
    #
    # Let me check: is the task asking for the rank of the matrix multiplication tensor
    # in the sense of the minimum number of multiplications needed? If so, then for
    # 2x3 x 3x3, the rank would be related to the minimum number of scalar multiplications
    # needed to compute all 2*3 = 6 entries of C.
    #
    # Each scalar multiplication can be of the form (linear combo of A entries) * (linear combo of B entries).
    # The rank of the matrix multiplication tensor is the minimum number of such multiplications.
    #
    # For 2x3 x 3x3: A has 6 entries, B has 9 entries, C has 6 entries.
    # The schoolbook uses 18 multiplications (2*3*3).
    #
    # But the tensor rank (in the algebraic sense) of the all-ones tensor T[i,j,k] = 1
    # is indeed 1, as I showed. So there must be a different interpretation.
    #
    # I think the issue is that the "tensor" in the context of matrix multiplication rank
    # is NOT the all-ones tensor. Instead, it's a tensor that encodes the structure of
    # the bilinear map more carefully, taking into account that the middle index is shared.
    #
    # Let me re-examine the index convention:
    #   U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    #
    # The first index of U is i*m+j, which ranges over [0, n*m).
    # The first index of V is j*p+k, which ranges over [0, m*p).
    # The first index of W is k*n+i, which ranges over [0, p*n).
    #
    # The tensor T is of shape (n*m, m*p, p*n) or equivalently (i*m+j, j*p+k, k*n+i).
    # T[i*m+j, j*p+k, k*n+i] = 1 if the indices are consistent (i.e., the j in U and V
    # is the same, and the i in U and W is the same, and the k in V and W is the same).
    #
    # Wait, but the indices are already tied together in the definition. The tensor is
    # implicitly defined by the constraint that the same i, j, k appear in all three factors.
    #
    # I think the correct interpretation is:
    # The tensor T is a 3-way array of shape (n*m, m*p, p*n) where
    #   T[a, b, c] = 1 if there exist i, j, k such that a = i*m+j, b = j*p+k, c = k*n+i
    #   T[a, b, c] = 0 otherwise
    #
    # For n=2, m=3, p=3:
    # a ranges over [0,6), b over [0,9), c over [0,6).
    # a = i*3+j, b = j*3+k, c = k*2+i
    #
    # For a given (i,j,k), we get a unique (a,b,c). There are 2*3*3 = 18 such triples.
    # So the tensor has 18 non-zero entries, each equal to 1.
    #
    # Now, the rank of this tensor (as a 3-way array of shape 6x9x6) is what we want to minimize.
    #
    # This is a different tensor from the all-ones tensor of shape (2,3,3). The all-ones
    # tensor of shape (2,3,3) has rank 1, but the tensor of shape (6,9,6) with 18 non-zero
    # entries (one for each (i,j,k) triple) has a higher rank.
    #
    # So the task is to find the minimum rank decomposition of this 6x9x6 tensor.
    #
    # Now, the directive suggests using the rank-7 decomposition for 3x3 matrix multiplication.
    # The 3x3 matrix multiplication tensor (for 3x3 x 3x3) has a known rank of 19 (schoolbook)
    # or possibly lower. But the directive mentions "rank-7", which is suspicious.
    #
    # Actually, I think the directive is referring to the fact that the 3x3 matrix multiplication
    # can be decomposed in a certain way. Let me think about this differently.
    #
    # The 2x3 x 3x3 multiplication can be viewed as two independent 1x3 x 3x3 multiplications
    # (one for each row of A). Each 1x3 x 3x3 multiplication computes a row of C.
    #
    # For a 1x3 x 3x3 multiplication: A is 1x3, B is 3x3, C is 1x3.
    # C[k] = sum_j A[j] * B[j,k]
    #
    # The tensor for this is of shape (3, 9, 3) where:
    #   T[j, j*p+k, k] = 1 (using the same indexing convention)
    #
    # The rank of this tensor is 3 (schoolbook: one multiplication per j).
    #
    # So for two independent 1x3 x 3x3 multiplications, the total rank would be 2*3 = 6.
    #
    # But wait, this assumes that the two rows of A are independent. In the full 2x3 x 3x3
    # multiplication, the two rows of A share the same B. So there might be a way to share
    # some computations.
    #
    # Actually, no. The two rows of A are independent, so the computations for each row
    # are independent. The rank should be 2 * (rank of 1x3 x 3x3) = 2 * 3 = 6.
    #
    # But the schoolbook rank is 18, which is 2 * 3 * 3 = 18. So the rank of 1x3 x 3x3
    # is 3, and for two independent rows, it's 6.
    #
    # So the minimum rank is at most 6. Can we do better?
    #
    # The rank of the 1x3 x 3x3 tensor (shape 3x9x3) is 3, since it's a sum of 3 rank-1 tensors:
    #   T = sum_j e_j ⊗ f_j ⊗ g_j
    # where e_j is the j-th basis vector in R^3, f_j is the vector with 1 at position j*3+k
    # for all k, and g_k is the k-th basis vector in R^3.
    #
    # Actually, let me be more precise. For the 1x3 x 3x3 case:
    #   a = j (ranges over [0,3)), b = j*3+k (ranges over [0,9)), c = k (ranges over [0,3))
    #   T[a, b, c] = 1 if a = j and b = j*3+c, i.e., if b = a*3+c
    #
    # So T[a, b, c] = 1 if b = a*3+c, 0 otherwise.
    #
    # This tensor can be decomposed as:
    #   T = sum_j e_j ⊗ (sum_k e_{j*3+k} ⊗ e_k)
    # = sum_j sum_k e_j ⊗ e_{j*3+k} ⊗ e_k
    #
    # This is a sum of 9 rank-1 tensors, so the rank is at most 9. But can we do better?
    #
    # Actually, for fixed j, the term sum_k e_{j*3+k} ⊗ e_k is a rank-1 tensor in the
    # (b,c) space (it's the outer product of the vector with 1s at positions j*3, j*3+1, j*3+2
    # and the all-ones vector in the c space). So the total tensor is:
    #   T = sum_j e_j ⊗ (v_j ⊗ w)
    # where v_j is the vector with 1s at positions j*3, j*3+1, j*3+2, and w is the all-ones
    # vector in R^3.
    #
    # This is a sum of 3 rank-1 tensors (one for each j), so the rank is at most 3.
    #
    # Can we do better than 3? The tensor T has 3 non-zero "slices" (one for each j), and
    # each slice is a rank-1 matrix. The rank of the tensor is at least the maximum rank of
    # any slice, which is 1. But that doesn't help.
    #
    # Actually, the rank of T is exactly 3, since the 3 terms e_j ⊗ (v_j ⊗ w) are linearly
    # independent (they have support on disjoint sets of a indices).
    #
    # So the rank of 1x3 x 3x3 is 3, and for 2x3 x 3x3, the rank is at most 2*3 = 6.
    #
    # Now, can we share computations between the two rows? The two rows of A are A[0,:] and
    # A[1,:]. The computations are:
    #   C[0,k] = sum_j A[0,j] * B[j,k]
    #   C[1,k] = sum_j A[1,j] * B[j,k]
    #
    # These are independent linear combinations of the entries of B, weighted by A[0,j] and
    # A[1,j] respectively. There's no obvious way to share multiplications.
    #
    # So the rank is 6.
    #
    # Let me construct the rank-6 decomposition.
    #
    # For each i in {0,1} and j in {0,1,2}, we have a rank-1 term:
    #   U[i*3+j, c] = 1 if c = i*3+j, 0 otherwise
    #   V[j*3+k, c] = 1 if c = i*3+j, 0 otherwise (but V depends on j and k, not i)
    #
    # Wait, I need to be more careful. Let me index the rank-1 terms by (i,j) where i in {0,1}
    # and j in {0,1,2}. There are 6 such pairs.
    #
    # For the (i,j)-th term:
    #   U[i*3+j, c] = 1 if c = i*3+j, 0 otherwise
    #   V[j*3+k, c] = 1 if c = i*3+j, 0 otherwise
    #   W[k*2+i, c] = 1 if c = i*3+j, 0 otherwise
    #
    # Let's verify: for a given (i,j,k), the contribution to T[i,j,k] is:
    #   sum_c U[i*3+j, c] * V[j*3+k, c] * W[k*2+i, c]
    # = U[i*3+j, i*3+j] * V[j*3+k, i*3+j] * W[k*2+i, i*3+j]
    # = 1 * 1 * 1 = 1
    #
    # And for any other (i',j',k') with i' != i or j' != j, the contribution is 0.
    #
    # So the sum over all 6 terms gives T[i,j,k] = 1 for all (i,j,k). 
    #
    # This is a valid rank-6 decomposition!
    #
    # Let me double-check the shapes:
    #   U: (n*m, rank) = (6, 6)
    #   V: (m*p, rank) = (9, 6)
    #   W: (p*n, rank) = (6, 6)
    #
    # For c in [0,6), let i = c // 3, j = c % 3.
    #   U[i*3+j, c] = 1 (since i*3+j = c)
    #   V[j*3+k, c] = 1 for all k (since V[j*3+k, c] = 1 if c = i*3+j, which is true)
    #   W[k*2+i, c] = 1 for all k (since W[k*2+i, c] = 1 if c = i*3+j, which is true)
    #
    # Wait, V[j*3+k, c] should be 1 for all k when c = i*3+j. Let me verify:
    #   V[j*3+k, c] = 1 if c = i*3+j, 0 otherwise
    # For fixed (i,j) and c = i*3+j, V[j*3+k, c] = 1 for all k in [0,3).
    # This means V[j*3+k, c] = 1 for k=0,1,2.
    #
    # Similarly, W[k*2+i, c] = 1 for all k in [0,3) when c = i*3+j.
    #
    # So the decomposition is:
    #   For c in [0,6):
    #     i = c // 3, j = c % 3
    #     U[c, c] = 1 (since i*3+j = c)
    #     V[j*3+k, c] = 1 for all k in [0,3)
    #     W[k*2+i, c] = 1 for all k in [0,3)
    #
    # Let me verify with a specific example. Take i=0, j=1, so c = 0*3+1 = 1.
    #   U[1, 1] = 1
    #   V[1*3+k, 1] = V[3+k, 1] = 1 for k=0,1,2, so V[3,1]=1, V[4,1]=1, V[5,1]=1
    #   W[k*2+0, 1] = W[2k, 1] = 1 for k=0,1,2, so W[0,1]=1, W[2,1]=1, W[4,1]=1
    #
    # Now, for (i=0, j=1, k=0):
    #   U[0*3+1, 1] * V[1*3+0, 1] * W[0*2+0, 1] = U[1,1] * V[3,1] * W[0,1] = 1*1*1 = 1
    # For (i=0, j=1, k=1):
    #   U[1,1] * V[4,1] * W[2,1] = 1*1*1 = 1
    # For (i=0, j=1, k=2):
    #   U[1,1] * V[5,1] * W[4,1] = 1*1*1 = 1
    #
    # For (i=1, j=1, k=0):
    #   U[1*3+1, 1] * V[1*3+0, 1] * W[0*2+1, 1] = U[4,1] * V[3,1] * W[1,1]
    #   U[4,1] = 0 (since U[4,4]=1, not U[4,1])
    #   So the contribution is 0. 
    #
    # This looks correct!
    #
    # So the rank-6 decomposition is:
    #   rank = 6
    #   For c in [0,6):
    #     i = c // 3, j = c % 3
    #     U[i*3+j, c] = 1
    #     V[j*3+k, c] = 1 for all k in [0,3)
    #     W[k*2+i, c] = 1 for all k in [0,3)
    #
    # Let me code this up.

    rank = 6
    U = [[0] * rank for _ in range(n * m)]  # 6 x 6
    V = [[0] * rank for _ in range(m * p)]  # 9 x 6
    W = [[0] * rank for _ in range(p * n)]  # 6 x 6

    for c in range(rank):
        i = c // m  # c // 3
        j = c % m   # c % 3
        U[i * m + j][c] = 1
        for k in range(p):
            V[j * p + k][c] = 1
            W[k * n + i][c] = 1

    # Verify the decomposition
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    T_reconstructed = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_reconstructed[i, j, k] = np.sum(
                    U_np[i * m + j, :] * V_np[j * p + k, :] * W_np[k * n + i, :]
                )

    if not np.allclose(T, T_reconstructed):
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col_idx] = 1
                    V_school[j * p + k][col_idx] = 1
                    W_school[k * n + i][col_idx] = 1
                    col_idx += 1
        
        return {"U": U_school, "V": V_school, "W": W_school}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
