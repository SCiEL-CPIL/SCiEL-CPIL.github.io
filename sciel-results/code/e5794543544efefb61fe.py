"""Matrix multiplication 3x4x6 with block tensor decomposition."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strategy: Split the m=4 dimension into m1=3 and m2=1.
    # A is n x m = 3 x 4. Split columns into A1 (3x3) and A2 (3x1).
    # B is m x p = 4 x 6. Split rows into B1 (3x6) and B2 (1x6).
    # C = A1 * B1 + A2 * B2
    #
    # For A1 * B1: A1 is 3x3, B1 is 3x6.
    # We can further split B1 into B1a (3x3) and B1b (3x3) by columns.
    # Then A1 * B1 = A1 * B1a + A1 * B1b.
    # Each is a 3x3x3 matrix multiplication.
    # Schoolbook rank for 3x3x3 is 27.
    #
    # For A2 * B2: A2 is 3x1, B2 is 1x6.
    # This is an outer product-like structure.
    # C2[i,k] = A2[i] * B2[k]
    # This is a rank-3 tensor (sum over r=0..2 of U[i,r] * V[r,k] * W[i,k,r])
    # Actually, let's think about this more carefully.
    # The tensor for (n,1,p) matrix multiplication has rank n.
    # For n=3, p=6, the rank is 3.
    # U[i,r] = delta(i,r) for i in 0..2, r in 0..2
    # V[r,k] = 1 for all r,k? No.
    # Let's derive:
    # C[i,k] = A[i,0] * B[0,k]
    # We want sum_r U[i,r] * V[r,k] * W[i,k,r] = A[i,0] * B[0,k]
    # If we set r to range over 0..2:
    # U[i,r] = delta(i,r)
    # V[r,k] = B[0,k] ... no, V depends on B values which vary.
    # Wait, the decomposition must be fixed coefficients.
    # C[i,k] = sum_r U[i,r] * (A[i,0]) * V[r,k] * (B[0,k]) * W[i,k,r]
    # This doesn't work directly because A[i,0] and B[0,k] are separate factors.
    #
    # Let's re-examine the tensor structure.
    # The matrix multiplication tensor T has entries T[i,j,k] such that
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The rank is the minimum number of rank-1 tensors to sum to T.
    # A rank-1 term is u_i * v_j * w_{ik}.
    # So C[i,k] += u_i * v_j * w_{ik} * A[i,j] * B[j,k]
    #
    # For the (3,1,6) case:
    # C[i,k] = A[i,0] * B[0,k]
    # We need sum_r u_{i,r} * v_{0,r} * w_{i,k,r} = A[i,0] * B[0,k]
    # This is not a fixed tensor decomposition in the usual sense because
    # the "tensor" here is the identity map on the space of (3x1) and (1x6) matrices.
    # Actually, the multiplication map is:
    # (A, B) -> A * B
    # For (n,1,p), this map is bilinear.
    # The rank of this bilinear map is n.
    # We can write:
    # A[i,0] * B[0,k] = sum_{r=0}^{n-1} delta(i,r) * A[i,0] * 1 * B[0,k] * delta(r, r)
    # Wait, that's not right either.
    #
    # Let's use the standard basis.
    # Let e_i be the standard basis for R^n, f_k for R^p.
    # A = sum_i a_i e_i (as a column vector)
    # B = sum_k b_k f_k^T (as a row vector)
    # C = A B = sum_{i,k} a_i b_k e_i f_k^T
    # This is a sum of n*p rank-1 matrices in the output space, but we want
    # the rank of the bilinear map.
    #
    # The bilinear map is (a, b) -> a b^T (outer product).
    # The rank of the outer product map R^n x R^p -> R^{n x p} is n*p?
    # No, the rank of a bilinear map is the minimum number of rank-1 bilinear forms.
    # A rank-1 bilinear form is (a, b) -> (u^T a) (v^T b).
    # The outer product map is (a, b) -> a b^T.
    # This can be written as sum_{i=0}^{n-1} (e_i^T a) * (a? no)
    #
    # Actually, for matrix multiplication, the tensor is:
    # T = sum_{i,j,k} E_{i,k}^{(j)} where E_{i,k}^{(j)} is the tensor that picks
    # out A[i,j] and B[j,k] and puts it in C[i,k].
    #
    # For (n,1,p):
    # T = sum_{i,k} E_{i,k}^{(0)}
    # Each E_{i,k}^{(0)} is a rank-1 tensor: u_i * v_0 * w_{i,k}
    # where u_i = delta(i,i), v_0 = 1, w_{i,k} = delta(i,i)delta(k,k)
    # But we can combine terms.
    # C[i,k] = A[i,0] B[0,k]
    # = sum_{r=0}^{n-1} delta(i,r) * A[i,0] * B[0,k] * 1
    # This suggests rank n.
    # Let's verify:
    # Term r: U[i,r] = delta(i,r), V[r,k] = 1 (for the B part, but B only has index 0)
    # Wait, V is indexed by (j,k) where j is the middle dimension.
    # For (n,1,p), j=0 only.
    # V[0,k] is the coefficient for B[0,k].
    # So V[r, 0, k] ... no, V is a matrix of size (m*p) x rank.
    # Here m=1, p=6, so V is 6 x rank.
    # V[j*p+k, r] = V[k, r] since j=0.
    #
    # Let's try rank n=3:
    # For r in 0..2:
    # U[i, r] = delta(i, r)  (U is n*m x rank = 3x3 x 3)
    # V[k, r] = 1           (V is m*p x rank = 1*6 x 3 = 6x3)
    # W[i, k, r] = delta(i, r)  (W is p*n x rank = 6*3 x 3 = 18x3)
    #
    # Sum_r U[i,r] * A[i,0] * V[0,k] * B[0,k] * W[i,k,r]
    # = Sum_r delta(i,r) * A[i,0] * 1 * B[0,k] * delta(i,r)
    # = A[i,0] * B[0,k] * delta(i,i)
    # = A[i,0] * B[0,k]
    # This works! Rank 3 for the (3,1,6) part.
    
    # Total rank = 27 + 27 + 3 = 57
    
    rank = 57
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: A1 (3x3) * B1a (3x3) -> C1a (3x3)
    # A1 columns 0-2, B1a columns 0-2 of B1 (which are columns 0-2 of B)
    # i in 0..2, j in 0..2, k in 0..2
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # U index: i*m + j = i*4 + j
                U[i * m + j][col] = 1
                # V index: j*p + k = j*6 + k
                V[j * p + k][col] = 1
                # W index: k*n + i = k*3 + i
                W[k * n + i][col] = 1
                col += 1
    
    # Block 2: A1 (3x3) * B1b (3x3) -> C1b (3x3)
    # A1 columns 0-2, B1b columns 3-5 of B1 (which are columns 3-5 of B)
    # i in 0..2, j in 0..2, k in 3..5
    for i in range(3):
        for j in range(3):
            for k in range(3, 6):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Block 3: A2 (3x1) * B2 (1x6) -> C2 (3x6)
    # A2 is column 3 of A, B2 is row 3 of B
    # Rank 3 decomposition
    for r in range(3):
        # U[i, r] = delta(i, r) for i in 0..2
        # U index: i*m + 3 = i*4 + 3
        for i in range(3):
            if i == r:
                U[i * m + 3][col] = 1
        
        # V[k, r] = 1 for k in 0..5
        # V index: 3*p + k = 18 + k
        for k in range(6):
            V[3 * p + k][col] = 1
        
        # W[i, k, r] = delta(i, r)
        # W index: k*n + i = k*3 + i
        for i in range(3):
            for k in range(6):
                if i == r:
                    W[k * n + i][col] = 1
        
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
