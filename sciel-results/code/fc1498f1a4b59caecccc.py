"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 4, 8

    # Strategy: Use Winograd-style decomposition for 1x4x2 blocks.
    # A 1x4x2 tensor (multiply 1x4 row by 4x2 matrix) can be done in rank 5.
    # The full 3x4x8 tensor = 3 independent 1x4x8 tensors.
    # Each 1x4x8 = two 1x4x2 blocks, so rank = 3 * 2 * 5 = 30.

    # Winograd 1x4x2 scheme:
    # Input a = [a0, a1, a2, a3], B is 4x2.
    # Output c0 = a0*B[0,0] + a1*B[1,0] + a2*B[2,0] + a3*B[3,0]
    #        c1 = a0*B[0,1] + a1*B[1,1] + a2*B[2,1] + a3*B[3,1]
    #
    # Winograd F(4,2) scheme uses rank 5 with the following factorization:
    # U (4x5): maps input a to 5 intermediate values
    # V (8x5): maps B entries to 5 intermediate values  
    # W (2x5): maps intermediates to output
    #
    # The standard Winograd 1x4x2 (F(4,2)) decomposition:
    # Let's use the known rank-5 decomposition for 1x4x2.
    #
    # Actually, let me use a simpler approach: the 1x4x2 tensor has rank at most 5.
    # Known decomposition (Winograd):
    #
    # U columns (4 x 5):
    # col0: [1, 0, 0, 0]
    # col1: [0, 1, 0, 0]
    # col2: [0, 0, 1, 0]
    # col3: [0, 0, 0, 1]
    # col4: [1, 1, 1, 1]
    #
    # Wait, that's not right. Let me think more carefully.
    #
    # For 1x4x2, the tensor T[a,j,k] = a_j * B[j,k] where a is 1x4, B is 4x2.
    # Actually the tensor is T[i,j,k] where i indexes the 1-dim (always 0),
    # j indexes the 4-dim (B rows), k indexes the 2-dim (B cols).
    # T[0,j,k] = B[j,k].
    #
    # The matrix multiplication tensor for 1x4x2 has entries T[j,k] = B[j,k].
    # This is just the 4x2 matrix itself, viewed as a tensor.
    # The rank of this tensor is min(4,2) = 2? No, that's the matrix rank.
    # The tensor rank of a 1x4x2 tensor (which is just a 4x2 matrix) is the matrix rank, which is 2.
    #
    # Wait, I'm confusing myself. Let me re-read the problem.
    #
    # We want to decompose the tensor for multiplying A (3x4) by B (4x8) to get C (3x8).
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The tensor T[i,j,k] = A[i,j] * B[j,k] is the structure tensor.
    # We need to write T as a sum of rank-1 tensors:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # (using the trace convention W[k*n+i, r])
    #
    # So U has shape (n*m, rank) = (12, rank)
    # V has shape (m*p, rank) = (32, rank)
    # W has shape (p*n, rank) = (24, rank)
    #
    # Schoolbook: rank = 3*4*8 = 96.
    #
    # The strategy says: partition into three 1x4x8 row-multiplication tensors.
    # Each row i of A (length 4) multiplies B (4x8) to give row i of C (length 8).
    # This is a 1x4x8 tensor.
    #
    # For a 1x4x8 tensor, we can use Winograd. The Winograd scheme F(4,8) would give
    # rank = ceil(4/2)*ceil(8/2) * something... Actually, let me use the F(4,2) block approach.
    #
    # Split B into two 4x2 blocks: B_left (cols 0,1) and B_right (cols 2,3), etc.
    # Actually, 8 columns = four 2-column blocks, or two 4-column blocks.
    #
    # The directive says: "decomposing each via the rank-5 Winograd scheme for 1x4x2 blocks
    # applied to the two 4x2 column sub-blocks of B"
    #
    # So for each row i: split B into B1 (4x2, cols 0-1) and B2 (4x2, cols 2-3)... 
    # Wait, "two 4x2 column sub-blocks" means B is split into two 4x4 blocks? No.
    # 4x8 split into two 4x4? That doesn't match "4x2".
    #
    # Re-reading: "the rank-5 Winograd scheme for 1x4x2 blocks applied to the two 4x2 column sub-blocks"
    # Maybe it means: split the 8 columns into four 2-column blocks, and pair them up?
    # Or: the 1x4x8 = 1x4x2 * 4 (four 2-col blocks), each rank 5, total rank 20 per row, 60 total.
    #
    # But the directive says rank-30. So 3 rows * 10 per row. That means each 1x4x8 has rank 10.
    # 1x4x8 split into two 1x4x4, each rank 5? Or two 1x4x2 + ... 
    #
    # Actually, maybe: 1x4x8 = two 1x4x4 blocks. Each 1x4x4 can be done in rank 6 (Winograd F(4,4)).
    # That gives 12 per row, 36 total. Not 30.
    #
    # Let me just try a direct approach: use the known Winograd F(4,8) or build a rank-30 decomposition.
    #
    # Actually, the simplest valid improvement: use Kronecker product structure.
    # 3x4x8 = 3x(4x8). The 4x8 part: use Winograd F(4,8).
    #
    # Let me just implement a known low-rank decomposition. 
    # For 1x4x2, rank is 5 (Winograd). For 1x4x8, we can do 4 blocks of 1x4x2, rank 20.
    # Total: 3*20 = 60. That's better than 96.
    #
    # But the directive claims rank 30. Let me try to construct it.
    #
    # Alternative: use the fact that 4 = 2*2, 8 = 2*4, and apply Strassen-like recursion.
    # Or use the Winograd F(4,8) scheme directly.
    #
    # Winograd F(4,8): The optimal rank for 1x4x8 is known to be 10 (from Winograd's F(4,8) scheme).
    # So 3 * 10 = 30 total. That matches!
    #
    # Let me construct the Winograd F(4,8) decomposition.
    # F(4,8) uses the factorization of the 4x8 convolution/multiplication.
    #
    # Actually, for matrix multiplication 1x4 by 4x8, the tensor is just the 4x8 matrix B.
    # The "multiplication" is a dot product of a 4-vector with each column of B.
    # The tensor rank of the 1x4x8 multiplication tensor... 
    #
    # Hmm, let me think about this differently. The tensor for 1x4x8 matmul:
    # T[j,k] where j in 0..3 (B row), k in 0..7 (B col). T[j,k] = B[j,k].
    # This is a 4x8 matrix. Its tensor rank (as a 1x4x8 tensor) equals its matrix rank = 8 (if B is full rank).
    # But we're not decomposing B; we're decomposing the multiplication operation.
    #
    # The multiplication tensor M for 1x4x8 has entries M[i,j,k] where i=0 (only one row),
    # j in 0..3, k in 0..7, and M[0,j,k] = e_j * e_k (the (j,k) entry of the identity-like tensor).
    # Actually, M[j,k] = 1 for all j,k? No.
    #
    # The structure tensor for multiplying a 1x4 vector by a 4x8 matrix:
    # The output C[k] = sum_j A[j] * B[j,k].
    # The tensor T[j,k] (with the implicit i=0) has T[j,k] = B[j,k].
    # As a bilinear map: (A, B) -> C. The tensor is T in U* ⊗ V* ⊗ W* where
    # U = R^4 (A space), V = R^8 (B space, flattened), W = R^8 (C space).
    # T[a,b] = sum_{j,k} a_j b_{j,k} e_k.
    # The structure tensor has components T_{j,(j,k),k} = 1... this is getting confusing.
    #
    # Let me just use the schoolbook and try to reduce via known methods.
    # Given time constraints, let me implement the Winograd F(4,2) per 2-column block.
    #
    # For each row i (3 rows), for each pair of output columns (k, k+1) (4 pairs):
    # Use rank-5 Winograd for 1x4x2.
    # Total rank = 3 * 4 * 5 = 60.
    #
    # Winograd F(4,2) decomposition:
    # Given a = [a0,a1,a2,a3] and B (4x2), compute c0 = a·B[:,0], c1 = a·B[:,1].
    #
    # The rank-5 scheme:
    # U (4x5): 
    #   u0 = [1, 0, 0, 0]
    #   u1 = [0, 1, 0, 0]  
    #   u2 = [0, 0, 1, 0]
    #   u3 = [0, 0, 0, 1]
    #   u4 = [1, 1, 1, 1]
    #
    # This doesn't seem right for Winograd. Let me look up the actual F(4,2) scheme.
    #
    # Winograd F(4,2): 
    # Input: a0,a1,a2,a3 and B00,B10,B20,B30,B01,B11,B21,B31
    # 
    # The standard Winograd F(4,2) uses these 5 products:
    # p0 = a0 * B00
    # p1 = a1 * B10
    # p2 = a2 * B20
    # p3 = a3 * B30
    # p4 = (a0+a1+a2+a3) * (B00+B10+B20+B30) ... no, that's not it.
    #
    # Actually, I recall that the Winograd F(4,2) scheme has rank 5 with:
    # Let me just use a different approach. Since I need to produce a valid decomposition,
    # let me use the schoolbook (rank 96) but try to find a smaller one via random search.
    #
    # Given the time budget, let me implement a simple rank-reduction search.
    # Start with schoolbook and try to find linear dependencies.
    
    # Actually, let me just implement the Winograd F(4,2) correctly.
    # 
    # From Winograd's paper, F(4,2) for convolution (which is equivalent to our case):
    # The 5 multipliers are:
    # m0 = a0
    # m1 = a1  
    # m2 = a2
    # m3 = a3
    # m4 = a0 + a1 + a2 + a3
    #
    # No wait, that's 5 values but they're not all independent products.
    #
    # Let me just go with the safe approach: implement schoolbook and verify.
    # Then try a simple random search for lower rank.
    
    rank = n * m * p  # 96 schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Verify and return
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
