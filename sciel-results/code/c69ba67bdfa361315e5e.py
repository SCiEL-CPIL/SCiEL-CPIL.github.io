import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 140.0
    
    # Decompose 2x2 * 2x9 as 3 blocks of 2x2 * 2x3
    # For each 2x2 * 2x3 block, use a rank-10 decomposition
    # Total rank = 30
    
    # Rank-10 decomposition for 2x2 * 2x3:
    # A = [[a, b], [c, d]] (2x2)
    # B = [[x1, y1, z1], [x2, y2, z2]] (2x3)
    # C = [[a*x1 + b*x2, a*y1 + b*y2, a*z1 + b*z2],
    #      [c*x1 + d*x2, c*y1 + d*y2, c*z1 + d*z2]]
    #
    # Terms 0-3: schoolbook for column 0 (x1, x2)
    # Terms 4-9: Strassen-style for columns 1,2 (y1,y2,z1,z2)
    #
    # For columns 1 and 2, we need to compute:
    # C[0,1] = a*y1 + b*y2, C[1,1] = c*y1 + d*y2
    # C[0,2] = a*z1 + b*z2, C[1,2] = c*z1 + d*z2
    #
    # Use 6 terms with half-integer coefficients:
    # Let's define the 6 terms for columns 1,2:
    
    # The 10 terms for one 2x2*2x3 block (M_r is 2x2, N_r is 2x3, w_r is scalar):
    block_decomp = [
        # Term 0: a*x1
        (np.array([[1, 0], [0, 0]]), np.array([[1, 0, 0], [0, 0, 0]]), 1),
        # Term 1: b*x2
        (np.array([[0, 1], [0, 0]]), np.array([[0, 0, 0], [1, 0, 0]]), 1),
        # Term 2: c*x1
        (np.array([[0, 0], [1, 0]]), np.array([[1, 0, 0], [0, 0, 0]]), 1),
        # Term 3: d*x2
        (np.array([[0, 0], [0, 1]]), np.array([[0, 0, 0], [1, 0, 0]]), 1),
        # Terms 4-9 for columns 1,2 using Strassen-like approach
        # We need: C[0,1]=a*y1+b*y2, C[1,1]=c*y1+d*y2, C[0,2]=a*z1+b*z2, C[1,2]=c*z1+d*z2
        # 
        # Use these 6 terms:
        # Term 4: (a+b)*(y1+z1) contributes to C[0,1], C[0,2]
        (np.array([[1, 1], [0, 0]]), np.array([[0, 1, 1], [0, 0, 0]]), 1/2),
        # Term 5: (a+b)*(y2+z2) contributes to C[0,1], C[0,2]
        (np.array([[1, 1], [0, 0]]), np.array([[0, 0, 0], [0, 1, 1]]), 1/2),
        # Term 6: (a-b)*(y1-z1) 
        (np.array([[1, -1], [0, 0]]), np.array([[0, 1, -1], [0, 0, 0]]), 1/2),
        # Term 7: (a-b)*(y2-z2)
        (np.array([[1, -1], [0, 0]]), np.array([[0, 0, 0], [0, 1, -1]]), 1/2),
        # Term 8: (c+d)*(y1+z1)
        (np.array([[0, 0], [1, 1]]), np.array([[0, 1, 1], [0, 0, 0]]), 1/2),
        # Term 9: (c+d)*(y2+z2)
        (np.array([[0, 0], [1, 1]]), np.array([[0, 0, 0], [0, 1, 1]]), 1/2),
    ]
    
    # Wait, this doesn't work. Let me think more carefully.
    # We need 4 outputs: C[0,1], C[1,1], C[0,2], C[1,2]
    # Each is a bilinear form in a,b,c,d and y1,y2,z1,z2
    # 
    # C[0,1] = a*y1 + b*y2
    # C[1,1] = c*y1 + d*y2
    # C[0,2] = a*z1 + b*z2
    # C[1,2] = c*z1 + d*z2
    #
    # This is essentially 2 independent 2x2*1x1 multiplications for each column pair.
    # For column 1: [a,b] * [y1;y2] and [c,d] * [y1;y2]
    # For column 2: [a,b] * [z1;z2] and [c,d] * [z1;z2]
    #
    # A 2x2*1x1 multiplication has rank 2 (schoolbook) or we can use Strassen for 2x2*2x2.
    # But here we have two separate 2x1*1x1 multiplications sharing the same [a,b] and [c,d].
    #
    # Actually, let's think of it as:
    # Row 0 of C: [a,b] * B_row0 where B_row0 = [[y1,z1],[y2,z2]] (1x2)
    # Row 1 of C: [c,d] * B_row1 where B_row1 = [[y1,z1],[y2,z2]] (1x2)
    #
    # So we need [a,b]*B_row0 and [c,d]*B_row1, where B_row0 = B_row1 = [[y1,z1],[y2,z2]]
    #
    # This is 2 independent 1x2 * 2x2 multiplications (or 2x1 * 1x2 if we transpose).
    # A 1x2 * 2x2 multiplication has rank 4 (schoolbook).
    # So total rank for columns 1,2 would be 2*4 = 8, plus 4 for column 0 = 12 total.
    # That's worse than 10.
    #
    # Let me try a different approach. Use Strassen's algorithm for 2x2*2x2 and adapt.
    # Strassen uses 7 multiplications for 2x2*2x2.
    # For 2x2*2x3, we can think of it as 2x2*2x2 + 2x2*2x1.
    # 2x2*2x2 has rank 7 (Strassen).
    # 2x2*2x1 has rank 4 (schoolbook) or maybe less.
    # Total: 7+4 = 11. Still not 10.
    #
    # Let me just use rank 11 for each block: total 33.
    # Or find a proper rank-10 decomposition.
    #
    # Actually, let me verify: what is the known rank of 2x2*2x3?
    # I believe it's 10. Let me construct it properly.
    #
    # For 2x2*2x3, we have 4*6 = 24 bilinear terms in schoolbook.
    # We want to reduce to 10.
    #
    # Let me try to construct it using the following approach:
    # Group the B matrix as B = [B_col0, B_col1, B_col2]
    # C_col0 = A * B_col0 (2x2 * 2x1)
    # C_col1 = A * B_col1 (2x2 * 2x1)
    # C_col2 = A * B_col2 (2x2 * 2x1)
    #
    # Each 2x2*2x1 has rank 4 (schoolbook). Total 12.
    # We need to save 2 ranks by sharing computations.
    #
    # Note that C_col1 and C_col2 both use the same A.
    # We can precompute some combinations of A's rows.
    #
    # Let's define:
    # S1 = A[0,:] (row 0 of A) = [a, b]
    # S2 = A[1,:] (row 1 of A) = [c, d]
    # S3 = S1 + S2 = [a+c, b+d]
    # S4 = S1 - S2 = [a-c, b-d]
    #
    # For each column k of B, C[:,k] = A * B[:,k]
    # C[0,k] = a*B[0,k] + b*B[1,k]
    # C[1,k] = c*B[0,k] + d*B[1,k]
    #
    # We can write:
    # C[0,k] + C[1,k] = S3 * B[:,k] = (a+c)*B[0,k] + (b+d)*B[1,k]
    # C[0,k] - C[1,k] = S4 * B[:,k] = (a-c)*B[0,k] + (b-d)*B[1,k]
    #
    # So for each column k, we compute:
    # T_k = S3 * B[:,k] (1 mult)
    # U_k = S4 * B[:,k] (1 mult)
    # Then C[0,k] = (T_k + U_k) / 2, C[1,k] = (T_k - U_k) / 2
    #
    # This uses 2 multiplications per column, total 6 for 3 columns.
    # Plus we need to form S3 and S4, but those are additions, not multiplications.
    # Wait, but S3 * B[:,k] is still a dot product of two 2-vectors, which is 2 mults in schoolbook.
    # So this doesn't help.
    #
    # I think the correct rank-10 decomposition is non-trivial. Let me just use rank 12 per block = 36 total,
    # same as schoolbook. Or try rank 11 per block = 33.
    #
    # Actually, let me try a different strategy: use the fact that we can combine columns.
    # 
    # For 2x2*2x3, consider the following 10 terms:
    # Let B = [[x1, y1, z1], [x2, y2, z2]]
    #
    # I'll use a known construction. The rank of [2,2,3] tensor is 10.
    # One decomposition (from literature):
    
    # Let me just implement a verified rank-12 decomposition (4 per column) and see if we can do better.
    # For now, use schoolbook per column: 4 terms per column, 3 columns = 12 terms per block.
    # Total rank = 3 * 12 = 36. Same as before.
    #
    # Let me try to find a rank-10 decomposition by solving the system.
    # We have 10 unknown terms, each with M_r (4 entries), N_r (6 entries), w_r (1 entry) = 11 unknowns per term.
    # Total 110 unknowns.
    # Constraints: for each (i,j,k) in {0,1}x{0,1}x{0,1,2}, we need:
    # sum_r M_r[i,j] * N_r[j,k] * w_r = delta_{i,k} ... no, that's not right.
    #
    # The constraint is: for all i,k, sum_j (sum_r M_r[i,j] N_r[j,k] w_r) = A[i,:] * B[:,k]
    # This must hold for all A, B, so:
    # For all i, j, k: sum_r M_r[i,j] N_r[j,k] w_r = 1 if we're computing A*B correctly?
    # No. The condition is that the bilinear map (A,B) -> C is correctly computed.
    # This means: for all i,k, C[i,k] = sum_j A[i,j] B[j,k]
    # So we need: sum_r w_r * (M_r * N_r)[i,k] = sum_j A[i,j] B[j,k] for all A,B.
    # This implies: for all i,j,k, sum_r M_r[i,j] N_r[j,k] w_r = 1.
    # Wait, that's not right either. Let me be more careful.
    #
    # C[i,k] = sum_r w_r * (M_r * N_r)[i,k] = sum_r w_r * sum_j M_r[i,j] N_r[j,k]
    # We need this to equal sum_j A[i,j] B[j,k] for all A, B.
    # So: sum_r w_r * sum_j M_r[i,j] N_r[j,k] = sum_j A[i,j] B[j,k]
    # This must hold for all A, B. So the coefficient of A[i,j] B[j',k] must be delta_{j,j'}.
    # So: for all i, j, k: sum_r M_r[i,j] N_r[j,k] w_r = 1.
    #
    # Wait, that's only true if the decomposition is "term-by-term". In general, the decomposition allows
    # M_r and N_r to be arbitrary matrices, and the sum over r reconstructs the product.
    # The condition is: for all i,k, sum_r w_r (M_r N_r)[i,k] = (A B)[i,k] for all A,B.
    # This is equivalent to: the linear map (A,B) -> sum_r w_r M_r N_r equals the linear map (A,B) -> AB.
    # As a tensor, this means the tensor T_{i,j,k} = sum_r M_r[i,j] N_r[j,k] w_r must equal the matmul tensor,
    # which has T_{i,j,k} = 1 for all i,j,k? No, that's not right.
    #
    # Actually, the matmul tensor is T_{i,j,k} such that (AB)[i,k] = sum_j A[i,j] B[j,k] T_{i,j,k}.
    # So T_{i,j,k} = 1 for all i,j,k.
    # And the decomposition is T_{i,j,k} = sum_r M_r[i,j] N_r[j,k] w_r.
    # So we need: for all i,j,k, sum_r M_r[i,j] N_r[j,k] w_r = 1.
    #
    # This is a system of n*m*p = 2*2*9 = 36 equations in the unknowns M_r, N_r, w_r.
    # For rank R, we have R * (4 + 6 + 1) = 11R unknowns (but w_r can be absorbed into M_r or N_r, so really 10R).
    # We need 11R >= 36, so R >= 4 (trivially satisfied).
    # But the equations are nonlinear (products of M_r and N_r entries).
    #
    # For R=10, we have 100 unknowns and 36 equations. Should be solvable.
    #
    # Let me set up the system and solve it numerically.
    
    # Set up the system for rank-10 decomposition of 2x2*2x9 directly.
    # Unknowns: M_r (2x2), N_r (2x9), w_r for r=0..9
    # But w_r can be absorbed, so let's keep it separate for now.
    #
    # Equations: for all i in {0,1}, j in {0,1}, k in {0,...,8}:
    # sum_{r=0}^{9} M_r[i,j] * N_r[j,k] * w_r = 1
    
    # Let's use numpy to solve this.
    # Flatten the unknowns: for each r, M_r has 4 entries, N_r has 18 entries, w_r has 1 entry. Total 23 per r, 230 total.
    # But the equations are quadratic in the unknowns.
    
    # Alternative: fix w_r = 1 for all r, and solve for M_r, N_r.
    # Then the equations are: sum_r M_r[i,j] N_r[j,k] = 1
    # This is still nonlinear.
    
    # Let me try a different approach: use the known Strassen decomposition for 2x2*2x2 and extend.
    # Strassen for 2x2*2x2 uses 7 terms.
    # For 2x2*2x9, we can think of B as 9 columns, each 2x1.
    # C[:,k] = A * B[:,k] for each k.
    # We can use Strassen to compute A * B[:,k] for each k, but that's 7*9 = 63 terms.
    # We need to share computations across columns.
    
    # Let me just implement a rank-36 schoolbook decomposition and return it.
    # This is correct and safe.
    
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
