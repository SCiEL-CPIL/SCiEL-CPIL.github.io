"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based hybrid decomposition for 2x3x7 matrix multiplication
    #
    # We exploit the structure: A is 2x3, B is 3x7, C is 2x7
    # Split inner dim: B = [B1 (2 cols) | B2 (2 cols) | B3 (3 cols)]
    # A = [A1 (1 col) | A2 (1 col) | A3 (1 col)]
    #
    # Strategy:
    # - Use Strassen for 2x2x2 blocks (rank 7 instead of 8)
    # - Use schoolbook for remaining parts
    #
    # For C = A @ B where A is 2x3, B is 3x7:
    # Split B into B_left (3x2) and B_right (3x5)
    # C_left = A @ B_left (2x3 @ 3x2 -> 2x2)
    # C_right = A @ B_right (2x3 @ 3x5 -> 2x5)
    #
    # Further split:
    # B_left (3x2): treat as [B1 (3x1) | B2 (3x1)]
    # C_left = A[:,0]*B1 + A[:,1]*B2 + A[:,2]*B3 ... 
    #
    # Let me think more carefully.
    #
    # A (2x3) @ B (3x7) = C (2x7)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Split the 7 columns of B into groups:
    # Group 1: 2 columns -> use 2x2x2 Strassen if we can fit
    # Actually, let me split differently.
    #
    # Split inner dimension (3) into parts:
    # C = A[:, 0:1] @ B[0:1, :] + A[:, 1:2] @ B[1:2, :] + A[:, 2:3] @ B[2:3, :]
    # Each term: (2x1) @ (1x7) = 2x7
    # That's just 3 * (2*1*7) = 42 scalar multiplications (schoolbook)
    #
    # Better: group columns of B:
    # C = A @ B where we split B's 7 columns into 2+2+3
    # B = [B_1 (3x2) | B_2 (3x2) | B_3 (3x3)]
    # C = A @ B_1 + A @ B_2 + A @ B_3 (all 2x7 results, but that's wrong dimension)
    #
    # Wait, C[i,k] = sum_j A[i,j] B[j,k]
    # If I split k into groups, each group gives a 2 x (group_size) submatrix
    # 
    # Let me try: split the 7 output columns into 2 + 2 + 3
    # C[:, 0:2] = A @ B[:, 0:2]   (2x3 @ 3x2 -> 2x2)
    # C[:, 2:4] = A @ B[:, 2:4]   (2x3 @ 3x2 -> 2x2)
    # C[:, 4:7] = A @ B[:, 4:7]   (2x3 @ 3x3 -> 2x3)
    #
    # For each 2x3 @ 3x2 block: schoolbook rank = 2*3*2 = 12
    # For 2x3 @ 3x3 block: schoolbook rank = 2*3*3 = 18
    # Total schoolbook: 12 + 12 + 18 = 42
    #
    # Can we do better? Use Strassen-like tricks.
    #
    # For 2x2 @ 2x2, Strassen uses 7 mults instead of 8.
    # For 2x3 @ 3x2: 
    #   Let A = [a b; c d; e f] (3x2) -- wait, A is 2x3
    #   A (2x3) @ B (3x2):
    #   C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    #   C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    #   C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    #   C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    #
    # This is equivalent to a 2x2 matrix multiply where "elements" are 1x2 vectors
    # C = A_vec @ B_vec where A_vec[i] = A[i,:] (row i), B_vec[j] = B[:,j] (col j)
    # Actually C[i,k] = sum_j A[i,j] B[j,k], so it's a standard 2x2 matmul with
    # entries being inner products of 3-vectors. Rank of this tensor is at most 8
    # (schoolbook for 2x2) but could be 7 (Strassen).
    #
    # Actually the tensor for 2x3@3x2 has rank at most 8 (standard 2x2 matmul rank).
    # Strassen achieves 7 for 2x2x2, and this is essentially the same structure.
    #
    # Let me use Strassen for the two 2x3@3x2 blocks (rank 7 each) and schoolbook
    # for the 2x3@3x3 block (rank 18).
    # Total: 7 + 7 + 18 = 32
    #
    # Wait, can I also optimize the 2x3@3x3 block?
    # 2x3@3x3: C[i,k] = sum_j A[i,j] B[j,k], i=0,1; j=0,1,2; k=0,1,2
    # This is like 2x2 matmul where entries are 3-vectors... no, it's 2x3 @ 3x3.
    # Rank is at most 2*3*3 = 18 (schoolbook).
    # Can we do better? The tensor M_{2,3,3} has rank at most 18, and I believe
    # it's exactly 18 (no known improvement for these small dimensions).
    #
    # So total rank: 7 + 7 + 18 = 32
    #
    # Let me also consider: split 7 columns as 3+2+2 or 2+3+2 etc.
    # 2x3@3x3 (rank 18) + 2x3@3x2 (rank 7) + 2x3@3x2 (rank 7) = 32
    # Same total.
    #
    # Let me implement Strassen for 2x3@3x2 and schoolbook for 2x3@3x3.
    
    n, m, p = 2, 3, 7
    U = []
    V = []
    W = []
    
    # Helper: add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1):
        """Add a term where U[u_idx] has val, V[v_idx] has val, W[w_idx] has val"""
        pass
    
    # We'll build U, V, W as lists of (index, value) pairs per rank column
    # Actually, let's build them as 2D arrays directly.
    
    # First, let's figure out the total rank:
    # Strassen for 2x3@3x2: rank 7
    # Strassen for 2x3@3x2: rank 7  
    # Schoolbook for 2x3@3x3: rank 18
    # Total: 32
    
    # For Strassen 2x3@3x2:
    # A is 2x3, B is 3x2
    # We can view this as: for each of the 2 output rows i and 2 output cols k,
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This is a 2x2 matmul over the field, but with "elements" being 3-dim inner products.
    # 
    # Actually, let me think of it as: C = A @ B where A is 2x3, B is 3x2.
    # We can pad A to 3x3 (add a row of zeros) and B to 3x3 (add a col of zeros),
    # then C = A_padded @ B_padded, and C is 3x3 but only top-left 2x2 is nonzero.
    # Strassen on 3x3... that's complicated.
    #
    # Simpler: think of it as 2x2 matmul where the "scalar" field is replaced by
    # the 3-dim inner product. So A_eff[i,j] = A[i,j] (scalar), B_eff[j,k] = B[j,k] (scalar),
    # and C[i,k] = sum_j A[i,j]*B[j,k]. This is literally 2x2 matrix multiplication!
    # 
    # Wait no. A is 2x3, B is 3x2. C[i,k] = sum_{j=0}^{2} A[i,j] B[j,k].
    # This is NOT 2x2 matmul. It's 2x3 @ 3x2. The inner dimension is 3, not 2.
    #
    # So I can't directly use Strassen here. Let me reconsider.
    #
    # The tensor for 2x3@3x2 has dimension 2*3*2 = 12 in the schoolbook.
    # Its rank: I need to check. It's the matrix multiplication tensor M_{2,3,2}.
    # Known: rank(M_{n,m,p}) <= n*m*p. For small cases:
    # M_{2,2,2} has rank 7 (Strassen).
    # M_{2,3,2}: Let's see. C[i,k] = sum_j A[i,j]B[j,k], i in {0,1}, k in {0,1}, j in {0,1,2}
    # = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k]
    # This can be seen as: for each j, we have a 2x2 outer product of A[:,j] and B[j,:].
    # So C = sum_j outer(A[:,j], B[j,:])
    # The rank of this tensor is at most 3 (since it's a sum of 3 rank-1 tensors).
    # Wait, that's the tensor rank of the multiplication map, not the decomposition rank.
    #
    # Actually, I'm confusing myself. Let me be precise.
    #
    # The matrix multiplication tensor T_{n,m,p} is in (F^{n*m})* x (F^{m*p})* x (F^{p*n})*
    # T(A,B,W) = sum_{i,j,k} A[i,j] B[j,k] W[k,i] = trace(A @ B @ W)
    #
    # The rank of T is the minimum number r such that T = sum_{r'=0}^{r-1} u_{r'} x v_{r'} x w_{r'}
    # where u_{r'} in F^{n*m}, v_{r'} in F^{m*p}, w_{r'} in F^{p*n}.
    #
    # For the full 2x3x7 problem: rank <= 42 (schoolbook).
    #
    # Now, if I split the output columns: C[:, 0:2] and C[:, 2:4] and C[:, 4:7]
    # The tensor for C[:, 0:2] = A @ B[:, 0:2] is T_{2,3,2} (with p=2).
    # Similarly for the others.
    #
    # The full tensor T_{2,3,7} can be written as the sum (direct sum) of T_{2,3,2}, T_{2,3,2}, T_{2,3,3}
    # because the output indices are disjoint.
    # So rank(T_{2,3,7}) <= rank(T_{2,3,2}) + rank(T_{2,3,2}) + rank(T_{2,3,3})
    #
    # Now, rank(T_{2,3,2}): C[i,k] = sum_j A[i,j] B[j,k], i=0,1; k=0,1; j=0,1,2
    # This is a sum of 3 terms: for j=0,1,2: sum_{i,k} A[i,j] B[j,k] W[k,i]
    # Each term is rank 1 (outer product of A[:,j], B[j,:], W[k,i] restricted).
    # Wait, the j-th term is: u_j = e_{i,j} (in F^{2*3}), v_j = e_{j,k} (in F^{3*2}), w_j = e_{k,i} (in F^{2*2})
    # Summed over i,k: this is not rank 1.
    #
    # Let me just count: the schoolbook decomposition of T_{2,3,2} has rank 2*3*2 = 12.
    # Can we do better? 
    #
    # T_{2,3,2}(A,B,W) = sum_{i=0}^1 sum_{j=0}^2 sum_{k=0}^1 A[i,j] B[j,k] W[k,i]
    # = sum_{i,k} W[k,i] * (sum_j A[i,j] B[j,k])
    # = sum_{i,k} W[k,i] * <A[i,:], B[:,k]>
    #
    # For fixed i,k, <A[i,:], B[:,k]> = sum_j A[i,j] B[j,k]
    # This is a bilinear form in A[i,:] and B[:,k].
    #
    # The rank of T_{2,3,2} is the minimum r such that we can write
    # sum_{i,j,k} A[i,j]B[j,k]W[k,i] = sum_{r'=0}^{r-1} u_{r'}(A) v_{r'}(B) w_{r'}(W)
    # where u_{r'} is linear in A, v_{r'} linear in B, w_{r'} linear in W.
    #
    # I believe rank(T_{2,3,2}) = 8 (same as 2x2x2 since we can embed).
    # Actually, let me think... T_{2,3,2} contains T_{2,2,2} as a sub-tensor (set A[*,2]=0).
    # So rank(T_{2,3,2}) >= rank(T_{2,2,2}) = 7.
    # And rank(T_{2,3,2}) <= 12 (schoolbook).
    # I suspect rank(T_{2,3,2}) = 8 or 9.
    #
    # For simplicity and correctness, let me just use schoolbook for all parts:
    # rank = 42. But the task says the seed has score 1.0 which corresponds to rank 42.
    # To improve, I need rank < 42.
    #
    # Let me try a different splitting. What if I use the fact that one dimension is 2?
    # 
    # C[i,k] = sum_j A[i,j] B[j,k], i in {0,1}
    # C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k] + A[0,2]B[2,k]
    # C[1,k] = A[1,0]B[0,k] + A[1,1]B[1,k] + A[1,2]B[2,k]
    #
    # Define:
    # s0 = A[0,0] + A[1,0], d0 = A[0,0] - A[1,0]
    # s1 = A[0,1] + A[1,1], d1 = A[0,1] - A[1,1]  
    # s2 = A[0,2] + A[1,2], d2 = A[0,2] - A[1,2]
    #
    # t0 = s0*B[0,k] + s1*B[1,k] + s2*B[2,k]
    # t1 = d0*B[0,k] + d1*B[1,k] + d2*B[2,k]
    #
    # C[0,k] = (t0 + t1)/2, C[1,k] = (t0 - t1)/2
    #
    # This saves 1 multiplication per k (2 mults -> 2 mults, no savings).
    # Hmm, that doesn't help.
    #
    # Let me try: for each k, C[0,k] and C[1,k] depend on A[:,0], A[:,1], A[:,2] and B[0,k], B[1,k], B[2,k].
    # C[:,k] = A^T @ B[:,k] where A^T is 3x2, B[:,k] is 3x1.
    # C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k] + A[0,2]B[2,k]
    # C[1,k] = A[1,0]B[0,k] + A[1,1]B[1,k] + A[1,2]B[2,k]
    #
    # These are two dot products of length 3. Total 6 mults per k, 42 total.
    #
    # I don't see an obvious way to reduce below 42 for these specific dimensions.
    # The known lower bound for rank(M_{2,3,7}) is... I'm not sure.
    #
    # Let me just implement the schoolbook (rank 42) correctly and ensure it passes.
    # If I can find a valid rank reduction, great, but correctness first.
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    rank = 42
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
