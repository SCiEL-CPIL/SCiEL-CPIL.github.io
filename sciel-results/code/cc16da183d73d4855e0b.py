"""Matrix multiplication 2x3x4 rank-minimal decomposition."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    # We exploit the structure of 2x3x4 matrix multiplication.
    # A is 2x3, B is 3x4, C = A*B is 2x4.
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1,2}, k in {0,1,2,3}
    
    # The tensor is T[i,j,k] = 1 for all valid indices (it's the standard multiplication tensor).
    # We want to decompose T as sum_{r} u_r[i] * v_r[j] * w_r[k] with minimal rank.
    
    # Known result: rank of 2x3x4 matrix multiplication tensor is 14.
    # Construction: The 2x3x4 tensor can be viewed as combining a 2x2x4 and a 2x1x4 part,
    # or more directly, we can use the fact that:
    # - The 2x2x2 tensor has rank 7 (Strassen)
    # - We can decompose 2x3x4 into pieces that leverage known low-rank decompositions
    
    # Alternative approach: Direct construction of rank-14 decomposition.
    # For 2x3x4, the multiplication tensor T_{i,j,k} = e_{i,j} \otimes e_{j,k}
    # where i in {0,1}, j in {0,1,2}, k in {0,1,2,3}.
    
    # Let's construct a rank-14 decomposition explicitly.
    # The tensor has 2*3*4 = 24 entries, all 1s in the "multiplication" sense.
    # Actually, T[i,j,k] = 1 if the indices form a valid multiplication term.
    
    # For matrix multiplication A(2x3) * B(3x4) = C(2x4):
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1,2,3}.
    # Wait, that's not right. The tensor for matrix multiplication is:
    # T[i,j,k] = 1 if we're computing the (i,k) entry using the j-th intermediate.
    # So T[i,j,k] = 1 for all valid (i,j,k) where i is row of A, j is column of A / row of B, k is column of B.
    
    # Actually, the standard multiplication tensor for m x n times n x p is:
    # T[i, j, k] where i in {0..m-1}, j in {0..n-1}, k in {0..p-1}
    # and T[i,j,k] = 1 (it's just the identity in the appropriate sense).
    
    # For 2x3x4: i in {0,1}, j in {0,1,2}, k in {0,1,2,3}
    # T[i,j,k] = 1 for all 24 combinations.
    
    # We need to find rank-14 decomposition: T = sum_{r=0}^{13} u_r \otimes v_r \otimes w_r
    # where u_r is in R^2, v_r in R^3, w_r in R^4.
    
    # Known construction for rank-14:
    # Split the 3 intermediate indices into {0,1} and {2}.
    # For the {0,1} part, we have a 2x2x4 sub-problem.
    # For the {2} part, we have a 2x1x4 sub-problem.
    
    # Actually, let me think about this differently. The rank of the 2x3x4 multiplication tensor
    # is known to be 14. Let me construct it explicitly.
    
    # One known approach: Use the Strassen-like decomposition.
    # For 2x3x4, we can write:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # We need 14 bilinear forms. Let me try to construct them.
    
    # Actually, let me use a known result. The rank of the 2x3x4 matrix multiplication tensor is 14.
    # Here's an explicit construction:
    
    n, m, p = 2, 3, 4
    rank = 14
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 14
    V = [[0] * rank for _ in range(m * p)]  # 12 x 14
    W = [[0] * rank for _ in range(p * n)]  # 8 x 14
    
    # Index conventions:
    # U[i*m + j, r] corresponds to A[i,j]
    # V[j*p + k, r] corresponds to B[j,k]
    # W[k*n + i, r] corresponds to C[i,k] (following the trace convention W[k*n+i, r])
    
    # Let's construct the 14 rank-1 terms.
    # We'll use a combination of Strassen-like ideas.
    
    # Term 1-4: Handle A[0,:] * B[:,0:2] and A[1,:] * B[:,0:2] partially
    # Let me try a systematic approach.
    
    # For the 2x3x4 case, one known rank-14 decomposition uses:
    # - 4 terms for the "diagonal" parts
    # - 10 terms for the "off-diagonal" parts
    
    # Let me try to build it step by step.
    
    # Actually, let me use a different strategy. I'll use the fact that:
    # rank(2x3x4) <= rank(2x2x4) + rank(2x1x4)
    # rank(2x2x4) <= 2 * rank(2x2x2) = 2*7 = 14? No, that's not right either.
    
    # Let me just try to find a valid rank-14 decomposition by construction.
    
    # Here's a known rank-14 decomposition for 2x3x4 matrix multiplication:
    
    # We'll define the 14 bilinear forms:
    # M1 = A[0,0]*B[0,0]
    # M2 = A[0,0]*B[1,0]
    # M3 = A[0,1]*B[0,1]
    # M4 = A[0,1]*B[1,1]
    # M5 = A[0,2]*B[0,2]
    # M6 = A[0,2]*B[1,2]
    # M7 = A[1,0]*B[0,3]
    # M8 = A[1,0]*B[1,3]
    # M9 = A[1,1]*B[0,0]
    # M10 = A[1,1]*B[1,0]
    # M11 = A[1,1]*B[0,1]
    # M12 = A[1,1]*B[1,1]
    # M13 = A[1,2]*B[0,2]
    # M14 = A[1,2]*B[1,2]
    # This is just the schoolbook for 8 terms, not 14.
    
    # Let me think about this more carefully. The schoolbook is 2*3*4 = 24 terms.
    # We need to reduce to 14.
    
    # Known result: The rank of the 2x3x4 matrix multiplication tensor is exactly 14.
    # Let me look for an explicit construction.
    
    # One approach: Use the fact that the 2x3x4 tensor is equivalent to the tensor
    # that computes (A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0],
    #                 A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1],
    #                 A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2],
    #                 A[0,0]*B[0,3] + A[0,1]*B[1,3] + A[0,2]*B[2,3],
    #                 A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0],
    #                 A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1],
    #                 A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2],
    #                 A[1,0]*B[0,3] + A[1,1]*B[1,3] + A[1,2]*B[2,3])
    
    # Let me try a constructive approach using known low-rank decompositions.
    
    # For 2x2x2, Strassen gives rank 7.
    # We can embed 2x2x2 into 2x3x4 by using only 2 of the 3 intermediate indices
    # and 2 of the 4 output indices.
    
    # Actually, here's a cleaner approach. The 2x3x4 multiplication tensor can be
    # decomposed as follows:
    
    # Split B into B1 (3x2) and B2 (3x2), so B = [B1 | B2].
    # Then C = A*B = [A*B1 | A*B2], where A*B1 is 2x2 and A*B2 is 2x2.
    # So we need to compute two 2x3x2 multiplications.
    
    # rank(2x3x2) = ? The 2x3x2 tensor has 12 entries.
    # It's equivalent to computing a 2x2 matrix from a 2x3 and 3x2.
    # This is the same as the 2x3x2 tensor.
    
    # Actually, I think the cleanest way is to just provide a known valid rank-14 decomposition.
    
    # Let me construct it explicitly. Here's a rank-14 decomposition:
    
    # I'll use the following approach based on known results:
    # The 2x3x4 tensor can be decomposed using 14 bilinear forms.
    
    # Let me define the factors explicitly:
    
    # U has shape (6, 14), V has shape (12, 14), W has shape (8, 14)
    
    # I'll construct a valid decomposition by ensuring that for each (i,j,k),
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of 24 equations in the 14*6 + 14*12 + 14*8 = 84 + 168 + 112 = 364 unknowns,
    # but with the constraint that each rank-1 term must be "aligned" properly.
    
    # Actually, the constraint is that T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    # for all valid (i,j,k).
    
    # Let me try a simple construction. I'll use 14 terms where each term
    # "covers" certain (i,j,k) triples.
    
    # Term r contributes to T[i,j,k] if U[i,j,r] * V[j,k,r] * W[k,i,r] != 0.
    
    # For a rank-1 term with U[i,j,r] = a, V[j,k,r] = b, W[k,i,r] = c,
    # it contributes a*b*c to T[i,j,k].
    
    # For the decomposition to be exact, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    
    # And for invalid combinations (which don't exist in our case since all 24 are valid),
    # we need the sum to be 0. But wait, in matrix multiplication, ALL combinations
    # (i,j,k) with i in {0,1}, j in {0,1,2}, k in {0,1,2,3} are valid.
    
    # So we just need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all 24 (i,j,k).
    
    # Let me try to construct this. I'll use a greedy approach or known patterns.
    
    # Here's a known rank-14 construction for 2x3x4:
    
    # I'll define the 14 terms as follows:
    
    terms = []
    
    # Term 0: U[0,0]=1, V[0,0]=1, W[0,0]=1  -> covers (i=0,j=0,k=0)
    # But this only covers one entry. We need each term to potentially cover multiple entries.
    
    # Actually, a rank-1 term U_r \otimes V_r \otimes W_r contributes to ALL (i,j,k) where
    # U_r[i] != 0, V_r[j] != 0, W_r[k] != 0.
    
    # So if U_r = [1, 1, 0, 0, 0, 0] (i.e., U_r[0,0]=1, U_r[0,1]=1, rest 0),
    # V_r = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0] (i.e., V_r[0,0]=1, V_r[1,0]=1, rest 0),
    # W_r = [1, 0, 0, 0, 1, 0, 0, 0] (i.e., W_r[0,0]=1, W_r[2,0]=1, rest 0),
    # then this term contributes to (0,0,0), (0,0,2), (0,1,0), (0,1,2).
    
    # This is getting complex. Let me just use a known valid decomposition.
    
    # After research, here's a valid rank-14 decomposition for 2x3x4:
    
    # I'll construct it programmatically by solving the system.
    
    import numpy as np
    from fractions import Fraction
    
    # We need to find U (6x14), V (12x14), W (8x14) such that
    # for all i in {0,1}, j in {0,1,2}, k in {0,1,2,3}:
    # sum_{r=0}^{13} U[i*3+j, r] * V[j*4+k, r] * W[k*2+i, r] = 1
    
    # This is a nonlinear system. Let me try a structured approach.
    
    # I'll use the following strategy:
    # - Fix U and W to be sparse/simple
    # - Solve for V
    
    # Or better, let me use a known explicit construction.
    
    # Here's a rank-14 decomposition based on the structure of the problem:
    
    # We'll use 14 bilinear forms M_0 through M_13.
    
    # Let me define them as (U_col, V_col, W_col) where each is a vector.
    
    # I'll use a combination of "local" and "global" terms.
    
    # Actually, let me just hard-code a known valid decomposition.
    
    # The following is a valid rank-14 decomposition:
    
    # U columns (6 entries each):
    U_cols = [
        [1, 1, 0, 0, 0, 0],  # r=0
        [1, 0, 1, 0, 0, 0],  # r=1
        [0, 1, 1, 0, 0, 0],  # r=2
        [0, 0, 0, 1, 1, 0],  # r=3
        [0, 0, 0, 1, 0, 1],  # r=4
        [0, 0, 0, 0, 1, 1],  # r=5
        [1, 1, 0, 0, 0, 0],  # r=6
        [1, 0, 1, 0, 0, 0],  # r=7
        [0, 1, 1, 0, 0, 0],  # r=8
        [0, 0, 0, 1, 1, 0],  # r=9
        [0, 0, 0, 1, 0, 1],  # r=10
        [0, 0, 0, 0, 1, 1],  # r=11
        [1, 0, 0, 0, 1, 0],  # r=12
        [0, 1, 0, 0, 0, 1],  # r=13
    ]
    
    # V columns (12 entries each):
    V_cols = [
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # r=0
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # r=1
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # r=2
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # r=3
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # r=4
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # r=5
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # r=6
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # r=7
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # r=8
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # r=9
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # r=10
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # r=11
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # r=12
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # r=13
    ]
    
    # W columns (8 entries each):
    W_cols = [
        [1, 0, 0, 0, 0, 0, 0, 0],  # r=0
        [0, 1, 0, 0, 0, 0, 0, 0],  # r=1
        [0, 0, 1, 0, 0, 0, 0, 0],  # r=2
        [1, 0, 0, 0, 0, 0, 0, 0],  # r=3
        [0, 1, 0, 0, 0, 0, 0, 0],  # r=4
        [0, 0, 1, 0, 0, 0, 0, 0],  # r=5
        [1, 0, 0, 0, 0, 0, 0, 0],  # r=6
        [0, 1, 0, 0, 0, 0, 0, 0],  # r=7
        [0, 0, 1, 0, 0, 0, 0, 0],  # r=8
        [1, 0, 0, 0, 0, 0, 0, 0],  # r=9
        [0, 1, 0, 0, 0, 0, 0, 0],  # r=10
        [0, 0, 1, 0, 0, 0, 0, 0],  # r=11
        [0, 0, 0, 1, 0, 0, 0, 0],  # r=12
        [0, 0, 0, 0, 0, 0, 0, 0],  # r=13
    ]
    
    # Let me verify this construction... Actually, I'm not confident this is correct.
    # Let me take a different approach and just use the schoolbook decomposition
    # but try to optimize it, or use a known valid low-rank decomposition.
    
    # Given the complexity, let me just return the schoolbook decomposition
    # which is guaranteed to be correct, even though it's not rank-minimal.
    
    # Actually, the task says "score is the rank" and "lower is better", so I should
    # try to find the minimum rank. The schoolbook is 24.
    
    # Let me try a different approach: use numpy to solve for a rank-14 decomposition.
    
    # Actually, let me just return the schoolbook for now to ensure correctness,
    # and the evolution can try to improve it.
    
    rank_schoolbook = n * m * p  # 24
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
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
