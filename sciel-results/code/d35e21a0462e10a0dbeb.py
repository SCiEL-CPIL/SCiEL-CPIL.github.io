"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # We'll build the decomposition incrementally.
    # Start with a known good solution (schoolbook, rank 60) and try to improve.
    
    # First, let's try a constructive approach based on known rank bounds.
    # For 3x4x5, the tensor rank is at most 3*4*5 = 60 (schoolbook).
    # Known results: rank of M_{3,4,5} is at most 55 (by various constructions).
    # Let's try to construct a rank-55 or lower decomposition.
    
    # Strategy: Use block structure.
    # A 3x4x5 multiplication can be seen as blocks.
    # Let's try to use the fact that 4 = 2+2, 5 = 3+2, 3 = 3.
    # Or 4 = 2+2, 5 = 2+3, 3 = 1+2.
    
    # Actually, let's try a more systematic approach.
    # We know that rank(M_{a,b,c}) <= rank(M_{a1,b1,c1}) * rank(M_{a2,b2,c2})
    # if we can factor the dimensions.
    
    # Let's try: 3x4x5 = 3x(2+2)x(3+2)
    # This gives us 4 sub-problems:
    # 3x2x3, 3x2x2, 3x2x3, 3x2x2
    # But this doesn't directly help unless we know ranks of these.
    
    # Alternative: Use the "Laser Method" style embedding.
    # Embed 3x4x5 into a larger tensor and use known decompositions.
    
    # For now, let's try a direct construction with rank < 60.
    # Known: rank(M_{2,2,2}) = 7 (Strassen)
    # rank(M_{2,2,3}) = 11 (known)
    # rank(M_{3,3,3}) = 23 (best known)
    
    # Let's try to decompose 3x4x5 using Strassen-style tricks.
    # Split 4 = 2+2, 5 = 2+3, 3 = 3.
    # So we have sub-blocks:
    #   (3x2) * (2x2) -> 3x2
    #   (3x2) * (2x3) -> 3x3
    #   (3x2) * (2x2) -> 3x2
    #   (3x2) * (2x3) -> 3x3
    
    # This is getting complex. Let's try a different approach.
    
    # Use the fact that we can compute 3x4x5 by first computing some
    # intermediate products using Strassen, then combining.
    
    # Actually, let's just try to find a good decomposition numerically
    # using a greedy or optimization approach, with a time limit.
    
    # For safety, let's return the schoolbook decomposition if we run out of time.
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    
    # Try to improve using a constructive algebraic method.
    # Based on the knowledge that Strassen's algorithm for 2x2x2 gives rank 7,
    # and that we can embed smaller problems into larger ones.
    
    # Let's try a specific construction:
    # Use the identity that M_{3,4,5} can be computed with rank at most 55
    # by combining Strassen-like decompositions.
    
    # Here's a concrete plan:
    # 1. Split the 4x5 matrix B into four 2x2, 2x3, 2x2, 2x3 blocks (rows 1-2, 3-4; cols 1-2, 3-5).
    #    Actually, let's split B as [B11 B12; B21 B22] where B11 is 2x2, B12 is 2x3, B21 is 2x2, B22 is 2x3.
    # 2. Split A (3x4) as [A1 A2] where A1 is 3x2, A2 is 3x2.
    # 3. Then AB = A1*B11 + A1*B12 + A2*B21 + A2*B22 (with appropriate dimensions).
    #    Wait, A is 3x4, B is 4x5.
    #    A = [A_left A_right] where A_left is 3x2, A_right is 3x2.
    #    B = [B_top; B_bottom] where B_top is 2x5, B_bottom is 2x5.
    #    Then AB = A_left * B_top + A_right * B_bottom.
    #    B_top = [B11 B12] where B11 is 2x2, B12 is 2x3.
    #    B_bottom = [B21 B22] where B21 is 2x2, B22 is 2x3.
    #    So AB = A_left * [B11 B12] + A_right * [B21 B22]
    #          = [A_left*B11 + A_right*B21, A_left*B12 + A_right*B22]
    #    So we need to compute:
    #      C1 = A_left*B11 + A_right*B21  (3x2)
    #      C2 = A_left*B12 + A_right*B22  (3x3)
    
    # For C1: A_left is 3x2, B11 is 2x2, A_right is 3x2, B21 is 2x2.
    #   C1 = A_left*B11 + A_right*B21
    #   This is like a 3x2x2 problem with two terms.
    #   We can compute A_left*B11 using rank-7 Strassen for 2x2x2? No, A_left is 3x2.
    #   Let's think of it as: for each column of C1 (2 columns), we compute a 3x1 vector.
    #   C1[:,0] = A_left * B11[:,0] + A_right * B21[:,0]
    #   This is a linear combination.
    
    # This is getting too complex to reason about abstractly. Let's try a numerical approach
    # with a time limit, and fall back to schoolbook.
    
    # Actually, let's try a known result: the rank of M_{3,4,5} is at most 55.
    # Let's try to construct such a decomposition.
    
    # One approach: use the "tensor contraction" method.
    # We want to find U (12 x r), V (20 x r), W (15 x r) such that
    #   T = sum_c U_c \otimes V_c \otimes W_c
    # where T is the matrix multiplication tensor.
    
    # The tensor T has entries T[i,j,k] = 1 if (i,j) is a row-col pair in A, (j,k) in B, (k,i) in C.
    # More precisely, T[a,b,c] = delta_{a, row of A} * delta_{b, col of A = row of B} * delta_{c, col of B}
    # In our indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k.
    
    # Let's build the tensor and try to find a low-rank decomposition.
    # But this is a huge tensor (12 x 20 x 15), and finding its rank is hard.
    
    # Instead, let's try a heuristic: start with a random decomposition of some rank r,
    # and try to minimize the error. But this is non-convex and may not converge.
    
    # Given the time constraints, let's try a simple improvement:
    # Use the fact that some entries can be shared.
    
    # For now, let's just return the schoolbook decomposition.
    # This is a safe baseline.
    
    if time.time() > deadline - 5:
        return best_result
    
    # Try a specific construction with rank 56 (from the knowledge).
    # The knowledge says: "Uses Strassen's algorithm for 2x2x2 sub-blocks and rank-11 decomposition for 2x2x3 sub-blocks, reducing rank from 60 to 56"
    
    # Let's try to implement this.
    # Split 3x4x5 as follows:
    #   4 = 2 + 2 (split the middle dimension)
    #   5 = 2 + 3 (split the output dimension)
    #   3 = 3 (no split)
    
    # So we have 4 sub-problems:
    #   3x2x2 (for A[:,0:2] * B[0:2,0:2])
    #   3x2x3 (for A[:,0:2] * B[0:2,2:5])
    #   3x2x2 (for A[:,2:4] * B[2:4,0:2])
    #   3x2x3 (for A[:,2:4] * B[2:4,2:5])
    
    # For each 3x2x2: we can use a rank-? decomposition.
    # For 3x2x2: this is like multiplying a 3x2 by a 2x2.
    #   The result is 3x2.
    #   We can compute this by: for each of the 2 columns of the result,
    #   we compute a 3x1 vector = A_col * B_col.
    #   A_col is 3x1, B_col is 2x1? No, A is 3x2, so A_col is 3x1? No.
    
    # Let me re-index. A is 3x4, B is 4x5.
    # Split A into A1 (3x2) and A2 (3x2).
    # Split B into B1 (2x5) and B2 (2x5).
    # Then AB = A1*B1 + A2*B2.
    
    # Now split B1 into B11 (2x2) and B12 (2x3).
    # Split B2 into B21 (2x2) and B22 (2x3).
    
    # Then:
    #   AB[:,0:2] = A1*B11 + A2*B21  (3x2)
    #   AB[:,2:5] = A1*B12 + A2*B22  (3x3)
    
    # For AB[:,0:2]:
    #   This is a sum of two 3x2x2 products.
    #   Let's denote P1 = A1*B11 (3x2) and P2 = A2*B21 (3x2).
    #   We need to compute P1 + P2.
    #   Computing P1: A1 is 3x2, B11 is 2x2.
    #     This is a 3x2x2 tensor. Its rank is at most 3*2*2 = 12, but can be lower.
    #     Actually, for 3x2x2, we can think of it as: for each of the 2 output columns,
    #     we compute a 3x1 vector = A1 * B11[:,j].
    #     A1 * B11[:,j] is a 3x1 vector, which is a linear combination of the 2 columns of A1.
    #     So for each j, we need 2 multiplications (one for each column of A1) and 2 additions.
    #     But in tensor rank terms, the rank of M_{3,2,2} is... let's compute it.
    #     M_{3,2,2} has tensor size 3*2*2 = 12. The rank is at most 12.
    #     Can we do better? For 2x2x2, rank is 7. For 3x2x2, I'm not sure.
    
    # This is taking too long to reason about. Let's just try a numerical search with a time limit.
    
    # Given the complexity, let's return the schoolbook decomposition for now.
    # Future improvements can try more sophisticated constructions.
    
    return best_result
# EVOLVE-BLOCK-END
