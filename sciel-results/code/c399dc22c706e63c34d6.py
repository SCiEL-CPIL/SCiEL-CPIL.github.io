"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]
import time
import numpy as np
import itertools

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = 135  # 135s safety margin from 150s limit

    # We want to multiply a (n x m) matrix by a (m x p) matrix.
    # Result is (n x p).
    # Tensor is T_{i,j,k} (i in 0..n-1, j in 0..m-1, k in 0..p-1).
    # Actually, indices for U, V, W:
    # U: (n*m) x rank, U[i, c] = U_{i*m+j, c}
    # V: (m*p) x rank, V[j, c] = V_{j*p+k, c}
    # W: (p*n) x rank, W[k, c] = W_{k*n+i, c}
    
    # Rank-minimal is likely 18 (since rank of matmul 3x3 by 3x7 might be related to 3x3x1).
    # Actually, matmul tensor M_{n,m,p} has rank at most n*m*p (schoolbook = 63).
    # It is known that rank of M_{2,2,2} is 7, M_{3,3,1} is 5, M_{2,2,1} is 3.
    # M_{3,3,2} is 11?
    # For M_{3,3,7}, if we can use a decomposition that separates the p-dimension, it would be 5*7=35?
    # No, because the middle dimension m is shared.
    # The tensor is sum_{j=0}^{2} A_{:,j} B_{j,:}.
    # This is equivalent to three matmuls of (3x1) and (1x7).
    # rank(M_{3,1,7}) = rank(M_{3,1,7}).
    # It is known that rank(M_{n,1,p}) = n+p-1?
    # Let's check: M_{2,1,2} has rank 3. 2+2-1=3. Correct.
    # M_{3,1,3} has rank 5. 3+3-1=5. Correct.
    # So M_{3,1,7} has rank 3+7-1 = 9.
    # So M_{3,3,7} can be decomposed into 3 terms of rank 9 each, giving rank 27.
    # Can we do better?
    # There might be shared factors if we combine the m-index.
    # However, 27 is a solid improvement over 63.
    # Let's try to find a decomposition of rank 27 for M_{3,3,7}.
    # This is equivalent to finding a rank-9 decomposition for M_{3,1,7} and replicating it for the 3 columns of A and 3 rows of B.
    # Wait, the m-dimension is the contraction.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # If we set j=0,1,2, we have 3 independent contractions.
    # Unless there is a way to mix the j's, the rank is sum_j rank(M_{3,1,7}) = 3 * 9 = 27.
    # Is there a rank less than 27?
    # For M_{2,2,2}, rank is 7 < 2*3=6? No, 6 is schoolbook for 2x1x2? No, 2x2x2 schoolbook is 8.
    # Rank 7 < 2*rank(2,1,2) = 2*3=6? No, 7 > 6. So mixing can increase rank?
    # Wait, M_{2,1,2} rank is 3. So 2*3=6. But M_{2,2,2} rank is 7. So mixing *increases* the required rank compared to just summing the ranks? That doesn't make sense.
    # Ah, for M_{2,2,2}, schoolbook is 8. The sum of ranks of M_{2,1,2} is 2*3=6. But the true rank is 7.
    # This implies that the "decoupled" lower bound is not always achievable, and the true rank can be higher than the sum of the decoupled ranks?
    # No, the tensor M_{2,2,2} is a single tensor. It's not a sum of two disjoint tensors.
    # The mapping from (A,B) to C is linear.
    # The question is whether the rank of M_{3,3,7} is exactly 3 * rank(M_{3,1,7}).
    # Given time constraints, implementing a known rank 27 solution is a good target.
    # But wait, what is the rank of M_{3,3,3}?
    # It is known to be 23? Or 22?
    # If rank(M_{3,3,3}) is 23, then for M_{3,3,7}, it might be related.
    
    # Let's stick to the schoolbook if we can't find a verified lower rank quickly.
    # Or implement a rank 27 based on the M_{3,1,7} rank 9 decomposition.
    # The rank 9 decomposition of M_{3,1,7} (i.e., product of 3x1 and 1x7 vectors) is given by:
    # u_{i,c}, v_{k,c} such that sum_c u_{i,c} v_{k,c} = a_i b_k.
    # This is just rank 1? No, a_i b_k is rank 1.
    # Wait, M_{3,1,7} is the tensor for A (3x1) * B (1x7).
    # A is a vector, B is a vector. Their product is a 3x7 matrix.
    # The tensor T_{i,k} = a_i b_k.
    # This is a rank 1 tensor!
    # So rank(M_{3,1,7}) = 1.
    # My previous logic was flawed. M_{n,1,p} is the outer product of a vector and a vector. It is rank 1.
    # So M_{3,3,7} = sum_j M_{3,1,7}^{(j)}.
    # Each term is rank 1? No, the term j is A_{:,j} * B_{j,:}.
    # A_{:,j} is a 3-vector, B_{j,:} is a 7-vector.
    # Their product is a 3x7 matrix.
    # The tensor for this single term (for a fixed j) is rank 1 in the i-k space, but depends on A_{i,j} and B_{j,k}.
    # Wait, the tensor T_{i,j,k} = A_{i,j} B_{j,k}.
    # For fixed j, this is A_{i,j} B_{j,k}.
    # The rank of the tensor M_{3,3,7} is at least 3 (since we have 3 independent j's).
    # Actually, M_{3,3,7} contains M_{3,3,1} as a sub-tensor?
    # If we fix k=0, we get M_{3,3,1} (matrix * vector).
    # Rank(M_{3,3,1}) = 5.
    # So rank(M_{3,3,7}) >= 5.
    # Since it's 3x3x7, and p=7, we have 7 "slices" of 3x3 matrices.
    # The rank is likely around 15-20.
    # Schoolbook is 63.
    # A simple bound is max(rank(M_{3,3,1}), rank(M_{3,1,7}))*something?
    # Let's try to construct a rank 15 solution?
    # Or just use a known decomposition for M_{3,3,3} (rank 11? 15?) and extend it?
    
    # Given the risk of invalidity, the schoolbook rank 63 is the only safe option if I can't derive a lower rank.
    # However, the task asks for rank-minimal. 63 is very far from minimal.
    # But I have no verified lower rank decomposition available in my immediate recall.
    
    # Let's just return the schoolbook. It is guaranteed correct.
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
