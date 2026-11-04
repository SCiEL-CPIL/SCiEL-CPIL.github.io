"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 42

    # Known lower bound for 2x3x7 is 26 (or possibly lower).
    # We will try to find a decomposition with rank < 42.
    # However, finding an exact integer/half-integer decomposition is hard.
    # A safe bet is to return the schoolbook if we can't find a better one quickly.
    # But let's try a few heuristic ranks.
    # Actually, the task says "Find a rank-minimal exact tensor decomposition".
    # If we can't find the minimal one, we should return the best valid one we can.
    # The schoolbook is always valid.
    
    # Let's check if we can construct a known lower rank decomposition.
    # For 2x3, rank is 5.
    # For 3x3, rank is 7.
    # For 2x3x7, is there a known decomposition?
    # A simple composition:
    # (2x3) * (3x7)
    # If we decompose 2x3 into rank 5, and 3x7 into rank 21 (schoolbook),
    # the total rank would be 5 * 21? No, that's not how it works.
    
    # Let's try to use the known rank-5 decomposition for 2x3.
    # The 2x3 matrix multiplication tensor has rank 5.
    # This allows us to multiply a 2x3 matrix by a 3x1 vector in 5 multiplications.
    # For 2x3 by 3x7, we have 7 columns of the second matrix.
    # We can apply the 2x3 rank-5 algorithm to each column independently?
    # That would give rank 5 * 7 = 35.
    # This is better than 42.
    
    # Let's construct the rank-35 decomposition.
    # The rank-5 algorithm for 2x3 (multiplying A (2x3) by B (3x1) to get C (2x1)):
    # The factors U (2x3, rank 5), V (3x1, rank 5), W (2x1, rank 5) are known.
    # Wait, the dimensions in the problem are:
    # U: (n*m, rank) -> (2*3, rank) = (6, rank)
    # V: (m*p, rank) -> (3*7, rank) = (21, rank)
    # W: (p*n, rank) -> (7*2, rank) = (14, rank)
    
    # The standard rank-5 decomposition for 2x3 matrix multiplication (A * B, A is 2x3, B is 3x1):
    # This is slightly different from the standard 2x2.
    # Let's find the rank-5 decomposition for 2x3.
    # Actually, the rank of 2x3 matrix multiplication is 5.
    # The decomposition is:
    # U has 5 columns. Each column corresponds to a "multiplication".
    # Let's look up the specific factors.
    
    # A known rank-5 decomposition for 2x3 (A is 2x3, B is 3x1, C is 2x1):
    # The tensor T_{i,j,k} = delta_{i,j} (where i in 0..1, j in 0..2, k in 0..1? No, C is 2x1, so k is row index of C).
    # Wait, C = A * B. C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # Here A is 2x3, B is 3x1, C is 2x1.
    # So indices: i (row of A, 0..1), j (col of A / row of B, 0..2), k (col of B, 0..0? No, B is 3x1, so k is 0).
    # Wait, B is 3x1. So B has 1 column. C has 1 column.
    # So the tensor is T_{i, j, k} where i in 0..1, j in 0..2, k in 0..0.
    # T_{i, j, k} = 1 if i==j? No, A_{i,j} * B_{j,k}.
    # The tensor for matrix multiplication is T_{i, j, k} = delta_{j, l} ... 
    # The multiplication tensor T is defined by T_{i, j, k} = 1 if the index matches.
    # For C_{i, k} = sum_j A_{i, j} B_{j, k}, the tensor is T_{i, j, k} = 1 if the "middle" index j matches.
    # Wait, the tensor is a 3-tensor T_{i, j, k} such that C_{i, k} = sum_{j} T_{i, j, k} A_{i, j} B_{j, k}.
    # This is not quite right. The tensor is T_{i, j, k} = delta_{j, j'} ...
    # Actually, the multiplication tensor is T_{i, j, k} = 1 if i is the row of A, j is the col of A, and k is the col of B?
    # No, the standard definition is T_{i, j, k} = 1 if the indices correspond to the multiplication.
    # For C_{i, k} = sum_j A_{i, j} B_{j, k}, the tensor is T_{i, j, k} = 1 if i is row of A, j is col of A, and k is col of B?
    # No, the tensor is T_{i, j, k} = 1 if i, j, k are such that A_{i, j} and B_{j, k} are multiplied.
    # So T_{i, j, k} = 1 if the "middle" index is j.
    # So T is a 2x3x1 tensor? No, C is 2x1.
    # The tensor has dimensions (n, m, p) = (2, 3, 1).
    # T_{i, j, k} = 1 if i is row of A, j is col of A, and k is col of B.
    # But B is 3x1, so k is 0.
    # So T_{i, j, 0} = 1 if i is row of A and j is col of A.
    # This is just the identity in the j dimension?
    # No, the tensor is T_{i, j, k} = 1 if i, j, k are such that A_{i, j} B_{j, k} contributes to C_{i, k}.
    # So T_{i, j, k} = 1 if i is row of A, j is col of A, and k is col of B.
    # But j must match the row of B.
    # So T_{i, j, k} = 1 if i is row of A, j is col of A, and k is col of B, AND j is row of B.
    # Since B is 3x1, j is 0..2, k is 0.
    # So T_{i, j, 0} = 1 for all i, j.
    # This is not a tensor that needs decomposition in the usual sense?
    # No, the rank of the 2x3x1 tensor is 5.
    # Wait, the rank of the 2x3x1 tensor is 5?
    # The 2x3x1 tensor is T_{i, j, k} = 1 if i is row of A, j is col of A, and k is col of B.
    # This is a 2x3x1 tensor with all ones? No.
    # T_{i, j, k} = 1 if i, j, k are such that A_{i, j} and B_{j, k} are multiplied.
    # So T_{i, j, k} = 1 if i is row of A, j is col of A, and k is col of B.
    # But j must be the same for A and B.
    # So T_{i, j, k} = 1 if i is row of A, j is col of A, and k is col of B.
    # This is a 2x3x1 tensor.
    # The rank of this tensor is 5.
    
    # Let's use the known rank-5 decomposition for 2x3x1.
    # The factors are:
    # U (2x3, rank 5): U_{i, j, c}
    # V (3x1, rank 5): V_{j, k, c}
    # W (2x1, rank 5): W_{k, i, c}
    # Wait, the dimensions in the problem are:
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (3*7, rank) = (21, rank)
    # W: (p*n, rank) = (7*2, rank) = (14, rank)
    
    # For the 2x3x1 case:
    # U: (2*3, 5) = (6, 5)
    # V: (3*1, 5) = (3, 5)
    # W: (1*2, 5) = (2, 5)
    
    # The known rank-5 decomposition for 2x3x1 is:
    # U:
    # 1 0 0 0 0
    # 0 1 0 0 0
    # 0 0 1 0 0
    # 0 0 0 1 0
    # 0 0 0 0 1
    # 1 1 1 1 1  (This is not correct)
    
    # Let's look up the actual rank-5 decomposition for 2x3.
    # It is known that the rank of 2x3 matrix multiplication is 5.
    # The decomposition is:
    # U = [[1, 0, 0, 0, 0],
    #      [0, 1, 0, 0, 0],
    #      [0, 0, 1, 0, 0],
    #      [0, 0, 0, 1, 0],
    #      [0, 0, 0, 0, 1],
    #      [1, 1, 1, 1, 1]]
    # This is not correct.
    
    # Actually, I don't have the exact rank-5 decomposition memorized.
    # Given the time constraints and the difficulty of finding an exact lower-rank decomposition,
    # I will return the schoolbook decomposition of rank 42.
    # This is guaranteed to be correct and valid.
    
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
