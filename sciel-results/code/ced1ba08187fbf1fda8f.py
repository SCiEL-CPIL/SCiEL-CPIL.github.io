import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Structured decomposition
    # Split m into two parts of size 2
    # Rank of 3x2 by 2x1 is 3
    # Total rank = 2 (parts) * 7 (cols) * 3 (rank per col) = 42
    
    rank_target = 42
    U_new = [[0] * rank_target for _ in range(nm)]
    V_new = [[0] * rank_target for _ in range(mp)]
    W_new = [[0] * rank_target for _ in range(pn)]
    
    # Mapping for 3x2 by 2x1 rank 3 decomposition
    # C_i = A_i1 B_1 + A_i2 B_2
    # Rank 3 decomposition of this tensor:
    # U_{i,1} = 1, V_{1,1} = 1, W_{1,i} = 1? No.
    # We need sum_r U_{i,j} V_{j,k} W_{k,i} = delta
    
    # For 3x2 by 2x1:
    # i in 0..2, j in 0..1, k=0
    # T_{i,j,0} = 1
    # Decomposition:
    # r=0: U_{0,0}=1, V_{0,0}=1, W_{0,0}=1?
    # Let's use the identity:
    # C_0 = A_00 B_0 + A_01 B_1
    # C_1 = A_10 B_0 + A_11 B_1
    # C_2 = A_20 B_0 + A_21 B_1
    
    # This is just 3 separate linear combinations.
    # Rank 3 decomposition:
    # For each output i, we need to compute dot product.
    # This is equivalent to 3 rank-1 tensors?
    # T_{i,j,k} = 1.
    # U_{i,j}^{(i)} = delta_{j,0}? No.
    
    # Actually, for 3x2 by 2x1, the rank is 3.
    # We can use:
    # r=0: U_{0,0}=1, V_{0,0}=1, W_{0,0}=1 -> C_0 += A_00 B_0
    # r=1: U_{0,1}=1, V_{1,0}=1, W_{0,0}=1 -> C_0 += A_01 B_1
    # r=2: U_{1,0}=1, V_{0,0}=1, W_{1,0}=1 -> C_1 += A_10 B_0
    # r=3: U_{1,1}=1, V_{1,0}=1, W_{1,0}=1 -> C_1 += A_11 B_1
    # r=4: U_{2,0}=1, V_{0,0}=1, W_{2,0}=1 -> C_2 += A_20 B_0
    # r=5: U_{2,1}=1, V_{1,0}=1, W_{2,0}=1 -> C_2 += A_21 B_1
    
    # This is rank 6?
    # Wait, B has only 1 column.
    # V index is (j, k). k=0.
    # So V_{j,0}.
    # We can share V?
    # No, V is part of the tensor.
    # But we can combine terms.
    # C_i = A_i0 B_0 + A_i1 B_1.
    # This is a rank 2 tensor for each i?
    # No, the tensor is 3x2x1.
    # Rank is 3?
    # Let's check.
    # T_{i,j,0} = 1.
    # If we use rank 3:
    # r=0: U_{0,0}=1, V_{0,0}=1, W_{0,0}=1
    # r=1: U_{0,1}=1, V_{1,0}=1, W_{0,0}=1
    # r=2: U_{1,0}=1, V_{0,0}=1, W_{1,0}=1
    # ...
    # This is rank 6.
    
    # Is there a rank 3 decomposition?
    # C = A B.
    # If we use Strassen for 2x2?
    # Pad 3x2 to 4x2?
    # Rank of 4x2 by 2x1?
    
    # Let's just use the schoolbook for the subproblems if rank is not obvious.
    # Schoolbook for 3x2 by 2x7 is 3*2*7 = 42.
    # Two of them is 84.
    # This is not better.
    
    # The hint says rank 42.
    # This implies R(3,2,7) = 21.
    # Which implies R(3,2,1) = 3.
    # My rank 6 analysis might be wrong.
    
    # Let's try to construct the rank 42 decomposition.
    
    return best_result
