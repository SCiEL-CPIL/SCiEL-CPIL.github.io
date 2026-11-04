import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Use a known optimal or near-optimal decomposition for 2x3x6
    # The theoretical lower bound is ceil(2*3*6 / something) but in practice
    # for small matrices, we can do better than naive 36
    
    # Known result: 2x3x6 can be done in rank 30 or 31
    # Let's construct a rank-30 solution
    
    rank = 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strategy: Use a hybrid approach
    # - First 2 columns of B with all of A: use 2x3 structure
    # - Last 4 columns: use 2x4 structure split into 2x2 blocks
    
    # Part 1: A (2x3) * B[:, 0:2] (3x2)
    # This is a 2x3x2 problem
    # Can be done in rank 7 (optimal for 2x2x2 is 7)
    # But we have 2 output columns, so we need more
    
    # Let's use a direct construction
    # For each pair (i, j) in A and (j, k) in B, we need A[i,j]*B[j,k] -> C[i,k]
    
    # Use a sparse pattern based on known fast matrix multiplication
    # For 2x3x6, one approach is to use the fact that 6 = 2*3
    
    # Construct using tensor network / circuit approach
    # This is complex, so let's use a verified rank-33 and try to optimize
    
    # Fallback: use the seed rank-33
    rank_baseline = 33
    U_base = [[0] * rank_baseline for _ in range(n * m)]
    V_base = [[0] * rank_baseline for _ in range(m * p)]
    W_base = [[0] * rank_baseline for _ in range(p * n)]
    
    col = 0
    for k in range(p):
        U_base[0][col] = 1; V_base[k][col] = 1; W_base[2*k][col] = 1; col += 1
        U_base[3][col] = 1; V_base[k][col] = 1; W_base[2*k+1][col] = 1; col += 1
    
    for x in range(3):
        c_start = 2 * x
        U_base[1][col]=1; U_base[5][col]=1; V_base[6+c_start][col]=1; V_base[12+c_start+1][col]=1
        W_base[2*c_start][col]=1; W_base[2*c_start+3][col]=1; col+=1
        U_base[4][col]=1; U_base[5][col]=1; V_base[6+c_start][col]=1
        W_base[2*c_start+1][col]=1; W_base[2*c_start+3][col]=-1; col+=1
        U_base[1][col]=1; V_base[6+c_start+1][col]=1; V_base[12+c_start+1][col]=-1
        W_base[2*c_start+2][col]=1; W_base[2*c_start+3][col]=1; col+=1
        U_base[5][col]=1; V_base[12+c_start][col]=1; V_base[6+c_start][col]=-1
        W_base[2*c_start][col]=1; W_base[2*c_start+1][col]=1; col+=1
        U_base[1][col]=1; U_base[2][col]=1; V_base[12+c_start+1][col]=1
        W_base[2*c_start][col]=-1; W_base[2*c_start+2][col]=1; col+=1
        U_base[4][col]=1; U_base[1][col]=-1; V_base[6+c_start][col]=1; V_base[6+c_start+1][col]=1
        W_base[2*c_start+3][col]=1; col+=1
        U_base[2][col]=1; U_base[5][col]=-1; V_base[12+c_start][col]=1; V_base[12+c_start+1][col]=1
        W_base[2*c_start][col]=1; col+=1
    
    return {"U": U_base, "V": V_base, "W": W_base}
