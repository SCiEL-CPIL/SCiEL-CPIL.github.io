import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Decomposition of 3x4x6 using 3x2x2 blocks
    # 4 = 2+2, 6 = 2+2+2
    # 6 blocks of 3x2x2
    # Each 3x2x2 has rank 3
    # Total rank 18
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)] # 3*4 = 12 rows
    V = [[0] * rank for _ in range(m * p)] # 4*6 = 24 rows
    W = [[0] * rank for _ in range(p * n)] # 6*3 = 18 rows
    
    col = 0
    for i_block in range(2): # A split into two 3x2
        for k_block in range(3): # B split into three 2x2
            # Block A is rows i_block*2 to i_block*2+1 of A (cols 0..1 of block)
            # Block B is rows k_block*2 to k_block*2+1 of B (rows 0..1 of block)
            # Wait, A is 3x4. Split into 3x2 and 3x2.
            # A1 is cols 0,1. A2 is cols 2,3.
            # B is 4x6. Split into 2x2 blocks?
            # B rows 0,1 correspond to A col 0,1.
            # B rows 2,3 correspond to A col 2,3.
            # B cols 0,1; 2,3; 4,5.
            
            # Term: A_i1 B_k1
            # A indices: i (0..2), j (0..1)
            # B indices: j (0..1), k (0..1)
            
            # Rank 1 decomposition of 3x2x2
            # U_{i,j,r}
            # V_{j,k,r}
            # W_{k,i,r}
            
            # For 1x2x2 rank 1:
            # U_{0,j}=1
            # V_{j,k}=1
            # W_{k,0}=1
            
            # For 3x2x2, we have 3 copies of 1x2x2 for i=0,1,2.
            # So for each i, we need 1 rank-1 term?
            # No, the rank-1 term for 1x2x2 covers all j,k.
            # So for fixed i, we have 1 term.
            # Total 3 terms per block.
            
            for i in range(n):
                # r = col
                # U[i, j, r] = 1 for j in block
                # V[j, k, r] = 1 for j,k in block
                # W[k, i, r] = 1 for k in block
                
                # A col offset: i_block * 2
                # B row offset: i_block * 2
                # B col offset: k_block * 2
                
                a_j_offset = i_block * 2
                b_j_offset = i_block * 2
                b_k_offset = k_block * 2
                
                # U: row i, col a_j_offset + j
                for j in range(2):
                    U[i * m + (a_j_offset + j)][col] = 1
                    
                # V: row b_j_offset + j, col b_k_offset + k
                for j in range(2):
                    for k in range(2):
                        V[(b_j_offset + j) * p + (b_k_offset + k)][col] = 1
                        
                # W: row b_k_offset + k, col i
                for k in range(2):
                    W[(b_k_offset + k) * n + i][col] = 1
                    
                col += 1
                
    return {"U": U, "V": V, "W": W}
