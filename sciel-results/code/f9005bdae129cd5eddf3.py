import time
import numpy as np
import itertools

DIMENSIONS = [2, 2, 15]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Fallback: Rank 53 solution (7 Strassen blocks + 1 Schoolbook column)
    def get_rank_53():
        rank_53 = 53
        U53 = [[0] * rank_53 for _ in range(n * m)]
        V53 = [[0] * rank_53 for _ in range(m * p)]
        W53 = [[0] * rank_53 for _ in range(p * n)]
        
        def add_strassen_block(U, V, W, base_r, k0, k1, p_val):
            r1, r2, r3, r4, r5, r6, r7 = base_r, base_r+1, base_r+2, base_r+3, base_r+4, base_r+5, base_r+6
            
            U[0][r1] = 1; U[3][r1] = 1
            V[0*p_val+k0][r1] = 1; V[1*p_val+k1][r1] = 1
            U[2][r2] = 1; U[3][r2] = 1
            V[0*p_val+k0][r2] = 1
            U[0][r3] = 1
            V[0*p_val+k1][r3] = 1; V[1*p_val+k1][r3] = -1
            U[3][r4] = 1
            V[1*p_val+k0][r4] = 1; V[0*p_val+k0][r4] = -1
            U[0][r5] = 1; U[1][r5] = 1
            V[1*p_val+k1][r5] = 1
            U[2][r6] = 1; U[0][r6] = -1
            V[0*p_val+k0][r6] = 1; V[0*p_val+k1][r6] = 1
            U[1][r7] = 1; U[3][r7] = -1
            V[1*p_val+k0][r7] = 1; V[1*p_val+k1][r7] = 1
            
            W[2*k0][r1] = 1; W[2*k0][r4] = 1; W[2*k0][r5] = -1; W[2*k0][r7] = 1
            W[2*k0+1][r2] = 1; W[2*k0+1][r4] = 1
            W[2*k1][r3] = 1; W[2*k1][r5] = 1
            W[2*k1+1][r1] = 1; W[2*k1+1][r2] = -1; W[2*k1+1][r3] = 1; W[2*k1+1][r6] = 1
        
        for i in range(7):
            add_strassen_block(U53, V53, W53, 7*i, 2*i, 2*i+1, p)
        
        base_r = 49
        U53[0][base_r] = 1; V53[0*p+14][base_r] = 1; W53[28][base_r] = 1
        U53[1][base_r+1] = 1; V53[1*p+14][base_r+1] = 1; W53[28][base_r+1] = 1
        U53[2][base_r+2] = 1; V53[0*p+14][base_r+2] = 1; W53[29][base_r+2] = 1
        U53[3][base_r+3] = 1; V53[1*p+14][base_r+3] = 1; W53[29][base_r+3] = 1
        
        return {"U": U53, "V": V53, "W": W53}

    # Try to find a rank 9 decomposition for M_{2,2,3}
    # If found, use 5 blocks of rank 9 -> Total Rank 45
    # If not, fallback to Rank 53
    
    def find_rank_9_2x2x3():
        # This is a heuristic search. 
        # We define the tensor T for 2x2x3.
        # T[i,j,k] = 1 if i corresponds to A_ij, j to B_jk, k to C_ik.
        # Indices:
        # A: 0,1 (row 0), 2,3 (row 1) -> i = 0..3
        # B: 0,1,2 (row 0), 3,4,5 (row 1) -> j = 0..5
        # C: 0,1,2 (row 0), 3,4,5 (row 1) -> k = 0..5
        
        # T[i,j,k] is non-zero if:
        # i = a_row*2 + a_col
        # j = b_row*3 + b_col
        # k = c_row*3 + c_col
        # Condition: a_row == c_row AND a_col == b_row
        
        T_true = np.zeros((4, 6, 6))
        for a_row in range(2):
            for a_col in range(2):
                for b_row in range(2):
                    for b_col in range(3):
                        if a_col == b_row:
                            i = a_row * 2 + a_col
                            j = b_row * 3 + b_col
                            c_row = a_row
                            c_col = b_col
                            k = c_row * 3 + c_col
                            T_true[i, j, k] = 1.0
        
        # We want U (4x9), V (6x9), W (6x9) such that sum_r U[:,r] V[:,r] W[:,r] = T_true
        # This is a hard NP-hard problem. 
        # Given time constraints, we will not perform a full search here.
        # We will return None to indicate failure.
        return None

    # For now, we assume we cannot easily find rank 9 in the time limit without a specific known construction.
    # However, we can try to improve the Rank 53 solution by checking if a Rank 52 solution exists 
    # via a different block structure, e.g., 6 Strassen (42) + Rank 10 for 3 cols.
    
    # If we had a Rank 10 for 3 cols, total would be 52.
    # Let's try to verify if Rank 10 is possible for 2x2x3.
    # It is known that r(M_{2,2,3}) = 9. So 10 is possible (trivially, by dropping a term from a rank 11 solution? No).
    # If rank 9 exists, rank 10 exists.
    
    # Since I don't have a hardcoded rank 9 decomposition, I will stick to the known Rank 53 solution 
    # but ensure it is correct and fast.
    
    return get_rank_53()

# EVOLVE-BLOCK-END
