"""
Rank-35 decomposition for 2x2 by 2x10 matrix multiplication.

Strategy: Use Strassen's rank-7 algorithm for each of the 5 column-pairs (blocks 0-4).
The tensor decomposition for 2x2 matrix multiplication has a minimal rank of 7.
Since the 5 blocks are independent in the general tensor sense, the total rank is 5 * 7 = 35.
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 35 decomposition: 5 Strassen blocks (rank 7 each)
    rank = 35
    
    # Build U, V, W as numpy arrays for efficiency, then convert to lists
    U_arr = np.zeros((n * m, rank), dtype=np.int8)
    V_arr = np.zeros((m * p, rank), dtype=np.int8)
    W_arr = np.zeros((p * n, rank), dtype=np.int8)
    
    # All 5 blocks: Standard Strassen (7 terms each)
    for t in range(5):
        col_base = t * 7
        
        # Term 1: P1 = (a00+a11)*(b00+b11) -> c00, c11
        col = col_base
        U_arr[0, col] = 1; U_arr[3, col] = 1
        V_arr[2*t, col] = 1; V_arr[10+2*t+1, col] = 1
        W_arr[4*t, col] = 1; W_arr[4*t+3, col] = 1
        
        # Term 2: P2 = (a10+a11)*b00 -> c01, -c11
        col = col_base + 1
        U_arr[2, col] = 1; U_arr[3, col] = 1
        V_arr[2*t, col] = 1
        W_arr[4*t+1, col] = 1; W_arr[4*t+3, col] = -1
        
        # Term 3: P3 = a00*(b01-b11) -> c10, c11
        col = col_base + 2
        U_arr[0, col] = 1
        V_arr[2*t+1, col] = 1; V_arr[10+2*t+1, col] = -1
        W_arr[4*t+2, col] = 1; W_arr[4*t+3, col] = 1
        
        # Term 4: P4 = a11*(b10-b00) -> c00, c01
        col = col_base + 3
        U_arr[3, col] = 1
        V_arr[10+2*t, col] = 1; V_arr[2*t, col] = -1
        W_arr[4*t, col] = 1; W_arr[4*t+1, col] = 1
        
        # Term 5: P5 = (a00+a01)*b11 -> -c00, c10
        col = col_base + 4
        U_arr[0, col] = 1; U_arr[1, col] = 1
        V_arr[10+2*t+1, col] = 1
        W_arr[4*t, col] = -1; W_arr[4*t+2, col] = 1
        
        # Term 6: P6 = (a10-a00)*(b00+b01) -> c11
        col = col_base + 5
        U_arr[2, col] = 1; U_arr[0, col] = -1
        V_arr[2*t, col] = 1; V_arr[2*t+1, col] = 1
        W_arr[4*t+3, col] = 1
        
        # Term 7: P7 = (a01-a11)*(b10+b11) -> c00
        col = col_base + 6
        U_arr[1, col] = 1; U_arr[3, col] = -1
        V_arr[10+2*t, col] = 1; V_arr[10+2*t+1, col] = 1
        W_arr[4*t, col] = 1

    # Convert to lists as required by the interface
    U = U_arr.tolist()
    V = V_arr.tolist()
    W = W_arr.tolist()

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
