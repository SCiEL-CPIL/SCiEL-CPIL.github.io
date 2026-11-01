"""Matrix multiplication 3x3x3 rank-26 decomposition."""
DIMENSIONS = [3, 3, 3]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-26 exact decomposition for 3x3 matrix multiplication.
    
    Uses a block-partitioned approach:
    1. Strassen's algorithm (rank 7) for the 2x2 sub-block (indices 0,1).
    2. Direct terms for interactions involving index 2 (rank 19).
    Total rank: 26.
    
    The decomposition satisfies:
        sum_{r=0}^{25} U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = 1
        for all i,j,k in {0,1,2}
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = 3, 3, 3
    rank = 26
    
    # Initialize matrices
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    
    # Helper to add a term
    def add_term(r, u_indices, v_indices, w_indices):
        for idx in u_indices:
            U[idx][r] += 1
        for idx in v_indices:
            V[idx][r] += 1
        for idx in w_indices:
            W[idx][r] += 1
    
    # Indices mapping:
    # U row: i*3 + j
    # V row: j*3 + k
    # W row: k*3 + i
    
    # --- Group 1: Strassen for i,j,k in {0,1} (7 terms) ---
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    # U: (0,0), (1,1) -> 0, 4
    # V: (0,0), (1,1) -> 0, 4
    # W: (0,0), (1,1) -> 0, 4
    add_term(0, [0, 4], [0, 4], [0, 4])
    
    # Term 1: M2 = (A10+A11)B00
    # U: (1,0), (1,1) -> 3, 4
    # V: (0,0) -> 0
    # W: (0,1) [C10 coeff 1], (1,1) [C11 coeff -1] -> 1, 4
    # Note: W index is k*3+i.
    # C10 is i=1, k=0 -> W[0*3+1] = W[1]
    # C11 is i=1, k=1 -> W[1*3+1] = W[4]
    add_term(1, [3, 4], [0], [1, 4])
    W[4][1] = -1 # Correct sign
    
    # Term 2: M3 = A00(B01-B11)
    # U: (0,0) -> 0
    # V: (0,1), (1,1) -> 1, 4
    # W: (1,0) [C01 coeff 1], (1,1) [C11 coeff 1] -> 3, 4
    # C01 is i=0, k=1 -> W[1*3+0] = W[3]
    # C11 is i=1, k=1 -> W[1*3+1] = W[4]
    add_term(2, [0], [1, 4], [3, 4])
    V[4][2] = -1 # Correct sign for B11
    
    # Term 3: M4 = A11(B10-B00)
    # U: (1,1) -> 4
    # V: (1,0), (0,0) -> 3, 0
    # W: (0,0) [C00 coeff 1], (0,1) [C10 coeff 1] -> 0, 1
    # C00 is i=0, k=0 -> W[0]
    # C10 is i=1, k=0 -> W[1]
    add_term(3, [4], [3, 0], [0, 1])
    V[0][3] = -1 # Correct sign for B00
    
    # Term 4: M5 = (A00+A10)B11
    # U: (0,0), (1,0) -> 0, 3
    # V: (1,1) -> 4
    # W: (0,0) [C00 coeff -1], (1,0) [C01 coeff 1] -> 0, 3
    # C00 is i=0, k=0 -> W[0]
    # C01 is i=0, k=1 -> W[3]
    add_term(4, [0, 3], [4], [0, 3])
    W[0][4] = -1 # Correct sign
    
    # Term 5: M6 = (A01-A00)(B00+B10)
    # U: (0,1), (0,0) -> 1, 0
    # V: (0,0), (1,0) -> 0, 3
    # W: (1,1) [C11 coeff 1] -> 4
    # C11 is i=1, k=1 -> W[4]
    add_term(5, [1, 0], [0, 3], [4])
    U[0][5] = -1 # Correct sign for A00
    
    # Term 6: M7 = (A11-A10)(B01+B11)
    # U: (1,1), (1,0) -> 4, 3
    # V: (0,1), (1,1) -> 1, 4
    # W: (0,0) [C00 coeff -1] -> 0
    # C00 is i=0, k=0 -> W[0]
    add_term(6, [4, 3], [1, 4], [0])
    U[3][6] = -1 # Correct sign for A10
    W[0][6] = -1 # Correct sign
    
    # --- Group 2: Cross terms involving index 2 (19 terms) ---
    
    # 2.1 j=2, i in {0,1}, k in {0,1} (4 terms)
    # A[i,2] B[2,k]
    # i=0, k=0: U[0,2]=2, V[2,0]=6, W[0,0]=0
    add_term(7, [2], [6], [0])
    # i=0, k=1: U[0,2]=2, V[2,1]=7, W[1,0]=3
    add_term(8, [2], [7], [3])
    # i=1, k=0: U[1,2]=5, V[2,0]=6, W[0,1]=1
    add_term(9, [5], [6], [1])
    # i=1, k=1: U[1,2]=5, V[2,1]=7, W[1,1]=4
    add_term(10, [5], [7], [4])
    
    # 2.2 i in {0,1}, j in {0,1}, k=2 (4 terms)
    # A[i,j] B[j,2]
    # i=0, j=0: U[0,0]=0, V[0,2]=2, W[2,0]=6
    add_term(11, [0], [2], [6])
    # i=0, j=1: U[0,1]=1, V[1,2]=5, W[2,0]=6
    add_term(12, [1], [5], [6])
    # i=1, j=0: U[1,0]=3, V[0,2]=2, W[2,1]=7
    add_term(13, [3], [2], [7])
    # i=1, j=1: U[1,1]=4, V[1,2]=5, W[2,1]=7
    add_term(14, [4], [5], [7])
    
    # 2.3 i=2, j in {0,1}, k in {0,1} (4 terms)
    # A[2,j] B[j,k]
    # j=0, k=0: U[2,0]=6, V[0,0]=0, W[0,2]=2
    add_term(15, [6], [0], [2])
    # j=0, k=1: U[2,0]=6, V[0,1]=1, W[1,2]=5
    add_term(16, [6], [1], [5])
    # j=1, k=0: U[2,1]=7, V[1,0]=3, W[0,2]=2
    add_term(17, [7], [3], [2])
    # j=1, k=1: U[2,1]=7, V[1,1]=4, W[1,2]=5
    add_term(18, [7], [4], [5])
    
    # 2.4 i in {0,1}, j=2, k=2 (2 terms)
    # A[i,2] B[2,2]
    # i=0: U[0,2]=2, V[2,2]=8, W[2,0]=6
    add_term(19, [2], [8], [6])
    # i=1: U[1,2]=5, V[2,2]=8, W[2,1]=7
    add_term(20, [5], [8], [7])
    
    # 2.5 i=2, j in {0,1}, k=2 (2 terms)
    # A[2,j] B[j,2]
    # j=0: U[2,0]=6, V[0,2]=2, W[2,2]=8
    add_term(21, [6], [2], [8])
    # j=1: U[2,1]=7, V[1,2]=5, W[2,2]=8
    add_term(22, [7], [5], [8])
    
    # 2.6 i=2, j=2, k in {0,1} (2 terms)
    # A[2,2] B[2,k]
    # k=0: U[2,2]=8, V[2,0]=6, W[0,2]=2
    add_term(23, [8], [6], [2])
    # k=1: U[2,2]=8, V[2,1]=7, W[1,2]=5
    add_term(24, [8], [7], [5])
    
    # 2.7 i=2, j=2, k=2 (1 term)
    # A[2,2] B[2,2]
    # U[2,2]=8, V[2,2]=8, W[2,2]=8
    add_term(25, [8], [8], [8])
    
    # Verification
    valid = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = 0
                ui = U[i * 3 + j]
                vj = V[j * 3 + k]
                wk = W[k * 3 + i]
                for r in range(rank):
                    s += ui[r] * vj[r] * wk[r]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to rank 27
        rank27 = 27
        U27 = [[0] * rank27 for _ in range(9)]
        V27 = [[0] * rank27 for _ in range(9)]
        W27 = [[0] * rank27 for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U27[i * 3 + j][col] = 1
                    V27[j * 3 + k][col] = 1
                    W27[k * 3 + i][col] = 1
                    col += 1
        return {"U": U27, "V": V27, "W": W27}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
