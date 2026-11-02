import time
import sys

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    # We use 3 blocks of 2x2 multiplication using Strassen's algorithm (rank 7 each)
    # Total rank = 3 * 7 = 21
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms definitions for 2x2 multiplication
    # A indices: 00, 01, 10, 11 -> global A: i*2+j
    # B indices: 00, 01, 10, 11 -> global B: j*6+k
    # C indices: 00, 01, 10, 11 -> global C: k*2+i
    
    # Term: M = L(A) * M(B)
    # L(A) is a vector of length 4 (for 2x2 A)
    # M(B) is a vector of length 4 (for 2x2 B block)
    # W(C) is a vector of length 4 (for 2x2 C block)
    
    # We define the 7 Strassen terms
    # Each term is (U_vec, V_vec, W_vec)
    # U_vec corresponds to coefficients of A_{00}, A_{01}, A_{10}, A_{11}
    # V_vec corresponds to coefficients of B_{00}, B_{01}, B_{10}, B_{11}
    # W_vec corresponds to coefficients for C_{00}, C_{01}, C_{10}, C_{11}
    
    # Note: The task's W index is k*n+i.
    # C_{00} -> k=0, i=0 -> idx 0
    # C_{01} -> k=1, i=0 -> idx 2
    # C_{10} -> k=0, i=1 -> idx 1
    # C_{11} -> k=1, i=1 -> idx 3
    
    # So W_vec order should be [coeff for C00, coeff for C10, coeff for C01, coeff for C11]
    # Wait, the index is k*n+i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # So the order in the W matrix for a 2x2 block is [C00, C10, C01, C11].
    
    # Strassen Formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A00+A11)B10
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A01+A11)B00
    # M6 = (A00-A01)(B01+B11)
    # M7 = (A10-A00)(B00+B10)
    
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M6
    
    # Let's define the vectors for each M term.
    # A indices: 0:A00, 1:A01, 2:A10, 3:A11
    # B indices: 0:B00, 1:B01, 2:B10, 3:B11
    
    terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # C00=1, C10=0, C01=0, C11=1
        # M2
        ([1, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, -1]), # C00=0, C10=1, C01=0, C11=-1
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]), # C00=0, C10=0, C01=1, C11=1
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [-1, 1, 0, 0]), # C00=-1, C10=1, C01=0, C11=0
        # M5
        ([0, 1, 0, 1], [1, 0, 0, 0], [-1, 0, 1, 0]), # C00=-1, C10=0, C01=1, C11=0
        # M6
        ([1, -1, 0, 0], [0, 1, 0, 1], [0, 0, 0, -1]), # C00=0, C10=0, C01=0, C11=-1
        # M7
        ([1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 0, 0]), # C00=1, C10=0, C01=0, C11=0
    ]
    
    # Verify C11 for M6:
    # C11 = M1 - M2 + M3 - M6
    # M6 coeff for C11 is -1. Correct.
    
    # Now, we assign these to the 3 blocks.
    block_starts = [0, 2, 4]
    
    for b_idx, k_start in enumerate(block_starts):
        for t_idx in range(7):
            col = b_idx * 7 + t_idx
            u_vec, v_vec, w_vec = terms[t_idx]
            
            # Map A indices to global U
            # A is 2x2. Global index i*m + j.
            # u_vec[0] is A00 (i=0, j=0) -> idx 0
            # u_vec[1] is A01 (i=0, j=1) -> idx 1
            # u_vec[2] is A10 (i=1, j=0) -> idx 2
            # u_vec[3] is A11 (i=1, j=1) -> idx 3
            for i in range(2):
                for j in range(2):
                    local_idx = i * 2 + j
                    global_idx = i * m + j
                    U[global_idx][col] = u_vec[local_idx]
                    
            # Map B indices to global V
            # B is 2x6. Block b_idx uses columns k_start, k_start+1.
            # v_vec[0] is B00 (j=0, k=k_start) -> global j*6 + k_start
            # v_vec[1] is B01 (j=0, k=k_start+1) -> global j*6 + k_start+1
            # v_vec[2] is B10 (j=1, k=k_start) -> global j*6 + k_start
            # v_vec[3] is B11 (j=1, k=k_start+1) -> global j*6 + k_start+1
            
            # v_vec index mapping:
            # 0: (0, 0) local -> (0, k_start) global
            # 1: (0, 1) local -> (0, k_start+1) global
            # 2: (1, 0) local -> (1, k_start) global
            # 3: (1, 1) local -> (1, k_start+1) global
            
            for j in range(2):
                for k_local in range(2):
                    local_idx = j * 2 + k_local
                    k_global = k_start + k_local
                    global_idx = j * p + k_global
                    V[global_idx][col] = v_vec[local_idx]
                    
            # Map C indices to global W
            # C is 6x2? No, C is 2x6.
            # Wait, the task says W index is k*n + i.
            # C_{ik} is the element at row i, col k.
            # The index in W is k*n + i.
            # So for a fixed block (k_start, k_start+1), the C elements are:
            # C_{0, k_start}, C_{1, k_start}, C_{0, k_start+1}, C_{1, k_start+1}
            
            # w_vec order: [C00, C10, C01, C11] in local block coords.
            # Local C00: i=0, k_local=0 -> global k=k_start, i=0 -> idx k_start*n + 0
            # Local C10: i=1, k_local=0 -> global k=k_start, i=1 -> idx k_start*n + 1
            # Local C01: i=0, k_local=1 -> global k=k_start+1, i=0 -> idx (k_start+1)*n + 0
            # Local C11: i=1, k_local=1 -> global k=k_start+1, i=1 -> idx (k_start+1)*n + 1
            
            # w_vec[0] is coeff for local C00
            # w_vec[1] is coeff for local C10
            # w_vec[2] is coeff for local C01
            # w_vec[3] is coeff for local C11
            
            # Map w_vec to global W
            # w_vec[0] -> W[k_start*n + 0]
            # w_vec[1] -> W[k_start*n + 1]
            # w_vec[2] -> W[(k_start+1)*n + 0]
            # w_vec[3] -> W[(k_start+1)*n + 1]
            
            W[k_start * n + 0][col] = w_vec[0]
            W[k_start * n + 1][col] = w_vec[1]
            W[(k_start + 1) * n + 0][col] = w_vec[2]
            W[(k_start + 1) * n + 1][col] = w_vec[3]
            
    return {"U": U, "V": V, "W": W}

