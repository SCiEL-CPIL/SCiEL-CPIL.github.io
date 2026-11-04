import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Total rank
    # Block 1: 2x2x2 (Strassen) -> 7
    # Block 2: 2x2x2 (Strassen) -> 7
    # Block 3: 2x2x2 (Strassen) -> 7
    # Block 4: 2x2x1 (Schoolbook) -> 4
    # Block 5: 2x1x7 (Schoolbook) -> 14
    total_rank = 7 + 7 + 7 + 4 + 14
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    current_col = 0
    
    # Helper to add a 2x2x2 Strassen block
    # A indices: i in {0, 1}, j in {j_start, j_start+1}
    # B indices: j in {j_start, j_start+1}, k in {k_start, k_start+1}
    # C indices: i in {0, 1}, k in {k_start, k_start+1}
    def add_strassen_2x2x2(j_start, k_start):
        nonlocal current_col
        
        # u_r[i, j]
        # v_r[j, k]
        # w_r[k, i]
        
        # r=1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        U[0*m + j_start, current_col] = 1
        U[1*m + (j_start+1), current_col] = 1
        V[j_start*p + k_start, current_col] = 1
        V[(j_start+1)*p + (k_start+1), current_col] = 1
        W[k_start*n + 0, current_col] = 1
        W[(k_start+1)*n + 1, current_col] = 1
        current_col += 1
        
        # r=2: (A10+A11)B00 -> C10 += 1
        U[1*m + j_start, current_col] = 1
        U[1*m + (j_start+1), current_col] = 1
        V[j_start*p + k_start, current_col] = 1
        W[k_start*n + 1, current_col] = 1
        current_col += 1
        
        # r=3: A00(B01-B11) -> C01 += 1, C11 += 1
        U[0*m + j_start, current_col] = 1
        V[j_start*p + (k_start+1), current_col] = 1
        V[(j_start+1)*p + (k_start+1), current_col] = -1
        W[(k_start+1)*n + 0, current_col] = 1
        W[(k_start+1)*n + 1, current_col] = 1
        current_col += 1
        
        # r=4: A11(B10-B00) -> C00 += 1, C10 += 1
        U[1*m + (j_start+1), current_col] = 1
        V[(j_start+1)*p + k_start, current_col] = 1
        V[j_start*p + k_start, current_col] = -1
        W[k_start*n + 0, current_col] = 1
        W[k_start*n + 1, current_col] = 1
        current_col += 1
        
        # r=5: (A00+A01)B11 -> C00 -= 1, C01 += 1
        U[0*m + j_start, current_col] = 1
        U[0*m + (j_start+1), current_col] = 1
        V[(j_start+1)*p + (k_start+1), current_col] = 1
        W[k_start*n + 0, current_col] = -1
        W[(k_start+1)*n + 0, current_col] = 1
        current_col += 1
        
        # r=6: (A10-A00)(B00+B01) -> C11 += 1
        U[1*m + j_start, current_col] = 1
        U[0*m + j_start, current_col] = -1
        V[j_start*p + k_start, current_col] = 1
        V[j_start*p + (k_start+1), current_col] = 1
        W[(k_start+1)*n + 1, current_col] = 1
        current_col += 1
        
        # r=7: (A01-A11)(B10+B11) -> C00 += 1
        U[0*m + (j_start+1), current_col] = 1
        U[1*m + (j_start+1), current_col] = -1
        V[(j_start+1)*p + k_start, current_col] = 1
        V[(j_start+1)*p + (k_start+1), current_col] = 1
        W[k_start*n + 0, current_col] = 1
        current_col += 1

    # Helper to add a 2x2x1 block
    # A indices: i in {0, 1}, j in {j_start, j_start+1}
    # B indices: j in {j_start, j_start+1}, k in {k_start}
    # C indices: i in {0, 1}, k in {k_start}
    def add_2x2x1(j_start, k_start):
        nonlocal current_col
        
        # 1. A00 B0 -> C0
        U[0*m + j_start, current_col] = 1
        V[j_start*p + k_start, current_col] = 1
        W[k_start*n + 0, current_col] = 1
        current_col += 1
        
        # 2. A01 B1 -> C0
        U[0*m + (j_start+1), current_col] = 1
        V[(j_start+1)*p + k_start, current_col] = 1
        W[k_start*n + 0, current_col] = 1
        current_col += 1
        
        # 3. A10 B0 -> C1
        U[1*m + j_start, current_col] = 1
        V[j_start*p + k_start, current_col] = 1
        W[k_start*n + 1, current_col] = 1
        current_col += 1
        
        # 4. A11 B1 -> C1
        U[1*m + (j_start+1), current_col] = 1
        V[(j_start+1)*p + k_start, current_col] = 1
        W[k_start*n + 1, current_col] = 1
        current_col += 1

    # Helper to add a 2x1x7 block
    # A indices: i in {0, 1}, j in {j_start}
    # B indices: j in {j_start}, k in {0..6}
    # C indices: i in {0, 1}, k in {0..6}
    def add_2x1x7(j_start):
        nonlocal current_col
        
        for i in range(2):
            for k in range(7):
                U[i*m + j_start, current_col] = 1
                V[j_start*p + k, current_col] = 1
                W[k*n + i, current_col] = 1
                current_col += 1

    # Apply blocks
    # Block 1: j=0,1; k=0,1
    add_strassen_2x2x2(0, 0)
    
    # Block 2: j=0,1; k=2,3
    add_strassen_2x2x2(0, 2)
    
    # Block 3: j=0,1; k=4,5
    add_strassen_2x2x2(0, 4)
    
    # Block 4: j=0,1; k=6
    add_2x2x1(0, 6)
    
    # Block 5: j=2; k=0..6
    add_2x1x7(2)
    
    # Convert to list format
    U_list = [[int(U[i,j]) for j in range(total_rank)] for i in range(n*m)]
    V_list = [[int(V[i,j]) for j in range(total_rank)] for i in range(m*p)]
    W_list = [[int(W[i,j]) for j in range(total_rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
