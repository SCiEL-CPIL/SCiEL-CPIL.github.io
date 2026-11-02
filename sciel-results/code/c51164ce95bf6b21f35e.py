import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    R = 28
    
    U = np.zeros((nm, R), dtype=float)
    V = np.zeros((mp, R), dtype=float)
    W = np.zeros((pn, R), dtype=float)
    
    r = 0
    
    # Helper to add a Strassen 2x2x2 block
    # u_base: base index for U rows (i*3 + j_offset)
    # v_base: base index for V rows (j*5 + k_offset)
    # w_base: base index for W rows (k*2 + i)
    # k_offset: shift in k dimension for B and C
    # j_offset: shift in j dimension for A and B
    def add_strassen_block(j_off, k_off, r_start):
        # U indices:
        # A00: i=0, j=0 -> 0*3 + 0 = 0
        # A01: i=0, j=1 -> 0*3 + 1 = 1
        # A10: i=1, j=0 -> 1*3 + 0 = 3
        # A11: i=1, j=1 -> 1*3 + 1 = 4
        
        # V indices:
        # B00: j=0, k=k_off -> 0*5 + k_off
        # B01: j=0, k=k_off+1 -> 0*5 + k_off + 1
        # B10: j=1, k=k_off -> 1*5 + k_off
        # B11: j=1, k=k_off+1 -> 1*5 + k_off + 1
        
        # W indices:
        # C00: k=k_off, i=0 -> k_off*2 + 0
        # C01: k=k_off+1, i=0 -> (k_off+1)*2 + 0
        # C10: k=k_off, i=1 -> k_off*2 + 1
        # C11: k=k_off+1, i=1 -> (k_off+1)*2 + 1
        
        # Strassen Formulas:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        u_idx_A00 = 0 * 3 + 0
        u_idx_A01 = 0 * 3 + 1
        u_idx_A10 = 1 * 3 + 0
        u_idx_A11 = 1 * 3 + 1
        
        v_idx_B00 = 0 * 5 + k_off
        v_idx_B01 = 0 * 5 + (k_off + 1)
        v_idx_B10 = 1 * 5 + k_off
        v_idx_B11 = 1 * 5 + (k_off + 1)
        
        w_idx_C00 = k_off * 2 + 0
        w_idx_C01 = (k_off + 1) * 2 + 0
        w_idx_C10 = k_off * 2 + 1
        w_idx_C11 = (k_off + 1) * 2 + 1
        
        # Term 1: M1
        U[u_idx_A00, r_start + 0] += 1
        U[u_idx_A11, r_start + 0] += 1
        V[v_idx_B00, r_start + 0] += 1
        V[v_idx_B11, r_start + 0] += 1
        W[w_idx_C00, r_start + 0] += 1
        W[w_idx_C11, r_start + 0] += 1
        
        # Term 2: M2
        U[u_idx_A10, r_start + 1] += 1
        U[u_idx_A11, r_start + 1] += 1
        V[v_idx_B00, r_start + 1] += 1
        W[w_idx_C10, r_start + 1] += 1
        
        # Term 3: M3
        U[u_idx_A00, r_start + 2] += 1
        V[v_idx_B10, r_start + 2] += 1
        V[v_idx_B11, r_start + 2] += -1
        W[w_idx_C01, r_start + 2] += 1
        W[w_idx_C11, r_start + 2] += 1 # Wait, C11 has M3? No.
        # Check Strassen:
        # C01 = M3 + M5
        # C11 = M1 - M2 + M3 + M6
        # So M3 is in C01 and C11.
        W[w_idx_C01, r_start + 2] += 1
        W[w_idx_C11, r_start + 2] += 1
        
        # Term 4: M4
        U[u_idx_A11, r_start + 3] += 1
        V[v_idx_B01, r_start + 3] += 1
        V[v_idx_B00, r_start + 3] += -1
        W[w_idx_C00, r_start + 3] += 1
        W[w_idx_C10, r_start + 3] += 1
        
        # Term 5: M5
        U[u_idx_A00, r_start + 4] += 1
        U[u_idx_A01, r_start + 4] += 1
        V[v_idx_B11, r_start + 4] += 1
        W[w_idx_C00, r_start + 4] += -1
        W[w_idx_C01, r_start + 4] += 1
        
        # Term 6: M6
        U[u_idx_A10, r_start + 5] += 1
        U[u_idx_A00, r_start + 5] += -1
        V[v_idx_B00, r_start + 5] += 1
        V[v_idx_B01, r_start + 5] += 1
        W[w_idx_C11, r_start + 5] += 1
        
        # Term 7: M7
        U[u_idx_A01, r_start + 6] += 1
        U[u_idx_A11, r_start + 6] += -1
        V[v_idx_B10, r_start + 6] += 1
        V[v_idx_B11, r_start + 6] += 1
        W[w_idx_C00, r_start + 6] += 1
        W[w_idx_C11, r_start + 6] += 1
        
        return r_start + 7

    # Part 1: j in {0, 1}
    # k in {0, 1}
    r = add_strassen_block(0, 0, r)
    
    # k in {2, 3}
    r = add_strassen_block(0, 2, r)
    
    # k in {4}
    # Standard 2x2x1
    # C04 = A00 B04 + A01 B14
    # C14 = A10 B04 + A11 B14
    
    # Term 1: A00 B04 C04
    U[0, r] += 1
    V[4, r] += 1
    W[8, r] += 1
    r += 1
    
    # Term 2: A01 B14 C04
    U[1, r] += 1
    V[9, r] += 1
    W[8, r] += 1
    r += 1
    
    # Term 3: A10 B04 C14
    U[3, r] += 1
    V[4, r] += 1
    W[9, r] += 1
    r += 1
    
    # Term 4: A11 B14 C14
    U[4, r] += 1
    V[9, r] += 1
    W[9, r] += 1
    r += 1
    
    # Part 2: j in {2}
    # Outer product
    for i in range(2):
        for k in range(5):
            u_idx = i * 3 + 2
            v_idx = 2 * 5 + k
            w_idx = k * 2 + i
            
            U[u_idx, r] += 1
            V[v_idx, r] += 1
            W[w_idx, r] += 1
            r += 1
            
    # Verify r == 28
    if r != 28:
        raise ValueError(f"Expected rank 28, got {r}")
        
    # Convert to list of lists for return
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
