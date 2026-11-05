import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    dim_U = n * m  # 12
    dim_V = m * p  # 28
    dim_W = p * n  # 21
    
    # Rank 40 decomposition
    rank = 40
    U = np.zeros((dim_U, rank))
    V = np.zeros((dim_V, rank))
    W = np.zeros((dim_W, rank))
    
    # --- Block 1: M_{2,4,4} using 4x M_{2,2,2} (Rank 7 each) ---
    # Sub-blocks defined by (j_range, k_range)
    # j_range: (0,1) or (2,3)
    # k_range: (0,1) or (2,3)
    
    # Strassen Rank 7 Decomposition for M_{2,2,2}
    # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U_local[i, r], V_local[j, r], W_local[k, r]
    
    # We will define the 7 terms for a generic M_{2,2,2}
    # Term 1: (A00+A11)(B00+B11)
    # Term 2: (A10+A11)B00
    # Term 3: A00(B10-B11)
    # Term 4: A11(B01-B00)
    # Term 5: (A00+A01)B11
    # Term 6: (A10-A00)(B00+B01)
    # Term 7: (A01-A11)(B10+B11)
    
    # Mapping to U, V, W coefficients
    # U[i, r] corresponds to A_i
    # V[j, r] corresponds to B_j (local j)
    # W[k, r] corresponds to C_k (local k)
    
    # We need to map local indices to global indices
    # Global A index: i_global, j_global
    # Global B index: j_global, k_global
    # Global C index: k_global, i_global
    
    # For Block 1 (i in 0,1; j in 0..3; k in 0..3):
    # Sub-block (j_offset, k_offset)
    
    def add_strassen_block(j_offset, k_offset, base_rank):
        """
        Adds a rank-7 Strassen decomposition for M_{2,2,2} 
        at A rows 0,1; B cols j_offset..j_offset+1; C cols k_offset..k_offset+1.
        """
        # Local j indices 0,1 map to global j_offset, j_offset+1
        # Local k indices 0,1 map to global k_offset, k_offset+1
        # Local i indices 0,1 map to global i 0,1
        
        # Term 1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        # U: A00 (i=0, j=j_off+0) += 1, A11 (i=1, j=j_off+1) += 1
        # V: B00 (j=j_off+0, k=k_off+0) += 1, B11 (j=j_off+1, k=k_off+1) += 1
        # W: C00 (k=k_off+0, i=0) += 1, C11 (k=k_off+1, i=1) += 1
        
        r = base_rank
        
        # T1
        U[0*m + (j_offset+0), r] += 1
        U[1*m + (j_offset+1), r] += 1
        V[(j_offset+0)*p + (k_offset+0), r] += 1
        V[(j_offset+1)*p + (k_offset+1), r] += 1
        W[(k_offset+0)*n + 0, r] += 1
        W[(k_offset+1)*n + 1, r] += 1
        
        # T2: (A10+A11)B00 -> C10 += 1, C11 += -1 ? No.
        # Strassen:
        # M2 = (A10+A11)B00
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # So W for M2: C10 += 1, C11 += -1
        r += 1
        U[1*m + (j_offset+0), r] += 1
        U[1*m + (j_offset+1), r] += 1
        V[(j_offset+0)*p + (k_offset+0), r] += 1
        W[(k_offset+0)*n + 1, r] += 1
        W[(k_offset+1)*n + 1, r] += -1
        
        # T3: A00(B10-B11) -> C01 += 1, C10 += 1
        # C01 = M3 + M5
        # C10 = M2 + M4 ... wait.
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Re-evaluating T3:
        # M3 = A00(B10-B11)
        # U: A00 (i=0, j=j_off+0) += 1
        # V: B10 (j=j_off+1, k=k_off+0) += 1, B11 (j=j_off+1, k=k_off+1) += -1
        # W: C01 (k=k_off+1, i=0) += 1, C11 (k=k_off+1, i=1) += 1
        r += 1
        U[0*m + (j_offset+0), r] += 1
        V[(j_offset+1)*p + (k_offset+0), r] += 1
        V[(j_offset+1)*p + (k_offset+1), r] += -1
        W[(k_offset+1)*n + 0, r] += 1
        W[(k_offset+1)*n + 1, r] += 1
        
        # T4: A11(B01-B00) -> C00 += 1, C10 += 1
        # C00 = ... + M4 ...
        # C10 = M2 + M4
        r += 1
        U[1*m + (j_offset+1), r] += 1
        V[(j_offset+0)*p + (k_offset+1), r] += 1
        V[(j_offset+0)*p + (k_offset+0), r] += -1
        W[(k_offset+0)*n + 0, r] += 1
        W[(k_offset+0)*n + 1, r] += 1
        
        # T5: (A00+A01)B11 -> C00 += -1, C01 += 1
        # C00 = ... - M5 ...
        # C01 = M3 + M5
        r += 1
        U[0*m + (j_offset+0), r] += 1
        U[0*m + (j_offset+1), r] += 1
        V[(j_offset+1)*p + (k_offset+1), r] += 1
        W[(k_offset+0)*n + 0, r] += -1
        W[(k_offset+1)*n + 0, r] += 1
        
        # T6: (A10-A00)(B00+B01) -> C11 += 1
        # C11 = ... + M6 ...
        r += 1
        U[1*m + (j_offset+0), r] += 1
        U[0*m + (j_offset+0), r] += -1
        V[(j_offset+0)*p + (k_offset+0), r] += 1
        V[(j_offset+0)*p + (k_offset+1), r] += 1
        W[(k_offset+1)*n + 1, r] += 1
        
        # T7: (A01-A11)(B10+B11) -> C00 += 1
        # C00 = ... + M7 ...
        r += 1
        U[0*m + (j_offset+1), r] += 1
        U[1*m + (j_offset+1), r] += -1
        V[(j_offset+1)*p + (k_offset+0), r] += 1
        V[(j_offset+1)*p + (k_offset+1), r] += 1
        W[(k_offset+0)*n + 0, r] += 1
        
        return r + 1 # Next base rank

    # Apply 4 Strassen blocks
    # Block (0,0): j 0-1, k 0-1
    next_rank = add_strassen_block(0, 0, 0)
    # Block (0,1): j 0-1, k 2-3
    next_rank = add_strassen_block(0, 2, next_rank)
    # Block (1,0): j 2-3, k 0-1
    next_rank = add_strassen_block(2, 0, next_rank)
    # Block (1,1): j 2-3, k 2-3
    next_rank = add_strassen_block(2, 2, next_rank)
    
    # --- Block 2: M_{1,4,3} (Rank 12) ---
    # i=2, j in 0..3, k in 0..2
    # Schoolbook: C_k = sum_j A_{2,j} B_{j,k}
    # Each term (j, k) is a rank-1 contribution.
    
    for j in range(4):
        for k in range(3):
            r = next_rank
            U[2*m + j, r] = 1
            V[j*p + k, r] = 1
            W[k*n + 2, r] = 1
            next_rank += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
