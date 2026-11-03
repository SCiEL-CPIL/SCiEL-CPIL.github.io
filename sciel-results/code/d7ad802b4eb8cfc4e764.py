"""Matrix multiplication 2x4x7 - tensor decomposition search."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    """
    Find a low-rank exact tensor decomposition for 2x4 x 4x7 matrix multiplication.
    Uses a decomposition based on splitting the middle dimension into two 2x2 blocks,
    and decomposing each 2x2x7 multiplication using Strassen for 2x2x2 sub-blocks
    and schoolbook for the remainder.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    # We decompose M[2,4,7] into two M[2,2,7] tensors.
    # A = [A1, A2], B = [B1; B2]. C = A1*B1 + A2*B2.
    
    # Function to generate decomposition for M[2,2,p]
    def decomp_2_2_p(p_val):
        # Returns U, V, W for M[2,2,p_val]
        # U shape (4, R), V shape (2*p_val, R), W shape (2*p_val, R)
        # Indices:
        # U[i*2 + j, r]
        # V[j*p_val + k, r]
        # W[k*2 + i, r]
        
        # We use the grouping [2, 2, ..., 1]
        # For p_val=7: [2, 2, 2, 1]
        
        U_list = []
        V_list = []
        W_list = []
        
        # Helper to add a rank-1 term
        def add_term(u_idx, v_idx, w_idx, val=1.0):
            # u_idx: (i, j)
            # v_idx: (j, k)
            # w_idx: (k, i)
            U_list.append(0.0) # Placeholder, will fill
            # Actually, let's build matrices directly
            
        # Let's build the matrices
        # We will determine the total rank R first.
        # Groups of size 2 use Strassen (rank 7).
        # Groups of size 1 use Schoolbook (rank 4).
        
        groups = []
        remaining = p_val
        while remaining > 0:
            if remaining >= 2:
                groups.append(2)
                remaining -= 2
            else:
                groups.append(1)
                remaining -= 1
                
        # Calculate total rank
        total_rank = 0
        for g in groups:
            if g == 2:
                total_rank += 7
            elif g == 1:
                total_rank += 4
                
        U = np.zeros((4, total_rank))
        V = np.zeros((2 * p_val, total_rank))
        W = np.zeros((2 * p_val, total_rank))
        
        col = 0
        k_offset = 0
        
        for g in groups:
            if g == 2:
                # Strassen for M[2,2,2]
                # Columns k_offset, k_offset+1
                # Strassen constants:
                # m1 = (A00 + A11) * (B00 + B11)
                # m2 = (A10 + A11) * B00
                # m3 = A00 * (B10 - B11)
                # m4 = A11 * (B01 - B00)
                # m5 = (A00 + A01) * B11
                # m6 = (A10 - A00) * (B00 + B10)
                # m7 = (A01 - A11) * (B01 + B11)
                
                # U indices: i in 0..1, j in 0..1
                # V indices: j in 0..1, k in 0..1 (relative to offset)
                # W indices: k in 0..1, i in 0..1
                
                # m1
                U[0, col] = 1; U[3, col] = 1 # A00, A11
                V[0, col] = 1; V[1, col] = 1 # B00, B11 (k=0,1? No, B is 2x2)
                # B indices: j*p + k. j=0,1. k=0,1.
                # V[0*2+0] = B00, V[0*2+1] = B01
                # V[1*2+0] = B10, V[1*2+1] = B11
                # Strassen m1: (A00+A11)(B00+B11)
                # U: A00, A11 -> 1
                # V: B00, B11 -> 1
                # W: C00, C11 -> 1? No, W contributes to C.
                # C00 = A00B00 + A01B10
                # C01 = A00B01 + A01B11
                # C10 = A10B00 + A11B10
                # C11 = A10B01 + A11B11
                
                # Let's use the standard Strassen decomposition for M[2,2,2]
                # R1 = (A00+A11)(B00+B11)
                # R2 = (A10+A11)B00
                # R3 = A00(B10-B11)
                # R4 = A11(B01-B00)
                # R5 = (A00+A01)B11
                # R6 = (A10-A00)(B00+B10)
                # R7 = (A01-A11)(B01+B11)
                
                # C00 = R1 - R4 + R5 - R7
                # C01 = R2 + R7
                # C10 = R3 + R6
                # C11 = R1 - R2 + R3 + R4
                
                # We need to map these to U, V, W.
                # U[i,j,r], V[j,k,r], W[k,i,r]
                
                # Term 1: R1
                # U: A00, A11 -> 1
                # V: B00, B11 -> 1
                # W: C00, C11 -> 1 (coeff 1)
                # C00 also gets -R4, +R5, -R7.
                
                # Let's just add the 7 terms.
                # r=0 (R1)
                U[0, col] += 1; U[3, col] += 1
                V[0, col] += 1; V[3, col] += 1 # B00, B11
                W[0, col] += 1; W[3, col] += 1 # C00, C11
                col += 1
                
                # r=1 (R2)
                # U: A10, A11 -> 1
                # V: B00 -> 1
                # W: C01 -> 1, C11 -> -1
                U[2, col] += 1; U[3, col] += 1
                V[0, col] += 1
                W[1, col] += 1; W[3, col] += -1
                col += 1
                
                # r=2 (R3)
                # U: A00 -> 1
                # V: B10 -> 1, B11 -> -1
                # W: C10 -> 1, C11 -> 1
                U[0, col] += 1
                V[2, col] += 1; V[3, col] += -1
                W[2, col] += 1; W[3, col] += 1
                col += 1
                
                # r=3 (R4)
                # U: A11 -> 1
                # V: B01 -> 1, B00 -> -1
                # W: C00 -> -1, C11 -> 1
                U[3, col] += 1
                V[1, col] += 1; V[0, col] += -1
                W[0, col] += -1; W[3, col] += 1
                col += 1
                
                # r=4 (R5)
                # U: A00, A01 -> 1
                # V: B11 -> 1
                # W: C00 -> 1
                U[0, col] += 1; U[1, col] += 1
                V[3, col] += 1
                W[0, col] += 1
                col += 1
                
                # r=5 (R6)
                # U: A10 -> 1, A00 -> -1
                # V: B00 -> 1, B10 -> 1
                # W: C10 -> 1
                U[2, col] += 1; U[0, col] += -1
                V[0, col] += 1; V[2, col] += 1
                W[2, col] += 1
                col += 1
                
                # r=6 (R7)
                # U: A01 -> 1, A11 -> -1
                # V: B01 -> 1, B11 -> 1
                # W: C01 -> 1, C00 -> -1
                U[1, col] += 1; U[3, col] += -1
                V[1, col] += 1; V[3, col] += 1
                W[1, col] += 1; W[0, col] += -1
                col += 1
                
            elif g == 1:
                # Schoolbook for M[2,2,1]
                # C00 = A00 B00
                # C01 = A00 B01 (Wait, B is 2x1, so only k=0? No, B is 2x1, so j=0,1, k=0)
                # M[2,2,1]: A is 2x2, B is 2x1. C is 2x1.
                # C0 = A00 B00 + A01 B10
                # C1 = A10 B00 + A11 B10
                
                # r=0: A00 B00 -> C0
                U[0, col] += 1
                V[0, col] += 1 # B00
                W[0, col] += 1 # C0
                col += 1
                
                # r=1: A01 B10 -> C0
                U[1, col] += 1
                V[1, col] += 1 # B10
                W[0, col] += 1 # C0
                col += 1
                
                # r=2: A10 B00 -> C1
                U[2, col] += 1
                V[0, col] += 1 # B00
                W[1, col] += 1 # C1
                col += 1
                
                # r=3: A11 B10 -> C1
                U[3, col] += 1
                V[1, col] += 1 # B10
                W[1, col] += 1 # C1
                col += 1
                
            k_offset += g
            
        return U, V, W, total_rank

    # Get decomposition for M[2,2,7]
    U1, V1, W1, R1 = decomp_2_2_p(7)
    
    # Total rank for M[2,4,7] is 2 * R1
    R_total = 2 * R1
    
    U = np.zeros((8, R_total))
    V = np.zeros((28, R_total))
    W = np.zeros((14, R_total))
    
    # First half: A1 (rows 0,1 of A), B1 (rows 0,1 of B)
    # U indices: i in 0..1, j in 0..1 -> i*4+j
    # V indices: j in 0..1, k in 0..6 -> j*7+k
    # W indices: k in 0..6, i in 0..1 -> k*2+i
    
    for r in range(R1):
        for i in range(2):
            for j in range(2):
                U[i*4 + j, r] = U1[i*2 + j, r]
                
        for j in range(2):
            for k in range(7):
                V[j*7 + k, r] = V1[j*2 + k, r]
                
        for k in range(7):
            for i in range(2):
                W[k*2 + i, r] = W1[k*2 + i, r]
                
    # Second half: A2 (rows 2,3 of A), B2 (rows 2,3 of B)
    # U indices: i in 0..1, j in 2..3 -> i*4 + (j+2)
    # V indices: j in 2..3, k in 0..6 -> (j+2)*7 + k
    # W indices: k in 0..6, i in 0..1 -> k*2+i
    
    for r in range(R1):
        for i in range(2):
            for j in range(2):
                U[i*4 + (j+2), r + R1] = U1[i*2 + j, r]
                
        for j in range(2):
            for k in range(7):
                V[(j+2)*7 + k, r + R1] = V1[j*2 + k, r]
                
        for k in range(7):
            for i in range(2):
                W[k*2 + i, r + R1] = W1[k*2 + i, r]

    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
