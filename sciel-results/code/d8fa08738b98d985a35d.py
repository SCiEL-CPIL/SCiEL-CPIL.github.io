import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 recursive Strassen decomposition for 4x4 matrix multiplication.
    
    Convention:
      C[i,j] = sum_k A[i,k] * B[k,j]
      U index: i*4 + k  (A row i, A col k)
      V index: k*4 + j  (B row k, B col j)
      W index: j*4 + i  (C col j, C row i)  [trace convention W[k*n+i]]
    """
    n = m = p = 4
    rank = 49
    
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)
    
    def u_idx(i, k):
        return i * 4 + k
    
    def v_idx(k, j):
        return k * 4 + j
    
    def w_idx(i, j):
        return j * 4 + i
    
    # Strassen 7 products for 2x2 matrices [[a,b],[c,d]]*[[e,f],[g,h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Each product: (U_factor, V_factor, W_contrib)
    # U_factor: {pos_in_A: coeff}, V_factor: {pos_in_B: coeff}, W_contrib: {pos_in_C: coeff}
    # Positions: (0,0)=a/e/C11, (0,1)=b/f/C12, (1,0)=c/g/C21, (1,1)=d/h/C22
    
    strassen = [
        # P1 = (a+d)(e+h)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, 
         {(0,0): 1, (1,1): 1}),
        # P2 = (c+d)e
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, 
         {(1,0): 1, (1,1): -1}),
        # P3 = a(f-h)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, 
         {(0,1): 1, (1,1): 1}),
        # P4 = d(g-e)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, 
         {(0,0): 1, (1,0): 1}),
        # P5 = (a+b)h
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, 
         {(0,0): -1, (0,1): 1}),
        # P6 = (c-a)(e+f)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, 
         {(1,1): 1}),
        # P7 = (b-d)(g+h)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, 
         {(0,0): 1}),
    ]
    
    # Build 4x4 by nesting: outer 2x2 blocks of 2x2 submatrices, each using Strassen
    for t in range(7):
        outer_U, outer_V, outer_W = strassen[t]
        for s in range(7):
            r_idx = t * 7 + s
            inner_U, inner_V, inner_W = strassen[s]
            
            u_vec = np.zeros(16)
            v_vec = np.zeros(16)
            w_vec = np.zeros(16)
            
            # Combine U factors: outer block position + inner position -> actual index
            for (bl_r, bl_c), l_coeff in outer_U.items():
                if l_coeff == 0:
                    continue
                for (ir, ic), i_coeff in inner_U.items():
                    total = l_coeff * i_coeff
                    if total == 0:
                        continue
                    a_row = bl_r * 2 + ir
                    a_col = bl_c * 2 + ic
                    u_vec[u_idx(a_row, a_col)] += total
            
            # Combine V factors
            for (bl_r, bl_c), r_coeff in outer_V.items():
                if r_coeff == 0:
                    continue
                for (ir, ic), i_coeff in inner_V.items():
                    total = r_coeff * i_coeff
                    if total == 0:
                        continue
                    b_row = bl_r * 2 + ir
                    b_col = bl_c * 2 + ic
                    v_vec[v_idx(b_row, b_col)] += total
            
            # Combine W factors
            for (bl_r, bl_c), c_coeff in outer_W.items():
                if c_coeff == 0:
                    continue
                for (ir, ic), i_coeff in inner_W.items():
                    total = c_coeff * i_coeff
                    if total == 0:
                        continue
                    c_row = bl_r * 2 + ir
                    c_col = bl_c * 2 + ic
                    w_vec[w_idx(c_row, c_col)] += total
            
            U[:, r_idx] = u_vec
            V[:, r_idx] = v_vec
            W[:, r_idx] = w_vec
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
