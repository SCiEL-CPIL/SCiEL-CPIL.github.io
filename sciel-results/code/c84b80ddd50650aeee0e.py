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
    start_time = time.time()
    deadline = 150  # seconds, with margin for 120s limit
    
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
    
    # Build 4x4 by nesting
    for t in range(7):
        outer_U, outer_V, outer_W = strassen[t]
        for s in range(7):
            r_idx = t * 7 + s
            inner_U, inner_V, inner_W = strassen[s]
            
            u_vec = np.zeros(16)
            v_vec = np.zeros(16)
            w_vec = np.zeros(16)
            
            for (bl_r, bl_c), l_coeff in outer_U.items():
                if l_coeff == 0: continue
                for (ir, ic), i_coeff in inner_U.items():
                    total = l_coeff * i_coeff
                    if total == 0: continue
                    a_row = bl_r * 2 + ir
                    a_col = bl_c * 2 + ic
                    u_vec[u_idx(a_row, a_col)] += total
            
            for (bl_r, bl_c), r_coeff in outer_V.items():
                if r_coeff == 0: continue
                for (ir, ic), i_coeff in inner_V.items():
                    total = r_coeff * i_coeff
                    if total == 0: continue
                    b_row = bl_r * 2 + ir
                    b_col = bl_c * 2 + ic
                    v_vec[v_idx(b_row, b_col)] += total
            
            for (bl_r, bl_c), c_coeff in outer_W.items():
                if c_coeff == 0: continue
                for (ir, ic), i_coeff in inner_W.items():
                    total = c_coeff * i_coeff
                    if total == 0: continue
                    c_row = bl_r * 2 + ir
                    c_col = bl_c * 2 + ic
                    w_vec[w_idx(c_row, c_col)] += total
            
            U[:, r_idx] = u_vec
            V[:, r_idx] = v_vec
            W[:, r_idx] = w_vec
            
            # Check deadline (though unlikely to be hit)
            if time.time() - start_time > deadline:
                break
        if time.time() - start_time > deadline:
            break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
