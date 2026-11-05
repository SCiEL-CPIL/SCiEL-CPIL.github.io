import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Partitions
    # I: {0,1}, {2}
    # J: {0,1}, {2,3}
    # K: {0,1}, {2,3}, {4,5}, {6}
    
    I_blocks = [ [0, 1], [2] ]
    J_blocks = [ [0, 1], [2, 3] ]
    K_blocks = [ [0, 1], [2, 3], [4, 5], [6] ]
    
    # Initialize U, V, W with 0s. We don't know rank yet, so we build lists of columns.
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vals, v_vals, w_vals):
        """
        u_vals: dict mapping (i, j) to coeff
        v_vals: dict mapping (j, k) to coeff
        w_vals: dict mapping (k, i) to coeff
        """
        U_row = [0] * (n * m)
        V_row = [0] * (m * p)
        W_row = [0] * (p * n)
        
        for (i, j), val in u_vals.items():
            U_row[i * m + j] = val
        for (j, k), val in v_vals.items():
            V_row[j * p + k] = val
        for (k, i), val in w_vals.items():
            W_row[k * n + i] = val
            
        U_cols.append(U_row)
        V_cols.append(V_row)
        W_cols.append(W_row)

    def add_strassen_block(I_idx, J_idx, K_idx):
        """
        Adds Strassen decomposition for a 2x2x2 block.
        I_idx, J_idx, K_idx are lists of 2 indices.
        """
        i0, i1 = I_idx
        j0, j1 = J_idx
        k0, k1 = K_idx
        
        # M1 = (A00 + A11)(B00 + B11)
        # C00 += 1, C11 += 1
        u = {(i0,j0): 1, (i1,j1): 1}
        v = {(j0,k0): 1, (j1,k1): 1}
        w = {(k0,i0): 1, (k1,i1): 1}
        add_term(u, v, w)
        
        # M2 = (A10 + A11)B00
        # C10 += 1
        u = {(i1,j0): 1, (i1,j1): 1}
        v = {(j0,k0): 1}
        w = {(k0,i1): 1}
        add_term(u, v, w)
        
        # M3 = A00(B10 - B11)
        # C01 += 1
        u = {(i0,j0): 1}
        v = {(j1,k0): 1, (j1,k1): -1}
        w = {(k0,i0): 1}
        add_term(u, v, w)
        
        # M4 = A11(B01 - B00)
        # C00 += 1, C10 += 1
        u = {(i1,j1): 1}
        v = {(j0,k1): 1, (j0,k0): -1}
        w = {(k1,i0): 1, (k0,i1): 1}
        add_term(u, v, w)
        
        # M5 = (A00 + A01)B11
        # C00 -= 1, C01 += 1
        u = {(i0,j0): 1, (i0,j1): 1}
        v = {(j1,k1): 1}
        w = {(k1,i0): -1, (k1,i0): 1} # Wait, C00 and C01 are different indices.
        # C00 is (k0, i0). C01 is (k1, i0).
        # M5 contributes to C00 with -1 and C01 with +1.
        # So w = {(k0,i0): -1, (k1,i0): 1}
        w = {(k0,i0): -1, (k1,i0): 1}
        add_term(u, v, w)
        
        # M6 = (A10 - A00)(B00 + B10)
        # C11 += 1
        u = {(i1,j0): 1, (i0,j0): -1}
        v = {(j0,k0): 1, (j1,k0): 1}
        w = {(k1,i1): 1}
        add_term(u, v, w)
        
        # M7 = (A01 - A11)(B01 + B11)
        # C00 += 1, C11 -= 1
        u = {(i0,j1): 1, (i1,j1): -1}
        v = {(j0,k1): 1, (j1,k1): 1}
        w = {(k0,i0): 1, (k1,i1): -1}
        add_term(u, v, w)

    def add_schoolbook_block(I_idx, J_idx, K_idx):
        """
        Adds schoolbook decomposition for a block.
        """
        for i in I_idx:
            for j in J_idx:
                for k in K_idx:
                    u = {(i,j): 1}
                    v = {(j,k): 1}
                    w = {(k,i): 1}
                    add_term(u, v, w)

    # Iterate over all combinations of blocks
    for I_b in I_blocks:
        for J_b in J_blocks:
            for K_b in K_blocks:
                if len(I_b) == 2 and len(J_b) == 2 and len(K_b) == 2:
                    add_strassen_block(I_b, J_b, K_b)
                else:
                    add_schoolbook_block(I_b, J_b, K_b)

    # Convert to numpy arrays
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    # Ensure integer/half-integer type (floats are fine if they are exact, but ints are safer)
    # The coefficients are integers.
    U = U.astype(int)
    V = V.astype(int)
    W = W.astype(int)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
