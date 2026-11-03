import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split A (4x5) into A1 (4x4) and A2 (4x1)
    # Split B (5x6) into B1 (4x6) and B2 (1x6)
    # C = A1 B1 + A2 B2
    
    # Part 1: A1 (4x4) * B1 (4x6)
    # Split B1 into B1a (4x4) and B1b (4x2)
    # C1 = A1 B1a + A1 B1b
    
    # A1 B1a is 4x4x4. Rank 49 (Bini et al.)
    # A1 B1b is 4x4x2. Rank 8 (Schoolbook for m=2? No, R(4x4x2) is 8? 
    # C_{i,k} = sum_{j=1}^2 A_{i,j} B_{j,k}. 
    # For each j,k, term is A_{i,j} B_{j,k} E_{i,k}. 
    # This is 4x2=8 terms? 
    # Yes, for each j,k, we multiply vector A_{:,j} by scalar B_{j,k}. 
    # This is a rank 1 operation in the tensor sense? 
    # No, the tensor rank of M_{n,2,p} is 2*np? 
    # Let's assume rank 8 for 4x4x2.
    
    # A2 B2 is 4x1x6. Rank 24.
    
    # Total rank = 49 + 8 + 24 = 81.
    
    # We need the actual matrices.
    # Since I don't have the 49 rank matrices hardcoded, I will use schoolbook for 4x4x4 (rank 64).
    # Total rank = 64 + 8 + 24 = 96.
    
    # Let's try to get 96.
    
    U = []
    V = []
    W = []
    
    # Helper to add schoolbook decomposition for n x m x p
    def add_schoolbook(n_sub, m_sub, p_sub, i_offset, j_offset, k_offset):
        # i_offset: start row of A
        # j_offset: start col of A / start row of B
        # k_offset: start col of B / start col of C
        rank_sub = n_sub * m_sub * p_sub
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    U.append(0)
                    V.append(0)
                    W.append(0)
                    # U index: (i_offset+i)*m + (j_offset+j)
                    U[-1] = 1
                    # V index: (j_offset+j)*p + (k_offset+k)
                    V[-1] = 1
                    # W index: (k_offset+k)*n + (i_offset+i)
                    W[-1] = 1
    
    # This is inefficient. Let's build matrices directly.
    
    rank_total = 0
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_val, v_val, w_val):
        U_cols.append(u_val)
        V_cols.append(v_val)
        W_cols.append(w_val)
        
    # Part 1: A1 (4x4) * B1a (4x4)
    # i in 0..3, j in 0..3, k in 0..3
    for i in range(4):
        for j in range(4):
            for k in range(4):
                u = [0]*(4*5)
                v = [0]*(5*6)
                w = [0]*(6*4)
                u[i*5 + j] = 1
                v[j*6 + k] = 1
                w[k*4 + i] = 1
                add_term(u, v, w)
                
    # Part 2: A1 (4x4) * B1b (4x2)
    # i in 0..3, j in 0..3, k in 4..5
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                u = [0]*(4*5)
                v = [0]*(5*6)
                w = [0]*(6*4)
                u[i*5 + j] = 1
                v[j*6 + k] = 1
                w[k*4 + i] = 1
                add_term(u, v, w)
                
    # Part 3: A2 (4x1) * B2 (1x6)
    # i in 0..3, j in 4, k in 0..5
    for i in range(4):
        for k in range(6):
            u = [0]*(4*5)
            v = [0]*(5*6)
            w = [0]*(6*4)
            u[i*5 + 4] = 1
            v[4*6 + k] = 1
            w[k*4 + i] = 1
            add_term(u, v, w)
            
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
