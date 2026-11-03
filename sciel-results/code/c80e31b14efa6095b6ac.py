import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (8, 36)
    # V: (m*p, rank) = (20, 36)
    # W: (p*n, rank) = (10, 36)
    
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    def add_strassen_block(u_rows, v_rows, w_rows):
        """
        Adds a 2x2x2 Strassen decomposition.
        u_rows: [u00, u01, u10, u11] indices in U
        v_rows: [v00, v01, v10, v11] indices in V
        w_rows: [w00, w01, w10, w11] indices in W (order: c00, c01, c10, c11)
        Returns new column index.
        """
        nonlocal col
        u00, u01, u10, u11 = u_rows
        v00, v01, v10, v11 = v_rows
        w00, w01, w10, w11 = w_rows
        
        # m1 = (a00 + a11)(b00 + b11)
        U[u00, col] += 1
        U[u11, col] += 1
        V[v00, col] += 1
        V[v11, col] += 1
        W[w00, col] += 1
        W[w10, col] += 1
        W[w01, col] += 1
        W[w11, col] += 1
        col += 1
        
        # m2 = (a10 + a11)b00
        U[u10, col] += 1
        U[u11, col] += 1
        V[v00, col] += 1
        W[w10, col] += 1
        W[w11, col] += 1
        col += 1
        
        # m3 = a00(b10 - b11)
        U[u00, col] += 1
        V[v10, col] += 1
        V[v11, col] += -1
        W[w01, col] += 1
        W[w11, col] += 1
        col += 1
        
        # m4 = a11(b00 - b10)
        U[u11, col] += 1
        V[v00, col] += 1
        V[v10, col] += -1
        W[w00, col] += 1
        W[w10, col] += 1
        col += 1
        
        # m5 = (a00 + a01)b11
        U[u00, col] += 1
        U[u01, col] += 1
        V[v11, col] += 1
        W[w00, col] += -1
        W[w01, col] += -1
        col += 1
        
        # m6 = (a10 - a00)(b00 + b01)
        U[u10, col] += 1
        U[u00, col] += -1
        V[v00, col] += 1
        V[v01, col] += 1
        W[w11, col] += -1
        col += 1
        
        # m7 = (a01 - a11)(b10 + b11)
        U[u01, col] += 1
        U[u11, col] += -1
        V[v10, col] += 1
        V[v11, col] += 1
        W[w00, col] += 1
        W[w11, col] += 1
        col += 1
        
        return col

    def add_schoolbook_221_block(u_rows, v_rows, w_rows):
        """
        Adds a 2x2x1 schoolbook decomposition.
        u_rows: [u00, u01, u10, u11] indices in U
        v_rows: [v00, v10] indices in V (only 2 rows, col 0 of the 2x1 block)
        w_rows: [w00, w10] indices in W (c00, c10)
        Returns new column index.
        """
        nonlocal col
        u00, u01, u10, u11 = u_rows
        v00, v10 = v_rows
        w00, w10 = w_rows
        
        # a00 * v00
        U[u00, col] += 1
        V[v00, col] += 1
        W[w00, col] += 1
        col += 1
        
        # a01 * v10
        U[u01, col] += 1
        V[v10, col] += 1
        W[w00, col] += 1
        col += 1
        
        # a10 * v00
        U[u10, col] += 1
        V[v00, col] += 1
        W[w10, col] += 1
        col += 1
        
        # a11 * v10
        U[u11, col] += 1
        V[v10, col] += 1
        W[w10, col] += 1
        col += 1
        
        return col

    # Part 1: j in {0, 1}
    # A1 indices: 0, 1, 4, 5
    u_rows_1 = [0, 1, 4, 5]
    
    # B1a (cols 0,1): rows 0,1 of B
    # v00=0, v01=1, v10=5, v11=6
    v_rows_1a = [0, 1, 5, 6]
    # W indices for cols 0,1: w00=0, w01=2, w10=1, w11=3
    w_rows_1a = [0, 2, 1, 3]
    
    add_strassen_block(u_rows_1, v_rows_1a, w_rows_1a)
    
    # B1b (cols 2,3): rows 0,1 of B
    # v00=2, v01=3, v10=7, v11=8
    v_rows_1b = [2, 3, 7, 8]
    # W indices for cols 2,3: w00=4, w01=6, w10=5, w11=7
    w_rows_1b = [4, 6, 5, 7]
    
    add_strassen_block(u_rows_1, v_rows_1b, w_rows_1b)
    
    # B1c (col 4): rows 0,1 of B
    # v00=4, v10=9
    v_rows_1c = [4, 9]
    # W indices for col 4: w00=8, w10=9
    w_rows_1c = [8, 9]
    
    add_schoolbook_221_block(u_rows_1, v_rows_1c, w_rows_1c)
    
    # Part 2: j in {2, 3}
    # A2 indices: 2, 3, 6, 7
    u_rows_2 = [2, 3, 6, 7]
    
    # B2a (cols 0,1): rows 2,3 of B
    # v00=10, v01=11, v10=15, v11=16
    v_rows_2a = [10, 11, 15, 16]
    # W indices for cols 0,1: w00=0, w01=2, w10=1, w11=3
    w_rows_2a = [0, 2, 1, 3]
    
    add_strassen_block(u_rows_2, v_rows_2a, w_rows_2a)
    
    # B2b (cols 2,3): rows 2,3 of B
    # v00=12, v01=13, v10=17, v11=18
    v_rows_2b = [12, 13, 17, 18]
    # W indices for cols 2,3: w00=4, w01=6, w10=5, w11=7
    w_rows_2b = [4, 6, 5, 7]
    
    add_strassen_block(u_rows_2, v_rows_2b, w_rows_2b)
    
    # B2c (col 4): rows 2,3 of B
    # v00=14, v10=19
    v_rows_2c = [14, 19]
    # W indices for col 4: w00=8, w10=9
    w_rows_2c = [8, 9]
    
    add_schoolbook_221_block(u_rows_2, v_rows_2c, w_rows_2c)
    
    # Verify col == 36
    if col != rank:
        raise ValueError(f"Expected rank {rank}, got {col}")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
