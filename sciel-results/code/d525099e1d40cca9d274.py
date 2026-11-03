"""Matrix multiplication 3,4,7 - rank-minimal exact tensor decomposition"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]
TIME_LIMIT = 130  # seconds, with safety margin

def _strassen_222():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    Local indices:
    A: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
    B: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
    C: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # m1 = (b+d)(e+g)
    U[1, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[2, 0] = 1
    W[2, 0] = 1; W[3, 0] = -1
    
    # m2 = (a+b)h
    U[0, 1] = 1; U[1, 1] = 1
    V[3, 1] = 1
    W[0, 1] = -1; W[1, 1] = 1
    
    # m3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # m4 = d(e-g)
    U[3, 3] = 1
    V[0, 3] = 1; V[2, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # m5 = (a+d)(e+h)
    U[0, 4] = 1; U[3, 4] = 1
    V[0, 4] = 1; V[3, 4] = 1
    W[0, 4] = 1; W[3, 4] = 1
    
    # m6 = (c+d)f
    U[2, 5] = 1; U[3, 5] = 1
    V[1, 5] = 1
    W[0, 5] = 1
    
    # m7 = (a-b)(g+e)
    U[0, 6] = 1; U[1, 6] = -1
    V[2, 6] = 1; V[0, 6] = 1
    W[3, 6] = 1
    
    return U, V, W

def _dot_1x2x3():
    """
    Returns U, V, W for (1x2) * (2x3). Rank 6.
    A: (0,0), (0,1) -> 0, 1
    B: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2) -> 0, 1, 2, 3, 4, 5
    C: (0,0), (0,1), (0,2) -> 0, 1, 2
    """
    rank = 6
    U = np.zeros((2, rank))
    V = np.zeros((6, rank))
    W = np.zeros((3, rank))
    
    # C0 = A0*B00 + A1*B10
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    U[1, 1] = 1; V[3, 1] = 1; W[0, 1] = 1
    
    # C1 = A0*B01 + A1*B11
    U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
    U[1, 3] = 1; V[4, 3] = 1; W[1, 3] = 1
    
    # C2 = A0*B02 + A1*B12
    U[0, 4] = 1; V[2, 4] = 1; W[2, 4] = 1
    U[1, 5] = 1; V[5, 5] = 1; W[2, 5] = 1
    
    return U, V, W

def _dot_1x2x4():
    """
    Returns U, V, W for (1x2) * (2x4). Rank 8.
    A: (0,0), (0,1) -> 0, 1
    B: (0,0)..(0,3), (1,0)..(1,3) -> 0..7
    C: (0,0)..(0,3) -> 0..3
    """
    rank = 8
    U = np.zeros((2, rank))
    V = np.zeros((8, rank))
    W = np.zeros((4, rank))
    
    for k in range(4):
        # Ck = A0*B0k + A1*B1k
        U[0, 2*k] = 1; V[k, 2*k] = 1; W[k, 2*k] = 1
        U[1, 2*k+1] = 1; V[4+k, 2*k+1] = 1; W[k, 2*k+1] = 1
        
    return U, V, W

def _dot_2x2x1():
    """
    Returns U, V, W for (2x2) * (2x1). Rank 4.
    A: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
    B: (0,0), (1,0) -> 0, 1
    C: (0,0), (1,0) -> 0, 1
    """
    rank = 4
    U = np.zeros((4, rank))
    V = np.zeros((2, rank))
    W = np.zeros((2, rank))
    
    # C0 = A00*B00 + A01*B10
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
    
    # C1 = A10*B00 + A11*B10
    U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
    U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
    
    return U, V, W

def construct():
    """Find rank-minimal exact tensor decomposition for 3x4 * 4x7."""
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Global dimensions
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (21, rank)
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    # Helper to embed local matrices into global blocks
    def embed_block(U_loc, V_loc, W_loc, row_offsets_A, col_offsets_A, row_offsets_B, col_offsets_B, row_offsets_C, col_offsets_C):
        """
        row_offsets_A: list of (start_row, num_rows) for A partition
        col_offsets_A: list of (start_col, num_cols) for A partition
        etc.
        This function is too complex for simple concatenation. 
        Instead, we will manually construct the global U, V, W by iterating over the terms.
        """
        pass

    # We will construct the global matrices directly.
    # Total rank is sum of ranks of all sub-blocks.
    # Sub-blocks:
    # 1. C11: 1x4 * 4x3. Split A cols 2+2, B rows 2+2.
    #    - A[0, 0:2] * B[0:2, 0:3] -> 1x2 * 2x3. Rank 6.
    #    - A[0, 2:4] * B[2:4, 0:3] -> 1x2 * 2x3. Rank 6.
    # 2. C12: 1x4 * 4x4. Split A cols 2+2, B rows 2+2.
    #    - A[0, 0:2] * B[0:2, 3:7] -> 1x2 * 2x4. Rank 8.
    #    - A[0, 2:4] * B[2:4, 3:7] -> 1x2 * 2x4. Rank 8.
    # 3. C21: 2x4 * 4x3. Split A cols 2+2, B rows 2+2.
    #    - A[1:3, 0:2] * B[0:2, 0:3] -> 2x2 * 2x3.
    #       Split B cols 2+1.
    #       - A[1:3, 0:2] * B[0:2, 0:2] -> 2x2 * 2x2. Rank 7.
    #       - A[1:3, 0:2] * B[0:2, 2:3] -> 2x2 * 2x1. Rank 4.
    #    - A[1:3, 2:4] * B[2:4, 0:3] -> 2x2 * 2x3.
    #       Split B cols 2+1.
    #       - A[1:3, 2:4] * B[2:4, 0:2] -> 2x2 * 2x2. Rank 7.
    #       - A[1:3, 2:4] * B[2:4, 2:3] -> 2x2 * 2x1. Rank 4.
    # 4. C22: 2x4 * 4x4. Split A cols 2+2, B rows 2+2.
    #    - A[1:3, 0:2] * B[0:2, 3:7] -> 2x2 * 2x4.
    #       Split B cols 2+2.
    #       - A[1:3, 0:2] * B[0:2, 3:5] -> 2x2 * 2x2. Rank 7.
    #       - A[1:3, 0:2] * B[0:2, 5:7] -> 2x2 * 2x2. Rank 7.
    #    - A[1:3, 2:4] * B[2:4, 3:7] -> 2x2 * 2x4.
    #       Split B cols 2+2.
    #       - A[1:3, 2:4] * B[2:4, 3:5] -> 2x2 * 2x2. Rank 7.
    #       - A[1:3, 2:4] * B[2:4, 5:7] -> 2x2 * 2x2. Rank 7.
    
    total_rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    def add_block(U_loc, V_loc, W_loc, 
                  a_rows, a_cols, 
                  b_rows, b_cols, 
                  c_rows, c_cols):
        """
        Adds a local block decomposition to the global lists.
        a_rows: list of global row indices for A
        a_cols: list of global col indices for A
        b_rows: list of global row indices for B
        b_cols: list of global col indices for B
        c_rows: list of global row indices for C
        c_cols: list of global col indices for C
        
        Note: The local matrices U_loc, V_loc, W_loc are indexed by local indices.
        We need to map local indices to global indices.
        """
        r = U_loc.shape[1]
        U_g = np.zeros((12, r))
        V_g = np.zeros((28, r))
        W_g = np.zeros((21, r))
        
        # Map A indices
        # Local A index -> (local_row, local_col)
        # We assume the local matrices are constructed with standard row-major ordering of the sub-matrix.
        # So local index l corresponds to (l // num_cols, l % num_cols) in the sub-matrix.
        # We need to map (local_row, local_col) to global (a_rows[local_row], a_cols[local_col]).
        
        num_a_rows = len(a_rows)
        num_a_cols = len(a_cols)
        num_b_rows = len(b_rows)
        num_b_cols = len(b_cols)
        num_c_rows = len(c_rows)
        num_c_cols = len(c_cols)
        
        for l in range(len(a_rows) * len(a_cols)):
            lr = l // num_a_cols
            lc = l % num_a_cols
            gr = a_rows[lr]
            gc = a_cols[lc]
            U_g[gr * 4 + gc, :] = U_loc[l, :]
            
        for l in range(len(b_rows) * len(b_cols)):
            lr = l // num_b_cols
            lc = l % num_b_cols
            gr = b_rows[lr]
            gc = b_cols[lc]
            V_g[gr * 7 + gc, :] = V_loc[l, :]
            
        for l in range(len(c_rows) * len(c_cols)):
            lr = l // num_c_cols
            lc = l % num_c_cols
            gr = c_rows[lr]
            gc = c_cols[lc]
            W_g[gr * 7 + gc, :] = W_loc[l, :]
            
        U_list.append(U_g)
        V_list.append(V_g)
        W_list.append(W_g)
        
    # 1. C11: A[0, 0:2] * B[0:2, 0:3]
    U11a, V11a, W11a = _dot_1x2x3()
    add_block(U11a, V11a, W11a, 
              a_rows=[0], a_cols=[0, 1], 
              b_rows=[0, 1], b_cols=[0, 1, 2], 
              c_rows=[0], c_cols=[0, 1, 2])
              
    # 2. C11: A[0, 2:4] * B[2:4, 0:3]
    U11b, V11b, W11b = _dot_1x2x3()
    add_block(U11b, V11b, W11b, 
              a_rows=[0], a_cols=[2, 3], 
              b_rows=[2, 3], b_cols=[0, 1, 2], 
              c_rows=[0], c_cols=[0, 1, 2])
              
    # 3. C12: A[0, 0:2] * B[0:2, 3:7]
    U12a, V12a, W12a = _dot_1x2x4()
    add_block(U12a, V12a, W12a, 
              a_rows=[0], a_cols=[0, 1], 
              b_rows=[0, 1], b_cols=[3, 4, 5, 6], 
              c_rows=[0], c_cols=[3, 4, 5, 6])
              
    # 4. C12: A[0, 2:4] * B[2:4, 3:7]
    U12b, V12b, W12b = _dot_1x2x4()
    add_block(U12b, V12b, W12b, 
              a_rows=[0], a_cols=[2, 3], 
              b_rows=[2, 3], b_cols=[3, 4, 5, 6], 
              c_rows=[0], c_cols=[3, 4, 5, 6])
              
    # 5. C21 Term 1: A[1:3, 0:2] * B[0:2, 0:2] (2x2x2)
    U21a1, V21a1, W21a1 = _strassen_222()
    add_block(U21a1, V21a1, W21a1, 
              a_rows=[1, 2], a_cols=[0, 1], 
              b_rows=[0, 1], b_cols=[0, 1], 
              c_rows=[1, 2], c_cols=[0, 1])
              
    # 6. C21 Term 1b: A[1:3, 0:2] * B[0:2, 2:3] (2x2x1)
    U21a2, V21a2, W21a2 = _dot_2x2x1()
    add_block(U21a2, V21a2, W21a2, 
              a_rows=[1, 2], a_cols=[0, 1], 
              b_rows=[0, 1], b_cols=[2], 
              c_rows=[1, 2], c_cols=[2])
              
    # 7. C21 Term 2: A[1:3, 2:4] * B[2:4, 0:2] (2x2x2)
    U21b1, V21b1, W21b1 = _strassen_222()
    add_block(U21b1, V21b1, W21b1, 
              a_rows=[1, 2], a_cols=[2, 3], 
              b_rows=[2, 3], b_cols=[0, 1], 
              c_rows=[1, 2], c_cols=[0, 1])
              
    # 8. C21 Term 2b: A[1:3, 2:4] * B[2:4, 2:3] (2x2x1)
    U21b2, V21b2, W21b2 = _dot_2x2x1()
    add_block(U21b2, V21b2, W21b2, 
              a_rows=[1, 2], a_cols=[2, 3], 
              b_rows=[2, 3], b_cols=[2], 
              c_rows=[1, 2], c_cols=[2])
              
    # 9. C22 Term 1a: A[1:3, 0:2] * B[0:2, 3:5] (2x2x2)
    U22a1, V22a1, W22a1 = _strassen_222()
    add_block(U22a1, V22a1, W22a1, 
              a_rows=[1, 2], a_cols=[0, 1], 
              b_rows=[0, 1], b_cols=[3, 4], 
              c_rows=[1, 2], c_cols=[3, 4])
              
    # 10. C22 Term 1b: A[1:3, 0:2] * B[0:2, 5:7] (2x2x2)
    U22a2, V22a2, W22a2 = _strassen_222()
    add_block(U22a2, V22a2, W22a2, 
              a_rows=[1, 2], a_cols=[0, 1], 
              b_rows=[0, 1], b_cols=[5, 6], 
              c_rows=[1, 2], c_cols=[5, 6])
              
    # 11. C22 Term 2a: A[1:3, 2:4] * B[2:4, 3:5] (2x2x2)
    U22b1, V22b1, W22b1 = _strassen_222()
    add_block(U22b1, V22b1, W22b1, 
              a_rows=[1, 2], a_cols=[2, 3], 
              b_rows=[2, 3], b_cols=[3, 4], 
              c_rows=[1, 2], c_cols=[3, 4])
              
    # 12. C22 Term 2b: A[1:3, 2:4] * B[2:4, 5:7] (2x2x2)
    U22b2, V22b2, W22b2 = _strassen_222()
    add_block(U22b2, V22b2, W22b2, 
              a_rows=[1, 2], a_cols=[2, 3], 
              b_rows=[2, 3], b_cols=[5, 6], 
              c_rows=[1, 2], c_cols=[5, 6])

    # Concatenate all blocks
    U_total = np.hstack(U_list)
    V_total = np.hstack(V_list)
    W_total = np.hstack(W_list)
    
    # Convert to lists
    U_res = U_total.tolist()
    V_res = V_total.tolist()
    W_res = W_total.tolist()
    
    return {"U": U_res, "V": V_res, "W": W_res}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
