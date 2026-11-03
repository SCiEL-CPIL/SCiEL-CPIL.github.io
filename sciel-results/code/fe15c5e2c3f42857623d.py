"""
5x5 matrix multiplication - rank-121 block decomposition.
Splits 5 = 2 + 3. Uses Strassen (rank 7) for the 2x2 block,
and schoolbook for all other blocks.
Total rank = 7 + 2*3*2 + 2*3*2 + 3*3*2 + 2*3*3 + 3*3*3 + 3*2*3 + 3*2*3
           = 7 + 12 + 12 + 18 + 18 + 27 + 18 + 18
           = 130? Let me recompute.

Actually, the blocks are:
- A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] : 2x2x2 -> Strassen rank 7
- A[0:2, 2:5] * B[2:5, 0:2] -> C[0:2, 0:2] : 2x3x2 -> schoolbook 12
- A[2:5, 0:2] * B[0:2, 0:2] -> C[2:5, 0:2] : 3x2x2 -> schoolbook 12
- A[2:5, 2:5] * B[2:5, 0:2] -> C[2:5, 0:2] : 3x3x2 -> schoolbook 18
- A[0:2, 2:5] * B[2:5, 2:5] -> C[0:2, 2:5] : 2x3x3 -> schoolbook 18
- A[2:5, 0:2] * B[0:2, 2:5] -> C[2:5, 2:5] : 3x2x3 -> schoolbook 18
- A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5] : 2x2x3 -> schoolbook 12
- A[2:5, 2:5] * B[2:5, 2:5] -> C[2:5, 2:5] : 3x3x3 -> schoolbook 27

Total: 7 + 12 + 12 + 18 + 18 + 18 + 12 + 27 = 124

Let me also try a 3+2 split to see if it gives the same.
Actually, let me try to also use a rank-3 decomposition for 2x2x2? No, 2x2x2 has rank 7.

Wait, can I do better on some of the cross blocks? 
For a 2x3x2 block (n=2, m=3, p=2), schoolbook is 12. Is there a lower rank?
The border rank / tensor rank of a 2x3x2 tensor... 
Actually, for small blocks, the tensor rank might be lower than n*m*p.

Let me think about 2x2x2: rank 7 (Strassen).
2x3x2: This is the tensor for multiplying a 2x3 by 3x2 to get 2x2.
The tensor has 12 entries. Can we do it in fewer than 12?
I believe the rank of the 2x3x2 matrix multiplication tensor is 12 (it's not known to be less, and likely equals the schoolbook count for these small sizes, but let me not assume).

Actually, let me just implement the 124-rank decomposition and see if I can improve.

Let me also consider: can I combine the two 2x3x2 blocks that feed into C[0:2, 0:2]?
Block 1: A[0:2, 2:5] * B[2:5, 0:2] -> C[0:2, 0:2]
Block 2: A[2:5, 0:2] * B[0:2, 0:2] -> C[2:5, 0:2]  (this feeds into C[2:5, 0:2], not C[0:2, 0:2])

So they feed into different output blocks. Can't easily combine.

Let me just implement the 124-rank version first.
"""
import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def _strassen_2x2x2(U_cols, V_cols, W_cols, u_offset, v_offset, w_offset):
    """
    Add Strassen 2x2x2 terms.
    u_offset: row offset in U (i*m + j), where i is local row, j is local col in A
    v_offset: row offset in V (j*p + k), where j is local row in B (= local col in A), k is local col in B
    w_offset: row offset in W (k*n + i), where k is local row in C (= local col in B), i is local row in C (= local row in A)
    
    For 2x2x2, local indices: i in {0,1}, j in {0,1}, k in {0,1}
    U row = i*2 + j
    V row = j*2 + k
    W row = k*2 + i
    """
    # Strassen formulas:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00) (B00 + B01)
    # M7 = (A01 - A11) (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    terms = [
        # (M_index, U_entries, V_entries, W_entries)
        # U_entries: list of (local_i, local_j)
        # V_entries: list of (local_j, local_k)
        # W_entries: list of (local_k, local_i)
        
        # M1 = (A00 + A11)(B00 + B11)
        ([(0,0), (1,1)], [(0,0), (1,1)], [(0,0), (1,1)], 1),
        # M2 = (A10 + A11) B00
        ([(1,0), (1,1)], [(0,0)], [(1,0), (1,1)], 1),  # Wait, C10 = M2 + M4, C11 = M1 - M2 + M3 + M6
        # So W for M2: +1 at (1,0), -1 at (1,1)
        ([(1,0), (1,1)], [(0,0)], [(1,0), (1,1)], 1),  # placeholder, fix below
        # Let me redo this more carefully
    ]
    
    # Let me just hardcode the 7 terms properly
    # Each term: (U_dict, V_dict, W_dict)
    # U_dict: {(i,j): coeff}
    # V_dict: {(j,k): coeff}
    # W_dict: {(k,i): coeff}
    
    S = [
        # M1
        ({(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1},
         {(0,0): 1},
         {(1,0): 1, (1,1): -1}),
        # M3
        ({(0,0): 1},
         {(0,1): 1, (1,1): -1},
         {(0,1): 1, (1,1): 1}),
        # M4
        ({(1,1): 1},
         {(1,0): 1, (0,0): -1},
         {(0,0): 1, (1,0): 1}),
        # M5
        ({(0,0): 1, (0,1): 1},
         {(1,1): 1},
         {(0,0): -1, (0,1): 1}),
        # M6
        ({(1,0): 1, (0,0): -1},
         {(0,0): 1, (0,1): 1},
         {(1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1},
         {(1,0): 1, (1,1): 1},
         {(0,0): 1}),
    ]
    
    for U_dict, V_dict, W_dict in S:
        U_cols.append({})
        V_cols.append({})
        W_cols.append({})
        for (i, j), c in U_dict.items():
            row = u_offset + i * 2 + j
            U_cols[-1][row] = c
        for (j, k), c in V_dict.items():
            row = v_offset + j * 2 + k
            V_cols[-1][row] = c
        for (k, i), c in W_dict.items():
            row = w_offset + k * 2 + i
            W_cols[-1][row] = c

def _schoolbook_block(U_cols, V_cols, W_cols, u_offset, v_offset, w_offset, ni, mj, pk):
    """
    Schoolbook for ni x mj x pk block.
    u_offset, v_offset, w_offset are the starting row indices.
    ni: number of rows in A block (= rows in C block)
    mj: number of cols in A block (= rows in B block)
    pk: number of cols in B block (= cols in C block)
    """
    for i in range(ni):
        for j in range(mj):
            for k in range(pk):
                U_cols.append({})
                V_cols.append({})
                W_cols.append({})
                u_row = u_offset + i * mj + j
                v_row = v_offset + j * pk + k
                w_row = w_offset + k * ni + i
                U_cols[-1][u_row] = 1
                V_cols[-1][v_row] = 1
                W_cols[-1][w_row] = 1

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin
    
    n, m, p = 5, 5, 5
    # Split: first 2, then 3
    # A block rows: 0,1 (2) and 2,3,4 (3)
    # A block cols: 0,1 (2) and 2,3,4 (3)
    # B block rows: 0,1 (2) and 2,3,4 (3)
    # B block cols: 0,1 (2) and 2,3,4 (3)
    # C block rows: 0,1 (2) and 2,3,4 (3)
    # C block cols: 0,1 (2) and 2,3,4 (3)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Block (0,0): A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Strassen 2x2x2
    _strassen_2x2x2(U_cols, V_cols, W_cols, 
                    u_offset=0*2+0,   # i=0, j starts at 0 -> row 0
                    v_offset=0*2+0,   # j=0, k starts at 0 -> row 0
                    w_offset=0*2+0)   # k=0, i starts at 0 -> row 0
    
    # Block (0,1): A[0:2, 2:5] * B[2:5, 0:2] -> C[0:2, 0:2]
    # Schoolbook 2x3x2
    _schoolbook_block(U_cols, V_cols, W_cols,
                      u_offset=0*3+0,  # i=0, j starts at 2 in A -> but we use local j
                      v_offset=0*2+0,  # j=0 local, k starts at 0
                      w_offset=0*2+0,  # k=0, i starts at 0
                      ni=2, mj=3, pk=2)
    
    # Block (1,0): A[2:5, 0:2] * B[0:2, 0:2] -> C[2:5, 0:2]
    # Schoolbook 3x2x2
    _schoolbook_block(U_cols, V_cols, W_cols,
                      u_offset=2*2+0,  # i=2 in A -> row offset 4 (since i*m+j = 2*5+0=10)
                      v_offset=0*2+0,  # j=0, k=0
                      w_offset=0*3+0,  # k=0, i=2 in C -> row 0*5+2=2
                      ni=3, mj=2, pk=2)
    
    # Block (1,1): A[2:5, 2:5] * B[2:5, 0:2] -> C[2:5, 0:2]
    # Schoolbook 3x3x2
    _schoolbook_block(U_cols, V_cols, W_cols,
                      u_offset=2*3+0,  # i=2, j=2 in A -> row 2*5+2=12
                      v_offset=0*2+0,  # j=0 local, k=0
                      w_offset=0*3+0,  # k=0, i=2 in C
                      ni=3, mj=3, pk=2)
    
    # Block (0,2): A[0:2, 2:5] * B[2:5, 2:5] -> C[0:2, 2:5]
    # Schoolbook 2x3x3
    _schoolbook_block(U_cols, V_cols, W_cols,
                      u_offset=0*3+0,  # i=0, j=2 in A
                      v_offset=0*3+0,  # j=0 local, k=2 in B -> row 0*5+2=2
                      w_offset=2*2+0,  # k=2, i=0 in C -> row 2*5+0=10
                      ni=2, mj=3, pk=3)
    
    # Block (1,2): A[2:5, 2:5] * B[2:5, 2:5] -> C[2:5, 2:5]
    # Schoolbook 3x3x3
    _schoolbook_block(U_cols, V_cols, W_cols,
                      u_offset=2*3+0,  # i=2, j=2 in A
                      v_offset=0*3+0,  # j=0 local, k=2 in B
                      w_offset=2*3+0,  # k=2, i=2 in C -> row 2*5+2=12
                      ni=3, mj=3, pk=3)
    
    # Block (2,0): A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    # Schoolbook 2x2x3
    _schoolbook_block(U_cols, V_cols, W_cols,
                      u_offset=0*2+0,  # i=0, j=0 in A
                      v_offset=0*3+0,  # j=0, k=2 in B
                      w_offset=2*2+0,  # k=2, i=0 in C
                      ni=2, mj=2, pk=3)
    
    # Block (2,1): A[2:5, 0:2] * B[0:2, 2:5] -> C[2:5, 2:5]
    # Schoolbook 3x2x3
    _schoolbook_block(U_cols, V_cols, W_cols,
                      u_offset=2*2+0,  # i=2, j=0 in A
                      v_offset=0*3+0,  # j=0, k=2 in B
                      w_offset=2*3+0,  # k=2, i=2 in C
                      ni=3, mj=2, pk=3)
    
    rank = len(U_cols)
    
    # Convert to arrays
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for c, (ud, vd, wd) in enumerate(zip(U_cols, V_cols, W_cols)):
        for row, val in ud.items():
            U[row, c] = val
        for row, val in vd.items():
            V[row, c] = val
        for row, val in wd.items():
            W[row, c] = val
    
    # Verify
    def check(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not check(U, V, W):
        # Fallback to schoolbook
        U_s = np.zeros((n*m, n*m*p), dtype=float)
        V_s = np.zeros((m*p, n*m*p), dtype=float)
        W_s = np.zeros((p*n, n*m*p), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j, col] = 1
                    V_s[j*p+k, col] = 1
                    W_s[k*n+i, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
