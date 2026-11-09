"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # Block decomposition: split 5x5x5 into blocks
    # We use a 2+2+1 split on each dimension
    # Blocks: (2,2,2), (2,2,1), (2,1,2), (2,1,1), (1,2,2), (1,2,1), (1,1,2), (1,1,1)
    
    # For 2x2x2 blocks, we use Strassen (rank 7)
    # For 2x2x1, 2x1x2, 2x1x1, 1x2x2, 1x2x1, 1x1x2, 1x1x1 blocks, we use schoolbook
    
    # Strassen formulas for 2x2x2:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(f-h)
    # M4 = (d)(g-c)
    # M5 = (a+c)(h)
    # M6 = (b+d)(g)
    # M7 = (a-b)(e+f)
    #
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M4
    # C21 = M2 - M1 + M7
    # C22 = M1 - M2 + M3 + M6
    
    # Let's define the Strassen decomposition for a 2x2x2 block
    # A indices: a=00, b=01, c=10, d=11 (row, col)
    # B indices: e=00, f=01, g=10, h=11 (row, col)
    # C indices: C11=00, C12=01, C21=10, C22=11 (row, col)
    
    # Strassen coefficients:
    # For each of the 7 multiplications, we have:
    # U (for A), V (for B), W (for C)
    
    strassen_2x2x2 = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # M1 = (a+d)(e+h)
        ([1,0,0,1], [1,0,0,1], [1,0,0,-1,0,0,0,1]),  # Wait, C is 2x2=4 elements
        # Let me reconsider. C has 4 elements: C11, C12, C21, C22
        # Flat index: C11=0, C12=1, C21=2, C22=3
    ]
    
    # Let me redo this more carefully
    # For a 2x2x2 block:
    # A is 2x2: indices (0,0), (0,1), (1,0), (1,1) -> flat: 0,1,2,3
    # B is 2x2: indices (0,0), (0,1), (1,0), (1,1) -> flat: 0,1,2,3
    # C is 2x2: indices (0,0), (0,1), (1,0), (1,1) -> flat: 0,1,2,3
    
    # Strassen:
    # M1 = (a+d)(e+h)
    #   U1 = [1,0,0,1] (a and d)
    #   V1 = [1,0,0,1] (e and h)
    #   W1: M1 contributes to C11, C21, C22
    #   C11 = M1 - M4 + M5 - M7
    #   C12 = M3 + M4
    #   C21 = M2 - M1 + M7
    #   C22 = M1 - M2 + M3 + M6
    #   So M1: C11=+1, C12=0, C21=-1, C22=+1
    #   W1 = [1, 0, -1, 1]
    
    # M2 = (c+d)(e)
    #   U2 = [0,0,1,1]
    #   V2 = [1,0,0,0]
    #   M2: C21=-1, C22=-1
    #   W2 = [0, 0, -1, -1]
    
    # M3 = (a)(f-h)
    #   U3 = [1,0,0,0]
    #   V3 = [0,1,0,-1]
    #   M3: C12=+1, C22=+1
    #   W3 = [0, 1, 0, 1]
    
    # M4 = (d)(g-c)
    #   U4 = [0,0,0,1]
    #   V4 = [0,0,1,-1]
    #   M4: C11=-1, C12=+1
    #   W4 = [-1, 1, 0, 0]
    
    # M5 = (a+c)(h)
    #   U5 = [1,0,1,0]
    #   V5 = [0,0,0,1]
    #   M5: C11=+1
    #   W5 = [1, 0, 0, 0]
    
    # M6 = (b+d)(g)
    #   U6 = [0,1,0,1]
    #   V6 = [0,0,1,0]
    #   M6: C22=+1
    #   W6 = [0, 0, 0, 1]
    
    # M7 = (a-b)(e+f)
    #   U7 = [1,-1,0,0]
    #   V7 = [1,1,0,0]
    #   M7: C11=-1, C21=+1
    #   W7 = [-1, 0, 1, 0]
    
    strassen_terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,-1,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,-1,-1]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
        ([0,0,0,1], [0,0,1,-1], [-1,1,0,0]),
        ([1,0,1,0], [0,0,0,1], [1,0,0,0]),
        ([0,1,0,1], [0,0,1,0], [0,0,0,1]),
        ([1,-1,0,0], [1,1,0,0], [-1,0,1,0]),
    ]
    
    # Now, let's define the block decomposition
    # We split each dimension (5) into 2+2+1
    # Block sizes: for i in [0,1,2], j in [0,1,2], k in [0,1,2]
    # Block (i,j,k) has size (2,2,2) if i,j,k all < 2
    # Otherwise, it's a smaller block
    
    # Let's define the block structure
    # Dimension split: [2, 2, 1]
    splits = [2, 2, 1]
    
    # For each block, determine if it's a 2x2x2 block (use Strassen) or schoolbook
    # Block (bi, bj, bk) where bi, bj, bk in {0,1,2}
    # Size: (splits[bi], splits[bj], splits[bk])
    
    # Let's build the decomposition
    U_list = []
    V_list = []
    W_list = []
    
    # For each block
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                si = splits[bi]
                sj = splits[bj]
                sk = splits[bk]
                
                # Offset in each dimension
                off_i = sum(splits[:bi])
                off_j = sum(splits[:bj])
                off_k = sum(splits[:bk])
                
                if si == 2 and sj == 2 and sk == 2:
                    # Use Strassen for this 2x2x2 block
                    for (u_coeffs, v_coeffs, w_coeffs) in strassen_terms:
                        u = [Fraction(0)] * (n*m)
                        v = [Fraction(0)] * (m*p)
                        w = [Fraction(0)] * (p*n)
                        
                        for idx, val in enumerate(u_coeffs):
                            if val != 0:
                                local_i, local_j = divmod(idx, 2)
                                global_i = off_i + local_i
                                global_j = off_j + local_j
                                u[global_i * m + global_j] = Fraction(val)
                        
                        for idx, val in enumerate(v_coeffs):
                            if val != 0:
                                local_j, local_k = divmod(idx, 2)
                                global_j = off_j + local_j
                                global_k = off_k + local_k
                                v[global_j * p + global_k] = Fraction(val)
                        
                        for idx, val in enumerate(w_coeffs):
                            if val != 0:
                                local_k, local_i = divmod(idx, 2)
                                global_k = off_k + local_k
                                global_i = off_i + local_i
                                w[global_k * n + global_i] = Fraction(val)
                        
                        U_list.append(u)
                        V_list.append(v)
                        W_list.append(w)
                else:
                    # Use schoolbook for smaller blocks
                    for local_i in range(si):
                        for local_j in range(sj):
                            for local_k in range(sk):
                                u = [Fraction(0)] * (n*m)
                                v = [Fraction(0)] * (m*p)
                                w = [Fraction(0)] * (p*n)
                                
                                global_i = off_i + local_i
                                global_j = off_j + local_j
                                global_k = off_k + local_k
                                
                                u[global_i * m + global_j] = Fraction(1)
                                v[global_j * p + global_k] = Fraction(1)
                                w[global_k * n + global_i] = Fraction(1)
                                
                                U_list.append(u)
                                V_list.append(v)
                                W_list.append(w)
    
    rank = len(U_list)
    
    # Convert to lists of lists
    U = [list(row) for row in U_list]
    V = [list(row) for row in V_list]
    W = [list(row) for row in W_list]
    
    # Verify the decomposition
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Should be 1 for all i, j, k
    
    # Quick verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if total != 1:
                    # Fallback to schoolbook
                    U = [[0] * (n*m*p) for _ in range(n*m)]
                    V = [[0] * (n*m*p) for _ in range(m*p)]
                    W = [[0] * (n*m*p) for _ in range(p*n)]
                    column = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                U[i*m+j][column] = 1
                                V[j*p+k][column] = 1
                                W[k*n+i][column] = 1
                                column += 1
                    return {"U": U, "V": V, "W": W}
    
    # Check time
    if time.time() > deadline:
        # Return schoolbook as fallback
        U = [[0] * (n*m*p) for _ in range(n*m)]
        V = [[0] * (n*m*p) for _ in range(m*p)]
        W = [[0] * (n*m*p) for _ in range(p*n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][column] = 1
                    V[j*p+k][column] = 1
                    W[k*n+i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
