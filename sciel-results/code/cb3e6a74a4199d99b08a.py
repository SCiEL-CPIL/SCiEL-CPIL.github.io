"""Program to find a rank-minimal exact tensor decomposition for 3x5 * 5x5 matrix multiplication."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]
n, m, p = DIMENSIONS

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    Strategy: 
    1. Use Strassen's 7-rank decomposition for the 2x2x2 sub-block (indices 0,1 in all dims).
    2. Use schoolbook decomposition for the remaining 67 terms.
    Total rank: 7 + 67 = 74.
    """
    # Define the sub-block indices
    block_idx = [0, 1]
    rest_idx = [2] # For n=3, only index 2 is outside 0,1
    
    # We will construct the columns
    U_cols = []
    V_cols = []
    W_cols = []
    
    # 1. Strassen terms for the 2x2x2 block
    # Indices in the block are i,j,k in {0,1}
    # We map these to global indices:
    # i_global = i, j_global = j, k_global = k
    
    # Strassen formulas for C = A * B where A,B are 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each term c corresponds to a column in U, V, W.
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # Helper to add a term
    def add_term(u_dict, v_dict, w_dict):
        U_cols.append(u_dict)
        V_cols.append(v_dict)
        W_cols.append(w_dict)
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    # U: A00=1, A11=1
    # V: B00=1, B11=1
    # W: C00=1, C11=1
    add_term(
        {0*m+0: 1, 1*m+1: 1}, # A00, A11
        {0*p+0: 1, 1*p+1: 1}, # B00, B11
        {0*n+0: 1, 1*n+1: 1}  # C00, C11 (W[k*n+i]) -> W[0,0]=1, W[1,1]=1
    )
    
    # Term 2: M2 = (A10+A11)B00
    # U: A10=1, A11=1
    # V: B00=1
    # W: C10=1
    add_term(
        {1*m+0: 1, 1*m+1: 1},
        {0*p+0: 1},
        {0*n+1: 1} # C10 -> W[0,1]
    )
    
    # Term 3: M3 = A00(B10-B11)
    # U: A00=1
    # V: B10=1, B11=-1
    # W: C01=1
    add_term(
        {0*m+0: 1},
        {1*p+0: 1, 1*p+1: -1},
        {1*n+0: 1} # C01 -> W[1,0]
    )
    
    # Term 4: M4 = A11(B01-B00)
    # U: A11=1
    # V: B01=1, B00=-1
    # W: C00=1, C10=1
    add_term(
        {1*m+1: 1},
        {0*p+1: 1, 0*p+0: -1},
        {0*n+0: 1, 0*n+1: 1} # C00, C10
    )
    
    # Term 5: M5 = (A00+A01)B11
    # U: A00=1, A01=1
    # V: B11=1
    # W: C00=-1, C01=1
    add_term(
        {0*m+0: 1, 0*m+1: 1},
        {1*p+1: 1},
        {0*n+0: -1, 1*n+0: 1} # C00, C01
    )
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    # U: A10=1, A00=-1
    # V: B00=1, B01=1
    # W: C11=1
    add_term(
        {1*m+0: 1, 0*m+0: -1},
        {0*p+0: 1, 0*p+1: 1},
        {1*n+1: 1} # C11
    )
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    # U: A01=1, A11=-1
    # V: B10=1, B11=1
    # W: C00=1
    add_term(
        {0*m+1: 1, 1*m+1: -1},
        {1*p+0: 1, 1*p+1: 1},
        {0*n+0: 1} # C00
    )
    
    # 2. Schoolbook terms for the complement
    # All (i,j,k) such that NOT (i in {0,1} AND j in {0,1} AND k in {0,1})
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this term is in the Strassen block
                if i in block_idx and j in block_idx and k in block_idx:
                    continue # Skip, already handled by Strassen
                
                # Add schoolbook term
                u_dict = {i*m + j: 1}
                v_dict = {j*p + k: 1}
                w_dict = {k*n + i: 1}
                add_term(u_dict, v_dict, w_dict)
    
    # Convert lists of dicts to matrices
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for idx, val in U_cols[c].items():
            U[idx][c] = val
        for idx, val in V_cols[c].items():
            V[idx][c] = val
        for idx, val in W_cols[c].items():
            W[idx][c] = val
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    return construct()
# EVOLVE-BLOCK-END
