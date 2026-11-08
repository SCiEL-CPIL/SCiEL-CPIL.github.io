import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # We will build U, V, W matrices
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (27, rank)
    # W shape: (p*n, rank) = (18, rank)
    
    # Rank calculation
    # 4 blocks of 2x3x2 (rank 11 each)
    # 1 block of 2x3x1 (rank 6)
    rank = 4 * 11 + 6
    rank = 50
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col_offset = 0
    
    # Helper to add a 2x2x2 Strassen decomposition
    # A is 2x2 (indices in U), B is 2x2 (indices in V), C is 2x2 (indices in W)
    # We need to map the global indices
    # A indices: i in {0,1}, j in {0,1} -> U[i*m + j]
    # B indices: j in {0,1}, k in {0,1} -> V[j*p + k]
    # C indices: k in {0,1}, i in {0,1} -> W[k*n + i]
    
    def add_strassen_222(a_rows, b_cols, c_cols, offset):
        # a_rows: list of 2 global row indices for A (i*m+j)
        # b_cols: list of 2 global col indices for B (j*p+k)
        # c_cols: list of 2 global col indices for C (k*n+i)
        # This function adds 7 columns to U, V, W starting at 'offset'
        
        # Strassen 7 multiplications
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A10)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # We need to define the U, V, W entries for each of the 7 terms.
        # Each term is a product of a linear combo of A elements and a linear combo of B elements.
        # This results in a rank-1 update to the tensor.
        
        # U entries: coefficients for A elements
        # V entries: coefficients for B elements
        # W entries: coefficients for C elements
        
        # Term 1: (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00+=1, C01+=1, C10+=1, C11+=1? No.
        # Strassen formula:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We add 7 columns.
        # Col 0: M1
        U[a_rows[0]][offset+0] += 1
        U[a_rows[3]][offset+0] += 1  # A11 is row 1, col 1 -> index 1*m+1?
        # Wait, a_rows are global indices.
        # A00: i=0, j=0 -> idx 0
        # A11: i=1, j=1 -> idx 1*m+1 = 4
        # So a_rows should be the global indices.
        
        # Let's refine the helper to take global indices directly.
        
        # M1
        U[a_rows[0]][offset+0] += 1
        U[a_rows[3]][offset+0] += 1
        V[b_cols[0]][offset+0] += 1
        V[b_cols[3]][offset+0] += 1
        # W updates for M1:
        # C00 += 1, C11 += 1? No.
        # C00 = M1 + ...
        # C11 = M1 + ...
        # So W for C00 and C11 get +1.
        # C00 index: k=0, i=0 -> 0*n+0 = 0
        # C11 index: k=1, i=1 -> 1*n+1 = 3
        # But c_cols are the global column indices for the block.
        # The block corresponds to specific k values.
        # Let's say the block uses k indices k0, k1.
        # Then C00 is W[k0*n + i0], C01 is W[k1*n + i0], etc.
        # i0, i1 are the row indices of A.
        
        # This is getting complex. I will implement the 50-rank decomposition explicitly.
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
