import time
import numpy as np

def construct():
    """
    Return a rank-96 exact tensor decomposition for 4x5 * 5x5 matrix multiplication.
    Uses Strassen's algorithm on four disjoint 2x2x2 sub-problems.
    """
    n, m, p = 4, 5, 5
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20
    
    # Start with the trivial rank-100 decomposition
    rank = n * m * p
    U = np.zeros((nm, rank), dtype=np.int64)
    V = np.zeros((mp, rank), dtype=np.int64)
    W = np.zeros((pn, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Identify 4 disjoint 2x2x2 sub-problems to replace with Strassen (rank 7 vs 8)
    # Sub-problem 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Sub-problem 2: i in {0,1}, j in {2,3}, k in {2,3}
    # Sub-problem 3: i in {2,3}, j in {0,1}, k in {0,1}
    # Sub-problem 4: i in {2,3}, j in {2,3}, k in {2,3}
    
    # We need to remove the 8 terms for each sub-problem and add 7 Strassen terms.
    # Total rank change: 4 * (7 - 8) = -4. New rank 96.
    
    # Let's construct the Strassen decomposition for a generic 2x2x2 block.
    # Indices: i' in {0,1}, j' in {0,1}, k' in {0,1}.
    # A block A' (2x2), B' (2x2), C' (2x2).
    # Strassen 7 products:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will manually construct the U, V, W columns for these 4 blocks.
    # It is safer to build the full rank-96 matrix from scratch to avoid index confusion.
    
    R = 96
    U_new = np.zeros((nm, R), dtype=np.int64)
    V_new = np.zeros((mp, R), dtype=np.int64)
    W_new = np.zeros((pn, R), dtype=np.int64)
    
    # Helper to add a Strassen block
    def add_strassen_block(i_off, j_off, k_off, start_col):
        # i_off, j_off, k_off are the offsets for the 2x2 block
        # i in {i_off, i_off+1}, j in {j_off, j_off+1}, k in {k_off, k_off+1}
        
        # Strassen factors for U (A), V (B), W (C)
        # U indices: (i, j) -> i*m + j
        # V indices: (j, k) -> j*p + k
        # W indices: (k, i) -> k*n + i
        
        # M1: (A00+A11)(B00+B11)
        # A part: A00, A11. B part: B00, B11. C part: C00, C11.
        # U: (i0,j0)=1, (i1,j1)=1. V: (j0,k0)=1, (j1,k1)=1. W: (k0,i0)=1, (k1,i1)=1.
        # This is not a single rank-1 term. Strassen uses 7 rank-1 terms.
        
        # Term 1: (A00+A11) * (B00+B11) * (C00+C11 - C01 - C10)? No.
        # The decomposition is:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We add 7 columns.
        
        # Col 1: M1
        # U: A00=1, A11=1. Others 0.
        # V: B00=1, B11=1. Others 0.
        # W: C00=1, C11=1. Others 0.
        U_new[i_off*m + j_off, start_col] = 1
        U_new[(i_off+1)*m + (j_off+1), start_col] = 1
        V_new[j_off*p + k_off, start_col] = 1
        V_new[(j_off+1)*p + (k_off+1), start_col] = 1
        W_new[k_off*n + i_off, start_col] = 1
        W_new[(k_off+1)*n + (i_off+1), start_col] = 1
        
        # Col 2: M2 = (A10+A11) * B00 * (C10 - C11)
        # U: A10=1, A11=1.
        # V: B00=1.
        # W: C10=1, C11=-1.
        U_new[(i_off+1)*m + j_off, start_col+1] = 1
        U_new[(i_off+1)*m + (j_off+1), start_col+1] = 1
        V_new[j_off*p + k_off, start_col+1] = 1
        W_new[(k_off+1)*n + (i_off+1), start_col+1] = 1
        W_new[(k_off+1)*n + i_off, start_col+1] = -1
        
        # Col 3: M3 = A00 * (B01-B11) * (C01 + C11)
        # U: A00=1.
        # V: B01=1, B11=-1.
        # W: C01=1, C11=1.
        U_new[i_off*m + j_off, start_col+2] = 1
        V_new[j_off*p + (k_off+1), start_col+2] = 1
        V_new[(j_off+1)*p + (k_off+1), start_col+2] = -1
        W_new[(k_off+1)*n + i_off, start_col+2] = 1
        W_new[(k_off+1)*n + (i_off+1), start_col+2] = 1
        
        # Col 4: M4 = A11 * (B10-B00) * (C00 + C10)
        # U: A11=1.
        # V: B10=1, B00=-1.
        # W: C00=1, C10=1.
        U_new[(i_off+1)*m + (j_off+1), start_col+3] = 1
        V_new[(j_off+1)*p + k_off, start_col+3] = 1
        V_new[j_off*p + k_off, start_col+3] = -1
        W_new[k_off*n + i_off, start_col+3] = 1
        W_new[(k_off+1)*n + (i_off+1), start_col+3] = 1
        
        # Col 5: M5 = (A00+A01) * B11 * (C00 - C01)
        # U: A00=1, A01=1.
        # V: B11=1.
        # W: C00=1, C01=-1.
        U_new[i_off*m + j_off, start_col+4] = 1
        U_new[i_off*m + (j_off+1), start_col+4] = 1
        V_new[(j_off+1)*p + (k_off+1), start_col+4] = 1
        W_new[k_off*n + i_off, start_col+4] = 1
        W_new[(k_off+1)*n + i_off, start_col+4] = -1
        
        # Col 6: M6 = (A10-A00) * (B00+B01) * (C11 - C10)
        # U: A10=1, A00=-1.
        # V: B00=1, B01=1.
        # W: C11=1, C10=-1.
        U_new[(i_off+1)*m + j_off, start_col+5] = 1
        U_new[i_off*m + j_off, start_col+5] = -1
        V_new[j_off*p + k_off, start_col+5] = 1
        V_new[j_off*p + (k_off+1), start_col+5] = 1
        W_new[(k_off+1)*n + (i_off+1), start_col+5] = 1
        W_new[(k_off+1)*n + i_off, start_col+5] = -1
        
        # Col 7: M7 = (A01-A11) * (B10+B11) * (C00 + C11)
        # U: A01=1, A11=-1.
        # V: B10=1, B11=1.
        # W: C00=1, C11=1.
        U_new[i_off*m + (j_off+1), start_col+6] = 1
        U_new[(i_off+1)*m + (j_off+1), start_col+6] = -1
        V_new[(j_off+1)*p + k_off, start_col+6] = 1
        V_new[(j_off+1)*p + (k_off+1), start_col+6] = 1
        W_new[k_off*n + i_off, start_col+6] = 1
        W_new[(k_off+1)*n + (i_off+1), start_col+6] = 1

    # We need to fill the rest of the matrix with the trivial decomposition for the non-block parts.
    # This is complex to manage with offsets.
    # Instead, we can just use the trivial decomposition for everything, and then 
    # "patch" the 4 blocks by replacing their 8 columns with 7 Strassen columns.
    
    # Let's revert to the patching strategy on the original U, V, W.
    # We have U, V, W of size (nm, 100), (mp, 100), (pn, 100).
    # We will create new arrays of size 96.
    
    # Mapping of columns:
    # For each of the 4 blocks, we replace 8 columns with 7.
    # The other 100 - 4*8 = 68 columns remain as is.
    
    # Let's identify the column indices for the 4 blocks in the trivial decomposition.
    # Trivial col index = i * m * p + j * p + k.
    
    block_configs = [
        (0, 0, 0), # i0=0, j0=0, k0=0
        (0, 2, 2), # i0=0, j0=2, k0=2
        (2, 0, 0), # i0=2, j0=0, k0=0
        (2, 2, 2)  # i0=2, j0=2, k0=2
    ]
    
    # We will build the new matrices by iterating through the 96 columns.
    # 4 blocks * 7 cols = 28 cols.
    # 100 - 32 = 68 trivial cols.
    # Total 96.
    
    U_final = np.zeros((nm, 96), dtype=np.int64)
    V_final = np.zeros((mp, 96), dtype=np.int64)
    W_final = np.zeros((pn, 96), dtype=np.int64)
    
    col_idx = 0
    
    # Add Strassen blocks
    for i_off, j_off, k_off in block_configs:
        # Call the helper to fill 7 columns starting at col_idx
        # The helper above assumes it writes to U_new, etc. 
        # Let's refactor the helper to take the target arrays and start col.
        
        # M1
        U_final[i_off*m + j_off, col_idx] = 1
        U_final[(i_off+1)*m + (j_off+1), col_idx] = 1
        V_final[j_off*p + k_off, col_idx] = 1
        V_final[(j_off+1)*p + (k_off+1), col_idx] = 1
        W_final[k_off*n + i_off, col_idx] = 1
        W_final[(k_off+1)*n + (i_off+1), col_idx] = 1
        col_idx += 1
        
        # M2
        U_final[(i_off+1)*m + j_off, col_idx] = 1
        U_final[(i_off+1)*m + (j_off+1), col_idx] = 1
        V_final[j_off*p + k_off, col_idx] = 1
        W_final[(k_off+1)*n + (i_off+1), col_idx] = 1
        W_final[(k_off+1)*n + i_off, col_idx] = -1
        col_idx += 1
        
        # M3
        U_final[i_off*m + j_off, col_idx] = 1
        V_final[j_off*p + (k_off+1), col_idx] = 1
        V_final[(j_off+1)*p + (k_off+1), col_idx] = -1
        W_final[(k_off+1)*n + i_off, col_idx] = 1
        W_final[(k_off+1)*n + (i_off+1), col_idx] = 1
        col_idx += 1
        
        # M4
        U_final[(i_off+1)*m + (j_off+1), col_idx] = 1
        V_final[(j_off+1)*p + k_off, col_idx] = 1
        V_final[j_off*p + k_off, col_idx] = -1
        W_final[k_off*n + i_off, col_idx] = 1
        W_final[(k_off+1)*n + (i_off+1), col_idx] = 1
        col_idx += 1
        
        # M5
        U_final[i_off*m + j_off, col_idx] = 1
        U_final[i_off*m + (j_off+1), col_idx] = 1
        V_final[(j_off+1)*p + (k_off+1), col_idx] = 1
        W_final[k_off*n + i_off, col_idx] = 1
        W_final[(k_off+1)*n + i_off, col_idx] = -1
        col_idx += 1
        
        # M6
        U_final[(i_off+1)*m + j_off, col_idx] = 1
        U_final[i_off*m + j_off, col_idx] = -1
        V_final[j_off*p + k_off, col_idx] = 1
        V_final[j_off*p + (k_off+1), col_idx] = 1
        W_final[(k_off+1)*n + (i_off+1), col_idx] = 1
        W_final[(k_off+1)*n + i_off, col_idx] = -1
        col_idx += 1
        
        # M7
        U_final[i_off*m + (j_off+1), col_idx] = 1
        U_final[(i_off+1)*m + (j_off+1), col_idx] = -1
        V_final[(j_off+1)*p + k_off, col_idx] = 1
        V_final[(j_off+1)*p + (k_off+1), col_idx] = 1
        W_final[k_off*n + i_off, col_idx] = 1
        W_final[(k_off+1)*n + (i_off+1), col_idx] = 1
        col_idx += 1
        
    # Add trivial columns for the rest
    # We need to skip the columns corresponding to the 4 blocks.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if (i,j,k) is in one of the blocks
                in_block = False
                for i_off, j_off, k_off in block_configs:
                    if i in [i_off, i_off+1] and j in [j_off, j_off+1] and k in [k_off, k_off+1]:
                        in_block = True
                        break
                if not in_block:
                    U_final[i * m + j, col_idx] = 1
                    V_final[j * p + k, col_idx] = 1
                    W_final[k * n + i, col_idx] = 1
                    col_idx += 1
                    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
