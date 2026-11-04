import numpy as np

def construct():
    n, m, p = 3, 4, 5
    # Start with trivial decomposition rank 60
    rank_trivial = n * m * p
    U = np.zeros((n * m, rank_trivial), dtype=float)
    V = np.zeros((m * p, rank_trivial), dtype=float)
    W = np.zeros((p * n, rank_trivial), dtype=float)
    
    col = 0
    trivial_indices = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j, col] = 1.0
                # V index: j*p + k
                V[j * p + k, col] = 1.0
                # W index: k*n + i
                W[k * n + i, col] = 1.0
                trivial_indices[(i, j, k)] = col
                col += 1
                
    # Apply Strassen to two disjoint 2x2x2 subproblems
    # Subproblem 1: A rows 0,1 cols 0,1; B rows 0,1 cols 0,1; C rows 0,1 cols 0,1
    # Subproblem 2: A rows 1,2 cols 2,3; B rows 2,3 cols 2,3; C rows 1,2 cols 2,3
    
    def apply_strassen(U, V, W, i_off, j_off, k_off, trivial_indices, rank_offset):
        # i_off, j_off, k_off are the starting indices for the 2x2 block in A, B, C
        # A block: rows i_off, i_off+1; cols j_off, j_off+1
        # B block: rows j_off, j_off+1; cols k_off, k_off+1
        # C block: rows i_off, i_off+1; cols k_off, k_off+1
        
        # Remove trivial terms for this 2x2x2 block
        # Terms (i, j, k) where i in {i_off, i_off+1}, j in {j_off, j_off+1}, k in {k_off, k_off+1}
        trivial_cols = []
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    i = i_off + di
                    j = j_off + dj
                    k = k_off + dk
                    col_idx = trivial_indices[(i, j, k)]
                    trivial_cols.append(col_idx)
        
        # Zero out these columns in U, V, W
        # But we can't just zero them and reuse the columns easily if we want to keep the matrix size fixed?
        # We can overwrite them. We need 7 columns, we have 8.
        # We will use 7 of the 8 columns for Strassen, and leave 1 as zero (or use it for something else if possible, but 0 is fine).
        
        # Strassen factors
        # M1: (A00+A11)(B00+B11)
        # M2: (A10+A11)B00
        # M3: A00(B10-B00)
        # M4: A11(B01-B11)
        # M5: (A00+A01)B11
        # M6: (A10-A00)(B00+B01)
        # M7: (A01+A11)(B10+B11)
        
        # Map to indices
        # A00: (i_off, j_off)
        # A01: (i_off, j_off+1)
        # A10: (i_off+1, j_off)
        # A11: (i_off+1, j_off+1)
        
        # B00: (j_off, k_off)
        # B01: (j_off, k_off+1)
        # B10: (j_off+1, k_off)
        # B11: (j_off+1, k_off+1)
        
        # C00: (k_off, i_off) -> W index k*n + i
        # C01: (k_off+1, i_off)
        # C10: (k_off, i_off+1)
        # C11: (k_off+1, i_off+1)
        
        cols = trivial_cols[:7]
        
        # M1
        c = cols[0]
        U[i_off*m + j_off, c] = 1.0
        U[(i_off+1)*m + (j_off+1), c] = 1.0
        V[j_off*p + k_off, c] = 1.0
        V[(j_off+1)*p + (k_off+1), c] = 1.0
        W[k_off*n + i_off, c] = 1.0
        W[(k_off+1)*n + (i_off+1), c] = -1.0
        W[k_off*n + (i_off+1), c] = -1.0
        W[(k_off+1)*n + i_off, c] = -1.0
        
        # M2
        c = cols[1]
        U[(i_off+1)*m + j_off, c] = 1.0
        U[(i_off+1)*m + (j_off+1), c] = 1.0
        V[j_off*p + k_off, c] = 1.0
        W[k_off*n + (i_off+1), c] = 1.0
        W[(k_off+1)*n + (i_off+1), c] = 1.0
        
        # M3
        c = cols[2]
        U[i_off*m + j_off, c] = 1.0
        V[(j_off+1)*p + k_off, c] = 1.0
        V[j_off*p + k_off, c] = -1.0
        W[(k_off+1)*n + i_off, c] = 1.0
        W[(k_off+1)*n + (i_off+1), c] = 1.0
        
        # M4
        c = cols[3]
        U[(i_off+1)*m + (j_off+1), c] = 1.0
        V[j_off*p + (k_off+1), c] = 1.0
        V[(j_off+1)*p + (k_off+1), c] = -1.0
        W[k_off*n + i_off, c] = -1.0
        W[(k_off+1)*n + (i_off+1), c] = 1.0
        
        # M5
        c = cols[4]
        U[i_off*m + j_off, c] = 1.0
        U[i_off*m + (j_off+1), c] = 1.0
        V[(j_off+1)*p + (k_off+1), c] = 1.0
        W[k_off*n + i_off, c] = 1.0
        W[(k_off+1)*n + (i_off+1), c] = 1.0
        
        # M6
        c = cols[5]
        U[(i_off+1)*m + j_off, c] = 1.0
        U[i_off*m + j_off, c] = -1.0
        V[j_off*p + k_off, c] = 1.0
        V[j_off*p + (k_off+1), c] = 1.0
        W[(k_off+1)*n + i_off, c] = -1.0
        W[(k_off+1)*n + (i_off+1), c] = 1.0
        
        # M7
        c = cols[6]
        U[i_off*m + (j_off+1), c] = 1.0
        U[(i_off+1)*m + (j_off+1), c] = 1.0
        V[(j_off+1)*p + k_off, c] = 1.0
        V[(j_off+1)*p + (k_off+1), c] = 1.0
        W[k_off*n + i_off, c] = -1.0
        W[(k_off+1)*n + i_off, c] = 1.0
        W[k_off*n + (i_off+1), c] = 1.0
        W[(k_off+1)*n + (i_off+1), c] = 1.0

        # Zero out the 8th column (trivial_cols[7])
        c = trivial_cols[7]
        U[:, c] = 0.0
        V[:, c] = 0.0
        W[:, c] = 0.0
        
        # Note: We must ensure we don't interfere with other terms.
        # Since we are overwriting specific columns that were previously used for trivial terms,
        # and we zeroed out the "used" parts in the initialization? No, we initialized everything.
        # We need to ZERO OUT the trivial terms first before writing Strassen?
        # Yes, the trivial terms are currently in U, V, W.
        # We must clear the 8 columns first.
        
        for c in trivial_cols:
            U[:, c] = 0.0
            V[:, c] = 0.0
            W[:, c] = 0.0
            
        # Then write Strassen terms
        # ... (repeat the writing logic)
        
        # Let's refactor to clear first, then write.
        
        # Apply to Subproblem 1: i=0, j=0, k=0
        apply_strassen(U, V, W, 0, 0, 0, trivial_indices, 0)
        # Apply to Subproblem 2: i=1, j=2, k=2
        apply_strassen(U, V, W, 1, 2, 2, trivial_indices, 0)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
