"""Matrix multiplication 3x3x4 rank-minimal exact tensor decomposition."""
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix
    by a 3x4 matrix using randomized linear algebra over GF(2).
    
    Strategy: Work over GF(2) where the equations become linear. Fix V and W 
    randomly over GF(2), then solve for U over GF(2). Since solutions over GF(2)
    have entries in {0,1}, they are already integers and lift exactly.
    
    For rank r, fix V (12xr) and W (12xr) over GF(2). For each i in {0,1,2},
    solve the 12x(3r) linear system over GF(2) for U[i*3:(i+1)*3, :].
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Try ranks from 20 upward
    for rank in [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]:
        np.random.seed(42 + rank)
        
        # Random initialization for V and W over GF(2)
        V = np.random.randint(0, 2, size=(m * p, rank))  # 12 x rank, entries in {0,1}
        W = np.random.randint(0, 2, size=(p * n, rank))  # 12 x rank, entries in {0,1}
        
        # Solve for U over GF(2)
        U = np.zeros((n * m, rank), dtype=np.int64)
        
        success = True
        for i in range(n):
            # Build the 12 x (m*rank) system over GF(2)
            # Equations: for each (j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 (mod 2)
            # Unknowns: U[i*m+j, r] for j in {0,1,2}, r in {0,...,rank-1}
            
            num_eqs = m * p  # 12
            num_unknowns = m * rank  # 3*rank
            
            A_mat = np.zeros((num_eqs, num_unknowns), dtype=np.int64)
            b_vec = np.ones(num_eqs, dtype=np.int64)
            
            eq_idx = 0
            for j in range(m):
                for k in range(p):
                    for jj in range(m):
                        for r in range(rank):
                            # A_mat[eq_idx, jj*rank + r] = V[j*p+k, r] * W[k*n+i, r] mod 2
                            A_mat[eq_idx, jj * rank + r] = (V[j * p + k, r] * W[k * n + i, r]) % 2
                    eq_idx += 1
            
            # Solve over GF(2) using Gaussian elimination
            # Augmented matrix [A | b]
            augmented = np.hstack([A_mat, b_vec.reshape(-1, 1)]) % 2
            
            # Gaussian elimination over GF(2)
            rows, cols = augmented.shape
            pivot_cols = []
            current_row = 0
            for col in range(cols - 1):  # Don't pivot on last column
                # Find pivot
                pivot = -1
                for r in range(current_row, rows):
                    if augmented[r, col] % 2 == 1:
                        pivot = r
                        break
                if pivot == -1:
                    continue
                # Swap rows
                augmented[[current_row, pivot]] = augmented[[pivot, current_row]]
                # Eliminate
                for r in range(rows):
                    if r != current_row and augmented[r, col] % 2 == 1:
                        augmented[r] = (augmented[r] + augmented[current_row]) % 2
                pivot_cols.append(col)
                current_row += 1
                if current_row == rows:
                    break
            
            # Check for consistency: no row with all zeros in A part but 1 in b part
            consistent = True
            for r in range(current_row, rows):
                if np.any(augmented[r, :cols-1] % 2 == 1):
                    consistent = False
                    break
                elif augmented[r, cols-1] % 2 == 1:
                    consistent = False
                    break
            
            if not consistent:
                success = False
                break
            
            # Extract solution (set free variables to 0 for simplicity)
            U_block = np.zeros(num_unknowns, dtype=np.int64)
            for r in range(current_row):
                col = pivot_cols[r]
                U_block[col] = augmented[r, cols-1] % 2
            
            U[i * m:(i + 1) * m, :] = U_block.reshape(m, rank)
        
        if not success:
            continue
        
        # Verify the full decomposition over integers (values are already 0 or 1)
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(int(U[i * m + j, r]) * int(V[j * p + k, r]) * int(W[k * n + i, r]) for r in range(rank))
                    if val != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Convert to list of lists (values are already integers)
            U_list = [[int(val) for val in row] for row in U.tolist()]
            V_list = [[int(val) for val in row] for row in V.tolist()]
            W_list = [[int(val) for val in row] for row in W.tolist()]
            
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook if no low-rank found
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
