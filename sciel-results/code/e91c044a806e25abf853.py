"""Matrix multiplication 2x4x8 via Strassen block decomposition with exact rational rank reduction."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 8]

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Construct the 56-rank Strassen decomposition
    n_blocks_i = 1
    n_blocks_j = 2
    n_blocks_k = 4
    total_blocks = n_blocks_i * n_blocks_j * n_blocks_k
    rank = total_blocks * 7  # 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for bi in range(n_blocks_i):
        for bj in range(n_blocks_j):
            for bk in range(n_blocks_k):
                i0 = 2 * bi
                i1 = 2 * bi + 1
                j0 = 2 * bj
                j1 = 2 * bj + 1
                k0 = 2 * bk
                k1 = 2 * bk + 1
                
                u_a = i0 * m + j0
                u_b = i0 * m + j1
                u_c = i1 * m + j0
                u_d = i1 * m + j1
                
                v_e = j0 * p + k0
                v_f = j0 * p + k1
                v_g = j1 * p + k0
                v_h = j1 * p + k1
                
                w_c00 = k0 * n + i0
                w_c01 = k1 * n + i0
                w_c10 = k0 * n + i1
                w_c11 = k1 * n + i1
                
                # M1 = (a+d)(e+h)
                U[u_a][col] += 1; U[u_d][col] += 1
                V[v_e][col] += 1; V[v_h][col] += 1
                W[w_c00][col] += 1; W[w_c11][col] += 1
                col += 1
                
                # M2 = (c+d)e
                U[u_c][col] += 1; U[u_d][col] += 1
                V[v_e][col] += 1
                W[w_c10][col] += 1; W[w_c11][col] -= 1
                col += 1
                
                # M3 = a(f-h)
                U[u_a][col] += 1
                V[v_f][col] += 1; V[v_h][col] -= 1
                W[w_c01][col] += 1; W[w_c11][col] += 1
                col += 1
                
                # M4 = d(g-e)
                U[u_d][col] += 1
                V[v_g][col] += 1; V[v_e][col] -= 1
                W[w_c00][col] += 1; W[w_c10][col] += 1
                col += 1
                
                # M5 = (a+b)h
                U[u_a][col] += 1; U[u_b][col] += 1
                V[v_h][col] += 1
                W[w_c00][col] -= 1; W[w_c01][col] += 1
                col += 1
                
                # M6 = (c-a)(e+f)
                U[u_c][col] += 1; U[u_a][col] -= 1
                V[v_e][col] += 1; V[v_f][col] += 1
                W[w_c11][col] += 1
                col += 1
                
                # M7 = (b-d)(g+h)
                U[u_b][col] += 1; U[u_d][col] -= 1
                V[v_g][col] += 1; V[v_h][col] += 1
                W[w_c00][col] += 1
                col += 1
    
    # Convert to Fraction for exact arithmetic
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    # Exact rank reduction via Gaussian elimination on the tensor's mode-1 flattening
    # We treat each rank term r as a vector in the space of tensors.
    # To check linear dependence, we can flatten the tensor T = sum_r U_r V_r W_r.
    # However, a simpler approach for this specific structure is to check if the columns of U, V, W
    # can be combined.
    
    # The tensor T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want to find the minimal number of terms.
    # This is equivalent to finding the rank of the tensor.
    # Since the dimension is small, we can attempt to reduce the rank by checking
    # if any column is a linear combination of others.
    
    # Let's build a matrix M where each row corresponds to a unique (i,j,k) triplet
    # and each column corresponds to a rank term r.
    # M[i,j,k, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # The rank of the tensor is the rank of this matrix.
    
    num_triplets = n * m * p
    M = [[Fraction(0) for _ in range(rank)] for _ in range(num_triplets)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    M[idx][r] = U_frac[i*m+j][r] * V_frac[j*p+k][r] * W_frac[k*n+i][r]
                idx += 1
    
    # Perform Gaussian elimination to find the rank
    # We want to identify which columns are linearly independent.
    # We'll do row-reduced echelon form (RREF) on M.
    
    # Copy M for elimination
    M_elim = [row[:] for row in M]
    rows = len(M_elim)
    cols = len(M_elim[0])
    
    rank_found = 0
    pivot_cols = []
    
    for c in range(cols):
        if rank_found >= rows:
            break
        # Find pivot
        pivot = -1
        for r in range(rank_found, rows):
            if M_elim[r][c] != 0:
                pivot = r
                break
        if pivot == -1:
            continue
        
        # Swap rows
        M_elim[rank_found], M_elim[pivot] = M_elim[pivot], M_elim[rank_found]
        
        # Normalize pivot row
        pivot_val = M_elim[rank_found][c]
        for cc in range(c, cols):
            M_elim[rank_found][cc] /= pivot_val
        
        # Eliminate other rows
        for r in range(rows):
            if r == rank_found:
                continue
            factor = M_elim[r][c]
            if factor != 0:
                for cc in range(c, cols):
                    M_elim[r][cc] -= factor * M_elim[rank_found][cc]
        
        pivot_cols.append(c)
        rank_found += 1
        
        # Time check
        if time.time() > deadline - 5:
            break
    
    # The pivot columns correspond to the independent terms.
    # We can reconstruct the decomposition using only these terms.
    # However, the coefficients might not be 1.
    # In the original decomposition, each term was a rank-1 tensor.
    # If we keep only the pivot columns, the tensor is the sum of those terms.
    # But wait, the Gaussian elimination was on the *flattened tensor*.
    # This means the set of pivot columns spans the tensor space.
    # So T = sum_{r in pivot_cols} c_r * T_r ?
    # No, the columns of M are the vectors representing the terms.
    # M = [T_0, T_1, ..., T_{R-1}]
    # The rank is the number of pivot columns.
    # The tensor T is the sum of all columns? No.
    # The tensor T is defined by the original U, V, W.
    # The matrix M encodes T.
    # If we remove a column that is linearly dependent, does the tensor change?
    # Yes, unless we adjust the coefficients.
    # But here, the "coefficients" are fixed to 1 in the sum.
    # T = sum_r T_r.
    # If T_k = sum_{j != k} c_j T_j, then T = sum_{j != k} (1 + c_k) T_j ... wait.
    # T = T_k + sum_{j != k} T_j = sum_{j != k} c_j T_j + sum_{j != k} T_j = sum_{j != k} (1 + c_j) T_j.
    # This changes the coefficients.
    # The task requires U, V, W such that T = sum_r U_r V_r W_r.
    # If we change the coefficients, we can absorb them into U, V, or W.
    # e.g., if the coefficient becomes (1+c_j), we can multiply U_j by (1+c_j).
    
    # So, we need to find the linear dependencies.
    # From RREF, we can express the non-pivot columns as linear combinations of pivot columns.
    # Let P be the set of pivot columns.
    # For each non-pivot column q, T_q = sum_{p in P} M_elim[q, p] * T_p ?
    # No, the rows of M_elim are linear combinations of the original rows.
    # The columns of M_elim are the transformed columns.
    # If column q is not a pivot, its entries in the pivot rows are the coefficients.
    # Specifically, T_q = sum_{p in P} M_elim[pivot_row_of_p, q] * T_p.
    
    # Let's extract the coefficients.
    # pivot_rows: list of row indices corresponding to pivot columns.
    # Actually, in the loop above, rank_found was the row index.
    # Let's re-run elimination more carefully to track coefficients.
    
    # Re-implement elimination to get dependency coefficients.
    M_elim2 = [row[:] for row in M]
    rows = len(M_elim2)
    cols = len(M_elim2[0])
    
    pivot_cols2 = []
    pivot_rows2 = []
    
    for c in range(cols):
        if len(pivot_rows2) >= rows:
            break
        pivot = -1
        for r in range(len(pivot_rows2), rows):
            if M_elim2[r][c] != 0:
                pivot = r
                break
        if pivot == -1:
            continue
        
        M_elim2[len(pivot_rows2)], M_elim2[pivot] = M_elim2[pivot], M_elim2[len(pivot_rows2)]
        
        pivot_val = M_elim2[len(pivot_rows2)][c]
        for cc in range(c, cols):
            M_elim2[len(pivot_rows2)][cc] /= pivot_val
        
        for r in range(rows):
            if r == len(pivot_rows2):
                continue
            factor = M_elim2[r][c]
            if factor != 0:
                for cc in range(c, cols):
                    M_elim2[r][cc] -= factor * M_elim2[len(pivot_rows2)][cc]
        
        pivot_cols2.append(c)
        pivot_rows2.append(len(pivot_rows2))
        
        if time.time() > deadline - 5:
            break
    
    # Now, for any column q not in pivot_cols2:
    # The entries in M_elim2[pivot_rows2, q] give the coefficients to express T_q in terms of T_p.
    # T_q = sum_{i, p_i in pivot_cols2} M_elim2[i, q] * T_{p_i}
    
    # We want to rewrite T = sum_{r} T_r as a sum of only pivot terms with adjusted coefficients.
    # T = sum_{p in P} T_p + sum_{q not in P} T_q
    #   = sum_{p in P} T_p + sum_{q not in P} (sum_{p in P} M_elim2[row_p, q] T_p)
    #   = sum_{p in P} (1 + sum_{q not in P} M_elim2[row_p, q]) T_p
    
    # Let's compute the new coefficients.
    new_coeffs = {p: Fraction(1) for p in pivot_cols2}
    
    non_pivot_cols = [c for c in range(cols) if c not in pivot_cols2]
    
    for q in non_pivot_cols:
        for i, p in enumerate(pivot_cols2):
            # row i in M_elim2 corresponds to pivot p
            coeff = M_elim2[i][q]
            if coeff != 0:
                new_coeffs[p] += coeff
    
    # Check if we actually reduced the rank
    if len(pivot_cols2) < rank:
        # Reconstruct U, V, W with new coefficients
        U_new = [[0] * len(pivot_cols2) for _ in range(n * m)]
        V_new = [[0] * len(pivot_cols2) for _ in range(m * p)]
        W_new = [[0] * len(pivot_cols2) for _ in range(p * n)]
        
        for idx_new, p in enumerate(pivot_cols2):
            coeff = new_coeffs[p]
            if coeff == 0:
                continue
            # Absorb coefficient into U
            for i in range(n * m):
                U_new[i][idx_new] = U_frac[i][p] * coeff
            # V and W remain same
            for j in range(m * p):
                V_new[j][idx_new] = V_frac[j][p]
            for k in range(p * n):
                W_new[k][idx_new] = W_frac[k][p]
        
        # Verify
        # T_new = sum_r U_new_r V_new_r W_new_r
        # Should equal T_old
        # We can do a quick check with random matrices
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((p, n))
        for r in range(len(pivot_cols2)):
            if new_coeffs[pivot_cols2[r]] == 0:
                continue
            # Compute contribution
            # U_r is vector of length n*m
            # V_r is vector of length m*p
            # W_r is vector of length p*n
            # C[i,k] += sum_j U[i*m+j] A[i,j] V[j*p+k] B[j,k] W[k*n+i]
            # This is O(n*m*p*rank)
            # Since rank is small, it's fine.
            for i in range(n):
                for k in range(p):
                    w_val = W_new[k*n+i][r]
                    if w_val == 0:
                        continue
                    total = 0
                    for j in range(m):
                        u_val = U_new[i*m+j][r]
                        v_val = V_new[j*p+k][r]
                        total += u_val * A_test[i,j] * v_val * B_test[j,k]
                    C_computed[k,i] += w_val * total
        
        # Check error
        error = np.max(np.abs(C_expected - C_computed))
        if error < 1e-9:
            # Convert back to float for output (or keep as list of numbers)
            # The task allows integer/half-integer entries. Fractions are exact.
            # We can return lists of floats or integers if they are integers.
            # Let's return floats for simplicity, as numpy handles them.
            U_out = [[float(x) for x in row] for row in U_new]
            V_out = [[float(x) for x in row] for row in V_new]
            W_out = [[float(x) for x in row] for row in W_new]
            
            # Clean up tiny floating point errors
            for mat in [U_out, V_out, W_out]:
                for row in mat:
                    for i in range(len(row)):
                        if abs(row[i]) < 1e-9:
                            row[i] = 0.0
                        elif abs(row[i] - round(row[i])) < 1e-9:
                            row[i] = float(round(row[i]))
        
        return {"U": U_out, "V": V_out, "W": W_out}
    
    # If no reduction found, return original
    return {"U": U, "V": V, "W": W}
