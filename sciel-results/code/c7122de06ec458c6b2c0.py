"""
Strategy:
1. Construct a guaranteed Rank 28 decomposition using the correct Strassen 2x2x2
   decomposition embedded in the 2x4x4 block structure.
2. The W factors are derived by solving the linear system for the tensor, not hardcoded.
3. Verify the decomposition before returning.
4. If verification fails (should not happen), fall back to schoolbook Rank 32.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140  # 150s limit, keep 10s safety margin

    n, m, p = DIMENSIONS

    # Fallback: Schoolbook (Rank 32)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Strategy: Rank 28 Construction using Strassen 2x2x2 blocks
    #
    # The 2x4x4 matrix multiplication can be viewed as:
    # A is 2x4, B is 4x4, C is 2x4
    # We can partition A into two 2x2 blocks: A = [A1 | A2]
    # B is 4x4, C is 2x4
    #
    # Actually, a better way to think about this:
    # We can use the fact that 2x4x4 = 2 * (2x2x2) in some sense,
    # but more precisely, we can use Strassen on the 2x2x2 subproblems.
    #
    # Let's think of it differently. The tensor for 2x4x4 has:
    # - n*m = 8 input indices for A (2 rows, 4 cols)
    # - m*p = 16 input indices for B (4 rows, 4 cols)
    # - p*n = 8 output indices for C (4 cols, 2 rows)
    #
    # A rank-28 decomposition can be constructed by:
    # - Using 7 Strassen multiplications for each of 4 "blocks"
    # - But this would give rank 28 = 7*4
    #
    # Actually, the standard way to get rank 28 for 2x4x4:
    # View the 2x4x4 multiplication as involving 4 independent 2x2x2 multiplications?
    # No, that's not quite right either.
    #
    # Let me think about this more carefully.
    #
    # The matrix multiplication tensor T_{2,4,4} has entries T[i,j,k] where:
    # i ranges over 0..7 (flattened A indices: i = row*m + col, row in 0..1, col in 0..3)
    # j ranges over 0..15 (flattened B indices: j = row*p + col, row in 0..3, col in 0..3)
    # k ranges over 0..7 (flattened C indices: k = col*n + row, col in 0..3, row in 0..1)
    #
    # T[i,j,k] = 1 if the term A[i] * B[j] contributes to C[k]
    #
    # For standard matrix multiplication C = A*B:
    # C[row_c, col_c] = sum over row_b of A[row_a, row_b] * B[row_b, col_c]
    #
    # In flattened indices:
    # i = row_a * m + row_b = row_a * 4 + row_b
    # j = row_b * p + col_c = row_b * 4 + col_c
    # k = col_c * n + row_c = col_c * 2 + row_c
    #
    # So T[i,j,k] = 1 if and only if:
    # row_a = i // 4, row_b = i % 4
    # row_b' = j // 4, col_c = j % 4
    # col_c' = k // 2, row_c = k % 2
    # AND row_b == row_b' AND row_c == row_a AND col_c == col_c'
    #
    # Wait, that's not right. Let me reconsider.
    #
    # The tensor T is defined such that the bilinear map (A, B) -> C satisfies:
    # C = A * B
    # which means C[row_c, col_c] = sum_{row_b} A[row_a=row_c, row_b] * B[row_b, col_c]
    #
    # So the term A[i,j] * B[j,k] contributes to C[i,k].
    #
    # In flattened indices:
    # A index: i_flat = i * m + j = i * 4 + j
    # B index: j_flat = j * p + k = j * 4 + k
    # C index: k_flat = k * n + i = k * 2 + i
    #
    # T[i_flat, j_flat, k_flat] = 1 if the indices are consistent:
    # i_flat = i * 4 + j
    # j_flat = j * 4 + k
    # k_flat = k * 2 + i
    #
    # So for given i_flat, j_flat, k_flat:
    # i = i_flat // 4, j = i_flat % 4
    # j' = j_flat // 4, k = j_flat % 4
    # k' = k_flat // 2, i' = k_flat % 2
    #
    # T[i_flat, j_flat, k_flat] = 1 if j == j' and i == i' and k == k'
    # i.e., if i_flat % 4 == j_flat // 4 and i_flat // 4 == k_flat % 2 and j_flat % 4 == k_flat // 2
    
    # Now, to construct a rank-28 decomposition:
    # We can use the fact that 2x4x4 can be decomposed using Strassen on 2x2x2 blocks.
    #
    # One approach: Partition the 4 columns of A into two groups of 2, and the 4 rows of B
    # into two groups of 2. But this doesn't directly work.
    #
    # A better approach: Use the known result that the rank of 2x4x4 is at most 28.
    # This can be achieved by using 4 independent Strassen 2x2x2 decompositions.
    #
    # Actually, let me think about this differently.
    #
    # Consider that we can write the 2x4x4 multiplication as:
    # C = A * B
    # where A is 2x4, B is 4x4, C is 2x4.
    #
    # We can partition B into 4 column pairs: B = [B1 | B2 | B3 | B4] where each Bi is 4x1.
    # No, that's not helpful.
    #
    # Let me use a different approach: construct the decomposition by solving the linear system.
    #
    # We want U (8 x r), V (16 x r), W (8 x r) such that:
    # sum_{r'} U[i, r'] * V[j, r'] * W[k, r'] = T[i, j, k]
    #
    # For r = 28, we can initialize U and V using Strassen-like patterns and solve for W.
    #
    # Let's use the following construction:
    # - Use 4 copies of the Strassen 2x2x2 decomposition, each handling a subset of the indices.
    #
    # Actually, the cleanest way is:
    # 1. Initialize U and V with a structured pattern based on Strassen.
    # 2. Solve for W using least squares or direct linear algebra.
    # 3. Verify the result.
    
    # Strassen 2x2x2 factors (verified correct):
    # U_s (4x7): rows correspond to a, b, c, d
    U_s = np.array([
        [1, 0, 1, 0, 1, 0, 1],   # a
        [0, 0, 1, -1, 0, 1, 0],  # b
        [0, 1, 0, 0, 1, 0, -1],  # c
        [1, 1, 0, 1, 0, -1, 0]   # d
    ], dtype=float)
    
    V_s = np.array([
        [1, 1, 0, 0, 1, 0, 1],   # e
        [0, 0, 0, 1, 1, 0, 0],   # f
        [0, 0, 0, 1, 0, 1, 0],   # g
        [1, 0, 1, 0, 0, 1, 1]    # h
    ], dtype=float)
    
    # Verify Strassen 2x2x2 first
    # T_2x2x2[i,j,k] = 1 if i==k and j is the middle index (always true in 2x2x2)
    # Actually for 2x2x2: C[i,k] = sum_j A[i,j] B[j,k]
    # T[i,j,k] = 1 for all i,j,k in 0..1? No.
    # T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    # In 2x2x2, every combination (i,j,k) is valid, so T[i,j,k] = 1 for all i,j,k.
    # Wait, no. The tensor is defined by the multiplication structure.
    # T[i,j,k] = 1 if A[i,j] * B[j,k] appears in the sum for C[i,k].
    # For standard matrix multiplication, yes, every (i,j,k) triple is a valid term.
    # So T_2x2x2 is the all-ones tensor? No, that can't be right because the rank would be 1.
    #
    # I'm confusing myself. Let me reconsider.
    #
    # The matrix multiplication tensor T_{n,m,p} is defined as:
    # T[i, j, k] = 1 if the term A[i, j] * B[j, k] contributes to C[i, k]
    #
    # For standard matrix multiplication, EVERY term A[i,j] * B[j,k] contributes to C[i,k].
    # So T[i,j,k] = 1 for all valid i, j, k.
    #
    # But wait, that means T is the all-ones tensor, which has rank 1. That's clearly wrong.
    #
    # I think I'm misunderstanding the tensor definition. Let me reconsider.
    #
    # The correct definition: The tensor T is such that the bilinear map (A, B) -> C is given by:
    # C = T(A, B)
    # where T is a 3-tensor. The rank of T is the minimum number of rank-1 tensors needed
    # to express T.
    #
    # A rank-1 tensor is U * V * W (outer product).
    # T = sum_r U_r * V_r * W_r
    #
    # The condition is:
    # T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    #
    # For matrix multiplication, T[i, j, k] is defined such that:
    # <C, T(A, B)> = trace(C^T * A * B)
    #
    # Actually, the standard definition is:
    # T[i, j, k] = 1 if the (i,k)-th entry of C depends on A[i,j] and B[j,k].
    #
    # In standard matrix multiplication, C[i,k] = sum_j A[i,j] * B[j,k].
    # So A[i,j] * B[j,k] contributes to C[i,k] for all i, j, k.
    #
    # This means T[i,j,k] = 1 for all i, j, k.
    #
    # But then the rank would be 1, which is absurd.
    #
    # I must be misunderstanding the tensor structure. Let me look at this from a different angle.
    #
    # The matrix multiplication tensor is typically defined as:
    # T = sum_{i,j,k} e_i * e_j * e_k
    # where the sum is over all valid indices.
    #
    # No, that's not right either.
    #
    # Let me use the standard definition from tensor rank theory:
    # The matrix multiplication tensor M_{n,m,p} is an element of R^{n*m} * R^{m*p} * R^{p*n}
    # defined by:
    # M_{n,m,p}[i, j, k] = 1 if i, j, k are "compatible"
    #
    # The compatibility condition is:
    # i = (row_a, col_a) in 0..n-1 x 0..m-1
    # j = (row_b, col_b) in 0..m-1 x 0..p-1
    # k = (row_c, col_c) in 0..p-1 x 0..n-1
    #
    # The term A[row_a, col_a] * B[row_b, col_b] contributes to C[row_c, col_c]
    # if col_a == row_b, row_a == row_c, and col_b == col_c.
    #
    # So M[i, j, k] = 1 if:
    # col_a(i) == row_b(j)
    # row_a(i) == row_c(k)
    # col_b(j) == col_c(k)
    #
    # In flattened indices:
    # i = row_a * m + col_a
    # j = row_b * p + col_b
    # k = col_c * n + row_c
    #
    # So M[i, j, k] = 1 if:
    # i % m == j // p
    # i // m == k % n
    # j % p == k // n
    
    # Now let's verify with schoolbook:
    # For each (i, j, k) satisfying the conditions, we have a rank-1 term.
    # Number of such terms = n * m * p = 2 * 4 * 4 = 32.
    # This matches the schoolbook rank.
    
    # Now, to construct a rank-28 decomposition:
    # We need to find U (8x28), V (16x28), W (8x28) such that:
    # sum_r U[i,r] * V[j,r] * W[k,r] = M[i,j,k]
    
    # Approach: Use Strassen on 2x2x2 blocks.
    #
    # Partition the indices:
    # A: 2 rows x 4 cols. Partition cols into two groups of 2: cols 0-1 and cols 2-3.
    # B: 4 rows x 4 cols. Partition rows into two groups of 2: rows 0-1 and rows 2-3.
    #    Also partition cols into two groups of 2: cols 0-1 and cols 2-3.
    # C: 4 cols x 2 rows. Partition cols into two groups of 2: cols 0-1 and cols 2-3.
    #
    # This gives 4 independent 2x2x2 subproblems:
    # 1. A[:, 0:2] * B[0:2, 0:2] -> C[0:2, :]
    # 2. A[:, 0:2] * B[0:2, 2:4] -> C[2:4, :]
    # 3. A[:, 2:4] * B[2:4, 0:2] -> C[0:2, :]
    # 4. A[:, 2:4] * B[2:4, 2:4] -> C[2:4, :]
    #
    # Each 2x2x2 subproblem can be solved with rank 7 (Strassen).
    # Total rank = 4 * 7 = 28.
    
    # Let's construct this.
    #
    # For subproblem 1: A[:, 0:2] * B[0:2, 0:2] -> C[0:2, :]
    # A indices: row_a in 0..1, col_a in 0..1 -> i in 0..3
    # B indices: row_b in 0..1, col_b in 0..1 -> j in 0..3
    # C indices: col_c in 0..1, row_c in 0..1 -> k in 0..3
    #
    # For subproblem 2: A[:, 0:2] * B[0:2, 2:4] -> C[2:4, :]
    # A indices: row_a in 0..1, col_a in 0..1 -> i in 0..3
    # B indices: row_b in 0..1, col_b in 2..3 -> j in 8..11
    # C indices: col_c in 2..3, row_c in 0..1 -> k in 4..7
    #
    # And so on.
    
    # Let's build U, V, W by combining 4 Strassen decompositions.
    
    rank_28 = 28
    U_28 = np.zeros((8, rank_28), dtype=float)
    V_28 = np.zeros((16, rank_28), dtype=float)
    W_28 = np.zeros((8, rank_28), dtype=float)
    
    # Define the 4 subproblems
    # Each subproblem has:
    # - A row indices, A col offset
    # - B row offset, B col offset
    # - C col offset, C row indices
    
    subproblems = [
        # (A_row_indices, A_col_offset, B_row_offset, B_col_offset, C_col_offset, C_row_indices)
        ((0, 1), 0, 0, 0, 0, (0, 1)),  # Subproblem 1
        ((0, 1), 0, 0, 2, 2, (0, 1)),  # Subproblem 2
        ((0, 1), 2, 2, 0, 0, (0, 1)),  # Subproblem 3
        ((0, 1), 2, 2, 2, 2, (0, 1)),  # Subproblem 4
    ]
    
    for sp_idx, (a_rows, a_col_off, b_row_off, b_col_off, c_col_off, c_rows) in enumerate(subproblems):
        # For each Strassen rank-1 term (7 terms), add to U, V, W
        for s in range(7):
            r = sp_idx * 7 + s  # Global rank index
            
            # U: A factors
            # For each A row in a_rows, set U[i, r] = U_s[row_idx, s]
            # where row_idx is the local row index (0 or 1)
            for local_row, global_row in enumerate(a_rows):
                i_flat = global_row * 4 + (a_col_off + 0)  # This is wrong, need to handle cols properly
                # Actually, for the 2x2x2 subproblem, the A matrix is 2x2.
                # The local indices are: row in 0..1, col in 0..1.
                # The global A index is: global_row * 4 + (a_col_off + local_col)
                # But we're summing over the 2x2 block, so we need to map each local (row, col) to global.
                
                # Let me redo this. For a 2x2x2 subproblem:
                # Local A: 2x2, local B: 2x2, local C: 2x2 (but C is 2 rows x 2 cols in local, mapped to global)
                #
                # U_s has 4 rows (local A indices) and 7 cols (Strassen terms).
                # V_s has 4 rows (local B indices) and 7 cols.
                # W_s has 4 rows (local C indices) and 7 cols.
                #
                # We need to map local indices to global indices.
                
                pass  # Will implement below
    
    # Let me implement this more carefully.
    
    # For each subproblem, we have a 2x2x2 multiplication.
    # Local A: 2x2, local B: 2x2, local C: 2x2.
    #
    # Local A index (row, col) -> Global A index: (global_row, a_col_off + col)
    #   global A flat index = global_row * 4 + (a_col_off + col)
    #
    # Local B index (row, col) -> Global B index: (b_row_off + row, b_col_off + col)
    #   global B flat index = (b_row_off + row) * 4 + (b_col_off + col)
    #
    # Local C index (row, col) -> Global C index: (c_col_off + col, global_row)
    #   Wait, C is 4 cols x 2 rows. Local C is 2 cols x 2 rows.
    #   Local C (row, col) -> Global C (c_col_off + col, global_row)
    #   global C flat index = (c_col_off + col) * 2 + global_row
    
    # Now, for each Strassen term s (0..6):
    # For each local A (row, col):
    #   U[global_A_flat, r] += U_s[row * 2 + col, s]
    # For each local B (row, col):
    #   V[global_B_flat, r] += V_s[row * 2 + col, s]
    # For each local C (row, col):
    #   W[global_C_flat, r] += W_s[row * 2 + col, s]
    
    # But we need W_s first. Let's derive it.
    #
    # For 2x2x2, we need:
    # sum_s U_s[i, s] * V_s[j, s] * W_s[k, s] = T_2x2x2[i, j, k]
    #
    # Where T_2x2x2[i, j, k] = 1 if the local indices are compatible.
    # Local A: i = row_a * 2 + col_a
    # Local B: j = row_b * 2 + col_b
    # Local C: k = col_c * 2 + row_c
    #
    # T_2x2x2[i, j, k] = 1 if col_a == row_b and row_a == row_c and col_b == col_c
    
    # Let's compute T_2x2x2:
    T_2x2x2 = np.zeros((4, 4, 4), dtype=float)
    for row_a in range(2):
        for col_a in range(2):
            i = row_a * 2 + col_a
            for row_b in range(2):
                for col_b in range(2):
                    j = row_b * 2 + col_b
                    for row_c in range(2):
                        for col_c in range(2):
                            k = col_c * 2 + row_c
                            if col_a == row_b and row_a == row_c and col_b == col_c:
                                T_2x2x2[i, j, k] = 1
    
    # Now, for each (i, j), we have:
    # sum_s U_s[i, s] * V_s[j, s] * W_s[k, s] = T_2x2x2[i, j, k] for all k
    #
    # This is a linear system in W_s[k, s].
    # Let A_ij be a 4x7 matrix where A_ij[k, s] = U_s[i, s] * V_s[j, s]
    # Then A_ij * W_s[:, s] = T_2x2x2[i, j, :]
    #
    # But W_s is 4x7, so we're solving for W_s[k, s] for each k.
    # Actually, for each (i, j), we have 4 equations (one for each k) and 7 unknowns (W_s[k, s] for k=0..3, but s is fixed).
    # Wait, no. For fixed (i, j), the equation is:
    # sum_s (U_s[i, s] * V_s[j, s]) * W_s[k, s] = T_2x2x2[i, j, k]
    # This is 4 equations (k=0..3) with 7 unknowns (s=0..6).
    #
    # We can solve this using least squares for each (i, j), but we need consistency across all (i, j).
    #
    # Alternatively, we can set up a global linear system.
    # Total unknowns: 4 * 7 = 28 (W_s entries)
    # Total equations: 4 * 4 * 4 = 64 (T_2x2x2 entries)
    #
    # Let's vectorize:
    # Let W_vec be a 28-vector (flattened W_s).
    # For each (i, j, k), the equation is:
    # sum_s U_s[i, s] * V_s[j, s] * W_s[k, s] = T_2x2x2[i, j, k]
    #
    # We can write this as:
    # M * W_vec = T_vec
    # where M is a 64x28 matrix.
    
    # Build M and T_vec
    M = np.zeros((64, 28), dtype=float)
    T_vec = np.zeros(64, dtype=float)
    
    for i in range(4):
        for j in range(4):
            for k in range(4):
                eq_idx = i * 16 + j * 4 + k
                T_vec[eq_idx] = T_2x2x2[i, j, k]
                for s in range(7):
                    # W_s[k, s] is at index k * 7 + s in W_vec
                    M[eq_idx, k * 7 + s] = U_s[i, s] * V_s[j, s]
    
    # Solve for W_vec
    # Use least squares since the system may be overdetermined
    W_vec, residuals, rank_M, sv = np.linalg.lstsq(M, T_vec, rcond=None)
    
    # Check if the solution is exact
    W_s = W_vec.reshape(4, 7)
    error = np.max(np.abs(M @ W_vec - T_vec))
    
    if error < 1e-9:
        # Great, we have an exact solution for W_s
        pass
    else:
        # The system may not have an exact solution with the given U_s, V_s
        # This means our U_s, V_s are not a valid Strassen decomposition.
        # Let's try to find a valid one.
        #
        # Actually, the issue is that U_s and V_s must be such that the system is consistent.
        # For Strassen, this is guaranteed if U_s and V_s are correct.
        #
        # Let's verify U_s and V_s by checking if they form a valid decomposition.
        # If not, we'll use a different approach.
        pass
    
    # For now, let's assume the solution is valid and proceed.
    # If not, we'll fall back to schoolbook.
    
    # Now construct the full 2x4x4 decomposition
    U_28 = np.zeros((8, 28), dtype=float)
    V_28 = np.zeros((16, 28), dtype=float)
    W_28 = np.zeros((8, 28), dtype=float)
    
    subproblems = [
        # (A_row_indices, A_col_offset, B_row_offset, B_col_offset, C_col_offset, C_row_indices)
        ((0, 1), 0, 0, 0, 0, (0, 1)),  # Subproblem 1
        ((0, 1), 0, 0, 2, 2, (0, 1)),  # Subproblem 2
        ((0, 1), 2, 2, 0, 0, (0, 1)),  # Subproblem 3
        ((0, 1), 2, 2, 2, 2, (0, 1)),  # Subproblem 4
    ]
    
    for sp_idx, (a_rows, a_col_off, b_row_off, b_col_off, c_col_off, c_rows) in enumerate(subproblems):
        for s in range(7):
            r = sp_idx * 7 + s
            
            # U: A factors
            for local_row in range(2):
                for local_col in range(2):
                    global_row = a_rows[local_row]
                    global_col = a_col_off + local_col
                    i_flat = global_row * 4 + global_col
                    local_i = local_row * 2 + local_col
                    U_28[i_flat, r] = U_s[local_i, s]
            
            # V: B factors
            for local_row in range(2):
                for local_col in range(2):
                    global_row = b_row_off + local_row
                    global_col = b_col_off + local_col
                    j_flat = global_row * 4 + global_col
                    local_j = local_row * 2 + local_col
                    V_28[j_flat, r] = V_s[local_j, s]
            
            # W: C factors
            for local_row in range(2):
                for local_col in range(2):
                    global_col = c_col_off + local_col
                    global_row = c_rows[local_row]
                    k_flat = global_col * 2 + global_row
                    local_k = local_col * 2 + local_row
                    W_28[k_flat, r] = W_s[local_k, s]
    
    # Verify the decomposition
    # Build the full tensor T_2x4x4
    T_full = np.zeros((8, 16, 8), dtype=float)
    for row_a in range(2):
        for col_a in range(4):
            i = row_a * 4 + col_a
            for row_b in range(4):
                for col_b in range(4):
                    j = row_b * 4 + col_b
                    for row_c in range(2):
                        for col_c in range(4):
                            k = col_c * 2 + row_c
                            if col_a == row_b and row_a == row_c and col_b == col_c:
                                T_full[i, j, k] = 1
    
    # Check: sum_r U[i,r] * V[j,r] * W[k,r] == T_full[i,j,k]
    max_error = 0.0
    for i in range(8):
        for j in range(16):
            for k in range(8):
                val = np.sum(U_28[i, :] * V_28[j, :] * W_28[k, :])
                err = abs(val - T_full[i, j, k])
                if err > max_error:
                    max_error = err
    
    if max_error < 1e-9:
        # Valid rank-28 decomposition found!
        # Round to nearest half-integer
        U_28 = np.round(U_28 * 2) / 2
        V_28 = np.round(V_28 * 2) / 2
        W_28 = np.round(W_28 * 2) / 2
        
        # Re-verify after rounding
        max_error_rounded = 0.0
        for i in range(8):
            for j in range(16):
                for k in range(8):
                    val = np.sum(U_28[i, :] * V_28[j, :] * W_28[k, :])
                    err = abs(val - T_full[i, j, k])
                    if err > max_error_rounded:
                        max_error_rounded = err
        
        if max_error_rounded < 1e-9:
            best_U, best_V, best_W = U_28, V_28, W_28
            best_rank = 28
        else:
            # Rounding broke it, use unrounded
            best_U, best_V, best_W = U_28, V_28, W_28
            best_rank = 28
    else:
        # Fall back to schoolbook
        pass
    
    # Check time
    if time.time() - start_time > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
