import time
import sys
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor: T[i,j,k] = 1 for all i,j,k
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Start with rank-64 schoolbook decomposition
    rank_64 = n * m * p  # 64
    U64 = np.zeros((nm, rank_64))
    V64 = np.zeros((mp, rank_64))
    W64 = np.zeros((pn, rank_64))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U64[i * m + j, r] = 1.0
                V64[j * p + k, r] = 1.0
                W64[k * n + i, r] = 1.0
    
    def compute_tensor(U_arr, V_arr, W_arr, rank_val):
        """Compute the decomposed tensor from factor matrices."""
        tensor = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_vals = U_arr[i*m+j, :rank_val]
                    v_vals = V_arr[j*p+k, :rank_val]
                    w_vals = W_arr[k*n+i, :rank_val]
                    tensor[i,j,k] = np.dot(u_vals, v_vals * w_vals)
        return tensor
    
    def is_valid(U_arr, V_arr, W_arr, rank_val):
        """Check if the decomposition exactly computes matrix multiplication."""
        tensor = compute_tensor(U_arr, V_arr, W_arr, rank_val)
        expected = np.ones((n, m, p))
        return np.allclose(tensor, expected, atol=1e-9)
    
    # Structural approach: Alternating projection to find lower-rank exact decompositions
    # Key idea: For a given rank r, we can use the fact that the problem is polynomial
    # in the entries. We'll use a "deflation" approach:
    # 1. Start with rank-64
    # 2. Try to find a rank-r solution by solving a system of equations
    
    # Alternative structural approach: Use the block structure more cleverly
    # For 2x4x8, we can think of it as:
    # A is 2x4, B is 4x8, C is 2x8
    # C[0,:] = A[0,:] * B[0:2,:] + A[0,:] * B[2:4,:]  (not quite, that's just splitting)
    
    # Let me try a direct algebraic approach:
    # Parameterize with a smaller rank and solve using nonlinear least squares
    # but with a smarter initialization based on known low-rank constructions
    
    # Known: For 2x2x2, rank 7 (Strassen)
    # For 2x4x8, we can use block Strassen:
    # Split B into two 4x4 blocks: B = [B1 | B2]
    # Then C = A * B = A*B1 | A*B2
    # Each A*B is 2x4*4x4 = 2x4x4, which can be done with rank 7*2=14 per block
    # Total: 14 + 14 = 28? No, that's not right.
    
    # Actually, let's use the standard block Strassen for 2x4x4:
    # A is 2x4, split into [A1 | A2] where A1,A2 are 2x2
    # B is 4x4, split into [[B11, B12], [B21, B22]]
    # Then C = A*B can be computed with 7 multiplications of 2x2 matrices
    # Each 2x2 multiplication has rank 7
    # So 2x4x4 has rank 7*4 = 28? No...
    
    # Let me just implement the rank-56 Strassen properly and then try to reduce
    
    # Rank-56 Strassen for 2x4x8:
    # Split A (2x4) into two 2x2 blocks: A = [A_left | A_right]
    # Split B (4x8) into four 2x4 blocks: B = [[B_tl, B_tr], [B_bl, B_br]]
    # C = A*B = [A_left*B_tl + A_right*B_bl | A_left*B_tr + A_right*B_br]
    # Each 2x2 * 2x4 multiplication can be done with rank 7 (Strassen for 2x2x2 applied to columns)
    # Actually 2x2 * 2x4: treat the 4 columns as 2 groups of 2
    # Each 2x2*2x2 has rank 7, so 2x2*2x4 has rank 14
    # We have 4 such products: A_left*B_tl, A_left*B_tr, A_right*B_bl, A_right*B_br
    # Total rank: 4*14 = 56
    
    rank = 56
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Strassen for 2x2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    strassen_U = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1},
    ]
    strassen_V = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1},
    ]
    strassen_W = [
        {(0,0): 1, (1,1): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(0,0): -1, (1,0): 1},
        {(1,1): 1},
        {(0,0): 1},
    ]
    
    def add_strassen_block(base_rank, i_base, j_base, k_base):
        """Add a Strassen block for 2x2 * 2x2 at position (i_base, j_base, k_base)"""
        for r in range(7):
            cr = base_rank + r
            for (li, lj), val in strassen_U[r].items():
                gi = i_base + li
                gj = j_base + lj
                U[gi * 4 + gj, cr] += val
            for (lj, lk), val in strassen_V[r].items():
                gj = j_base + lj
                gk = k_base + lk
                V[gj * 8 + gk, cr] += val
            for (lk, li), val in strassen_W[r].items():
                gk = k_base + lk
                gi = i_base + li
                W[gk * 2 + gi, cr] += val
    
    # Four 2x2*2x2 blocks for the 2x4x8 product:
    # A = [A0 | A1] where A0, A1 are 2x2 (rows 0-1, cols 0-1 and 2-3)
    # B = [[B00, B01], [B10, B11]] where each is 2x4
    # Wait, B is 4x8, so splitting rows into 2+2 and cols into 4+4:
    # B = [[B_tl(2x4), B_tr(2x4)], [B_bl(2x4), B_br(2x4)]]
    # C = A*B: C[i,k] = sum_j A[i,j]*B[j,k]
    # C = A0*B_tl + A1*B_bl for left half, A0*B_tr + A1*B_br for right half
    # But A0*B_tl is 2x2 * 2x4, not 2x2*2x2
    
    # I need to split further. Let me split B's 4 columns into 2+2 as well.
    # So B has 4 row blocks of size 2 and 4 col blocks of size 2? No, B is 4x8.
    # Let me split B into 2x2 blocks: 2 row-blocks of size 2, 4 col-blocks of size 2
    # B = [[B00, B01, B02, B03], [B10, B11, B12, B13]] where each is 2x2
    # A = [A0, A1] where A0, A1 are 2x2 (col blocks)
    # C = A*B: C[i, k] = A0[i, :] * B[:, k] + A1[i, :] * B[:, k+2] etc.
    # This gets complicated. Let me just use the 8 blocks of 2x2*2x2.
    
    # Actually, the cleanest way: 
    # A is 2x4 = [A_col0 | A_col1] where A_col0, A_col1 are 2x2
    # B is 4x8. Split B's rows into 2 groups of 2: B = [B_top; B_bottom] where B_top, B_bottom are 2x8
    # Split B's cols into 4 groups of 2: B_top = [B_top0 | B_top1 | B_top2 | B_top3]
    # Similarly for B_bottom
    # Then C = A_col0 * B_top + A_col1 * B_bottom
    # C_col_k = A_col0 * B_top_col_k + A_col1 * B_bottom_col_k
    # Each A_colX * B_Y_col_k is 2x2 * 2x2, rank 7
    # We have 2 (col blocks of A) * 4 (col blocks of B) = 8 such products
    # Total rank: 8 * 7 = 56
    
    # Block (a, b) where a in {0,1} is the A col block, b in {0,1,2,3} is the B col block
    # A col block a: rows 0-1, cols 2a to 2a+1
    # B col block b: rows 0-3, cols 2b to 2b+1
    # But B col block b needs to be split into top (rows 0-1) and bottom (rows 2-3)
    
    # For product a=0, b=0:
    # A_part = A[0:2, 0:2] (2x2)
    # B_part = B[0:4, 0:2] (4x2) = [B_top(2x2); B_bottom(2x2)]
    # C_part = A_part * B_part = A[0:2,0:2] * B[0:4,0:2]
    # = A_col0 * [B_top0; B_bottom0]
    # = A_col0 * B_top0 + A_col0 * B_bottom0
    # First term: A_col0(2x2) * B_top0(2x2) -> 2x2 result, contributes to C[0:2, 0:2]
    # Second term: A_col0(2x2) * B_bottom0(2x2) -> 2x2 result, contributes to C[0:2, 0:2]
    
    # So for each (a, b) pair, we need two Strassen decompositions:
    # One for A_col_a * B_top_b and one for A_col_a * B_bottom_b
    # That's 2 * 2 * 4 = 16 Strassen blocks? No, 2 (a values) * 4 (b values) * 2 (top/bottom) = 16
    # 16 * 7 = 112, that's too many.
    
    # I think I'm overcomplicating this. Let me just use the original 8-block approach
    # where each block is a 2x2 * 2x2 Strassen, but I need to be more careful.
    
    # Reset and redo:
    U[:] = 0
    V[:] = 0
    W[:] = 0
    
    # The 8 blocks: for each combination of A's 2 col-blocks and B's 4 col-blocks
    # A col-block a (a=0,1): A[0:2, 2a:2a+2]
    # B col-block b (b=0,1,2,3): B[0:4, 2b:2b+2]
    # But B col-block is 4x2, so we split it into B[0:2, 2b:2b+2] and B[2:4, 2b:2b+2]
    # Product: A_col_a * B_col_b = A_col_a * B_col_b_top + A_col_a * B_col_b_bottom
    # Each is 2x2 * 2x2
    
    # For (a, b, part) where part in {0,1} (0=top, 1=bottom):
    # A indices: i in 0-1, j in 2a to 2a+1
    # B indices: j' in part*2 to part*2+1, k in 2b to 2b+1
    # C indices: i in 0-1, k in 2b to 2b+1
    
    block_count = 0
    for a in range(2):  # A col block
        for b in range(4):  # B col block
            for part in range(2):  # top/bottom of B
                base_rank = block_count * 7
                i_base = 0  # A rows always 0-1
                j_base = 2 * a  # A col offset
                b_j_base = part * 2  # B row offset (0 or 2)
                k_base = 2 * b  # B col offset
                
                for r in range(7):
                    cr = base_rank + r
                    for (li, lj), val in strassen_U[r].items():
                        gi = i_base + li
                        gj = j_base + lj
                        U[gi * 4 + gj, cr] += val
                    for (lj, lk), val in strassen_V[r].items():
                        gj = b_j_base + lj
                        gk = k_base + lk
                        V[gj * 8 + gk, cr] += val
                    for (lk, li), val in strassen_W[r].items():
                        gk = k_base + lk
                        gi = i_base + li
                        W[gk * 2 + gi, cr] += val
                block_count += 1
    
    # Verify the rank-56 decomposition
    if not is_valid(U, V, W, rank):
        # Fallback to rank-64
        return {"U": U64.tolist(), "V": V64.tolist(), "W": W64.tolist()}
    
    best_rank = rank
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # Now try to reduce rank using alternating least squares with rank constraint
    # Structural approach: Try rank r < 56 by solving the polynomial system
    # We'll use a "fix two, solve for third" alternating scheme
    
    def try_reduce_rank(target_rank, max_iters=200):
        """Try to find a rank-target_rank exact decomposition using alternating projection."""
        if time.time() > deadline - 10:
            return None
        
        # Initialize from the rank-56 solution, truncated
        U_cur = best_U[:, :target_rank].copy()
        V_cur = best_V[:, :target_rank].copy()
        W_cur = best_W[:, :target_rank].copy()
        
        # Add small random perturbation to escape local minima
        np.random.seed(42 + target_rank)
        U_cur += np.random.randn(nm, target_rank) * 0.01
        V_cur += np.random.randn(mp, target_rank) * 0.01
        W_cur += np.random.randn(pn, target_rank) * 0.01
        
        for iteration in range(max_iters):
            if time.time() > deadline - 10:
                return None
            
            # Solve for U given V, W
            # For each (i,j), U[i*m+j, :] should satisfy:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
            # This is linear in U[i*m+j, :] for fixed V, W
            U_new = np.zeros_like(U_cur)
            for i in range(n):
                for j in range(m):
                    # Build system: for each k, sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
                    A_mat = np.zeros((p, target_rank))
                    for k in range(p):
                        A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve A_mat @ U[i*m+j, :] = ones(p)
                    try:
                        sol, residuals, rank_val, sv = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                        U_new[i*m+j, :] = sol
                    except:
                        U_new[i*m+j, :] = U_cur[i*m+j, :]
            
            # Solve for V given U, W
            V_new = np.zeros_like(V_cur)
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, target_rank))
                    for i in range(n):
                        A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        sol, residuals, rank_val, sv = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)
                        V_new[j*p+k, :] = sol
                    except:
                        V_new[j*p+k, :] = V_cur[j*p+k, :]
            
            # Solve for W given U, V
            W_new = np.zeros_like(W_cur)
            for k in range(p):
                for i in range(n):
                    A_mat = np.zeros((m, target_rank))
                    for j in range(m):
                        A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        sol, residuals, rank_val, sv = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                        W_new[k*n+i, :] = sol
                    except:
                        W_new[k*n+i, :] = W_cur[k*n+i, :]
            
            U_cur = U_new
            V_cur = V_new
            W_cur = W_new
            
            # Check convergence
            tensor = compute_tensor(U_cur, V_cur, W_cur, target_rank)
            error = np.max(np.abs(tensor - 1.0))
            if error < 1e-9:
                return (U_cur, V_cur, W_cur)
        
        return None
    
    # Try to reduce from rank 55 down to rank 30
    for target_rank in range(rank - 1, max(28, rank - 20), -1):
        if time.time() > deadline - 15:
            break
        
        result = try_reduce_rank(target_rank)
        if result is not None:
            U_sol, V_sol, W_sol = result
            if is_valid(U_sol, V_sol, W_sol, target_rank):
                best_rank = target_rank
                best_U = U_sol.copy()
                best_V = V_sol.copy()
                best_W = W_sol.copy()
    
    # Convert to lists
    U_final = best_U[:, :best_rank].tolist()
    V_final = best_V[:, :best_rank].tolist()
    W_final = best_W[:, :best_rank].tolist()
    
    # Round to nearest half-integer if close
    for mat in [U_final, V_final, W_final]:
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                val = mat[i][j]
                rounded = round(val * 2) / 2
                if abs(val - rounded) < 1e-8:
                    mat[i][j] = rounded
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
