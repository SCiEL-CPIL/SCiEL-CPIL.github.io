"""Program for rank-minimal 5x5x5 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify tensor decomposition exactly using integer scaling by 2."""
    def scale2(mat):
        arr = np.array(mat, dtype=np.float64) * 2
        if np.max(np.abs(arr - np.round(arr))) > 1e-9:
            return None
        return np.round(arr).astype(np.int64)
    
    U2 = scale2(U)
    V2 = scale2(V)
    W2 = scale2(W)
    if U2 is None or V2 is None or W2 is None:
        return False
    
    # Target tensor: T[a*m+b, b*p+c, c*n+a] = 1 for all (a,b,c)
    target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1
    
    # T_computed[i,j,k] = sum_c U2[i,c] * V2[j,c] * W2[k,c]
    T_computed = np.einsum('ic,jc,kc->ijk', U2, V2, W2, optimize=True)
    
    # Since each factor is scaled by 2, product is scaled by 8
    if np.max(np.abs(T_computed - target * 8)) > 1e-6:
        return False
    return True

def _build_strassen_2x2():
    """Return U, V, W factors for 2x2x2 Strassen (rank 7)."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[1][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3: A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4: A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A10-A00)*(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_strassen_4x4():
    """
    Build rank-49 decomposition for 4x4x4 matrix multiplication.
    Uses Kronecker product of two 2x2 Strassen decompositions.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for macro_k in range(7):
        for micro_l in range(7):
            col = macro_k * 7 + micro_l
            # U: index = (macro_block_row * 2 + micro_row) * 4 + (macro_block_col * 2 + micro_col)
            # For A: global index = (block_row*2 + local_row)*4 + (block_col*2 + local_col)
            # U2[local_idx] where local_idx = local_row*2 + local_col
            # macro_k determines which blocks of A are involved
            # micro_l determines the local Strassen term
            
            # The 4x4 A matrix is partitioned into 2x2 blocks:
            # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
            # Global row index for A: (block_row*2 + local_row)*4 + (block_col*2 + local_col)
            
            # For macro term k, the blocks of A that contribute are determined by Strassen
            # M1: (A00+A11) -> blocks (0,0) and (1,1)
            # M2: (A10+A11) -> blocks (1,0) and (1,1)
            # M3: A00 -> block (0,0)
            # M4: A11 -> block (1,1)
            # M5: (A00+A01) -> blocks (0,0) and (0,1)
            # M6: (A10-A00) -> blocks (1,0) and (0,0)
            # M7: (A01-A11) -> blocks (0,1) and (1,1)
            
            a_blocks = {
                0: [(0,0,1), (1,1,1)],
                1: [(1,0,1), (1,1,1)],
                2: [(0,0,1)],
                3: [(1,1,1)],
                4: [(0,0,1), (0,1,1)],
                5: [(1,0,1), (0,0,-1)],
                6: [(0,1,1), (1,1,-1)],
            }[macro_k]
            
            b_blocks = {
                0: [(0,0,1), (1,1,1)],
                1: [(0,0,1)],
                2: [(0,1,1), (1,1,-1)],
                3: [(1,0,1), (0,0,-1)],
                4: [(1,1,1)],
                5: [(0,0,1), (0,1,1)],
                6: [(1,0,1), (1,1,1)],
            }[macro_k]
            
            c_blocks = {
                0: [(0,0,1), (1,1,1)],
                1: [(1,0,1), (1,1,-1)],
                2: [(0,1,1), (1,1,1)],
                3: [(0,0,1), (1,0,1)],
                4: [(0,0,-1), (0,1,1)],
                5: [(1,1,1)],
                6: [(0,0,1)],
            }[macro_k]
            
            for local_idx in range(4):
                u_coeff = U2[local_idx][micro_l]
                v_coeff = V2[local_idx][micro_l]
                w_coeff = W2[local_idx][micro_l]
                
                local_row, local_col = local_idx // 2, local_idx % 2
                
                for (blk_r, blk_c, coeff) in a_blocks:
                    if u_coeff != 0 and coeff != 0:
                        global_idx = (blk_r * 2 + local_row) * 4 + (blk_c * 2 + local_col)
                        U[global_idx][col] += u_coeff * coeff
                
                for (blk_r, blk_c, coeff) in b_blocks:
                    if v_coeff != 0 and coeff != 0:
                        global_idx = (blk_r * 2 + local_row) * 4 + (blk_c * 2 + local_col)
                        V[global_idx][col] += v_coeff * coeff
                
                for (blk_r, blk_c, coeff) in c_blocks:
                    if w_coeff != 0 and coeff != 0:
                        # C index: (blk_c * 2 + local_col) * 4 + (blk_r * 2 + local_row)
                        global_idx = (blk_c * 2 + local_col) * 4 + (blk_r * 2 + local_row)
                        W[global_idx][col] += w_coeff * coeff
    
    return U, V, W

def _build_5x5_decomposition():
    """
    Build a decomposition for 5x5x5 matrix multiplication.
    Strategy: Start with rank-49 Strassen for 4x4, then use structured
    rank-1 terms to cover the cross-terms involving index 4.
    Uses a linear system approach with structured ansatz.
    """
    U4, V4, W4 = _build_strassen_4x4()
    
    n, m, p = 5, 5, 5
    nm, mp, pn = n*m, m*p, p*n
    
    # Build target tensor
    target = np.zeros((nm, mp, pn), dtype=np.float64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1.0
    
    # Compute what the 4x4 Strassen terms contribute
    U4_arr = np.array(U4, dtype=np.float64)
    V4_arr = np.array(V4, dtype=np.float64)
    W4_arr = np.array(W4, dtype=np.float64)
    
    # Extend to 5x5 size (pad with zeros)
    U4_full = np.zeros((nm, 49), dtype=np.float64)
    V4_full = np.zeros((mp, 49), dtype=np.float64)
    W4_full = np.zeros((pn, 49), dtype=np.float64)
    
    for i in range(16):
        U4_full[i, :] = U4_arr[i, :]
        V4_full[i, :] = V4_arr[i, :]
        W4_full[i, :] = W4_arr[i, :]
    
    # T_strassen[i,j,k] = sum_c U4_full[i,c] * V4_full[j,c] * W4_full[k,c]
    T_strassen = np.einsum('ic,jc,kc->ijk', U4_full, V4_full, W4_full, optimize=True)
    
    # Residual: what's left to cover
    residual = target - T_strassen
    
    # Structured ansatz for additional terms.
    # Each additional term r has form: u_r * v_r * w_r
    # We use a structured pattern where each term covers a subset of cross-terms.
    
    # Pattern 1: Terms covering a=4 (i in 20..24)
    # For each b,c in 0..3: need term with U[20+b]=1, V[b*5+c]=1, W[c*5+4]=1
    # That's 16 terms, but we can group them.
    
    # Let's use a more efficient structured approach.
    # Group cross-terms by their pattern.
    
    # Type A: a=4, b in 0..3, c in 0..3 (16 terms)
    # U support: {20,21,22,23} (indices 20+b), V support: {b*5+c}, W support: {c*5+4}
    
    # Type B: b=4, a in 0..3, c in 0..3 (16 terms)  
    # U support: {a*5+4}, V support: {20,21,22,23} (indices 20+c), W support: {c*5+a}
    
    # Type C: c=4, a in 0..3, b in 0..3 (16 terms)
    # U support: {a*5+b}, V support: {b*5+4}, W support: {20,21,22,23} (indices 20+a)
    
    # Type D: two indices = 4 (12 terms: (4,b,4), (4,4,c), (a,4,4) for a,b,c in 0..3)
    # Type E: all three = 4 (1 term)
    
    # Total: 16+16+16+12+1 = 61 terms (matches naive count)
    
    # Let's try to reduce by using shared structure.
    # For Type A: we need sum_r u_r[20+b] * v_r[b*5+c] * w_r[c*5+4] = 1 for all b,c in 0..3
    # This is a 4x4 matrix product: (U_A * diag(V_A)) * W_A^T where
    # U_A[b,r] = u_r[20+b], V_A[b*5+c, r] = v_r[b*5+c], W_A[c*5+4, r] = w_r[c*5+4]
    
    # Actually, let me just use the naive 61-term approach but verify it works,
    # then try to reduce.
    
    # Naive approach: one term per cross-entry
    U_add = []
    V_add = []
    W_add = []
    
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if a < 4 and b < 4 and c < 4:
                    continue
                u = np.zeros(nm)
                v = np.zeros(mp)
                w = np.zeros(pn)
                u[a*m + b] = 1.0
                v[b*p + c] = 1.0
                w[c*n + a] = 1.0
                U_add.append(u)
                V_add.append(v)
                W_add.append(w)
    
    num_add = len(U_add)
    total_rank = 49 + num_add
    
    # Assemble
    U = np.zeros((nm, total_rank), dtype=np.float64)
    V = np.zeros((mp, total_rank), dtype=np.float64)
    W = np.zeros((pn, total_rank), dtype=np.float64)
    
    U[:, :49] = U4_full
    V[:, :49] = V4_full
    W[:, :49] = W4_full
    
    for r in range(num_add):
        U[:, 49 + r] = U_add[r]
        V[:, 49 + r] = V_add[r]
        W[:, 49 + r] = W_add[r]
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return U_list, V_list, W_list

def _try_random_rank_reduction(U, V, W, n, m, p, target_rank, deadline, rng):
    """
    Attempt to reduce rank by finding linear dependencies.
    Uses a structured approach: try to find columns that are linear combinations
    of others in a way that preserves the tensor decomposition.
    """
    if time.time() > deadline - 10:
        return None
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    current_rank = U_arr.shape[1]
    if current_rank <= target_rank:
        return U, V, W
    
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the target tensor
    target = np.zeros((nm, mp, pn), dtype=np.float64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1.0
    
    # For each pair of columns, check if their sum can be absorbed into a single term.
    # Column r contributes: U[:,r] * V[:,r] * W[:,r] (outer product)
    # We want to check if col_a + col_b can be replaced by a single rank-1 term.
    
    # More practically: check if one column's contribution is a linear combination
    # of other columns' contributions in the tensor space.
    
    # The tensor contribution of column r is: T_r[i,j,k] = U[i,r]*V[j,r]*W[k,r]
    # We want to find if T_r = sum_{s != r} alpha_s * T_s for some alphas.
    
    # This is a system of nm*mp*pn equations in current_rank-1 unknowns.
    # For 5x5x5: 125*125*125 = ~2M equations, too many.
    
    # Instead, use a sampled approach: check on a subset of tensor entries.
    
    # For each column to remove, set up a least-squares problem on sampled entries.
    num_samples = 500
    rng_local = np.random.RandomState(12345)
    
    sample_indices = []
    for _ in range(num_samples):
        i = rng_local.randint(nm)
        j = rng_local.randint(mp)
        k = rng_local.randint(pn)
        sample_indices.append((i, j, k))
    
    for col_to_remove in range(current_rank):
        if time.time() > deadline - 5:
            break
        
        # Check if this column is redundant
        # Set up: for each sample (i,j,k), we need:
        # U[i,col]*V[j,col]*W[k,col] = sum_{s != col} alpha_s * U[i,s]*V[j,s]*W[k,s]
        
        # Build the system
        A_mat = np.zeros((num_samples, current_rank - 1), dtype=np.float64)
        b_vec = np.zeros(num_samples, dtype=np.float64)
        
        col_idx = 0
        for s in range(current_rank):
            if s == col_to_remove:
                continue
            for idx, (i, j, k) in enumerate(sample_indices):
                A_mat[idx, col_idx] = U_arr[i, s] * V_arr[j, s] * W_arr[k, s]
            col_idx += 1
        
        for idx, (i, j, k) in enumerate(sample_indices):
            b_vec[idx] = U_arr[i, col_to_remove] * V_arr[j, col_to_remove] * W_arr[k, col_to_remove]
        
        # Solve least squares
        try:
            alpha, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except:
            continue
        
        # Check residual
        predicted = A_mat @ alpha
        residual_norm = np.linalg.norm(b_vec - predicted)
        b_norm = np.linalg.norm(b_vec)
        
        if b_norm > 1e-10 and residual_norm / b_norm < 1e-8:
            # Column is redundant! Remove it.
            U_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in U]
            V_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in V]
            W_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in W]
            
            # Verify the full decomposition
            if _verify(U_new, V_new, W_new, n, m, p):
                return U_new, V_new, W_new
    
    return None

def construct():
    deadline = time.time() + 135.0
    n, m, p = DIMENSIONS
    
    # Build the decomposition (49 Strassen + 61 cross terms = 110)
    U, V, W = _build_5x5_decomposition()
    
    # Verify
    if _verify(U, V, W, n, m, p):
        print(f"Found valid rank-{len(U[0])} decomposition")
        best_rank = len(U[0])
        best_result = {"U": U, "V": V, "W": W}
    else:
        print("Decomposition verification failed, falling back to schoolbook")
        best_rank = n * m * p
        rank_sb = best_rank
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][column] = 1
                    V_sb[j * p + k][column] = 1
                    W_sb[k * n + i][column] = 1
                    column += 1
        best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try rank reduction
    rng = np.random.RandomState(42)
    
    current_U = best_result["U"]
    current_V = best_result["V"]
    current_W = best_result["W"]
    current_rank = best_rank
    
    max_attempts = 20
    for attempt in range(max_attempts):
        if time.time() > deadline - 20:
            break
        
        target_rank = current_rank - 1
        result = _try_random_rank_reduction(
            current_U, current_V, current_W,
            n, m, p, target_rank, deadline, rng
        )
        
        if result is not None:
            U_new, V_new, W_new = result
            if _verify(U_new, V_new, W_new, n, m, p):
                print(f"Found valid rank-{target_rank} decomposition")
                current_rank = target_rank
                current_U, current_V, current_W = U_new, V_new, W_new
                best_result = {"U": U_new, "V": V_new, "W": W_new}
            else:
                break
        else:
            break
    
    print(f"Best rank found: {current_rank}")
    return best_result
# EVOLVE-BLOCK-END
