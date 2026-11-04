"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Build the rank-42 Strassen decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    def add_strassen_block(a_idx, b_idx, w_idx, start_col):
        c = start_col
        # M1 = (a00+a11)(b00+b11)
        U[a_idx[0]][c] += 1; U[a_idx[3]][c] += 1
        V[b_idx[0]][c] += 1; V[b_idx[3]][c] += 1
        W[w_idx[0]][c] += 1; W[w_idx[3]][c] += 1
        # M2 = (a10+a11)b00
        U[a_idx[2]][c+1] += 1; U[a_idx[3]][c+1] += 1
        V[b_idx[0]][c+1] += 1
        W[w_idx[2]][c+1] += 1; W[w_idx[3]][c+1] -= 1
        # M3 = a00(b01-b11)
        U[a_idx[0]][c+2] += 1
        V[b_idx[1]][c+2] += 1; V[b_idx[3]][c+2] -= 1
        W[w_idx[1]][c+2] += 1; W[w_idx[3]][c+2] += 1
        # M4 = a11(b10-b00)
        U[a_idx[3]][c+3] += 1
        V[b_idx[2]][c+3] += 1; V[b_idx[0]][c+3] -= 1
        W[w_idx[0]][c+3] += 1; W[w_idx[2]][c+3] += 1
        # M5 = (a00+a01)b11
        U[a_idx[0]][c+4] += 1; U[a_idx[1]][c+4] += 1
        V[b_idx[3]][c+4] += 1
        W[w_idx[0]][c+4] -= 1; W[w_idx[1]][c+4] += 1
        # M6 = (a10-a00)(b00+b01)
        U[a_idx[2]][c+5] += 1; U[a_idx[0]][c+5] -= 1
        V[b_idx[0]][c+5] += 1; V[b_idx[1]][c+5] += 1
        W[w_idx[3]][c+5] += 1
        # M7 = (a01-a11)(b10+b11)
        U[a_idx[1]][c+6] += 1; U[a_idx[3]][c+6] -= 1
        V[b_idx[2]][c+6] += 1; V[b_idx[3]][c+6] += 1
        W[w_idx[0]][c+6] += 1

    a_L = [0, 1, 4, 5]
    a_R = [2, 3, 6, 7]
    
    b_1_T = [0, 1, 6, 7]
    b_1_B = [12, 13, 18, 19]
    w_1 = [0, 2, 1, 3]
    
    add_strassen_block(a_L, b_1_T, w_1, 0)
    add_strassen_block(a_R, b_1_B, w_1, 7)
    
    b_2_T = [2, 3, 8, 9]
    b_2_B = [14, 15, 20, 21]
    w_2 = [4, 6, 5, 7]
    
    add_strassen_block(a_L, b_2_T, w_2, 14)
    add_strassen_block(a_R, b_2_B, w_2, 21)
    
    b_3_T = [4, 5, 10, 11]
    b_3_B = [16, 17, 22, 23]
    w_3 = [8, 10, 9, 11]
    
    add_strassen_block(a_L, b_3_T, w_3, 28)
    add_strassen_block(a_R, b_3_B, w_3, 35)
    
    # Verify the decomposition
    def verify_decomp(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True
    
    if not verify_decomp(U, V, W, rank):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to float arrays
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the tensor matrix: column r = flattened (u_r ⊗ v_r ⊗ w_r)
    flat_terms = []
    for r in range(rank):
        term = np.outer(np.outer(U_np[:, r], V_np[:, r]), W_np[:, r])
        flat_terms.append(term.flatten())
    M = np.array(flat_terms).T  # shape (2304, 42)
    
    # Best result so far
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    def to_list(arr):
        result = []
        for row in arr:
            result.append([])
            for val in row:
                # Round to nearest half-integer
                frac = round(val * 2) / 2
                if abs(frac - round(frac)) < 1e-10:
                    frac = int(round(frac))
                result[-1].append(frac)
        return result
    
    def verify_float(final_U, final_V, final_W, new_rank, tol=1e-7):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(new_rank):
                        total += final_U[i*m+j, r] * final_V[j*p+k, r] * final_W[k*n+i, r]
                    if abs(total - 1.0) > tol:
                        return False
        return True
    
    # Strategy: Use SVD to find the effective rank and eliminate dependent columns
    # The key insight: if the tensor matrix M has rank < 42, we can eliminate columns
    
    def eliminate_dependent_columns(U_in, V_in, W_in, M_in, target_rank, tol=1e-10):
        """Try to reduce the rank to target_rank by eliminating dependent columns."""
        if M_in.shape[1] <= target_rank:
            return U_in, V_in, W_in, M_in.shape[1]
        
        # Compute SVD
        U_sub, s_sub, Vt_sub = np.linalg.svd(M_in, full_matrices=False)
        eff_rank = int(np.sum(s_sub > tol))
        
        if eff_rank >= target_rank:
            # Can't reduce to target_rank with this tolerance
            return None, None, None, M_in.shape[1]
        
        # We can reduce to eff_rank
        # Find null space vectors
        null_space = Vt_sub[eff_rank:, :]
        
        new_U = U_in.copy()
        new_V = V_in.copy()
        new_W = W_in.copy()
        
        # Eliminate columns using null space vectors
        eliminated = set()
        for nv in null_space:
            if time.time() > deadline:
                break
            if len(eliminated) >= M_in.shape[1] - target_rank:
                break
            
            # Find the column with the largest coefficient in this null vector
            max_idx = np.argmax(np.abs(nv))
            max_val = nv[max_idx]
            
            if abs(max_val) < 1e-10:
                continue
            
            # Eliminate this column
            for j in range(len(nv)):
                if j == max_idx or j in eliminated:
                    continue
                coeff = -nv[j] / max_val
                if abs(coeff) > 1e-10:
                    new_U[:, j] += coeff * new_U[:, max_idx]
                    new_V[:, j] += coeff * new_V[:, max_idx]
                    new_W[:, j] += coeff * new_W[:, max_idx]
            
            eliminated.add(max_idx)
        
        if not eliminated:
            return None, None, None, M_in.shape[1]
        
        keep = [c for c in range(M_in.shape[1]) if c not in eliminated]
        new_rank = len(keep)
        
        return new_U[:, keep], new_V[:, keep], new_W[:, keep], new_rank
    
    # Try to eliminate columns
    for target in range(40, 42):
        if time.time() > deadline - 20:
            break
        
        elim_U, elim_V, elim_W, elim_rank = eliminate_dependent_columns(
            U_np, V_np, W_np, M, target
        )
        
        if elim_U is not None and elim_rank < best_rank:
            if verify_float(elim_U, elim_V, elim_W, elim_rank, tol=1e-7):
                best_rank = elim_rank
                best_result = {
                    "U": to_list(elim_U),
                    "V": to_list(elim_V),
                    "W": to_list(elim_W)
                }
    
    # If we couldn't reduce, return the original
    return best_result
# EVOLVE-BLOCK-END
