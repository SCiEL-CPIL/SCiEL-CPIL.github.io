import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build the target tensor T of shape (dim_u, dim_v, dim_w)
    # T[a, b, c] = 1 if indices are consistent, 0 otherwise
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.int8)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Fallback: Schoolbook decomposition (guaranteed valid, rank 112)
    def get_schoolbook():
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

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Strategy: Greedy Peeling
    # We attempt to decompose T into a sum of rank-1 tensors u_k v_k w_k.
    # We restrict u, v, w to have entries in {-1, 0, 1} to keep the search space manageable
    # and to encourage sparse, structured solutions.
    # If we successfully peel until R=0, we have a valid decomposition.
    # The rank is the number of terms.
    
    # We start with R = T
    R = T.copy().astype(np.float64)
    
    U_list = []
    V_list = []
    W_list = []
    
    max_terms = n * m * p  # We don't expect to do worse than schoolbook
    
    # Helper to check if R is zero
    def is_zero(tensor):
        return np.all(tensor == 0)

    # We will try to find a sequence of rank-1 terms that sum to T.
    # This is equivalent to finding a decomposition.
    
    # To improve success rate, we can use a randomized restart strategy if stuck.
    # But first, let's try a deterministic-ish greedy approach with sign updates.
    
    # However, greedy peeling can get stuck in local minima.
    # A better approach for small sizes might be to search for the rank directly.
    # But rank 112 is high.
    
    # Let's stick to the peeling idea but with a twist:
    # Instead of just peeling T, we can try to peel a modified tensor or use a different initialization.
    
    # Actually, the inspiration 2 uses Strassen.
    # Strassen for 4x4x4 is rank 49.
    # The remaining part is 4x4x3.
    # If we can decompose the 4x4x3 part efficiently, we can beat 112.
    
    # Let's try to construct a solution based on Strassen for the first 4 columns of B,
    # and schoolbook for the last 3 columns.
    # Rank = 49 + (4 * 4 * 3) = 49 + 48 = 97.
    # This is a structural improvement over 112.
    
    # Let's implement this hybrid approach directly.
    # It is guaranteed to be valid if implemented correctly.
    
    # Part 1: 4x4x4 (Strassen)
    # Part 2: 4x4x3 (Schoolbook)
    
    # Let's build this manually.
    
    U_final = [[0.0] * 97 for _ in range(n * m)]
    V_final = [[0.0] * 97 for _ in range(m * p)]
    W_final = [[0.0] * 97 for _ in range(p * n)]
    
    current_rank = 0
    
    # --- Part 1: Strassen for 4x4x4 ---
    # This handles B columns 0-3.
    # C rows 0-3.
    # A rows 0-3, cols 0-3.
    
    # Strassen terms for 4x4x4.
    # We can view 4x4 as 2x2 blocks of 2x2.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
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
    
    # We need to map these 2x2 block operations to the 4x4 indices.
    # Block (bi, bj) corresponds to rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
    
    # Let's define the linear combinations for U, V, W for each Strassen term.
    # There are 7 Strassen terms for the 4x4x4 block.
    # But wait, Strassen reduces 8 multiplications to 7.
    # The standard Strassen is for 2x2x2.
    # For 4x4x4, we apply Strassen recursively or just use the 2x2 block structure.
    # If we use 2x2 blocks, we have 4x4 = 2x2 of 2x2 matrices.
    # So we need 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication can be done with 7 scalar multiplications (Strassen 2x2x2).
    # So total rank = 7 * 7 = 49.
    
    # Let's generate the 49 terms.
    
    # Strassen 2x2x2 terms (u, v, w)
    # u, v, w are vectors of length 4 (for 2x2 matrix indices).
    # Index mapping: (r, c) -> r*2 + c
    
    s_u = []
    s_v = []
    s_w = []
    
    # M1: (A00+A11)*(B00+B11)
    # u: A00(1), A11(1) -> indices 0, 3
    # v: B00(1), B11(1) -> indices 0, 3
    # w: C00(1), C11(1) -> indices 0, 3
    s_u.append([1, 0, 0, 1])
    s_v.append([1, 0, 0, 1])
    s_w.append([1, 0, 0, 1])
    
    # M2: (A10+A11)*B00
    # u: A10(1), A11(1) -> indices 2, 3
    # v: B00(1) -> index 0
    # w: C10(1) -> index 2
    s_u.append([0, 0, 1, 1])
    s_v.append([1, 0, 0, 0])
    s_w.append([0, 0, 1, 0])
    
    # M3: A00*(B01-B11)
    # u: A00(1) -> index 0
    # v: B01(1), B11(-1) -> indices 1, 3
    # w: C01(1) -> index 1
    s_u.append([1, 0, 0, 0])
    s_v.append([0, 1, 0, -1])
    s_w.append([0, 1, 0, 0])
    
    # M4: A11*(B10-B00)
    # u: A11(1) -> index 3
    # v: B10(1), B00(-1) -> indices 2, 0
    # w: C10(1), C00(1) -> indices 2, 0
    s_u.append([0, 0, 0, 1])
    s_v.append([-1, 0, 1, 0])
    s_w.append([1, 0, 1, 0])
    
    # M5: (A00+A01)*B11
    # u: A00(1), A01(1) -> indices 0, 1
    # v: B11(1) -> index 3
    # w: C11(-1), C00(-1) -> indices 3, 0 (Note: C00 = M1+M4-M5+M7, so M5 coeff is -1. C11 = M1-M2+M3+M6, so M5 coeff is 0? Wait.)
    # Let's re-verify Strassen formulas.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So for M5:
    # w: C00(-1), C01(1) -> indices 0, 1
    s_u.append([1, 1, 0, 0])
    s_v.append([0, 0, 0, 1])
    s_w.append([-1, 1, 0, 0])
    
    # M6: (A10-A00)*(B00+B01)
    # u: A10(1), A00(-1) -> indices 2, 0
    # v: B00(1), B01(1) -> indices 0, 1
    # w: C11(1) -> index 3
    s_u.append([-1, 0, 1, 0])
    s_v.append([1, 1, 0, 0])
    s_w.append([0, 0, 0, 1])
    
    # M7: (A01-A11)*(B10+B11)
    # u: A01(1), A11(-1) -> indices 1, 3
    # v: B10(1), B11(1) -> indices 2, 3
    # w: C00(1) -> index 0
    s_u.append([0, 1, 0, -1])
    s_v.append([0, 0, 1, 1])
    s_w.append([1, 0, 0, 0])
    
    # Now, expand these 2x2 terms to 4x4.
    # For each of the 7 high-level Strassen terms (mh, vh, wh),
    # we expand them using the 7 low-level Strassen terms (mu, mv, mw).
    # Total 49 terms.
    
    for h_idx in range(7):
        u_h = s_u[h_idx]
        v_h = s_v[h_idx]
        w_h = s_w[h_idx]
        
        for l_idx in range(7):
            u_l = s_u[l_idx]
            v_l = s_v[l_idx]
            w_l = s_w[l_idx]
            
            # Create full 16-dim vectors for U, V, W
            U_vec = [0.0] * 16
            V_vec = [0.0] * 28
            W_vec = [0.0] * 28
            
            # Expand U (4x4)
            # u_h is 2x2 block coefficients.
            # u_l is 2x2 intra-block coefficients.
            for bi in range(2):
                for bj in range(2):
                    coeff_h = u_h[bi*2 + bj]
                    if coeff_h == 0: continue
                    for li in range(2):
                        for lj in range(2):
                            coeff_l = u_l[li*2 + lj]
                            if coeff_l == 0: continue
                            i = 2*bi + li
                            j = 2*bj + lj
                            idx = i*4 + j
                            U_vec[idx] += coeff_h * coeff_l
            
            # Expand V (4x4, but B is 4x7, so only first 4 cols used in this part)
            # v_h is 2x2 block coefficients.
            # v_l is 2x2 intra-block coefficients.
            for bi in range(2):
                for bj in range(2):
                    coeff_h = v_h[bi*2 + bj]
                    if coeff_h == 0: continue
                    for li in range(2):
                        for lj in range(2):
                            coeff_l = v_l[li*2 + lj]
                            if coeff_l == 0: continue
                            i = 2*bi + li
                            j = 2*bj + lj
                            # B index: i * 7 + j
                            idx = i*7 + j
                            V_vec[idx] += coeff_h * coeff_l
            
            # Expand W (4x4)
            # w_h is 2x2 block coefficients.
            # w_l is 2x2 intra-block coefficients.
            for bi in range(2):
                for bj in range(2):
                    coeff_h = w_h[bi*2 + bj]
                    if coeff_h == 0: continue
                    for li in range(2):
                        for lj in range(2):
                            coeff_l = w_l[li*2 + lj]
                            if coeff_l == 0: continue
                            i = 2*bi + li
                            k = 2*bj + lj
                            # W index: k * 4 + i
                            idx = k*4 + i
                            W_vec[idx] += coeff_h * coeff_l
            
            # Assign to current_rank column
            for i in range(16):
                U_final[i][current_rank] = U_vec[i]
            for j in range(28):
                V_final[j][current_rank] = V_vec[j]
            for k in range(28):
                W_final[k][current_rank] = W_vec[k]
            
            current_rank += 1
            
            if time.time() > deadline:
                return get_schoolbook()
                
    # --- Part 2: Schoolbook for 4x4x3 ---
    # This handles B columns 4-6.
    # A rows 0-3, cols 0-3.
    # C rows 0-3, cols 4-6.
    
    # For each i (0-3), j (0-3), k (4-6)
    # Term: A[i,j] * B[j,k] -> C[i,k]
    
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                # U: A[i,j] -> index i*4 + j
                U_final[i*4 + j][current_rank] = 1.0
                # V: B[j,k] -> index j*7 + k
                V_final[j*7 + k][current_rank] = 1.0
                # W: C[i,k] -> index k*4 + i
                W_final[k*4 + i][current_rank] = 1.0
                current_rank += 1
                
                if time.time() > deadline:
                    return get_schoolbook()
    
    # Verify the solution
    # Sum of U[:,t] V[:,t] W[:,t] should equal T
    # We can do a quick check on a few random indices or the whole thing if fast enough.
    # 97 terms, 16*28*28 = 12544 elements.
    # 97 * 12544 ~ 1.2M ops. Fast.
    
    T_check = np.zeros((16, 28, 28), dtype=np.float64)
    for t in range(current_rank):
        U_t = np.array(U_final)[:, t]
        V_t = np.array(V_final)[:, t]
        W_t = np.array(W_final)[:, t]
        T_check += np.einsum('a,b,c->abc', U_t, V_t, W_t)
        
    if not np.allclose(T_check, T):
        # Fallback if construction was flawed
        return get_schoolbook()
    
    # Return the constructed solution
    # Rank is 97.
    # This is better than 112.
    # Score = 112 / 97 > 1.0
    
    return {
        "U": U_final,
        "V": V_final,
        "W": W_final
    }

# EVOLVE-BLOCK-END
