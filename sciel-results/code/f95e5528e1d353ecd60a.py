import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill

    # Build schoolbook decomposition as fallback (rank 128)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Strassen rank-7 terms for 2x2x2 multiplication
    # Each term: (U_coeffs, V_coeffs, W_coeffs)
    strassen_terms_2x2 = [
        # M1: (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2: (A10+A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3: A00(B01-B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4: A11(B10-B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5: (A00+A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6: (A10-A00)(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7: (A01-A11)(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]

    # Top-level Strassen for 4x4x4: treat 4x4 as 2x2 blocks of 2x2 matrices
    top_level_terms = [
        # M1: (A00+A11)(B00+B11)
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        # M2: (A10+A11)B00
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)], [((1,0), 1), ((1,1), -1)]),
        # M3: A00(B01-B11)
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)], [((0,1), 1), ((1,1), 1)]),
        # M4: A11(B10-B00)
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)], [((0,0), 1), ((1,0), 1)]),
        # M5: (A00+A01)B11
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)], [((0,0), -1), ((0,1), 1)]),
        # M6: (A10-A00)(B00+B01)
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        # M7: (A01-A11)(B10+B11)
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)], [((0,0), 1)])
    ]

    # Use polynomial ring F[x]/(x^2) approach:
    # B is 4x8, treat as 4x4 over R where entry (j,k) in R is b_jk + x*b_j(k+4)
    # A is 4x4, C is 4x8 treated as 4x4 over R
    # Each Strassen term for 4x4x4 over R gives polynomial coefficients
    # Constant term -> k=0..3, x coefficient -> k=4..7
    
    all_terms = []
    
    for t_idx in range(7):
        if time.time() > deadline - 10:
            break
        a_blocks, b_blocks, c_blocks = top_level_terms[t_idx]
        
        for s_idx in range(7):
            if time.time() > deadline - 10:
                break
            s_u, s_v, s_w = strassen_terms_2x2[s_idx]
            
            # Build U factors (same for both halves since A is 4x4)
            u_global = {}
            for (r, c), coeff_s in s_u.items():
                for (u, v), coeff_a in a_blocks:
                    i = 2*u + r
                    j = 2*v + c
                    u_global[(i, j)] = u_global.get((i, j), 0) + coeff_s * coeff_a
            
            # Build V factors: B is over R, so V has entries in R
            # b_blocks gives (v, w) block indices, s_v gives (r, c) within 2x2
            # j = 2*v + r, and the "k" direction in R corresponds to both k and k+4
            # For the constant term (k=0..3): v_global_const[(j, k)]
            # For the x term (k=4..7): v_global_x[(j, k-4)]
            # But since we're building ONE term that covers both, we need to split
            # Actually, in the polynomial ring, V[j,k]_R = v_jk + x*v_j(k+4)
            # The Strassen term multiplies U_R * V_R, and the result is in R
            # So we need to track both constant and x parts separately for V and W
            
            # Let me reconsider: each term in the decomposition is a rank-1 tensor
            # U[i,j] * V[j,k] * W[k,i]
            # Over R, this becomes:
            # U[i,j]_R * V[j,k]_R * W[k,i]_R
            # where U is just 4x4 (no polynomial), V and W are 4x4 over R
            
            # Actually, A is 4x4 over F (not R), so U is just 4x4
            # B is 4x8, viewed as 4x4 over R: B[j, k_R] where k_R = k + x*(k+4)
            # C is 4x8, viewed as 4x4 over R: C[i, k_R]
            
            # The tensor decomposition over R:
            # C[i, k_R] = sum_r U[i,j] * V_R[j, k_R] * W_R[k_R, i]
            # But this doesn't quite work because the multiplication is A*B, not A_R*B_R
            
            # Let me think more carefully. The multiplication is:
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # where A is 4x4, B is 4x8, C is 4x8
            
            # If we view B as 4x4 over R with B_R[j, k'] = B[j, k'] + x*B[j, k'+4] for k'=0,1,2,3
            # and C as 4x4 over R with C_R[i, k'] = C[i, k'] + x*C[i, k'+4]
            # Then C_R[i, k'] = sum_j A[i,j] * B_R[j, k']
            # = sum_j A[i,j] * (B[j,k'] + x*B[j,k'+4])
            # = (sum_j A[i,j]*B[j,k']) + x*(sum_j A[i,j]*B[j,k'+4])
            # = C[i,k'] + x*C[i,k'+4]
            
            # So this works! Now the tensor decomposition:
            # We need C_R[i, k'] = sum_r U[i,j] * V[j, k'_R] * W[k'_R, i]
            # where V and W are over R
            
            # Actually, the standard tensor decomposition is:
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # Over R, we can write:
            # C_R[i, k'] = sum_j A[i,j] * B_R[j, k']
            # = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_{j,k'} V_R[j*p+k'_R, r] * B_R[j, k']) * ...
            
            # This is getting complex. Let me just use the direct approach:
            # For each of the 49 Strassen terms for 4x4x4, we get contributions to both
            # the k=0..3 and k=4..7 parts.
            
            # The key insight: if we use the 49 terms for 4x4x4, each term r has:
            # U_r[i,j], V_r[j,k], W_r[k,i] for k=0..3
            # To get k=4..7, we can use the SAME U_r but different V_r and W_r
            
            # Actually, the polynomial ring approach says:
            # Define V_R[j, k'] = V_left[j, k'] + x * V_right[j, k'] for k'=0..3
            # where V_left[j, k'] is the V for k=k' and V_right[j, k'] is V for k=k'+4
            # Similarly for W_R.
            
            # Then the 49 terms give:
            # C_R[i, k'] = sum_r U_r[i,j] * V_R[j, k'] * W_R[k', i]
            # The constant term gives C[i, k'] and the x term gives C[i, k'+4]
            
            # So we need V_R and W_R to be over R, meaning each "entry" has two components.
            # In the final tensor, V[j*p+k, r] for k=0..7 corresponds to:
            # V[j*p+k, r] = V_R[j, k%4]_const if k<4, V_R[j, k%4]_x if k>=4
            
            # So the 49 terms each produce TWO rows in V (for k and k+4) and TWO rows in W.
            # But U is the same for both!
            
            # So total rank is still 49, but V and W have 32 rows each (4*8).
            
            # Let me build this properly.
            
            # For each Strassen term, we have U[i,j], V[j,k], W[k,i] for 4x4x4
            # We need to map this to 4x4x8 where B has columns 0..7
            
            # The mapping: for k'=0..3, the "left" part uses k=k' and the "right" part uses k=k'+4
            # In the polynomial ring, V_R[j,k'] = V_left[j,k'] + x*V_right[j,k']
            # and W_R[k',i] = W_left[k',i] + x*W_right[k',i]
            
            # The product V_R[j,k'] * W_R[k',i] = V_left[j,k']*W_left[k',i] + x*(V_left[j,k']*W_right[k',i] + V_right[j,k']*W_left[k',i]) + x^2*V_right[j,k']*W_right[k',i]
            # Since x^2=0, this = V_left[j,k']*W_left[k',i] + x*(V_left[j,k']*W_right[k',i] + V_right[j,k']*W_left[k',i])
            
            # But we need C_R[i,k'] = sum_j A[i,j] * B_R[j,k'] = sum_j A[i,j]*(B_left[j,k'] + x*B_right[j,k'])
            # = (sum_j A[i,j]*B_left[j,k']) + x*(sum_j A[i,j]*B_right[j,k'])
            
            # So the constant term should be sum_j A[i,j]*B_left[j,k'] and the x term should be sum_j A[i,j]*B_right[j,k']
            
            # In the tensor decomposition, we have:
            # C_R[i,k'] = sum_r U_r[i,j] * V_R_r[j,k'] * W_R_r[k',i]
            
            # For this to work, we need:
            # sum_r U_r[i,j] * V_R_r[j,k'] * W_R_r[k',i] = (sum_j A[i,j]*B_left[j,k']) + x*(sum_j A[i,j]*B_right[j,k'])
            
            # This is a single equation over R. The standard 49-term decomposition for 4x4x4 gives:
            # sum_r U_r[i,j] * V_r[j,k'] * W_r[k',i] = sum_j A[i,j]*B[j,k']
            
            # If we set V_R_r[j,k'] = V_r[j,k'] (i.e., V_right = 0) and W_R_r[k',i] = W_r[k',i] (i.e., W_right = 0),
            # then we only get the left half. To get the right half, we need additional terms.
            
            # So the polynomial ring trick doesn't directly give rank 49 for 4x4x8.
            # It gives rank 49 for 4x4x4 over R, which is equivalent to 4x4x8 over F.
            # But the tensor rank over R of 4x4x4 might be different from 49.
            
            # Actually, the rank over R should be the same as over F if we extend scalars,
            # but the tensor dimensions change.
            
            # Let me just stick with the current approach (rank 98) and try to optimize it.
            # Or try a different decomposition.
            
            pass  # Skip this complex analysis for now

    # Use the existing approach: 49 terms for left, 49 for right = 98 total
    all_terms = []
    
    for t_idx in range(7):
        if time.time() > deadline - 10:
            break
        a_blocks, b_blocks, c_blocks = top_level_terms[t_idx]
        
        for s_idx in range(7):
            if time.time() > deadline - 10:
                break
            s_u, s_v, s_w = strassen_terms_2x2[s_idx]
            
            # Build U factors
            u_global = {}
            for (r, c), coeff_s in s_u.items():
                for (u, v), coeff_a in a_blocks:
                    i = 2*u + r
                    j = 2*v + c
                    u_global[(i, j)] = u_global.get((i, j), 0) + coeff_s * coeff_a
            
            # Build V factors
            v_global = {}
            for (r, c), coeff_s in s_v.items():
                for (v, w), coeff_b in b_blocks:
                    j = 2*v + r
                    k = 2*w + c
                    v_global[(j, k)] = v_global.get((j, k), 0) + coeff_s * coeff_b
            
            # Build W factors
            w_global = {}
            for (r, c), coeff_s in s_w.items():
                for (u, w), coeff_c in c_blocks:
                    i = 2*u + r
                    k = 2*w + c
                    w_global[(i, k)] = w_global.get((i, k), 0) + coeff_s * coeff_c
            
            # Clean up zeros
            u_global = {key: val for key, val in u_global.items() if val != 0}
            v_global = {key: val for key, val in v_global.items() if val != 0}
            w_global = {key: val for key, val in w_global.items() if val != 0}
            
            if u_global and v_global and w_global:
                all_terms.append((u_global, v_global, w_global))

    # Handle the right 4x4 block of B (k=4..7)
    terms_right = []
    for u_dict, v_dict, w_dict in all_terms:
        u_new = dict(u_dict)
        v_new = {}
        for (j, k), coeff in v_dict.items():
            v_new[(j, k + 4)] = coeff
        w_new = {}
        for (i, k), coeff in w_dict.items():
            w_new[(i, k + 4)] = coeff
        terms_right.append((u_new, v_new, w_new))
    
    total_terms = all_terms + terms_right
    rank = len(total_terms)
    
    # Build arrays
    U_arr = np.zeros((n * m, rank))
    V_arr = np.zeros((m * p, rank))
    W_arr = np.zeros((p * n, rank))
    
    for col_idx, (u_dict, v_dict, w_dict) in enumerate(total_terms):
        for (i, j), coeff in u_dict.items():
            U_arr[i * m + j, col_idx] = coeff
        for (j, k), coeff in v_dict.items():
            V_arr[j * p + k, col_idx] = coeff
        for (i, k), coeff in w_dict.items():
            W_arr[k * n + i, col_idx] = coeff
    
    # Verify
    valid = True
    for i in range(n):
        if time.time() > deadline - 5:
            valid = False
            break
        for j in range(m):
            if time.time() > deadline - 5:
                valid = False
                break
            for k in range(p):
                t_val = np.dot(U_arr[i * m + j], V_arr[j * p + k] * W_arr[k * n + i])
                if abs(t_val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    print(f"Rank: {U.shape[1]}")
