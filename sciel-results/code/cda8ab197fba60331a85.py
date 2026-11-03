import time
import numpy as np

DIMENSIONS = [2, 4, 8]
DEADLINE = 145  # safety margin before 150s kill

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    def time_left():
        return DEADLINE - (time.time() - start_time)

    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    # We seek U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_r U[ia,r]*V[jb,r]*W[kc,r] = T[ia, jb, kc]
    
    # Start with the rank-56 Strassen tiling, then attempt column merging.
    
    # Strassen 2x2 local factorization (rank 7).
    U_loc = [[0]*7 for _ in range(4)]
    V_loc = [[0]*7 for _ in range(4)]
    W_loc = [[0]*7 for _ in range(4)]
    
    U_loc[0][0]=1; U_loc[3][0]=1
    V_loc[0][0]=1; V_loc[3][0]=1
    W_loc[0][0]=1; W_loc[3][0]=1
    U_loc[2][1]=1; U_loc[3][1]=1
    V_loc[0][1]=1
    W_loc[2][1]=1; W_loc[3][1]=-1
    U_loc[0][2]=1
    V_loc[1][2]=1; V_loc[3][2]=-1
    W_loc[1][2]=1; W_loc[3][2]=1
    U_loc[3][3]=1
    V_loc[2][3]=1; V_loc[0][3]=-1
    W_loc[0][3]=1; W_loc[2][3]=1
    U_loc[0][4]=1; U_loc[1][4]=1
    V_loc[3][4]=1
    W_loc[0][4]=-1; W_loc[1][4]=1
    U_loc[2][5]=1; U_loc[0][5]=-1
    V_loc[0][5]=1; V_loc[1][5]=1
    W_loc[3][5]=1
    U_loc[1][6]=1; U_loc[3][6]=-1
    V_loc[2][6]=1; V_loc[3][6]=1
    W_loc[0][6]=1

    total_rank = 56
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]

    group = 0
    for jj in range(p // 2):
        for half in range(2):
            base_r = group * 7
            group += 1
            a_map = [0*m+(2*half+0), 0*m+(2*half+1), 1*m+(2*half+0), 1*m+(2*half+1)]
            b_map = [(2*half+0)*p+(2*jj+0), (2*half+0)*p+(2*jj+1),
                     (2*half+1)*p+(2*jj+0), (2*half+1)*p+(2*jj+1)]
            c_map = [(2*jj+0)*n+0, (2*jj+1)*n+0, (2*jj+0)*n+1, (2*jj+1)*n+1]
            for r in range(7):
                rr = base_r + r
                for la in range(4):
                    val = U_loc[la][r]
                    if val: U[a_map[la]][rr] += val
                for lb in range(4):
                    val = V_loc[lb][r]
                    if val: V[b_map[lb]][rr] += val
                for lc in range(4):
                    val = W_loc[lc][r]
                    if val: W[c_map[lc]][rr] += val

    # Convert to numpy for manipulation
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Verify correctness
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1.0
    
    def verify(Ua, Va, Wa):
        result = np.zeros((n*m, m*p, p*n))
        for r in range(Ua.shape[1]):
            result += np.einsum('i,j,k->ijk', Ua[:,r], Va[:,r], Wa[:,r])
        return np.allclose(result, T_expected, atol=1e-10)
    
    if not verify(U_arr, V_arr, W_arr):
        U_fb = np.zeros((n*m, 64))
        V_fb = np.zeros((m*p, 64))
        W_fb = np.zeros((p*n, 64))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m*p + j*p + k
                    U_fb[i*m+j, idx] = 1
                    V_fb[j*p+k, idx] = 1
                    W_fb[k*n+i, idx] = 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Remove zero columns
    for r in range(U_arr.shape[1] - 1, -1, -1):
        if np.allclose(U_arr[:, r], 0) and np.allclose(V_arr[:, r], 0) and np.allclose(W_arr[:, r], 0):
            U_arr = np.delete(U_arr, r, axis=1)
            V_arr = np.delete(V_arr, r, axis=1)
            W_arr = np.delete(W_arr, r, axis=1)
    
    # Valid rank reduction: try to find a column that can be eliminated
    # by expressing it as a linear combination of others in a way that
    # the tensor contribution cancels.
    # For a tensor decomposition, column r can be removed if there exist
    # coefficients c_s (s != r) such that for ALL (i,j,k):
    #   U[i,r]*V[j,r]*W[k,r] = sum_s c_s * U[i,s]*V[j,s]*W[k,s]
    # This is equivalent to: the rank-1 tensor u_r ⊗ v_r ⊗ w_r is in the
    # span of {u_s ⊗ v_s ⊗ w_s : s != r}.
    
    # More practical approach: try pairwise column elimination.
    # For columns a and b, can we find λ such that replacing column a with
    # column a - λ*column b (in all three matrices) preserves the tensor?
    # This requires: (u_a - λ u_b) ⊗ (v_a - λ v_b) ⊗ (w_a - λ w_b)
    #   = u_a ⊗ v_a ⊗ w_a - λ u_b ⊗ v_b ⊗ w_b
    # Expanding LHS: u_a⊗v_a⊗w_a - λ u_a⊗v_a⊗w_b - λ u_a⊗v_b⊗w_a
    #   + λ u_b⊗v_a⊗w_a - λ u_b⊗v_b⊗w_a + λ² u_a⊗v_b⊗w_b
    #   - λ² u_b⊗v_a⊗w_b + λ³ u_b⊗v_b⊗w_b
    # RHS: u_a⊗v_a⊗w_a - λ u_b⊗v_b⊗w_b
    # So we need: -λ u_a⊗v_a⊗w_b - λ u_a⊗v_b⊗w_a + λ u_b⊗v_a⊗w_a
    #   - λ u_b⊗v_b⊗w_a + λ² u_a⊗v_b⊗w_b - λ² u_b⊗v_a⊗w_b + λ³ u_b⊗v_b⊗w_b
    #   + λ u_b⊗v_b⊗w_b = 0
    # This is complex. Instead, use a simpler valid operation:
    # If u_a is a scalar multiple of u_b, AND v_a is a scalar multiple of v_b,
    # AND w_a is a scalar multiple of w_b, with the SAME scalar, then
    # column a is redundant (contributes the same as column b scaled).
    # Actually that's not right either - they'd add, not cancel.
    
    # The simplest valid elimination: if column r is entirely zero in one matrix
    # (say U), it contributes nothing and can be removed.
    
    # Another valid approach: if for some r, there exists s such that
    # u_r = α u_s, v_r = β v_s, w_r = γ w_s with αβγ = 1,
    # then we can potentially merge. But this is rare in Strassen.
    
    # For the Strassen tiling, the 8 groups are independent. Within each group,
    # rank-7 is minimal for 2x2x2. So rank 56 is likely optimal for this structure.
    # Let's verify the rank is truly 56 and try a more general elimination.
    
    # General column elimination: try removing each column and see if the
    # remaining columns can still span the tensor (via least squares).
    # This is expensive but let's try for a few candidates.
    
    if time_left() > 30:
        # Try to find columns that are linearly dependent in the tensor sense
        # For each pair (a,b), check if u_a⊗v_a⊗w_a can be expressed as
        # a linear combination of other columns.
        # Practical shortcut: check if u_a is in span of other u's,
        # v_a in span of other v's, w_a in span of other w's simultaneously.
        # This is sufficient but not necessary.
        
        r = U_arr.shape[1]
        for col in range(r):
            if time_left() < 10:
                break
            # Try to eliminate column 'col'
            if col >= U_arr.shape[1]:
                continue
            # Check if U[:,col] is in span of other U columns
            other_idx = [i for i in range(U_arr.shape[1]) if i != col]
            if not other_idx:
                continue
            U_other = U_arr[:, other_idx]
            # Check rank
            if np.linalg.matrix_rank(U_other) < np.linalg.matrix_rank(np.column_stack([U_other, U_arr[:, col]])):
                # U[:,col] not in span, can't eliminate
                continue
            # Similarly check V and W
            V_other = V_arr[:, other_idx]
            if np.linalg.matrix_rank(V_other) < np.linalg.matrix_rank(np.column_stack([V_other, V_arr[:, col]])):
                continue
            W_other = W_arr[:, other_idx]
            if np.linalg.matrix_rank(W_other) < np.linalg.matrix_rank(np.column_stack([W_other, W_arr[:, col]])):
                continue
            # All three are in their respective spans. Now check if the tensor
            # contribution can actually be reproduced.
            # Solve for coefficients
            try:
                coeffs_U, _, _, _ = np.linalg.lstsq(U_other, U_arr[:, col], rcond=None)
                coeffs_V, _, _, _ = np.linalg.lstsq(V_other, V_arr[:, col], rcond=None)
                coeffs_W, _, _, _ = np.linalg.lstsq(W_other, W_arr[:, col], rcond=None)
            except:
                continue
            
            # Check if these coefficients give the same tensor
            # u_col ⊗ v_col ⊗ w_col should equal sum_s c_U[s]*c_V[s]*c_W[s] * u_s ⊗ v_s ⊗ w_s
            # This is a strong condition. Let's verify numerically.
            new_U = np.delete(U_arr, col, axis=1)
            new_V = np.delete(V_arr, col, axis=1)
            new_W = np.delete(W_arr, col, axis=1)
            
            # Compute the tensor with the new columns plus the "virtual" contribution
            # Actually, we need to check if the tensor is preserved.
            # The contribution of column col is u_col⊗v_col⊗w_col.
            # We need: sum_s (c_U[s]*c_V[s]*c_W[s]) * u_s⊗v_s⊗w_s = u_col⊗v_col⊗w_col
            # This is generally NOT satisfied just from individual spans.
            # So this approach is too weak. Skip.
            pass
    
    # Convert to integer/half-integer if possible
    def to_rational(arr):
        arr2 = arr * 2
        if np.allclose(arr2, np.round(arr2), atol=1e-10):
            return np.round(arr2) / 2
        return arr
    
    U_final = to_rational(U_arr)
    V_final = to_rational(V_arr)
    W_final = to_rational(W_arr)
    
    # Final verification
    if not verify(U_final, V_final, W_final):
        return {"U": U, "V": V, "W": W}
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
