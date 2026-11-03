import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=4, p=8

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
        # Compute sum_r Ua[:,r] * Va[:,r] * Wa[:,r] and compare to T
        # Efficient: for each r, outer product of Ua[:,r] and Va[:,r], then contract with Wa
        result = np.zeros((n*m, m*p, p*n))
        for r in range(Ua.shape[1]):
            result += np.einsum('i,j,k->ijk', Ua[:,r], Va[:,r], Wa[:,r])
        return np.allclose(result, T_expected, atol=1e-10)
    
    if not verify(U_arr, V_arr, W_arr):
        # Fallback: return schoolbook rank-64
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
    
    # Attempt to reduce rank by finding linear dependencies among columns
    # Check if any column of U is zero
    for r in range(U_arr.shape[1]):
        if np.allclose(U_arr[:, r], 0):
            U_arr = np.delete(U_arr, r, axis=1)
            V_arr = np.delete(V_arr, r, axis=1)
            W_arr = np.delete(W_arr, r, axis=1)
            break
    
    # Try to merge columns: if two columns r1, r2 have U columns that are linearly dependent
    # or if we can find a change of basis that zeros out entries
    # Simple approach: try to find pairs of columns where U[:,r1] and U[:,r2] are proportional
    # and V[:,r1] and V[:,r2] are proportional, then merge
    
    merged = True
    while merged:
        merged = False
        r1, r2 = -1, -1
        for a in range(U_arr.shape[1]):
            for b in range(a+1, U_arr.shape[1]):
                # Check if U columns are proportional
                ua, ub = U_arr[:, a], U_arr[:, b]
                if np.allclose(ua, 0) or np.allclose(ub, 0):
                    continue
                # Find ratio
                non_zero_a = ua != 0
                if not np.any(non_zero_a):
                    continue
                ratio = ub[non_zero_a] / ua[non_zero_a]
                if np.allclose(ratio, ratio[0], atol=1e-10):
                    # Check V columns proportional with same ratio
                    va, vb = V_arr[:, a], V_arr[:, b]
                    non_zero_v = va != 0
                    if np.any(non_zero_v):
                        ratio_v = vb[non_zero_v] / va[non_zero_v]
                        if np.allclose(ratio_v, ratio[0], atol=1e-10):
                            # Check W columns
                            wa, wb = W_arr[:, a], W_arr[:, b]
                            non_zero_w = wa != 0
                            if np.any(non_zero_w):
                                ratio_w = wb[non_zero_w] / wa[non_zero_w]
                                if np.allclose(ratio_w, ratio[0], atol=1e-10):
                                    # Merge: column a absorbs column b
                                    U_arr[:, a] += ratio[0] * U_arr[:, b]
                                    V_arr[:, a] += ratio[0] * V_arr[:, b]
                                    W_arr[:, a] += ratio[0] * W_arr[:, b]
                                    U_arr = np.delete(U_arr, b, axis=1)
                                    V_arr = np.delete(V_arr, b, axis=1)
                                    W_arr = np.delete(W_arr, b, axis=1)
                                    merged = True
                                    break
            if merged:
                break
    
    # Verify again
    if not verify(U_arr, V_arr, W_arr):
        # Restore original
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
    
    # Convert to integer/half-integer if possible
    def to_rational(arr):
        # Check if all entries are integer or half-integer
        arr2 = arr * 2
        if np.allclose(arr2, np.round(arr2), atol=1e-10):
            return np.round(arr2) / 2
        return arr
    
    U_final = to_rational(U_arr)
    V_final = to_rational(V_arr)
    W_final = to_rational(W_arr)
    
    # Final verification
    if not verify(U_final, V_final, W_final):
        # Use the original rank-56
        return {"U": U, "V": V, "W": W}
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
