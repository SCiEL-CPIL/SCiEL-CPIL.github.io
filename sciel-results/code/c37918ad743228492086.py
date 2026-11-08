import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 145  # seconds, safety margin from 150s

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_baseline = 50
    U = [[0] * rank_baseline for _ in range(n * m)]
    V = [[0] * rank_baseline for _ in range(m * p)]
    W = [[0] * rank_baseline for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h)
        U[a_idx][col] += 1; U[d_idx][col] += 1
        V[e_idx][col] += 1; V[h_idx][col] += 1
        W[c00_idx][col] += 1; W[c11_idx][col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[c_idx][col] += 1; U[d_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1; W[c11_idx][col] -= 1
        col += 1
        
        # M3 = a(f-h)
        U[a_idx][col] += 1
        V[f_idx][col] += 1; V[h_idx][col] -= 1
        W[c01_idx][col] += 1; W[c11_idx][col] += 1
        col += 1
        
        # M4 = d(g-e)
        U[d_idx][col] += 1
        V[g_idx][col] += 1; V[e_idx][col] -= 1
        W[c00_idx][col] += 1; W[c10_idx][col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[a_idx][col] += 1; U[b_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] -= 1; W[c01_idx][col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[c_idx][col] += 1; U[a_idx][col] -= 1
        V[e_idx][col] += 1; V[f_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[b_idx][col] += 1; U[d_idx][col] -= 1
        V[g_idx][col] += 1; V[h_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        U[a_idx][col] += 1; V[e_idx][col] += 1; W[c00_idx][col] += 1
        col += 1
        U[b_idx][col] += 1; V[g_idx][col] += 1; W[c00_idx][col] += 1
        col += 1
        U[c_idx][col] += 1; V[e_idx][col] += 1; W[c10_idx][col] += 1
        col += 1
        U[d_idx][col] += 1; V[g_idx][col] += 1; W[c10_idx][col] += 1
        col += 1

    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)
    
    assert col == rank_baseline, f"Expected rank {rank_baseline}, got {col}"
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    def verify_decomposition(U_arr, V_arr, W_arr):
        n_val, m_val, p_val = n, m, p
        T_expected = np.zeros((n_val * m_val, m_val * p_val, p_val * n_val))
        for i in range(n_val):
            for j in range(m_val):
                for k in range(p_val):
                    a_idx = i * m_val + j
                    b_idx = j * p_val + k
                    c_idx = k * n_val + i
                    T_expected[a_idx, b_idx, c_idx] = 1
        
        T_decomposed = np.zeros((n_val * m_val, m_val * p_val, p_val * n_val))
        for r in range(U_arr.shape[1]):
            u_col = U_arr[:, r]
            v_col = V_arr[:, r]
            w_col = W_arr[:, r]
            T_decomposed += np.einsum('a,b,c->abc', u_col, v_col, w_col)
        
        return np.allclose(T_decomposed, T_expected, atol=1e-10)
    
    assert verify_decomposition(U_np, V_np, W_np), "Baseline decomposition is incorrect!"
    
    best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
    best_rank = rank_baseline
    
    # Aggressive column removal search
    # We try to remove pairs of columns (one from each j-block) to see if rank 48 is possible
    # Or just single columns to get 49
    
    current_U, current_V, current_W = U_np, V_np, W_np
    current_rank = rank_baseline
    
    # Try removing columns one by one, restarting from 0 each time a removal succeeds
    # This allows finding multiple removable columns
    removed = []
    while True:
        found_removal = False
        for r in range(current_U.shape[1]):
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
            U_test = np.delete(current_U, r, axis=1)
            V_test = np.delete(current_V, r, axis=1)
            W_test = np.delete(current_W, r, axis=1)
            if verify_decomposition(U_test, V_test, W_test):
                current_U, current_V, current_W = U_test, V_test, W_test
                current_rank -= 1
                found_removal = True
                break
        if not found_removal:
            break
    
    best_U, best_V, best_W = current_U, current_V, current_W
    best_rank = current_rank
    
    # Try to find a 2-column removal (rank reduction by 2)
    # This is more expensive but might find the rank 48 solution
    # We only do this if we have time and haven't reached a very low rank
    if best_rank > 40 and time.time() - START_TIME < TIME_LIMIT - 30:
        # Try removing pairs of columns
        # To save time, we only try pairs that are "similar" or in the same block
        # But for now, we'll just try a limited number of random pairs or structured pairs
        # Given the structure, removing one from j=0 block and one from j=1 block is promising
        
        # Let's try removing one column from the first half (indices 0-24) and one from the second half (25-49)
        # Note: The indices might have shifted if we removed some already.
        # We'll just try all pairs if rank is small enough, otherwise sample.
        
        rank = best_U.shape[1]
        if rank <= 50:
            for i in range(rank):
                if time.time() - START_TIME > TIME_LIMIT - 15:
                    break
                for j in range(i + 1, rank):
                    U_test = np.delete(best_U, [i, j], axis=1)
                    V_test = np.delete(best_V, [i, j], axis=1)
                    W_test = np.delete(best_W, [i, j], axis=1)
                    if verify_decomposition(U_test, V_test, W_test):
                        best_U, best_V, best_W = U_test, V_test, W_test
                        best_rank -= 2
                        break
                if best_rank < rank - 1:
                    break

    U_final = best_U.tolist()
    V_final = best_V.tolist()
    W_final = best_W.tolist()
    
    for arr in [U_final, V_final, W_final]:
        for row in arr:
            for idx in range(len(row)):
                val = row[idx]
                if abs(val - round(val)) < 1e-10:
                    row[idx] = int(round(val))
                elif abs(val * 2 - round(val * 2)) < 1e-10:
                    row[idx] = round(val * 2) / 2
    
    return {"U": U_final, "V": V_final, "W": W_final}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
