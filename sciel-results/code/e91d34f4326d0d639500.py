"""Combined program: Strassen-based rank-49 decomposition with greedy column removal."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x2 * 2x14 matrix multiplication.
    
    Strategy:
    1. Build the Strassen-based rank-49 decomposition (7 blocks x 7 Strassen terms).
    2. Apply greedy column removal with exact verification to see if rank can be reduced.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    nm = n * m  # 4
    mp = m * p  # 28
    pn = p * n  # 28

    # --- Build the Strassen-based rank-49 decomposition ---
    def build_strassen():
        num_blocks = p // 2  # 7
        rank = num_blocks * 7  # 49
        
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Strassen linear forms for A (2x2)
        # A indices: A00=0, A01=1, A10=2, A11=3
        U_forms = [
            [1, 0, 0, 1],   # M1 = (A00+A11)
            [0, 0, 1, 1],   # M2 = (A10+A11)
            [1, 0, 0, 0],   # M3 = A00
            [0, 0, 0, 1],   # M4 = A11
            [1, 1, 0, 0],   # M5 = (A00+A01)
            [-1, 0, 1, 0],  # M6 = (A10-A00)
            [0, 1, 0, -1]   # M7 = (A01-A11)
        ]
        
        # V forms for B (2x2 block): B00=0, B01=1, B10=2, B11=3
        V_forms = [
            {0: 1, 3: 1},   # M1: (B00+B11)
            {0: 1},          # M2: B00
            {1: 1, 3: -1},  # M3: (B01-B11)
            {2: 1, 0: -1},  # M4: (B10-B00)
            {3: 1},          # M5: B11
            {0: 1, 1: 1},   # M6: (B00+B01)
            {2: 1, 3: 1}    # M7: (B10+B11)
        ]
        
        # W forms for C (2x2 block): C00=0, C01=1, C10=2, C11=3
        W_forms = [
            {0: 1, 3: 1},   # M1: C00+C11
            {2: 1, 3: -1},  # M2: C10-C11
            {1: 1, 3: 1},   # M3: C01+C11
            {0: 1, 2: 1},   # M4: C00+C10
            {0: -1, 1: 1},  # M5: -C00+C01
            {3: 1},          # M6: C11
            {0: 1}           # M7: C00
        ]
        
        for t in range(num_blocks):
            for s in range(7):
                col = t * 7 + s
                
                # U: coefficients for A
                for i in range(4):
                    U[i, col] = U_forms[s][i]
                
                # V: coefficients for B block t
                # B[j, 2*t + b_col] -> V index = j*p + (2*t + b_col)
                for b_idx, coeff in V_forms[s].items():
                    b_j = b_idx // 2   # 0 or 1
                    b_col = b_idx % 2  # 0 or 1
                    v_row = b_j * p + (2 * t + b_col)
                    V[v_row, col] = coeff
                
                # W: coefficients for C block t
                # C[i, 2*t + c_col] -> W index = (2*t + c_col)*n + i
                for c_idx, coeff in W_forms[s].items():
                    c_i = c_idx // 2   # 0 or 1
                    c_col = c_idx % 2  # 0 or 1
                    w_row = (2 * t + c_col) * n + c_i
                    W[w_row, col] = coeff
        
        return U, V, W

    U, V, W = build_strassen()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W

    # --- Fast tensor verification ---
    def tensor_check(Uf, Vf, Wf):
        """Check that the decomposition exactly reconstructs the mult tensor."""
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                for k in range(p):
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = np.dot(Uf[u_idx, :], Vf[v_idx, :] * Wf[w_idx, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Verify the Strassen decomposition
    assert tensor_check(U, V, W), "Strassen decomposition failed verification!"

    # --- Greedy column removal ---
    if time.time() < deadline:
        current_U = best_U.copy()
        current_V = best_V.copy()
        current_W = best_W.copy()
        
        # Multiple passes of greedy removal
        for pass_num in range(3):
            if time.time() > deadline - 10:
                break
            
            removed_any = True
            while removed_any and time.time() < deadline - 5:
                removed_any = False
                current_rank = current_U.shape[1]
                for try_remove in range(current_rank):
                    if time.time() > deadline - 5:
                        break
                    
                    new_indices = [c for c in range(current_rank) if c != try_remove]
                    if len(new_indices) == 0:
                        break
                    
                    test_U = current_U[:, new_indices]
                    test_V = current_V[:, new_indices]
                    test_W = current_W[:, new_indices]
                    
                    if tensor_check(test_U, test_V, test_W):
                        current_U = test_U
                        current_V = test_V
                        current_W = test_W
                        best_rank = current_U.shape[1]
                        best_U = current_U
                        best_V = test_V
                        best_W = test_W
                        removed_any = True
                        break
        
        # Try randomized column merges if time allows
        if time.time() < deadline - 20:
            rng = np.random.RandomState(42)
            for _ in range(50):
                if time.time() > deadline - 10:
                    break
                
                r1, r2 = rng.choice(best_rank, 2, replace=False)
                alpha = rng.choice([-1, 0, 1])
                
                test_U = best_U.copy()
                test_V = best_V.copy()
                test_W = best_W.copy()
                
                test_U[:, r1] += alpha * test_U[:, r2]
                test_V[:, r1] += alpha * test_V[:, r2]
                test_W[:, r1] += alpha * test_W[:, r2]
                
                new_indices = [c for c in range(best_rank) if c != r2]
                test_U = test_U[:, new_indices]
                test_V = test_V[:, new_indices]
                test_W = test_W[:, new_indices]
                
                if tensor_check(test_U, test_V, test_W):
                    best_U = test_U
                    best_V = test_V
                    best_W = test_W
                    best_rank = best_U.shape[1]

    # Return in list-of-lists format
    best_U = best_U[:best_U.shape[0], :best_U.shape[1]]
    best_V = best_V[:best_V.shape[0], :best_V.shape[1]]
    best_W = best_W[:best_W.shape[0], :best_W.shape[1]]

    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    U_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U_list]
    V_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V_list]
    W_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W_list]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
