import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = 145
    
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 52
    
    # Build schoolbook
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
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    def verify(U, V, W, rank):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_decomp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for t in range(rank):
                        C_decomp[i, k] += U_np[i*m+j, t] * V_np[j*p+k, t] * W_np[k*n+i, t]
        
        return np.allclose(C_decomp, C_expected, atol=1e-10)
    
    # Try to find a valid rank-7 decomposition for 2x2 x 2x2
    # by brute force over small integer/half-integer coefficients
    # We know rank 7 is achievable (Strassen), so let's find valid terms
    
    # The 2x2x2 multiplication tensor has 8 nonzero entries:
    # T[0,0,0]=1, T[0,1,1]=1, T[1,0,0]=1, T[1,1,1]=1
    # Wait, T[i,j,k] where C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 if the term A[i,j]*B[j,k] appears in C[i,k]
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # So the tensor entries are:
    # T[0,0,0] = 1 (A[0,0]*B[0,0] in C[0,0])
    # T[0,1,0] = 1 (A[0,1]*B[1,0] in C[0,0])
    # T[0,0,1] = 1 (A[0,0]*B[0,1] in C[0,1])
    # T[0,1,1] = 1 (A[0,1]*B[1,1] in C[0,1])
    # T[1,0,0] = 1 (A[1,0]*B[0,0] in C[1,0])
    # T[1,1,0] = 1 (A[1,1]*B[1,0] in C[1,0])
    # T[1,0,1] = 1 (A[1,0]*B[0,1] in C[1,1])
    # T[1,1,1] = 1 (A[1,1]*B[1,1] in C[1,1])
    
    # For a rank-1 term with U_vec (4 entries), V_vec (2 entries for 2 cols), W_vec (2 entries for 2 rows of C)
    # It contributes U[i]*V[j]*W[k] to T[i,j,k] for the 2x2x2 block
    
    # Let me just try the known Strassen terms and verify
    def try_strassen_for_pair(k0, k1):
        """Return list of 7 terms (U_vec, V_dict, W_dict) for columns k0,k1, or None if invalid"""
        e_idx = 0 * p + k0
        f_idx = 1 * p + k0
        g_idx = 0 * p + k1
        h_idx = 1 * p + k1
        C00_idx = k0 * n + 0
        C01_idx = k1 * n + 0
        C10_idx = k0 * n + 1
        C11_idx = k1 * n + 1
        
        # Try multiple candidate Strassen decompositions
        candidates = [
            # Candidate 1
            [
                ([1, 1, 0, 0], {e_idx: 1, f_idx: 1}, {C00_idx: 1, C11_idx: 1}),
                ([0, 0, 1, 1], {e_idx: 1}, {C10_idx: 1, C11_idx: -1}),
                ([1, 0, 0, 0], {h_idx: 1, f_idx: -1}, {C01_idx: 1}),
                ([0, 0, 0, 1], {g_idx: 1, e_idx: -1}, {C00_idx: 1, C10_idx: 1}),
                ([1, 0, 0, 1], {e_idx: 1, h_idx: 1}, {C00_idx: -1, C01_idx: 1}),
                ([0, 1, 1, 0], {g_idx: 1, e_idx: -1}, {C11_idx: 1}),
                ([0, 1, 0, 1], {f_idx: 1, h_idx: -1}, {C00_idx: 1}),
            ],
            # Candidate 2 (different sign convention)
            [
                ([1, 1, 0, 0], {e_idx: 1, f_idx: 1}, {C00_idx: 1, C11_idx: 1}),
                ([0, 0, 1, 1], {e_idx: 1}, {C10_idx: 1}),
                ([1, 0, 0, 0], {h_idx: 1, f_idx: -1}, {C01_idx: 1}),
                ([0, 0, 0, 1], {g_idx: 1, e_idx: -1}, {C10_idx: 1}),
                ([1, 0, 0, 1], {e_idx: 1, h_idx: 1}, {C01_idx: 1}),
                ([0, 1, 1, 0], {g_idx: 1, e_idx: -1}, {C11_idx: 1}),
                ([0, 1, 0, 1], {f_idx: 1, h_idx: -1}, {C00_idx: -1}),
            ],
        ]
        
        for terms in candidates:
            # Build a local 2x2x2x7 tensor and verify
            local_U = np.zeros((4, 7))
            local_V = np.zeros((2, 2, 7))  # V[j][k_local] where k_local 0->k0, 1->k1
            local_W = np.zeros((2, 2, 7))  # W[k_local][i]
            
            for t, (U_vec, V_dict, W_dict) in enumerate(terms):
                for i in range(4):
                    local_U[i, t] = U_vec[i]
                for j_idx, val in V_dict.items():
                    if j_idx == e_idx:
                        local_V[0, 0, t] += val
                    elif j_idx == f_idx:
                        local_V[1, 0, t] += val
                    elif j_idx == g_idx:
                        local_V[0, 1, t] += val
                    elif j_idx == h_idx:
                        local_V[1, 1, t] += val
                for k_idx, val in W_dict.items():
                    if k_idx == C00_idx:
                        local_W[0, 0, t] += val
                    elif k_idx == C01_idx:
                        local_W[1, 0, t] += val
                    elif k_idx == C10_idx:
                        local_W[0, 1, t] += val
                    elif k_idx == C11_idx:
                        local_W[1, 1, t] += val
            
            # Verify: sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] should give the tensor
            # For 2x2x2: T[i,j,k] = sum_t U[i,j,t]*V[j,k,t]*W[k,i,t]
            # But our indexing: U[i*m+j], V[j*p+k], W[k*n+i]
            # For local: U[i,j], V[j,k_local], W[k_local,i]
            # T[i,j,k_local] = sum_t U[i,j,t]*V[j,k_local,t]*W[k_local,i,t]
            
            # Expected tensor for 2x2 multiplication:
            # T[0,0,0]=1, T[0,1,0]=1, T[0,0,1]=1, T[0,1,1]=1
            # T[1,0,0]=1, T[1,1,0]=1, T[1,0,1]=1, T[1,1,1]=1
            
            valid = True
            for i in range(2):
                for j in range(2):
                    for k_local in range(2):
                        val = 0
                        for t in range(7):
                            val += local_U[i*2+j, t] * local_V[j, k_local, t] * local_W[k_local, i, t]
                        expected = 1
                        if abs(val - expected) > 1e-10:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                return terms
        
        return None
    
    # Try different numbers of Strassen pairs
    for num_strassen_pairs in range(0, (p + 1) // 2 + 1):
        if time.time() - start_time > deadline - 5:
            break
        
        strassen_cols = 2 * num_strassen_pairs
        if strassen_cols > p:
            continue
        schoolbook_cols = p - strassen_cols
        total_rank = 7 * num_strassen_pairs + 4 * schoolbook_cols
        
        if total_rank >= best_rank:
            continue
        
        # Get Strassen terms for each pair
        all_strassen_terms = []
        valid_all = True
        for pair_idx in range(num_strassen_pairs):
            k0 = 2 * pair_idx
            k1 = 2 * pair_idx + 1
            terms = try_strassen_for_pair(k0, k1)
            if terms is None:
                valid_all = False
                break
            all_strassen_terms.extend(terms)
        
        if not valid_all:
            continue
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        for U_vec, V_dict, W_dict in all_strassen_terms:
            for i in range(4):
                if U_vec[i] != 0:
                    U[i][col] = U_vec[i]
            for j_idx, val in V_dict.items():
                V[j_idx][col] = val
            for k_idx, val in W_dict.items():
                W[k_idx][col] = val
            col += 1
        
        for k in range(strassen_cols, p):
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        if verify(U, V, W, total_rank):
            best_result = {"U": U, "V": V, "W": W}
            best_rank = total_rank
    
    return best_result
