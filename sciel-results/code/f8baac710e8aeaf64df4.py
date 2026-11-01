import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Schoolbook fallback: rank = n*m*p = 52
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition correctly computes matrix multiplication."""
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
        
        return np.allclose(C_decomp, C_expected, atol=1e-9)

    # Strategy: Use Strassen (rank 7) for 2x2*2x2 blocks, schoolbook for remaining columns
    # For p=13: use Strassen for 2 columns (rank 7), schoolbook for 11 columns (rank 4*11=44)
    # Total: 7 + 44 = 51 (saves 1 over schoolbook's 52)
    
    # Better: Try to use Strassen for as many 2-column blocks as possible
    # 13 columns = 6 blocks of 2 + 1 column of 1
    # 6 Strassen blocks: 6*7 = 42, plus 1 schoolbook column: 4, total = 46
    
    # But wait - Strassen works on 2x2 * 2x2. For 2x2 * 2xk where k>2,
    # we can process column pairs with Strassen.
    
    # Let's try: 6 Strassen blocks (12 columns) + 1 schoolbook column = 6*7 + 4 = 46
    if p >= 2:
        num_strassen_blocks = p // 2  # 6 for p=13
        remaining_cols = p % 2  # 1 for p=13
        total_rank = num_strassen_blocks * 7 + 4 * remaining_cols
        
        if total_rank < best_rank:
            U = [[0] * total_rank for _ in range(n * m)]   # 4 rows
            V = [[0] * total_rank for _ in range(m * p)]   # 26 rows
            W = [[0] * total_rank for _ in range(p * n)]   # 26 rows
            
            col = 0
            for block in range(num_strassen_blocks):
                col_start = block * 2  # starting column index in B and C
                col_end = col_start + 2
                
                # For this block, B columns are col_start, col_start+1
                # V indices: V[j*p + k] where j in {0,1}, k in {col_start, col_start+1}
                # W indices: W[k*n + i] where i in {0,1}, k in {col_start, col_start+1}
                
                # Strassen terms for 2x2 * 2x2:
                # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
                # U indices: 0=a, 1=b, 2=c, 3=d
                # V: e=V[0*p+col_start]=V[col_start], f=V[1*p+col_start+1]... 
                # Wait, V[j*p+k]: j is row of B (0 or 1), k is column of B
                # So e = B[0,col_start] -> V[0*p + col_start] = V[col_start]
                # f = B[0,col_start+1] -> V[0*p + col_start+1] = V[col_start+1]
                # g = B[1,col_start] -> V[1*p + col_start] = V[p + col_start]
                # h = B[1,col_start+1] -> V[1*p + col_start+1] = V[p + col_start+1]
                
                # W[k*n+i]: k is column of C (which equals column of B), i is row of C
                # C00 (i=0,k=col_start) -> W[col_start*n + 0] = W[col_start]
                # C01 (i=0,k=col_start+1) -> W[(col_start+1)*n + 0] = W[n + col_start]
                # C10 (i=1,k=col_start) -> W[col_start*n + 1] = W[col_start + 1]
                # C11 (i=1,k=col_start+1) -> W[(col_start+1)*n + 1] = W[n + col_start + 1]
                
                e_idx = col_start
                f_idx = col_start + 1
                g_idx = p + col_start
                h_idx = p + col_start + 1
                
                C00_idx = col_start * n + 0
                C01_idx = (col_start + 1) * n + 0
                C10_idx = col_start * n + 1
                C11_idx = (col_start + 1) * n + 1
                
                strassen_terms = [
                    # (U_vec, V_dict, W_dict)
                    ([1, 1, 0, 0], {e_idx: 1, f_idx: 1}, {C00_idx: 1, C11_idx: 1}),
                    ([0, 0, 1, 1], {e_idx: 1}, {C10_idx: 1}),
                    ([1, 0, 0, 0], {h_idx: 1, f_idx: -1}, {C01_idx: 1}),
                    ([0, 0, 0, 1], {g_idx: 1, e_idx: -1}, {C10_idx: 1}),
                    ([1, 0, 0, 1], {e_idx: 1, h_idx: 1}, {C01_idx: 1}),
                    ([0, 1, 1, 0], {g_idx: 1, e_idx: -1}, {C11_idx: 1}),
                    ([0, 1, 0, 1], {f_idx: 1, h_idx: -1}, {C00_idx: -1}),
                ]
                
                for U_vec, V_dict, W_dict in strassen_terms:
                    for i in range(4):
                        if U_vec[i] != 0:
                            U[i][col] = U_vec[i]
                    for j_idx, val in V_dict.items():
                        V[j_idx][col] = val
                    for k_idx, val in W_dict.items():
                        W[k_idx][col] = val
                    col += 1
            
            # Schoolbook for remaining column(s)
            for block in range(num_strassen_blocks):
                col_start = block * 2
                col_end = col_start + 2
                if col_end < p:
                    for i in range(n):
                        for j in range(m):
                            for k in range(col_end, p):
                                if k >= col_end:
                                    U[i * m + j][col] = 1
                                    V[j * p + k][col] = 1
                                    W[k * n + i][col] = 1
                                    col += 1
                        break  # only need to do this once
                    break
            
            # Simpler: just handle remaining columns
            for k in range(num_strassen_blocks * 2, p):
                for i in range(n):
                    for j in range(m):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
            
            # Verify
            if verify_decomposition(U, V, W, total_rank):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = total_rank
            
            if time.time() - start_time > deadline - 5:
                return best_result

    # Try even better: Can we do better than Strassen for 2x2*2x2?
    # The rank of 2x2*2x2 matrix multiplication is known to be 7 (Strassen is optimal for 2x2)
    # So 6*7 + 4 = 46 is the best with this approach
    
    # Alternative: use a different split. What if we use schoolbook for some columns
    # and Strassen for others in a mixed way?
    # Actually, let's verify our current best and also try a few more configurations
    
    # Try: 5 Strassen blocks + 3 schoolbook columns = 5*7 + 12 = 47 (worse)
    # Try: 7 Strassen blocks would need 14 columns, but we only have 13
    
    # The optimal is indeed 6 Strassen + 1 schoolbook = 46
    
    # Let's also try a completely different approach: 
    # Can we find a rank < 46 decomposition using more sophisticated methods?
    # For now, 46 is our best. Let's make sure it's correct.
    
    if time.time() - start_time < deadline - 2:
        # Double-check the 46-rank decomposition
        p_check = p
        num_blocks_check = p_check // 2
        rem_check = p_check % 2
        rank_check = num_blocks_check * 7 + 4 * rem_check
        
        if rank_check < best_rank:
            U2 = [[0] * rank_check for _ in range(n * m)]
            V2 = [[0] * rank_check for _ in range(m * p)]
            W2 = [[0] * rank_check for _ in range(p * n)]
            
            col2 = 0
            for block in range(num_blocks_check):
                cs = block * 2
                e_i = cs
                f_i = cs + 1
                g_i = p + cs
                h_i = p + cs + 1
                C00_i = cs * n
                C01_i = (cs + 1) * n
                C10_i = cs * n + 1
                C11_i = (cs + 1) * n + 1
                
                terms = [
                    ([1,1,0,0], {e_i:1, f_i:1}, {C00_i:1, C11_i:1}),
                    ([0,0,1,1], {e_i:1}, {C10_i:1}),
                    ([1,0,0,0], {h_i:1, f_i:-1}, {C01_i:1}),
                    ([0,0,0,1], {g_i:1, e_i:-1}, {C10_i:1}),
                    ([1,0,0,1], {e_i:1, h_i:1}, {C01_i:1}),
                    ([0,1,1,0], {g_i:1, e_i:-1}, {C11_i:1}),
                    ([0,1,0,1], {f_i:1, h_i:-1}, {C00_i:-1}),
                ]
                
                for uv, vd, wd in terms:
                    for i in range(4):
                        if uv[i] != 0:
                            U2[i][col2] = uv[i]
                    for idx, val in vd.items():
                        V2[idx][col2] = val
                    for idx, val in wd.items():
                        W2[idx][col2] = val
                    col2 += 1
            
            for k in range(num_blocks_check * 2, p):
                for i in range(n):
                    for j in range(m):
                        U2[i*m+j][col2] = 1
                        V2[j*p+k][col2] = 1
                        W2[k*n+i][col2] = 1
                        col2 += 1
            
            if verify_decomposition(U2, V2, W2, rank_check):
                best_result = {"U": U2, "V": V2, "W": W2}
                best_rank = rank_check
    
    return best_result
