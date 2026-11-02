import time
import numpy as np

def construct():
    n, m, p = 3, 4, 6
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Build the matrix multiplication tensor T[i,j,k] = 1 if we multiply A[i,j]*B[j,k]->C[i,k]
    # Flattened: A has nm=12 entries, B has mp=24 entries, C has pn=18 entries
    # Tensor T[a,b,c] = 1 iff a=(i,j), b=(j,k), c=(i,k)
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a,b,c] = 1
    
    # Try iterative rank-1 peeling (de Zeeuw's algorithm)
    def try_peeling(T_tensor, max_rank, time_budget):
        """Try to find a rank decomposition by iterative peeling."""
        T_rem = T_tensor.copy().astype(float)
        U_list = []
        V_list = []
        W_list = []
        
        for rank in range(1, max_rank+1):
            if time.time() > time_budget:
                break
            # Find a rank-1 component: vectors u, v, w such that
            # u ⊗ v ⊗ w is in the span of T_rem
            # Strategy: pick random support patterns and solve linear system
            
            found = False
            # Try several random initializations
            for attempt in range(50):
                if time.time() > time_budget:
                    break
                    
                # Choose random sparse supports
                u = np.zeros(n*m)
                v = np.zeros(m*p)
                w = np.zeros(p*n)
                
                # Pick random non-zero entries
                n_u = max(1, int(np.random.exponential(3)))
                n_v = max(1, int(np.random.exponential(4)))
                n_w = max(1, int(np.random.exponential(2)))
                
                u_idx = np.random.choice(n*m, min(n_u, n*m), replace=False)
                v_idx = np.random.choice(m*p, min(n_v, m*p), replace=False)
                w_idx = np.random.choice(p*n, min(n_w, p*n), replace=False)
                
                u[u_idx] = np.random.choice([-1, 1], size=len(u_idx))
                v[v_idx] = np.random.choice([-1, 1], size=len(v_idx))
                w[w_idx] = np.random.choice([-1, 1], size=len(w_idx))
                
                # Compute the rank-1 tensor
                rank1 = np.einsum('a,b,c->abc', u, v, w)
                
                # Check if rank1 is in the span of T_rem
                # Flatten and check
                T_flat = T_rem.flatten()
                r_flat = rank1.flatten()
                
                # Solve: find alpha such that T_rem - alpha*rank1 has lower "rank"
                # Use least squares to project
                try:
                    # Check if rank1 is approximately in the column space
                    # by checking if T_rem - alpha*rank1 can be made sparser
                    # Simple heuristic: try alpha values
                    best_alpha = None
                    best_score = float('inf')
                    
                    for alpha in [0.5, 1.0, -0.5, -1.0, 0.25, -0.25, 2.0, -2.0]:
                        T_test = T_rem - alpha * rank1
                        # Score: number of non-zero entries (lower is better for peeling)
                        score = np.count_nonzero(T_test)
                        if score < best_score:
                            best_score = score
                            best_alpha = alpha
                    
                    if best_alpha is not None and best_score < np.count_nonzero(T_rem):
                        T_rem = T_rem - best_alpha * rank1
                        U_list.append(u)
                        V_list.append(v)
                        W_list.append(w)
                        found = True
                        break
                except Exception:
                    continue
            
            if not found:
                break
        
        # Check if we've decomposed everything
        if np.allclose(T_rem, 0, atol=1e-10):
            return U_list, V_list, W_list
        return None
    
    # First, get the known 66-term decomposition as a baseline
    def get_baseline_66():
        U = [[0] * 66 for _ in range(n * m)]
        V = [[0] * 66 for _ in range(m * p)]
        W = [[0] * 66 for _ in range(p * n)]
        col = 0
        
        def add_schoolbook(i, j, k):
            nonlocal col
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
        
        def add_strassen_222(i_base, j_base, k_base):
            nonlocal col
            a_idx = i_base * m + j_base
            b_idx = i_base * m + (j_base + 1)
            c_idx = (i_base + 1) * m + j_base
            d_idx = (i_base + 1) * m + (j_base + 1)
            e_idx = j_base * p + k_base
            f_idx = j_base * p + (k_base + 1)
            g_idx = (j_base + 1) * p + k_base
            h_idx = (j_base + 1) * p + (k_base + 1)
            w00 = k_base * n + i_base
            w01 = (k_base + 1) * n + i_base
            w10 = k_base * n + (i_base + 1)
            w11 = (k_base + 1) * n + (i_base + 1)
            
            U[a_idx][col] = 1; U[d_idx][col] = 1
            V[e_idx][col] = 1; V[h_idx][col] = 1
            W[w00][col] = 1; W[w11][col] = 1
            col += 1
            U[c_idx][col] = 1; U[d_idx][col] = 1
            V[e_idx][col] = 1
            W[w10][col] = 1; W[w11][col] = -1
            col += 1
            U[a_idx][col] = 1
            V[f_idx][col] = 1; V[h_idx][col] = -1
            W[w01][col] = 1; W[w11][col] = 1
            col += 1
            U[d_idx][col] = 1
            V[g_idx][col] = 1; V[e_idx][col] = -1
            W[w00][col] = 1; W[w10][col] = 1
            col += 1
            U[a_idx][col] = 1; U[b_idx][col] = 1
            V[h_idx][col] = 1
            W[w00][col] = -1; W[w01][col] = 1
            col += 1
            U[c_idx][col] = 1; U[a_idx][col] = -1
            V[e_idx][col] = 1; V[f_idx][col] = 1
            W[w11][col] = 1
            col += 1
            U[b_idx][col] = 1; U[d_idx][col] = -1
            V[g_idx][col] = 1; V[h_idx][col] = 1
            W[w00][col] = 1
            col += 1
        
        # Row 0: schoolbook 24 terms
        for j in range(4):
            for k in range(6):
                add_schoolbook(0, j, k)
        
        # Rows 1-2: 6 Strassen blocks of 2x2x2
        for j_block in range(2):
            for k_block in range(3):
                j_base = j_block * 2
                k_base = k_block * 2
                add_strassen_222(1, j_base, k_base)
        
        return U, V, W
    
    # Start with baseline
    U_base, V_base, W_base = get_baseline_66()
    best_rank = 66
    best_result = {"U": U_base, "V": V_base, "W": W_base}
    
    # Try to improve via peeling
    if time.time() < deadline - 10:
        T_tensor = np.array(T, dtype=float)
        # Try peeling from the schoolbook rank-72 tensor
        T_schoolbook = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    T_schoolbook[a,b,c] = 1
        
        result = try_peeling(T_schoolbook, 72, deadline - 10)
        if result is not None:
            U_list, V_list, W_list = result
            new_rank = len(U_list)
            if new_rank < best_rank:
                U_new = [[0] * new_rank for _ in range(n*m)]
                V_new = [[0] * new_rank for _ in range(m*p)]
                W_new = [[0] * new_rank for _ in range(p*n)]
                for r in range(new_rank):
                    for a in range(n*m):
                        U_new[a][r] = U_list[r][a]
                    for b in range(m*p):
                        V_new[b][r] = V_list[r][b]
                    for c in range(p*n):
                        W_new[c][r] = W_list[r][c]
                # Verify
                T_check = np.einsum('ar,br,cr->abc', 
                    np.array(U_new).T, np.array(V_new).T, np.array(W_new).T)
                if np.allclose(T_check, T_tensor, atol=1e-10):
                    best_rank = new_rank
                    best_result = {"U": U_new, "V": V_new, "W": W_new}
    
    # Also try a smarter block-based approach: 3x2x3 blocks with optimized rank
    if time.time() < deadline - 5:
        # Try decomposing 3x4x6 as three 3x2x2 blocks + adjustments
        # Or try 2x2x2 Strassen on all possible 2x2 subblocks
        
        # Alternative: try rank 60 with a different block structure
        # Use 3x2x2 blocks: 3 rows, 2 cols of A, 2 cols of B
        # Each 3x2x2 block: can we do it in < 12 terms?
        # 3x2x2 schoolbook = 12 terms
        # With Strassen-like: maybe 10-11 terms?
        
        pass  # Keep it simple for now
    
    return best_result
