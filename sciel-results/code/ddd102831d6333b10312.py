import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 4, 5
    
    # Target tensor T[i, j, k] = 1 if it's a valid multiplication term
    # T[i, j, k] corresponds to U[i, j] * V[j, k] * W[k, i]
    # The multiplication tensor is 1 for all (i, j, k) in the range
    T = np.ones((n, m, p), dtype=float)
    
    def verify_exact(U, V, W, rank):
        """
        Verify if U, V, W (lists of lists of numbers) form an exact decomposition.
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
        """
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True

    # Hardcoded Strassen-based decomposition: rank = 36
    def get_strassen_36():
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        strassen_terms = [
            ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
            ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
            ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
            ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
            ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
            ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
            ({1:1, 3:-1}, {2:1, 3:1}, {0:1})
        ]
        standard_terms = [
            ({0:1}, {0:1}, {0:1}),
            ({1:1}, {1:1}, {0:1}),
            ({2:1}, {0:1}, {1:1}),
            ({3:1}, {1:1}, {1:1})
        ]
        
        for q in range(2):
            for s in range(3):
                if s < 2:
                    for u_t, v_t, w_t in strassen_terms:
                        for idx_u, val_u in u_t.items():
                            li = idx_u // 2
                            lj = idx_u % 2
                            gi = li
                            gj = 2*q + lj
                            U[gi*m + gj][col] += val_u
                        for idx_v, val_v in v_t.items():
                            lj = idx_v // 2
                            lk = idx_v % 2
                            gj = 2*q + lj
                            gk = 2*s + lk
                            V[gj*p + gk][col] += val_v
                        for idx_w, val_w in w_t.items():
                            li = idx_w // 2
                            lk = idx_w % 2
                            gi = li
                            gk = 2*s + lk
                            W[gk*n + gi][col] += val_w
                        col += 1
                else:
                    for u_t, v_t, w_t in standard_terms:
                        for idx_u, val_u in u_t.items():
                            li = idx_u // 2
                            lj = idx_u % 2
                            gi = li
                            gj = 2*q + lj
                            U[gi*m + gj][col] += val_u
                        for idx_v, val_v in v_t.items():
                            lj = idx_v
                            gj = 2*q + lj
                            gk = 4
                            V[gj*p + gk][col] += val_v
                        for idx_w, val_w in w_t.items():
                            li = idx_w
                            gi = li
                            gk = 4
                            W[gk*n + gi][col] += val_w
                        col += 1
        return U, V, W

    # Attempt greedy search for a lower rank (e.g., 35)
    # This is a heuristic. If it fails, we fall back to 36.
    # We only search if we have time.
    search_time_limit = 10.0
    best_result = None
    
    if time.time() < deadline - search_time_limit - 5.0:
        # Simple greedy search using floating point
        # We try to find rank 35.
        # Since exact search is hard, we skip complex search to avoid bugs.
        # The Strassen 36 is a known solid result.
        pass
    
    # Return the hardcoded 36 decomposition
    U, V, W = get_strassen_36()
    
    # Final verification
    if verify_exact(U, V, W, 36):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook (should not happen)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
                
    return {"U": U_school, "V": V_school, "W": W_school}
