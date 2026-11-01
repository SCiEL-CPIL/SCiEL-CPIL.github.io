import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2 by 3 matrix by a 3 by 4 matrix.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 3, 4
    nm, mp, pn = n * m, m * p, p * n  # 6, 12, 8
    
    # Build the schoolbook decomposition of rank 24 as fallback
    rank_schoolbook = n * m * p  # 24
    U_sb = np.zeros((nm, rank_schoolbook), dtype=np.int64)
    V_sb = np.zeros((mp, rank_schoolbook), dtype=np.int64)
    W_sb = np.zeros((pn, rank_schoolbook), dtype=np.int64)
    
    triples = [(i,j,k) for i in range(n) for j in range(m) for k in range(p)]
    col = 0
    for i, j, k in triples:
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U_sb[u_idx, col] = 1
        V_sb[v_idx, col] = 1
        W_sb[w_idx, col] = 1
        col += 1
    
    def verify_decomposition(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    def try_merge_pair(idx1, idx2, entries):
        """Try to merge two schoolbook terms into one column with given entry set."""
        i1,j1,k1 = triples[idx1]
        i2,j2,k2 = triples[idx2]
        
        u1_idx = i1*m+j1
        u2_idx = i2*m+j2
        v1_idx = j1*p+k1
        v2_idx = j2*p+k2
        w1_idx = k1*n+i1
        w2_idx = k2*n+i2
        
        for u1 in entries:
            if u1 == 0:
                continue
            for u2 in entries:
                if u2 == 0:
                    continue
                for v1 in entries:
                    if v1 == 0:
                        continue
                    for v2 in entries:
                        if v2 == 0:
                            continue
                        for w1 in entries:
                            if w1 == 0:
                                continue
                            for w2 in entries:
                                if w2 == 0:
                                    continue
                                
                                if u1 * v1 * w1 != 1:
                                    continue
                                if u2 * v2 * w2 != 1:
                                    continue
                                
                                U_col = np.zeros(nm, dtype=np.int64)
                                V_col = np.zeros(mp, dtype=np.int64)
                                W_col = np.zeros(pn, dtype=np.int64)
                                
                                U_col[u1_idx] = u1
                                U_col[u2_idx] = u2
                                V_col[v1_idx] = v1
                                V_col[v2_idx] = v2
                                W_col[w1_idx] = w1
                                W_col[w2_idx] = w2
                                
                                valid = True
                                for i in range(n):
                                    if not valid:
                                        break
                                    for j in range(m):
                                        if not valid:
                                            break
                                        for k in range(p):
                                            if (i,j,k) == (i1,j1,k1) or (i,j,k) == (i2,j2,k2):
                                                continue
                                            u_idx = i*m+j
                                            v_idx = j*p+k
                                            w_idx = k*n+i
                                            if U_col[u_idx] * V_col[v_idx] * W_col[w_idx] != 0:
                                                valid = False
                                                break
                                
                                if valid:
                                    U_final = np.zeros((nm, 23), dtype=np.int64)
                                    V_final = np.zeros((mp, 23), dtype=np.int64)
                                    W_final = np.zeros((pn, 23), dtype=np.int64)
                                    
                                    col = 0
                                    for t_idx, (i,j,k) in enumerate(triples):
                                        if t_idx == idx1 or t_idx == idx2:
                                            continue
                                        u_idx = i*m+j
                                        v_idx = j*p+k
                                        w_idx = k*n+i
                                        U_final[u_idx, col] = 1
                                        V_final[v_idx, col] = 1
                                        W_final[w_idx, col] = 1
                                        col += 1
                                    
                                    U_final[:, col] = U_col
                                    V_final[:, col] = V_col
                                    W_final[:, col] = W_col
                                    
                                    if verify_decomposition(U_final, V_final, W_final):
                                        return (U_final, V_final, W_final)
        return None
    
    # Try integer entries {-1, 0, 1}
    entries = [-1, 0, 1]
    for idx1 in range(len(triples)):
        if time.time() > deadline - 10:
            break
        for idx2 in range(idx1+1, len(triples)):
            if time.time() > deadline - 10:
                break
            result = try_merge_pair(idx1, idx2, entries)
            if result is not None:
                U, V, W = result
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try with entries {-2, -1, 0, 1, 2} - but need product = 1, so limited options
    # Products of 3 from {-2,-1,0,1,2} that equal 1: (1,1,1), (-1,-1,1), (-1,1,-1), (1,-1,-1)
    # And permutations. So effectively same as {-1,0,1} for product=1
    # Unless we allow non-integer intermediate... but we need integer entries
    
    # Fallback: return schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
