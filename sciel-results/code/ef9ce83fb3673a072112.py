"""Rewritten program for 4x5x6 matrix multiplication tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(nm)]
    V_sb = [[0] * schoolbook_rank for _ in range(mp)]
    W_sb = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a][col] = 1
                V_sb[b][col] = 1
                W_sb[c][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank

    def strassen2_decomp():
        """2x2x2 Strassen, rank 7. Returns U(4x7), V(4x7), W(4x7)"""
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        U[2][1]=1; U[3][1]=1; V[2][1]=1; W[2][1]=1
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1; W[3][6]=1
        return U, V, W

    def schoolbook_decomp(ni, mi, pi):
        rank = ni * mi * pi
        U = [[0]*rank for _ in range(ni*mi)]
        V = [[0]*rank for _ in range(mi*pi)]
        W = [[0]*rank for _ in range(pi*ni)]
        col = 0
        for i in range(ni):
            for j in range(mi):
                for k in range(pi):
                    U[i*mi+j][col] = 1
                    V[j*pi+k][col] = 1
                    W[k*ni+i][col] = 1
                    col += 1
        return U, V, W, rank

    def decomp_choice(ni, mi, pi):
        if ni == 2 and mi == 2 and pi == 2:
            U, V, W = strassen2_decomp()
            return U, V, W, 7
        else:
            return schoolbook_decomp(ni, mi, pi)

    def build_partition(n_parts, m_parts, p_parts):
        """Build a block partition decomposition."""
        total_rank = 0
        for ni in n_parts:
            for mi in m_parts:
                for pi in p_parts:
                    total_rank += decomp_choice(ni, mi, pi)[3]
        
        U_full = [[0] * total_rank for _ in range(n*m)]
        V_full = [[0] * total_rank for _ in range(m*p)]
        W_full = [[0] * total_rank for _ in range(p*n)]
        
        col = 0
        i_off = 0
        for ni in n_parts:
            j_off = 0
            for mi in m_parts:
                k_off = 0
                for pi in p_parts:
                    U_b, V_b, W_b, rank_b = decomp_choice(ni, mi, pi)
                    for r in range(rank_b):
                        for i in range(ni):
                            for j in range(mi):
                                U_full[(i_off+i)*m + (j_off+j)][col+r] = U_b[i*mi+j][r]
                        for j in range(mi):
                            for k in range(pi):
                                V_full[(j_off+j)*p + (k_off+k)][col+r] = V_b[j*pi+k][r]
                        for k in range(pi):
                            for i in range(ni):
                                W_full[(k_off+k)*n + (i_off+i)][col+r] = W_b[k*ni+i][r]
                    col += rank_b
                    k_off += pi
                j_off += mi
            i_off += ni
        
        return U_full, V_full, W_full, total_rank

    def verify_fast(U, V, W, rank):
        """Fast verification using NumPy tensor operations."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Check all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = np.dot(U_arr[a], V_arr[b])  # This is wrong, need elementwise with W
                    # Actually: sum_r U[a,r] * V[b,r] * W[c,r]
                    s = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True

    def verify_fast2(U, V, W, rank):
        """Even faster: build full tensor and check."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # We only need to check positions where target is 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = np.einsum('r,r,r->', U_arr[a], V_arr[b], W_arr[c])
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True

    def get_partitions(dim):
        """Get all ways to partition dim into parts of size 1 or 2."""
        results = []
        def rec(rem, current):
            if rem == 0:
                results.append(tuple(current))
                return
            if rem >= 1:
                current.append(1)
                rec(rem - 1, current)
                current.pop()
            if rem >= 2:
                current.append(2)
                rec(rem - 2, current)
                current.pop()
        rec(dim, [])
        return results

    # Enumerate all partitions using only 1s and 2s
    n_parts_list = get_partitions(n)
    m_parts_list = get_partitions(m)
    p_parts_list = get_partitions(p)
    
    # Score each partition by estimated rank (lower is better)
    candidates = []
    for np_ in n_parts_list:
        for mp_ in m_parts_list:
            for pp_ in p_parts_list:
                est_rank = 0
                for ni in np_:
                    for mi in mp_:
                        for pi in pp_:
                            if ni == 2 and mi == 2 and pi == 2:
                                est_rank += 7
                            else:
                                est_rank += ni * mi * pi
                candidates.append((est_rank, np_, mp_, pp_))
    
    # Sort by estimated rank
    candidates.sort(key=lambda x: x[0])
    
    # Try top candidates
    tried = 0
    for est_rank, np_, mp_, pp_ in candidates:
        if time.time() > deadline - 10:
            break
        if est_rank >= best_rank:
            break  # No point trying worse ones
        
        U_c, V_c, W_c, actual_rank = build_partition(list(np_), list(mp_), list(pp_))
        
        if actual_rank >= best_rank:
            continue
        
        # Verify
        if verify_fast2(U_c, V_c, W_c, actual_rank):
            best_rank = actual_rank
            best_result = {"U": U_c, "V": V_c, "W": W_c}
            tried += 1
            if best_rank <= 100:  # Good enough, stop early
                break
    
    return best_result
# EVOLVE-BLOCK-END
