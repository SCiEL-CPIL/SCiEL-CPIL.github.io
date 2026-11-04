"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

_start_time = time.time()

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    deadline = _start_time + 140.0  # safety margin before 150s kill
    
    def generate_partitions(total, max_part=3):
        """Generate all ways to partition 'total' into parts of size 1..max_part."""
        if total == 0:
            return [[]]
        results = []
        for first in range(1, min(max_part, total) + 1):
            for rest in generate_partitions(total - first, max_part):
                results.append([first] + rest)
        return results
    
    def block_rank(n_s, m_s, p_s):
        """Return the rank needed for an n_s x m_s x p_s block."""
        if n_s == 1 and m_s == 1 and p_s == 1:
            return 1
        if n_s == 2 and m_s == 2 and p_s == 2:
            return 7  # Strassen
        # Otherwise schoolbook
        return n_s * m_s * p_s
    
    def build_decomposition(n_parts, m_parts, p_parts, n, m, p):
        """Build the full decomposition from a partitioning. Returns (U, V, W, rank)."""
        terms = []  # each: (u_entries, v_entries, w_entries)
        
        # Compute offsets
        n_offsets = [0]
        for sz in n_parts[:-1]:
            n_offsets.append(n_offsets[-1] + sz)
        m_offsets = [0]
        for sz in m_parts[:-1]:
            m_offsets.append(m_offsets[-1] + sz)
        p_offsets = [0]
        for sz in p_parts[:-1]:
            p_offsets.append(p_offsets[-1] + sz)
        
        for ni, ns in enumerate(n_parts):
            for mi, ms in enumerate(m_parts):
                for pi, ps in enumerate(p_parts):
                    a_rows = list(range(n_offsets[ni], n_offsets[ni] + ns))
                    c_cols = list(range(m_offsets[mi], m_offsets[mi] + ms))
                    d_cols = list(range(p_offsets[pi], p_offsets[pi] + ps))
                    
                    def U_idx(la, lc): return a_rows[la] * m + c_cols[lc]
                    def V_idx(lc, ld): return c_cols[lc] * p + d_cols[ld]
                    def W_idx(ld, la): return d_cols[ld] * n + a_rows[la]
                    
                    if ns == 2 and ms == 2 and ps == 2:
                        # Strassen 7 terms
                        a0, a1 = a_rows
                        c0, c1 = c_cols
                        d0, d1 = d_cols
                        
                        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
                        # C[0,0] += M1, C[1,1] += M1
                        terms.append((
                            [(U_idx(0,0), 1), (U_idx(1,1), 1)],
                            [(V_idx(0,0), 1), (V_idx(1,1), 1)],
                            [(W_idx(0,0), 1), (W_idx(1,1), 1)]
                        ))
                        
                        # M2 = (A[1,0]+A[1,1])*B[0,0]
                        # C[1,0] += M2, C[1,1] += -M2
                        terms.append((
                            [(U_idx(1,0), 1), (U_idx(1,1), 1)],
                            [(V_idx(0,0), 1)],
                            [(W_idx(0,1), 1), (W_idx(1,1), -1)]
                        ))
                        
                        # M3 = A[0,0]*(B[0,1]-B[1,1])
                        # C[0,1] += M3, C[1,1] += M3
                        terms.append((
                            [(U_idx(0,0), 1)],
                            [(V_idx(0,1), 1), (V_idx(1,1), -1)],
                            [(W_idx(1,0), 1), (W_idx(1,1), 1)]
                        ))
                        
                        # M4 = A[1,1]*(B[1,0]-B[0,0])
                        # C[0,0] += M4, C[1,0] += M4
                        terms.append((
                            [(U_idx(1,1), 1)],
                            [(V_idx(1,0), 1), (V_idx(0,0), -1)],
                            [(W_idx(0,0), 1), (W_idx(0,1), 1)]
                        ))
                        
                        # M5 = (A[0,0]+A[0,1])*B[1,1]
                        # C[0,0] += -M5, C[0,1] += M5
                        terms.append((
                            [(U_idx(0,0), 1), (U_idx(0,1), 1)],
                            [(V_idx(1,1), 1)],
                            [(W_idx(0,0), -1), (W_idx(1,0), 1)]
                        ))
                        
                        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
                        # C[1,1] += M6
                        terms.append((
                            [(U_idx(1,0), 1), (U_idx(0,0), -1)],
                            [(V_idx(0,0), 1), (V_idx(0,1), 1)],
                            [(W_idx(1,1), 1)]
                        ))
                        
                        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
                        # C[0,0] += M7
                        terms.append((
                            [(U_idx(0,1), 1), (U_idx(1,1), -1)],
                            [(V_idx(1,0), 1), (V_idx(1,1), 1)],
                            [(W_idx(0,0), 1)]
                        ))
                    else:
                        # Schoolbook for this block
                        for la in range(ns):
                            for lc in range(ms):
                                for ld in range(ps):
                                    terms.append((
                                        [(U_idx(la, lc), 1)],
                                        [(V_idx(lc, ld), 1)],
                                        [(W_idx(ld, la), 1)]
                                    ))
        
        # Build matrices
        rank = len(terms)
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        for ci, (u_entries, v_entries, w_entries) in enumerate(terms):
            for row, coeff in u_entries:
                U[row][ci] += coeff
            for row, coeff in v_entries:
                V[row][ci] += coeff
            for row, coeff in w_entries:
                W[row][ci] += coeff
        
        return U, V, W, rank
    
    def verify(U_mat, V_mat, W_mat, n, m, p, tol=1e-9):
        """Check if U,V,W correctly decompose the matmul tensor."""
        U_np = np.array(U_mat, dtype=float)
        V_np = np.array(V_mat, dtype=float)
        W_np = np.array(W_mat, dtype=float)
        r = U_np.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_np[i*m+j, :], V_np[j*p+k, :] * W_np[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # First, quickly build and verify the known good partition
    # n: [2,1], m: [2,2,1], p: [2,2,2]
    known_good = (
        [2, 1],  # n partition
        [2, 2, 1],  # m partition
        [2, 2, 2]  # p partition
    )
    
    best_rank = float('inf')
    best_U = best_V = best_W = None
    best_partition = None
    
    # Try the known good partition first
    U, V, W, rank = build_decomposition(*known_good, n, m, p)
    if verify(U, V, W, n, m, p):
        best_rank = rank
        best_U, best_V, best_W = U, V, W
        best_partition = known_good
    
    # Now search for potentially better partitions
    n_partitions = generate_partitions(n, max_part=3)
    m_partitions = generate_partitions(m, max_part=3)
    p_partitions = generate_partitions(p, max_part=3)
    
    # Sort partitions by estimated rank to find good ones first
    def est_rank(np_, mp_, pp_):
        total = 0
        for ns, ms, ps in product(np_, mp_, pp_):
            total += block_rank(ns, ms, ps)
        return total
    
    all_partitions = []
    for np_ in n_partitions:
        for mp_ in m_partitions:
            for pp_ in p_partitions:
                r = est_rank(np_, mp_, pp_)
                all_partitions.append((r, np_, mp_, pp_))
    
    all_partitions.sort(key=lambda x: x[0])
    
    for r_est, np_, mp_, pp_ in all_partitions:
        if time.time() > deadline:
            break
        
        # Skip if we already have a good result and this partition can't beat it
        if r_est >= best_rank:
            continue
        
        U, V, W, rank = build_decomposition(np_, mp_, pp_, n, m, p)
        if verify(U, V, W, n, m, p):
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            best_partition = (np_, mp_, pp_)
    
    # If no better partition found, use the known good one
    if best_U is None:
        U, V, W, rank = build_decomposition(*known_good, n, m, p)
        best_U, best_V, best_W = U, V, W
        best_rank = rank
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
