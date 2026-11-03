"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Rank function for a block of size (ns, ms, ps)
    def block_rank(ns, ms, ps):
        if ns == 1 and ms == 1 and ps == 1:
            return 1
        if ns == 1:
            return ms * ps
        if ms == 1:
            return ns * ps
        if ps == 1:
            return ns * ms
        if ns == 2 and ms == 2 and ps == 2:
            return 7  # Strassen
        if ns == 2 and ms == 2 and ps == 1:
            return 4
        if ns == 2 and ms == 1 and ps == 2:
            return 4
        if ns == 1 and ms == 2 and ps == 2:
            return 4
        if ns == 2 and ms == 1 and ps == 1:
            return 2
        if ns == 1 and ms == 2 and ps == 1:
            return 2
        if ns == 1 and ms == 1 and ps == 2:
            return 2
        # For other cases, use schoolbook as fallback
        # But check if we can decompose further
        return ns * ms * ps
    
    # Generate all compositions of a number into parts >= 1
    def compositions(total, min_part=1):
        if total == min_part:
            yield (min_part,)
            return
        for first in range(min_part, total + 1):
            for rest in compositions(total - first, min_part):
                yield (first,) + rest
    
    # Find optimal block decomposition
    best_rank = n * m * p
    best_decomp = None
    
    for nb in compositions(n):
        for mb in compositions(m):
            for pb in compositions(p):
                if len(nb) != len(mb) or len(mb) != len(pb):
                    continue
                total_r = 0
                for ns, ms, ps in zip(nb, mb, pb):
                    total_r += block_rank(ns, ms, ps)
                if total_r < best_rank:
                    best_rank = total_r
                    best_decomp = (nb, mb, pb)
    
    # Now build the actual decomposition
    U_vecs = []
    V_vecs = []
    W_vecs = []
    
    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        u = np.zeros(n * m, dtype=float)
        v = np.zeros(m * p, dtype=float)
        w = np.zeros(p * n, dtype=float)
        for idx, val in zip(u_indices, u_vals):
            u[idx] += val
        for idx, val in zip(v_indices, v_vals):
            v[idx] += val
        for idx, val in zip(w_indices, w_vals):
            w[idx] += val
        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)
    
    def gen_block(ns, ms, ps, i_off, j_off, k_off):
        """Generate rank-1 terms for a block of size ns x ms x ps at offsets."""
        if ns == 1 and ms == 1 and ps == 1:
            i = i_off; j = j_off; k = k_off
            add_rank1_term([i*m+j], [1.0], [j*p+k], [1.0], [k*n+i], [1.0])
            return
        
        if ns == 1:
            for j in range(ms):
                for k in range(ps):
                    ii = i_off
                    jj = j_off + j
                    kk = k_off + k
                    add_rank1_term([ii*m+jj], [1.0], [jj*p+kk], [1.0], [kk*n+ii], [1.0])
            return
        
        if ms == 1:
            for i in range(ns):
                for k in range(ps):
                    ii = i_off + i
                    jj = j_off
                    kk = k_off + k
                    add_rank1_term([ii*m+jj], [1.0], [jj*p+kk], [1.0], [kk*n+ii], [1.0])
            return
        
        if ps == 1:
            for i in range(ns):
                for j in range(ms):
                    ii = i_off + i
                    jj = j_off + j
                    kk = k_off
                    add_rank1_term([ii*m+jj], [1.0], [jj*p+kk], [1.0], [kk*n+ii], [1.0])
            return
        
        if ns == 2 and ms == 2 and ps == 2:
            # Strassen
            i0 = i_off; i1 = i_off+1
            j0 = j_off; j1 = j_off+1
            k0 = k_off; k1 = k_off+1
            
            add_rank1_term([i0*m+j0, i1*m+j1], [1, 1], [j0*p+k0, j1*p+k1], [1, 1], [k0*n+i0, k1*n+i1], [1, 1])
            add_rank1_term([i1*m+j0, i1*m+j1], [1, 1], [j0*p+k0], [1], [k0*n+i1, k1*n+i1], [1, -1])
            add_rank1_term([i0*m+j0], [1], [j0*p+k1, j1*p+k1], [1, -1], [k1*n+i0, k1*n+i1], [1, 1])
            add_rank1_term([i1*m+j1], [1], [j1*p+k0, j0*p+k0], [1, -1], [k0*n+i0, k0*n+i1], [1, 1])
            add_rank1_term([i0*m+j0, i0*m+j1], [1, 1], [j1*p+k1], [1], [k0*n+i0, k1*n+i0], [-1, 1])
            add_rank1_term([i1*m+j0, i0*m+j0], [1, -1], [j0*p+k0, j0*p+k1], [1, 1], [k1*n+i1], [1])
            add_rank1_term([i0*m+j1, i1*m+j1], [1, -1], [j1*p+k0, j1*p+k1], [1, 1], [k0*n+i0], [1])
            return
        
        # Fallback: schoolbook
        for i in range(ns):
            for j in range(ms):
                for k in range(ps):
                    ii = i_off + i
                    jj = j_off + j
                    kk = k_off + k
                    add_rank1_term([ii*m+jj], [1.0], [jj*p+kk], [1.0], [kk*n+ii], [1.0])
    
    if best_decomp is not None:
        nb, mb, pb = best_decomp
        i_off = 0
        for ni, ns in enumerate(nb):
            j_off = 0
            for mi, ms in enumerate(mb):
                k_off = 0
                for pi, ps in enumerate(pb):
                    gen_block(ns, ms, ps, i_off, j_off, k_off)
                    k_off += ps
                j_off += ms
            i_off += ns
    
    U = np.array(U_vecs).T
    V = np.array(V_vecs).T
    W = np.array(W_vecs).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
