import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Theoretical rank for 2x4x5 is 36 based on 2 * Rank(2x2x5) = 2 * 18
    # We construct this directly.
    
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col_idx = 0
    
    # Strassen terms for 2x2x2
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    strassen_terms = [
        # m1: (x00+x11)(y00+y11) -> z00+1, z11+1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # m2: (x10+x11)y00 -> z10+1, z11-1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # m3: x00(y01-y11) -> z01+1, z11+1
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # m4: x11(y10-y00) -> z00+1, z10+1
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # m5: (x00+x01)y11 -> z00-1, z01+1
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # m6: (x10-x00)(y00+y01) -> z11+1
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # m7: (x01-x11)(y10+y11) -> z00+1
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    # Standard terms for 2x2x1
    # z0 = x00*y0 + x01*y1
    # z1 = x10*y0 + x11*y1
    standard_terms_2x2x1 = [
        ({0: 1}, {0: 1}, {0: 1}),
        ({1: 1}, {1: 1}, {0: 1}),
        ({2: 1}, {0: 1}, {1: 1}),
        ({3: 1}, {1: 1}, {1: 1})
    ]
    
    def add_term(u_coeffs, v_coeffs, w_coeffs, idx):
        for (i, j), val in u_coeffs.items():
            U[i * m + j, idx] += val
        for (j, k), val in v_coeffs.items():
            V[j * p + k, idx] += val
        for (k, i), val in w_coeffs.items():
            W[k * n + i, idx] += val

    # Iterate over the two blocks of the middle dimension (j)
    for block_q in range(2):
        # j_offset = block_q * 2
        j_offset = block_q * 2
        
        # Iterate over the three segments of the output dimension (k)
        # Segments: 0-1 (Strassen), 2-3 (Strassen), 4 (Standard)
        for seg_s in range(3):
            if seg_s < 2:
                # 2x2x2 Strassen block
                k_offset = seg_s * 2
                for u_c, v_c, w_c in strassen_terms:
                    # Map local indices to global
                    u_g = {}
                    v_g = {}
                    w_g = {}
                    
                    for (li, lj), val in u_c.items():
                        u_g[(li, j_offset + lj)] = val
                    for (lj, lk), val in v_c.items():
                        v_g[(j_offset + lj, k_offset + lk)] = val
                    for (lk, li), val in w_c.items():
                        w_g[(k_offset + lk, li)] = val
                        
                    add_term(u_g, v_g, w_g, col_idx)
                    col_idx += 1
            else:
                # 2x2x1 Standard block
                k_offset = 4
                for u_c, v_c, w_c in standard_terms_2x2x1:
                    u_g = {}
                    v_g = {}
                    w_g = {}
                    
                    for (li, lj), val in u_c.items():
                        u_g[(li, j_offset + lj)] = val
                    for (lj, _), val in v_c.items():
                        v_g[(j_offset + lj, k_offset)] = val
                    for (_, li), val in w_c.items():
                        w_g[(k_offset, li)] = val
                        
                    add_term(u_g, v_g, w_g, col_idx)
                    col_idx += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
