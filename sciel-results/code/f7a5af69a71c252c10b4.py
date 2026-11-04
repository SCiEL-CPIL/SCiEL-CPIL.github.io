# EVOLVE-BLOCK-START
import time

def construct():
    t0 = time.time()
    n, m, p = 2, 3, 10
    rank = 55
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    terms = []
    
    # Generate Strassen terms for 5 blocks of 2x2
    for l in range(5):
        k0 = 2 * l
        k1 = 2 * l + 1
        
        # A indices (0-5)
        a00, a01, a10, a11 = 0, 1, 3, 4
        
        # B indices (0-29)
        b0k0, b0k1 = 0 * 10 + k0, 0 * 10 + k1
        b1k0, b1k1 = 1 * 10 + k0, 1 * 10 + k1
        
        # W indices (0-19)
        c0k0, c0k1 = k0 * 2 + 0, k1 * 2 + 0
        c1k0, c1k1 = k0 * 2 + 1, k1 * 2 + 1
        
        # M1
        terms.append(([(a00, 1), (a11, 1)], [(b0k0, 1), (b1k1, 1)], [(c0k0, 1), (c1k1, 1)]))
        # M2
        terms.append(([(a10, 1), (a11, 1)], [(b0k0, 1)], [(c1k0, 1), (c1k1, -1)]))
        # M3
        terms.append(([(a00, 1)], [(b0k1, 1), (b1k1, -1)], [(c0k1, 1), (c1k1, 1)]))
        # M4
        terms.append(([(a11, 1)], [(b1k0, 1), (b0k0, -1)], [(c0k0, 1), (c1k0, 1)]))
        # M5
        terms.append(([(a00, 1), (a01, 1)], [(b1k1, 1)], [(c0k0, -1), (c0k1, 1)]))
        # M6
        terms.append(([(a10, 1), (a00, -1)], [(b0k0, 1), (b0k1, 1)], [(c1k1, 1)]))
        # M7
        terms.append(([(a01, 1), (a11, -1)], [(b1k0, 1), (b1k1, 1)], [(c0k0, 1)]))

    # Schoolbook for j=2
    for i in range(2):
        for k in range(10):
            u_idx = i * 3 + 2
            v_idx = 2 * 10 + k
            w_idx = k * 2 + i
            terms.append(([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)]))
            
    # Fill matrices
    for t_idx, (u_terms, v_terms, w_terms) in enumerate(terms):
        for idx, coeff in u_terms:
            U_mat[idx][t_idx] = coeff
        for idx, coeff in v_terms:
            V_mat[idx][t_idx] = coeff
        for idx, coeff in w_terms:
            W_mat[idx][t_idx] = coeff

    elapsed = time.time() - t0
    if elapsed > 145:
        # Should not happen for this static construction, but guard just in case
        pass

    return {'U': U_mat, 'V': V_mat, 'W': W_mat}
# EVOLVE-BLOCK-END
