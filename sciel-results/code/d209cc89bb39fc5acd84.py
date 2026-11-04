"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Matrix multiplication tensor decomposition for 3x5x7."""
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Optimal block decomposition:
    # n: [0,1] (size 2), [2] (size 1)
    # m: [0,1] (size 2), [2,3] (size 2), [4] (size 1)
    # p: [0,1] (size 2), [2,3] (size 2), [4,5,6] (size 3)
    
    n_splits = [([0,1], 2), ([2], 1)]
    m_splits = [([0,1], 2), ([2,3], 2), ([4], 1)]
    p_splits = [([0,1], 2), ([2,3], 2), ([4,5,6], 3)]
    
    terms = []
    
    def add_term(u_coeff, v_coeff, w_coeff):
        terms.append((u_coeff, v_coeff, w_coeff))
        
    def strassen_222_terms(n_rows, m_cols, p_rows):
        """Add Strassen terms for 2x2x2 block defined by indices."""
        nr = list(n_rows)
        mc = list(m_cols)
        pr = list(p_rows)
        
        # p1 = (a00+a11)(b00+b11)
        u = {(nr[0], mc[0]): 1, (nr[1], mc[1]): 1}
        v = {(mc[0], pr[0]): 1, (mc[1], pr[1]): 1}
        w = {(pr[0], nr[0]): 1, (pr[1], nr[1]): 1}
        add_term(u, v, w)
        
        # p2 = (a10+a11)b00
        u = {(nr[1], mc[0]): 1, (nr[1], mc[1]): 1}
        v = {(mc[0], pr[0]): 1}
        w = {(pr[0], nr[1]): 1}
        add_term(u, v, w)
        
        # p3 = a00(b01-b11)
        u = {(nr[0], mc[0]): 1}
        v = {(mc[0], pr[1]): 1, (mc[1], pr[1]): -1}
        w = {(pr[1], nr[0]): 1, (pr[1], nr[1]): 1}
        add_term(u, v, w)
        
        # p4 = a11(b10-b00)
        u = {(nr[1], mc[1]): 1}
        v = {(mc[1], pr[0]): 1, (mc[0], pr[0]): -1}
        w = {(pr[0], nr[0]): -1, (pr[0], nr[1]): 1}
        add_term(u, v, w)
        
        # p5 = (a00+a01)b11
        u = {(nr[0], mc[0]): 1, (nr[0], mc[1]): 1}
        v = {(mc[1], pr[1]): 1}
        w = {(pr[0], nr[0]): 1, (pr[1], nr[0]): 1}
        add_term(u, v, w)
        
        # p6 = (a10-a11)(b01+b10)
        u = {(nr[1], mc[0]): 1, (nr[1], mc[1]): -1}
        v = {(mc[0], pr[1]): 1, (mc[1], pr[0]): 1}
        w = {(pr[1], nr[1]): 1}
        add_term(u, v, w)
        
        # p7 = (a01-a11)(b00+b11)
        u = {(nr[0], mc[1]): 1, (nr[1], mc[1]): -1}
        v = {(mc[0], pr[0]): 1, (mc[1], pr[1]): 1}
        w = {(pr[0], nr[0]): -1}
        add_term(u, v, w)

    def schoolbook_terms(n_rows, m_cols, p_rows):
        """Add schoolbook terms for a block."""
        for i in n_rows:
            for j in m_cols:
                for k in p_rows:
                    u = {(i,j): 1}
                    v = {(j,k): 1}
                    w = {(k,i): 1}
                    add_term(u, v, w)

    def handle_block(n_rows, m_cols, p_rows, n_size, m_size, p_size):
        """Handle a block with optimal decomposition."""
        if n_size == 2 and m_size == 2:
            if p_size == 2:
                strassen_222_terms(n_rows, m_cols, p_rows)
            elif p_size == 3:
                # Split p into 2+1
                p_sub1 = list(p_rows[:2])
                p_sub2 = list(p_rows[2:])
                strassen_222_terms(n_rows, m_cols, p_sub1)
                schoolbook_terms(n_rows, m_cols, p_sub2)
            elif p_size == 1:
                schoolbook_terms(n_rows, m_cols, p_rows)
        else:
            schoolbook_terms(n_rows, m_cols, p_rows)

    for n_idx, (n_rows, n_size) in enumerate(n_splits):
        for m_idx, (m_cols, m_size) in enumerate(m_splits):
            for p_idx, (p_rows, p_size) in enumerate(p_splits):
                handle_block(n_rows, m_cols, p_rows, n_size, m_size, p_size)

    rank = len(terms)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for c, (u_coeff, v_coeff, w_coeff) in enumerate(terms):
        for (i,j), coeff in u_coeff.items():
            U[i*m+j, c] = coeff
        for (j,k), coeff in v_coeff.items():
            V[j*p+k, c] = coeff
        for (k,i), coeff in w_coeff.items():
            W[k*n+i, c] = coeff
            
    # Verify
    if time.time() < deadline:
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err > 1e-9:
            # Fallback to schoolbook
            rank = n * m * p
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
