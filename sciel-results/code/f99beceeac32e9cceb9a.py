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
    
    # We need to find U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # such that for all i in [0,n), j in [0,m), k in [0,p):
    #   T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Strategy: Use block decomposition with Strassen where beneficial.
    # Split n=3 into 2+1, m=5 into 2+3, p=7 into 3+4
    # This gives 8 blocks: (2,2,3), (2,2,4), (2,3,3), (2,3,4), (1,2,3), (1,2,4), (1,3,3), (1,3,4)
    
    # For each block (a,b,c), we use the best known decomposition.
    # Strassen gives rank 7 for 2x2x2 blocks.
    # We can split larger blocks into 2x2x2 sub-blocks plus remainders.
    
    # Let's build the decomposition by processing each block.
    
    def strassen_222():
        """Return (U, V, W) for M_{2,2,2} with rank 7.
        U: (4, 7), V: (4, 7), W: (4, 7)
        Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # c=0: p1 = (a+d)(e+h)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # c=1: p2 = (c+d)e
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        
        # c=2: p3 = a(f-h)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1
        
        # c=3: p4 = d(g-e)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = -1; W[1, 3] = 1
        
        # c=4: p5 = (a+b)h
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1; W[2, 4] = 1
        
        # c=5: p6 = (c-d)(f+g)
        U[2, 5] = 1; U[3, 5] = -1
        V[1, 5] = 1; V[2, 5] = 1
        W[3, 5] = 1
        
        # c=6: p7 = (b-d)(e+h)
        U[1, 6] = 1; U[3, 6] = -1
        V[0, 6] = 1; V[3, 6] = 1
        W[0, 6] = -1
        
        return U, V, W
    
    # Define the splits:
    # n-split: [0,1] (size 2), [2] (size 1)
    # m-split: [0,1] (size 2), [2,3,4] (size 3)
    # p-split: [0,1,2] (size 3), [3,4,5,6] (size 4)
    
    n_splits = [([0,1], 2), ([2], 1)]
    m_splits = [([0,1], 2), ([2,3,4], 3)]
    p_splits = [([0,1,2], 3), ([3,4,5,6], 4)]
    
    # I'll build U, V, W as lists of terms, then combine.
    # Each term is (u_dict, v_dict, w_dict) where u_dict maps (i,j) -> coeff, etc.
    
    terms = []
    
    def add_term(u_coeff, v_coeff, w_coeff):
        """Add a single rank-1 term.
        u_coeff: dict mapping (i,j) -> coeff for U
        v_coeff: dict mapping (j,k) -> coeff for V
        w_coeff: dict mapping (k,i) -> coeff for W
        """
        terms.append((u_coeff, v_coeff, w_coeff))
    
    def add_strassen_block(n_rows, m_cols, p_rows):
        """Add Strassen decomposition for a 2x2x2 block."""
        U_s, V_s, W_s = strassen_222()
        for col in range(7):
            u_coeff = {}
            v_coeff = {}
            w_coeff = {}
            for i_local in range(2):
                for j_local in range(2):
                    i = n_rows[i_local]
                    j = m_cols[j_local]
                    coeff = U_s[i_local*2+j_local, col]
                    if coeff != 0:
                        u_coeff[(i,j)] = coeff
            for j_local in range(2):
                for k_local in range(2):
                    j = m_cols[j_local]
                    k = p_rows[k_local]
                    coeff = V_s[j_local*2+k_local, col]
                    if coeff != 0:
                        v_coeff[(j,k)] = coeff
            for k_local in range(2):
                for i_local in range(2):
                    k = p_rows[k_local]
                    i = n_rows[i_local]
                    coeff = W_s[k_local*2+i_local, col]
                    if coeff != 0:
                        w_coeff[(k,i)] = coeff
            add_term(u_coeff, v_coeff, w_coeff)
    
    def add_schoolbook_block(n_rows, m_cols, p_rows):
        """Add schoolbook decomposition for a block."""
        for i_local in range(len(n_rows)):
            for j_local in range(len(m_cols)):
                for k_local in range(len(p_rows)):
                    i = n_rows[i_local]
                    j = m_cols[j_local]
                    k = p_rows[k_local]
                    u_coeff = {(i,j): 1.0}
                    v_coeff = {(j,k): 1.0}
                    w_coeff = {(k,i): 1.0}
                    add_term(u_coeff, v_coeff, w_coeff)
    
    # Process each block
    for n_idx, (n_rows, n_size) in enumerate(n_splits):
        for m_idx, (m_cols, m_size) in enumerate(m_splits):
            for p_idx, (p_rows, p_size) in enumerate(p_splits):
                a, b, c = n_size, m_size, p_size
                
                if a == 2 and b == 2:
                    # For 2x2xc blocks, use Strassen for 2x2x2 sub-blocks
                    if c >= 2:
                        # Use Strassen for first 2 columns
                        p_sub = p_rows[:2]
                        add_strassen_block(n_rows, m_cols, p_sub)
                        # Schoolbook for remaining columns
                        if c > 2:
                            p_rem = p_rows[2:]
                            add_schoolbook_block(n_rows, m_cols, p_rem)
                    else:
                        # c == 1: schoolbook
                        add_schoolbook_block(n_rows, m_cols, p_rows)
                elif a == 2 and b == 3:
                    # For 2x3xc blocks, split m into 2+1
                    m_sub1 = m_cols[:2]
                    m_sub2 = m_cols[2:]
                    
                    # (2,2,c) part
                    if c >= 2:
                        p_sub = p_rows[:2]
                        add_strassen_block(n_rows, m_sub1, p_sub)
                        if c > 2:
                            p_rem = p_rows[2:]
                            add_schoolbook_block(n_rows, m_sub1, p_rem)
                    else:
                        add_schoolbook_block(n_rows, m_sub1, p_rows)
                    
                    # (2,1,c) part: schoolbook
                    add_schoolbook_block(n_rows, m_sub2, p_rows)
                else:
                    # Schoolbook for all other cases (a=1)
                    add_schoolbook_block(n_rows, m_cols, p_rows)
    
    # Now combine all terms into U, V, W matrices
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
