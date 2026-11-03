"""Rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # We'll build U, V, W dynamically
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Each factor matrix entry is a coefficient for the corresponding variable
    
    # Start with empty lists - we'll append columns as we add terms
    U = []  # Will be list of lists, each inner list is a row
    V = []
    W = []
    
    # Initialize with n*m, m*p, p*n rows respectively, each row starts empty
    for i in range(n * m):
        U.append([])
    for i in range(m * p):
        V.append([])
    for i in range(p * n):
        W.append([])
    
    def add_term(u_coeff, v_coeff, w_coeff):
        """Add one term to the decomposition.
        u_coeff: dict mapping (i,j) -> coefficient for A[i,j]
        v_coeff: dict mapping (j,k) -> coefficient for B[j,k]  
        w_coeff: dict mapping (i,k) -> coefficient for C[i,k]
        """
        # Append to each row
        for i in range(n * m):
            ij = (i // m, i % m)
            U[i].append(u_coeff.get(ij, 0))
        for i in range(m * p):
            jk = (i // p, i % p)
            V[i].append(v_coeff.get(jk, 0))
        for i in range(p * n):
            ki = (i // n, i % n)
            W[i].append(w_coeff.get(ki, 0))
    
    def schoolbook_block(n_start, n_end, m_start, m_end, p_start, p_end):
        """Add schoolbook terms for the block A[n_start:n_end, m_start:m_end] * B[m_start:m_end, p_start:p_end]"""
        for i in range(n_start, n_end):
            for j in range(m_start, m_end):
                for k in range(p_start, p_end):
                    add_term(
                        {(i, j): 1},
                        {(j, k): 1},
                        {(i, k): 1}
                    )
    
    def strassen_block(n_start, m_start, p_start):
        """Add Strassen terms for a 2x2x2 block starting at (n_start, m_start, p_start)"""
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[i, j], [k, l]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        n0, n1 = n_start, n_start + 1
        m0, m1 = m_start, m_start + 1
        p0, p1 = p_start, p_start + 1
        
        # M1 = (a+d)(e+h)
        add_term(
            {(n0, m0): 1, (n1, m0): 1},
            {(m0, p0): 1, (m0, p1): 1},
            {(n0, p0): 1, (n1, p1): 1}
        )
        
        # M2 = (c+d)e
        add_term(
            {(n1, m0): 1, (n1, m1): 1},
            {(m0, p0): 1},
            {(n1, p0): 1, (n1, p1): -1}
        )
        
        # M3 = a(f-h)
        add_term(
            {(n0, m0): 1},
            {(m0, p1): 1, (m1, p1): -1},
            {(n0, p1): 1, (n1, p0): 1}
        )
        
        # M4 = d(g-e)
        add_term(
            {(n1, m1): 1},
            {(m1, p0): 1, (m0, p0): -1},
            {(n0, p0): 1, (n1, p0): 1}
        )
        
        # M5 = (a+b)h
        add_term(
            {(n0, m0): 1, (n0, m1): 1},
            {(m1, p1): 1},
            {(n0, p0): -1, (n0, p1): 1}
        )
        
        # M6 = (c-a)(e+f)
        add_term(
            {(n1, m0): 1, (n0, m0): -1},
            {(m0, p0): 1, (m0, p1): 1},
            {(n1, p1): 1}
        )
        
        # M7 = (b-d)(g+h)
        add_term(
            {(n0, m1): 1, (n1, m1): -1},
            {(m1, p0): 1, (m1, p1): 1},
            {(n0, p0): 1}
        )
    
    # Strategy: Split the 3x5x7 problem into blocks
    # A is 3x5, B is 5x7
    # Split m into 2+2+1
    # For m=0,1: use Strassen for n=0,1 and p=0,1 / p=2,3 / p=4,5, then schoolbook for p=6
    # For m=2,3: same as above
    # For m=4: schoolbook for all n, p
    
    # Block 1: n=0,1, m=0,1, p=0,1 -> Strassen
    strassen_block(0, 0, 0)
    # Block 2: n=0,1, m=0,1, p=2,3 -> Strassen
    strassen_block(0, 0, 2)
    # Block 3: n=0,1, m=0,1, p=4,5 -> Strassen
    strassen_block(0, 0, 4)
    # Block 4: n=0,1, m=0,1, p=6,6 -> Schoolbook
    schoolbook_block(0, 2, 0, 2, 6, 7)
    # Block 5: n=2, m=0,1, p=0,6 -> Schoolbook
    schoolbook_block(2, 3, 0, 2, 0, 7)
    
    # Block 6: n=0,1, m=2,3, p=0,1 -> Strassen
    strassen_block(0, 2, 0)
    # Block 7: n=0,1, m=2,3, p=2,3 -> Strassen
    strassen_block(0, 2, 2)
    # Block 8: n=0,1, m=2,3, p=4,5 -> Strassen
    strassen_block(0, 2, 4)
    # Block 9: n=0,1, m=2,3, p=6,6 -> Schoolbook
    schoolbook_block(0, 2, 2, 4, 6, 7)
    # Block 10: n=2, m=2,3, p=0,6 -> Schoolbook
    schoolbook_block(2, 3, 2, 4, 0, 7)
    
    # Block 11: n=0,2, m=4, p=0,6 -> Schoolbook
    schoolbook_block(0, 3, 4, 5, 0, 7)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
