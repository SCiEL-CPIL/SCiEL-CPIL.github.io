"""Matrix multiplication 3x5x7 rank-minimal decomposition."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

def construct():
    """Matrix multiplication tensor decomposition for 3x5x7."""
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # We'll build a list of rank-1 terms.
    # Each term: (u_dict, v_dict, w_dict)
    # u_dict: {(i,j): coeff} for A[i,j]
    # v_dict: {(j,k): coeff} for B[j,k]  
    # w_dict: {(k,i): coeff} for C[k,i] (trace convention W[k*n+i])
    
    terms = []
    
    def add_term(u_coeff, v_coeff, w_coeff):
        terms.append((u_coeff, v_coeff, w_coeff))
    
    def schoolbook_block(n_rows, m_cols, p_rows):
        """Add schoolbook terms for a block."""
        for i in n_rows:
            for j in m_cols:
                for k in p_rows:
                    add_term({(i,j): 1.0}, {(j,k): 1.0}, {(k,i): 1.0})
    
    def strassen_222(n_rows, m_cols, p_rows):
        """
        Strassen rank-7 for 2x2x2 block.
        n_rows = [n0, n1], m_cols = [m0, m1], p_rows = [p0, p1]
        
        Standard Strassen:
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B10-B00)
        M4 = A11(B11-B01)
        M5 = (A00+A01)B11
        M6 = (A10-A00)(B00+B01)
        M7 = (A01-A11)(B10+B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        n0, n1 = n_rows[0], n_rows[1]
        m0, m1 = m_cols[0], m_cols[1]
        p0, p1 = p_rows[0], p_rows[1]
        
        # M1 = (A[n0,m0]+A[n1,m1])(B[m0,p0]+B[m0,p1])
        # Wait, that's not right. Let me be more careful.
        # A is 2x2: A[0,0], A[0,1], A[1,0], A[1,1]
        # B is 2x2: B[0,0], B[0,1], B[1,0], B[1,1]
        # In our notation:
        # A[i,j] where i in {n0,n1}, j in {m0,m1}
        # B[j,k] where j in {m0,m1}, k in {p0,p1}
        
        # M1 = (A[n0,m0]+A[n1,m1])*(B[m0,p0]+B[m1,p1])
        add_term(
            {(n0,m0): 1.0, (n1,m1): 1.0},
            {(m0,p0): 1.0, (m1,p1): 1.0},
            {(p0,n0): 1.0, (p1,n0): 1.0, (p0,n1): 1.0, (p1,n1): 1.0}
        )
        
        # M2 = (A[n1,m0]+A[n1,m1])*B[m0,p0]
        add_term(
            {(n1,m0): 1.0, (n1,m1): 1.0},
            {(m0,p0): 1.0},
            {(p0,n1): 1.0}
        )
        
        # M3 = A[n0,m0]*(B[m1,p0]-B[m0,p0])
        add_term(
            {(n0,m0): 1.0},
            {(m1,p0): 1.0, (m0,p0): -1.0},
            {(p0,n0): 1.0, (p0,n1): 1.0}
        )
        
        # M4 = A[n1,m1]*(B[m1,p1]-B[m0,p1])
        add_term(
            {(n1,m1): 1.0},
            {(m1,p1): 1.0, (m0,p1): -1.0},
            {(p1,n0): 1.0, (p1,n1): 1.0}
        )
        
        # M5 = (A[n0,m0]+A[n0,m1])*B[m1,p1]
        add_term(
            {(n0,m0): 1.0, (n0,m1): 1.0},
            {(m1,p1): 1.0},
            {(p1,n0): -1.0, (p1,n1): -1.0}
        )
        
        # M6 = (A[n1,m0]-A[n0,m0])*(B[m0,p0]+B[m0,p1])
        add_term(
            {(n1,m0): 1.0, (n0,m0): -1.0},
            {(m0,p0): 1.0, (m0,p1): 1.0},
            {(p1,n1): 1.0}
        )
        
        # M7 = (A[n0,m1]-A[n1,m1])*(B[m1,p0]+B[m1,p1])
        add_term(
            {(n0,m1): 1.0, (n1,m1): -1.0},
            {(m1,p0): 1.0, (m1,p1): 1.0},
            {(p0,n0): 1.0, (p0,n1): -1.0}
        )
    
    def laderman_233(n_rows, m_cols, p_rows):
        """
        Laderman rank-7 for 2x3x3 block.
        n_rows = [n0, n1] (2 rows of A)
        m_cols = [m0, m1, m2] (3 cols of A, 3 rows of B)
        p_rows = [p0, p1, p2] (3 cols of B, 3 rows of C)
        
        A is 2x3, B is 3x3, C is 2x3.
        
        Laderman's algorithm for 2x3x3:
        Let A = [[a0, a1, a2], [b0, b1, b2]]
        Let B = [[c0, c1, c2], [d0, d1, d2], [e0, e1, e2]]
        
        M1 = (a0+a2)*(c0+e0)
        M2 = (a1+a2)*(d0+e0)  
        M3 = a0*(e1-c1)
        M4 = a2*(c2-d2)
        M5 = (a0+a1)*(d2+e2)
        M6 = (b0-b2)*(c0+c1)
        M7 = (b1-b2)*(d0+d1)
        
        C[0,0] = M1 + M4 - M5 + M7
        C[0,1] = M3 + M5
        C[0,2] = M1 + M2 - M3 - M6
        C[1,0] = M6
        C[1,1] = M7
        C[1,2] = M2
        
        Wait, I need to verify this. Let me use a known correct formulation.
        
        Actually, let me use a simpler approach: just use schoolbook for all blocks.
        The total rank would be 3*5*7 = 105.
        
        But we can do better with some Strassen blocks.
        """
        # For now, fall back to schoolbook
        schoolbook_block(n_rows, m_cols, p_rows)
    
    # Strategy: partition into blocks and use best algorithm for each
    # n=3: split as [0,1] (size 2) and [2] (size 1)
    # m=5: split as [0,1] (size 2), [2,3] (size 2), [4] (size 1)
    # p=7: split as [0,1] (size 2), [2,3] (size 2), [4,5] (size 2), [6] (size 1)
    
    n_blocks = [([0,1], 2), ([2], 1)]
    m_blocks = [([0,1], 2), ([2,3], 2), ([4], 1)]
    p_blocks = [([0,1], 2), ([2,3], 2), ([4,5], 2), ([6], 1)]
    
    for n_rows, n_size in n_blocks:
        for m_cols, m_size in m_blocks:
            for p_rows, p_size in p_blocks:
                if n_size == 2 and m_size == 2 and p_size == 2:
                    strassen_222(n_rows, m_cols, p_rows)
                else:
                    schoolbook_block(n_rows, m_cols, p_rows)
    
    # Combine terms into matrices
    rank = len(terms)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for col, (u_coeff, v_coeff, w_coeff) in enumerate(terms):
        for (i,j), coeff in u_coeff.items():
            U[i*m+j, col] = coeff
        for (j,k), coeff in v_coeff.items():
            V[j*p+k, col] = coeff
        for (k,i), coeff in w_coeff.items():
            W[k*n+i, col] = coeff
    
    # Verify with numpy
    T = np.einsum('ic,jc,kc->ijk', U, V, W)
    
    # Check that T[i,j,k] = 1 if this corresponds to C[i,k] = sum_j A[i,j]*B[j,k]
    # Actually T[i,j,k] should be 1 for the "diagonal" terms where we're computing C[i,k]
    # The tensor for matrix mult has T[i,j,k] = 1 if we're summing over the middle index
    # More precisely: C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i,j,k (it's the standard matrix mult tensor)
    
    max_err = np.max(np.abs(T - 1.0))
    
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
