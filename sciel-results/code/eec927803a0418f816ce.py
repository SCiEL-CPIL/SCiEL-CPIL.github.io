"""Matrix multiplication 4,4,5 - Strassen block decomposition with rank minimization."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Index mappings:
    # A[i,j] -> U[i*m+j]
    # B[j,k] -> V[j*p+k]
    # C[i,k] -> W[k*n+i]
    
    # We'll build terms as (U_dict, V_dict, W_dict)
    terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
    
    def strassen_2x2(A_idx, B_idx, C_idx):
        """
        Add 7 Strassen terms for 2x2 * 2x2 = 2x2.
        A_idx: dict {(0,0),(0,1),(1,0),(1,1)} -> U row indices
        B_idx: dict {(0,0),(0,1),(1,0),(1,1)} -> V row indices
        C_idx: dict {(0,0),(0,1),(1,0),(1,1)} -> W row indices
        """
        x11, x12 = A_idx[(0,0)], A_idx[(0,1)]
        x21, x22 = A_idx[(1,0)], A_idx[(1,1)]
        y11, y12 = B_idx[(0,0)], B_idx[(0,1)]
        y21, y22 = B_idx[(1,0)], B_idx[(1,1)]
        z11, z12 = C_idx[(0,0)], C_idx[(0,1)]
        z21, z22 = C_idx[(1,0)], C_idx[(1,1)]
        
        # M1 = (x11+x22)*(y11+y22) -> z11 + z22
        add_term({x11:1, x22:1}, {y11:1, y22:1}, {z11:1, z22:1})
        # M2 = (x21+x22)*y11 -> z21 - z22
        add_term({x21:1, x22:1}, {y11:1}, {z21:1, z22:-1})
        # M3 = x11*(y12-y22) -> z12 + z22
        add_term({x11:1}, {y12:1, y22:-1}, {z12:1, z22:1})
        # M4 = x22*(y21-y11) -> z11 + z21
        add_term({x22:1}, {y21:1, y11:-1}, {z11:1, z21:1})
        # M5 = (x11+x12)*y22 -> z12 - z11
        add_term({x11:1, x12:1}, {y22:1}, {z12:1, z11:-1})
        # M6 = (x21-x11)*(y11+y12) -> z22
        add_term({x21:1, x11:-1}, {y11:1, y12:1}, {z22:1})
        # M7 = (x12-x22)*(y21+y22) -> z11 + z22
        add_term({x12:1, x22:-1}, {y21:1, y22:1}, {z11:1, z22:1})
    
    def schoolbook_2x1(A_idx, B_idx, C_idx):
        """
        Add 4 schoolbook terms for 2x2 * 2x1 = 2x1.
        A_idx: dict {(0,0),(0,1),(1,0),(1,1)} -> U row indices
        B_idx: dict {(0,0),(1,0)} -> V row indices
        C_idx: dict {(0,0),(1,0)} -> W row indices
        """
        for i in [0, 1]:
            for j in [0, 1]:
                add_term({A_idx[(i,j)]: 1}, {B_idx[(j,0)]: 1}, {C_idx[(i,0)]: 1})
    
    # Block index mappings
    # A blocks (2x2 each)
    A11 = {(0,0): 0, (0,1): 1, (1,0): 4, (1,1): 5}
    A12 = {(0,0): 2, (0,1): 3, (1,0): 6, (1,1): 7}
    A21 = {(0,0): 8, (0,1): 9, (1,0): 12, (1,1): 13}
    A22 = {(0,0): 10, (0,1): 11, (1,0): 14, (1,1): 15}
    
    # B blocks
    B11 = {(0,0): 0, (0,1): 1, (1,0): 5, (1,1): 6}
    B12 = {(0,0): 2, (0,1): 3, (1,0): 7, (1,1): 8}
    B13 = {(0,0): 4, (1,0): 9}
    B21 = {(0,0): 10, (0,1): 11, (1,0): 15, (1,1): 16}
    B22 = {(0,0): 12, (0,1): 13, (1,0): 17, (1,1): 18}
    B23 = {(0,0): 14, (1,0): 19}
    
    # C blocks (W index = k*n + i)
    C11 = {(0,0): 0, (0,1): 4, (1,0): 1, (1,1): 5}
    C12 = {(0,0): 8, (0,1): 12, (1,0): 9, (1,1): 13}
    C13 = {(0,0): 16, (1,0): 17}
    C21 = {(0,0): 2, (0,1): 6, (1,0): 3, (1,1): 7}
    C22 = {(0,0): 10, (0,1): 14, (1,0): 11, (1,1): 15}
    C23 = {(0,0): 18, (1,0): 19}
    
    # Build the decomposition
    # C11 = A11*B11 + A12*B21 (two 2x2*2x2 products)
    strassen_2x2(A11, B11, C11)
    strassen_2x2(A12, B21, C11)
    
    # C12 = A11*B12 + A12*B22
    strassen_2x2(A11, B12, C12)
    strassen_2x2(A12, B22, C12)
    
    # C13 = A11*B13 + A12*B23 (two 2x2*2x1 products)
    schoolbook_2x1(A11, B13, C13)
    schoolbook_2x1(A12, B23, C13)
    
    # C21 = A21*B11 + A22*B21
    strassen_2x2(A21, B11, C21)
    strassen_2x2(A22, B21, C21)
    
    # C22 = A21*B12 + A22*B22
    strassen_2x2(A21, B12, C22)
    strassen_2x2(A22, B22, C22)
    
    # C23 = A21*B13 + A22*B23
    schoolbook_2x1(A21, B13, C23)
    schoolbook_2x1(A22, B23, C23)
    
    rank = len(terms)
    
    # Build U, V, W matrices
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for row, coeff in u_dict.items():
            U[row, c] = coeff
        for row, coeff in v_dict.items():
            V[row, c] = coeff
        for row, coeff in w_dict.items():
            W[row, c] = coeff
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fall back to schoolbook
        rank_sb = n*m*p
        U = np.zeros((nm, rank_sb), dtype=np.float64)
        V = np.zeros((mp, rank_sb), dtype=np.float64)
        W = np.zeros((pn, rank_sb), dtype=np.float64)
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U[ii*m+jj, col] = 1.0
                    V[jj*p+kk, col] = 1.0
                    W[kk*n+ii, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by merging terms with identical W patterns
    # Check if any two terms have the same W column (up to scaling)
    # and can be combined
    
    # For now, return the Strassen decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
