import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Global dimensions
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # We will build lists of (U_row, V_row, W_row, coeff) for each term
    # Or just build the matrices directly.
    # Since we are stacking, we can just append columns.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        """
        u_vec: list of (idx, val)
        v_vec: list of (idx, val)
        w_vec: list of (idx, val)
        """
        u_col = np.zeros(n_m)
        v_col = np.zeros(m_p)
        w_col = np.zeros(p_n)
        
        for idx, val in u_vec:
            u_col[idx] = val
        for idx, val in v_vec:
            v_col[idx] = val
        for idx, val in w_vec:
            w_col[idx] = val
            
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
        
    def schoolbook(n_prime, m_prime, p_prime, i_offset, j_offset, k_offset):
        """
        Adds schoolbook decomposition for a sub-block.
        i_offset: global i start
        j_offset: global j start
        k_offset: global k start
        """
        for i_loc in range(n_prime):
            for j_loc in range(m_prime):
                for k_loc in range(p_prime):
                    i_glob = i_offset + i_loc
                    j_glob = j_offset + j_loc
                    k_glob = k_offset + k_loc
                    
                    I = i_glob * m + j_glob
                    J = j_glob * p + k_glob
                    K = k_glob * n + i_glob
                    
                    u_vec = [(I, 1)]
                    v_vec = [(J, 1)]
                    w_vec = [(K, 1)]
                    add_term(u_vec, v_vec, w_vec)
                    
    def strassen_2x2x2(i_offset, j_offset, k_offset):
        """
        Adds Strassen decomposition for a 2x2x2 sub-block.
        """
        # Local indices 0,1 map to global i_offset+0, i_offset+1 etc.
        
        # Term 1: (A00+A11)(B00+B11) -> C00, C11
        # U: A00 (I=0), A11 (I=3)
        # V: B00 (J=0), B11 (J=3)
        # W: C00 (K=0), C11 (K=3)
        
        # Helper to get global I, J, K
        def get_I(i_l, j_l): return (i_offset + i_l) * m + (j_offset + j_l)
        def get_J(j_l, k_l): return (j_offset + j_l) * p + (k_offset + k_l)
        def get_K(k_l, i_l): return (k_offset + k_l) * n + (i_offset + i_l)
        
        # M1
        add_term(
            [(get_I(0,0), 1), (get_I(1,1), 1)],
            [(get_J(0,0), 1), (get_J(1,1), 1)],
            [(get_K(0,0), 1), (get_K(1,1), 1)]
        )
        
        # M2: (A10+A11)B00 -> C10 (+1), C11 (-1)
        add_term(
            [(get_I(1,0), 1), (get_I(1,1), 1)],
            [(get_J(0,0), 1)],
            [(get_K(0,1), 1), (get_K(1,1), -1)]
        )
        
        # M3: A00(B01-B11) -> C01 (+1), C11 (+1)
        add_term(
            [(get_I(0,0), 1)],
            [(get_J(0,1), 1), (get_J(1,1), -1)],
            [(get_K(1,0), 1), (get_K(1,1), 1)]
        )
        
        # M4: A11(B10-B00) -> C00 (+1), C10 (+1)
        add_term(
            [(get_I(1,1), 1)],
            [(get_J(1,0), 1), (get_J(0,0), -1)],
            [(get_K(0,0), 1), (get_K(0,1), 1)]
        )
        
        # M5: (A00+A01)B11 -> C00 (-1), C01 (+1)
        add_term(
            [(get_I(0,0), 1), (get_I(0,1), 1)],
            [(get_J(1,1), 1)],
            [(get_K(0,0), -1), (get_K(1,0), 1)]
        )
        
        # M6: (A10-A00)(B00+B01) -> C11 (+1)
        add_term(
            [(get_I(1,0), 1), (get_I(0,0), -1)],
            [(get_J(0,0), 1), (get_J(0,1), 1)],
            [(get_K(1,1), 1)]
        )
        
        # M7: (A01-A11)(B10+B11) -> C00 (+1)
        add_term(
            [(get_I(0,1), 1), (get_I(1,1), -1)],
            [(get_J(1,0), 1), (get_J(1,1), 1)],
            [(get_K(0,0), 1)]
        )

    # Sub-problem 1: A11 B11 (2x2x2) -> Strassen
    # i in {0,1}, j in {0,1}, k in {0,1}
    strassen_2x2x2(0, 0, 0)
    
    # Sub-problem 2: A12 B21 (2x3x2) -> Schoolbook
    # i in {0,1}, j in {2,3,4}, k in {0,1}
    schoolbook(2, 3, 2, 0, 2, 0)
    
    # Sub-problem 3: A11 B12 (2x2x3) -> Schoolbook
    # i in {0,1}, j in {0,1}, k in {2,3,4}
    schoolbook(2, 2, 3, 0, 0, 2)
    
    # Sub-problem 4: A12 B22 (2x3x3) -> Schoolbook
    # i in {0,1}, j in {2,3,4}, k in {2,3,4}
    schoolbook(2, 3, 3, 0, 2, 2)
    
    # Sub-problem 5: A21 B11 (2x2x2) -> Strassen
    # i in {2,3}, j in {0,1}, k in {0,1}
    strassen_2x2x2(2, 0, 0)
    
    # Sub-problem 6: A22 B21 (2x3x2) -> Schoolbook
    # i in {2,3}, j in {2,3,4}, k in {0,1}
    schoolbook(2, 3, 2, 2, 2, 0)
    
    # Sub-problem 7: A21 B12 (2x2x3) -> Schoolbook
    # i in {2,3}, j in {0,1}, k in {2,3,4}
    schoolbook(2, 2, 3, 2, 0, 2)
    
    # Sub-problem 8: A22 B22 (2x3x3) -> Schoolbook
    # i in {2,3}, j in {2,3,4}, k in {2,3,4}
    schoolbook(2, 3, 3, 2, 2, 2)
    
    # Stack columns
    if not U_cols:
        # Fallback
        return {"U": [[0]*100 for _ in range(20)], "V": [[0]*100 for _ in range(25)], "W": [[0]*100 for _ in range(20)]}
        
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
