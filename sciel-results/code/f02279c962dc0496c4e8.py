import numpy as np

def construct():
    n, m, p = 3, 3, 4
    
    dim_A = n * m
    dim_B = m * p
    dim_C = p * n
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_component(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        u = np.zeros(dim_A)
        v = np.zeros(dim_B)
        w = np.zeros(dim_C)
        for idx, val in zip(u_indices, u_vals):
            u[idx] = val
        for idx, val in zip(v_indices, v_vals):
            v[idx] = val
        for idx, val in zip(w_indices, w_vals):
            w[idx] = val
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
        
    def a_idx(i, j): return i * m + j
    def b_idx(j, k): return j * p + k
    def c_idx(k, i): return k * n + i

    def strassen_block(a_i_start, a_j_start, b_j_start, b_k_start, c_k_start, c_i_start):
        A00 = a_idx(a_i_start, a_j_start)
        A01 = a_idx(a_i_start, a_j_start+1)
        A10 = a_idx(a_i_start+1, a_j_start)
        A11 = a_idx(a_i_start+1, a_j_start+1)
        
        B00 = b_idx(b_j_start, b_k_start)
        B01 = b_idx(b_j_start, b_k_start+1)
        B10 = b_idx(b_j_start+1, b_k_start)
        B11 = b_idx(b_j_start+1, b_k_start+1)
        
        C00 = c_idx(c_k_start, c_i_start)
        C01 = c_idx(c_k_start+1, c_i_start)
        C10 = c_idx(c_k_start, c_i_start+1)
        C11 = c_idx(c_k_start+1, c_i_start+1)
        
        add_component([A00, A11], [1, 1], [B00, B11], [1, 1], [C00, C11], [1, 1])
        add_component([A10, A11], [1, 1], [B00], [1], [C10, C11], [1, -1])
        add_component([A00], [1], [B10, B11], [1, -1], [C01, C11], [1, 1])
        add_component([A11], [1], [B01, B00], [1, -1], [C00, C10], [1, 1])
        add_component([A00, A01], [1, 1], [B11], [1], [C00, C01], [-1, 1])
        add_component([A10, A00], [1, -1], [B00, B10], [1, 1], [C11], [1])
        add_component([A01, A11], [1, -1], [B11, B01], [1, 1], [C00, C11], [1, 1])
        
    strassen_block(0, 0, 0, 0, 0, 0)
    strassen_block(0, 0, 0, 2, 2, 0)
    
    # Block 3: A12 (0,2), B21 (2,0), C11 (0,0)
    add_component([a_idx(0,2)], [1], [b_idx(2,0)], [1], [c_idx(0,0)], [1])
    add_component([a_idx(0,2)], [1], [b_idx(2,1)], [1], [c_idx(1,0)], [1])
    add_component([a_idx(1,2)], [1], [b_idx(2,0)], [1], [c_idx(0,1)], [1])
    add_component([a_idx(1,2)], [1], [b_idx(2,1)], [1], [c_idx(1,1)], [1])
    
    # Block 4: A12 (0,2), B22 (2,2), C12 (2,0)
    add_component([a_idx(0,2)], [1], [b_idx(2,2)], [1], [c_idx(2,0)], [1])
    add_component([a_idx(0,2)], [1], [b_idx(2,3)], [1], [c_idx(3,0)], [1])
    add_component([a_idx(1,2)], [1], [b_idx(2,2)], [1], [c_idx(2,1)], [1])
    add_component([a_idx(1,2)], [1], [b_idx(2,3)], [1], [c_idx(3,1)], [1])
    
    # Block 5: A21 (2,0), B11 (0,0), C21 (0,2)
    add_component([a_idx(2,0), a_idx(2,1)], [1, 1], [b_idx(0,0), b_idx(1,0)], [1, 1], [c_idx(0,2)], [1])
    add_component([a_idx(2,0), a_idx(2,1)], [1, 1], [b_idx(0,1), b_idx(1,1)], [1, 1], [c_idx(1,2)], [1])
    
    # Block 6: A21 (2,0), B12 (0,2), C22 (2,2)
    add_component([a_idx(2,0), a_idx(2,1)], [1, 1], [b_idx(0,2), b_idx(1,2)], [1, 1], [c_idx(2,2)], [1])
    add_component([a_idx(2,0), a_idx(2,1)], [1, 1], [b_idx(0,3), b_idx(1,3)], [1, 1], [c_idx(3,2)], [1])
    
    # Block 7: A22 (2,2), B21 (2,0), C21 (0,2)
    add_component([a_idx(2,2)], [1], [b_idx(2,0)], [1], [c_idx(0,2)], [1])
    add_component([a_idx(2,2)], [1], [b_idx(2,1)], [1], [c_idx(1,2)], [1])
    
    # Block 8: A22 (2,2), B22 (2,2), C22 (2,2)
    add_component([a_idx(2,2)], [1], [b_idx(2,2)], [1], [c_idx(2,2)], [1])
    add_component([a_idx(2,2)], [1], [b_idx(2,3)], [1], [c_idx(3,2)], [1])
    
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
