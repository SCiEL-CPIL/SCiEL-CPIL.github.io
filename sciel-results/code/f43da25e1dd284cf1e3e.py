import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (20, rank)
    # W shape: (p*n, rank) = (10, rank)
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_dict, v_dict, w_dict):
        """Add a rank-1 term. u_dict: {A_index: coeff}, v_dict: {B_index: coeff}, w_dict: {C_index: coeff}"""
        u = [0.0] * (n * m)
        for idx, coeff in u_dict.items():
            u[idx] = coeff
        v = [0.0] * (m * p)
        for idx, coeff in v_dict.items():
            v[idx] = coeff
        w = [0.0] * (p * n)
        for idx, coeff in w_dict.items():
            w[idx] = coeff
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
    
    def a_idx(i, j):
        """A[i,j] index in U"""
        return i * m + j
    
    def b_idx(j, k):
        """B[j,k] index in V"""
        return j * p + k
    
    def c_idx(i, k):
        """C[i,k] index in W, using convention W[k*n+i]"""
        return k * n + i
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A local: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # B local: (0,0)->0, (0,1)->1, (1,0)->5, (1,1)->6
    # C local: (0,0)->c_idx(0,0)=0, (0,1)->c_idx(0,1)=2, (1,0)->c_idx(1,0)=1, (1,1)->c_idx(1,1)=3
    
    # Strassen for block (i_offset=0, j_offset=0, k_offset=0)
    def strassen_block(i_off, j_off, k_off):
        """
        A block: rows i_off..i_off+1, cols j_off..j_off+1
        B block: rows j_off..j_off+1, cols k_off..k_off+1
        C block: rows i_off..i_off+1, cols k_off..k_off+1
        """
        # A indices
        a00 = a_idx(i_off, j_off)
        a01 = a_idx(i_off, j_off + 1)
        a10 = a_idx(i_off + 1, j_off)
        a11 = a_idx(i_off + 1, j_off + 1)
        
        # B indices
        b00 = b_idx(j_off, k_off)
        b01 = b_idx(j_off, k_off + 1)
        b10 = b_idx(j_off + 1, k_off)
        b11 = b_idx(j_off + 1, k_off + 1)
        
        # C indices (W convention: k*n + i)
        c00 = c_idx(i_off, k_off)
        c01 = c_idx(i_off, k_off + 1)
        c10 = c_idx(i_off + 1, k_off)
        c11 = c_idx(i_off + 1, k_off + 1)
        
        # Strassen terms
        # m1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        add_term({a00: 1, a11: 1}, {b00: 1, b11: 1}, {c00: 1, c11: 1})
        # m2 = (A10+A11)B00 -> C10 += 1, C11 -= 1
        add_term({a10: 1, a11: 1}, {b00: 1}, {c10: 1, c11: -1})
        # m3 = A00(B01-B11) -> C01 += 1, C11 += 1
        add_term({a00: 1}, {b01: 1, b11: -1}, {c01: 1, c11: 1})
        # m4 = A11(B10-B00) -> C00 += 1, C10 += 1
        add_term({a11: 1}, {b10: 1, b00: -1}, {c00: 1, c10: 1})
        # m5 = (A00+A01)B11 -> C00 -= 1, C01 += 1
        add_term({a00: 1, a01: 1}, {b11: 1}, {c00: -1, c01: 1})
        # m6 = (A10-A00)(B00+B01) -> C11 += 1
        add_term({a10: 1, a00: -1}, {b00: 1, b01: 1}, {c11: 1})
        # m7 = (A01-A11)(B10+B11) -> C00 += 1, C10 -= 1
        add_term({a01: 1, a11: -1}, {b10: 1, b11: 1}, {c00: 1, c10: -1})
    
    def block_2x2x1(i_off, j_off, k_off):
        """
        A block: rows i_off..i_off+1, cols j_off..j_off+1
        B block: rows j_off..j_off+1, col k_off
        C block: rows i_off..i_off+1, col k_off
        Schoolbook rank 4.
        """
        a00 = a_idx(i_off, j_off)
        a01 = a_idx(i_off, j_off + 1)
        a10 = a_idx(i_off + 1, j_off)
        a11 = a_idx(i_off + 1, j_off + 1)
        
        b0 = b_idx(j_off, k_off)
        b1 = b_idx(j_off + 1, k_off)
        
        c0 = c_idx(i_off, k_off)
        c1 = c_idx(i_off + 1, k_off)
        
        add_term({a00: 1}, {b0: 1}, {c0: 1})
        add_term({a01: 1}, {b1: 1}, {c0: 1})
        add_term({a10: 1}, {b0: 1}, {c1: 1})
        add_term({a11: 1}, {b1: 1}, {c1: 1})
    
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    strassen_block(0, 0, 0)
    # Block 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    strassen_block(0, 0, 2)
    # Block 3: A[0:2,0:2] * B[0:2,4:5] -> C[0:2,4:5]
    block_2x2x1(0, 0, 4)
    # Block 4: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    strassen_block(0, 2, 0)
    # Block 5: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    strassen_block(0, 2, 2)
    # Block 6: A[0:2,2:4] * B[2:4,4:5] -> C[0:2,4:5]
    block_2x2x1(0, 2, 4)
    
    rank = len(U_list)
    U = [row for row in U_list]
    V = [row for row in V_list]
    W = [row for row in W_list]
    
    return {"U": U, "V": V, "W": W}
