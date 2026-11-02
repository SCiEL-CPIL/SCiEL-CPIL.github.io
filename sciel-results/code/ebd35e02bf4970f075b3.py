import time
import sys

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1

    def strassen_222(k_offset, k_base):
        # k_base is 0 for k in [0,1], 2 for k in [2,3] relative to k_offset
        # i in [0,1], j in [0,1]
        # U indices: i*3 + j
        # V indices: j*8 + (k_offset + k_base + k_local)
        # W indices: (k_offset + k_base + k_local)*3 + i
        
        # Strassen variables M1..M7
        # M1 = (A00+A11)(B00+B11)
        # This corresponds to multiple terms in the tensor decomposition?
        # No, Strassen is a decomposition into 7 rank-1 tensors.
        # We need to map the Strassen terms to U, V, W.
        
        # Term 1: (A00+A11) * (B00+B11) -> C00, C11
        # In tensor form:
        # U has A00=1, A11=1. V has B00=1, B11=1.
        # This is not a single rank-1 term.
        # Strassen decomposes the tensor into 7 terms.
        # Each term is of the form u * v * w.
        
        # Let's list the 7 terms explicitly.
        # T1: u=(1,0,1,0), v=(1,1,0,0), w=(1,0,0,1) ? No.
        
        # Standard Strassen:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b01-b00)
        # M5 = (a00+a10)b11
        # M6 = (a11-a00)(b00+b10)
        # M7 = (a01-a11)(b11+b01)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each Mi is a product of a linear form in A and a linear form in B.
        # So each Mi corresponds to ONE rank-1 term in the tensor decomposition?
        # Yes. The tensor is sum of Mi * (coeffs for C).
        # Wait, the tensor T = sum_{i,j,k} A_ij B_jk C_ki.
        # Strassen says T = sum_{r=1}^7 u_r * v_r * w_r.
        # Where u_r is a vector in A-space, v_r in B-space, w_r in C-space.
        
        # Let's define u_r, v_r, w_r for each r.
        
        # r=1: M1
        # u1: A00=1, A11=1. (Indices 0, 3 in 2x2 A? No, local indices 0,1)
        # Local A indices: 00, 01, 10, 11.
        # u1: [1, 0, 0, 1] (A00, A01, A10, A11) -> A00=1, A11=1.
        # v1: B00=1, B11=1. (B00, B01, B10, B11) -> [1, 0, 0, 1].
        # w1: C00=1, C11=1. (C00, C01, C10, C11) -> [1, 0, 0, 1].
        
        # r=2: M2
        # u2: A10=1, A11=1. -> [0, 0, 1, 1].
        # v2: B00=1. -> [1, 0, 0, 0].
        # w2: C10=1. -> [0, 0, 1, 0]. (C10 is index 2 in C 2x2? C00=0, C01=1, C10=2, C11=3)
        
        # r=3: M3
        # u3: A00=1. -> [1, 0, 0, 0].
        # v3: B10=1, B11=-1. -> [0, 0, 1, -1].
        # w3: C01=1, C11=1. -> [0, 1, 0, 1].
        
        # r=4: M4
        # u4: A11=1. -> [0, 0, 0, 1].
        # v4: B01=1, B00=-1. -> [-1, 1, 0, 0].
        # w4: C00=1, C10=1. -> [1, 0, 1, 0].
        
        # r=5: M5
        # u5: A00=1, A10=1. -> [1, 0, 1, 0].
        # v5: B11=1. -> [0, 0, 0, 1].
        # w5: C00=-1, C01=1. -> [-1, 1, 0, 0].
        
        # r=6: M6
        # u6: A11=1, A00=-1. -> [-1, 0, 0, 1].
        # v6: B00=1, B10=1. -> [1, 0, 1, 0].
        # w6: C11=1. -> [0, 0, 0, 1].
        
        # r=7: M7
        # u7: A01=1, A11=-1. -> [0, 1, 0, -1].
        # v7: B11=1, B01=1. -> [0, 1, 0, 1].
        # w7: C00=1, C11=1. -> [1, 0, 0, 1].

        # We need to map local indices to global indices.
        # Local A (2x2) is at global i in [0,1], j in [0,1].
        # Local B (2x2) is at global j in [0,1], k in [k_base, k_base+1].
        # Local C (2x2) is at global k in [k_base, k_base+1], i in [0,1].
        
        # We will iterate through the 7 terms and add them.
        
        terms = [
            # (u_coeffs, v_coeffs, w_coeffs)
            # u: [A00, A01, A10, A11]
            # v: [B00, B01, B10, B11]
            # w: [C00, C01, C10, C11]
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # M2
            ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]), # M3
            ([0, 0, 0, 1], [-1, 1, 0, 0], [1, 0, 1, 0]), # M4
            ([1, 0, 1, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
            ([-1, 0, 0, 1], [1, 0, 1, 0], [0, 0, 0, 1]), # M6
            ([0, 1, 0, -1], [0, 1, 0, 1], [1, 0, 0, 1])  # M7
        ]
        
        for u_c, v_c, w_c in terms:
            # Expand to add terms for each non-zero coefficient
            # u_c indices: 0->A00, 1->A01, 2->A10, 3->A11
            # Global U index for A_ij: i*3 + j
            # i_local in [0,1], j_local in [0,1]
            # A00: i=0,j=0 -> 0
            # A01: i=0,j=1 -> 1
            # A10: i=1,j=0 -> 3
            # A11: i=1,j=1 -> 4
            
            # v_c indices: 0->B00, 1->B01, 2->B10, 3->B11
            # Global V index for B_jk: j*8 + k_global
            # j_local in [0,1]
            # k_local in [0,1] -> k_global = k_offset + k_base + k_local
            # B00: j=0, k=k_offset+k_base -> 0*8 + ...
            # B01: j=0, k=k_offset+k_base+1 -> 0*8 + ...
            # B10: j=1, k=k_offset+k_base -> 1*8 + ...
            # B11: j=1, k=k_offset+k_base+1 -> 1*8 + ...
            
            # w_c indices: 0->C00, 1->C01, 2->C10, 3->C11
            # Global W index for C_ki: k_global*3 + i
            # k_local in [0,1] -> k_global = k_offset + k_base + k_local
            # i_local in [0,1]
            # C00: k=k_offset+k_base, i=0
            # C01: k=k_offset+k_base+1, i=0
            # C10: k=k_offset+k_base, i=1
            # C11: k=k_offset+k_base+1, i=1

            # We need to combine the linear forms.
            # A term is u_val * v_val * w_val for specific indices.
            # Since each u_c, v_c, w_c has multiple non-zeros, this represents a sum of rank-1 tensors?
            # No. The term is (sum u_i A_i) * (sum v_j B_j) * (sum w_k C_k).
            # This expands to sum_{i,j,k} u_i v_j w_k A_i B_j C_k.
            # This is a sum of rank-1 tensors.
            # So for each non-zero u_i, v_j, w_k, we add a term to U, V, W.
            
            # Wait, the rank is 7.
            # If I expand M1 = (A00+A11)(B00+B11)(C00+C11), it becomes 4 terms.
            # That would increase the rank.
            # Strassen's algorithm provides 7 rank-1 tensors directly.
            # The vectors u_r, v_r, w_r ARE the vectors for the rank-1 tensors.
            # So I should NOT expand them.
            # I should add ONE term for each r, with U, V, W vectors defined by u_c, v_c, w_c.
            
            # But U, V, W are matrices of shape (dim, rank).
            # Each column is a vector.
            # So for term r, I set U[:, col] = u_r (mapped to global), V[:, col] = v_r (mapped), W[:, col] = w_r (mapped).
            
            u_vec = [0]*9
            v_vec = [0]*24
            w_vec = [0]*24
            
            # Map u_c to u_vec
            # u_c[0] -> A00 -> U[0]
            # u_c[1] -> A01 -> U[1]
            # u_c[2] -> A10 -> U[3]
            # u_c[3] -> A11 -> U[4]
            u_vec[0] = u_c[0]
            u_vec[1] = u_c[1]
            u_vec[3] = u_c[2]
            u_vec[4] = u_c[3]
            
            # Map v_c to v_vec
            # v_c[0] -> B00 -> V[0*8 + k_offset + k_base]
            # v_c[1] -> B01 -> V[0*8 + k_offset + k_base + 1]
            # v_c[2] -> B10 -> V[1*8 + k_offset + k_base]
            # v_c[3] -> B11 -> V[1*8 + k_offset + k_base + 1]
            v_vec[0*8 + k_offset + k_base] = v_c[0]
            v_vec[0*8 + k_offset + k_base + 1] = v_c[1]
            v_vec[1*8 + k_offset + k_base] = v_c[2]
            v_vec[1*8 + k_offset + k_base + 1] = v_c[3]
            
            # Map w_c to w_vec
            # w_c[0] -> C00 -> W[(k_offset + k_base)*3 + 0]
            # w_c[1] -> C01 -> W[(k_offset + k_base + 1)*3 + 0]
            # w_c[2] -> C10 -> W[(k_offset + k_base)*3 + 1]
            # w_c[3] -> C11 -> W[(k_offset + k_base + 1)*3 + 1]
            w_vec[(k_offset + k_base)*3 + 0] = w_c[0]
            w_vec[(k_offset + k_base + 1)*3 + 0] = w_c[1]
            w_vec[(k_offset + k_base)*3 + 1] = w_c[2]
            w_vec[(k_offset + k_base + 1)*3 + 1] = w_c[3]
            
            for idx in range(9):
                if u_vec[idx] != 0:
                    U[idx][col] = u_vec[idx]
            for idx in range(24):
                if v_vec[idx] != 0:
                    V[idx][col] = v_vec[idx]
            for idx in range(24):
                if w_vec[idx] != 0:
                    W[idx][col] = w_vec[idx]
            col += 1

    # Loop over p_offset (0 and 4)
    for p_off in [0, 4]:
        # Part A1a: M222 for k_base 0
        strassen_222(p_off, 0)
        # Part A1b: M222 for k_base 2
        strassen_222(p_off, 2)
        
        # Part A2: M214
        # i in [0,1], j=2, k in [0,3]
        for i in range(2):
            for k_local in range(4):
                k_global = p_off + k_local
                u_idx = i * 3 + 2
                v_idx = 2 * 8 + k_global
                w_idx = k_global * 3 + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
        # Part B: M134
        # i=2, j in [0,2], k in [0,3]
        for j in range(3):
            for k_local in range(4):
                k_global = p_off + k_local
                u_idx = 2 * 3 + j
                v_idx = j * 8 + k_global
                w_idx = k_global * 3 + 2
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Simple check
    res = construct()
    print("Rank:", len(res["U"][0]))
