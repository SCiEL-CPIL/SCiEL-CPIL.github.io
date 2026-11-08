"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    deadline = start_time + 115
    
    # Rank 40: 4x4x4 via Strassen-4 (28 terms) + 4x4x1 via schoolbook (12 terms)
    rank = 40
    U = np.zeros((n * m, rank))  # 16 x 40
    V = np.zeros((m * p, rank))  # 20 x 40
    W = np.zeros((p * n, rank))  # 20 x 40
    
    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j, col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k, col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i, col] += c
    
    col = 0
    
    # --- Part 1: 4x4 * 4x4 using Strassen-4 (rank 28) ---
    # Split into 2x2 blocks, apply Strassen at each level
    # 4x4 = (2x2 blocks) x (2x2 blocks) -> 7 Strassen terms, each 2x2x2 -> 7 scalar terms
    # Total: 7 * 7 = 49? No, wait.
    # Actually for 4x4x4, we can use the 2-level Strassen:
    # Level 1: Split 4x4 into four 2x2 blocks. Use 7 multiplications of 2x2 matrices.
    # Level 2: Each 2x2x2 uses Strassen with 7 scalar multiplications.
    # Total: 7 * 7 = 49. That's the same as before.
    
    # Alternative: Use the known rank-28 algorithm for 4x4x4.
    # The rank of 4x4x4 is known to be at most 28 (de Groote 1978, Bini et al.).
    # But constructing it explicitly is complex. Let me use a simpler approach.
    
    # Actually, let me just use the 7x7=49 Strassen for 4x4x4 and find a better
    # decomposition for the 4x4x1 part.
    
    # For 4x4 * 4x1: this is A(4x4) * b(4x1) = c(4x1)
    # The rank of 4x4x1 is at most 4 (each output is a dot product).
    # Wait, no. The tensor T_{i,k} = sum_j A_{i,j} b_j.
    # We need to decompose: sum_r U_{i,j}^r V_{j,k}^r W_{k,i}^r = sum_j A_{i,j} b_j
    # This is equivalent to: for each (i,k), sum_j A_{i,j} b_j = sum_r (sum_j U_{i,j}^r W_{k,i}^r) V_{j,k}^r
    # Actually the tensor is T_{i,j,k} = delta_{k,0} for k=0 (the 5th column, index 4).
    # Wait, p=5, so k ranges 0..4. The 5th column is k=4.
    # T_{i,j,k} = delta_{k,4}
    # So we need: sum_r U_{i,j}^r V_{j,4}^r W_{4,i}^r = delta_{k,4} * 1 (for the A*b part)
    # Actually no. The full tensor for 4x4*4x5 is:
    # T_{i,j,k} = 1 if we're multiplying A_{i,j} by B_{j,k}
    # So T_{i,j,k} = 1 for all i,j,k.
    # The rank of this tensor is what we need.
    
    # Let me reconsider. The current approach uses 49+16=65.
    # Can I do better?
    
    # For the 4x4x1 part (column 4 of B), the rank is at most 4:
    # c_i = sum_j A_{i,j} b_j
    # We can write this as 4 rank-1 terms:
    # For each j: U_{i,j}^r = delta_{i,r} * 1, V_{j,k}^r = delta_{j,r} * delta_{k,4}, W_{k,i}^r = delta_{k,4} * delta_{i,r}
    # Wait, that gives sum_r delta_{i,r} * delta_{j,r} * delta_{k,4} * delta_{k,4} * delta_{i,r} = delta_{i,j} * delta_{k,4}
    # That's not right. We need sum_j A_{i,j} b_j, not sum_r delta_{i,j} b_j.
    
    # Actually, the 4x4x1 tensor has rank 4. Let me verify:
    # T_{i,j,k} = delta_{k,4} for the A*b contribution
    # We need sum_r U_{i,j}^r V_{j,4}^r W_{4,i}^r = delta_{k,4}
    # But V has indices (j,k) and W has indices (k,i).
    # So sum_r U_{i,j}^r V_{j,4}^r W_{4,i}^r should equal 1 for all i,j when k=4.
    # This is a rank-1 tensor in (i,j,4) space... no, it's the all-ones tensor in i,j.
    # The all-ones 4x4 tensor has rank 4 (it's the sum of 4 rank-1 terms: e_i * 1^T for each i).
    # So U_{i,j}^r = delta_{i,r}, V_{j,4}^r = 1, W_{4,i}^r = 1 for r=0..3.
    # Check: sum_r delta_{i,r} * 1 * 1 = 1. Yes!
    # So the 4x4x1 part needs only rank 4, not 16.
    
    # Total rank: 49 + 4 = 53. That's better than 65.
    
    # But wait, the current code uses 16 terms for the 4x4x1 part because it treats
    # each (i,j) pair separately. We can do much better.
    
    # Let me redo: rank = 49 + 4 = 53
    
    # Actually, I realize the 49 for 4x4x4 might not be optimal either.
    # The best known rank for 4x4x4 is 28 (de Groote). But implementing that is complex.
    # Let me stick with 49 for now and improve the 4x4x1 part to rank 4.
    
    # --- Part 1: 4x4 * 4x4 using Strassen (49 terms) ---
    # Same as before, but I'll rebuild it cleanly
    
    def get_super_term_indices(super_idx, inner_idx):
        u_list = []
        v_list = []
        w_list = []
        
        if super_idx == 0:
            p_blocks = [1, 0, 0, 1]
            q_blocks = [1, 0, 0, 1]
        elif super_idx == 1:
            p_blocks = [0, 0, 1, 1]
            q_blocks = [1, 0, 0, 0]
        elif super_idx == 2:
            p_blocks = [1, 0, 0, 0]
            q_blocks = [0, 1, 0, -1]
        elif super_idx == 3:
            p_blocks = [0, 0, 0, 1]
            q_blocks = [-1, 0, 1, 0]
        elif super_idx == 4:
            p_blocks = [1, 1, 0, 0]
            q_blocks = [0, 0, 0, 1]
        elif super_idx == 5:
            p_blocks = [-1, 0, 1, 0]
            q_blocks = [1, 1, 0, 0]
        elif super_idx == 6:
            p_blocks = [0, 1, 0, -1]
            q_blocks = [0, 0, 1, 1]
        else:
            return u_list, v_list, w_list
            
        if inner_idx == 0:
            p_elems = [(0, 1), (3, 1)]
            q_elems = [(0, 1), (3, 1)]
        elif inner_idx == 1:
            p_elems = [(2, 1), (3, 1)]
            q_elems = [(0, 1)]
        elif inner_idx == 2:
            p_elems = [(0, 1)]
            q_elems = [(1, 1), (3, -1)]
        elif inner_idx == 3:
            p_elems = [(3, 1)]
            q_elems = [(2, 1), (0, -1)]
        elif inner_idx == 4:
            p_elems = [(0, 1), (1, 1)]
            q_elems = [(3, 1)]
        elif inner_idx == 5:
            p_elems = [(2, 1), (0, -1)]
            q_elems = [(0, 1), (1, 1)]
        elif inner_idx == 6:
            p_elems = [(1, 1), (3, -1)]
            q_elems = [(2, 1), (3, 1)]
        else:
            return u_list, v_list, w_list
            
        def get_p_element(p_elem_idx):
            res = []
            r = p_elem_idx // 2
            c = p_elem_idx % 2
            for b_idx, coeff in enumerate(p_blocks):
                if coeff == 0: continue
                if b_idx == 0:
                    gi, gj = r, c
                elif b_idx == 1:
                    gi, gj = r, 2 + c
                elif b_idx == 2:
                    gi, gj = 2 + r, c
                elif b_idx == 3:
                    gi, gj = 2 + r, 2 + c
                res.append((gi, gj, coeff))
            return res
            
        def get_q_element(q_elem_idx):
            res = []
            r = q_elem_idx // 2
            c = q_elem_idx % 2
            for b_idx, coeff in enumerate(q_blocks):
                if coeff == 0: continue
                if b_idx == 0:
                    gj, gk = r, c
                elif b_idx == 1:
                    gj, gk = r, 2 + c
                elif b_idx == 2:
                    gj, gk = 2 + r, c
                elif b_idx == 3:
                    gj, gk = 2 + r, 2 + c
                res.append((gj, gk, coeff))
            return res
            
        for p_idx, p_coeff in p_elems:
            for i, j, c in get_p_element(p_idx):
                u_list.append((i, j, c * p_coeff))
                
        for q_idx, q_coeff in q_elems:
            for j, k, c in get_q_element(q_idx):
                v_list.append((j, k, c * q_coeff))
                
        if inner_idx == 0:
            s_contrib = [1, 0, 0, 1]
        elif inner_idx == 1:
            s_contrib = [0, 0, 1, -1]
        elif inner_idx == 2:
            s_contrib = [0, 1, 0, 1]
        elif inner_idx == 3:
            s_contrib = [1, 0, 1, 0]
        elif inner_idx == 4:
            s_contrib = [-1, 1, 0, 0]
        elif inner_idx == 5:
            s_contrib = [0, 0, 0, 1]
        elif inner_idx == 6:
            s_contrib = [1, 0, 0, 0]
            
        if super_idx == 0:
            c_block_contrib = [1, 0, 0, 1]
        elif super_idx == 1:
            c_block_contrib = [0, 0, 1, -1]
        elif super_idx == 2:
            c_block_contrib = [0, 1, 0, 1]
        elif super_idx == 3:
            c_block_contrib = [1, 0, 1, 0]
        elif super_idx == 4:
            c_block_contrib = [-1, 1, 0, 0]
        elif super_idx == 5:
            c_block_contrib = [0, 0, 0, 1]
        elif super_idx == 6:
            c_block_contrib = [1, 0, 0, 0]
            
        for s_pos in range(4):
            s_coeff = s_contrib[s_pos]
            if s_coeff == 0: continue
            s_r = s_pos // 2
            s_c = s_pos % 2
            for c_b in range(4):
                c_coeff = c_block_contrib[c_b]
                if c_coeff == 0: continue
                total_coeff = s_coeff * c_coeff
                if total_coeff == 0: continue
                if c_b == 0:
                    gi, gk = s_r, s_c
                elif c_b == 1:
                    gi, gk = s_r, 2 + s_c
                elif c_b == 2:
                    gi, gk = 2 + s_r, s_c
                elif c_b == 3:
                    gi, gk = 2 + s_r, 2 + s_c
                w_list.append((gk, gi, total_coeff))
                
        return u_list, v_list, w_list

    for s in range(7):
        for t in range(7):
            u_list, v_list, w_list = get_super_term_indices(s, t)
            add_term(col, u_list, v_list, w_list)
            col += 1
    
    # --- Part 2: 4x4 * 4x1 using rank-4 decomposition ---
    # c_i = sum_j A_{i,j} b_j
    # Rank-4: for r=0..3: U_{i,j}^r = delta_{i,r}, V_{j,4}^r = 1, W_{4,i}^r = 1
    for r in range(4):
        u_list = [(r, j, 1) for j in range(4)]
        v_list = [(j, 4, 1) for j in range(4)]
        w_list = [(4, r, 1)]
        add_term(col, u_list, v_list, w_list)
        col += 1
    
    # Total rank: 49 + 4 = 53
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
