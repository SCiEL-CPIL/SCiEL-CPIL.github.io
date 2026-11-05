import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = 5, 5, 5
    
    # Partition 4 + 1
    block_sizes = [4, 1]
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)
    
    def get_strassen_factors_2x2():
        """Return list of 7 (U_vec, V_vec, W_vec) for Strassen 2x2x2.
        Vectors have length 4.
        """
        factors = []
        # M1 = (a11+a22)(b11+b22)
        factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        # M2 = (a21+a22)b11
        factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
        # M3 = a11(b12-b22)
        factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
        # M4 = a22(b21-b11)
        factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
        # M5 = (a11+a12)b22
        factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        # M6 = (a21-a11)(b11+b12)
        factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        # M7 = (a12-a22)(b21+b22)
        factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        
        return factors

    strassen_2x2 = get_strassen_factors_2x2()
    
    # Strassen 4x4x4 factors
    # 7 outer terms, each expanded with 7 inner terms
    # Total 49 terms
    
    # Define the outer Strassen structure for 4x4
    # Each term k defines:
    # A_comb: linear combination of A blocks (11, 12, 21, 22)
    # B_comb: linear combination of B blocks
    # C_block: which C block (11, 12, 21, 22) it contributes to, and sign
    
    # A blocks indices: 0:11, 1:12, 2:21, 3:22
    # B blocks indices: 0:11, 1:12, 2:21, 3:22
    # C blocks indices: 0:11, 1:12, 2:21, 3:22
    
    outer_terms = [
        # M1: (A11+A22)(B11+B22) -> C11 (+), C22 (+)
        # Wait, standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Term 1: M1
        # A: 11, 22 (coeff 1)
        # B: 11, 22 (coeff 1)
        # C: 11 (+1), 22 (+1)
        {'A': {0: 1, 3: 1}, 'B': {0: 1, 3: 1}, 'C': {0: 1, 3: 1}},
        # Term 2: M2
        # A: 21, 22 (coeff 1)
        # B: 11 (coeff 1)
        # C: 21 (+1), 22 (-1)
        {'A': {2: 1, 3: 1}, 'B': {0: 1}, 'C': {2: 1, 3: -1}},
        # Term 3: M3
        # A: 11 (coeff 1)
        # B: 12, 22 (coeff 1, -1)
        # C: 12 (+1), 22 (+1)
        {'A': {0: 1}, 'B': {1: 1, 3: -1}, 'C': {1: 1, 3: 1}},
        # Term 4: M4
        # A: 22 (coeff 1)
        # B: 21, 11 (coeff 1, -1)
        # C: 11 (+1), 21 (+1)
        {'A': {3: 1}, 'B': {2: 1, 0: -1}, 'C': {0: 1, 2: 1}},
        # Term 5: M5
        # A: 11, 12 (coeff 1)
        # B: 22 (coeff 1)
        # C: 11 (-1), 12 (+1)
        {'A': {0: 1, 1: 1}, 'B': {3: 1}, 'C': {0: -1, 1: 1}},
        # Term 6: M6
        # A: 21, 11 (coeff 1, -1)
        # B: 11, 12 (coeff 1)
        # C: 22 (+1)
        {'A': {2: 1, 0: -1}, 'B': {0: 1, 1: 1}, 'C': {3: 1}},
        # Term 7: M7
        # A: 12, 22 (coeff 1, -1)
        # B: 21, 22 (coeff 1)
        # C: 11 (+1)
        {'A': {1: 1, 3: -1}, 'B': {2: 1, 3: 1}, 'C': {0: 1}}
    ]
    
    # Generate 4x4x4 factors
    factors_4x4 = []
    
    for outer in outer_terms:
        # Construct A_comb and B_comb vectors for the 2x2 block multiplication
        # The inner multiplication is 2x2 x 2x2 -> 2x2
        # A_comb is a 2x2 matrix formed by linear combination of A blocks
        # B_comb is a 2x2 matrix formed by linear combination of B blocks
        
        # We need to map the inner 2x2x2 factors to the global 4x4x4 factors
        
        for u, v, w in strassen_2x2:
            # u, v, w are length 4
            # u corresponds to A_comb elements
            # v corresponds to B_comb elements
            # w corresponds to C_block elements
            
            U_vec = [0.0] * 16
            V_vec = [0.0] * 16
            W_vec = [0.0] * 16
            
            # A_comb = sum_ai * A_block_i
            # u is a vector such that A_comb_flat = u dot A_flat? No.
            # The inner Strassen factorization is:
            # Product = sum_s (u_s dot A_comb) (v_s dot B_comb) w_s
            # Here A_comb is a 2x2 matrix.
            # u_s is a vector of length 4.
            # u_s dot A_comb = sum_{r,c} u_s[r,c] * A_comb[r,c]
            
            # A_comb[r,c] = sum_{block_i} coeff_A[block_i] * A_block_i[r,c]
            
            # So u_s dot A_comb = sum_{r,c} u_s[r,c] sum_{block_i} coeff_A[block_i] A_block_i[r,c]
            # = sum_{block_i} coeff_A[block_i] sum_{r,c} u_s[r,c] A_block_i[r,c]
            
            # We want to express this as sum_{global_idx} U_vec[global_idx] A_flat[global_idx]
            
            # For each block_i in outer['A']:
            # coeff = outer['A'][block_i]
            # For each r,c in 0..1:
            #   global_idx = block_offset_i + r*2 + c
            #   U_vec[global_idx] += coeff * u_s[r*2+c]
            
            block_offsets = [0, 2, 4, 6] # 0:11, 1:12, 2:21, 3:22 for 4x4 matrix?
            # Wait, 4x4 matrix flattened:
            # 11: (0,0)-(1,1) -> 0,1,4,5
            # 12: (0,2)-(1,3) -> 2,3,6,7
            # 21: (2,0)-(3,1) -> 8,9,12,13
            # 22: (2,2)-(3,3) -> 10,11,14,15
            
            # Let's define a mapping from block_idx to local (r,c) to global index
            def get_global_A_idx(block_idx, r, c):
                # block_idx 0:11, 1:12, 2:21, 3:22
                if block_idx == 0: return r * 4 + c
                if block_idx == 1: return r * 4 + (c + 2)
                if block_idx == 2: return (r + 2) * 4 + c
                if block_idx == 3: return (r + 2) * 4 + (c + 2)
            
            def get_global_B_idx(block_idx, r, c):
                # Same structure
                if block_idx == 0: return r * 4 + c
                if block_idx == 1: return r * 4 + (c + 2)
                if block_idx == 2: return (r + 2) * 4 + c
                if block_idx == 3: return (r + 2) * 4 + (c + 2)
            
            def get_global_C_idx(block_idx, r, c):
                # Same structure
                if block_idx == 0: return r * 4 + c
                if block_idx == 1: return r * 4 + (c + 2)
                if block_idx == 2: return (r + 2) * 4 + c
                if block_idx == 3: return (r + 2) * 4 + (c + 2)

            for block_idx, coeff in outer['A'].items():
                for r in range(2):
                    for c in range(2):
                        u_val = u[r * 2 + c]
                        if u_val != 0:
                            g_idx = get_global_A_idx(block_idx, r, c)
                            U_vec[g_idx] += coeff * u_val
            
            for block_idx, coeff in outer['B'].items():
                for r in range(2):
                    for c in range(2):
                        v_val = v[r * 2 + c]
                        if v_val != 0:
                            g_idx = get_global_B_idx(block_idx, r, c)
                            V_vec[g_idx] += coeff * v_val
            
            for block_idx, coeff in outer['C'].items():
                for r in range(2):
                    for c in range(2):
                        w_val = w[r * 2 + c]
                        if w_val != 0:
                            g_idx = get_global_C_idx(block_idx, r, c)
                            W_vec[g_idx] += coeff * w_val
            
            factors_4x4.append((U_vec, V_vec, W_vec))
    
    # Total rank calculation
    # 4x4x4 block: 49
    # 4x4x1 block: 16
    # 4x1x4 block: 16
    # 1x4x4 block: 16
    # 4x1x1 block: 4
    # 1x4x1 block: 4
    # 1x1x4 block: 4
    # 1x1x1 block: 1
    total_rank = 49 + 16 + 16 + 16 + 4 + 4 + 4 + 1
    total_rank = 110
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                
                if ni == 4 and mj == 4 and pk == 4:
                    for (uf, vf, wf) in factors_4x4:
                        for idx in range(16):
                            if uf[idx] != 0:
                                U[(offsets[i] + idx // 4) * m + (offsets[j] + idx % 4)][col] += uf[idx]
                            if vf[idx] != 0:
                                V[(offsets[j] + idx // 4) * p + (offsets[k] + idx % 4)][col] += vf[idx]
                            if wf[idx] != 0:
                                W[(offsets[k] + idx // 4) * n + (offsets[i] + idx % 4)][col] += wf[idx]
                        col += 1
                else:
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                U_idx = (offsets[i] + li) * m + (offsets[j] + lj)
                                V_idx = (offsets[j] + lj) * p + (offsets[k] + lk)
                                W_idx = (offsets[k] + lk) * n + (offsets[i] + li)
                                U[U_idx][col] = 1.0
                                V[V_idx][col] = 1.0
                                W[W_idx][col] = 1.0
                                col += 1
    
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    
    return {"U": U, "V": V, "W": W}
