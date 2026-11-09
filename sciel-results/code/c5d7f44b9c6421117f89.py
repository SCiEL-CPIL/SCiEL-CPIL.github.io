import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # We use a rank 98 decomposition based on Strassen's algorithm applied to 4x4x4 blocks.
    # B is split into B_left (4x4) and B_right (4x4).
    
    # Strassen rank-7 terms for 2x2x2 multiplication
    # Each term is (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict { (i, j): coeff } for A
    # V_coeffs: dict { (j, k): coeff } for B
    # W_coeffs: dict { (i, k): coeff } for C
    strassen_terms_2x2 = [
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]

    # Generate rank-49 terms for 4x4x4 multiplication
    # Top-level Strassen terms combine 2x2 blocks
    # A blocks: A[u,v] where u,v in 0,1. Each is 2x2.
    # B blocks: B[v,w] where v,w in 0,1. Each is 2x2.
    # C blocks: C[u,w] where u,w in 0,1. Each is 2x2.
    
    # Top-level Strassen definitions for 4x4x4
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We represent each top-level term as a list of (block_u, block_v, sign_A, block_u2, block_v2, sign_B)
    # Actually, it's easier to just define the linear combinations of the 2x2 blocks.
    # Let's create a list of 7 "virtual" multiplications of 2x2 blocks.
    # Each virtual multiplication is defined by:
    #   A_part: list of (block_u, block_v, coeff)
    #   B_part: list of (block_v, block_w, coeff)
    #   C_part: list of (block_u, block_w, coeff)
    
    top_level_terms = [
        # M1
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        # M2
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)], [((1,0), 1), ((1,1), -1)]),
        # M3
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)], [((0,1), 1), ((1,1), 1)]),
        # M4
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)], [((0,0), 1), ((1,0), 1)]),
        # M5
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)], [((0,0), -1), ((0,1), 1)]),
        # M6
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        # M7
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)], [((0,0), 1)])
    ]

    # Now we expand each top-level term into 7 bottom-level terms.
    # Total 49 terms for 4x4x4.
    # Each bottom-level term is a rank-1 tensor for the 2x2x2 multiplication of the virtual blocks.
    
    # We will store the final terms as:
    # U_coeffs: dict { global_i, global_j: coeff }
    # V_coeffs: dict { global_j, global_k: coeff }
    # W_coeffs: dict { global_i, global_k: coeff }
    
    all_terms = []
    
    # Helper to add a term
    def add_term(u_dict, v_dict, w_dict):
        all_terms.append((u_dict, v_dict, w_dict))

    for t_idx in range(7):
        a_blocks, b_blocks, c_blocks = top_level_terms[t_idx]
        
        # a_blocks is list of ((u, v), coeff)
        # b_blocks is list of ((v, w), coeff)
        # c_blocks is list of ((u, w), coeff)
        
        # For each combination of a_block and b_block? No.
        # The virtual multiplication is:
        # A_virtual = sum(a_coeff * A_block)
        # B_virtual = sum(b_coeff * B_block)
        # C_virtual = A_virtual * B_virtual
        # C_virtual is then distributed to C blocks via c_blocks.
        
        # A_virtual is a 2x2 matrix (sum of 2x2 blocks).
        # B_virtual is a 2x2 matrix.
        # Their product is a 2x2 matrix.
        # We use Strassen rank-7 to compute this product.
        
        # For each of the 7 Strassen terms of the 2x2x2 multiplication:
        for s_idx in range(7):
            s_u, s_v, s_w = strassen_terms_2x2[s_idx]
            
            # s_u: dict { (r, c): coeff } for A_virtual
            # s_v: dict { (r, c): coeff } for B_virtual
            # s_w: dict { (r, c): coeff } for C_virtual
            
            # We need to map A_virtual entries to global A entries.
            # A_virtual[r, c] = sum_{(u,v), coeff} coeff * A_block[u, v][r, c]
            # A_block[u, v][r, c] corresponds to global A[i, j] where:
            # i = 2*u + r, j = 2*v + c
            
            u_global = {}
            for (r, c), coeff_s in s_u.items():
                for (u, v), coeff_a in a_blocks:
                    # A_virtual[r, c] gets coeff_s * coeff_a * A_block[u, v][r, c]
                    # So global A[2*u+r, 2*v+c] gets coeff_s * coeff_a
                    i = 2*u + r
                    j = 2*v + c
                    u_global[(i, j)] = u_global.get((i, j), 0) + coeff_s * coeff_a
            
            # Similarly for B_virtual
            # B_virtual[r, c] = sum_{(v,w), coeff} coeff * B_block[v, w][r, c]
            # B_block[v, w][r, c] corresponds to global B[j, k] where:
            # j = 2*v + r, k = 2*w + c
            # Note: In 4x4x4, k is 0..3.
            
            v_global = {}
            for (r, c), coeff_s in s_v.items():
                for (v, w), coeff_b in b_blocks:
                    j = 2*v + r
                    k = 2*w + c
                    v_global[(j, k)] = v_global.get((j, k), 0) + coeff_s * coeff_b
            
            # Similarly for C_virtual
            # C_virtual[r, c] is the result of the multiplication.
            # It is distributed to C blocks via c_blocks.
            # C_virtual[r, c] = sum_{(u,w), coeff} coeff * C_block[u, w][r, c]
            # Wait, C_virtual is the matrix product A_virtual * B_virtual.
            # The Strassen term s_w gives the contribution of this specific rank-1 term to C_virtual.
            # So C_virtual[r, c] gets coeff_s_w from this term.
            # And C_block[u, w][r, c] gets coeff_c * C_virtual[r, c].
            
            w_global = {}
            for (r, c), coeff_s in s_w.items():
                for (u, w), coeff_c in c_blocks:
                    i = 2*u + r
                    k = 2*w + c
                    w_global[(i, k)] = w_global.get((i, k), 0) + coeff_s * coeff_c
            
            # Clean up zeros
            u_global = {k: v for k, v in u_global.items() if v != 0}
            v_global = {k: v for k, v in v_global.items() if v != 0}
            w_global = {k: v for k, v in w_global.items() if v != 0}
            
            if u_global and v_global and w_global:
                add_term(u_global, v_global, w_global)

    # We have 49 terms for 4x4x4 (left part).
    # Now we need to handle the right part (4x4x4 for columns 4-7).
    # The structure is the same, but the global k indices are shifted by 4.
    
    terms_right = []
    for u_dict, v_dict, w_dict in all_terms:
        u_new = u_dict # A is the same
        v_new = {}
        for (j, k), coeff in v_dict.items():
            v_new[(j, k + 4)] = coeff
        w_new = {}
        for (i, k), coeff in w_dict.items():
            w_new[(i, k + 4)] = coeff
        terms_right.append((u_new, v_new, w_new))

    # Total terms: 49 + 49 = 98
    total_terms = all_terms + terms_right
    rank = len(total_terms)

    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col_idx, (u_dict, v_dict, w_dict) in enumerate(total_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][col_idx] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][col_idx] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][col_idx] = coeff

    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    # This is the all-ones tensor of size 4x4x8
    
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U_arr[i*m + j, r] * V_arr[j*p + k, r] * W_arr[k*n + i, r]
                if abs(val - 1) > 1e-10:
                    errors += 1
    
    if errors > 0:
        # If there are errors, fall back to schoolbook (rank 128)
        U = [[0] * (n*m*p) for _ in range(n * m)]
        V = [[0] * (n*m*p) for _ in range(m * p)]
        W = [[0] * (n*m*p) for _ in range(p * n)]
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][idx] = 1
                    V[j*p + k][idx] = 1
                    W[k*n + i][idx] = 1
                    idx += 1
        
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    print(f"Rank: {U.shape[1]}")
