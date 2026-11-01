"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56  # 8 blocks * 7 Strassen terms
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # U_vecs are length 4 (a00, a01, a10, a11)
    # V_vecs are length 4 local (b00, b01, b10, b11)
    # W_contribs are dict mapping local c_idx (0,1,2,3 for c00, c01, c10, c11) to coeff
    
    # Note: V local index 0 is b00, 1 is b01, 2 is b10, 3 is b11
    # U local index 0 is a00, 1 is a01, 2 is a10, 3 is a11
    
    strassen_terms = [
        # (U_local, V_local, W_local_contribs)
        ([1, 0, 0, 1], [1, 0, 0, 0], {0: 1, 3: 1}),       # M1: (a00+a11)(b00+b11) -> c00+=, c11+=
        ([0, 0, 1, 1], [1, 0, 0, 0], {1: 1, 3: -1}),      # M2: (a10+a11)b00 -> c10+=, c11-=
        ([1, 0, 0, 0], [0, 1, 0, -1], {2: 1, 3: 1}),      # M3: a00(b01-b11) -> c01+=, c11+=
        ([0, 0, 0, 1], [-1, 0, 1, 0], {0: 1, 1: 1}),      # M4: a11(b10-b00) -> c00+=, c10+=
        ([1, 1, 0, 0], [0, 0, 0, 1], {0: -1, 2: 1}),      # M5: (a00+a01)b11 -> c00-=, c01+=
        ([-1, 0, 1, 0], [1, 1, 0, 0], {3: 1}),            # M6: (a10-a00)(b00+b01) -> c11+=
        ([0, 1, 0, -1], [0, 0, 1, 1], {0: 1})             # M7: (a01-a11)(b10+b11) -> c00+=
    ]
    
    # Mapping of local V index to global V index for block g
    # V_global_idx = j * p + k
    # j=0, k=2g -> 2g
    # j=0, k=2g+1 -> 2g+1
    # j=1, k=2g -> 16 + 2g
    # j=1, k=2g+1 -> 16 + 2g + 1
    # Local V indices: 0->b00, 1->b01, 2->b10, 3->b11
    
    # Mapping of local W index to global W index for block g
    # W_global_idx = k * n + i
    # c00: k=2g, i=0 -> 4g
    # c01: k=2g+1, i=0 -> 4g+2
    # c10: k=2g, i=1 -> 4g+1
    # c11: k=2g+1, i=1 -> 4g+3
    # Local W indices: 0->c00, 1->c01, 2->c10, 3->c11
    # Wait, my W_contribs dict keys:
    # 0: c00
    # 1: c10
    # 2: c01
    # 3: c11
    # Let's re-verify the Strassen formulas and key mapping.
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # My dict keys in strassen_terms:
    # M1: {0: 1, 3: 1} -> C00, C11. Correct.
    # M2: {1: 1, 3: -1} -> C10, C11. Correct.
    # M3: {2: 1, 3: 1} -> C01, C11. Correct.
    # M4: {0: 1, 1: 1} -> C00, C10. Correct.
    # M5: {0: -1, 2: 1} -> C00, C01. Correct.
    # M6: {3: 1} -> C11. Correct.
    # M7: {0: 1} -> C00. Correct.
    
    # Key 0 -> C00 -> Global W idx 4g
    # Key 1 -> C10 -> Global W idx 4g+1
    # Key 2 -> C01 -> Global W idx 4g+2
    # Key 3 -> C11 -> Global W idx 4g+3
    
    col = 0
    for g in range(8):
        for u_loc, v_loc, w_contribs in strassen_terms:
            # Update U
            for i in range(4):
                if u_loc[i] != 0:
                    U[i][col] = u_loc[i]
            
            # Update V
            # Local V indices 0,1,2,3 map to global V indices
            v_global_indices = [2*g, 2*g+1, 16+2*g, 16+2*g+1]
            for l in range(4):
                if v_loc[l] != 0:
                    V[v_global_indices[l]][col] = v_loc[l]
            
            # Update W
            # Local W indices 0,1,2,3 map to global W indices
            # 0->4g, 1->4g+1, 2->4g+2, 3->4g+3
            w_global_indices = [4*g, 4*g+1, 4*g+2, 4*g+3]
            for k_loc, coeff in w_contribs.items():
                W[w_global_indices[k_loc]][col] = coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
