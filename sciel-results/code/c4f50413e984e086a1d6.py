import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 15 decomposition
    rank = 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2
    # Indices for 2x2 block: i in {0,1}, j in {0,1}, k in {0,1}
    # We need to map these to the global indices.
    # Global j is 0,1 for the 2x2 block.
    # Global k is 0,1 for B1, 2,3 for B2.
    
    # Define Strassen terms for 2x2x2
    # Each term is (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict (i,j) -> val
    # v_coeffs: dict (j,k) -> val
    # w_coeffs: dict (k,i) -> val
    
    strassen_terms = []
    
    # M1 = (a+d)(e+h)
    # u: (0,0)=1, (1,1)=1
    # v: (0,0)=1, (0,1)=1, (1,0)=1, (1,1)=1
    # w: (0,0)=1, (1,1)=1 (contributes to C00 and C11)
    strassen_terms.append({
        'u': {(0,0): 1, (1,1): 1},
        'v': {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1},
        'w': {(0,0): 1, (1,1): 1}
    })
    
    # M2 = (c+d)e
    # u: (1,0)=1, (1,1)=1
    # v: (0,0)=1
    # w: (0,1)=1 (contributes to C10)
    strassen_terms.append({
        'u': {(1,0): 1, (1,1): 1},
        'v': {(0,0): 1},
        'w': {(0,1): 1}
    })
    
    # M3 = a(f-h)
    # u: (0,0)=1
    # v: (0,1)=1, (1,1)=-1
    # w: (1,0)=1, (1,1)=1 (contributes to C01 and C11)
    strassen_terms.append({
        'u': {(0,0): 1},
        'v': {(0,1): 1, (1,1): -1},
        'w': {(1,0): 1, (1,1): 1}
    })
    
    # M4 = d(g-e)
    # u: (1,1)=1
    # v: (1,0)=1, (0,0)=-1
    # w: (0,0)=1, (0,1)=1 (contributes to C00 and C10)
    strassen_terms.append({
        'u': {(1,1): 1},
        'v': {(1,0): 1, (0,0): -1},
        'w': {(0,0): 1, (0,1): 1}
    })
    
    # M5 = (a+b)h
    # u: (0,0)=1, (0,1)=1
    # v: (1,1)=1
    # w: (1,0)=-1, (1,1)=1 (contributes to C01 and C11)
    # Check C01: M3 + M5. w(1,0) for M5 should be 1?
    # C01 = M3 + M5. So w(1,0)=1.
    # C11 = M1 - M2 + M3 + M6. M5 is not in C11.
    # Wait, standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So M5 contributes to C01 with +1 and C11 with 0?
    # My previous w for M5 was {(1,0): -1, (1,1): 1}. That was wrong.
    # Let's re-verify M5.
    # M5 = (a+b)h.
    # u: (0,0)=1, (0,1)=1.
    # v: (1,1)=1.
    # w: (1,0)=1 (for C01). (1,1)=0 (for C11).
    strassen_terms.append({
        'u': {(0,0): 1, (0,1): 1},
        'v': {(1,1): 1},
        'w': {(1,0): 1}
    })
    
    # M6 = (c-a)(e+f)
    # u: (1,0)=1, (0,0)=-1
    # v: (0,0)=1, (0,1)=1
    # w: (1,1)=1 (for C11).
    strassen_terms.append({
        'u': {(1,0): 1, (0,0): -1},
        'v': {(0,0): 1, (0,1): 1},
        'w': {(1,1): 1}
    })
    
    # M7 = (b-d)(g+h)
    # u: (0,1)=1, (1,1)=-1
    # v: (1,0)=1, (1,1)=1
    # w: (0,0)=1 (for C00).
    strassen_terms.append({
        'u': {(0,1): 1, (1,1): -1},
        'v': {(1,0): 1, (1,1): 1},
        'w': {(0,0): 1}
    })
    
    # Add terms for B1 (k in {0,1})
    col_offset = 0
    for term in strassen_terms:
        for (i, j), val in term['u'].items():
            U[i * m + j][col_offset] = val
        for (j, k), val in term['v'].items():
            V[j * p + k][col_offset] = val
        for (k, i), val in term['w'].items():
            W[k * n + i][col_offset] = val
        col_offset += 1
        
    # Add terms for B2 (k in {2,3})
    # Map local k {0,1} to global k {2,3}
    for term in strassen_terms:
        for (i, j), val in term['u'].items():
            U[i * m + j][col_offset] = val
        for (j, k), val in term['v'].items():
            global_k = k + 2
            V[j * p + global_k][col_offset] = val
        for (k, i), val in term['w'].items():
            global_k = k + 2
            W[global_k * n + i][col_offset] = val
        col_offset += 1
            
    # Add term for j=2 (outer product)
    # A[:, 2] * B[2, :]
    # u: (0,2)=1, (1,2)=1
    # v: (2,0)=1, (2,1)=1, (2,2)=1, (2,3)=1
    # w: (0,0)=1, (0,1)=1, (1,0)=1, (1,1)=1
    # Check: C_i,k = A_i,2 B_2,k.
    # Term: u(0,2)=1, u(1,2)=1.
    # v(2,0)=1, v(2,1)=1, v(2,2)=1, v(2,3)=1.
    # w(k,i)=1 for all k,i.
    # Product: sum_j u(i,j) v(j,k) w(k,i) A_ij B_jk
    # j=2: u(i,2) v(2,k) w(k,i) A_i,2 B_2,k = 1*1*1*A_i,2 B_2,k.
    # Correct.
    
    for i in range(n):
        U[i * m + 2][col_offset] = 1
    for k in range(p):
        V[2 * p + k][col_offset] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][col_offset] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
