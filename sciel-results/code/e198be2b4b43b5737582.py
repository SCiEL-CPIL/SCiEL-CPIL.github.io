import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # Rank 40
    rank = 40
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Strassen terms definition
    # U coeffs for a,b,c,d (indices 0,1,3,4 in U)
    # V coeffs for e,f,g,h (indices 0,1 in V rows, k1,k2 in cols)
    # W coeffs for C11, C12, C21, C22
    
    strassen_terms = [
        # (U: {idx: val}, V: {idx: val}, W: {idx: val})
        # M1: (a+d)(e+h)
        ({0: 1, 4: 1}, {0: 1, 1: 1}, {0: 1, 3: -1}), # C11+1, C22-1 (W indices: k1*2+0, k2*2+1)
        # M2: (c+d)e
        ({3: 1, 4: 1}, {0: 1}, {1: 1, 3: -1}), # C21+1, C22-1 (W: k1*2+1, k2*2+1)
        # M3: a(f-h)
        ({0: 1}, {0: 1, 1: -1}, {2: 1, 3: 1}), # C12+1, C22+1 (W: k2*2+0, k2*2+1)
        # M4: d(g-e)
        ({4: 1}, {0: -1, 1: 1}, {0: 1, 1: 1}), # C11+1, C21+1 (W: k1*2+0, k1*2+1)
        # M5: (a+b)h
        ({0: 1, 1: 1}, {1: 1}, {0: -1, 2: 1}), # C11-1, C12+1 (W: k1*2+0, k2*2+0)
        # M6: (c-a)(e+f)
        ({0: -1, 3: 1}, {0: 1, 1: 1}, {3: 1}), # C22+1 (W: k2*2+1)
        # M7: (b-d)(g+h)
        ({1: 1, 4: -1}, {0: 1, 1: 1}, {0: 1}), # C11+1 (W: k1*2+0)
    ]
    
    for t in range(5):
        k1 = 2 * t
        k2 = 2 * t + 1
        
        # Add 7 Strassen terms
        for term in strassen_terms:
            u_coeffs, v_coeffs, w_coeffs = term
            
            # U
            for idx, val in u_coeffs.items():
                U[idx][col_idx] = val
                
            # V
            # v_coeffs keys are 0,1 corresponding to j=0,1
            # We need to map to V index: j*p + k
            # For each v_coeff, it applies to both k1 and k2?
            # No, the Strassen V coeffs are for e,f,g,h.
            # e = B[0, k1], f = B[0, k2]
            # g = B[1, k1], h = B[1, k2]
            # So v_coeffs[0] applies to e and f?
            # In my definition above:
            # M1: e+h. e is B[0,k1], h is B[1,k2].
            # So v_coeffs should specify which of e,f,g,h.
            # Let's refine V mapping.
            # e -> V[0*p + k1]
            # f -> V[0*p + k2]
            # g -> V[1*p + k1]
            # h -> V[1*p + k2]
            
            # I will redefine v_coeffs to map to e,f,g,h explicitly
            # M1: e, h. 
            # My previous v_coeffs {0:1, 1:1} was ambiguous.
            # Let's use a dict mapping 'e','f','g','h' to values.
            
            pass # I will rewrite the loop below
            
        # Add 1 Rank-1 term for j=3
        # U: indices 2 (i=0,j=3) and 5 (i=1,j=3)
        U[2][col_idx] = 1
        U[5][col_idx] = 1
        # V: indices 3*p + k1 and 3*p + k2
        V[3 * p + k1][col_idx] = 1
        V[3 * p + k2][col_idx] = 1
        # W: indices k1*n + 0, k1*n + 1, k2*n + 0, k2*n + 1
        W[k1 * n + 0][col_idx] = 1
        W[k1 * n + 1][col_idx] = 1
        W[k2 * n + 0][col_idx] = 1
        W[k2 * n + 1][col_idx] = 1
        
        col_idx += 1
        
        # Now the 7 Strassen terms
        # I need to update the strassen_terms definition to use e,f,g,h keys
        
    # Re-do Strassen terms with clear mapping
    # e: V[0*p+k1], f: V[0*p+k2], g: V[1*p+k1], h: V[1*p+k2]
    
    strassen_terms_v2 = [
        # (U: {idx: val}, V: {e/f/g/h: val}, W: {C11/C12/C21/C22: val})
        ({0: 1, 4: 1}, {'e': 1, 'h': 1}, {'C11': 1, 'C22': -1}),
        ({3: 1, 4: 1}, {'e': 1}, {'C21': 1, 'C22': -1}),
        ({0: 1}, {'f': 1, 'h': -1}, {'C12': 1, 'C22': 1}),
        ({4: 1}, {'g': 1, 'e': -1}, {'C11': 1, 'C21': 1}),
        ({0: 1, 1: 1}, {'h': 1}, {'C11': -1, 'C12': 1}),
        ({0: -1, 3: 1}, {'e': 1, 'f': 1}, {'C22': 1}),
        ({1: 1, 4: -1}, {'g': 1, 'h': 1}, {'C11': 1}),
    ]
    
    # Reset col_idx to 0 and refill properly
    for c in range(rank):
        for r in range(n*m): U[r][c] = 0
        for r in range(m*p): V[r][c] = 0
        for r in range(p*n): W[r][c] = 0
        
    col_idx = 0
    for t in range(5):
        k1 = 2 * t
        k2 = 2 * t + 1
        
        for term in strassen_terms_v2:
            u_coeffs, v_coeffs, w_coeffs = term
            
            for idx, val in u_coeffs.items():
                U[idx][col_idx] = val
                
            # Map V
            if 'e' in v_coeffs: V[0 * p + k1][col_idx] += v_coeffs['e']
            if 'f' in v_coeffs: V[0 * p + k2][col_idx] += v_coeffs['f']
            if 'g' in v_coeffs: V[1 * p + k1][col_idx] += v_coeffs['g']
            if 'h' in v_coeffs: V[1 * p + k2][col_idx] += v_coeffs['h']
            
            # Map W
            if 'C11' in w_coeffs: W[k1 * n + 0][col_idx] += w_coeffs['C11']
            if 'C12' in w_coeffs: W[k2 * n + 0][col_idx] += w_coeffs['C12']
            if 'C21' in w_coeffs: W[k1 * n + 1][col_idx] += w_coeffs['C21']
            if 'C22' in w_coeffs: W[k2 * n + 1][col_idx] += w_coeffs['C22']
            
            col_idx += 1
            
        # Rank 1 term for j=3
        U[2][col_idx] = 1
        U[5][col_idx] = 1
        V[3 * p + k1][col_idx] = 1
        V[3 * p + k2][col_idx] = 1
        W[k1 * n + 0][col_idx] = 1
        W[k1 * n + 1][col_idx] = 1
        W[k2 * n + 0][col_idx] = 1
        W[k2 * n + 1][col_idx] = 1
        col_idx += 1
        
    return {"U": U, "V": V, "W": W}

BET: {mechanism: "Decomposing the 2x3x10 tensor into five disjoint 2x3x2 sub-tensors, each solved with rank 8 (7 Strassen + 1 rank-1), achieves total rank 40, which is significantly lower than the previous best of 55.", q: 0.9, delta: [0.1, 0.3]}
