"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict

def _build_decomposition(n, m, p):
    """
    Builds a decomposition for 4x4 x 4x6 matrix multiplication.
    Structural approach: split into 4x4x4 + 4x4x2 blocks.
    - 4x4x4: Strassen on 2x2 blocks -> rank 49
    - 4x4x2: 4 independent 2x2x2 Strassen -> rank 28
    Then merge terms with identical (U, V) factors to reduce rank.
    """
    
    # Strassen 2x2x2 terms
    strassen_terms_2x2 = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (0, 1): 1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
    ]
    
    def _expand_2x2_to_4x4(u2, v2, w2, k_offset=0):
        """Expand a 2x2x2 term to 4x4x(k_offset..k_offset+1) global indices."""
        U_global = {}
        V_global = {}
        W_global = {}
        
        for (i1, j1), cu in u2.items():
            for (i2, j2), cu2 in u2.items():
                pass  # placeholder
        
        # Proper expansion for 4x4x4 block
        # i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
        for (i1, j1), cu1 in u2.items():
            for (i2, j2), cu2 in u2.items():
                gi = 2 * i1 + i2
                gj = 2 * j1 + j2
                coeff = cu1 * cu2
                key = (gi, gj)
                U_global[key] = U_global.get(key, 0) + coeff
        
        for (j1, k1), cv1 in v2.items():
            for (j2, k2), cv2 in v2.items():
                gj = 2 * j1 + j2
                gk = 2 * k1 + k2 + k_offset
                coeff = cv1 * cv2
                key = (gj, gk)
                V_global[key] = V_global.get(key, 0) + coeff
        
        for (k1, i1), cw1 in w2.items():
            for (k2, i2), cw2 in w2.items():
                gi = 2 * i1 + i2
                gk = 2 * k1 + k2 + k_offset
                coeff = cw1 * cw2
                key = (gk, gi)
                W_global[key] = W_global.get(key, 0) + coeff
        
        return U_global, V_global, W_global
    
    # Build 4x4x4 terms (rank 49)
    terms_4x4x4 = []
    for u2, v2, w2 in strassen_terms_2x2:
        # Tensor product of two Strassen 2x2x2 for 4x4x4
        for u2b, v2b, w2b in strassen_terms_2x2:
            U_g = {}
            V_g = {}
            W_g = {}
            
            # U: (i,j) where i=2*i1+i2, j=2*j1+j2
            for (i1, j1), c1 in u2.items():
                for (i2, j2), c2 in u2b.items():
                    gi = 2*i1 + i2
                    gj = 2*j1 + j2
                    key = (gi, gj)
                    U_g[key] = U_g.get(key, 0) + c1*c2
            
            # V: (j,k) where j=2*j1+j2, k=2*k1+k2
            for (j1, k1), c1 in v2.items():
                for (j2, k2), c2 in v2b.items():
                    gj = 2*j1 + j2
                    gk = 2*k1 + k2
                    key = (gj, gk)
                    V_g[key] = V_g.get(key, 0) + c1*c2
            
            # W: (k,i) where k=2*k1+k2, i=2*i1+i2
            for (k1, i1), c1 in w2.items():
                for (k2, i2), c2 in w2b.items():
                    gk = 2*k1 + k2
                    gi = 2*i1 + i2
                    key = (gk, gi)
                    W_g[key] = W_g.get(key, 0) + c1*c2
            
            terms_4x4x4.append((U_g, V_g, W_g))
    
    # Build 4x4x2 terms (rank 28)
    # 4 independent 2x2x2 Strassen for the last 2 columns
    terms_4x4x2 = []
    for i1 in range(2):
        for j1 in range(2):
            # Each 2x2x2 block handles rows 2*i1..2*i1+1, cols 2*j1..2*j1+1, output cols 4,5
            for u2, v2, w2 in strassen_terms_2x2:
                U_g = {}
                V_g = {}
                W_g = {}
                
                # U: (i, j) -> rows 2*i1+i2, cols 2*j1+j2
                for (i2, j2), c in u2.items():
                    gi = 2*i1 + i2
                    gj = 2*j1 + j2
                    key = (gi, gj)
                    U_g[key] = U_g.get(key, 0) + c
                
                # V: (j, k) -> cols 2*j1+j2, output cols 4+k2
                for (j2, k2), c in v2.items():
                    gj = 2*j1 + j2
                    gk = 4 + k2
                    key = (gj, gk)
                    V_g[key] = V_g.get(key, 0) + c
                
                # W: (k, i) -> output col 4+k2, row 2*i1+i2
                for (k2, i2), c in w2.items():
                    gk = 4 + k2
                    gi = 2*i1 + i2
                    key = (gk, gi)
                    W_g[key] = W_g.get(key, 0) + c
                
                terms_4x4x2.append((U_g, V_g, W_g))
    
    # Combine all terms and merge those with identical (U, V) factors
    all_terms = terms_4x4x4 + terms_4x4x2
    
    # Merge terms with identical (U, V) by summing their W contributions
    merged = {}
    for u_dict, v_dict, w_dict in all_terms:
        # Create hashable key from U and V
        u_key = tuple(sorted(u_dict.items()))
        v_key = tuple(sorted(v_dict.items()))
        key = (u_key, v_key)
        
        if key not in merged:
            merged[key] = (dict(u_dict), dict(v_dict), dict(w_dict))
        else:
            # Sum W contributions
            existing_w = merged[key][2]
            for k, w in w_dict.items():
                existing_w[k] = existing_w.get(k, 0) + w
    
    # Convert merged dict back to list
    rank_terms = list(merged.values())
    rank = len(rank_terms)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j, r] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k, r] = coeff
        for (k, i), coeff in w_dict.items():
            W[k * n + i, r] = coeff
            
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify the decomposition is correct."""
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(U.shape[1]):
        alpha_r = np.sum(A_test * U[:, r].reshape(n, m))
        beta_r = np.sum(B_test * V[:, r].reshape(m, p))
        C_computed += alpha_r * beta_r * W[:, r].reshape(p, n).T
        
    return np.allclose(C_computed, C_expected, atol=1e-10)

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 110.0
    
    U, V, W = _build_decomposition(n, m, p)
    
    if not _verify(U, V, W, n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
