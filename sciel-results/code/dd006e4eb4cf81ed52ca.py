"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_decomposition(n, m, p):
    """
    Builds a rank-77 decomposition for 4x4 x 4x6 matrix multiplication.
    Uses the tensor product of a 2x2x3 (rank 11) and a 2x2x2 (rank 7) multiplication tensor.
    Rank = 11 * 7 = 77.
    """
    
    # 1. Define Strassen 2x2x2 terms as sparse dicts
    # Format: (U_dict, V_dict, W_dict)
    # Indices are (row, col) for 2x2 matrices
    strassen_terms_2x2 = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (0, 1): 1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
    ]
    
    # 2. Define 2x2x3 terms (Rank 11)
    # 7 Strassen terms for cols 0,1
    # 4 Schoolbook terms for col 2
    terms_2x2x3 = []
    
    # Add Strassen terms for k=0,1
    for u_s, v_s, w_s in strassen_terms_2x2:
        # w_s only has keys for k=0,1 which is correct for 2x2x2 part
        terms_2x2x3.append((u_s, v_s, w_s))
        
    # Add Schoolbook terms for k=2
    for i in range(2):
        for j in range(2):
            u_dict = {(i, j): 1}
            v_dict = {(j, 2): 1}
            w_dict = {(2, i): 1}
            terms_2x2x3.append((u_dict, v_dict, w_dict))
            
    # Total 11 terms for 2x2x3
    
    rank_terms = []
    
    # 3. Generate 77 terms via tensor product
    # Global indices:
    # i = 2*i1 + i2, j = 2*j1 + j2
    # k = 2*k1 + k2  (k1 in 0..2, k2 in 0..1)
    
    for u1, v1, w1 in terms_2x2x3:
        for u2, v2, w2 in strassen_terms_2x2:
            
            U_global = {}
            V_global = {}
            W_global = {}
            
            # U: (i, j) -> (i1, i2), (j1, j2)
            for (i1, j1), cu1 in u1.items():
                for (i2, j2), cu2 in u2.items():
                    gi = 2 * i1 + i2
                    gj = 2 * j1 + j2
                    coeff = cu1 * cu2
                    key = (gi, gj)
                    U_global[key] = U_global.get(key, 0) + coeff
                    
            # V: (j, k) -> (j1, j2), (k1, k2)
            # Note: v1 has keys (j1, k1) with k1 in 0..2
            # v2 has keys (j2, k2) with k2 in 0..1
            for (j1, k1), cv1 in v1.items():
                for (j2, k2), cv2 in v2.items():
                    gj = 2 * j1 + j2
                    gk = 2 * k1 + k2
                    coeff = cv1 * cv2
                    key = (gj, gk)
                    V_global[key] = V_global.get(key, 0) + coeff
                    
            # W: (k, i) -> (k1, i1), (k2, i2)
            # w1 has keys (k1, i1) with k1 in 0..2
            # w2 has keys (k2, i2) with k2 in 0..1
            for (k1, i1), cw1 in w1.items():
                for (k2, i2), cw2 in w2.items():
                    gi = 2 * i1 + i2
                    gk = 2 * k1 + k2
                    coeff = cw1 * cw2
                    key = (gk, gi) # W index is (k, i)
                    W_global[key] = W_global.get(key, 0) + coeff
                    
            rank_terms.append((U_global, V_global, W_global))
            
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
    deadline = start_time + 110.0 # Safety margin
    
    # Build decomposition
    U, V, W = _build_decomposition(n, m, p)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
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
