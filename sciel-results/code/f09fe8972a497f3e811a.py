import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a tensor decomposition for 3x4x4 matrix multiplication.
    Uses a block decomposition with Strassen's algorithm for 2x2x2 blocks.
    Target rank: 44 (28 from Strassen blocks + 16 from schoolbook row).
    Falls back to schoolbook rank 48 if verification fails.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    # n=3, m=4, p=4
    
    # U shape: (12, rank)
    # V shape: (16, rank)
    # W shape: (12, rank)
    
    # We will build the decomposition term by term.
    # To allow for rank reduction, we keep a list of terms.
    # Each term is a tuple (u_indices, u_values, v_indices, v_values, w_indices, w_values)
    # But for efficiency, we can just build the matrices and then check independence.
    
    # Let's try to construct the rank 44 decomposition.
    terms = []
    
    # 1. Handle row i=2 (1x4x4 schoolbook)
    # i=2
    for j in range(4):
        for k in range(4):
            u_idx = 2 * 4 + j
            v_idx = j * 4 + k
            w_idx = k * 3 + 2
            terms.append(([u_idx], [1.0], [v_idx], [1.0], [w_idx], [1.0]))
            
    # 2. Handle rows i=0,1 (2x4x4 using Strassen)
    # A_1 is 2x4. Split into A_10 (2x2, cols 0,1) and A_11 (2x2, cols 2,3).
    # B is 4x4. Split into B_00, B_01, B_10, B_11 (2x2 blocks).
    # C_1 is 2x4. Split into C_10 (2x2, cols 0,1) and C_11 (2x2, cols 2,3).
    
    # C_10 = A_10 B_00 + A_11 B_10
    # C_11 = A_10 B_01 + A_11 B_11
    
    # We need to add Strassen terms for each of the 4 products.
    
    def add_strassen(u_base, v_base, w_base):
        """
        Adds 7 Strassen terms for a 2x2x2 multiplication.
        u_base: index of u_00 in U
        v_base: index of v_00 in V
        w_base: index of w_00 in W
        
        U indices for 2x2 block:
        u_00 = u_base
        u_01 = u_base + 1
        u_10 = u_base + 4
        u_11 = u_base + 5
        
        V indices for 2x2 block:
        v_00 = v_base
        v_01 = v_base + 1
        v_10 = v_base + 4
        v_11 = v_base + 5
        
        W indices for 2x2 block:
        w_00 = w_base
        w_01 = w_base + 3
        w_10 = w_base + 1
        w_11 = w_base + 4
        """
        
        u_00, u_01, u_10, u_11 = u_base, u_base + 1, u_base + 4, u_base + 5
        v_00, v_01, v_10, v_11 = v_base, v_base + 1, v_base + 4, v_base + 5
        w_00, w_01, w_10, w_11 = w_base, w_base + 3, w_base + 1, w_base + 4
        
        # M1 = (X00 + X11)(Y00 + Y11)
        terms.append(([u_00, u_11], [1.0, 1.0], [v_00, v_11], [1.0, 1.0], [w_00, w_11], [1.0, 1.0]))
        
        # M2 = (X10 + X11)Y00
        terms.append(([u_10, u_11], [1.0, 1.0], [v_00], [1.0], [w_10], [1.0]))
        
        # M3 = X00(Y10 - Y11)
        terms.append(([u_00], [1.0], [v_10, v_11], [1.0, -1.0], [w_01], [1.0]))
        
        # M4 = X11(Y01 - Y10)
        terms.append(([u_11], [1.0], [v_01, v_10], [1.0, -1.0], [w_00, w_10], [-1.0, 1.0]))
        
        # M5 = (X00 + X10)Y11
        terms.append(([u_00, u_10], [1.0, 1.0], [v_11], [1.0], [w_01, w_11], [1.0, 1.0]))
        
        # M6 = (X11 - X00)(Y00 + Y10)
        terms.append(([u_11, u_00], [1.0, -1.0], [v_00, v_10], [1.0, 1.0], [w_00, w_11], [1.0, -1.0]))
        
        # M7 = (X01 - X11)(Y11 + Y10)
        terms.append(([u_01, u_11], [1.0, -1.0], [v_11, v_10], [1.0, 1.0], [w_00], [-1.0]))
        
    # Product 1: A_10 (rows 0,1; cols 0,1) * B_00 (rows 0,1; cols 0,1) -> C_10 (rows 0,1; cols 0,1)
    # u_base: i=0, j=0 -> 0*4 + 0 = 0
    # v_base: j=0, k=0 -> 0*4 + 0 = 0
    # w_base: k=0, i=0 -> 0*3 + 0 = 0
    add_strassen(0, 0, 0)
    
    # Product 2: A_11 (rows 0,1; cols 2,3) * B_10 (rows 2,3; cols 0,1) -> C_10 (rows 0,1; cols 0,1)
    # u_base: i=0, j=2 -> 0*4 + 2 = 2
    # v_base: j=2, k=0 -> 2*4 + 0 = 8
    # w_base: k=0, i=0 -> 0
    add_strassen(2, 8, 0)
    
    # Product 3: A_10 (rows 0,1; cols 0,1) * B_01 (rows 0,1; cols 2,3) -> C_11 (rows 0,1; cols 2,3)
    # u_base: i=0, j=0 -> 0
    # v_base: j=0, k=2 -> 0*4 + 2 = 2
    # w_base: k=2, i=0 -> 2*3 + 0 = 6
    add_strassen(0, 2, 6)
    
    # Product 4: A_11 (rows 0,1; cols 2,3) * B_11 (rows 2,3; cols 2,3) -> C_11 (rows 0,1; cols 2,3)
    # u_base: i=0, j=2 -> 2
    # v_base: j=2, k=2 -> 2*4 + 2 = 10
    # w_base: k=2, i=0 -> 6
    add_strassen(2, 10, 6)
    
    # Total terms: 16 + 4*7 = 44.
    
    # Build U, V, W matrices
    rank = len(terms)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r, term in enumerate(terms):
        u_idxs, u_vals, v_idxs, v_vals, w_idxs, w_vals = term
        for idx, val in zip(u_idxs, u_vals):
            U[idx][r] += val
        for idx, val in zip(v_idxs, v_vals):
            V[idx][r] += val
        for idx, val in zip(w_idxs, w_vals):
            W[idx][r] += val
            
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        # Try to reduce rank?
        # For now, return rank 44.
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        U_fb, V_fb, W_fb = _schoolbook_direct(n, m, p)
        return {"U": U_fb, "V": V_fb, "W": W_fb}


def _schoolbook_direct(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1.0
                V[v_idx][col] = 1.0
                W[w_idx][col] = 1.0
                col += 1
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    
    # Check main terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                total = 0.0
                for r in range(rank):
                    total += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                
                if abs(total - 1.0) > 1e-9:
                    return False
                    
    # Check orthogonality
    for u_idx in range(len(U)):
        i = u_idx // m
        j = u_idx % m
        
        for v_idx in range(len(V)):
            j_prime = v_idx // p
            k = v_idx % p
            
            for w_idx in range(len(W)):
                k_prime = w_idx // n
                i_prime = w_idx % n
                
                if (j == j_prime) and (k == k_prime) and (i == i_prime):
                    expected = 1.0
                else:
                    expected = 0.0
                
                total = 0.0
                for r in range(rank):
                    total += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                
                if abs(total - expected) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
