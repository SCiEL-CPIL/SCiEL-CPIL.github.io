import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition term by term
    # Using a list of tuples (i, j, k, coeff_u, coeff_v, coeff_w) is not enough
    # because one term (rank 1 tensor) has one u, one v, one w vector.
    # We need to accumulate into U, V, W matrices.
    
    # Initialize U, V, W
    # We don't know the rank ahead of time exactly, but we can estimate or use a list and convert.
    # Max rank estimate: 55. Let's use a dynamic list or a large enough array.
    # Better: collect terms in a list, then assemble.
    
    terms = []
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        # u_idx is (i, j), v_idx is (j, k), w_idx is (k, i)
        # This adds a contribution to a specific rank-1 component?
        # No, a rank-1 component is defined by a full vector u, v, w.
        # If we add multiple non-zero entries to u, v, w for the same 'r', they must belong to the same rank-1 tensor.
        # The block decompositions above define specific rank-1 tensors.
        # For Strassen, each of the 7 products is a rank-1 tensor.
        # For Schoolbook, each product $A_{ij} B_{jk}$ is a rank-1 tensor.
        
        # So we should add a whole rank-1 tensor at a time.
        pass

    # Let's define functions to add specific blocks.
    
    def add_strassen_222(i0, j0, k0):
        # Adds 7 rank-1 tensors for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        # Indices:
        # i in {i0, i0+1}
        # j in {j0, j0+1}
        # k in {k0, k0+1}
        
        # P1: (A00+A11)(B00+B11) -> C00, C01, C10, C11
        u = {(i0, j0): 1, (i0+1, j0+1): 1}
        v = {(j0, k0): 1, (j0+1, k0+1): 1}
        w = {(k0, i0): 1, (k0, i0+1): 1, (k0+1, i0): 1, (k0+1, i0+1): 1}
        terms.append((u, v, w))
        
        # P2: (A10+A11)B00 -> C10
        u = {(i0+1, j0): 1, (i0+1, j0+1): 1}
        v = {(j0, k0): 1}
        w = {(k0, i0+1): 1}
        terms.append((u, v, w))
        
        # P3: A00(B01-B11) -> C01
        u = {(i0, j0): 1}
        v = {(j0, k0+1): 1, (j0+1, k0+1): -1}
        w = {(k0+1, i0): 1}
        terms.append((u, v, w))
        
        # P4: A11(B10-B00) -> C00, C10
        u = {(i0+1, j0+1): 1}
        v = {(j0+1, k0): 1, (j0, k0): -1}
        w = {(k0, i0): 1, (k0, i0+1): 1}
        terms.append((u, v, w))
        
        # P5: (A00+A10)B11 -> C01, C11
        u = {(i0, j0): 1, (i0+1, j0): 1}
        v = {(j0+1, k0+1): 1}
        w = {(k0+1, i0): 1, (k0+1, i0+1): 1}
        terms.append((u, v, w))
        
        # P6: (A11-A00)(B00+B01) -> C11
        u = {(i0+1, j0+1): 1, (i0, j0): -1}
        v = {(j0, k0): 1, (j0, k0+1): 1}
        w = {(k0+1, i0+1): 1}
        terms.append((u, v, w))
        
        # P7: (A01-A11)(B10+B11) -> C00, C11
        u = {(i0, j0+1): 1, (i0+1, j0+1): -1}
        v = {(j0+1, k0): 1, (j0+1, k0+1): 1}
        w = {(k0, i0): 1, (k0+1, i0+1): 1}
        terms.append((u, v, w))

    def add_schoolbook_221(i0, j0, k0):
        # A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+1]
        # 4 terms
        for di in range(2):
            for dj in range(2):
                u = {(i0+di, j0+dj): 1}
                v = {(j0+dj, k0): 1}
                w = {(k0, i0+di): 1}
                terms.append((u, v, w))

    def add_schoolbook_213(i0, j0, k0):
        # A[i0:i0+2, j0:j0+1] * B[j0:j0+1, k0:k0+3]
        # 6 terms
        for di in range(2):
            for dk in range(3):
                u = {(i0+di, j0): 1}
                v = {(j0, k0+dk): 1}
                w = {(k0+dk, i0+di): 1}
                terms.append((u, v, w))

    # T1: j in {0,1}, k in {0..5}
    # k blocks: 0-1, 2-3, 4-5
    add_strassen_222(0, 0, 0)
    add_strassen_222(0, 0, 2)
    add_strassen_222(0, 0, 4)
    
    # T2: j in {2,3,4}, k in {0..5}
    # k blocks: 0-2, 3-5
    
    # k block 0-2
    # j in {2,3} (2x2x2 part)
    add_strassen_222(0, 2, 0)
    # j in {2,3}, k=2 (2x2x1 part)
    add_schoolbook_221(0, 2, 2)
    # j=4 (2x1x3 part)
    add_schoolbook_213(0, 4, 0)
    
    # k block 3-5
    # j in {2,3} (2x2x2 part)
    add_strassen_222(0, 2, 3)
    # j in {2,3}, k=5 (2x2x1 part)
    add_schoolbook_221(0, 2, 5)
    # j=4 (2x1x3 part)
    add_schoolbook_213(0, 4, 3)

    # Assemble U, V, W
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for (i, j), val in u.items():
            U[i * m + j][r] += val
        for (j, k), val in v.items():
            V[j * p + k][r] += val
        for (k, i), val in w.items():
            W[k * n + i][r] += val
            
    return {"U": U, "V": V, "W": W}

