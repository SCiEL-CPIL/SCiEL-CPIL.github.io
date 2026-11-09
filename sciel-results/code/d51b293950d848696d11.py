import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # We'll build rank-1 terms using a list of (u_vec, v_vec, w_vec) tuples
    terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        """
        u_dict: {(i,j): coeff}
        v_dict: {(j,k): coeff}
        w_dict: {(k,i): coeff}
        """
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        
        for (i, j), val in u_dict.items():
            u_vec[i * m + j] = val
        for (j, k), val in v_dict.items():
            v_vec[j * p + k] = val
        for (k, i), val in w_dict.items():
            w_vec[k * n + i] = val
            
        terms.append((u_vec, v_vec, w_vec))
    
    # Standard Strassen for a 2x2x2 block
    # A is 2x2 (rows i1,i2; cols j1,j2)
    # B is 2x2 (rows j1,j2; cols k1,k2)
    # C is 2x2 (rows i1,i2; cols k1,k2)
    # C = A * B
    #
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        # M1
        u = {(i1, j1): 1, (i2, j2): 1}
        v = {(j1, k1): 1, (j2, k2): 1}
        w = {(k1, i1): 1, (k2, i2): 1}
        add_term(u, v, w)
        
        # M2
        u = {(i2, j1): 1, (i2, j2): 1}
        v = {(j1, k1): 1}
        w = {(k1, i2): 1, (k2, i2): -1}
        add_term(u, v, w)
        
        # M3
        u = {(i1, j1): 1}
        v = {(j1, k2): 1, (j2, k2): -1}
        w = {(k2, i1): 1, (k2, i2): 1}
        add_term(u, v, w)
        
        # M4
        u = {(i2, j2): 1}
        v = {(j2, k1): 1, (j1, k1): -1}
        w = {(k1, i1): 1, (k1, i2): 1}
        add_term(u, v, w)
        
        # M5
        u = {(i1, j1): 1, (i1, j2): 1}
        v = {(j2, k2): 1}
        w = {(k1, i1): -1, (k2, i1): 1}
        add_term(u, v, w)
        
        # M6
        u = {(i2, j1): 1, (i1, j1): -1}
        v = {(j1, k1): 1, (j1, k2): 1}
        w = {(k2, i2): 1}
        add_term(u, v, w)
        
        # M7
        u = {(i1, j2): 1, (i2, j2): -1}
        v = {(j2, k1): 1, (j2, k2): 1}
        w = {(k1, i1): 1}
        add_term(u, v, w)
        
    # Schoolbook term for a single (i, j, k)
    def add_schoolbook_term(i, j, k):
        u = {(i, j): 1}
        v = {(j, k): 1}
        w = {(k, i): 1}
        add_term(u, v, w)

    # Partition the middle dimension (m=5) and output dimension (p=5)
    # We partition m into sets of size 2, 2, 1
    # We partition p into sets of size 2, 2, 1
    # For each pair of sets (J, K), if both have size 2, use Strassen.
    # Otherwise, use schoolbook terms for each (i, j, k) in the cross product.
    
    J_sets = [(0, 1), (2, 3), (4,)]
    K_sets = [(0, 1), (2, 3), (4,)]
    
    for J in J_sets:
        for K in K_sets:
            if len(J) == 2 and len(K) == 2:
                j1, j2 = J
                k1, k2 = K
                # n=2, so rows are 0 and 1
                add_strassen_block(0, 1, j1, j2, k1, k2)
            else:
                for j in J:
                    for k in K:
                        for i in range(n):
                            add_schoolbook_term(i, j, k)
                            
    rank = len(terms)
    
    # Build U, V, W matrices
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        u_vec, v_vec, w_vec = terms[r]
        for i in range(n * m):
            U[i][r] = u_vec[i]
        for i in range(m * p):
            V[i][r] = v_vec[i]
        for i in range(p * n):
            W[i][r] = w_vec[i]
            
    return {"U": U, "V": V, "W": W}

BET: {mechanism: "Reconstructed the exact same Strassen-partitioning logic (rank 46) with cleaner code structure to ensure correctness and stability, confirming the baseline performance without attempting unverified rank reductions.", q: 0.5, delta: [0, 0]}
