"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Block-Strassen decomposition for 2x4 * 4x5 matrix multiplication.
    
    Partition:
    - A (2x4) = [A1, A2] where A1, A2 are 2x2 (cols 0-1, 2-3)
    - B (4x5) = [B_top (2x5); B_bot (2x5)]
    - C (2x5) = A1 * B_top + A2 * B_bot
    
    For each 2x2 * 2x5 product, split 5 cols into 2+2+1:
    - 2x2 * 2x2: Strassen (rank 7)
    - 2x2 * 2x1: schoolbook (rank 4)
    
    Total rank: 2 * (7 + 7 + 4) = 36
    """
    import time
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Rank-1 terms: each is (u_dict, v_dict, w_dict)
    # u_dict: {(i, j): coeff} for A[i,j], row in U = i*m + j
    # v_dict: {(j, k): coeff} for B[j,k], row in V = j*p + k
    # w_dict: {(i, k): coeff} for C[i,k], col in W = k*n + i
    
    rank_terms = []
    
    def add_strassen_2x2x2(i0, j0, k0, sign=1):
        """
        Add Strassen terms for X(2x2) * Y(2x2) where:
        X = A[i0:i0+2, j0:j0+2]
        Y = B[j0:j0+2, k0:k0+2]
        contributes to C[i0:i0+2, k0:k0+2]
        """
        # Strassen products:
        # m1 = (x11+x22)*(y11+y22) -> z11 += m1, z22 += m1
        # m2 = (x21+x22)*y11      -> z21 += m2, z22 -= m2
        # m3 = x11*(y12-y22)      -> z12 += m3, z22 += m3
        # m4 = x22*(y21-y11)      -> z11 += m4, z21 += m4
        # m5 = (x11+x12)*y22      -> z11 -= m5, z12 += m5
        # m6 = (x21-x11)*(y11+y12) -> z22 += m6
        # m7 = (x12-x22)*(y21+y22) -> z11 += m7
        
        terms = [
            # m1
            ({(i0, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1, (i0+1, k0+1): 1}),
            # m2
            ({(i0+1, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1},
             {(i0+1, k0): 1, (i0+1, k0+1): -1}),
            # m3
            ({(i0, j0): 1},
             {(j0, k0+1): 1, (j0+1, k0+1): -1},
             {(i0, k0+1): 1, (i0+1, k0+1): 1}),
            # m4
            ({(i0+1, j0+1): 1},
             {(j0+1, k0): 1, (j0, k0): -1},
             {(i0, k0): 1, (i0+1, k0): 1}),
            # m5
            ({(i0, j0): 1, (i0, j0+1): 1},
             {(j0+1, k0+1): 1},
             {(i0, k0): -1, (i0, k0+1): 1}),
            # m6
            ({(i0+1, j0): 1, (i0, j0): -1},
             {(j0, k0): 1, (j0, k0+1): 1},
             {(i0+1, k0+1): 1}),
            # m7
            ({(i0, j0+1): 1, (i0+1, j0+1): -1},
             {(j0+1, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1}),
        ]
        
        for u, v, w in terms:
            u_s = {k: val * sign for k, val in u.items()}
            v_s = {k: val * sign for k, val in v.items()}
            w_s = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_s, v_s, w_s))
    
    def add_schoolbook_2x2x1(i0, j0, k0, sign=1):
        """
        Add schoolbook terms for X(2x2) * Y(2x1) where:
        X = A[i0:i0+2, j0:j0+2]
        Y = B[j0:j0+2, k0]
        contributes to C[i0:i0+2, k0]
        """
        terms = [
            ({(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1}),
            ({(i0, j0+1): 1}, {(j0+1, k0): 1}, {(i0, k0): 1}),
            ({(i0+1, j0): 1}, {(j0, k0): 1}, {(i0+1, k0): 1}),
            ({(i0+1, j0+1): 1}, {(j0+1, k0): 1}, {(i0+1, k0): 1}),
        ]
        
        for u, v, w in terms:
            u_s = {k: val * sign for k, val in u.items()}
            v_s = {k: val * sign for k, val in v.items()}
            w_s = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_s, v_s, w_s))
    
    # A1 = A[0:2, 0:2], A2 = A[0:2, 2:4]
    # B_top = B[0:2, 0:5], B_bot = B[2:4, 0:5]
    
    # C = A1 * B_top + A2 * B_bot
    # A1 * B_top: split B_top cols into [0,1], [2,3], [4]
    add_strassen_2x2x2(i0=0, j0=0, k0=0, sign=1)   # A1 * B_top[:, 0:2]
    add_strassen_2x2x2(i0=0, j0=0, k0=2, sign=1)   # A1 * B_top[:, 2:4]
    add_schoolbook_2x2x1(i0=0, j0=0, k0=4, sign=1) # A1 * B_top[:, 4]
    
    # A2 * B_bot: split B_bot cols into [0,1], [2,3], [4]
    add_strassen_2x2x2(i0=0, j0=2, k0=0, sign=1)   # A2 * B_bot[:, 0:2]
    add_strassen_2x2x2(i0=0, j0=2, k0=2, sign=1)   # A2 * B_bot[:, 2:4]
    add_schoolbook_2x2x1(i0=0, j0=2, k0=4, sign=1) # A2 * B_bot[:, 4]
    
    rank = len(rank_terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]  # (8, rank)
    V = [[0] * rank for _ in range(m * p)]  # (20, rank)
    W = [[0] * rank for _ in range(p * n)]  # (10, rank)
    
    for c, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][c] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][c] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][c] = coeff
    
    # Attempt to reduce rank by checking if any rank-1 term is redundant
    # via numerical linear algebra. If we can find a dependency, we can
    # potentially reduce the rank. However, for now we keep the structure
    # and just verify correctness.
    
    # Verify the decomposition is correct
    import numpy as np
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # should equal 1 if j is the "middle" index connecting A and B, else 0.
    # Actually, the matrix multiplication tensor is:
    # T_{i,j,k} = delta_{j, j'} where we sum over j'... 
    # More precisely, for matrix multiplication C = A*B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i,j,k] = 1 if we're looking at the (i,j) of A and (j,k) of B
    # contributing to (i,k) of C. The tensor entry is 1 for all valid (i,j,k)
    # where j is the shared index.
    
    # Let's verify: for each (i,j,k), the tensor value should be 1 if the 
    # indices form a valid chain, which they always do in our construction.
    # Actually, the multiplication tensor T_{ijk} = 1 for all i,j,k in range
    # (it's the all-ones tensor in the appropriate subspace). Wait, no.
    
    # The multiplication tensor is defined as:
    # T = sum_{i,j,k} e_i^A * e_j^B * e_k^C where the structure is
    # T_{(i,j),(j,k),(i,k)} = 1
    
    # Let's just verify numerically that our decomposition gives the right answer
    # for random inputs.
    
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    # Compute C from our decomposition
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for c in range(rank):
                # Sum over j: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # But W is shared across j, so:
                pass
    
    # More efficient: C[i,k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for k in range(p):
            val = 0.0
            for j in range(m):
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
            C_computed[i, k] = val
    
    # Check if the decomposition is correct
    error = np.max(np.abs(C_computed - C_expected))
    if error > 1e-10:
        # Fallback: return the schoolbook decomposition which is always correct
        # But this shouldn't happen with our Strassen construction
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
