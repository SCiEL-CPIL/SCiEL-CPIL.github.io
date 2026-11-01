# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    Uses Strassen's algorithm decomposition (optimal rank for 2x2).
    Constructed programmatically to ensure structural distinctness and clarity.
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    # Initialize U, V, W as zeros
    # U shape: (n*m, rank) = (4, 7)
    # V shape: (m*p, rank) = (4, 7)
    # W shape: (p*n, rank) = (4, 7)
    # Note: W index is k*n + i.
    # U index is i*m + j.
    # V index is j*p + k.
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Strassen Products M1..M7
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Helper to add to U, V, W
    def add_product(idx, a_coeffs, b_coeffs, c_coeffs):
        """
        idx: 0-6 for M1-M7
        a_coeffs: dict { (i,j): coeff }
        b_coeffs: dict { (j,k): coeff }
        c_coeffs: dict { (i,k): coeff }
        """
        for (i, j), val in a_coeffs.items():
            U[i*m + j, idx] += val
        for (j, k), val in b_coeffs.items():
            V[j*p + k, idx] += val
        for (i, k), val in c_coeffs.items():
            # W index is k*n + i
            W[k*n + i, idx] += val

    # M1
    add_product(0, 
                {(0,0): 1, (1,1): 1}, 
                {(0,0): 1, (1,1): 1}, 
                {(0,0): 1, (1,1): 1}) # Wait, C coefficients are in W, not here directly. 
                # Actually, W is the combination of M's for each C element.
                # The U and V define the M's. W defines C as sum of M's.
                # So I just need to define U and V for each M, and W for each C.
                
    # Let's reset and do it properly.
    U[:] = 0
    V[:] = 0
    W[:] = 0
    
    # Define U and V for each Strassen product
    # M1: (A00+A11)(B00+B11)
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    
    # M2: (A10+A11)B00
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    
    # M3: A00(B01-B11)
    U[0, 2] += 1
    V[1, 2] += 1; V[3, 2] -= 1
    
    # M4: A11(B10-B00)
    U[3, 3] += 1
    V[2, 3] += 1; V[0, 3] -= 1
    
    # M5: (A00+A01)B11
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    
    # M6: (A10-A00)(B00+B01)
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[1, 5] += 1
    
    # M7: (A01-A11)(B10+B11)
    U[1, 6] += 1; U[3, 6] -= 1
    V[2, 6] += 1; V[3, 6] += 1
    
    # Define W (C coefficients)
    # C00 = M1 + M4 - M5 - M7
    W[0, 0] += 1; W[0, 3] += 1; W[0, 4] -= 1; W[0, 6] -= 1
    # C10 = M2 + M4  (Index 1: k=0, i=1)
    W[1, 1] += 1; W[1, 3] += 1
    # C01 = M3 + M5  (Index 2: k=1, i=0)
    W[2, 2] += 1; W[2, 4] += 1
    # C11 = M1 - M2 + M3 + M6 (Index 3: k=1, i=1)
    W[3, 0] += 1; W[3, 1] -= 1; W[3, 2] += 1; W[3, 5] += 1
    
    # Cleanup -0.0
    U[U == 0] = 0.0
    V[V == 0] = 0.0
    W[W == 0] = 0.0

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
