"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs):
        nonlocal col
        for i, c in zip(u_indices, u_coeffs):
            U[i][col] = c
        for i, c in zip(v_indices, v_coeffs):
            V[i][col] = c
        for i, c in zip(w_indices, w_coeffs):
            W[i][col] = c
        col += 1

    # Strassen blocks for columns (0,1), (2,3), (4,5)
    for b in range(3):
        c0 = 2 * b
        c1 = 2 * b + 1
        
        # P1 = (A00 + A11) * (B0,c0 + B1,c1)
        # U: A00(0), A11(3)
        # V: B0,c0(7*c0), B1,c1(7+c1)
        # W: C0,c0(2*c0) coeff 1, C1,c1(2*c1+1) coeff 1
        add_term([0, 3], [1, 1], [7*c0, 7+c1], [1, 1], [2*c0, 2*c1+1], [1, 1])
        
        # P2 = (A10 + A11) * B0,c0
        # U: A10(2), A11(3)
        # V: B0,c0(7*c0)
        # W: C1,c0(2*c0+1) coeff 1
        add_term([2, 3], [1, 1], [7*c0], [1], [2*c0+1], [1])
        
        # P3 = A00 * (B0,c1 - B1,c1)
        # U: A00(0)
        # V: B0,c1(7*c1), B1,c1(7+c1)
        # W: C0,c1(2*c1) coeff 1, C1,c1(2*c1+1) coeff 1
        add_term([0], [1], [7*c1, 7+c1], [1, -1], [2*c1, 2*c1+1], [1, 1])
        
        # P4 = A11 * (B1,c0 - B0,c0)
        # U: A11(3)
        # V: B1,c0(7+c0), B0,c0(7*c0)
        # W: C0,c0(2*c0) coeff 1, C1,c0(2*c0+1) coeff 1
        add_term([3], [1], [7+c0, 7*c0], [1, -1], [2*c0, 2*c0+1], [1, 1])
        
        # P5 = (A00 + A01) * B1,c1
        # U: A00(0), A01(1)
        # V: B1,c1(7+c1)
        # W: C0,c0(2*c0) coeff -1, C0,c1(2*c1) coeff 1
        add_term([0, 1], [1, 1], [7+c1], [1], [2*c0, 2*c1], [-1, 1])
        
        # P6 = (A10 - A00) * (B0,c0 + B0,c1)
        # U: A10(2), A00(0)
        # V: B0,c0(7*c0), B0,c1(7*c1)
        # W: C1,c1(2*c1+1) coeff 1
        add_term([2, 0], [1, -1], [7*c0, 7*c1], [1, 1], [2*c1+1], [1])
        
        # P7 = (A01 - A11) * (B1,c0 + B1,c1)
        # U: A01(1), A11(3)
        # V: B1,c0(7+c0), B1,c1(7+c1)
        # W: C0,c0(2*c0) coeff 1
        add_term([1, 3], [1, -1], [7+c0, 7+c1], [1, 1], [2*c0], [1])

    # Schoolbook for column 6
    c = 6
    
    # Q1 = A00 * B0,6
    add_term([0], [1], [7*c], [1], [2*c], [1])
    
    # Q2 = A01 * B1,6
    add_term([1], [1], [7+c], [1], [2*c], [1])
    
    # Q3 = A10 * B0,6
    add_term([2], [1], [7*c], [1], [2*c+1], [1])
    
    # Q4 = A11 * B1,6
    add_term([3], [1], [7+c], [1], [2*c+1], [1])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
