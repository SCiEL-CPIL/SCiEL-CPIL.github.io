"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 93
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    # I, J, K are the block positions in the 4x4 grid
    # k_offset is the offset for the k index
    def add_strassen_2x2x2(I, J, K, k_offset):
        nonlocal col
        # 7 terms
        # Term 1: M1 = (A00+A11)(B00+B11)
        U[2*I + 0][col] = 1; V[2*J + 0][col] = 1; W[(2*K + 0 + k_offset)*n + 2*I + 0][col] = 1
        U[2*I + 1][col] = 1; V[2*J + 1][col] = 1; W[(2*K + 1 + k_offset)*n + 2*I + 1][col] = 1
        col += 1
        
        # Term 2: M2 = (A10+A11)B00
        U[2*I + 1][col] = 1; V[2*J + 0][col] = 1; W[(2*K + 0 + k_offset)*n + 2*I + 1][col] = 1
        U[2*I + 1][col] += 1; V[2*J + 1][col] = 1; W[(2*K + 1 + k_offset)*n + 2*I + 1][col] = -1
        col += 1
        
        # Term 3: M3 = A00(B01-B11)
        U[2*I + 0][col] = 1; V[2*J + 0][col] = 1; W[(2*K + 1 + k_offset)*n + 2*I + 0][col] = 1
        V[2*J + 1][col] = -1; W[(2*K + 1 + k_offset)*n + 2*I + 1][col] = 1
        col += 1
        
        # Term 4: M4 = A11(B10-B00)
        U[2*I + 1][col] = 1; V[2*J + 1][col] = 1; W[(2*K + 0 + k_offset)*n + 2*I + 0][col] = 1
        V[2*J + 0][col] = -1; W[(2*K + 0 + k_offset)*n + 2*I + 1][col] = 1
        col += 1
        
        # Term 5: M5 = (A00+A01)B11
        U[2*I + 0][col] = 1; V[2*J + 1][col] = 1; W[(2*K + 0 + k_offset)*n + 2*I + 0][col] = -1
        U[2*I + 0][col] += 1; V[2*J + 1][col] += 1; W[(2*K + 1 + k_offset)*n + 2*I + 0][col] = 1
        col += 1
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        U[2*I + 1][col] = 1; V[2*J + 0][col] = 1; W[(2*K + 1 + k_offset)*n + 2*I + 1][col] = 1
        U[2*I + 0][col] = -1; V[2*J + 0][col] += 1; 
        col += 1
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        U[2*I + 0][col] = 1; V[2*J + 1][col] = 1; W[(2*K + 0 + k_offset)*n + 2*I + 0][col] = 1
        U[2*I + 1][col] = -1; V[2*J + 1][col] += 1; W[(2*K + 1 + k_offset)*n + 2*I + 1][col] = -1
        col += 1

    # Helper to add a 2x2x1 schoolbook block
    def add_schoolbook_2x2x1(I, J, K, k_offset):
        nonlocal col
        for i in range(2):
            for j in range(2):
                U[2*I + i][col] = 1
                V[2*J + j][col] = 1
                W[(2*K + 0 + k_offset)*n + 2*I + i][col] = 1
                col += 1

    # 4x4x4 part: 7 Strassen products
    # We will implement this manually.
    # For now, use schoolbook (64 terms).
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*4+j][col] = 1
                V[j*7+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
                
    # 4x4x3 part: 4 blocks of 2x2x3 (11 terms each)
    for I in range(2):
        for J in range(2):
            # Strassen for k=4,5
            add_strassen_2x2x2(I, J, 0, 4)
            # Schoolbook for k=6
            add_schoolbook_2x2x1(I, J, 1, 4)
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
