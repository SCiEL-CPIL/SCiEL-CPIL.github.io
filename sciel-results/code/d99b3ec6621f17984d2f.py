import time

DIMENSIONS = [2, 2, 7]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (4, 25)
    # V: (m*p, rank) = (14, 25)
    # W: (p*n, rank) = (14, 25)
    rank = 25
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block for column offset k0
    def add_strassen_block(k0):
        nonlocal col
        # Indices for B
        b00 = 0 * p + k0
        b01 = 0 * p + (k0 + 1)
        b10 = 1 * p + k0
        b11 = 1 * p + (k0 + 1)
        
        # Indices for C (Trace: C[k*n + i])
        # Block is C[:, k0:k0+2]
        # C[0, k0] -> index 2*k0
        # C[0, k0+1] -> index 2*k0 + 1
        # C[1, k0] -> index 2*k0 + 2
        # C[1, k0+1] -> index 2*k0 + 3
        c00 = 2 * k0
        c01 = 2 * k0 + 1
        c10 = 2 * k0 + 2
        c11 = 2 * k0 + 3
        
        # Term 1: P1 = (A00 + A11)(B00 + B11)
        # U: 1 at A00 (0), 1 at A11 (3)
        U[0][col] += 1
        U[3][col] += 1
        # V: 1 at B00, 1 at B11
        V[b00][col] += 1
        V[b11][col] += 1
        # W: 1 at C00, 1 at C11
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1
        
        # Term 2: P2 = (A10 + A11) B00
        # U: 1 at A10 (2), 1 at A11 (3)
        U[2][col] += 1
        U[3][col] += 1
        # V: 1 at B00
        V[b00][col] += 1
        # W: -1 at C00
        W[c00][col] += -1
        col += 1
        
        # Term 3: P3 = A00 (B10 - B00)
        # U: 1 at A00 (0)
        U[0][col] += 1
        # V: -1 at B00, 1 at B10
        V[b00][col] += -1
        V[b10][col] += 1
        # W: 1 at C01, -1 at C11
        W[c01][col] += 1
        W[c11][col] += -1
        col += 1
        
        # Term 4: P4 = A11 (B01 - B11)
        # U: 1 at A11 (3)
        U[3][col] += 1
        # V: 1 at B01, -1 at B11
        V[b01][col] += 1
        V[b11][col] += -1
        # W: -1 at C00, 1 at C10
        W[c00][col] += -1
        W[c10][col] += 1
        col += 1
        
        # Term 5: P5 = (A00 + A01) B11
        # U: 1 at A00 (0), 1 at A01 (1)
        U[0][col] += 1
        U[1][col] += 1
        # V: 1 at B11
        V[b11][col] += 1
        # W: 1 at C01, -1 at C11
        W[c01][col] += 1
        W[c11][col] += -1
        col += 1
        
        # Term 6: P6 = (A10 - A00) (B00 + B01)
        # U: -1 at A00 (0), 1 at A10 (2)
        U[0][col] += -1
        U[2][col] += 1
        # V: 1 at B00, 1 at B01
        V[b00][col] += 1
        V[b01][col] += 1
        # W: 1 at C10
        W[c10][col] += 1
        col += 1
        
        # Term 7: P7 = (A01 - A11) (B10 + B11)
        # U: 1 at A01 (1), -1 at A11 (3)
        U[1][col] += 1
        U[3][col] += -1
        # V: 1 at B10, 1 at B11
        V[b10][col] += 1
        V[b11][col] += 1
        # W: 1 at C00, 1 at C11
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1

    # Add Strassen blocks for k=0, 2, 4
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    # Add Schoolbook for k=6
    # C06 = A00 B06 + A01 B16
    # C16 = A10 B06 + A11 B16
    b06 = 0 * p + 6
    b16 = 1 * p + 6
    c06 = 2 * 6
    c16 = 2 * 6 + 1
    
    # Term 8: A00 B06 -> C06
    U[0][col] += 1
    V[b06][col] += 1
    W[c06][col] += 1
    col += 1
    
    # Term 9: A01 B16 -> C06
    U[1][col] += 1
    V[b16][col] += 1
    W[c06][col] += 1
    col += 1
    
    # Term 10: A10 B06 -> C16
    U[2][col] += 1
    V[b06][col] += 1
    W[c16][col] += 1
    col += 1
    
    # Term 11: A11 B16 -> C16
    U[3][col] += 1
    V[b16][col] += 1
    W[c16][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
