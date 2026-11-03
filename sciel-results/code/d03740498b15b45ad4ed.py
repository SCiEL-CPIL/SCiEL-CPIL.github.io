"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Total rank estimate
    # Strassen blocks: 3 * 7 = 21
    # Trivial blocks: 4 + 12 + 4 + 4 + 2 = 26
    # Total: 47
    rank = 47
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        # Strassen 2x2x2
        # Indices: i in [0,1], j in [0,1], k in [0,1] relative to offsets
        # M1: (A00+A11)(B00+B11)
        # U: (i0, j0)=1, (i0+1, j0+1)=1
        # V: (j0, k0)=1, (j0+1, k0+1)=1
        # W: (k0, i0)=1, (k0+1, i0+1)=1
        U[i0 * m + j0, col] = 1
        U[(i0+1) * m + (j0+1), col] = 1
        V[j0 * p + k0, col] = 1
        V[(j0+1) * p + (k0+1), col] = 1
        W[k0 * n + i0, col] = 1
        W[(k0+1) * n + (i0+1), col] = 1
        col += 1
        
        # M2: (A10+A11)B00
        U[i0 * m + (j0+1), col] = 1 # A10? No, A10 is i=1, j=0.
        # Wait, standard indexing A_ij. i=row, j=col.
        # A10 is row 1, col 0.
        # My previous U mapping: U[i*m+j].
        # So A10 is U[1*m+0].
        # In relative coords: i=1, j=0.
        # So U[(i0+1)*m + j0] = 1.
        # A11 is i=1, j=1. U[(i0+1)*m + (j0+1)] = 1.
        U[i0 * m + (j0+1), col] = 0 # Clear previous if any? No, we are writing new col.
        U[(i0+1) * m + j0, col] = 1
        U[(i0+1) * m + (j0+1), col] = 1
        V[j0 * p + k0, col] = 1
        W[k0 * n + (i0+1), col] = 1 # C10 gets M2
        W[(k0+1) * n + (i0+1), col] = -1 # C11 gets -M2
        col += 1
        
        # M3: A00(B01-B11)
        U[i0 * m + j0, col] = 1
        V[j0 * p + (k0+1), col] = 1
        V[(j0+1) * p + (k0+1), col] = -1
        W[k0 * n + i0, col] = 0 # C00 no M3
        W[(k0+1) * n + i0, col] = 1 # C01 gets M3
        W[k0 * n + (i0+1), col] = 0 # C10 no M3
        W[(k0+1) * n + (i0+1), col] = 1 # C11 gets M3
        col += 1
        
        # M4: A11(B10-B00)
        U[(i0+1) * m + (j0+1), col] = 1
        V[(j0+1) * p + k0, col] = 1
        V[j0 * p + k0, col] = -1
        W[k0 * n + i0, col] = -1 # C00 gets -M4
        W[(k0+1) * n + i0, col] = 0
        W[k0 * n + (i0+1), col] = 1 # C10 gets M4
        W[(k0+1) * n + (i0+1), col] = 0
        col += 1
        
        # M5: (A00+A01)B11
        U[i0 * m + j0, col] = 1
        U[i0 * m + (j0+1), col] = 1
        V[(j0+1) * p + (k0+1), col] = 1
        W[k0 * n + i0, col] = 1 # C00 gets M5
        W[(k0+1) * n + i0, col] = 1 # C01 gets M5
        W[k0 * n + (i0+1), col] = 0
        W[(k0+1) * n + (i0+1), col] = 0
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[(i0+1) * m + j0, col] = 1
        U[i0 * m + j0, col] = -1
        V[j0 * p + k0, col] = 1
        V[j0 * p + (k0+1), col] = 1
        W[k0 * n + i0, col] = 0
        W[(k0+1) * n + i0, col] = 0
        W[k0 * n + (i0+1), col] = 0
        W[(k0+1) * n + (i0+1), col] = 1 # C11 gets M6
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[i0 * m + (j0+1), col] = 1
        U[(i0+1) * m + (j0+1), col] = -1
        V[(j0+1) * p + k0, col] = 1
        V[(j0+1) * p + (k0+1), col] = 1
        W[k0 * n + i0, col] = -1 # C00 gets -M7
        W[(k0+1) * n + i0, col] = 0
        W[k0 * n + (i0+1), col] = 0
        W[(k0+1) * n + (i0+1), col] = 0
        col += 1

    def add_trivial_block(n_b, m_b, p_b, i0, j0, k0):
        nonlocal col
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U[(i0+i) * m + (j0+j), col] = 1
                    V[(j0+j) * p + (k0+k), col] = 1
                    W[(k0+k) * n + (i0+i), col] = 1
                    col += 1

    # 1. A1 B11 (2x2x2) Strassen at (0,0,0)
    add_strassen_block(0, 0, 0)
    
    # 2. A1 B12,1 (2x2x2) Strassen at (0,0,2)
    add_strassen_block(0, 0, 2)
    
    # 3. A1 B12,2 (2x2x1) Trivial at (0,0,4)
    add_trivial_block(2, 2, 1, 0, 0, 4)
    
    # 4. A2 B21 (2x3x2) Trivial at (0,2,0)
    add_trivial_block(2, 3, 2, 0, 2, 0)
    
    # 5. A2,1 B22,11 (2x2x2) Strassen at (0,2,2)
    add_strassen_block(0, 2, 2)
    
    # 6. A2,1 B22,12 (2x2x1) Trivial at (0,2,4)
    add_trivial_block(2, 2, 1, 0, 2, 4)
    
    # 7. A2,2 B22,21 (2x1x2) Trivial at (0,4,2)
    add_trivial_block(2, 1, 2, 0, 4, 2)
    
    # 8. A2,2 B22,22 (2x1x1) Trivial at (0,4,4)
    add_trivial_block(2, 1, 1, 0, 4, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
