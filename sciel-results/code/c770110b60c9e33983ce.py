import time
import sys

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen block (2x2x2)
    # i_base: not used directly as i is always 0,1, but we need to map local i to global i
    # j_base: starting column of A / starting row of B
    # k_base: starting column of B
    # col: starting column index in U, V, W
    def add_strassen(col, j_base, k_base):
        # Local indices for A (2x2), B (2x2), C (2x2)
        # A local (i, ja) -> Global U[i*m + (j_base + ja)]
        # B local (jb, k) -> Global V[(j_base + jb)*p + (k_base + k)]
        # C local (i, kc) -> Global W[(k_base + kc)*n + i]
        
        # Strassen products M1..M7
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Coefficients for A (U)
        # c is column index offset 0..6
        # A coefficients:
        # M1: A00:1, A11:1
        # M2: A10:1, A11:1
        # M3: A00:1
        # M4: A11:1
        # M5: A00:1, A01:1
        # M6: A10:1, A00:-1
        # M7: A01:1, A11:-1
        
        # Coefficients for B (V)
        # M1: B00:1, B11:1
        # M2: B00:1
        # M3: B01:1, B11:-1
        # M4: B10:1, B00:-1
        # M5: B11:1
        # M6: B00:1, B01:1
        # M7: B10:1, B11:1
        
        # Coefficients for C (W)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Fill U
        # Local A indices: 00, 01, 10, 11
        # Global U index: i*m + (j_base + ja)
        
        # M1 (col+0)
        U[0*m + (j_base+0)][col+0] += 1
        U[1*m + (j_base+1)][col+0] += 1
        # M2 (col+1)
        U[1*m + (j_base+0)][col+1] += 1
        U[1*m + (j_base+1)][col+1] += 1
        # M3 (col+2)
        U[0*m + (j_base+0)][col+2] += 1
        # M4 (col+3)
        U[1*m + (j_base+1)][col+3] += 1
        # M5 (col+4)
        U[0*m + (j_base+0)][col+4] += 1
        U[0*m + (j_base+1)][col+4] += 1
        # M6 (col+5)
        U[1*m + (j_base+0)][col+5] += 1
        U[0*m + (j_base+0)][col+5] -= 1
        # M7 (col+6)
        U[0*m + (j_base+1)][col+6] += 1
        U[1*m + (j_base+1)][col+6] -= 1

        # Fill V
        # Local B indices: 00, 01, 10, 11
        # Global V index: (j_base + jb)*p + (k_base + kc)
        
        # M1 (col+0)
        V[(j_base+0)*p + (k_base+0)][col+0] += 1
        V[(j_base+1)*p + (k_base+1)][col+0] += 1
        # M2 (col+1)
        V[(j_base+0)*p + (k_base+0)][col+1] += 1
        # M3 (col+2)
        V[(j_base+0)*p + (k_base+1)][col+2] += 1
        V[(j_base+1)*p + (k_base+1)][col+2] -= 1
        # M4 (col+3)
        V[(j_base+1)*p + (k_base+0)][col+3] += 1
        V[(j_base+0)*p + (k_base+0)][col+3] -= 1
        # M5 (col+4)
        V[(j_base+1)*p + (k_base+1)][col+4] += 1
        # M6 (col+5)
        V[(j_base+0)*p + (k_base+0)][col+5] += 1
        V[(j_base+0)*p + (k_base+1)][col+5] += 1
        # M7 (col+6)
        V[(j_base+1)*p + (k_base+0)][col+6] += 1
        V[(j_base+1)*p + (k_base+1)][col+6] += 1

        # Fill W
        # Local C indices: 00, 01, 10, 11
        # Global W index: (k_base + kc)*n + i
        
        # C00 (i=0, kc=0) -> W[(k_base+0)*n + 0]
        W[(k_base+0)*n + 0][col+0] += 1  # M1
        W[(k_base+0)*n + 0][col+3] += 1  # M4
        W[(k_base+0)*n + 0][col+4] -= 1  # M5
        W[(k_base+0)*n + 0][col+6] += 1  # M7
        
        # C01 (i=0, kc=1) -> W[(k_base+1)*n + 0]
        W[(k_base+1)*n + 0][col+2] += 1  # M3
        W[(k_base+1)*n + 0][col+4] += 1  # M5
        
        # C10 (i=1, kc=0) -> W[(k_base+0)*n + 1]
        W[(k_base+0)*n + 1][col+1] += 1  # M2
        W[(k_base+0)*n + 1][col+3] += 1  # M4
        
        # C11 (i=1, kc=1) -> W[(k_base+1)*n + 1]
        W[(k_base+1)*n + 1][col+0] += 1  # M1
        W[(k_base+1)*n + 1][col+1] -= 1  # M2
        W[(k_base+1)*n + 1][col+2] += 1  # M3
        W[(k_base+1)*n + 1][col+5] += 1  # M6

    # Helper to add a Schoolbook block (2x2x1)
    def add_schoolbook(col, j_base, k_base):
        # A (2x2) * B (2x1) -> C (2x1)
        # C00 = A00*B00 + A01*B10
        # C10 = A10*B00 + A11*B10
        
        # 4 products
        # P1 = A00 B00
        # P2 = A01 B10
        # P3 = A10 B00
        # P4 = A11 B10
        
        # U
        U[0*m + (j_base+0)][col+0] += 1 # P1 A00
        U[0*m + (j_base+1)][col+1] += 1 # P2 A01
        U[1*m + (j_base+0)][col+2] += 1 # P3 A10
        U[1*m + (j_base+1)][col+3] += 1 # P4 A11
        
        # V
        V[(j_base+0)*p + (k_base+0)][col+0] += 1 # P1 B00
        V[(j_base+1)*p + (k_base+0)][col+1] += 1 # P2 B10
        V[(j_base+0)*p + (k_base+0)][col+2] += 1 # P3 B00
        V[(j_base+1)*p + (k_base+0)][col+3] += 1 # P4 B10
        
        # W
        # C00 (i=0, kc=0) -> W[(k_base+0)*n + 0]
        W[(k_base+0)*n + 0][col+0] += 1 # P1
        W[(k_base+0)*n + 0][col+1] += 1 # P2
        # C10 (i=1, kc=0) -> W[(k_base+0)*n + 1]
        W[(k_base+0)*n + 1][col+2] += 1 # P3
        W[(k_base+0)*n + 1][col+3] += 1 # P4

    # Block 1: j in 0,1
    add_strassen(0, 0, 0)
    add_strassen(7, 0, 2)
    add_strassen(14, 0, 4)
    add_schoolbook(21, 0, 6)
    
    # Block 2: j in 2,3
    add_strassen(25, 2, 0)
    add_strassen(32, 2, 2)
    add_strassen(39, 2, 4)
    add_schoolbook(46, 2, 6)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
