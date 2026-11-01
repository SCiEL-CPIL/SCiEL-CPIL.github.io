import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=14
    # Rank 49 decomposition using 7 blocks of Strassen 2x2x2
    
    rank = 49
    # U: (n*m, rank) = (4, 49)
    # V: (m*p, rank) = (28, 49)
    # W: (p*n, rank) = (28, 49)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products for 2x2x2
    # A is 2x2, B is 2x2, C is 2x2
    # Indices for A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Strassen Formulas:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We map these to the global indices.
    # For block s (0 to 6):
    # B indices: j*p + k. j in {0,1}, k in {2s, 2s+1}.
    #   B00_s -> V[0*p + 2s]
    #   B01_s -> V[0*p + 2s+1]
    #   B10_s -> V[1*p + 2s]
    #   B11_s -> V[1*p + 2s+1]
    # C indices: k*n + i. k in {2s, 2s+1}, i in {0,1}.
    #   C00_s -> W[2s*n + 0]
    #   C01_s -> W[(2s+1)*n + 0]
    #   C10_s -> W[2s*n + 1]
    #   C11_s -> W[(2s+1)*n + 1]
    
    # U indices are always 0,1,2,3 for A00, A01, A10, A11.
    
    for s in range(7):
        base_col = s * 7
        
        # Column offset for this block
        # M1
        col = base_col
        U[0][col] += 1  # A00
        U[3][col] += 1  # A11
        V[0*p + 2*s][col] += 1      # B00
        V[1*p + 2*s+1][col] += 1    # B11
        W[2*s*n + 0][col] += 1      # C00
        W[2*s*n + 1][col] += 1      # C10? No, C11? 
        # Let's check Strassen C indices:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So M1 contributes to C00 and C11.
        W[2*s*n + 0][col] = 1       # C00
        W[(2*s+1)*n + 1][col] = 1   # C11
        
        # M2
        col = base_col + 1
        U[2][col] += 1  # A10
        U[3][col] += 1  # A11
        V[0*p + 2*s][col] += 1      # B00
        W[2*s*n + 1][col] += 1      # C10
        W[(2*s+1)*n + 1][col] += -1 # C11
        
        # M3
        col = base_col + 2
        U[0][col] += 1  # A00
        V[0*p + 2*s+1][col] += 1    # B01
        V[1*p + 2*s+1][col] += -1   # B11
        W[(2*s+1)*n + 0][col] += 1  # C01
        W[(2*s+1)*n + 1][col] += 1  # C11
        
        # M4
        col = base_col + 3
        U[3][col] += 1  # A11
        V[1*p + 2*s][col] += 1      # B10
        V[0*p + 2*s][col] += -1     # B00
        W[2*s*n + 0][col] += 1      # C00
        W[2*s*n + 1][col] += 1      # C10
        
        # M5
        col = base_col + 4
        U[0][col] += 1  # A00
        U[1][col] += 1  # A01
        V[1*p + 2*s+1][col] += 1    # B11
        W[2*s*n + 0][col] += -1     # C00
        W[(2*s+1)*n + 0][col] += 1  # C01
        
        # M6
        col = base_col + 5
        U[2][col] += 1  # A10
        U[0][col] += -1 # A00
        V[0*p + 2*s][col] += 1      # B00
        V[0*p + 2*s+1][col] += 1    # B01
        W[(2*s+1)*n + 1][col] += 1  # C11
        
        # M7
        col = base_col + 6
        U[1][col] += 1  # A01
        U[3][col] += -1 # A11
        V[1*p + 2*s][col] += 1      # B10
        V[1*p + 2*s+1][col] += 1    # B11
        W[2*s*n + 0][col] += 1      # C00
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
