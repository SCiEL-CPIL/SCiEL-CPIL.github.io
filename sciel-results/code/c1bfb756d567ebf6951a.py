import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    # Rank 11 decomposition using Strassen for 2x2x2 and standard for 2x2x1
    n, m, p = DIMENSIONS
    rank = 11
    
    # Initialize matrices
    # U: (n*m, rank) = (4, 11)
    # V: (m*p, rank) = (6, 11)
    # W: (p*n, rank) = (6, 11)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set values
    def set_U(i, j, c, val):
        U[i * m + j][c] = val
        
    def set_V(j, k, c, val):
        V[j * p + k][c] = val
        
    def set_W(k, i, c, val):
        W[k * n + i][c] = val

    # Term 1: M1 = (a00 + a11)(b00 + b11)
    # A: (0,0)=1, (1,1)=1
    # B: (0,0)=1, (1,1)=1
    # C: (0,0)=1, (1,1)=1
    set_U(0, 0, 0, 1); set_U(1, 1, 0, 1)
    set_V(0, 0, 0, 1); set_V(1, 1, 0, 1)
    set_W(0, 0, 0, 1); set_W(1, 1, 0, 1)
    
    # Term 2: M2 = (a10 + a11)b00
    # A: (1,0)=1, (1,1)=1
    # B: (0,0)=1
    # C: (0,1)=1 (c10)
    set_U(1, 0, 1, 1); set_U(1, 1, 1, 1)
    set_V(0, 0, 1, 1)
    set_W(0, 1, 1, 1)
    
    # Term 3: M3 = a00(b01 - b11)
    # A: (0,0)=1
    # B: (0,1)=1, (1,1)=-1
    # C: (1,0)=1 (c01)
    set_U(0, 0, 2, 1)
    set_V(0, 1, 2, 1); set_V(1, 1, 2, -1)
    set_W(1, 0, 2, 1)
    
    # Term 4: M4 = a11(b10 - b00)
    # A: (1,1)=1
    # B: (1,0)=1, (0,0)=-1
    # C: (0,0)=-1 (c00), (0,1)=1 (c10)
    set_U(1, 1, 3, 1)
    set_V(1, 0, 3, 1); set_V(0, 0, 3, -1)
    set_W(0, 0, 3, -1); set_W(0, 1, 3, 1)
    
    # Term 5: M5 = (a00 + a01)b11
    # A: (0,0)=1, (0,1)=1
    # B: (1,1)=1
    # C: (1,0)=1 (c01)
    set_U(0, 0, 4, 1); set_U(0, 1, 4, 1)
    set_V(1, 1, 4, 1)
    set_W(1, 0, 4, 1)
    
    # Term 6: M6 = (a10 - a00)(b00 + b01)
    # A: (1,0)=1, (0,0)=-1
    # B: (0,0)=1, (0,1)=1
    # C: (1,1)=1 (c11)
    set_U(1, 0, 5, 1); set_U(0, 0, 5, -1)
    set_V(0, 0, 5, 1); set_V(0, 1, 5, 1)
    set_W(1, 1, 5, 1)
    
    # Term 7: M7 = (a01 - a11)(b10 + b11)
    # A: (0,1)=1, (1,1)=-1
    # B: (1,0)=1, (1,1)=1
    # C: (0,0)=1 (c00)
    set_U(0, 1, 6, 1); set_U(1, 1, 6, -1)
    set_V(1, 0, 6, 1); set_V(1, 1, 6, 1)
    set_W(0, 0, 6, 1)
    
    # Term 8: M8 = a00 b02
    # A: (0,0)=1
    # B: (0,2)=1
    # C: (2,0)=1 (c02)
    set_U(0, 0, 7, 1)
    set_V(0, 2, 7, 1)
    set_W(2, 0, 7, 1)
    
    # Term 9: M9 = a01 b12
    # A: (0,1)=1
    # B: (1,2)=1
    # C: (2,0)=1 (c02)
    set_U(0, 1, 8, 1)
    set_V(1, 2, 8, 1)
    set_W(2, 0, 8, 1)
    
    # Term 10: M10 = a10 b02
    # A: (1,0)=1
    # B: (0,2)=1
    # C: (2,1)=1 (c12)
    set_U(1, 0, 9, 1)
    set_V(0, 2, 9, 1)
    set_W(2, 1, 9, 1)
    
    # Term 11: M11 = a11 b12
    # A: (1,1)=1
    # B: (1,2)=1
    # C: (2,1)=1 (c12)
    set_U(1, 1, 10, 1)
    set_V(1, 2, 10, 1)
    set_W(2, 1, 10, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
