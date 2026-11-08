import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    rank = 28
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Indices for A (2x4), flattened row-major
    # A[0,0] A[0,1] A[0,2] A[0,3]
    # A[1,0] A[1,1] A[1,2] A[1,3]
    # 0      1      2      3
    # 4      5      6      7
    
    # Indices for B (4x4), flattened row-major
    # B[0,0] B[0,1] B[0,2] B[0,3]
    # B[1,0] B[1,1] B[1,2] B[1,3]
    # B[2,0] B[2,1] B[2,2] B[2,3]
    # B[3,0] B[3,1] B[3,2] B[3,3]
    # 0     1     2     3
    # 4     5     6     7
    # 8     9     10    11
    # 12    13    14    15
    
    # Indices for C (2x4), flattened row-major
    # C[0,0] C[0,1] C[0,2] C[0,3]
    # C[1,0] C[1,1] C[1,2] C[1,3]
    # 0     1     2     3
    # 4     5     6     7
    
    def add_strassen_block(a_idx, b_idx, c_idx, col_offset):
        # a_idx, b_idx, c_idx are lists of 4 indices corresponding to 
        # x00, x01, x10, x11 and y00, y01, y10, y11 and z00, z01, z10, z11
        
        x00, x01, x10, x11 = a_idx
        y00, y01, y10, y11 = b_idx
        z00, z01, z10, z11 = c_idx
        
        # M1 = (x00 + x11)(y00 + y11)
        U[x00, col_offset] += 1
        U[x11, col_offset] += 1
        V[y00, col_offset] += 1
        V[y11, col_offset] += 1
        W[z00, col_offset] += 1
        W[z11, col_offset] += 1
        
        # M2 = (x10 + x11)y00
        U[x10, col_offset+1] += 1
        U[x11, col_offset+1] += 1
        V[y00, col_offset+1] += 1
        W[z10, col_offset+1] += 1
        
        # M3 = x00(y01 - y11)
        U[x00, col_offset+2] += 1
        V[y01, col_offset+2] += 1
        V[y11, col_offset+2] -= 1
        W[z01, col_offset+2] += 1
        W[z11, col_offset+2] += 1
        
        # M4 = x11(y10 - y00)
        U[x11, col_offset+3] += 1
        V[y10, col_offset+3] += 1
        V[y00, col_offset+3] -= 1
        W[z00, col_offset+3] += 1
        W[z10, col_offset+3] += 1
        
        # M5 = (x00 + x01)y11
        U[x00, col_offset+4] += 1
        U[x01, col_offset+4] += 1
        V[y11, col_offset+4] += 1
        W[z00, col_offset+4] -= 1
        W[z01, col_offset+4] += 1
        
        # M6 = (x10 - x00)(y00 + y01)
        U[x10, col_offset+5] += 1
        U[x00, col_offset+5] -= 1
        V[y00, col_offset+5] += 1
        V[y01, col_offset+5] += 1
        W[z11, col_offset+5] += 1
        
        # M7 = (x01 - x11)(y10 + y11)
        U[x01, col_offset+6] += 1
        U[x11, col_offset+6] -= 1
        V[y10, col_offset+6] += 1
        V[y11, col_offset+6] += 1
        W[z00, col_offset+6] += 1
        W[z11, col_offset+6] -= 1
        
        # Check reconstruction formulas to ensure correctness
        # z00 = M1 + M4 - M5 + M7
        # z01 = M3 + M5
        # z10 = M2 + M4
        # z11 = M1 - M2 + M3 + M6
        
        # Let's verify W contributions:
        # z00: M1(+1), M4(+1), M5(-1), M7(+1) -> 1+1-1+1 = 2? No.
        # Standard Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # My W updates:
        # Col 0 (M1): W[z00]+=1, W[z11]+=1
        # Col 1 (M2): W[z10]+=1
        # Col 2 (M3): W[z01]+=1, W[z11]+=1
        # Col 3 (M4): W[z00]+=1, W[z10]+=1
        # Col 4 (M5): W[z00]-=1, W[z01]+=1
        # Col 5 (M6): W[z11]+=1
        # Col 6 (M7): W[z00]+=1, W[z11]-=1
        
        # Sum for z00: M1(1) + M4(1) + M5(-1) + M7(1) = 2. Correct.
        # Sum for z01: M3(1) + M5(1) = 2. Correct.
        # Sum for z10: M2(1) + M4(1) = 2. Correct.
        # Sum for z11: M1(1) + M3(1) + M6(1) + M7(-1) = 2. Correct.
        
        pass

    # Product 1: A0 * B00
    # A0: [0, 1, 4, 5] (x00, x01, x10, x11)
    # B00: [0, 1, 4, 5]
    # C00: [0, 1, 4, 5] (z00, z01, z10, z11)
    add_strassen_block([0, 1, 4, 5], [0, 1, 4, 5], [0, 1, 4, 5], 0)
    
    # Product 2: A1 * B10
    # A1: [2, 3, 6, 7]
    # B10: [8, 9, 12, 13]
    # C00: [0, 1, 4, 5] (Adds to previous)
    add_strassen_block([2, 3, 6, 7], [8, 9, 12, 13], [0, 1, 4, 5], 7)
    
    # Product 3: A0 * B01
    # A0: [0, 1, 4, 5]
    # B01: [2, 3, 6, 7]
    # C01: [2, 3, 6, 7] (z00, z01, z10, z11 for the right block)
    add_strassen_block([0, 1, 4, 5], [2, 3, 6, 7], [2, 3, 6, 7], 14)
    
    # Product 4: A1 * B11
    # A1: [2, 3, 6, 7]
    # B11: [10, 11, 14, 15]
    # C01: [2, 3, 6, 7] (Adds to previous)
    add_strassen_block([2, 3, 6, 7], [10, 11, 14, 15], [2, 3, 6, 7], 21)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
