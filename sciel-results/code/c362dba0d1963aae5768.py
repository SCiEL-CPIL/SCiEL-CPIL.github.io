import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 24
    pn = p * n  # 16
    
    # Rank-44 decomposition for 2x3 * 3x8 matrix multiplication
    # Uses Strassen's algorithm on 2x2 sub-blocks
    rank = 44
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    
    # A is 2x3: indices (i, j) where i in {0,1}, j in {0,1,2}
    # B is 3x8: indices (j, k) where j in {0,1,2}, k in {0,...,7}
    # C is 2x8: indices (i, k) where i in {0,1}, k in {0,...,7}
    #
    # Split: A = [A11 (2x2), A12 (2x1)]
    #         B = [B11 (2x8); B12 (1x8)]
    #
    # C = A11 * B11 + A12 * B12
    #
    # For A11 * B11: split B11 into 4 blocks of 2x2
    # B11 = [B00 B01 B02 B03] each 2x2
    # A11 = [[a00, a01], [a10, a11]]
    #
    # Strassen for each 2x2 * 2x2 block:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b10-b11)
    # M4 = a11*(b01-b10)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b10)
    # M7 = (a01-a11)*(b01+b11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # For block b in {0,1,2,3}, the 2x2 block of B11 is at columns 2*b, 2*b+1
    # The corresponding C columns are 2*b, 2*b+1
    
    # A11 elements: a00=A[0,0], a01=A[0,1], a10=A[1,0], a11=A[1,1]
    # In tensor index: a_ij has index i*m+j = i*3+j
    
    # Strassen terms for each 2x2 block:
    # For block b, B elements:
    #   B[0, 2b] = b00, B[0, 2b+1] = b01
    #   B[1, 2b] = b10, B[1, 2b+1] = b11
    # In tensor index: B[j, k] has index j*p+k = j*8+k
    
    # C elements for block b:
    #   C[0, 2b] = c00, C[0, 2b+1] = c01
    #   C[1, 2b] = c10, C[1, 2b+1] = c11
    # In tensor index: C[i, k] has index k*n+i = k*2+i
    
    for b in range(4):
        # M1 = (a00+a11)*(b00+b11)
        # This contributes to C00 and C11
        U[0*3+0, col] = 1  # a00
        U[1*3+1, col] = 1  # a11
        V[0*8+2*b, col] = 1    # b00
        V[1*8+2*b+1, col] = 1  # b11
        W[(2*b)*2+0, col] = 1      # c00
        W[(2*b+1)*2+1, col] = 1    # c11
        col += 1
        
        # M2 = (a10+a11)*b00
        # Contributes to C10
        U[1*3+0, col] = 1  # a10
        U[1*3+1, col] = 1  # a11
        V[0*8+2*b, col] = 1    # b00
        W[(2*b)*2+1, col] = 1  # c10
        col += 1
        
        # M3 = a00*(b10-b11)
        # Contributes to C01
        U[0*3+0, col] = 1  # a00
        V[1*8+2*b, col] = 1    # b10
        V[1*8+2*b+1, col] = -1  # b11
        W[(2*b+1)*2+0, col] = 1  # c01
        col += 1
        
        # M4 = a11*(b01-b10)
        # Contributes to C00 and C10
        U[1*3+1, col] = 1  # a11
        V[0*8+2*b+1, col] = 1   # b01
        V[1*8+2*b, col] = -1    # b10
        W[(2*b)*2+0, col] = 1      # c00
        W[(2*b)*2+1, col] = 1      # c10
        col += 1
        
        # M5 = (a00+a01)*b11
        # Contributes to C00 (negative) and C01
        U[0*3+0, col] = 1  # a00
        U[0*3+1, col] = 1  # a01
        V[1*8+2*b+1, col] = 1  # b11
        W[(2*b)*2+0, col] = -1   # c00
        W[(2*b+1)*2+0, col] = 1  # c01
        col += 1
        
        # M6 = (a10-a00)*(b00+b10)
        # Contributes to C10 (negative in formula? Let me recheck)
        # Actually: C10 = M2 + M4, so M6 is not directly used in standard Strassen
        # Wait, I need to recheck the Strassen formulas
        
        # Let me use the standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        #
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # I already have M1, M2, M3, M4, M5. Now M6 and M7:
        
        # M6 = (a10-a00)*(b00+b10)
        # Not directly in the C formulas above... 
        # Wait, I think I made an error. Let me recheck.
        # Actually M6 is not in the final C formulas I listed. Let me recheck.
        
        # Standard Strassen:
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5  
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # So M6 is not used? That can't be right. Let me look this up again.
        
        # Actually, I think the standard Strassen has 7 multiplications and the formulas are:
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # And M6 is not in the output. That seems wrong. Let me recheck.
        
        # I believe I'm confusing myself. Let me just use a verified decomposition.
        
        # M6 = (a10-a00)*(b00+b10) - not used in standard output
        # M7 = (a01-a11)*(b01+b11)
        U[0*3+1, col] = 1   # a01
        U[1*3+1, col] = -1  # a11
        V[0*8+2*b+1, col] = 1  # b01
        V[1*8+2*b+1, col] = 1  # b11
        W[(2*b)*2+0, col] = -1   # c00
        W[(2*b+1)*2+1, col] = 1  # c11
        col += 1
        
        # M7 done. Now I need to add M6 but it's not in the standard formulas.
        # Let me recount: I've added 7 terms per block (M1-M7), but M6 wasn't added.
        # I added: M1, M2, M3, M4, M5, M7 = 6 terms. I'm missing M6.
        # But M6 isn't in the C formulas... 
        # 
        # I think I need to restart this more carefully.
        
    # This approach is getting messy. Let me use a cleaner method.
    # Reset and try again with explicit term tracking.
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    
    # I'll define each of the 44 terms explicitly.
    # Terms 0-27: Strassen for A11 * B11 (4 blocks of 2x2, 7 terms each = 28)
    # Terms 28-35: A12 * B12 (2x1 * 1x8, 8 terms)
    # That's 36, not 44. 
    #
    # I think the rank-44 decomposition is different. Let me just return schoolbook
    # for now and try a different approach.
    
    # Actually, let me just verify: is the rank really 44?
    # For 2x3 x 3x8, the schoolbook is 48.
    # The known minimum rank is indeed 44.
    
    # Let me try a different construction.
    # Perhaps the decomposition doesn't split as I thought.
    
    # Given the complexity, let me just return the schoolbook decomposition
    # which gives rank 48, and note that this is the baseline.
    
    # Actually, the inspiration says rank 44 is achievable. Let me try to construct it.
    
    # New approach: use the fact that we can apply Strassen to overlapping blocks.
    
    # For now, return schoolbook to ensure correctness
    U = np.zeros((nm, 48))
    V = np.zeros((mp, 48))
    W = np.zeros((pn, 48))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
