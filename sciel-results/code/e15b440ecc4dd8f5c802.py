import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Use Strassen's 2x2x2 decomposition tiled appropriately
    # A is 2x4, B is 4x8
    # We can view this as computing 2x4 x 4x8 = 2x8
    # Split B into two 4x4 blocks: B = [B1, B2]
    # C = [A*B1, A*B2] where each A*B is 2x4
    # For each 2x4 x 4x4 product, we can use a tiling approach
    
    # Actually, let's use a more direct approach.
    # The schoolbook rank is 2*4*8 = 64.
    # Using Strassen on 2x2 blocks:
    # A is 2x4 = [A1 | A2] where A1, A2 are 2x2
    # B is 4x8 = [B1; B2] where B1, B2 are 2x8
    # C = A1*B1 + A2*B2, where each product is 2x8
    
    # For 2x2 * 2x8:
    # Split B1 into [B11 | B12] where B11, B12 are 2x4
    # Split B12 into [B12a | B12b] where B12a, B12b are 2x2
    # This gets complicated. Let's just do a proper tiling with Strassen.
    
    # Better approach: Use the known fact that we can use Strassen recursively.
    # Pad A to 4x4 (add zero rows), pad B to 4x4 (split into two 4x4 blocks).
    # Compute 4x4 * 4x4 using Winograd rank-20, do it twice for the two 4x4 blocks of B.
    # But Winograd 4x4x4 is complex. Let's use a simpler approach.
    
    # Direct approach with Strassen 2x2x2:
    # A (2x4) = [A_top_left, A_top_right] no...
    # A is 2 rows, 4 cols. Split columns: A = [A1 A2] where A1, A2 are 2x2.
    # B is 4 rows, 8 cols. Split rows: B = [B1; B2] where B1, B2 are 2x8.
    # C = A1*B1 + A2*B2, each product is 2x8.
    
    # For 2x2 * 2x8:
    # Split the 2x8 into four 2x2 blocks: B = [B11 B12 B13 B14]
    # Result is 2x8 = [C11 C12 C13 C14] where Ci = A*Bii (2x2 * 2x2)
    # Each 2x2*2x2 uses Strassen rank 7.
    # So 2x2*2x8 uses 4*7 = 28 rank.
    # Total for 2x4*2x8: we need A1*B1 and A2*B2.
    # A1*B1: A1 is 2x2, B1 is 2x8 -> 28 rank
    # A2*B2: A2 is 2x2, B2 is 2x8 -> 28 rank
    # Total rank = 56.
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2x2 coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B01-B11)
    # M4 = (A11)(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    def add_strassen_block(u_offset, v_offset, w_offset, rank_offset):
        """Add a Strassen 2x2x2 block.
        u_offset: offset in U for the 2x2 A block (i*m+j within this block)
        v_offset: offset in V for the 2x2 B block (j*p+k within this block)  
        w_offset: offset in W for the 2x2 C block (k*n+i within this block)
        rank_offset: starting rank column
        """
        # Local indices: i,j in {0,1} for A (2x2), j,k in {0,1} for B (2x2), k,i in {0,1} for C (2x2)
        # U index: i*2+j
        # V index: j*2+k
        # W index: k*2+i
        
        # M1 = (A00+A11)(B00+B11)
        U[u_offset + 0, rank_offset + 0] = 1   # A00
        U[u_offset + 3, rank_offset + 0] = 1   # A11
        V[v_offset + 0, rank_offset + 0] = 1   # B00
        V[v_offset + 3, rank_offset + 0] = 1   # B11
        W[w_offset + 0, rank_offset + 0] = 1   # C00 (k=0,i=0)
        W[w_offset + 3, rank_offset + 0] = 1   # C11 (k=1,i=1)
        
        # M2 = (A10+A11)(B00)
        U[u_offset + 2, rank_offset + 1] = 1   # A10
        U[u_offset + 3, rank_offset + 1] = 1   # A11
        V[v_offset + 0, rank_offset + 1] = 1   # B00
        W[w_offset + 1, rank_offset + 1] = 1   # C10 (k=0,i=1)
        
        # M3 = (A00)(B01-B11)
        U[u_offset + 0, rank_offset + 2] = 1   # A00
        V[v_offset + 1, rank_offset + 2] = 1   # B01
        V[v_offset + 3, rank_offset + 2] = -1  # B11
        W[w_offset + 2, rank_offset + 2] = 1   # C01 (k=1,i=0)
        
        # M4 = (A11)(B10-B00)
        U[u_offset + 3, rank_offset + 3] = 1   # A11
        V[v_offset + 2, rank_offset + 3] = 1   # B10
        V[v_offset + 0, rank_offset + 3] = -1  # B00
        W[w_offset + 0, rank_offset + 3] = 1   # C00 (k=0,i=0)
        W[w_offset + 1, rank_offset + 3] = 1   # C10 (k=0,i=1)
        
        # M5 = (A00+A01)(B11)
        U[u_offset + 0, rank_offset + 4] = 1   # A00
        U[u_offset + 1, rank_offset + 4] = 1   # A01
        V[v_offset + 3, rank_offset + 4] = 1   # B11
        W[w_offset + 0, rank_offset + 4] = -1  # C00 (k=0,i=0)
        W[w_offset + 2, rank_offset + 4] = 1   # C01 (k=1,i=0)
        
        # M6 = (A10-A00)(B00+B01)
        U[u_offset + 2, rank_offset + 5] = 1   # A10
        U[u_offset + 0, rank_offset + 5] = -1  # A00
        V[v_offset + 0, rank_offset + 5] = 1   # B00
        V[v_offset + 1, rank_offset + 5] = 1   # B01
        # M6 contributes to C11? Let me check.
        # Actually in standard Strassen, M6 is used in C11 = M1 - M2 + M3 + M7
        # Wait, I think I have the formulas wrong. Let me use the standard ones.
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        # So M6 is not used in the output? That can't be right.
        # Actually, I recall now: the standard 7 multiplications don't use M6 in the final sum
        # because it's already accounted for. Let me verify...
        # Actually, I think I'm confusing myself. Let me just not use M6 if it's not needed.
        # Wait, checking again: the 7 products are M1-M7, and all are used.
        # C11 = M1 - M2 + M3 + M7... that's only 4 terms.
        # I must be missing something. Let me look this up mentally.
        # Actually, I believe the correct formulas are:
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        # And M6 is indeed not used in the standard formulation I recall.
        # This is suspicious. Let me just drop M6 and use rank 6 per block instead of 7.
        # But that would be incorrect. Let me just use the schoolbook for safety and accept rank 64.
        
        # Actually, let me just implement schoolbook properly to ensure correctness.
        pass
    
    # Given the complexity, let's just do schoolbook with rank 64 for correctness.
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
