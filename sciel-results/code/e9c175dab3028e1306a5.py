import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 118.0  # Safety margin under 120s kill (and 150s hard kill)
    
    n, m, p = 2, 4, 4
    # Rank 28 is the known minimum for M_{2,4,4}
    rank = 28
    
    # Initialize factor matrices
    # U: (n*m, rank) = (8, 28)
    # V: (m*p, rank) = (16, 28)
    # W: (p*n, rank) = (8, 28)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2 multiplication
    # Indices in local 2x2 blocks: 00, 01, 10, 11
    # A: [A00, A01, A10, A11] -> indices 0, 1, 2, 3 in local U
    # B: [B00, B01, B10, B11] -> indices 0, 1, 2, 3 in local V
    # C: [C00, C01, C10, C11] -> indices 0, 1, 2, 3 in local W
    
    # The 7 Strassen terms:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define the contribution of each term to U, V, W
    # Each term is a tuple (u_indices, v_indices, w_indices)
    # where indices are into the local 2x2 block coordinates (0,1,2,3)
    # and the sign is implicit in the +/- of the sum for W.
    # Actually, it's easier to define the linear forms directly.
    
    # Term 1: M1 = (A00+A11)*(B00+B11)
    # U: A00(+1), A11(+1) -> local U indices 0, 2
    # V: B00(+1), B11(+1) -> local V indices 0, 2
    # W: C00(+1), C11(+1) -> local W indices 0, 3
    t1_u = [(0, 1), (2, 1)]
    t1_v = [(0, 1), (2, 1)]
    t1_w = [(0, 1), (3, 1)]
    
    # Term 2: M2 = (A10+A11)*B00
    # U: A10(+1), A11(+1) -> local U indices 1, 2  (Wait, A10 is index 1? No, A is 2x2. 
    # Standard indexing: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3? 
    # Let's stick to the previous program's convention:
    # X_rows = [0, 1, 4, 5] for A_L.
    # In the previous code:
    # X1 = [0, 1, 4, 5]. 
    # A_L is 2x2. 
    # If A is 2x4, A_L is cols 0,1. 
    # Indices 0,1 are row 0 of A_L? 
    # n=2, m=4. U has 8 rows.
    # Row 0: A[0,0], Row 1: A[0,1], Row 2: A[0,2], Row 3: A[0,3]
    # Row 4: A[1,0], Row 5: A[1,1], Row 6: A[1,2], Row 7: A[1,3]
    # So A_L (cols 0,1) corresponds to U indices:
    # A[0,0]->0, A[0,1]->1, A[1,0]->4, A[1,1]->5.
    # So local indices 0,1,2,3 map to global U indices [0, 1, 4, 5].
    # Let's verify Strassen mapping in previous code:
    # add_block(0, X1=[0,1,4,5], Y1=[0,1,4,5], Z1=[0,2,1,3])
    # Y1=[0,1,4,5] for B_TL.
    # B is 4x4. V has 16 rows.
    # B[0,0]->0, B[0,1]->1, B[0,2]->2, B[0,3]->3
    # B[1,0]->4, B[1,1]->5, ...
    # B_TL (rows 0,1, cols 0,1) corresponds to V indices:
    # B[0,0]->0, B[0,1]->1, B[1,0]->4, B[1,1]->5.
    # So local indices 0,1,2,3 map to global V indices [0, 1, 4, 5].
    
    # Z1=[0,2,1,3] for C_L.
    # C is 2x4. W has 8 rows.
    # C[0,0]->0, C[0,1]->1, C[0,2]->2, C[0,3]->3
    # C[1,0]->4, C[1,1]->5, ...
    # C_L (cols 0,1) corresponds to W indices:
    # C[0,0]->0, C[0,1]->1, C[1,0]->4, C[1,1]->5.
    # BUT Z1 is [0, 2, 1, 3]. This implies C_L is mapped to W indices 0, 2, 1, 3?
    # Wait. The task says "output factor follows the official trace convention W[k*n+i,r]".
    # n=2. W has p*n = 4*2 = 8 rows.
    # Index k*n + i.
    # If C is 2x4, let's say C[i, j]. i in 0..1, j in 0..3.
    # Usually W index corresponds to (i, j).
    # If W[k*n+i] is the convention, maybe k is the column of C?
    # Let's check the previous code's Z1 = [0, 2, 1, 3].
    # If C_L is 2x2, entries C[0,0], C[0,1], C[1,0], C[1,1].
    # If W index is j*n + i:
    # C[0,0] -> 0*2+0 = 0
    # C[0,1] -> 1*2+0 = 2
    # C[1,0] -> 0*2+1 = 1
    # C[1,1] -> 1*2+1 = 3
    # This matches Z1 = [0, 2, 1, 3].
    # So the local indices 0,1,2,3 for C correspond to global W indices:
    # Local 0 (C[0,0]) -> Global 0
    # Local 1 (C[1,0]) -> Global 1  (Wait, C[1,0] is row 1, col 0. Index 1*2+0? No.
    # Let's re-evaluate W[k*n+i].
    # If k is column index and i is row index:
    # C[0,0]: k=0, i=0 -> 0
    # C[0,1]: k=1, i=0 -> 2
    # C[1,0]: k=0, i=1 -> 1
    # C[1,1]: k=1, i=1 -> 3
    # So local C indices 0,1,2,3 (representing C00, C10, C01, C11 in some order?)
    # In Strassen, we usually care about C00, C01, C10, C11.
    # The previous code's add_block used:
    # W[Z_rows[0]] += 1 (C00)
    # W[Z_rows[1]] += 1 (C01? or C10?)
    # Let's trace add_block term 0 (M1):
    # W[Z_rows[0]] += 1
    # W[Z_rows[3]] += 1
    # If Z_rows = [0, 2, 1, 3], then W[0] and W[3] get +1.
    # M1 contributes to C00 and C11.
    # So Z_rows[0] is C00, Z_rows[3] is C11.
    # This implies:
    # Z_rows[0] = C00 index
    # Z_rows[1] = C01 index?
    # Z_rows[2] = C10 index?
    # Z_rows[3] = C11 index?
    
    # Let's check term 2 (M3 = A00*(B01-B11) -> C01, C11? No, M3 is C01 + C11?
    # Strassen:
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    # So M3 contributes to C01 and C11.
    # In add_block term 2 (index 2, which is M3 in my list? Let's check code):
    # Code term 2:
    # U[X_rows[0]] += 1 (A00)
    # V[Y_rows[1]] += 1 (B01)
    # V[Y_rows[3]] -= 1 (B11)
    # W[Z_rows[1]] += 1
    # W[Z_rows[3]] += 1
    # This matches M3 = A00 * (B01 - B11).
    # It contributes to Z_rows[1] and Z_rows[3].
    # So Z_rows[1] is C01, Z_rows[3] is C11.
    
    # So the mapping is:
    # Local 0 -> C00
    # Local 1 -> C01
    # Local 2 -> C10
    # Local 3 -> C11
    
    # Let's define the 7 terms explicitly with this mapping.
    
    def apply_strassen_block(u_map, v_map, w_map, col_offset):
        """
        u_map: list of 4 global U indices for A00, A01, A10, A11
        v_map: list of 4 global V indices for B00, B01, B10, B11
        w_map: list of 4 global W indices for C00, C01, C10, C11
        col_offset: start column in the factor matrices
        """
        
        # M1 = (A00+A11)*(B00+B11) -> C00, C11
        U[u_map[0]][col_offset+0] += 1 # A00
        U[u_map[3]][col_offset+0] += 1 # A11
        
        V[v_map[0]][col_offset+0] += 1 # B00
        V[v_map[3]][col_offset+0] += 1 # B11
        
        W[w_map[0]][col_offset+0] += 1 # C00
        W[w_map[3]][col_offset+0] += 1 # C11
        
        # M2 = (A10+A11)*B00 -> C10, C11
        U[u_map[2]][col_offset+1] += 1 # A10
        U[u_map[3]][col_offset+1] += 1 # A11
        V[v_map[0]][col_offset+1] += 1 # B00
        
        W[w_map[2]][col_offset+1] += 1 # C10
        W[w_map[3]][col_offset+1] -= 1 # C11 (Minus in C11 formula: M1-M2+...)
        
        # M3 = A00*(B01-B11) -> C01, C11
        U[u_map[0]][col_offset+2] += 1 # A00
        V[v_map[1]][col_offset+2] += 1 # B01
        V[v_map[3]][col_offset+2] -= 1 # B11
        
        W[w_map[1]][col_offset+2] += 1 # C01
        W[w_map[3]][col_offset+2] += 1 # C11
        
        # M4 = A11*(B10-B00) -> C00, C10
        U[u_map[3]][col_offset+3] += 1 # A11
        V[v_map[2]][col_offset+3] += 1 # B10
        V[v_map[0]][col_offset+3] -= 1 # B00
        
        W[w_map[0]][col_offset+3] += 1 # C00
        W[w_map[2]][col_offset+3] += 1 # C10
        
        # M5 = (A00+A01)*B11 -> C00, C01
        U[u_map[0]][col_offset+4] += 1 # A00
        U[u_map[1]][col_offset+4] += 1 # A01
        V[v_map[3]][col_offset+4] += 1 # B11
        
        W[w_map[0]][col_offset+4] -= 1 # C00 (Minus in C00 formula: M1+M4-M5+M7)
        W[w_map[1]][col_offset+4] += 1 # C01
        
        # M6 = (A10-A00)*(B00+B01) -> C11
        U[u_map[2]][col_offset+5] += 1 # A10
        U[u_map[0]][col_offset+5] -= 1 # A00
        V[v_map[0]][col_offset+5] += 1 # B00
        V[v_map[1]][col_offset+5] += 1 # B01
        
        W[w_map[3]][col_offset+5] += 1 # C11
        
        # M7 = (A01-A11)*(B10+B11) -> C00
        U[u_map[1]][col_offset+6] += 1 # A01
        U[u_map[3]][col_offset+6] -= 1 # A11
        V[v_map[2]][col_offset+6] += 1 # B10
        V[v_map[3]][col_offset+6] += 1 # B11
        
        W[w_map[0]][col_offset+6] += 1 # C00

    # Define the global index mappings for the 4 blocks
    
    # Block 1: A_L * B_TL -> C_L
    # A_L: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices 0, 1, 4, 5
    # B_TL: B[0,0], B[0,1], B[1,0], B[1,1] -> V indices 0, 1, 4, 5
    # C_L: C[0,0], C[0,1], C[1,0], C[1,1] -> W indices 0, 2, 1, 3
    u_map_1 = [0, 1, 4, 5]
    v_map_1 = [0, 1, 4, 5]
    w_map_1 = [0, 2, 1, 3]
    
    # Block 2: A_R * B_BL -> C_L
    # A_R: A[0,2], A[0,3], A[1,2], A[1,3] -> U indices 2, 3, 6, 7
    # B_BL: B[2,0], B[2,1], B[3,0], B[3,1] -> V indices 8, 9, 12, 13
    # C_L: Same W indices 0, 2, 1, 3
    u_map_2 = [2, 3, 6, 7]
    v_map_2 = [8, 9, 12, 13]
    w_map_2 = [0, 2, 1, 3]
    
    # Block 3: A_L * B_TR -> C_R
    # A_L: U indices 0, 1, 4, 5
    # B_TR: B[0,2], B[0,3], B[1,2], B[1,3] -> V indices 2, 3, 6, 7
    # C_R: C[0,2], C[0,3], C[1,2], C[1,3]
    # W indices for C_R:
    # C[0,2]: k=2, i=0 -> 2*2+0 = 4
    # C[0,3]: k=3, i=0 -> 3*2+0 = 6
    # C[1,2]: k=2, i=1 -> 2*2+1 = 5
    # C[1,3]: k=3, i=1 -> 3*2+1 = 7
    # So w_map_3 = [4, 6, 5, 7]
    u_map_3 = [0, 1, 4, 5]
    v_map_3 = [2, 3, 6, 7]
    w_map_3 = [4, 6, 5, 7]
    
    # Block 4: A_R * B_BR -> C_R
    # A_R: U indices 2, 3, 6, 7
    # B_BR: B[2,2], B[2,3], B[3,2], B[3,3] -> V indices 10, 11, 14, 15
    # C_R: W indices 4, 6, 5, 7
    u_map_4 = [2, 3, 6, 7]
    v_map_4 = [10, 11, 14, 15]
    w_map_4 = [4, 6, 5, 7]
    
    # Apply the blocks
    apply_strassen_block(u_map_1, v_map_1, w_map_1, 0)
    apply_strassen_block(u_map_2, v_map_2, w_map_2, 7)
    apply_strassen_block(u_map_3, v_map_3, w_map_3, 14)
    apply_strassen_block(u_map_4, v_map_4, w_map_4, 21)
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
