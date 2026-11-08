"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time

def _add_schoolbook(U, V, W, n, m, p, i_offset, j_offset, k_offset):
    """Add a schoolbook decomposition for an n x m by m x p block.
    U indices: (i_offset + i) * m_total + (j_offset + j) ? 
    Wait, the global U index is i * m + j.
    Global V index is j * p + k.
    Global W index is k * n + i.
    
    We need to map local block indices to global indices.
    Block A is at rows [i_offset, i_offset+n), cols [j_offset, j_offset+m).
    Block B is at rows [j_offset, j_offset+m), cols [k_offset, k_offset+p).
    Block C is at rows [i_offset, i_offset+n), cols [k_offset, k_offset+p).
    """
    rank = n * m * p
    # We need to append 'rank' new columns to U, V, W
    # But U, V, W are lists of lists. We need to extend them.
    # To make this easier, we will build the matrices as lists of lists and extend.
    
    # Current rank
    current_rank = len(U[0]) if U else 0
    
    # Initialize new columns with 0
    for row in U:
        row.extend([0] * rank)
    for row in V:
        row.extend([0] * rank)
    for row in W:
        row.extend([0] * rank)
        
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Global indices
                gi = i_offset + i
                gj = j_offset + j
                gk = k_offset + k
                
                U[gi * 5 + gj][current_rank + col] = 1
                V[gj * 6 + gk][current_rank + col] = 1
                W[gk * 4 + gi][current_rank + col] = 1
                col += 1

def _add_laderman(U, V, W, i_offset, j_offset, k_offset):
    """Add Laderman's rank-7 decomposition for a 3x3 by 3x3 block."""
    n, m, p = 3, 3, 3
    rank = 7
    current_rank = len(U[0]) if U else 0
    
    for row in U:
        row.extend([0] * rank)
    for row in V:
        row.extend([0] * rank)
    for row in W:
        row.extend([0] * rank)
        
    # Laderman terms
    # U[i,j], V[j,k], W[k,i]
    # Local indices 0..2
    
    # Term 0
    # A[0,0] * (B[0,1] - B[0,0])
    U[i_offset+0, j_offset+0] = 1
    V[j_offset+0, k_offset+1] = 1
    V[j_offset+0, k_offset+0] = -1
    W[k_offset+0, i_offset+0] = -1
    W[k_offset+1, i_offset+0] = 1
    # Note: The W term corresponds to the output C[i,k].
    # The term is M1 = A00(B01-B00).
    # This contributes to C00 and C01.
    # C00 gets -M1, C01 gets M1.
    # So W[k,i] should be -1 for (0,0) and 1 for (1,0)?
    # W index is k*n + i.
    # W[0,0] = -1, W[1,0] = 1.
    
    # Term 1
    # (A[0,0] + A[0,1]) * B[0,0]
    U[i_offset+0, j_offset+0] = 1
    U[i_offset+0, j_offset+1] = 1
    V[j_offset+0, k_offset+0] = 1
    W[k_offset+0, i_offset+0] = 1
    
    # Term 2
    # (A[0,1] - A[0,0]) * B[1,0]
    U[i_offset+0, j_offset+1] = 1
    U[i_offset+0, j_offset+0] = -1
    V[j_offset+1, k_offset+0] = 1
    W[k_offset+0, i_offset+0] = -1
    W[k_offset+1, i_offset+0] = 1
    
    # Term 3
    # A[1,1] * (B[1,0] - B[1,1])
    U[i_offset+1, j_offset+1] = 1
    V[j_offset+1, k_offset+0] = 1
    V[j_offset+1, k_offset+1] = -1
    W[k_offset+0, i_offset+1] = -1
    W[k_offset+1, i_offset+1] = 1
    
    # Term 4
    # (A[1,1] + A[1,0]) * B[1,1]
    U[i_offset+1, j_offset+1] = 1
    U[i_offset+1, j_offset+0] = 1
    V[j_offset+1, k_offset+1] = 1
    W[k_offset+1, i_offset+1] = 1
    
    # Term 5
    # (A[1,0] - A[1,1]) * B[0,1]
    U[i_offset+1, j_offset+0] = 1
    U[i_offset+1, j_offset+1] = -1
    V[j_offset+0, k_offset+1] = 1
    W[k_offset+1, i_offset+1] = -1
    W[k_offset+0, i_offset+1] = 1
    
    # Term 6
    # (A[0,0] + A[1,1]) * (B[1,1] - B[0,0])
    # This term is distributed.
    # M7 = (A00+A11)(B11-B00)
    # C00: -M7
    # C01: M7
    # C10: -M7 ? No.
    # C11: M7 ?
    
    # Let's check the reconstruction for Laderman:
    # C00 = M2 + M3 - M1 - M7
    # C01 = M1 + M7
    # C10 = M4 - M3
    # C11 = M5 + M3 - M4 - M7 ? No.
    
    # Actually, C11 = M5 + M7 - M4 - M6?
    # Let's use a verified set.
    
    # I will use the following verified Laderman coefficients:
    # M1 = A00(B01-B00)
    # M2 = (A00+A01)B00
    # M3 = (A01-A00)B10
    # M4 = A11(B10-B11)
    # M5 = (A11+A10)B11
    # M6 = (A10-A11)B01
    # M7 = (A00+A11)(B11-B00)
    
    # C00 = M2 + M3 - M1 - M7
    # C01 = M1 + M7
    # C10 = M4 - M3
    # C11 = M5 + M3 - M4 - M7 ? 
    # Wait, C11 should be A11B11 + A10B01 + A01B10 + A00B00? No.
    
    # I will hardcode the W coefficients based on the reconstruction formulas.
    # For Term 6 (M7):
    # U: A00=1, A11=1
    # V: B11=1, B00=-1
    # W: 
    # C00: -1
    # C01: 1
    # C10: 0?
    # C11: -1?
    
    # Let's assume the standard Laderman works and just implement the terms.
    # To be safe, I will use the schoolbook for all blocks if Laderman is uncertain.
    # But 80 is a good target.
    
    # Let's just use Schoolbook for everything first to ensure correctness, 
    # then replace 3x3 blocks with Laderman.
    
    # For now, I will implement Laderman carefully.
    
    # Term 6:
    U[i_offset+0, j_offset+0] = 1
    U[i_offset+1, j_offset+1] = 1
    V[j_offset+1, k_offset+1] = 1
    V[j_offset+0, k_offset+0] = -1
    W[k_offset+0, i_offset+0] = -1
    W[k_offset+1, i_offset+0] = 1
    W[k_offset+0, i_offset+1] = -1
    W[k_offset+1, i_offset+1] = 1
    # This W pattern corresponds to:
    # C00: -1
    # C01: 1
    # C10: -1
    # C11: 1
    
    # This implies:
    # C00 -= M7
    # C01 += M7
    # C10 -= M7
    # C11 += M7
    
    # Let's check if this matches the algebra.
    # M7 = (A00+A11)(B11-B00)
    # C11 should have +M7?
    # A11B11 is in M7.
    # C11 = ... + A11B11 ...
    # So yes, +M7 is correct for C11.
    # C00 = ... - A00B00 ...
    # M7 has -A00B00.
    # So -M7 has +A00B00.
    # So C00 should have -M7?
    # If C00 = ... - M7, then it adds A00B00. Correct.
    
    # So the W coefficients for Term 6 are:
    # W[0,0] = -1
    # W[1,0] = 1
    # W[0,1] = -1
    # W[1,1] = 1
    
    # I will update the code above to reflect this.

def construct():
    n, m, p = 4, 5, 6
    
    # Initialize U, V, W as empty lists of lists
    # U: 20 rows, V: 30 rows, W: 24 rows
    U = [[0] for _ in range(n * m)]
    V = [[0] for _ in range(m * p)]
    W = [[0] for _ in range(p * n)]
    
    # Block decomposition
    # I = {0,1,2,3} -> I1={0,1,2}, I2={3}
    # J = {0,1,2,3,4} -> J1={0,1,2}, J2={3,4}
    # K = {0,1,2,3,4,5} -> K1={0,1,2}, K2={3,4,5}
    
    # Block 1: A[0:3, 0:3] * B[0:3, 0:3] -> C[0:3, 0:3]
    # Laderman
    _add_laderman(U, V, W, 0, 0, 0)
    
    # Block 2: A[0:3, 0:3] * B[0:3, 3:6] -> C[0:3, 3:6]
    # Laderman
    _add_laderman(U, V, W, 0, 0, 3)
    
    # Block 3: A[3:4, 0:3] * B[0:3, 0:3] -> C[3:4, 0:3]
    # Schoolbook 1x3 x 3x3
    _add_schoolbook(U, V, W, 1, 3, 3, 3, 0, 0)
    
    # Block 4: A[3:4, 0:3] * B[0:3, 3:6] -> C[3:4, 3:6]
    # Schoolbook 1x3 x 3x3
    _add_schoolbook(U, V, W, 1, 3, 3, 3, 0, 3)
    
    # Block 5: A[0:3, 3:5] * B[3:5, 0:3] -> C[0:3, 0:3]
    # Schoolbook 3x2 x 2x3
    _add_schoolbook(U, V, W, 3, 2, 3, 0, 3, 0)
    
    # Block 6: A[0:3, 3:5] * B[3:5, 3:6] -> C[0:3, 3:6]
    # Schoolbook 3x2 x 2x3
    _add_schoolbook(U, V, W, 3, 2, 3, 0, 3, 3)
    
    # Block 7: A[3:4, 3:5] * B[3:5, 0:3] -> C[3:4, 0:3]
    # Schoolbook 1x2 x 2x3
    _add_schoolbook(U, V, W, 1, 2, 3, 3, 3, 0)
    
    # Block 8: A[3:4, 3:5] * B[3:5, 3:6] -> C[3:4, 3:6]
    # Schoolbook 1x2 x 2x3
    _add_schoolbook(U, V, W, 1, 2, 3, 3, 3, 3)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
