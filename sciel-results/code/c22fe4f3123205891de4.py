import numpy as np
import time

def construct():
    deadline = time.time() + 120
    
    n, m, p = 2, 4, 5
    
    # Try to find rank-32 decomposition using 2+3 output split
    # First, try to find a rank-9 decomposition for 2x2x3 tensor
    
    def try_find_223_rank9():
        """Try to find rank-9 decomposition for 2x2x3 tensor."""
        # Tensor: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
        # A is 2x2 (indices 0-3), B is 2x3 (indices 0-5), C is 2x3 (indices 0-5)
        # A: a00, a01, a10, a11
        # B: b00, b01, b02, b10, b11, b12
        # C: c00, c01, c02, c10, c11, c12
        
        # We want U (4x9), V (6x9), W (6x9) such that
        # sum_r U[i,r]*V[j,r]*W[k,r] = tensor[i,j,k]
        
        # Use a parameterized search
        # Try structured forms
        
        # Strategy: Use 4 "direct" terms and 5 "Strassen-like" terms
        # Direct: a00*b00->c00, a01*b10->c00, a10*b00->c10, a11*b10->c10
        # Then handle cols 1,2 with shared computations
        
        # Let's try:
        # M1: a00*b00 -> c00
        # M2: a01*b10 -> c00
        # M3: a10*b00 -> c10
        # M4: a11*b10 -> c10
        # M5: (a00+a01)*(b01+b11) -> c01 (+), c11 needs handling
        # M6: a00*b11 -> c01 (-) to fix
        # M7: a01*b01 -> c01 (-) to fix
        # M8: (a10+a11)*(b01+b11) -> c11 (+)
        # M9: a10*b11 -> c11 (-) to fix
        # That's 9 but we still have col 2!
        
        # This approach doesn't work for all 3 columns with 9 terms.
        # Let me try a different structure.
        
        # Alternative: 
        # M1: a00*(b00+b01+b02) -> c00,c01,c02
        # M2: a01*(b10+b11+b12) -> c00,c01,c02
        # M3: a10*(b00+b01+b02) -> c10,c11,c12
        # M4: a11*(b10+b11+b12) -> c10,c11,c12
        # M5: a00*b01 -> c01 (-)
        # M6: a00*b02 -> c02 (-)
        # M7: a01*b11 -> c01 (-)
        # M8: a01*b12 -> c02 (-)
        # M9: a10*b01 -> c11 (-)
        # M10: a10*b02 -> c12 (-)
        # M11: a11*b11 -> c11 (-)
        # M12: a11*b12 -> c12 (-)
        # That's 12, back to schoolbook.
        
        # The issue is that 2x2x3 genuinely might need rank >= 10 or 11.
        # Known result: rank of 2x2x3 tensor is 9 (Bini et al.)
        # Let me try the Bini decomposition.
        
        # Actually, I recall that the minimal rank for 2x2x3 is indeed 9.
        # Let me try a specific construction.
        
        # Use the fact that 2x2x3 = 2x2x2 + 2x2x1, but with sharing.
        
        # Try this structure:
        # M1: (a00+a01) * (b00+b10) -> c00(+)
        # M2: a00 * (b00+b10) -> c00(-) [to get a01*b10 part]
        # Hmm, this is getting complicated.
        
        # Let me just try a numerical optimization approach with a time limit.
        t0 = time.time()
        
        # Parameterize U, V, W with some structure
        # Use 9 rank-1 terms
        
        # Actually, let me just try a few known constructions and verify.
        
        # Construction attempt 1:
        # Based on the idea that we can share the "row" computations
        
        U = np.zeros((4, 9))
        V = np.zeros((6, 9))
        W = np.zeros((6, 9))
        
        # A indices: 0=a00, 1=a01, 2=a10, 3=a11
        # B indices: 0=b00, 1=b01, 2=b02, 3=b10, 4=b11, 5=b12
        # C indices: 0=c00, 1=c01, 2=c02, 3=c10, 4=c11, 5=c12
        
        # Term 1: a00 * b00 -> c00
        U[0,0] = 1; V[0,0] = 1; W[0,0] = 1
        # Term 2: a01 * b10 -> c00
        U[1,1] = 1; V[3,1] = 1; W[0,1] = 1
        # Term 3: a10 * b00 -> c10
        U[2,2] = 1; V[0,2] = 1; W[3,2] = 1
        # Term 4: a11 * b10 -> c10
        U[3,3] = 1; V[3,3] = 1; W[3,3] = 1
        # Term 5: (a00+a01) * (b01+b11) -> c01(+)
        U[0,4] = 1; U[1,4] = 1; V[1,4] = 1; V[4,4] = 1; W[1,4] = 1
        # Term 6: a00 * b11 -> c01(-)
        U[0,5] = 1; V[4,5] = 1; W[1,5] = -1
        # Term 7: a01 * b01 -> c01(-)
        U[1,6] = 1; V[1,6] = 1; W[1,6] = -1
        # Term 8: (a10+a11) * (b01+b11) -> c11(+)
        U[2,7] = 1; U[3,7] = 1; V[1,7] = 1; V[4,7] = 1; W[4,7] = 1
        # Term 9: a10 * b11 -> c11(-)
        U[2,8] = 1; V[4,8] = 1; W[4,8] = -1
        # But we're missing col 2! This only gives cols 0 and 1.
        
        # This construction doesn't cover col 2. We need a different approach.
        
        # Let me try a fully general numerical search with a time limit.
        if time.time() - t0 > 30:
            return None
        
        # Use scipy optimization or random restarts
        try:
            from scipy.optimize import minimize
        except ImportError:
            return None
        
        # This is getting too complex for the time budget.
        # Let me just return the current rank-36 solution.
        return None
    
    # Fall back to rank-36 construction
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(a_row, a_col, b_row, b_col, c_row, c_col):
        nonlocal col
        idx_p00 = a_row * m + a_col
        idx_p01 = a_row * m + (a_col + 1)
        idx_p10 = (a_row + 1) * m + a_col
        idx_p11 = (a_row + 1) * m + (a_col + 1)
        
        idx_q00 = b_row * p + b_col
        idx_q01 = b_row * p + (b_col + 1)
        idx_q10 = (b_row + 1) * p + b_col
        idx_q11 = (b_row + 1) * p + (b_col + 1)
        
        idx_c00 = c_col * n + c_row
        idx_c01 = (c_col + 1) * n + c_row
        idx_c10 = c_col * n + (c_row + 1)
        idx_c11 = (c_col + 1) * n + (c_row + 1)
        
        U[idx_p00][col] += 1
        U[idx_p11][col] += 1
        V[idx_q00][col] += 1
        V[idx_q11][col] += 1
        W[idx_c00][col] += 1
        W[idx_c11][col] += 1
        col += 1
        
        U[idx_p10][col] += 1
        U[idx_p11][col] += 1
        V[idx_q00][col] += 1
        W[idx_c10][col] += 1
        W[idx_c11][col] -= 1
        col += 1
        
        U[idx_p00][col] += 1
        V[idx_q01][col] += 1
        V[idx_q11][col] -= 1
        W[idx_c01][col] += 1
        W[idx_c11][col] += 1
        col += 1
        
        U[idx_p11][col] += 1
        V[idx_q10][col] += 1
        V[idx_q00][col] -= 1
        W[idx_c00][col] += 1
        W[idx_c10][col] += 1
        col += 1
        
        U[idx_p00][col] += 1
        U[idx_p01][col] += 1
        V[idx_q11][col] += 1
        W[idx_c00][col] -= 1
        W[idx_c01][col] += 1
        col += 1
        
        U[idx_p10][col] += 1
        U[idx_p00][col] -= 1
        V[idx_q00][col] += 1
        V[idx_q01][col] += 1
        W[idx_c11][col] += 1
        col += 1
        
        U[idx_p01][col] += 1
        U[idx_p11][col] -= 1
        V[idx_q10][col] += 1
        V[idx_q11][col] += 1
        W[idx_c00][col] += 1
        col += 1

    def add_schoolbook_221(a_row, a_col, b_row, b_col, c_row, c_col):
        nonlocal col
        idx_p00 = a_row * m + a_col
        idx_p01 = a_row * m + (a_col + 1)
        idx_p10 = (a_row + 1) * m + a_col
        idx_p11 = (a_row + 1) * m + (a_col + 1)
        
        idx_q0 = b_row * p + b_col
        idx_q1 = (b_row + 1) * p + b_col
        
        idx_c0 = c_col * n + c_row
        idx_c1 = c_col * n + (c_row + 1)
        
        U[idx_p00][col] += 1
        V[idx_q0][col] += 1
        W[idx_c0][col] += 1
        col += 1
        
        U[idx_p01][col] += 1
        V[idx_q1][col] += 1
        W[idx_c0][col] += 1
        col += 1
        
        U[idx_p10][col] += 1
        V[idx_q0][col] += 1
        W[idx_c1][col] += 1
        col += 1
        
        U[idx_p11][col] += 1
        V[idx_q1][col] += 1
        W[idx_c1][col] += 1
        col += 1

    add_strassen(0, 0, 0, 0, 0, 0)
    add_strassen(0, 2, 2, 0, 0, 0)
    add_strassen(0, 0, 0, 2, 0, 2)
    add_strassen(0, 2, 2, 2, 0, 2)
    add_schoolbook_221(0, 0, 0, 4, 0, 4)
    add_schoolbook_221(0, 2, 2, 4, 0, 4)
    
    return {"U": U, "V": V, "W": W}
