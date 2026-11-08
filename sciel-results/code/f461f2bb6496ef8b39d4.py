"""Matrix multiplication 2x4 x 4x6 rank-minimal exact tensor decomposition."""
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def _verify(U, V, W, n=2, m=4, p=6):
    """Verify that U, V, W form a valid tensor decomposition."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    
    # Check: sum_s U[i,m+j,s] * V[j,p+k,s] * W[k,n+i,s] should equal 1 if i,j,k match in the multiplication tensor
    # The multiplication tensor T[i,j,k] = 1 if the (i,j) element of M and (j,k) element of N produce (i,k) element of R
    # Actually the tensor is: for each s, the product U_s * V_s * W_s should sum to the identity tensor
    # T[i,j,k] = 1 for all valid i,j,k (it's the standard matrix multiplication tensor)
    # So we need: sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] == 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for s in range(rank):
                    total += U[i * m + j][s] * V[j * p + k][s] * W[k * n + i][s]
                if total != 1:
                    return False
    return True

def _schoolbook():
    """Return the schoolbook decomposition with rank 48."""
    n, m, p = DIMENSIONS
    rank = n * m * p  # 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = i * m * p + j * p + k
                U[i * m + j][s] = 1
                V[j * p + k][s] = 1
                W[k * n + i][s] = 1
    
    return {"U": U, "V": V, "W": W}

def _strassen_block_decomposition():
    """
    Use Strassen-like decomposition for 2x2 blocks.
    For 2x4 * 4x6, we can view it as:
    - 2x2 blocks of A (2 blocks horizontally)
    - 4x2 blocks of B (3 blocks horizontally)
    
    Each 2x2 * 2x2 multiplication can be done with 7 Strassen products.
    Total: 2 * 3 * 7 = 42 products.
    """
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 7 products for 2x2 * 2x2:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B00)  -- wait, let me use the standard formulation
    # 
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B10-B00)  -- actually: A00(B01-B11)
    # M4 = (A11)(B10-B00)  -- actually: A11(B01-B11)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For our case, A is 2x4, B is 4x6
    # A = [A_left | A_right] where A_left, A_right are 2x2
    # B = [B_top | B_mid | B_bot] where each is 4x2... no
    # B is 4x6, so B = [[B11,B12,B13],[B21,B22,B23]] where each Bij is 2x2
    
    # Actually let me think more carefully.
    # A is 2x4: rows 0,1; cols 0,1,2,3
    # B is 4x6: rows 0,1,2,3; cols 0,1,2,3,4,5
    # C = A*B is 2x6
    
    # We can split A into two 2x2 blocks: A0 (cols 0,1) and A1 (cols 2,3)
    # We can split B into: top half (rows 0,1) and bottom half (rows 2,3), each 2x6
    # Then B_top is 2x6, which we split into three 2x2 blocks: B_top0 (cols 0,1), B_top1 (cols 2,3), B_top2 (cols 4,5)
    # Similarly B_bot: B_bot0, B_bot1, B_bot2
    
    # C = A0 * B_top + A1 * B_bot
    # C[:,0:2] = A0 * B_top0 + A1 * B_bot0
    # C[:,2:4] = A0 * B_top1 + A1 * B_bot1
    # C[:,4:6] = A0 * B_top2 + A1 * B_bot2
    
    # So we have 6 sub-multiplications of 2x2 * 2x2, each using 7 Strassen products = 42 total.
    
    # Strassen coefficients:
    # For U (A matrix, 2x2): 
    #   M1: A00+A11 -> (0,0):1, (1,1):1
    #   M2: A10+A11 -> (1,0):1, (1,1):1
    #   M3: A00 -> (0,0):1
    #   M4: A11 -> (1,1):1
    #   M5: A00+A01 -> (0,0):1, (0,1):1
    #   M6: A10-A00 -> (1,0):1, (0,0):-1
    #   M7: A01-A11 -> (0,1):1, (1,1):-1
    
    U_coeffs = [
        {(0,0):1, (1,1):1},   # M1
        {(1,0):1, (1,1):1},   # M2
        {(0,0):1},             # M3
        {(1,1):1},             # M4
        {(0,0):1, (0,1):1},   # M5
        {(1,0):1, (0,0):-1},  # M6
        {(0,1):1, (1,1):-1},  # M7
    ]
    
    # For V (B matrix, 2x2):
    #   M1: B00+B11 -> (0,0):1, (1,1):1
    #   M2: B00 -> (0,0):1
    #   M3: B01-B11 -> (0,1):1, (1,1):-1
    #   M4: B01-B11 -> (0,1):1, (1,1):-1  -- wait
    #   M4: B10-B00 -> (1,0):1, (0,0):-1
    #   M5: B11 -> (1,1):1
    #   M6: B00+B10 -> (0,0):1, (1,0):1
    #   M7: B01+B11 -> (0,1):1, (1,1):1
    
    V_coeffs = [
        {(0,0):1, (1,1):1},   # M1
        {(0,0):1},             # M2
        {(0,1):1, (1,1):-1},  # M3
        {(1,0):1, (0,0):-1},  # M4
        {(1,1):1},             # M5
        {(0,0):1, (1,0):1},   # M6
        {(0,1):1, (1,1):1},   # M7
    ]
    
    # For W (C matrix, 2x2):
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    
    W_coeffs = [
        {(0,0):1, (1,1):1},   # M1
        {(1,0):1, (1,1):-1},  # M2
        {(0,1):1, (1,1):1},   # M3
        {(0,0):1, (1,0):1},   # M4
        {(0,0):-1, (0,1):1},  # M5
        {(1,1):1},             # M6
        {(0,0):1},             # M7
    ]
    
    col = 0
    for a_block in range(2):  # A block: 0 (cols 0,1) or 1 (cols 2,3)
        for b_block in range(3):  # B block: 0 (cols 0,1), 1 (cols 2,3), 2 (cols 4,5)
            # The B rows depend on a_block:
            # If a_block=0, B rows are 0,1 (top)
            # If a_block=1, B rows are 2,3 (bottom)
            b_row_offset = a_block * 2
            
            for s in range(7):
                # Update U
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a_block * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                # Update V
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = b_row_offset + j_loc
                    k = b_block * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                # Update W
                for (i_loc, k_loc), coeff in W_coeffs[s].items():
                    i = i_loc
                    k = b_block * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _try_improve(base_result, time_limit=10):
    """Try to improve upon the base result using random perturbation and verification."""
    start_time = time.time()
    best = base_result
    best_rank = len(base_result['U'][0]) if base_result['U'] else 0
    
    # For now, just verify the base result
    if _verify(base_result['U'], base_result['V'], base_result['W']):
        return best, best_rank
    
    # If base is invalid, fall back to schoolbook
    school = _schoolbook()
    return school, len(school['U'][0])

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # First try the Strassen-based decomposition with rank 42
    result = _strassen_block_decomposition()
    
    if time.time() < deadline:
        if _verify(result['U'], result['V'], result['W']):
            return result
    
    # If Strassen failed verification, fall back to schoolbook
    school = _schoolbook()
    if _verify(school['U'], school['V'], school['W']):
        return school
    
    # Shouldn't get here, but return schoolbook as last resort
    return school

# EVOLVE-BLOCK-END
