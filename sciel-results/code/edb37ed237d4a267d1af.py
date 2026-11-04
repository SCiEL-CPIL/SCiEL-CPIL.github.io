import time
import numpy as np

DIMENSIONS = [4, 4, 5]
DEADLINE = 145.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    """Return a valid tensor decomposition for 4x4 by 4x5 matrix multiplication."""
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Construct Strassen 4x4x4 decomposition (rank 49)
    try:
        result_444 = _strassen_4x4x4()
        # Extend to 4x4x5 by adding schoolbook terms for the 5th column
        result = _extend_to_445(result_444, n, m, p)
        
        if _verify_decomposition(result, n, m, p):
            return result
    except Exception as e:
        # Fallback to schoolbook if construction fails
        pass
    
    # Fallback: schoolbook decomposition (rank 80)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """Return U, V, W (each 4x7) for Strassen 2x2x2 matrix multiplication."""
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    # Indices: A[i,j] -> U[i*2+j], B[j,k] -> V[j*2+k], C[i,k] -> W[k*2+i]
    # Note: W index is k*2+i.
    # C[0,0] -> W[0*2+0] = W[0]
    # C[0,1] -> W[1*2+0] = W[2]  <-- Wait.
    # Let's check the index convention in the original code.
    # Original code: W[k*n+i]. For n=2, W[k*2+i].
    # C[0,0] (i=0, k=0) -> W[0]
    # C[0,1] (i=0, k=1) -> W[2]
    # C[1,0] (i=1, k=0) -> W[1]
    # C[1,1] (i=1, k=1) -> W[3]
    
    # Let's re-map the standard Strassen formulas to this W indexing.
    # C[0,0] is W[0]
    # C[0,1] is W[2]
    # C[1,0] is W[1]
    # C[1,1] is W[3]
    
    # Formulas:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # P1 = (A00 + A11) * (B00 + B11)
    # U: A00 (0), A11 (3)
    # V: B00 (0), B11 (3)
    # W: C00 (0) +1, C11 (3) +1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (A10 + A11) * B00
    # U: A10 (2), A11 (3)
    # V: B00 (0)
    # W: C10 (1) +1, C11 (3) -1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = A00 * (B10 - B00)
    # U: A00 (0)
    # V: B10 (2), B00 (0) -> B10 is V[2], B00 is V[0]
    # W: C01 (2) +1, C11 (3) +1
    U[0][2] = 1
    V[2][2] = 1; V[0][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = A11 * (B01 - B11)
    # U: A11 (3)
    # V: B01 (1), B11 (3)
    # W: C00 (0) +1, C10 (1) +1
    U[3][3] = 1
    V[1][3] = 1; V[3][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (A00 + A01) * B11
    # U: A00 (0), A01 (1)
    # V: B11 (3)
    # W: C01 (2) +1, C00 (0) -1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[2][4] = 1; W[0][4] = -1
    
    # P6 = (A01 - A11) * (B00 + B01)
    # U: A01 (1), A11 (3)
    # V: B00 (0), B01 (1)
    # W: C11 (3) +1
    U[1][5] = 1; U[3][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (A00 - A10) * (B10 + B11)
    # U: A00 (0), A10 (2)
    # V: B10 (2), B11 (3)
    # W: C00 (0) +1
    U[0][6] = 1; U[2][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _strassen_4x4x4():
    """Construct rank 49 decomposition for 4x4x4."""
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    U2, V2, W2 = _strassen_2x2x2()
    
    # Outer Strassen steps definitions
    # (A_block_coeffs, B_block_coeffs, C_block_contributions)
    # A_block_coeffs: dict { (bi, bj): coeff }
    # C_block_contributions: dict { (ci, ck): coeff }
    
    steps = [
        # M1 = (A00 + A11) * (B00 + B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A10 + A11) * B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = A00 * (B10 - B00)
        ({(0,0): 1}, {(1,0): 1, (0,0): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = A11 * (B01 - B11)
        ({(1,1): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A00 + A01) * B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        # M6 = (A01 - A11) * (B00 + B01)
        ({(0,1): 1, (1,1): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (A00 - A10) * (B10 + B11)
        ({(0,0): 1, (1,0): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    for r, (a_coeffs, b_coeffs, c_coeffs) in enumerate(steps):
        for s in range(7):
            col = r * 7 + s
            
            # Process U (A)
            for bi in range(2):
                for bj in range(2):
                    coeff_a = a_coeffs.get((bi, bj), 0)
                    if coeff_a == 0: continue
                    for li in range(2):
                        for lj in range(2):
                            local_u = U2[li*2+lj][s]
                            if local_u == 0: continue
                            global_i = bi*2 + li
                            global_j = bj*2 + lj
                            U[global_i*4 + global_j][col] += coeff_a * local_u
            
            # Process V (B)
            for bj in range(2):
                for bk in range(2):
                    coeff_b = b_coeffs.get((bj, bk), 0)
                    if coeff_b == 0: continue
                    for lj in range(2):
                        for lk in range(2):
                            local_v = V2[lj*2+lk][s]
                            if local_v == 0: continue
                            global_j = bj*2 + lj
                            global_k = bk*2 + lk
                            V[global_j*4 + global_k][col] += coeff_b * local_v
            
            # Process W (C)
            for ci in range(2):
                for ck in range(2):
                    coeff_c = c_coeffs.get((ci, ck), 0)
                    if coeff_c == 0: continue
                    for li in range(2):
                        for lk in range(2):
                            # W2 index is k*2+i.
                            # local_w corresponds to C[li, lk] in the 2x2 block.
                            # So index in W2 is lk*2 + li.
                            local_w = W2[lk*2+li][s] 
                            if local_w == 0: continue
                            global_k = ck*2 + lk
                            global_i = ci*2 + li
                            # Global W index is k*4 + i
                            W[global_k*4 + global_i][col] += coeff_c * local_w
                            
    return {"U": U, "V": V, "W": W}

def _extend_to_445(decomp_444, n, m, p):
    """Extend 4x4x4 decomposition to 4x4x5."""
    U4 = decomp_444["U"]
    V4 = decomp_444["V"]
    W4 = decomp_444["W"]
    rank_444 = len(U4[0])
    
    # New dimensions
    new_rank = rank_444 + 4
    U = [[0] * new_rank for _ in range(n * m)]
    V = [[0] * new_rank for _ in range(m * p)]
    W = [[0] * new_rank for _ in range(p * n)]
    
    # Copy 4x4x4 part
    # U: 16 x 49 -> 16 x 53
    for i in range(16):
        for r in range(rank_444):
            U[i][r] = U4[i][r]
            
    # V: 16 x 49 -> 20 x 53
    # V4 is for B[0:4, 0:4].
    # V is for B[0:4, 0:5].
    for j in range(4):
        for k in range(4):
            for r in range(rank_444):
                V[j*5 + k][r] = V4[j*4 + k][r]
                
    # W: 16 x 49 -> 20 x 53
    # W4 is for C[0:4, 0:4].
    # W is for C[0:5, 0:4].
    for k in range(4):
        for i in range(4):
            for r in range(rank_444):
                W[k*4 + i][r] = W4[k*4 + i][r]
                
    # Add schoolbook for 5th column (k=4)
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    for j in range(4):
        col = rank_444 + j
        for i in range(4):
            U[i*4 + j][col] = 1
        V[j*5 + 4][col] = 1
        for i in range(4):
            W[4*4 + i][col] = 1
            
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(decomp, n, m, p, num_checks=100):
    """Verify the decomposition."""
    import random
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    rank = len(U[0]) if U else 0
    
    # Check all valid (i,j,k) triples give 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(val - 1) > 1e-9:
                    return False
    
    # Check some invalid positions give 0
    random.seed(42)
    for _ in range(num_checks):
        a = random.randint(0, n*m - 1)
        b = random.randint(0, m*p - 1)
        c = random.randint(0, p*n - 1)
        i = a // m; j = a % m
        j2 = b // p; k = b % p
        k2 = c // n; i2 = c % n
        if i == i2 and j == j2 and k == k2:
            continue
        val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
        if abs(val) > 1e-9:
            return False
    
    return True

def _try_optimized_extension(decomp_444, n, m, p):
    """Try to find a lower-rank extension by sharing terms."""
    # This is a placeholder for optimization attempts
    # For now, return the standard extension
    return _extend_to_445(decomp_444, n, m, p)
# EVOLVE-BLOCK-END
