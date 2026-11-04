import time
import numpy as np

DIMENSIONS = [4, 4, 5]
TIME_LIMIT = 150
SAFETY_MARGIN = 15

# EVOLVE-BLOCK-START
def construct():
    """Return a valid tensor decomposition for 4x4 by 4x5 matrix multiplication."""
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Try rank 53
    try:
        if time.time() - start_time < TIME_LIMIT - SAFETY_MARGIN:
            result = _build_445_rank53(n, m, p)
            if _verify_decomposition_fast(result, n, m, p):
                return result
    except Exception as e:
        # Fallback if error
        pass
    
    # Fallback: schoolbook decomposition (rank 80)
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _build_445_rank53(n, m, p):
    """Build rank-53 decomposition for 4x4x5."""
    rank_444 = 49
    new_rank = rank_444 + 4
    
    U = np.zeros((n * m, new_rank))
    V = np.zeros((m * p, new_rank))
    W = np.zeros((p * n, new_rank))
    
    # Strassen 2x2x2 factors
    # U2, V2, W2 are 4x7
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Indices: U2[i*2+j], V2[j*2+k], W2[k*2+i]
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2: (A10+A11)B00 -> C10
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1
    
    # M3: A00(B10-B00) -> C01, C11
    U2[0, 2] = 1
    V2[2, 2] = 1; V2[0, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # M4: A11(B01-B11) -> C00, C10
    U2[3, 3] = 1
    V2[1, 3] = 1; V2[3, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # M5: (A00+A01)B11 -> C01, C00(-)
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[1, 4] = 1; W2[0, 4] = -1
    
    # M6: (A01-A11)(B00+B01) -> C11
    U2[1, 5] = 1; U2[3, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7: (A00-A10)(B10+B11) -> C00
    U2[0, 6] = 1; U2[2, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Block Strassen 4x4x4
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]], C = [[C00, C01], [C10, C11]]
    # Each block is 2x2.
    
    # Coefficients for A blocks in each of the 7 steps
    # Step r: A_r = sum coeff * A_block
    # We represent A_r as a 2x2 matrix of coefficients for the 4 blocks?
    # No, the standard Strassen 4x4x4 uses 49 multiplications.
    # It is constructed by applying Strassen 2x2x2 to the "blocks" which are themselves 2x2 matrices.
    # But here we are doing scalar multiplications.
    # The 4x4x4 Strassen algorithm computes 49 scalar products.
    # The factors U, V, W are 16x49.
    
    # Let's define the 49 terms explicitly or generate them.
    # Generation:
    # The 4x4x4 Strassen decomposition can be seen as the Kronecker product of the 2x2x2 Strassen?
    # No, it's a specific set of 49 terms.
    
    # We can generate the 49 terms by iterating over the 7x7 combinations of the 2x2 Strassen steps?
    # Yes, the "block Strassen" approach:
    # We treat the 4x4 matrices as 2x2 blocks of 2x2 matrices.
    # We apply the 2x2 Strassen algorithm to these blocks.
    # This results in 7 block-multiplications.
    # Each block-multiplication is a 2x2 by 2x2 multiplication, which costs 7 scalar multiplications.
    # Total 49 scalar multiplications.
    
    # So we iterate r (0..6) for the outer Strassen step, and s (0..6) for the inner Strassen step.
    
    a_steps = [
        {(0,0): 1, (1,1): 1},       # M1: A00 + A11
        {(1,0): 1, (1,1): 1},       # M2: A10 + A11
        {(0,0): 1},                  # M3: A00
        {(1,1): 1},                  # M4: A11
        {(0,0): 1, (0,1): 1},       # M5: A00 + A01
        {(0,1): 1, (1,1): -1},      # M6: A01 - A11
        {(0,0): 1, (1,0): -1},      # M7: A00 - A10
    ]
    
    b_steps = [
        {(0,0): 1, (1,1): 1},       # M1: B00 + B11
        {(0,0): 1},                  # M2: B00
        {(1,0): 1, (0,0): -1},      # M3: B10 - B00
        {(0,1): 1, (1,1): -1},      # M4: B01 - B11
        {(1,1): 1},                  # M5: B11
        {(0,0): 1, (0,1): 1},       # M6: B00 + B01
        {(1,0): 1, (1,1): 1},       # M7: B10 + B11
    ]
    
    c_steps = [
        {(0,0): 1, (1,1): 1},       # M1: C00 + C11
        {(1,0): 1},                  # M2: C10
        {(0,1): 1, (1,1): 1},       # M3: C01 + C11
        {(0,0): 1, (1,0): 1},       # M4: C00 + C10
        {(0,1): 1, (0,0): -1},      # M5: C01 - C00
        {(1,1): 1},                  # M6: C11
        {(0,0): 1},                  # M7: C00
    ]
    
    for r in range(7):
        for s in range(7):
            col = r * 7 + s
            
            # U: A matrix, U[i*4+j]
            for (bi, bj), coeff_a in a_steps[r].items():
                if coeff_a == 0:
                    continue
                for li in range(2):
                    for lj in range(2):
                        local_u = U2[li*2 + lj, s]
                        if local_u == 0:
                            continue
                        global_i = bi * 2 + li
                        global_j = bj * 2 + lj
                        U[global_i * 4 + global_j, col] += coeff_a * local_u
            
            # V: B matrix, V[j*5+k]
            for (bj, bk), coeff_b in b_steps[r].items():
                if coeff_b == 0:
                    continue
                for lj in range(2):
                    for lk in range(2):
                        local_v = V2[lj*2 + lk, s]
                        if local_v == 0:
                            continue
                        global_j = bj * 2 + lj
                        global_k = bk * 2 + lk
                        V[global_j * 5 + global_k, col] += coeff_b * local_v
            
            # W: C matrix, W[k*4+i]
            for (ci, ck), coeff_c in c_steps[r].items():
                if coeff_c == 0:
                    continue
                for li in range(2):
                    for lk in range(2):
                        local_w = W2[lk*2 + li, s]
                        if local_w == 0:
                            continue
                        global_k = ck * 2 + lk
                        global_i = ci * 2 + li
                        W[global_k * 4 + global_i, col] += coeff_c * local_w
    
    # Add schoolbook for 5th column (k=4)
    for j in range(4):
        col = rank_444 + j
        for i in range(4):
            U[i * 4 + j, col] = 1
        V[j * 5 + 4, col] = 1
        for i in range(4):
            W[4 * 4 + i, col] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition_fast(decomp, n, m, p):
    U = np.array(decomp["U"], dtype=np.float64)
    V = np.array(decomp["V"], dtype=np.float64)
    W = np.array(decomp["W"], dtype=np.float64)
    
    # Check all valid (i,j,k) triples give 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a], V[b] * W[c])
                if abs(val - 1) > 1e-9:
                    return False
                    
    # Check all invalid positions give 0 (comprehensive check)
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i = a // m; j = a % m
                j2 = b // p; k = b % p
                k2 = c // n; i2 = c % n
                if i == i2 and j == j2 and k == k2:
                    continue
                val = np.dot(U[a], V[b] * W[c])
                if abs(val) > 1e-9:
                    return False
    
    # Check no zero columns (wasted rank)
    rank = U.shape[1]
    for r in range(rank):
        if np.all(U[:, r] == 0) or np.all(V[:, r] == 0) or np.all(W[:, r] == 0):
            return False
    
    return True
# EVOLVE-BLOCK-END
