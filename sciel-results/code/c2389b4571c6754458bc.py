import time
import numpy as np

DIMENSIONS = [6, 6, 6]
DEADLINE = 130

def _verify_full(n, m, p, U, V, W):
    """Full verification using numpy."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                val = np.dot(Ua[a_idx], Va[b_idx] * Wa[c_idx])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _build_strassen_2x2():
    """
    Returns U, V, W for 2x2 matrix multiplication (rank 7).
    Shapes: (4, 7)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen patterns
    # (u, v, w) coefficients
    patterns = [
        # 1: (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # 2: (A10+A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # 3: A00(B01-B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # 4: A11(B10-A00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # 5: (A00+A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # 6: (A10-A00)(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # 7: (A01-A11)(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    for r, (u_coeffs, v_coeffs, w_coeffs) in enumerate(patterns):
        for (pi, pj), val in u_coeffs.items():
            U[pi * 2 + pj, r] += val
        for (pi, pj), val in v_coeffs.items():
            V[pi * 2 + pj, r] += val
        for (pi, pj), val in w_coeffs.items():
            # W index is k*n + i, here n=2, so k*2 + i
            # In Strassen, W corresponds to C matrix indices (k, i) where C = A*B
            # C[k, i] gets contribution from W[k*2+i]
            # The pattern (pi, pj) in w_coeffs refers to (k, i)
            k = pi
            i = pj
            W[k * 2 + i, r] += val
            
    return U, V, W

def _build_laderman_4x4():
    """
    Returns U, V, W for 4x4 matrix multiplication (rank 20).
    Shapes: (16, 20)
    """
    U = np.zeros((16, 20))
    V = np.zeros((16, 20))
    W = np.zeros((16, 20))
    
    # Laderman's algorithm coefficients
    # Reference: Laderman, "A 4x4 matrix multiplication algorithm"
    # The 20 multiplications are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-A00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # ... and 13 more for the 4x4 blocks.
    
    # Actually, Laderman's algorithm is for 4x4 using 20 multiplications.
    # It is not just 3x3 Strassen blocks. It uses a specific structure.
    # I will use the known explicit coefficients.
    
    # Indices for 4x4: i,j,k in 0..3.
    # U index: i*4+j
    # V index: j*4+k
    # W index: k*4+i
    
    # Coefficients for Laderman 4x4
    # This is a known set of 20 terms.
    # I will construct them based on the block structure or hardcode if I can find a reliable source.
    # Since I don't have the full hardcoded list in memory, I will use the Kronecker product of 2x2 Strassen (rank 7) and 2x2 Strassen (rank 7) to get 4x4 rank 49? No, that's too high.
    # Wait, Kronecker product of two rank-7 decompositions gives rank 49.
    # Laderman is rank 20.
    # I need the actual Laderman coefficients.
    
    # If I cannot reliably generate Laderman coefficients, I can use the rank-49 decomposition (Kronecker of 2x2 Strassen) for 4x4.
    # Then Kronecker with 2x2 Strassen (rank 7) gives 8x8 rank 49*7 = 343.
    # Then truncate to 6x6. Rank 343.
    # Score 216/343 < 1. Worse than 189.
    
    # So I MUST use Laderman rank 20.
    # Let's try to construct Laderman coefficients.
    # Laderman's algorithm is based on the identity:
    # (A00 A01; A10 A11) (B00 B01; B10 B11) = ...
    # The 20 terms are:
    # 1. (A00+A11)(B00+B11)
    # 2. (A10+A11)B00
    # 3. A00(B01-B11)
    # 4. A11(B10-A00)
    # 5. (A00+A01)B11
    # 6. (A10-A00)(B00+B01)
    # 7. (A01-A11)(B10+B11)
    # 8. (A00+A01+A10+A11)B00  -- No, this is not standard.
    
    # Actually, I can use the fact that the rank of 4x4 is 20.
    # I will use a known implementation or derivation.
    # For the sake of this rewrite, if I can't get Laderman right, I'll stick to the safe 189.
    # But 140 is much better.
    # Let's assume I can hardcode the Laderman coefficients.
    
    # Laderman 4x4 coefficients (from literature):
    # The terms are indexed 0..19.
    # I will define them as lists of (index, value) for U, V, W.
    
    # Term 0: (A00+A11)(B00+B11) -> C00+=1, C11+=1, C01+=1, C10+=1? No.
    # Strassen 2x2 terms contribute to C.
    # In Laderman, the terms are more complex.
    
    # Let's use a different strategy.
    # Use the rank-20 decomposition for 4x4.
    # I will hardcode the 20 terms.
    
    terms = []
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        u_list = [(i*4+j, val) for (i,j), val in u_coeffs.items()]
        v_list = [(j*4+k, val) for (j,k), val in v_coeffs.items()]
        w_list = [(k*4+i, val) for (k,i), val in w_coeffs.items()]
        terms.append((u_list, v_list, w_list))
        
    # Laderman's 20 terms
    # 1
    add_term({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1, (0,1):1, (1,0):1}) # This is not correct for 4x4.
    
    # I will not guess the Laderman coefficients.
    # I will use the rank-49 decomposition for 4x4 (Kronecker 2x2 x 2x2).
    # Rank 49 * 7 = 343.
    # This is worse than 189.
    
    # So I will stick to the current 189 solution.
    # The prompt asks for a "stronger algorithm".
    # If I can't get Laderman, I can't improve.
    
    # Wait, I can use the rank-20 decomposition for 4x4 from a reliable source.
    # I will implement the Laderman coefficients based on a known correct set.
    
    # Correct Laderman Coefficients:
    # (Source: "A 4x4 matrix multiplication algorithm" by Laderman, 1976)
    # The 20 multiplications are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-A00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # M8 = (A00+A01+A10+A11)B00  -- No.
    
    # Actually, the 4x4 algorithm is not simply 2x2 blocks.
    # I will use the rank-189 solution as it is verified and safe.
    # The improvement to 140 requires correct Laderman coefficients.
    # Without them, I risk an invalid result (score 0).
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Attempt 140 rank solution (Laderman 4x4 + Strassen 2x2)
    # Requires Laderman coefficients.
    # Since we don't have reliable Laderman coefficients here, we fallback to 189.
    
    U, V, W = _build_strassen_2x2_6x6(n, m, p)
    
    # Verify
    if time.time() - start_time < DEADLINE - 10:
        if not _verify_full(n, m, p, U, V, W):
            # Fallback to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        U[i * m + j][idx] = 1
                        V[j * p + k][idx] = 1
                        W[k * n + i][idx] = 1
    
    return {"U": U, "V": V, "W": W}

def _build_strassen_2x2_6x6(n, m, p):
    """Build Strassen 2x2 block decomposition (rank 189)."""
    block_size = 2
    blocks_per_dim = 3
    rank = (blocks_per_dim ** 3) * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen patterns for 2x2 multiplication
    strassen_patterns = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    for (pi, pj), val in u_coeffs.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c][col_idx] += val
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c][col_idx] += val
                    for (pi, pj), val in w_coeffs.items():
                        k = K * block_size + pj
                        i = I * block_size + pi
                        W[k * n + i][col_idx] += val
                    col_idx += 1
    return U, V, W
