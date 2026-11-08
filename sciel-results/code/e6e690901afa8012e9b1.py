import numpy as np
import time

def _strassen_2x2_terms():
    """
    Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is a tuple of three dictionaries representing the sparse 
    linear combinations for A, B, and C.
    """
    terms = []
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1, (1,1): -1}
    })
    # M3 = A00(B10-B11) -> C01 += M3, C11 += M3
    # Note: Standard Strassen M3 = A00(B10 - B11) is not standard.
    # Standard:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11) -> This is often M3 = A00(B10 - B11) in some variants?
    # Let's check standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # 
    # Let's verify M3: A00(B10-B11).
    # C01 = A00B01 + A01B11.
    # C11 = A10B01 + A11B11.
    # 
    # Using the terms:
    # C00 = M1 - M4 - M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 - M3 + M6
    # 
    # Let's check M3 in C01: M3 + M5 = A00(B10-B11) + (A00+A01)B11 = A00B10 - A00B11 + A00B11 + A01B11 = A00B10 + A01B11.
    # Wait, C01 should be A00B01 + A01B11.
    # My M3 definition gives A00B10. That's wrong.
    # Standard Strassen M3 is usually A00(B10 - B11) is NOT correct for C01.
    # Let's look up the exact coefficients.
    
    # Correct Standard Strassen:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)  <-- Wait, if M3 contributes to C01, it must involve B01 or B11?
    # Actually, C01 = A00B01 + A01B11.
    # If M3 = A00(B10 - B11), it has B10.
    # Let's re-verify the standard 7 multiplications.
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 - M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 - M3 + M6
    
    # Check C01 = M3 + M5 = A00(B10-B11) + (A00+A01)B11 = A00B10 - A00B11 + A00B11 + A01B11 = A00B10 + A01B11.
    # But C01 = A00B01 + A01B11.
    # So M3 must be related to B01?
    # Ah, the indices in Strassen are often permuted.
    # Let's use the dictionary definitions from Inspiration 1 which were verified to work (score 1.306).
    # Inspiration 1 M3:
    # 'U': {(0,0): 1}, 'V': {(0,1): 1, (1,1): -1}, 'W': {(0,1): 1, (1,1): 1}
    # This corresponds to A00 * (B01 - B11).
    # Let's check C01 with this:
    # C01 contribution from M3: W(0,1)=1 * M3 = A00(B01-B11).
    # C11 contribution from M3: W(1,1)=1 * M3 = A00(B01-B11).
    # 
    # Let's check the full reconstruction with Inspiration 1 terms.
    # I will trust Inspiration 1's terms as they are from a scored solution.

    terms[2] = {
        'U': {(0,0): 1},
        'V': {(0,1): 1, (1,1): -1},
        'W': {(0,1): 1, (1,1): 1}
    }
    # M4 = A11(B10-B00) -> In Insp 1:
    # 'U': {(1,1): 1}, 'V': {(1,0): 1, (0,0): -1}, 'W': {(0,0): 1, (1,0): 1}
    # This is A11 * (B10 - B00).
    
    terms[3] = {
        'U': {(1,1): 1},
        'V': {(1,0): 1, (0,0): -1},
        'W': {(0,0): 1, (1,0): 1}
    }
    # M5 = (A00+A01)B11 -> In Insp 1:
    # 'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,0): -1, (0,1): 1}
    
    terms[4] = {
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,0): -1, (0,1): 1}
    }
    # M6 = (A10-A00)(B00+B01) -> In Insp 1:
    # 'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(1,1): 1}
    
    terms[5] = {
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(1,1): 1}
    }
    # M7 = (A01-A11)(B10+B11) -> In Insp 1:
    # 'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1}
    
    terms[6] = {
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1}
    }
    
    return terms

def _build_4x4x4_strassen():
    """Build rank-49 decomposition for 4x4x4 using recursive Strassen."""
    strassen_terms = _strassen_2x2_terms()
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    col = 0
    for meta in strassen_terms:
        for inner in strassen_terms:
            # U factor
            for (bi, bj), cu in meta['U'].items():
                for (ii, jj), cu_i in inner['U'].items():
                    row = bi * 2 + ii
                    c = bj * 2 + jj
                    U[row * 4 + c][col] += cu * cu_i
            # V factor
            for (bj, bk), cv in meta['V'].items():
                for (jj, kk), cv_i in inner['V'].items():
                    row = bj * 2 + jj
                    c = bk * 2 + kk
                    V[row * 4 + c][col] += cv * cv_i
            # W factor
            for (bi, bk), cw in meta['W'].items():
                for (ii, kk), cw_i in inner['W'].items():
                    i = bi * 2 + ii
                    k = bk * 2 + kk
                    W[k * 4 + i][col] += cw * cw_i
            col += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify decomposition correctness using random matrices."""
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    rng = np.random.RandomState(42)
    A = rng.randint(-5, 6, size=(n, m)).astype(float)
    B = rng.randint(-5, 6, size=(m, p)).astype(float)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for r in range(len(U[0])):
        u = U_mat[:, r].reshape(n, m)
        v = V_mat[:, r].reshape(m, p)
        w = W_mat[:, r].reshape(p, n)
        
        a_proj = np.sum(A * u)
        b_proj = np.sum(B * v)
        C_computed += a_proj * b_proj * w.T
    
    return np.allclose(C_computed, C_expected, atol=1e-9)

def construct():
    """Construct rank-minimal decomposition for 4x4x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 4, 4, 8
    
    # Build 4x4x4 Strassen decomposition
    U_444, V_444, W_444 = _build_4x4x4_strassen()
    
    # Combine for 4x4x8
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # First half (k=0..3)
    for r in range(49):
        for row in range(16):
            U[row][r] = U_444[row][r]
            j = row // 4
            k = row % 4
            V[j * 8 + k][r] = V_444[row][r]
            W[k * 4 + (row % 4)][r] = W_444[row][r] # W index k*n+i -> k*4+i. Row in W_444 is k*4+i.
            # Wait, W_444 row index is k*4+i.
            # In the loop `for row in range(16)`, row = k*4+i.
            # So W_444[row] is correct.
            # We need to place it in W[k*4+i].
            # So W[row][r] = W_444[row][r] is correct for the first half?
            # W is (p*n, rank) = (32, 98).
            # Indices 0..15 correspond to k=0..3.
            # So yes, W[row][r] = W_444[row][r].
            
    # Correcting the W assignment in the loop above:
    # The loop `for row in range(16)` iterates through all 16 indices of the 4x4 matrix.
    # For U: row is i*4+j. Correct.
    # For V: row is j*4+k. We map to V[j*8+k]. Correct.
    # For W: row is k*4+i. We map to W[k*4+i]. Since k<4, k*4+i < 16. Correct.
    
    # Let's re-run the assignment logic cleanly.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(49):
        for idx in range(16):
            # U: idx is i*4+j
            U[idx][r] = U_444[idx][r]
            
            # V: idx is j*4+k. Target is j*8+k
            j = idx // 4
            k = idx % 4
            V[j * 8 + k][r] = V_444[idx][r]
            
            # W: idx is k*4+i. Target is k*4+i (since k<4)
            W[idx][r] = W_444[idx][r]
            
            # Second half (k=4..7)
            r2 = r + 49
            U[idx][r2] = U_444[idx][r]
            V[j * 8 + (k + 4)][r2] = V_444[idx][r]
            W[(k + 4) * 4 + i][r2] = W_444[idx][r]
            # Need i for W second half.
            # idx = k*4+i -> i = idx % 4.
            i = idx % 4
            W[(k + 4) * 4 + i][r2] = W_444[idx][r]

    if time.time() < deadline - 10.0:
        if not _verify(U, V, W, n, m, p):
            # Fallback to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
    
    return {"U": U, "V": V, "W": W}
