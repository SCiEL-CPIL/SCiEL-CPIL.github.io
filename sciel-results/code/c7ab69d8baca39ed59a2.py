"""Matrix multiplication 4x4 x 4x8 with rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is a rank-1 tensor contribution.
    """
    terms = []
    # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
    terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    # M2 = (A10+A11)*B00 -> C10 += M2, C11 -= M2
    terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1, (1,1): -1}
    })
    # M3 = A00*(B01-B11) -> C01 += M3, C11 += M3
    terms.append({
        'U': {(0,0): 1},
        'V': {(0,1): 1, (1,1): -1},
        'W': {(0,1): 1, (1,1): 1}
    })
    # M4 = A11*(B10-B00) -> C00 += M4, C10 += M4
    terms.append({
        'U': {(1,1): 1},
        'V': {(1,0): 1, (0,0): -1},
        'W': {(0,0): 1, (1,0): 1}
    })
    # M5 = (A00+A01)*B11 -> C00 -= M5, C01 += M5
    terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,0): -1, (0,1): 1}
    })
    # M6 = (A10-A00)*(B00+B01) -> C11 += M6
    terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(1,1): 1}
    })
    # M7 = (A01-A11)*(B10+B11) -> C00 += M7
    terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1}
    })
    return terms

def _build_4x4x4_strassen():
    """Build rank-49 decomposition for 4x4x4 using recursive Strassen.
    Returns U (16,49), V (16,49), W (16,49) with integer entries.
    Indexing: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    strassen_terms = _strassen_2x2_terms()
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    col = 0
    for meta in strassen_terms:
        for inner in strassen_terms:
            # U factor: A[bi*2+ii, bj*2+jj]
            for (bi, bj), cu in meta['U'].items():
                for (ii, jj), cu_i in inner['U'].items():
                    row = bi * 2 + ii
                    c = bj * 2 + jj
                    U[row * 4 + c][col] += cu * cu_i
            # V factor: B[bj*2+jj, bk*2+kk]
            for (bj, bk), cv in meta['V'].items():
                for (jj, kk), cv_i in inner['V'].items():
                    row = bj * 2 + jj
                    c = bk * 2 + kk
                    V[row * 4 + c][col] += cv * cv_i
            # W factor: C[bi*2+ii, bk*2+kk] -> W[k*4+i]
            for (bi, bk), cw in meta['W'].items():
                for (ii, kk), cw_i in inner['W'].items():
                    i = bi * 2 + ii
                    k = bk * 2 + kk
                    W[k * 4 + i][col] += cw * cw_i
            col += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify decomposition correctness using multiple random tests."""
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    rank = len(U[0])
    
    # Check with multiple random matrices for robustness
    rng = np.random.RandomState(42)
    for trial in range(3):
        A = rng.randint(-5, 6, size=(n, m)).astype(float)
        B = rng.randint(-5, 6, size=(m, p)).astype(float)
        C_expected = A @ B
        
        # Efficient computation: C = sum_r (A*U_r) * (B*V_r) * W_r^T
        # Vectorize: for each r, a_proj_r = sum(A * U[:,r].reshape(n,m))
        # Use broadcasting where possible
        U_reshaped = U_mat.reshape(n, m, rank)
        V_reshaped = V_mat.reshape(m, p, rank)
        W_reshaped = W_mat.reshape(p, n, rank)
        
        # a_proj[r] = sum over i,j of A[i,j] * U_reshaped[i,j,r]
        a_proj = np.einsum('ij,ijr->r', A, U_reshaped)
        b_proj = np.einsum('jk,jkr->r', B, V_reshaped)
        
        # C[i,k] = sum_r a_proj[r] * b_proj[r] * W_reshaped[k,i,r]
        C_computed = np.einsum('r,r,ki r->ik', a_proj, b_proj, W_reshaped)
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    
    return True

def construct():
    """Construct rank-minimal decomposition for 4x4x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Strategy: Split p=8 into two halves of 4, use Strassen for each 4x4x4
    # This gives rank 49*2 = 98
    U_444, V_444, W_444 = _build_4x4x4_strassen()
    
    # For 4x4x8: A is 4x4, B is 4x8, C is 4x8
    # B = [B_left (4x4) | B_right (4x4)]
    # C = [C_left (4x4) | C_right (4x4)]
    # C_left = A @ B_left, C_right = A @ B_right
    
    # U: (n*m, rank) = (16, 98)
    # V: (m*p, rank) = (32, 98)  
    # W: (p*n, rank) = (32, 98)
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # First half: C_left = A @ B_left
    for r in range(49):
        for row in range(16):
            U[row][r] = U_444[row][r]
        for row in range(16):
            j = row // 4
            k = row % 4
            V[j * 8 + k][r] = V_444[row][r]
        for row in range(16):
            k = row // 4
            i = row % 4
            W[k * 4 + i][r] = W_444[row][r]
    
    # Second half: C_right = A @ B_right
    for r in range(49):
        r2 = r + 49
        for row in range(16):
            U[row][r2] = U_444[row][r]
        for row in range(16):
            j = row // 4
            k = row % 4
            V[j * 8 + (k + 4)][r2] = V_444[row][r]
        for row in range(16):
            k = row // 4
            i = row % 4
            W[(k + 4) * 4 + i][r2] = W_444[row][r]
    
    # Verify
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
            if time.time() < deadline - 5.0:
                if not _verify(U, V, W, n, m, p):
                    raise RuntimeError("Both primary and fallback decompositions failed verification")
    
    return {"U": U, "V": V, "W": W}
