"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # A: n x m = 2x2, B: m x p = 2x8, C: n x p = 2x8
    # U: (n*m) x rank = 4 x rank
    # V: (m*p) x rank = 16 x rank  
    # W: (p*n) x rank = 16 x rank
    # W index: k*n + i (trace convention)
    
    deadline = time.time() + 145  # safety margin
    
    # Try rank 24 construction
    rank = 24
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # A indices: (i,j) -> i*m + j = i*2 + j
    #   A00=0, A01=1, A10=2, A11=3
    # B indices: (j,k) -> j*p + k = j*8 + k
    #   B0k = k, B1k = 8+k for k=0..7
    # C indices: (i,k) -> k*n + i = k*2 + i (trace convention W[k*n+i])
    #   C[i,k] -> k*2 + i
    
    # Define 24 rank-1 terms
    # Each term: (a_form, b_form, c_form)
    # a_form: dict mapping A_idx -> coeff
    # b_form: dict mapping B_idx -> coeff  
    # c_form: dict mapping C_idx -> coeff
    
    # For block q (columns 2q, 2q+1), local columns kl=0,1 map to global k=2q+kl
    
    # Strassen-like 7-term pattern per block, but we'll use 6 per block with sharing
    # Actually let's just define 24 terms explicitly
    
    # Pattern for each block: 6 terms using these A-forms:
    # Term type 0: A00 * B0k -> C0k
    # Term type 1: A01 * B1k -> C0k  
    # Term type 2: A10 * B0k -> C1k
    # Term type 3: A11 * B1k -> C1k
    # Term type 4: (A00+A11) * (B0k+B1k) -> contributes to C0k and C1k
    # Term type 5: (A10+A01) * (B1k-B0k) -> contributes to C0k and C1k
    
    # Wait, that's 6 terms but they don't give a valid 2x2x2 decomposition (need 7)
    
    # Let me use the actual Strassen 7-term per block but share between blocks
    # 4 blocks * 7 = 28, share 4 terms to get 24
    
    # Strassen terms for 2x2x2:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)(B01-B11) -> C10 += M2, C11 -= M2  
    # M3 = A00(B10-B11) -> C01 += M3, C11 -= M3
    # M4 = A11(B00-B10) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)(B00-B01) -> C00 += M5, C01 -= M5
    # M6 = (A10+A01)(B11-B01) -> C10 += M6, C11 += M6
    # M7 = (A01+A10)(B10-B11) -> C01 += M7, C10 -= M7
    
    # For block q, local (kl1, kl2) = (2q, 2q+1)
    # B indices: B0*(2q+kl) and B1*(2q+kl)
    # C indices: C*i*(2q+kl)
    
    # I'll create 24 terms by using 6 per block with a different sharing pattern
    # Or just use 28 and see if we can reduce
    
    # Let me try: use the 7 Strassen terms but make the A-side forms shared
    # Block 0: terms 0-6
    # Block 1: terms 7-13, but term 7 shares A-form with term 6
    # etc.
    
    # Actually, the A-forms in Strassen are:
    # M1: A00+A11
    # M2: A10+A11
    # M3: A00
    # M4: A11
    # M5: A00+A01
    # M6: A10+A01
    # M7: A01+A10
    
    # These are all different, so no natural sharing.
    
    # Let me just return rank 28 for now, but let me first verify it's correct
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for each of the 7 terms:
    # term: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict A_idx -> val
    # V_coeffs: dict local (j,kl) -> val, where kl is 0 or 1 within block
    # W_coeffs: dict local (i,kl) -> val
    
    strassen = [
        # M1 = (A00+A11)(B00+B11)
        ({0:1, 3:1}, {0:1, 1:1}, {0:1, 3:1}),
        # M2 = (A10+A11)(B01-B11)  
        ({2:1, 3:1}, {1:1, 3:-1}, {2:1, 3:-1}),
        # M3 = A00(B10-B11)
        ({0:1}, {2:1, 3:-1}, {1:1, 3:-1}),
        # M4 = A11(B00-B10)
        ({3:1}, {0:1, 2:-1}, {0:1, 2:1}),
        # M5 = (A00+A01)(B00-B01)
        ({0:1, 1:1}, {0:1, 1:-1}, {0:1, 1:-1}),
        # M6 = (A10+A01)(B11-B01)
        ({2:1, 1:1}, {3:1, 1:-1}, {2:1, 3:1}),
        # M7 = (A01+A10)(B10-B11)
        ({1:1, 2:1}, {2:1, 3:-1}, {1:1, 2:-1}),
    ]
    
    # V local index: j*2 + kl where j in {0,1}, kl in {0,1}
    # So: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # W local index: kl*2 + i where kl in {0,1}, i in {0,1}
    # So: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Wait, W[k*n+i] so for local block, W_idx = kl*n + i = kl*2 + i
    
    for q in range(4):
        for s in range(7):
            r = q * 7 + s
            u_c, v_c, w_c = strassen[s]
            
            for a_idx, val in u_c.items():
                U[a_idx, r] = val
            
            for v_local, val in v_c.items():
                j = v_local // 2
                kl = v_local % 2
                k = 2 * q + kl
                v_idx = j * p + k
                V[v_idx, r] = val
            
            for w_local, val in w_c.items():
                kl = w_local // 2
                i = w_local % 2
                k = 2 * q + kl
                w_idx = k * n + i
                W[w_idx, r] = val
    
    # Verify
    if time.time() < deadline:
        # Test with random matrices
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        # Compute C from decomposition
        C_comp = np.zeros((n, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        C_comp[i, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        if not np.allclose(C_comp, C_expected, atol=1e-10):
            print(f"Verification failed! Max diff: {np.max(np.abs(C_comp - C_expected))}")
            # Fall back to schoolbook rank 32
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = (i * m + j) * p + k
                        U[i*m+j, r] = 1
                        V[j*p+k, r] = 1
                        W[k*n+i, r] = 1
        else:
            print(f"Rank {rank} decomposition verified successfully")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
