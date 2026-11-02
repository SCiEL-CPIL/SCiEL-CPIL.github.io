import time
import random
import numpy as np

DIMENSIONS = [2, 2, 7]

def construct():
    n, m, p = 2, 2, 7
    deadline = time.time() + 115  # Safety margin

    def verify_decomp(U, V, W, trials=30):
        """Fast verification using numpy vectorized ops."""
        Ua = np.array(U)
        Va = np.array(V)
        Wa = np.array(W)
        rank = Ua.shape[1]
        for _ in range(trials):
            A = np.random.randint(-10, 11, size=(n, m))
            B = np.random.randint(-10, 11, size=(m, p))
            C_true = A @ B
            u_vals = (Ua * A.flatten()[:, None]).sum(axis=0)
            v_vals = (Va * B.flatten()[:, None]).sum(axis=0)
            W_r = Wa.reshape(p, n, rank)
            C_calc = (W_r * (u_vals * v_vals)[None, None, :]).sum(axis=2).T
            if not np.allclose(C_true, C_calc, atol=1e-9):
                return False
        return True

    def build_rank25():
        """Build the known rank-25 Strassen-based decomposition for 2x2 * 2x7."""
        rank = 25
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        def add_strassen(col_start, k_start):
            # Strassen for 2x2 * 2x2 block starting at column k_start
            # M1 = (A00+A11)*(B0,ks + B1,ks+1)
            U[0, col_start+0] += 1; U[3, col_start+0] += 1
            V[7*0+k_start, col_start+0] += 1; V[7*1+k_start+1, col_start+0] += 1
            W[k_start*n+0, col_start+0] += 1; W[(k_start+1)*n+1, col_start+0] += 1
            
            # M2 = (A10+A11)*B0,ks
            U[2, col_start+1] += 1; U[3, col_start+1] += 1
            V[7*0+k_start, col_start+1] += 1
            W[k_start*n+1, col_start+1] += 1; W[(k_start+1)*n+1, col_start+1] -= 1
            
            # M3 = A00*(B0,ks+1 - B1,ks+1)
            U[0, col_start+2] += 1
            V[7*0+k_start+1, col_start+2] += 1; V[7*1+k_start+1, col_start+2] -= 1
            W[(k_start+1)*n+0, col_start+2] += 1; W[(k_start+1)*n+1, col_start+2] += 1
            
            # M4 = A11*(B1,ks - B0,ks)
            U[3, col_start+3] += 1
            V[7*1+k_start, col_start+3] += 1; V[7*0+k_start, col_start+3] -= 1
            W[k_start*n+0, col_start+3] += 1; W[k_start*n+1, col_start+3] += 1
            
            # M5 = (A00+A01)*B1,ks+1
            U[0, col_start+4] += 1; U[1, col_start+4] += 1
            V[7*1+k_start+1, col_start+4] += 1
            W[k_start*n+0, col_start+4] -= 1; W[(k_start+1)*n+0, col_start+4] += 1
            
            # M6 = (A10-A00)*(B0,ks+B0,ks+1)
            U[2, col_start+5] += 1; U[0, col_start+5] -= 1
            V[7*0+k_start, col_start+5] += 1; V[7*0+k_start+1, col_start+5] += 1
            W[(k_start+1)*n+1, col_start+5] += 1
            
            # M7 = (A01-A11)*(B1,ks+B1,ks+1)
            U[1, col_start+6] += 1; U[3, col_start+6] -= 1
            V[7*1+k_start, col_start+6] += 1; V[7*1+k_start+1, col_start+6] += 1
            W[k_start*n+0, col_start+6] += 1

        add_strassen(0, 0)
        add_strassen(7, 2)
        add_strassen(14, 4)
        
        # Schoolbook for column 6
        cs = 21
        U[0, cs+0] += 1; V[0*7+6, cs+0] += 1; W[6*n+0, cs+0] += 1
        U[1, cs+1] += 1; V[1*7+6, cs+1] += 1; W[6*n+0, cs+1] += 1
        U[2, cs+2] += 1; V[0*7+6, cs+2] += 1; W[6*n+1, cs+2] += 1
        U[3, cs+3] += 1; V[1*7+6, cs+3] += 1; W[6*n+1, cs+3] += 1

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def build_dft_rank49():
        """
        DFT-based decomposition: embed 2x2x7 into 2x2x8, apply Strassen at each
        frequency, then invert DFT. Total rank = 8 * 7 = 56, but we can try
        to compress. Actually, let's just use the direct approach:
        
        The idea: B is 2x7. Pad to 2x8. For each frequency f in 0..7, 
        the DFT coefficient B_f = sum_k B[k] * omega^(f*k) where omega = exp(2*pi*i/8).
        But we need real/rational entries. Instead, use the fact that the 8-point
        DFT can be decomposed via FFT butterfly structure.
        
        Alternative: Use the cyclic convolution structure. The tensor for 2x2x7
        multiplication has the property that C[i,j] = sum_k A[i,k]*B[k,j].
        If we think of B as a 2x7 matrix, we can pad to 2x8 and use the fact that
        multiplication by a circulant matrix can be diagonalized by DFT.
        
        However, the simplest DFT-based approach: use the 8th roots of unity.
        omega = exp(2*pi*i/8) = exp(pi*i/4) = (1+i)/sqrt(2).
        
        For each frequency f, define:
          B_f[k] = sum_{j=0}^{7} B[k,j] * omega^(f*j)  (with B[k,7]=0)
        
        Then A @ B_f gives the DFT of the product.
        
        The issue is that omega is not rational. We need to work over an extension
        field or use a different approach.
        
        Let's try a different angle: use the fact that 7 = 2^3 - 1, and use
        the 7th roots of unity instead. The 7th cyclotomic polynomial is
        x^6 + x^5 + x^4 + x^3 + x^2 + x + 1.
        
        Actually, let me try a more practical approach: use a rank-reducing
        transformation based on the structure of the 7-column dimension.
        
        Key insight: For 2x2 * 2x7, we can write B = [B1 | B2 | B3 | B4 | B5 | B6 | B7]
        where each Bj is 2x1. The product is:
        C[:,j] = A @ B[:,j]
        
        If we group columns and use Strassen-like combinations across all 7 columns
        simultaneously, we might reduce the rank below 25.
        
        Let's try: use 4 groups of columns with overlapping Strassen patterns.
        """
        # Actually, let me try a simpler improvement: use a better grouping.
        # Current: 3 blocks of 2 columns (Strassen, rank 7 each) + 1 column (schoolbook, rank 4) = 25
        # Alternative: Can we do 2 blocks of 3 columns + 1 block of 1 column?
        # For 2x2 * 2x3, what's the minimal rank?
        # Schoolbook: 6. Strassen doesn't directly apply to 2x3.
        # But we can use: 2x2*2x3 = 2x2*2x2 + 2x2*2x1
        # Rank = 7 + 4 = 11? That's worse.
        
        # Let me try: 2x2*2x7 with a different Strassen decomposition.
        # What if we use the 7 columns more cleverly?
        
        # Actually, the best known bound for 2x2*2x7 is 25 (from the block approach).
        # Let me verify if there's a rank-24 or lower.
        
        # For now, let's try to see if we can get rank 24 by merging some terms.
        
        return None  # Placeholder - we'll keep rank 25 for now

    result = build_rank25()
    if verify_decomp(result["U"], result["V"], result["W"]):
        return result

    # Fallback: schoolbook rank 28
    rank_sb = 28
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
