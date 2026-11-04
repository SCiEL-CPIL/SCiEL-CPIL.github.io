"""
Winograd-style decomposition for 4x5 by 5x6 matrix multiplication.
"""
import numpy as np

DIMENSIONS = [4, 5, 6]

# Winograd 4-point FFT matrices (using complex roots of unity for correctness)
# For real Winograd, we use the standard 4-point DFT/IDFT matrices
# W4: 4-point DFT matrix (forward)
# W4_H: 4-point IDFT matrix (with 1/4 scaling)

def construct():
    n, m, p = 4, 5, 6
    
    # 4-point DFT and IDFT matrices
    # F4: DFT, F4_inv: IDFT (with 1/4 scaling)
    w = np.exp(-2j * np.pi / 4)
    F4 = np.array([[1, 1, 1, 1],
                    [1, w, w**2, w**3],
                    [1, w**2, w**4, w**6],
                    [1, w**3, w**6, w**9]], dtype=complex)
    F4_inv = (1.0/4) * np.conjugate(F4).T  # IDFT matrix
    
    # For 4x4 matrix multiplication C = A @ B:
    # C = F4_inv.T @ ((F4 @ A.T) * (B @ F4_inv)).T
    # where * is element-wise
    # This uses 16 scalar multiplications (element-wise product of two 4x4 matrices)
    
    # Verify the formula:
    # Let A_hat = F4 @ A.T (4x4), B_hat = B @ F4_inv (4x4)
    # C_hat = A_hat * B_hat (element-wise, 4x4)
    # C = (F4_inv.T @ C_hat).T
    
    # For the 4x4x4 part, each of the 16 elements of C_hat corresponds to one rank-1 term
    # U factor: for element (i, freq) of A_hat = F4 @ A.T, the coefficient is F4[i, j] for A[j, i]
    #   So U[i*m + j, col] = F4[i, freq] where col corresponds to (freq, k)
    # V factor: for element (freq, k) of B_hat = B @ F4_inv, the coefficient is F4_inv[j, k] for B[j, k]
    #   So V[j*p + k, col] = F4_inv[j, k]
    # W factor: for C[i, k], the coefficient from C_hat[i, k] after inverse transform is F4_inv[i, freq]
    #   Wait, let me re-derive.
    
    # C = (F4_inv.T @ C_hat).T
    # C.T = F4_inv.T @ C_hat
    # C.T[i, k] = sum_freq F4_inv.T[i, freq] * C_hat[freq, k] = sum_freq F4_inv[freq, i] * C_hat[freq, k]
    # C[k, i] = sum_freq F4_inv[freq, i] * C_hat[freq, k]
    # C[i, k] = sum_freq F4_inv[freq, k] * C_hat[k, i]... 
    
    # Let me just verify numerically and build the factors.
    
    # Total rank components:
    # 1. Winograd 4x4x4 for A1(4x4) @ B1a(4x4): rank 16
    # 2. Schoolbook for A1(4x4) @ B1b(4x2): rank 32
    # 3. Schoolbook for A2(4x1) @ B1(4x6): rank 24
    # 4. Schoolbook for A2(4x1) @ B2(1x6): rank 24
    # Total: 16 + 32 + 24 + 24 = 96
    
    # Wait, that's not right either. Let me reconsider the partition.
    # C = A @ B where A is 4x5, B is 5x6
    # A = [A1 (4x4) | A2 (4x1)]
    # B = [B1 (4x6); B2 (1x6)]
    # C = A1 @ B1 + A2 @ B2
    # A1 @ B1: 4x4 @ 4x6, split into A1@B1[0:4] (4x4@4x4) + A1@B1[4:6] (4x4@4x2)
    # A2 @ B2: 4x1 @ 1x6
    
    # So terms:
    # - A1 @ B1[0:4]: Winograd, rank 16
    # - A1 @ B1[4:6]: schoolbook, rank 4*4*2 = 32
    # - A2 @ B2: schoolbook, rank 4*1*6 = 24
    # Total: 16 + 32 + 24 = 72
    
    rank = 16 + 32 + 24
    U = [[0.0] * rank for _ in range(n * m)]  # 20 x rank
    V = [[0.0] * rank for _ in range(m * p)]  # 30 x rank
    W = [[0.0] * rank for _ in range(p * n)]  # 24 x rank
    
    col = 0
    
    # Part 1: Winograd for A1(4x4) @ B1a(4x4) where B1a = B[0:4, :]
    # C_part[i, k] = sum_j A[i,j] * B[j,k] for i in 0..3, j in 0..3, k in 0..5
    # Wait, B1 is 4x6, so B1a should be 4x4 (first 4 columns)
    # But B1 has 6 columns. Let me re-partition.
    # B1 is 4x6. Split into B1a (4x4) columns 0-3 and B1b (4x2) columns 4-5.
    # A1 @ B1a: 4x4 @ 4x4 -> Winograd rank 16
    # A1 @ B1b: 4x4 @ 4x2 -> schoolbook rank 32
    # A2 @ B2: 4x1 @ 1x6 -> schoolbook rank 24
    # Total: 72
    
    # Winograd part: A1 (4x4) @ B1a (4x4)
    # Using FFT: C = (F4_inv.T @ ((F4 @ A1.T) * (B1a @ F4_inv))).T
    # Each element (freq, k) of the intermediate product is a rank-1 term
    # For term (freq, k):
    #   U[i*m + j, col] = F4[i, freq] * delta(j in 0..3)  -- but A1.T[j, i] = A1[i, j]
    #   Actually A_hat = F4 @ A1.T, so A_hat[freq, i] = sum_j F4[freq, j] * A1.T[j, i] = sum_j F4[freq, j] * A1[i, j]
    #   So for the U factor corresponding to A1[i, j]: U[i*5 + j, col] = F4[freq, j]
    #   B_hat = B1a @ F4_inv, so B_hat[j, k] = sum_freq2 B1a[j, k2] * F4_inv[k2, k]... 
    #   Wait, B1a @ F4_inv: (B1a @ F4_inv)[j, k] = sum_l B1a[j, l] * F4_inv[l, k]
    #   So for the V factor corresponding to B1a[j, l]: V[j*6 + l, col] = F4_inv[l, k]
    #   W factor: C[i, k] gets contribution F4_inv[freq, i] * (A_hat[freq, i] * B_hat[i, k])... 
    #   Hmm, I'm getting confused with indices. Let me just build it numerically and verify.
    
    # Let me use a different approach: build the full 4x4x4 tensor decomposition numerically.
    # The 4x4x4 multiplication tensor T[i,j,k] = delta(i,j,k) means T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    # Actually T is the tensor such that C[i,k] = sum_j A[i,j] B[j,k], so T[i,j,k] = 1 for all valid i,j,k with the constraint that j is the shared index.
    # T[i,j,k] = 1 if j is the middle index connecting A[i,j] and B[j,k] to C[i,k].
    # So T[i,j,k] = 1 for all i in 0..3, j in 0..3, k in 0..3 (for the 4x4x4 case).
    # Wait no, T[i,j,k] = delta_{i,j,k} means it's 1 when the indices match for the multiplication.
    # For matrix mult, T[i,j,k] = 1 for all i,j,k in the valid range (it's the identity tensor for the multiplication).
    # Actually the tensor for matrix multiplication is T[i,j,k] = 1 for all i,j,k, and the contraction is sum_j A[i,j] T[i,j,k] B[j,k] = sum_j A[i,j] B[j,k] = C[i,k].
    
    # For the Winograd decomposition of the 4x4x4 tensor:
    # U[i,j, (freq,k)] = F4[freq, j]  (for i in 0..3, j in 0..3, freq in 0..3, k in 0..3)
    # V[j,k, (freq,k')] = F4_inv[k, k']  (for j in 0..3, k in 0..3, freq in 0..3, k' in 0..3)
    # W[k',i, (freq,k)] = F4_inv[freq, i]  (for k' in 0..3, i in 0..3, freq in 0..3, k in 0..3)
    
    # Wait, I need to be careful. Let me define:
    # For the 4x4x4 core, the output is C[i, k] for i in 0..3, k in 0..3 (but we have 6 output columns)
    # Actually for A1 @ B1a, both are 4x4, so C_part is 4x4.
    
    # Let me just verify the formula numerically first.
    A_test = np.random.rand(4, 4)
    B_test = np.random.rand(4, 4)
    C_expected = A_test @ B_test
    
    A_hat = F4 @ A_test.T  # 4x4
    B_hat = B_test @ F4_inv  # 4x4
    C_hat = A_hat * B_hat  # element-wise, 4x4
    C_result = (F4_inv.T @ C_hat).T  # 4x4
    
    # Check if this equals C_expected
    if not np.allclose(C_result, C_expected):
        # Try different formula
        C_result2 = (F4_inv.T @ (A_hat * B_hat)).T
        if not np.allclose(C_result2, C_expected):
            # Maybe the formula is different
            # Try: C = F4_inv @ ( (F4 @ A) * (B @ F4_inv) )
            C_result3 = F4_inv @ ((F4 @ A_test) * (B_test @ F4_inv))
            if not np.allclose(C_result3, C_expected):
                # Try yet another
                C_result4 = (F4_inv @ ((F4 @ A_test.T).T * (B_test @ F4_inv))).T
                pass
    
    # Let me try the correct formula. For C[i,k] = sum_j A[i,j] B[j,k]:
    # This is equivalent to: for each i,k, C[i,k] is the inner product of row i of A and column k of B.
    # Using FFT: if we take DFT of A's rows and B's columns...
    # Actually, let's think of it as: C[i,k] = sum_j A[i,j] B[j,k]
    # If we define for each i: a_i = A[i,:], and for each k: b_k = B[:,k]
    # Then C[i,k] = a_i . b_k (dot product)
    # The FFT trick: a_i . b_k = IDFT(DFT(a_i) * DFT(b_k))[0]... no that's for convolution.
    # Actually a_i . b_k is NOT a convolution. It's a dot product.
    # The FFT doesn't directly help with dot products - it helps with convolutions.
    
    # So the Winograd/FFT approach for matrix multiplication is NOT straightforward.
    # The rank-16 claim for 4x4x4 via FFT is incorrect.
    
    # Let me just use schoolbook for everything to be safe, or find a valid Strassen-like decomposition.
    
    # For now, let me implement schoolbook to ensure correctness:
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
