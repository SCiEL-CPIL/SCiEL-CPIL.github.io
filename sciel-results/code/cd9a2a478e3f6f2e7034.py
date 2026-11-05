"""Matrix multiplication 4x5x5 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    deadline = time.time() + 110  # Safety margin before 120s limit

    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def verify_decomposition(U, V, W, n, m, p, tol=1e-8):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for j2 in range(m):
                    for k in range(p):
                        b = j2 * p + k
                        for k2 in range(p):
                            for i2 in range(n):
                                c = k2 * n + i2
                                val = 0.0
                                for cc in range(rank):
                                    val += U[a, cc] * V[b, cc] * W[c, cc]
                                if i == i2 and j == j2 and k == k2:
                                    expected = 1.0
                                else:
                                    expected = 0.0
                                if abs(val - expected) > tol:
                                    return False
        return True

    def verify_fast(U, V, W, n, m, p, tol=1e-8):
        """Faster verification using tensor contraction."""
        rank = U.shape[1]
        # Check all tensor entries
        for i in range(n):
            for j in range(m):
                for k2 in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k in range(p):
                                # T[i,j,k2,i2,j2,k] = sum_r U[i*m+j, r] * V[j2*p+k, r] * W[k2*n+i2, r]
                                val = np.dot(U[i*m+j], V[j2*p+k] * W[k2*n+i2])
                                if i == i2 and j == j2 and k == k2:
                                    expected = 1.0
                                else:
                                    expected = 0.0
                                if abs(val - expected) > tol:
                                    return False
        return True

    def winograd_2x5x5():
        """
        Winograd-style decomposition for 2x5 * 5x5.
        
        For 2x5 * 5x5, we can use the following approach:
        Split the 5 columns into groups and use Winograd's technique.
        
        Actually, let me use a direct approach:
        For 2x5 * 5x5, schoolbook is 50.
        We can potentially reduce by using the fact that some products can be shared.
        
        Let me try a simpler Winograd-like approach:
        For each output element C[i,k] = sum_j A[i,j] * B[j,k]
        
        With n=2, m=5, p=5, we have 10 output elements, each requiring 5 multiplications = 50 total.
        
        One Winograd trick: for pairs of columns j and j', we can compute:
        (A[i,j] + A[i,j']) * (B[j,k] + B[j',k]) = A[i,j]*B[j,k] + A[i,j]*B[j',k] + A[i,j']*B[j,k] + A[i,j']*B[j',k]
        
        This gives 4 products with 1 multiplication, but we still need to subtract cross terms.
        
        For now, let me just return schoolbook for 2x5x5 (rank 50).
        """
        rank = 50
        U = np.zeros((2 * 5, rank))
        V = np.zeros((5 * 5, rank))
        W = np.zeros((5 * 2, rank))
        col = 0
        for i in range(2):
            for j in range(5):
                for k in range(5):
                    U[i * 5 + j, col] = 1
                    V[j * 5 + k, col] = 1
                    W[k * 2 + i, col] = 1
                    col += 1
        return U, V, W

    def try_strassen_4x5x5():
        """
        Try to use Strassen-like decomposition for 4x5x5.
        
        Idea: Split the 4x5 matrix into two 2x5 blocks (rows 0-1 and rows 2-3).
        The 5x5 matrix is shared.
        
        C_top = A_top * B (2x5 * 5x5 = 2x5)
        C_bottom = A_bottom * B (2x5 * 5x5 = 2x5)
        
        If we can compute each 2x5*5x5 in rank r, total rank is 2r.
        Schoolbook: 2*50 = 100.
        
        To beat 100, we need r < 50 for each 2x5*5x5.
        
        Let me try a Winograd approach for 2x5*5x5.
        """
        # For 2x5 * 5x5, let's try to use the following decomposition:
        # 
        # C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
        # 
        # Winograd trick for pairs:
        # Let's pair columns (0,1), (2,3), and leave 4 alone.
        # 
        # For pair (0,1):
        # S1 = A[i,0] + A[i,1]
        # T1_k = B[0,k] + B[1,k]
        # P1_k = S1 * T1_k = A[i,0]*B[0,k] + A[i,0]*B[1,k] + A[i,1]*B[0,k] + A[i,1]*B[1,k]
        # 
        # We need: A[i,0]*B[0,k], A[i,0]*B[1,k], A[i,1]*B[0,k], A[i,1]*B[1,k]
        # From P1_k we get the sum of all 4.
        # We also need:
        # S2 = A[i,0] - A[i,1]
        # T2_k = B[0,k] - B[1,k]
        # P2_k = S2 * T2_k = A[i,0]*B[0,k] - A[i,0]*B[1,k] - A[i,1]*B[0,k] + A[i,1]*B[1,k]
        # 
        # Then:
        # A[i,0]*B[0,k] = (P1_k + P2_k)/4 + ... hmm, this is getting complicated.
        # 
        # Actually, the standard Winograd for 2x2*2x2 uses 3 multiplications instead of 4.
        # For our case, we have 5 columns, not 2.
        # 
        # Let me just return None and fall back to schoolbook.
        return None

    # Start with schoolbook
    U_sb, V_sb, W_sb = schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p

    # Try various approaches
    if time.time() < deadline - 60:
        # Approach 1: Two 2x5x5 blocks
        U1, V1, W1 = winograd_2x5x5()
        U2, V2, W2 = winograd_2x5x5()
        
        # Combine
        total_rank = U1.shape[1] + U2.shape[1]
        U_comb = np.zeros((n * m, total_rank))
        V_comb = np.zeros((m * p, total_rank))
        W_comb = np.zeros((p * n, total_rank))
        
        r1 = U1.shape[1]
        # Block 1: rows 0,1
        for i in range(2):
            for j in range(m):
                U_comb[i * m + j, 0:r1] = U1[i * m + j, :]
            for k in range(p):
                W_comb[k * n + i, 0:r1] = W1[k * 2 + i, :]
        for j in range(m):
            for k in range(p):
                V_comb[j * p + k, 0:r1] = V1[j * p + k, :]
        
        # Block 2: rows 2,3
        for i in range(2, 4):
            for j in range(m):
                U_comb[i * m + j, r1:total_rank] = U2[(i - 2) * m + j, :]
            for k in range(p):
                W_comb[k * n + i, r1:total_rank] = W2[k * 2 + (i - 2), :]
        for j in range(m):
            for k in range(p):
                V_comb[j * p + k, r1:total_rank] = V2[j * p + k, :]
        
        if verify_fast(U_comb, V_comb, W_comb, n, m, p):
            best_U, best_V, best_W = U_comb, V_comb, W_comb
            best_rank = total_rank

    # Try Strassen-like approach
    if time.time() < deadline - 30:
        result = try_strassen_4x5x5()
        if result is not None:
            U_s, V_s, W_s = result
            if verify_fast(U_s, V_s, W_s, n, m, p):
                if U_s.shape[1] < best_rank:
                    best_U, best_V, best_W = U_s, V_s, W_s
                    best_rank = U_s.shape[1]

    # Return the best result
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
