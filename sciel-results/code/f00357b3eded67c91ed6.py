"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # The current best known approach: use Strassen's algorithm on 2x2 blocks.
    # For 4x4 * 4x6, we split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01, B02], [B10, B11, B12]] where each Bj is 2x2
    # C = [[C00, C01, C02], [C10, C11, C12]] where each Ck is 2x2
    #
    # C0k = A00*B0k + A01*B1k for k=0,1,2
    # C1k = A10*B0k + A11*B1k for k=0,1,2
    #
    # Each C block is a sum of two 2x2*2x2 products.
    # Using Strassen for each 2x2*2x2: 7 multiplications.
    # Naively: 6 C-blocks * 2 products * 7 = 84 multiplications.
    #
    # But we can share: for fixed k, both C0k and C1k use B0k and B1k.
    # The A blocks differ (A00,A01 vs A10,A11), but the B blocks are shared.
    #
    # Actually, let's think about this differently. We can use a rank-18 
    # decomposition for 4x4*4x4, and extend to 4x4*4x6.
    #
    # Known result: 4x4*4x4 has rank at most 18 (Lader's algorithm).
    # For 4x4*4x6, we can use the 4x4*4x4 rank-18 as a building block
    # and handle the extra 2 columns separately.
    #
    # Alternative: use the direct Strassen approach with careful sharing.
    # 
    # Let me use a cleaner approach: the rank-84 Strassen decomposition
    # is straightforward and verifiable. Let me implement it correctly.
    #
    # For each of the 3 column pairs k=0,1,2:
    #   C[:,k] = A * B[:,k] where B[:,k] is 4x2
    #   Using 2x2 block Strassen: 4 products of 2x2*2x2, each with 7 mults = 28
    #   Total: 3 * 28 = 84
    #
    # But wait - we can do better by sharing across k values!
    # The A blocks are the same for all k. If we precompute Strassen 
    # combinations of A blocks, we can reuse them.
    #
    # For Strassen 2x2*2x2:
    #   M1 = (A00+A11)*(B00+B11)
    #   M2 = (A10+A11)*B00
    #   M3 = A00*(B01-B11)
    #   M4 = A11*(B10-B00)
    #   M5 = (A00+A01)*B11
    #   M6 = (A10-A00)*(B00+B01)
    #   M7 = (A01-A11)*(B10+B11)
    #
    # For C0k = A00*B0k + A01*B1k, we need:
    #   7 mults for A00*B0k and 7 mults for A01*B1k = 14 mults
    # For C1k = A10*B0k + A11*B1k, we need:
    #   7 mults for A10*B0k and 7 mults for A11*B1k = 14 mults
    # Total per k: 28 mults, total: 84
    #
    # Can we share? The Strassen mults for A00*B0k and A10*B0k both use B0k.
    # But the A combinations differ, so no direct sharing.
    #
    # However, we can use a different decomposition. Let me try:
    # For fixed k, compute the 4x4 * 4x2 product using a rank-14 algorithm.
    # A 4x4 * 4x2 product has tensor of size 4*4*2 = 32 entries.
    # The rank is at most 28 (naive Strassen), but can be lower.
    #
    # Actually, let me just use the rank-84 solution which is simple and correct,
    # and see if we can optimize it.
    
    rank = 84
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(2):  # A row block index (0 or 1)
        for k in range(3):  # C col block index (0, 1, or 2)
            for j in range(2):  # Sum over j: A_ij * B_jk
                # Block A_ij: rows [2*i, 2*i+1], cols [2*j, 2*j+1]
                # Block B_jk: rows [2*j, 2*j+1], cols [2*k, 2*k+1]
                
                # A_ij elements indices in U (row index = i*m+j, flattened)
                a_idx = 2*i * m + 2*j
                b_idx = 2*i * m + 2*j + 1
                c_idx = (2*i+1) * m + 2*j
                d_idx = (2*i+1) * m + 2*j + 1
                
                # B_jk elements indices in V (row index = j*p+k, flattened)
                e_idx = 2*j * p + 2*k
                f_idx = 2*j * p + 2*k + 1
                g_idx = (2*j+1) * p + 2*k
                h_idx = (2*j+1) * p + 2*k + 1
                
                # C_ik block: rows [2*i, 2*i+1], cols [2*k, 2*k+1]
                # W index: W[col*n + row, r] where col is the C column, row is C row
                c11_idx = (2*k) * n + 2*i
                c12_idx = (2*k+1) * n + 2*i
                c21_idx = (2*k) * n + 2*i + 1
                c22_idx = (2*k+1) * n + 2*i + 1
                
                # Strassen 7 multiplications for A_ij * B_jk
                # M1 = (a+d)(e+h)
                U[a_idx][col] = 1
                U[d_idx][col] = 1
                V[e_idx][col] = 1
                V[h_idx][col] = 1
                W[c11_idx][col] = 1
                W[c22_idx][col] = 1
                col += 1
                
                # M2 = (c+d)e
                U[c_idx][col] = 1
                U[d_idx][col] = 1
                V[e_idx][col] = 1
                W[c21_idx][col] = 1
                W[c22_idx][col] = -1
                col += 1
                
                # M3 = a(f-h)
                U[a_idx][col] = 1
                V[f_idx][col] = 1
                V[h_idx][col] = -1
                W[c12_idx][col] = 1
                W[c22_idx][col] = 1
                col += 1
                
                # M4 = d(g-e)
                U[d_idx][col] = 1
                V[g_idx][col] = 1
                V[e_idx][col] = -1
                W[c11_idx][col] = 1
                W[c21_idx][col] = 1
                col += 1
                
                # M5 = (a+b)h
                U[a_idx][col] = 1
                U[b_idx][col] = 1
                V[h_idx][col] = 1
                W[c11_idx][col] = -1
                W[c12_idx][col] = 1
                col += 1
                
                # M6 = (c-a)(e+f)
                U[c_idx][col] = 1
                U[a_idx][col] = -1
                V[e_idx][col] = 1
                V[f_idx][col] = 1
                W[c22_idx][col] = 1
                col += 1
                
                # M7 = (b-d)(g+h)
                U[b_idx][col] = 1
                U[d_idx][col] = -1
                V[g_idx][col] = 1
                V[h_idx][col] = 1
                W[c11_idx][col] = 1
                col += 1
    
    # Verify the decomposition
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1
    
    T_decomp = np.zeros_like(T_expected)
    for r in range(rank):
        T_decomp += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    if not np.allclose(T_decomp, T_expected, atol=1e-10):
        # Fallback to naive
        rank_naive = n * m * p
        U_naive = np.zeros((n * m, rank_naive), dtype=int)
        V_naive = np.zeros((m * p, rank_naive), dtype=int)
        W_naive = np.zeros((p * n, rank_naive), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_naive[i*m+j][col] = 1
                    V_naive[j*p+k][col] = 1
                    W_naive[k*n+i][col] = 1
                    col += 1
        return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
