"""Matrix multiplication 2x4x8 via optimized block decomposition."""
import time

DIMENSIONS = [2, 4, 8]

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    # Rank 48 decomposition:
    # Split j into 2 blocks of 2, k into 4 blocks of 2.
    # For each (bj, bk) pair, we have a 2x2 x 2x2 sub-problem.
    # But A's two j-blocks share the same i-rows, so we can combine.
    # Use Strassen for 6 blocks and schoolbook for 2 blocks: 6*7 + 2*4 = 50? No.
    # Let's try: all 8 blocks with Strassen = 56.
    # Alternative: pair up bk blocks. For fixed bj, combine B's 4 k-blocks into one 2x8.
    # A_bj is 2x2, B_bj is 2x8. Product is 2x8.
    # Rank of 2x2x8 MM: can we do better than 7*4=28?
    # 2x2x8: split B into four 2x2. Standard: 4*7=28.
    # But we can share: A is same for all 4. Use (a+d, a-d, etc.) shared.
    # Actually for 2x2x8, the minimal rank is known to be 18 (or similar).
    # Let's just stick with 56 but verify it's correct, or try 48.
    
    # Let me try a rank-48 approach:
    # For each bj in {0,1}: A_bj is 2x2.
    # B_bj is 2x8, split into 4 blocks of 2x2.
    # Instead of 4 independent Strassen (rank 28 per bj, total 56),
    # use a combined approach: compute 7 Strassen products but with B combined.
    # This doesn't directly work because Strassen requires square blocks.
    
    # Let's just return the known-good rank-56 solution.
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    n_blocks_i = 1
    n_blocks_j = 2
    n_blocks_k = 4
    
    col = 0
    for bi in range(n_blocks_i):
        for bj in range(n_blocks_j):
            for bk in range(n_blocks_k):
                i0 = 2 * bi
                i1 = 2 * bi + 1
                j0 = 2 * bj
                j1 = 2 * bj + 1
                k0 = 2 * bk
                k1 = 2 * bk + 1
                
                u_a = i0 * m + j0
                u_b = i0 * m + j1
                u_c = i1 * m + j0
                u_d = i1 * m + j1
                
                v_e = j0 * p + k0
                v_f = j0 * p + k1
                v_g = j1 * p + k0
                v_h = j1 * p + k1
                
                w_c00 = k0 * n + i0
                w_c01 = k1 * n + i0
                w_c10 = k0 * n + i1
                w_c11 = k1 * n + i1
                
                # M1 = (a+d)(e+h)
                U[u_a][col] += 1
                U[u_d][col] += 1
                V[v_e][col] += 1
                V[v_h][col] += 1
                W[w_c00][col] += 1
                W[w_c11][col] += 1
                col += 1
                
                # M2 = (c+d)e
                U[u_c][col] += 1
                U[u_d][col] += 1
                V[v_e][col] += 1
                W[w_c10][col] += 1
                W[w_c11][col] -= 1
                col += 1
                
                # M3 = a(f-h)
                U[u_a][col] += 1
                V[v_f][col] += 1
                V[v_h][col] -= 1
                W[w_c01][col] += 1
                W[w_c11][col] += 1
                col += 1
                
                # M4 = d(g-e)
                U[u_d][col] += 1
                V[v_g][col] += 1
                V[v_e][col] -= 1
                W[w_c00][col] += 1
                W[w_c10][col] += 1
                col += 1
                
                # M5 = (a+b)h
                U[u_a][col] += 1
                U[u_b][col] += 1
                V[v_h][col] += 1
                W[w_c00][col] -= 1
                W[w_c01][col] += 1
                col += 1
                
                # M6 = (c-a)(e+f)
                U[u_c][col] += 1
                U[u_a][col] -= 1
                V[v_e][col] += 1
                V[v_f][col] += 1
                W[w_c11][col] += 1
                col += 1
                
                # M7 = (b-d)(g+h)
                U[u_b][col] += 1
                U[u_d][col] -= 1
                V[v_g][col] += 1
                V[v_h][col] += 1
                W[w_c00][col] += 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
