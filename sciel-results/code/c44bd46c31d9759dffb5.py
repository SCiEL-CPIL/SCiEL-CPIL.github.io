import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    n, m, p = 4, 4, 6

    # Strassen 2x2x2 coefficients
    # alpha[s][r]: coefficient of A block r in the s-th Strassen product's A-factor
    # beta[s][c]: coefficient of B block c in the s-th Strassen product's B-factor
    # recon[s]: list of (i, j, gamma) meaning C_block[i,j] += gamma * M_s
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1}
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1}
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]

    # Build rank-28 decomposition for 4x4 x 4x4 using Strassen
    # A is 4x4, B is 4x4, C is 4x4
    # T[i,j,k] = sum_l A[i,l] * B[l,k]
    # Using Strassen: T = sum_s sum_{i2,j2} gamma_s[i2,j2] * (A_s * B_s)[i2,j2]
    # where A_s = sum_r alpha[s][r] * A_block_r, B_s = sum_c beta[s][c] * B_block_c
    # A_block_r is at rows (r//2)*2 + {0,1}, cols (r%2)*2 + {0,1}
    # B_block_c is at rows (c//2)*2 + {0,1}, cols (c%2)*2 + {0,1}
    # (A_s * B_s)[i2,j2] = sum_{k2} A_s[i2,k2] * B_s[k2,j2]
    # = sum_{k2} sum_r alpha[s][r] * A_block_r[i2,k2] * sum_c beta[s][c] * B_block_c[k2,j2]
    # Each (s, i2, k2, j2) gives a rank-1 term with:
    #   U: sum_r alpha[s][r] * A_block_r[i2,k2] -> element at (r//2*2+i2, r%2*2+k2)
    #   V: sum_c beta[s][c] * B_block_c[k2,j2] -> element at (c//2*2+k2, c%2*2+j2)
    #   W: gamma_s[i2,j2] -> element at (j2*2+?, i2*2+?)
    # Wait, W should be indexed by output position. C[i,j] where i is row, j is col.
    # C_block[i2,j2] corresponds to C rows i2*2+{0,1}, cols j2*2+{0,1}
    # So W element is at (j2*2 + col_offset, i2*2 + row_offset)? No.
    # W has shape (p*n, rank) = (24, rank) for the full 4x4x6 case.
    # For the 4x4x4 part, W should have shape (16, rank) with W[k*n+i, r] convention.
    # W[k*n+i] where k is output column, i is output row.
    # So for C[i,j] (i=row, j=col), W index is j*n + i = j*4 + i.
    # C_block[i2,j2] has entries C[i2*2+di, j2*2+dj] for di,dj in {0,1}
    # But wait, the Strassen reconstruction gives C_block[i2,j2] as a 2x2 matrix.
    # Actually, the Strassen algorithm computes C = sum_s (recon) * M_s where M_s is 2x2.
    # C is 4x4, divided into 2x2 blocks C00, C01, C10, C11.
    # C_block[i2,j2] is a 2x2 matrix at rows i2*2+{0,1}, cols j2*2+{0,1}.
    # So for the (i2,j2) block, the output elements are at positions (i2*2+di, j2*2+dj).
    # W index for C[i,j] is j*4 + i.
    # So for the (i2,j2) block element (di,dj), W index is (j2*2+dj)*4 + (i2*2+di).
    
    # But in the rank-1 decomposition, each term contributes to all 4 elements of C_block[i2,j2].
    # So for a given (s, i2, k2, j2) term, W has non-zero entries at all 4 positions of C_block[i2,j2].
    # W[(j2*2+dj)*4 + (i2*2+di)] = gamma_s[i2,j2] for di,dj in {0,1}.
    
    R_strassen = 28  # 7 Strassen products * 2*2*2 = 56? No, 7*4 = 28 if we do it right.
    # Actually, for each s, we have 2x2x2 = 8 rank-1 terms (i2, k2, j2 each in {0,1})
    # So 7 * 8 = 56? That's too many.
    
    # Let me reconsider. The 4x4x4 tensor T[i,j,k] = sum_l A[i,l]*B[l,k]
    # Using Strassen at the 2x2 block level:
    # T = sum_s sum_{i2,j2} gamma_s[i2,j2] * (A_s * B_s)[i2,j2]
    # (A_s * B_s)[i2,j2] = sum_{k2} A_s[i2,k2] * B_s[k2,j2]
    # So T = sum_s sum_{i2,k2,j2} gamma_s[i2,j2] * A_s[i2,k2] * B_s[k2,j2]
    # Each (s, i2, k2, j2) is a rank-1 term.
    # There are 7 * 2 * 2 * 2 = 56 such terms.
    
    # But we can do better. The standard Strassen for 4x4x4 has rank 28.
    # How? By using the fact that (A_s * B_s) is a 2x2 matrix product, which has rank 4.
    # So 7 * 4 = 28 rank-1 terms.
    # The 4 rank-1 terms for a 2x2 product M = A*B are:
    # M[i,j] = sum_k A[i,k]*B[k,j]
    # This gives 4*2 = 8 terms if we expand fully, but we can group them.
    # Actually, a 2x2 matrix product has 4 entries, each a sum of 2 products.
    # So there are 4*2 = 8 rank-1 terms per 2x2 product, giving 7*8 = 56.
    
    # I think the rank-28 claim is wrong. Let me just build it correctly.
    # For now, let's use the straightforward expansion.
    
    # Actually, let me just build the full decomposition directly.
    # 4x4 x 4x6 multiplication.
    # T[i,j,k] = sum_l A[i,l] * B[l,k] for i in 0..3, j in 0..5, k in 0..3
    # Wait, the tensor is T[i,j,k] where i is row of A (0..3), k is col of A / row of B (0..3), j is col of B (0..5).
    # C[i,j] = sum_k A[i,k] * B[k,j]
    
    # Schoolbook: 4*4*6 = 96 rank-1 terms.
    # Strassen for the 4x4 part: use Strassen for columns 0-3 of B, schoolbook for columns 4-5.
    # Strassen 4x4x4: rank 28 (if we can achieve it) or 56 (if we expand fully).
    # Schoolbook 4x4x2: rank 16.
    # Total: 28 + 16 = 44 or 56 + 16 = 72.
    
    # Let me build the Strassen 4x4x4 part with rank 56 (full expansion) for now.
    R_strassen = 56
    R_schoolbook = 16
    R = R_strassen + R_schoolbook
    
    U = np.zeros((n * m, R))  # (16, R)
    V = np.zeros((m * p, R))  # (24, R)
    W = np.zeros((p * n, R))  # (24, R)
    
    col = 0
    # Strassen part for B columns 0-3
    for s in range(7):
        for i2 in range(2):
            for k2 in range(2):
                for j2 in range(2):
                    # U: sum_r alpha[s][r] * A_block_r[i2, k2]
                    # A_block_r is at rows (r//2)*2 + {0,1}, cols (r%2)*2 + {0,1}
                    for r in range(4):
                        if alpha[s][r] != 0:
                            u_row = (r // 2) * 2 + i2
                            u_col = (r % 2) * 2 + k2
                            U[u_row * 4 + u_col, col] += alpha[s][r]
                    
                    # V: sum_c beta[s][c] * B_block_c[k2, j2]
                    # B_block_c is at rows (c//2)*2 + {0,1}, cols (c%2)*2 + {0,1}
                    # But B is 4x6, so for the 4x4 part, B columns 0-3.
                    # B_block_c is at rows (c//2)*2 + {0,1}, cols (c%2)*2 + {0,1}
                    for c in range(4):
                        if beta[s][c] != 0:
                            v_row = (c // 2) * 2 + k2
                            v_col = (c % 2) * 2 + j2
                            V[v_row * 6 + v_col, col] += beta[s][c]
                    
                    # W: gamma_s[i2, j2] for all 4 elements of C_block[i2, j2]
                    # C_block[i2, j2] is at rows i2*2+{0,1}, cols j2*2+{0,1}
                    for di in range(2):
                        for dj in range(2):
                            w_row = (j2 * 2 + dj) * 4 + (i2 * 2 + di)
                            # Find gamma for this (i2, j2)
                            gamma = 0
                            for (ri, rj, g) in recon[s]:
                                if ri == i2 and rj == j2:
                                    gamma = g
                                    break
                            W[w_row, col] = gamma
                    
                    col += 1
    
    # Schoolbook part for B columns 4-5
    for i in range(4):
        for k in range(4):
            for j in range(2):  # columns 4 and 5
                U[i * 4 + k, col] = 1
                V[k * 6 + (4 + j), col] = 1
                W[(4 + j) * 4 + i, col] = 1
                col += 1
    
    # Verify the decomposition
    # C[i,j] = sum_r U[i*4 + ?, r] * V[?, r] * W[j*4 + i, r]
    # Wait, the tensor is T[i,k,j] = sum_r U[i*4+k, r] * V[k*6+j, r] * W[j*4+i, r]
    # Let me verify with a random matrix.
    try:
        A_test = np.random.randn(4, 4)
        B_test = np.random.randn(4, 6)
        C_test = A_test @ B_test
        
        C_comp = np.zeros((4, 6))
        for r in range(R):
            for i in range(4):
                for k in range(4):
                    for j in range(6):
                        C_comp[i, j] += U[i * 4 + k, r] * V[k * 6 + j, r] * W[j * 4 + i, r]
        
        if np.allclose(C_test, C_comp, atol=1e-9):
            pass
        else:
            # Fallback to schoolbook
            R_fb = 96
            U = np.zeros((16, R_fb))
            V = np.zeros((24, R_fb))
            W = np.zeros((24, R_fb))
            col = 0
            for i in range(4):
                for k in range(4):
                    for j in range(6):
                        U[i * 4 + k, col] = 1
                        V[k * 6 + j, col] = 1
                        W[j * 4 + i, col] = 1
                        col += 1
            R = R_fb
    except Exception:
        pass
    
    # Try to reduce rank by finding linear dependencies
    if time.time() < DEADLINE - 30:
        try:
            M = np.vstack([U, V, W])
            U_svd, s_vals, Vt = np.linalg.svd(M, full_matrices=False)
            tol = 1e-9
            rank_M = int(np.sum(s_vals > tol))
            
            if rank_M < R:
                kept = []
                for c in range(R):
                    if time.time() > DEADLINE - 15:
                        break
                    if len(kept) >= rank_M:
                        break
                    if len(kept) == 0:
                        kept.append(c)
                    else:
                        cols = M[:, kept]
                        try:
                            coeff, residuals, rank_c, _ = np.linalg.lstsq(cols, M[:, c], rcond=None)
                            residual = np.linalg.norm(M[:, c] - cols @ coeff)
                            if residual > tol:
                                kept.append(c)
                        except:
                            kept.append(c)
                
                if len(kept) == rank_M:
                    U = U[:, kept]
                    V = V[:, kept]
                    W = W[:, kept]
        except Exception:
            pass
    
    def to_list(mat):
        result = []
        for row in mat:
            result.append([])
            for x in row:
                x_rounded = round(x)
                if abs(x - x_rounded) < 1e-9:
                    result[-1].append(int(x_rounded))
                else:
                    result[-1].append(float(x))
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
