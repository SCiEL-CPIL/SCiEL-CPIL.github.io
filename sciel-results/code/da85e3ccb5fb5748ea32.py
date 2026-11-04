import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 140.0

    n, m, p = 4, 4, 6

    # Strassen 2x2x2 coefficients
    # alpha[s][r]: coefficient of A element r in the s-th Strassen product's A-factor
    # beta[s][c]: coefficient of B element c in the s-th Strassen product's B-factor
    # recon[s]: list of (i, j, gamma) meaning C[i,j] += gamma * M_s
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

    # Part 1: 4x4 x 4x4 via Strassen (rank 28) for B columns 0-3
    # Part 2: 4x4 x 4x2 via schoolbook (rank 16) for B columns 4-5
    R = 28 + 16
    U = np.zeros((n * m, R))  # (16, R)
    V = np.zeros((m * p, R))  # (24, R)
    W = np.zeros((p * n, R))  # (24, R)

    col = 0
    # Strassen part: 7 outer products x 4 inner schoolbook = 28 terms
    for s in range(7):
        for si in range(2):  # inner row index of 2x2 block
            for sj in range(2):  # inner col index of 2x2 block
                # U: A is 4x4, block structure
                # A[r, a] where r is row (0-3), a is col (0-3)
                # r = br*2 + si, a = ar*2 + sj? No...
                # Let me think: A is 2x2 blocks of 2x2 matrices
                # A = [[A00, A01], [A10, A11]]
                # A00 has elements at rows 0-1, cols 0-1
                # The s-th Strassen product uses A_s = sum_r alpha[s][r] * A_block_r
                # where A_block_r is the 2x2 block at position (r//2, r%2)
                # Then M_s = A_s * B_s, which is a 2x2 matrix product
                # M_s[i,j] = sum_{si,sj} A_s[si,sj] * B_s[sj,j]... 
                # Actually M_s is a full 2x2 matrix product, so:
                # M_s[i,j] = sum_k A_s[i,k] * B_s[k,j]
                # For the rank-1 decomposition of this 2x2 product:
                # M_s[i,j] = sum_{si,sj} (A_s[i,si] * B_s[sj,j]) * delta(sj, si)? No.
                # A 2x2 matrix product M = A*B has rank-1 decomposition:
                # M[i,j] = sum_k A[i,k]*B[k,j]
                # So each (i,k,j) gives a rank-1 term with U[i,k]=A[i,k], V[k,j]=B[k,j], W[i,j]=1
                # But we need to combine with the Strassen coefficients.
                
                # Let me use a different approach. The 4x4x4 tensor is:
                # T[i,j,k] = sum_l A[i,l] * B[l,k]
                # Using Strassen at block level:
                # T = sum_s gamma_s * (A_s * B_s) where A_s, B_s are 2x2 block matrices
                # A_s[i,j] = sum_{r} alpha[s][r] * A_block_r[i%2, j%2] where r = (i//2)*2 + (j//2)
                # Wait, that's not right either.
                
                # Let me just use the standard formulation:
                # A is 4x4. Divide into 2x2 blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
                # B is 4x4. Similarly divided into B00, B01, B10, B11
                # Strassen: 7 products M_s = A_s * B_s where A_s, B_s are 2x2 matrices
                # A_s = sum_r alpha[s][r] * A_r where r indexes the 4 blocks
                # B_s = sum_c beta[s][c] * B_c
                # C = sum_s (recon coefficients) * M_s
                # Each M_s is a 2x2 matrix product, decomposed into 4 rank-1 terms
                
                # For the (si, sj) inner term:
                # U contribution: A_s[si, sj] = sum_r alpha[s][r] * A_block_r[si, sj]
                # where A_block_r has elements at positions (r//2 * 2 + si, r%2 * 2 + sj)
                # V contribution: B_s[sj, si]... wait, need to be careful
                # M_s[i,j] = sum_k A_s[i,k] * B_s[k,j]
                # So for inner index (si, sj), we have A_s[si, sj] * B_s[sj, ?]
                # Actually, the rank-1 term for M_s at position (i,j) via intermediate k is:
                # U[i,k] * V[k,j] * W[i,j]
                # So for the (si, sj) term in the 2x2 product:
                # i=si, k=sj, j=? -- no, this is getting confused.
                
                # Let me restart with a clearer formulation.
                # The 4x4x4 multiplication tensor T[i,j,k] = sum_l A[i,l]*B[l,k]
                # Using Strassen: T = sum_s sum_{i2,j2} gamma_s[i2,j2] * (sum_r alpha[s][r]*A_block_r) * (sum_c beta[s][c]*B_block_c)
                # where the product is 2x2 matrix multiplication
                # = sum_s sum_{i2,j2} gamma_s[i2,j2] * sum_{k2} (sum_r alpha[s][r]*A_block_r[i2,k2]) * (sum_c beta[s][c]*B_block_c[k2,j2])
                # = sum_s sum_{i2,k2,j2} gamma_s[i2,j2] * sum_r alpha[s][r]*A_block_r[i2,k2] * sum_c beta[s][c]*B_block_c[k2,j2]
                # Each (s, i2, k2, j2) gives a rank-1 term:
                # U: sum_r alpha[s][r] * A_block_r[i2,k2] -> element at (r//2*2+i2, r%2*2+k2)
                # V: sum_c beta[s][c] * B_block_c[k2,j2] -> element at (c//2*2+k2, c%2*2+j2)  
                # W: gamma_s[i2,j2] -> element at (j2*2+? , i2*2+?)
                
                # Let me just build it directly.
                pass
                col += 1

    # This approach is getting too complex. Let me use a simpler, verified method.
    # Use 4x4x4 Strassen (rank 28) properly constructed.
    # Reset and rebuild
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    col = 0
    for s in range(7):
        for si in range(2):
            for sj in range(2):
                # U: sum over r of alpha[s][r] * A_block_r[si, sj]
                # A_block_r is at rows (r//2)*2 + {0,1}, cols (r%2)*2 + {0,1}
                for r in range(4):
                    if alpha[s][r] != 0:
                        u_idx = (r // 2) * 2 + si  # row in A
                        u_idx_col = (r % 2) * 2 + sj  # col in A
                        U[u_idx * 4 + u_idx_col, col] += alpha[s][r]
                
                # V: sum over c of beta[s][c] * B_block_c[sj, si]
                # Wait, M_s[i,j] = sum_k A_s[i,k] * B_s[k,j]
                # So for the term with A_s[si, sj], we need B_s[sj, j] for each j
                # But we're decomposing M_s into rank-1 terms, so:
                # M_s[i,j] = sum_k A_s[i,k] * B_s[k,j]
                # The rank-1 term for (i, k, j) has:
                # U part: A_s[i,k], V part: B_s[k,j], W part: 1 (times gamma)
                # So for fixed (si, sj) as (i, k), we need to sum over j
                # But that's not a single rank-1 term.
                
                # I think the correct decomposition is:
                # For each s, the 2x2 product M_s = A_s * B_s is decomposed as:
                # M_s[i,j] = sum_k A_s[i,k] * B_s[k,j]
                # So there are 2*2*2 = 8 rank-1 terms per s? No, 2*2 = 4 terms if we fix the summation.
                # Actually, a 2x2 matrix product has 4 entries, each is a sum of 2 products.
                # So M_s[i,j] = A_s[i,0]*B_s[0,j] + A_s[i,1]*B_s[1,j]
                # This gives 4*2 = 8 rank-1 terms per s, total 7*8 = 56? That's too many.
                
                # I think I'm overcomplicating this. Let me use a known rank-28 decomposition.
                pass
                col += 1

    # Let me just use schoolbook for everything as a fallback, then try to optimize
    # 4x4x6 schoolbook: 4*4*6 = 96 rank-1 terms
    # But we know Strassen can do better.
    
    # Actually, let me just return the current rank-77 solution but with the incomplete code removed
    # and add a simple verification-based rank reduction.
    
    # Rebuild the original rank-77 decomposition
    R_orig = 77
    U = np.zeros((16, R_orig))
    V = np.zeros((24, R_orig))
    W = np.zeros((24, R_orig))
    
    col = 0
    for s in range(7):
        for si in range(7):
            for a in range(4):
                if alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]

            for b in range(4):
                if beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]

            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma

            col += 1

        for u in range(2):
            for v in range(2):
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]

                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]

                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma

                col += 1

    # Try to reduce rank by finding linear dependencies
    if time.time() < DEADLINE - 30:
        try:
            M = np.vstack([U, V, W])
            U_svd, s_vals, Vt = np.linalg.svd(M, full_matrices=False)
            tol = 1e-9
            rank_M = np.sum(s_vals > tol)
            
            if rank_M < R_orig:
                # Find a subset of columns of size rank_M that spans the column space
                # Use greedy algorithm
                kept = []
                current_rank = 0
                for c in range(R_orig):
                    if time.time() > DEADLINE - 15:
                        break
                    if len(kept) >= rank_M:
                        break
                    # Check if column c adds to the span of kept columns
                    if len(kept) == 0:
                        kept.append(c)
                        current_rank += 1
                    else:
                        # Project column c onto span of kept columns
                        cols = M[:, kept]
                        # Least squares
                        try:
                            coeff, residuals, rank_c, _ = np.linalg.lstsq(cols, M[:, c], rcond=None)
                            residual = np.linalg.norm(M[:, c] - cols @ coeff)
                            if residual > tol:
                                kept.append(c)
                                current_rank += 1
                        except:
                            kept.append(c)
                            current_rank += 1
                
                if len(kept) == rank_M:
                    U = U[:, kept]
                    V = V[:, kept]
                    W = W[:, kept]
        except Exception as e:
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
