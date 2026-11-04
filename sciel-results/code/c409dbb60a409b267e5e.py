"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    deadline = time.time() + 115.0

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def strassen_2x2x2():
        # Standard Strassen decomposition for 2x2x2 matrix multiplication
        U = [
            [1, 0, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 0, 1],
            [1, 1, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 0, 1, 1],
        ]
        V = [
            [1, 1, 0, 1, 0, 1, 1],
            [0, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 1, 0, 1, 0],
            [1, 1, 1, 0, 1, 0, 1],
        ]
        W = [
            [1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0, 1, 0],
            [1, 0, 0, 1, -1, 0, 0],
        ]
        return U, V, W

    def verify_fast(U, V, W, n, m, p, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        return np.allclose(T, np.ones((n, m, p)), atol=tol, rtol=0)

    def verify_strassen_block():
        U_s, V_s, W_s = strassen_2x2x2()
        U_a = np.array(U_s, dtype=float)
        V_a = np.array(V_s, dtype=float)
        W_a = np.array(W_s, dtype=float)
        U_r = U_a.reshape(2, 2, 7)
        V_r = V_a.reshape(2, 2, 7)
        W_r = W_a.reshape(2, 2, 7)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        return np.allclose(T, np.ones((2, 2, 2)), atol=1e-9, rtol=0)

    best_result = schoolbook()
    best_rank = n * m * p

    if not verify_strassen_block():
        return best_result

    def build_decomposition(i_blocks, j_blocks, k_blocks):
        U_cols = []
        V_cols = []
        W_cols = []
        
        for i_blk in i_blocks:
            for j_blk in j_blocks:
                for k_blk in k_blocks:
                    ni = len(i_blk)
                    nj = len(j_blk)
                    nk = len(k_blk)
                    
                    if ni == 2 and nj == 2 and nk == 2:
                        U_s, V_s, W_s = strassen_2x2x2()
                        for c in range(7):
                            U_col = [0.0] * (n * m)
                            V_col = [0.0] * (m * p)
                            W_col = [0.0] * (p * n)
                            for ii in range(2):
                                for jj in range(2):
                                    U_col[i_blk[ii]*m + j_blk[jj]] = U_s[ii*2+jj][c]
                            for jj in range(2):
                                for kk in range(2):
                                    V_col[j_blk[jj]*p + k_blk[kk]] = V_s[jj*2+kk][c]
                            for kk in range(2):
                                for ii in range(2):
                                    W_col[k_blk[kk]*n + i_blk[ii]] = W_s[kk*2+ii][c]
                            U_cols.append(U_col)
                            V_cols.append(V_col)
                            W_cols.append(W_col)
                    else:
                        # Schoolbook for this sub-block
                        for ii in range(ni):
                            for jj in range(nj):
                                for kk in range(nk):
                                    U_col = [0.0] * (n * m)
                                    V_col = [0.0] * (m * p)
                                    W_col = [0.0] * (p * n)
                                    U_col[i_blk[ii]*m + j_blk[jj]] = 1.0
                                    V_col[j_blk[jj]*p + k_blk[kk]] = 1.0
                                    W_col[k_blk[kk]*n + i_blk[ii]] = 1.0
                                    U_cols.append(U_col)
                                    V_cols.append(V_col)
                                    W_cols.append(W_col)
        
        if not U_cols:
            return None, 0
        
        rank = len(U_cols)
        U_mat = [[U_cols[c][r] for c in range(rank)] for r in range(n*m)]
        V_mat = [[V_cols[c][r] for c in range(rank)] for r in range(m*p)]
        W_mat = [[W_cols[c][r] for c in range(rank)] for r in range(p*n)]
        
        return {"U": U_mat, "V": V_mat, "W": W_mat}, rank

    def generate_partitions(dim):
        """Generate all partitions of dim into blocks of size 1 or 2."""
        results = []
        def backtrack(remaining, current):
            if remaining == 0:
                results.append(list(current))
                return
            if remaining >= 1:
                current.append(1)
                backtrack(remaining - 1, current)
                current.pop()
            if remaining >= 2:
                current.append(2)
                backtrack(remaining - 2, current)
                current.pop()
        backtrack(dim, [])
        return results

    def blocks_from_sizes(sizes):
        """Convert block sizes to actual index blocks."""
        blocks = []
        start = 0
        for s in sizes:
            blocks.append(list(range(start, start + s)))
            start += s
        return blocks

    # Generate all partitions for each dimension
    i_size_options = generate_partitions(n)
    j_size_options = generate_partitions(m)
    k_size_options = generate_partitions(p)

    # Try all combinations
    for i_sizes in i_size_options:
        if time.time() > deadline:
            break
        i_blocks = blocks_from_sizes(i_sizes)
        for j_sizes in j_size_options:
            if time.time() > deadline:
                break
            j_blocks = blocks_from_sizes(j_sizes)
            for k_sizes in k_size_options:
                if time.time() > deadline:
                    break
                k_blocks = blocks_from_sizes(k_sizes)
                
                result, rank = build_decomposition(i_blocks, j_blocks, k_blocks)
                if result and rank < best_rank:
                    if verify_fast(result["U"], result["V"], result["W"], n, m, p):
                        best_result = result
                        best_rank = rank

    return best_result
# EVOLVE-BLOCK-END
