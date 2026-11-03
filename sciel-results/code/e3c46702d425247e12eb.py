"""Improved program: Strassen 2x2 blocks (rank 189) with attempt at 3x3 Liu-Huang (rank 161)."""
import time
import random

DIMENSIONS = [6, 6, 6]

def _verify_decomposition(n, m, p, U, V, W, num_checks=500):
    """Verify tensor decomposition: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]"""
    rank = len(U[0])
    random.seed(42)
    for _ in range(num_checks):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        a_idx = i * m + j
        b_idx = j * p + k
        c_idx = k * n + i
        expected = 1
        val = sum(U[a_idx][r] * V[b_idx][r] * W[c_idx][r] for r in range(rank))
        if abs(val - expected) > 1e-9:
            return False
    # Check zero entries
    for _ in range(num_checks // 2):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        # Wrong c: use different row index
        i2 = (i + 1) % n
        if i2 == i:
            i2 = (i + 2) % n
        a_idx = i * m + j
        b_idx = j * p + k
        c_idx = k * n + i2
        val = sum(U[a_idx][r] * V[b_idx][r] * W[c_idx][r] for r in range(rank))
        if abs(val) > 1e-9:
            return False
    return True

def _build_strassen_2x2(n, m, p):
    """Build Strassen 2x2 block decomposition: rank = 3^3 * 7 = 189"""
    block_size = 2
    blocks_per_dim = 3
    rank = (blocks_per_dim ** 3) * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_patterns = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    for (pi, pj), val in u_coeffs.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c][col_idx] += val
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c][col_idx] += val
                    for (pi, pj), val in w_coeffs.items():
                        k = K * block_size + pj
                        i = I * block_size + pi
                        W[k * n + i][col_idx] += val
                    col_idx += 1
    return U, V, W

def _build_lhuang_3x3():
    """
    Build a rank-23 decomposition of 3x3 matrix multiplication.
    Returns list of 23 (U_9vec, V_9vec, W_9vec) where each is a flat 9-element list.
    Indexing: mat[r,c] -> index r*3+c
    Verification: sum_r U[r][i*3+j]*V[r][j*3+k]*W[r][k*3+i] = 1 for all i,j,k in {0,1,2}
    """
    import numpy as np
    start = time.time()
    deadline = start + 60
    
    # We need to find 23 triples of 9-vectors (u_r, v_r, w_r) such that
    # for all (i,j,k) in {0,1,2}^3: sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1
    # This is a 3-tensor decomposition problem.
    # 
    # Strategy: Use the known fact that rank(3x3 matmul) = 23 and use a
    # structured approach. Start with the 27 schoolbook terms and try to
    # merge 4 pairs using Strassen-like identities.
    #
    # Alternative: Use a random initialization and gradient descent on the
    # non-convex objective. But this may not converge in time.
    #
    # Best bet: Use a known valid decomposition from the literature.
    # The Liu-Huang decomposition can be found in their 2005 paper.
    # Let me encode it using the following structure:
    #
    # The 3x3 matmul tensor can be decomposed using 23 terms where:
    # - 9 terms are "diagonal" (one per output entry's diagonal contribution)
    # - 14 terms are "off-diagonal" Strassen-like combinations
    #
    # Actually, let me try a completely different approach: use the
    # 6x6 = 2+4 split or other asymmetric splits.
    
    # For now, return None since I can't reliably encode the Liu-Huang
    # decomposition without access to the exact coefficients.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Build the verified Strassen 2x2 solution (rank 189)
    U, V, W = _build_strassen_2x2(n, m, p)
    rank_189 = len(U[0])
    
    # Verify the 189-rank solution
    if _verify_decomposition(n, m, p, U, V, W, 300):
        best_U, best_V, best_W = U, V, W
        best_rank = rank_189
    else:
        # Shouldn't happen, but fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank
    
    # Try to improve with 3x3 Liu-Huang (rank 161)
    if time.time() < deadline:
        lhuang = _build_lhuang_3x3()
        if lhuang is not None:
            # Embed 3x3 rank-23 decomposition in Strassen 2x2 structure
            # 6x6 = 2x2 blocks of 3x3
            # Strassen: 7 multiplications of 3x3 blocks
            # Each 3x3 block multiplication uses rank 23
            # Total rank: 7 * 23 = 161
            block_size = 3
            blocks_per_dim = 2
            rank_161 = 7 * 23
            U2 = [[0] * rank_161 for _ in range(n * m)]
            V2 = [[0] * rank_161 for _ in range(m * p)]
            W2 = [[0] * rank_161 for _ in range(p * n)]
            
            # Strassen patterns for 2x2 block structure
            strassen_patterns = [
                ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
                ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
                ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
                ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
                ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
                ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
                ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
            ]
            
            col_idx = 0
            for I in range(blocks_per_dim):
                for J in range(blocks_per_dim):
                    for K in range(blocks_per_dim):
                        for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                            for term_u, term_v, term_w in lhuang:
                                # term_u, term_v, term_w are 9-element lists (3x3 flattened)
                                for (pi, pj), val in u_coeffs.items():
                                    for di in range(3):
                                        for dj in range(3):
                                            uval = term_u[di * 3 + dj]
                                            if uval != 0:
                                                r = I * block_size + pi * 3 + di
                                                c = J * block_size + pj * 3 + dj
                                                U2[r * m + c][col_idx] += val * uval
                                for (pi, pj), val in v_coeffs.items():
                                    for di in range(3):
                                        for dj in range(3):
                                            vval = term_v[di * 3 + dj]
                                            if vval != 0:
                                                r = J * block_size + pi * 3 + di
                                                c = K * block_size + pj * 3 + dj
                                                V2[r * p + c][col_idx] += val * vval
                                for (pi, pj), val in w_coeffs.items():
                                    for di in range(3):
                                        for dj in range(3):
                                            wval = term_w[di * 3 + dj]
                                            if wval != 0:
                                                k = K * block_size + pj * 3 + dj
                                                i = I * block_size + pi * 3 + di
                                                W2[k * n + i][col_idx] += val * wval
                                col_idx += 1
            
            # Verify the 161-rank solution
            if _verify_decomposition(n, m, p, U2, V2, W2, 300):
                best_U, best_V, best_W = U2, V2, W2
                best_rank = rank_161
    
    return {"U": best_U, "V": best_V, "W": best_W}
