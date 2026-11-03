import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will use a block decomposition.
    # Split 5 into 1 + 2 + 2.
    # This creates 3x3 blocks.
    # Block sizes: 1, 2, 2.
    
    # Ranks for sub-problems
    # R(n, m, p)
    def get_rank(n, m, p):
        if n == 1 and m == 1 and p == 1:
            return 1
        if n == 1 and m == 1 and p == 2:
            return 2
        if n == 1 and m == 2 and p == 1:
            return 2
        if n == 1 and m == 2 and p == 2:
            return 4
        if n == 2 and m == 1 and p == 1:
            return 2
        if n == 2 and m == 1 and p == 2:
            return 4
        if n == 2 and m == 2 and p == 1:
            return 4
        if n == 2 and m == 2 and p == 2:
            return 7 # Strassen
        # Fallback to schoolbook
        return n * m * p

    # Define the split
    splits = [1, 2, 2]
    
    # Total rank
    total_rank = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                total_rank += get_rank(splits[i], splits[k], splits[j])
                
    # Initialize U, V, W
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    # Helper to get local indices
    def get_local_indices(size, split_sizes):
        # Returns list of (start, end) for each block
        indices = []
        current = 0
        for s in split_sizes:
            indices.append((current, current + s))
            current += s
        return indices

    row_splits = get_local_indices(5, splits)
    col_splits = get_local_indices(5, splits)
    
    for i in range(3):
        r_i_start, r_i_end = row_splits[i]
        for j in range(3):
            c_j_start, c_j_end = col_splits[j]
            for k in range(3):
                r_k_start, r_k_end = row_splits[k] # A's col k is B's row k
                # Sub-problem: A[r_i, r_k] * B[r_k, c_j]
                # A part: rows r_i_start..r_i_end, cols r_k_start..r_k_end
                # B part: rows r_k_start..r_k_end, cols c_j_start..c_j_end
                # C part: rows r_i_start..r_i_end, cols c_j_start..c_j_end
                
                n_sub = r_i_end - r_i_start
                m_sub = r_k_end - r_k_start
                p_sub = c_j_end - c_j_start
                
                rank_sub = get_rank(n_sub, m_sub, p_sub)
                
                # Construct sub-decomposition
                # For simplicity, we use schoolbook for all except 2x2x2 where we use Strassen
                # This is a valid, though not optimal, decomposition for the sub-blocks
                # We will use Strassen for 2x2x2 to save rank.
                
                if n_sub == 2 and m_sub == 2 and p_sub == 2:
                    # Strassen decomposition
                    # A = [[a11, a12], [a21, a22]]
                    # B = [[b11, b12], [b21, b22]]
                    # M1 = (a11+a22)(b11+b22)
                    # M2 = (a21+a22)b11
                    # M3 = a11(b12-b22)
                    # M4 = a22(b21-b11)
                    # M5 = (a11+a12)b22
                    # M6 = (a21-a11)(b11+b12)
                    # M7 = (a12-a22)(b21+b22)
                    # C11 = M1 + M4 - M5 + M7
                    # C12 = M3 + M5
                    # C21 = M2 + M4
                    # C22 = M1 - M2 + M3 + M6
                    
                    # We need to add these to U, V, W
                    # U indices: (row, col) in A
                    # V indices: (row, col) in B
                    # W indices: (row, col) in C
                    
                    # M1
                    # U: a11, a22
                    # V: b11, b22
                    # W: c11, c22
                    U[r_i_start][col] = 1
                    U[r_i_start+1][col] = 1
                    V[r_k_start][col] = 1
                    V[r_k_start+1][col] = 1
                    W[c_j_start][col] = 1
                    W[c_j_start+1][col] = 1
                    col += 1
                    
                    # M2
                    # U: a21, a22
                    # V: b11
                    # W: c21, c22
                    U[r_i_start+1][col] = 1
                    U[r_i_start+1][col] = 1 # Wait, a21 and a22 are both in row 1 of sub-block?
                    # Sub-block A:
                    # Row 0: a11, a12
                    # Row 1: a21, a22
                    # So a21 is at (r_i_start+1, r_k_start), a22 is at (r_i_start+1, r_k_start+1)
                    # M2 = (a21+a22)b11
                    # U should have 1 at a21 and a22.
                    U[r_i_start+1][col] = 1
                    U[r_i_start+1][col] = 1 # This is wrong, U is a matrix.
                    # U[row, col] = coeff.
                    # So U[r_i_start+1, col] = 1 for a21?
                    # a21 is at (r_i_start+1, r_k_start).
                    # a22 is at (r_i_start+1, r_k_start+1).
                    # So U[r_i_start+1, col] = 1 covers a21?
                    # No, U is (n*m, rank).
                    # Index is (row*m + col).
                    # So a21 is at (r_i_start+1)*m + r_k_start.
                    # a22 is at (r_i_start+1)*m + r_k_start+1.
                    
                    # Let's rewrite carefully.
                    # Reset col for this sub-block? No, col is global.
                    # I need to be precise.
                    
                    # M1: (a11+a22)(b11+b22)
                    # U: a11, a22
                    idx_a11 = (r_i_start) * m + r_k_start
                    idx_a22 = (r_i_start+1) * m + r_k_start+1
                    idx_b11 = (r_k_start) * p + c_j_start
                    idx_b22 = (r_k_start+1) * p + c_j_start+1
                    idx_c11 = (c_j_start) * n + r_i_start
                    idx_c22 = (c_j_start+1) * n + r_i_start+1
                    
                    U[idx_a11][col] = 1
                    U[idx_a22][col] = 1
                    V[idx_b11][col] = 1
                    V[idx_b22][col] = 1
                    W[idx_c11][col] = 1
                    W[idx_c22][col] = 1
                    col += 1
                    
                    # M2: (a21+a22)b11
                    idx_a21 = (r_i_start+1) * m + r_k_start
                    # U: a21, a22
                    U[idx_a21][col] = 1
                    U[idx_a22][col] = 1
                    V[idx_b11][col] = 1
                    # W: c21, c22
                    idx_c21 = (c_j_start) * n + r_i_start+1
                    idx_c22 = (c_j_start+1) * n + r_i_start+1
                    W[idx_c21][col] = 1
                    W[idx_c22][col] = 1
                    col += 1
                    
                    # M3: a11(b12-b22)
                    idx_b12 = (r_k_start) * p + c_j_start+1
                    idx_b22 = (r_k_start+1) * p + c_j_start+1
                    # U: a11
                    U[idx_a11][col] = 1
                    # V: b12, -b22
                    V[idx_b12][col] = 1
                    V[idx_b22][col] = -1
                    # W: c12
                    idx_c12 = (c_j_start+1) * n + r_i_start
                    W[idx_c12][col] = 1
                    col += 1
                    
                    # M4: a22(b21-b11)
                    idx_b21 = (r_k_start+1) * p + c_j_start
                    idx_b11 = (r_k_start) * p + c_j_start
                    # U: a22
                    U[idx_a22][col] = 1
                    # V: b21, -b11
                    V[idx_b21][col] = 1
                    V[idx_b11][col] = -1
                    # W: c11, c21
                    idx_c11 = (c_j_start) * n + r_i_start
                    idx_c21 = (c_j_start) * n + r_i_start+1
                    W[idx_c11][col] = 1
                    W[idx_c21][col] = 1
                    col += 1
                    
                    # M5: (a11+a12)b22
                    idx_a12 = (r_i_start) * m + r_k_start+1
                    idx_b22 = (r_k_start+1) * p + c_j_start+1
                    # U: a11, a12
                    U[idx_a11][col] = 1
                    U[idx_a12][col] = 1
                    # V: b22
                    V[idx_b22][col] = 1
                    # W: -c11, c12
                    idx_c11 = (c_j_start) * n + r_i_start
                    idx_c12 = (c_j_start+1) * n + r_i_start
                    W[idx_c11][col] = -1
                    W[idx_c12][col] = 1
                    col += 1
                    
                    # M6: (a21-a11)(b11+b12)
                    idx_a21 = (r_i_start+1) * m + r_k_start
                    idx_a11 = (r_i_start) * m + r_k_start
                    idx_b11 = (r_k_start) * p + c_j_start
                    idx_b12 = (r_k_start) * p + c_j_start+1
                    # U: a21, -a11
                    U[idx_a21][col] = 1
                    U[idx_a11][col] = -1
                    # V: b11, b12
                    V[idx_b11][col] = 1
                    V[idx_b12][col] = 1
                    # W: c22
                    idx_c22 = (c_j_start+1) * n + r_i_start+1
                    W[idx_c22][col] = 1
                    col += 1
                    
                    # M7: (a12-a22)(b21+b22)
                    idx_a12 = (r_i_start) * m + r_k_start+1
                    idx_a22 = (r_i_start+1) * m + r_k_start+1
                    idx_b21 = (r_k_start+1) * p + c_j_start
                    idx_b22 = (r_k_start+1) * p + c_j_start+1
                    # U: a12, -a22
                    U[idx_a12][col] = 1
                    U[idx_a22][col] = -1
                    # V: b21, b22
                    V[idx_b21][col] = 1
                    V[idx_b22][col] = 1
                    # W: c11, c22
                    idx_c11 = (c_j_start) * n + r_i_start
                    idx_c22 = (c_j_start+1) * n + r_i_start+1
                    W[idx_c11][col] = 1
                    W[idx_c22][col] = 1
                    col += 1
                    
                else:
                    # Schoolbook
                    for u in range(n_sub):
                        for v in range(m_sub):
                            for w in range(p_sub):
                                idx_a = (r_i_start + u) * m + (r_k_start + v)
                                idx_b = (r_k_start + v) * p + (c_j_start + w)
                                idx_c = (c_j_start + w) * n + (r_i_start + u)
                                U[idx_a][col] = 1
                                V[idx_b][col] = 1
                                W[idx_c][col] = 1
                                col += 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
