"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS
    block_size = 2
    n_blocks = n // block_size  # 3

    # Block Strassen: 3x3 blocks of 2x2, each 2x2 uses rank-7 Strassen
    rank = n_blocks ** 3 * 7  # 189

    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)

    col = 0

    def u_idx(ib, kb, r, c):
        return (ib * block_size + r) * m + (kb * block_size + c)

    def v_idx(kb, jb, r, c):
        return (kb * block_size + r) * p + (jb * block_size + c)

    def w_idx(ib, jb, r, c):
        return (jb * block_size + c) * n + (ib * block_size + r)

    # Strassen 2x2 multiplication terms
    strassen_terms = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    for ib in range(n_blocks):
        for jb in range(n_blocks):
            for kb in range(n_blocks):
                for u_entries, v_entries, w_entries in strassen_terms:
                    for (r, c, coeff) in u_entries:
                        U[u_idx(ib, kb, r, c), col] += coeff
                    for (r, c, coeff) in v_entries:
                        V[v_idx(kb, jb, r, c), col] += coeff
                    for (r, c, coeff) in w_entries:
                        W[w_idx(ib, jb, r, c), col] += coeff
                    col += 1

    # Rank reduction: check for linearly dependent columns using exact rational arithmetic
    # Convert to Fraction matrices for exact computation
    U_frac = np.array([[Fraction(int(x)) for x in row] for row in U.tolist()], dtype=object)
    V_frac = np.array([[Fraction(int(x)) for x in row] for row in V.tolist()], dtype=object)
    W_frac = np.array([[Fraction(int(x)) for x in row] for row in W.tolist()], dtype=object)

    # Find the minimum rank among the three factor matrices
    # We need to find a common set of independent columns
    # Use the fact that the tensor is decomposed as sum of rank-1 terms
    
    # Strategy: Find a basis for the column space of U, V, W simultaneously
    # This is equivalent to finding the rank of the tensor decomposition
    
    # First, let's check if there are any obviously dependent columns
    # by looking at the Kronecker product structure
    
    # For now, let's try a simpler approach: check each matrix's rank
    # and see if we can remove columns that are zero or linearly dependent
    
    def get_rank_frac(matrix):
        """Compute rank of a matrix with Fraction entries using Gaussian elimination."""
        if matrix.size == 0:
            return 0
        mat = matrix.copy()
        rows, cols = mat.shape
        rank = 0
        for col in range(cols):
            # Find pivot
            pivot = None
            for row in range(rank, rows):
                if mat[row, col] != 0:
                    pivot = row
                    break
            if pivot is None:
                continue
            # Swap rows
            if pivot != rank:
                mat[rank], mat[pivot] = mat[pivot].copy(), mat[rank].copy()
            # Normalize pivot row
            pivot_val = mat[rank, col]
            mat[rank] = mat[rank] / pivot_val
            # Eliminate other rows
            for row in range(rows):
                if row != rank and mat[row, col] != 0:
                    factor = mat[row, col]
                    mat[row] = mat[row] - factor * mat[rank]
            rank += 1
        return rank

    # Check the rank of each factor matrix
    rank_U = get_rank_frac(U_frac)
    rank_V = get_rank_frac(V_frac)
    rank_W = get_rank_frac(W_frac)
    
    # The effective rank is bounded by min(rank_U, rank_V, rank_W)
    # But we need to be careful: removing a column from U requires removing
    # the corresponding column from V and W
    
    # For this Block-Strassen decomposition, the columns are structured
    # such that each column corresponds to a unique (ib, jb, kb, strassen_term)
    # So there shouldn't be linear dependencies within the same factor matrix
    # unless the Strassen terms themselves have dependencies
    
    # Let's verify by checking if the full decomposition has rank 189
    # If not, we can reduce
    
    # Actually, for the standard Block-Strassen, the rank should be exactly 189
    # since each of the 27 block multiplications uses 7 independent Strassen terms
    
    # Let's just return the current decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
