import numpy as np
import time

# EVOLVE-BLOCK-START
_n, _m, _p = 4, 4, 4
_rank = 49

# Strassen terms for 2x2 multiplication:
# Each term is (A_coeff, B_coeff, C_coeff)
# A_coeff: 2x2 matrix of coefficients for A
# B_coeff: 2x2 matrix of coefficients for B
# C_coeff: 2x2 matrix of coefficients for C
# The term is: sum(A_coeff * A) * sum(B_coeff * B) -> adds to C via C_coeff
# We represent coefficients as sparse dicts {(row, col): val}

def _make_sparse_2x2(coeffs):
    """Convert a 2x2 list to a sparse dict {(r,c): val}"""
    d = {}
    for r in range(2):
        for c in range(2):
            if coeffs[r][c] != 0:
                d[(r,c)] = coeffs[r][c]
    return d

# Standard Strassen terms
# M1 = (A11+A22)(B11+B22)
# M2 = (A21+A22)B11
# M3 = A11(B12-B22)
# M4 = A22(B21-B11)
# M5 = (A11+A12)B22
# M6 = (A21-A11)(B11+B12)
# M7 = (A12-A22)(B21+B22)

_strassen_terms = [
    # (A_mask, B_mask, C_mask)
    (
        _make_sparse_2x2([[1,0],[0,1]]),
        _make_sparse_2x2([[1,0],[0,1]]),
        _make_sparse_2x2([[1,0],[0,1]])
    ),
    (
        _make_sparse_2x2([[0,0],[1,1]]),
        _make_sparse_2x2([[1,0],[0,0]]),
        _make_sparse_2x2([[0,0],[1,-1]])
    ),
    (
        _make_sparse_2x2([[1,0],[0,0]]),
        _make_sparse_2x2([[0,1],[0,-1]]),
        _make_sparse_2x2([[0,1],[0,1]])
    ),
    (
        _make_sparse_2x2([[0,0],[0,1]]),
        _make_sparse_2x2([[0,-1],[1,0]]),
        _make_sparse_2x2([[1,0],[0,1]])
    ),
    (
        _make_sparse_2x2([[1,1],[0,0]]),
        _make_sparse_2x2([[0,0],[0,1]]),
        _make_sparse_2x2([[-1,0],[0,1]])
    ),
    (
        _make_sparse_2x2([[0,0],[1,-1]]),
        _make_sparse_2x2([[1,1],[0,0]]),
        _make_sparse_2x2([[0,0],[0,1]])
    ),
    (
        _make_sparse_2x2([[0,1],[-1,0]]),
        _make_sparse_2x2([[0,0],[1,1]]),
        _make_sparse_2x2([[1,0],[0,0]])
    )
]

def _build_decomposition():
    """
    Recursively build the rank-49 decomposition using Strassen.
    The base case is 1x1 multiplication (rank 1).
    """
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # We build it for n=m=p=4.
    # We use a recursive helper that returns the factors for a given size.
    
    def build_factors(size):
        """
        Returns (U, V, W) for size x size matrix multiplication.
        U: (size*size, rank)
        V: (size*size, rank)
        W: (size*size, rank)
        """
        if size == 1:
            # 1x1 multiplication: A*B = C
            # U = [[1]], V = [[1]], W = [[1]]
            U = [[1]]
            V = [[1]]
            W = [[1]]
            return U, V, W
        
        half = size // 2
        # Recursively build for half size
        # We need to map indices from the full matrix to the sub-blocks.
        # For Strassen, we split A, B, C into 4 blocks of size half x half.
        
        # We will construct the factors by combining the recursive factors.
        # The rank for size 'size' is 7 * (rank of size 'half').
        
        # Get recursive factors
        U_rec, V_rec, W_rec = build_factors(half)
        rank_rec = len(U_rec[0])
        
        # Total rank
        total_rank = 7 * rank_rec
        
        # Initialize full factors
        U_full = [[0] * total_rank for _ in range(size * size)]
        V_full = [[0] * total_rank for _ in range(size * size)]
        W_full = [[0] * total_rank for _ in range(size * size)]
        
        # Helper to get index in full matrix from block (br, bc) and local (lr, lc)
        def full_idx(block_row, block_col, local_row, local_col):
            return (block_row * half + local_row) * size + (block_col * half + local_col)
        
        # Iterate over the 7 Strassen terms
        for t in range(7):
            A_mask, B_mask, C_mask = _strassen_terms[t]
            
            # For each recursive rank index
            for r_rec in range(rank_rec):
                r_full = t * rank_rec + r_rec
                
                # Update U (Left factor)
                # A_mask tells us which blocks of A contribute.
                # A_mask is a 2x2 sparse dict: {(br, bc): coeff}
                for (br, bc), coeff_A in A_mask.items():
                    # For each element in the recursive U factor
                    # U_rec[i_rec, r_rec] corresponds to element i_rec in the half-sized A
                    # We need to map i_rec to the full A index.
                    # i_rec is an index in a half x half matrix.
                    # local_row = i_rec // half, local_col = i_rec % half
                    for i_rec in range(half * half):
                        val = U_rec[i_rec][r_rec]
                        if val != 0:
                            lr = i_rec // half
                            lc = i_rec % half
                            idx_full = full_idx(br, bc, lr, lc)
                            U_full[idx_full][r_full] += coeff_A * val
                
                # Update V (Right factor)
                for (br, bc), coeff_B in B_mask.items():
                    for i_rec in range(half * half):
                        val = V_rec[i_rec][r_rec]
                        if val != 0:
                            lr = i_rec // half
                            lc = i_rec % half
                            idx_full = full_idx(br, bc, lr, lc)
                            V_full[idx_full][r_full] += coeff_B * val
                
                # Update W (Output factor)
                # W_mask tells us how the product contributes to C blocks.
                # The product is (sum A_mask * A) * (sum B_mask * B)
                # This product is a scalar (in the recursive sense, it's a matrix product of size half).
                # The recursive W factor W_rec[i_rec, r_rec] tells us how the recursive output C_rec
                # is formed from the recursive products.
                # However, in our formulation, W is the coefficient for the output C.
                # The term is: (A_part * B_part) contributes to C.
                # Specifically, the product P_t is a matrix of size half x half.
                # P_t is added to C blocks according to C_mask.
                # C_mask is a 2x2 sparse dict: {(br, bc): coeff_C}
                # So, for each block (br, bc) in C, we add coeff_C * P_t.
                # P_t is represented by the recursive W factor? 
                # Wait, the recursive decomposition gives us factors for the *multiplication* of the blocks.
                # The "product" in the Strassen step is a single scalar multiplication of two linear combinations?
                # No, Strassen multiplies two *matrices* (the linear combinations of blocks).
                # So P_t is a matrix of size half x half.
                # The recursive factors U_rec, V_rec, W_rec decompose the multiplication of two half-sized matrices.
                # So, the contribution of term t to the final C is:
                # Sum over r_rec of (U_rec * V_rec) ... wait.
                # The tensor decomposition is: T = sum_t U_t V_t W_t
                # For the Strassen step, the "multiplication" is of two half-sized matrices.
                # So the recursive factors apply to the *elements* of the half-sized matrices.
                # The term t produces a product matrix P_t of size half x half.
                # This P_t is then placed into the C blocks according to C_mask.
                # How is P_t constructed?
                # P_t = A_t * B_t, where A_t and B_t are the linear combinations.
                # The recursive decomposition provides the factors for the multiplication of A_t and B_t.
                # So, the elements of P_t are given by the recursive W factor?
                # Actually, the recursive W factor maps the "rank indices" to the output elements of the half-sized product.
                # So, for each output element (lr, lc) of P_t, its value is sum_r_rec W_rec[(lr,lc), r_rec] * (product term).
                # And this value is added to C at block (br, bc) and local position (lr, lc).
                
                for (br, bc), coeff_C in C_mask.items():
                    for i_rec in range(half * half):
                        val = W_rec[i_rec][r_rec]
                        if val != 0:
                            lr = i_rec // half
                            lc = i_rec % half
                            idx_full = full_idx(br, bc, lr, lc)
                            W_full[idx_full][r_full] += coeff_C * val
        
        return U_full, V_full, W_full

    U, V, W = build_factors(4)
    return U, V, W

# Build once at import time
_U, _V, _W = _build_decomposition()

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    Returns precomputed factor matrices.
    """
    # Explicit deadline guard (per directive, though construction is precomputed)
    # If this were a search, we would check time here.
    return {"U": _U, "V": _V, "W": _W}
# EVOLVE-BLOCK-END
