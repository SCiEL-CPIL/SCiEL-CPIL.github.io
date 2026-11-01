import numpy as np
import time

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-21 decomposition for M_{2,2,6}.
    This is the minimal possible rank for this tensor.
    """
    n, m, p = DIMENSIONS  # 2, 2, 6
    
    # Rank is 21 (3 blocks of 2x2x2 Strassen rank 7)
    rank = 21
    
    # Initialize matrices with zeros
    # U: (n*m, rank) = (4, 21)
    U = [[0] * rank for _ in range(n * m)]
    # V: (m*p, rank) = (12, 21)
    V = [[0] * rank for _ in range(m * p)]
    # W: (p*n, rank) = (12, 21)
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products for 2x2 matrix multiplication
    # Each product is defined by sparse vectors u, v and matrix w
    # u is in R^{n*m}, v is in R^{m*p}, w is in R^{p*n}
    # We represent them as dictionaries {index: value}
    
    # Indices for 2x2x2 block:
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Note: The local indices in the strassen list below are 0,1,2,3
    # corresponding to the 4 elements of the 2x2 matrices.
    
    # Standard Strassen coefficients for 2x2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # Let's map these to the dictionary format used in the seed.
    # The seed used a specific mapping. I will replicate the logic to ensure correctness.
    
    strassen_blocks = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
        # M3
        ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
        # M4
        ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
    ]
    
    # The matrix is 2x2 * 2x6.
    # We view the 2x6 matrix as three 2x2 blocks horizontally.
    # Block b (b=0,1,2) corresponds to columns 2b and 2b+1 of the 2x6 matrix.
    # The 2x2 result matrix C = A * B.
    # C is 2x2.
    # Wait, A is 2x2, B is 2x6. C is 2x6.
    # So we are computing C[:, 2b:2b+2] = A * B[:, 2b:2b+2].
    # This is a standard 2x2 * 2x2 multiplication for each block b.
    # The result C is 2x6.
    # The W matrix in the decomposition has shape (p*n, rank) = (12, 21).
    # The indices of W are k*n + i, where k is the row of C (0..p-1) and i is the col of C (0..n-1).
    # Wait, standard matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p.
    # Here n=2, m=2, p=6.
    # C is 2 x 6.
    # The index of C is (i, k) where i is row (0..n-1) and k is col (0..p-1).
    # The problem statement says: "output factor follows the official trace convention W[k*n+i,r]".
    # This means the index in W is `idx = k*n + i`.
    # Here k corresponds to the column of the output C (0..5), and i corresponds to the row of the output C (0..1).
    # So for block b (0,1,2), the columns of C are 2b and 2b+1.
    # So for a local result element (i, k_local) in the 2x2 block:
    # i is the row of C (0 or 1).
    # k_local is the local column (0 or 1).
    # The global column k is 2b + k_local.
    # The global index in W is k * n + i = (2b + k_local) * 2 + i.
    
    r = 0
    for b in range(3):
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen_blocks[t]
            
            # Fill U
            # U has shape (n*m, rank) = (4, 21).
            # The indices in u_coeffs are 0,1,2,3 corresponding to the 4 elements of A (2x2).
            # A is n x m = 2 x 2.
            # Index mapping for A: usually row-major or col-major?
            # In the seed code:
            # u_coeffs keys are 0,1,2,3.
            # It just assigns U[idx][r] = val.
            # We assume the standard ordering of the input tensor factors matches the index.
            # For 2x2 matrix A, indices 0,1,2,3 typically correspond to (0,0), (0,1), (1,0), (1,1) or similar.
            # As long as it's consistent with the tensor definition, it works.
            # The seed code used this and got a valid score, so we stick to it.
            
            for idx, val in u_coeffs.items():
                U[idx][r] = val
            
            # Fill V
            # V has shape (m*p, rank) = (12, 21).
            # v_coeffs keys are 0,1,2,3 corresponding to the 4 elements of the 2x2 block of B.
            # B is m x p = 2 x 6.
            # Block b uses columns 2b and 2b+1.
            # Local indices 0,1,2,3 in v_coeffs correspond to the 4 elements of B[:, 2b:2b+2].
            # The seed code mapped:
            # l=0 -> v_idx = 0*p + 2*b
            # l=1 -> v_idx = 0*p + 2*b + 1
            # l=2 -> v_idx = 1*p + 2*b
            # l=3 -> v_idx = 1*p + 2*b + 1
            # This corresponds to row 0 col 0, row 0 col 1, row 1 col 0, row 1 col 1 of the block.
            # This is consistent with row-major flattening of the 2x2 block.
            
            for l, val in v_coeffs.items():
                if l == 0:
                    v_idx = 0 * p + 2 * b
                elif l == 1:
                    v_idx = 0 * p + 2 * b + 1
                elif l == 2:
                    v_idx = 1 * p + 2 * b
                elif l == 3:
                    v_idx = 1 * p + 2 * b + 1
                V[v_idx][r] = val
            
            # Fill W
            # W has shape (p*n, rank) = (12, 21).
            # w_coeffs keys are (i, k_local) where i is row of C, k_local is local col of C.
            # i in {0,1}, k_local in {0,1}.
            # Global column k = 2b + k_local.
            # Index in W is k*n + i.
            
            for (i, k_local), val in w_coeffs.items():
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx][r] = val
            
            r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
