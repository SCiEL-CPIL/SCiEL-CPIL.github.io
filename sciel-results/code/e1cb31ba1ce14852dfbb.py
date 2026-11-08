"""Matrix multiplication 6x6x6 - rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Use Kronecker product structure: 6 = 2 * 3
    # rank(6x6) <= rank(2x2) * rank(3x3)
    # Use Strassen (rank 7) for 2x2 and a rank-23 decomposition for 3x3
    # Total rank: 7 * 23 = 161
    
    # Strassen decomposition for 2x2 matrix multiplication
    # Each entry is (u_dict, v_dict, w_dict) where keys are (row, col)
    strassen_2x2 = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Winograd's rank-23 decomposition for 3x3 matrix multiplication
    # This is a known result from Winograd (1980)
    # The decomposition uses 23 bilinear forms
    # 
    # We need to construct U, V, W matrices of shape (9, 23) for 3x3
    # where U[i*m+j, k], V[j*p+k, l], W[k*n+i, m] encode the decomposition
    #
    # For 3x3: n=m=p=3, so U is (9,23), V is (9,23), W is (9,23)
    # Indexing: U[row*3+col, factor], V[row*3+col, factor], W[row*3+col, factor]
    
    # Winograd's rank-23 decomposition for 3x3:
    # Based on the paper "On the number of multiplications required to compute
    # the product of two 3x3 matrices" by Shmuel Winograd, 1980
    
    # The 23 products can be organized as follows:
    # We use a systematic construction based on linear combinations
    
    # Let me use a known explicit rank-23 decomposition for 3x3.
    # Reference: The decomposition can be found in various sources on
    # matrix multiplication algorithms.
    
    # Here's one construction (adapted from known results):
    # The key idea is to exploit the structure of the 27 trivial products
    # and find 4 linear dependencies to reduce to 23.
    
    # Actually, let me use a different, more straightforward approach:
    # Use the fact that 3x3 can be decomposed using the "Kronecker" trick
    # with 2x2 and 1x1, but that gives 7*9=63 which is worse than 27.
    
    # Let me just use the trivial rank-27 for 3x3 to be safe, giving
    # total rank 7*27 = 189. But that's not an improvement.
    
    # Let me try to construct the rank-23 decomposition properly.
    # 
    # Winograd's decomposition for 3x3 uses the following 23 bilinear forms:
    # (I'll use a known explicit construction)
    
    # Actually, for correctness and to avoid errors, let me use a 
    # systematic approach: construct the rank-23 decomposition using
    # the known algorithm from Winograd's paper.
    
    # The decomposition is complex. Let me use a simpler but still
    # improved approach: use the Kronecker product of Strassen(2x2) 
    # with a better 3x3 decomposition.
    
    # For now, let me use the trivial 3x3 (rank 27) and accept 189,
    # but structure the code differently to allow future improvement.
    
    # Actually, let me try a completely different approach: use the
    # "decompose and conquer" with 2x3 blocks.
    
    # 6x6 can be seen as 2x3 blocks: 3 blocks of 2x3 in each dimension
    # No, that doesn't work cleanly.
    
    # Let me just implement the Kronecker product properly with
    # Strassen(2x2) and trivial(3x3) for now, giving rank 189.
    # But let me structure it so we can easily swap in a better 3x3.
    
    # Trivial decomposition for 3x3 (rank 27):
    trivial_3x3 = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                trivial_3x3.append((
                    {(i, j): 1},
                    {(j, k): 1},
                    {(k, i): 1}
                ))
    
    # Use Kronecker product of strassen_2x2 and trivial_3x3
    # Total rank: 7 * 27 = 189
    
    rank = len(strassen_2x2) * len(trivial_3x3)
    
    # Build U, V, W matrices
    # U has shape (n*m, rank) = (36, 189)
    # V has shape (m*p, rank) = (36, 189)  
    # W has shape (p*n, rank) = (36, 189)
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Kronecker product: for each pair (s2, s3) where s2 is a Strassen term
    # and s3 is a trivial 3x3 term, we create one rank-1 component
    # The indices are combined via Kronecker product
    
    for idx2, (u2, v2, w2) in enumerate(strassen_2x2):
        for idx3, (u3, v3, w3) in enumerate(trivial_3x3):
            col = idx2 * len(trivial_3x3) + idx3
            
            # U: combine u2 (2x2) and u3 (3x3) via Kronecker product
            # u2 has entries at positions (pi, pj) in 2x2
            # u3 has entries at positions (qi, qj) in 3x3
            # Combined position: (pi*3+qi, pj*3+qj) in 6x6
            for (pi, pj), val2 in u2.items():
                for (qi, qj), val3 in u3.items():
                    r = pi * 3 + qi
                    c = pj * 3 + qj
                    U[r * m + c, col] += val2 * val3
            
            # V: combine v2 (2x2) and v3 (3x3) via Kronecker product
            for (pi, pj), val2 in v2.items():
                for (qi, qj), val3 in v3.items():
                    r = pi * 3 + qi
                    c = pj * 3 + qj
                    V[r * p + c, col] += val2 * val3
            
            # W: combine w2 (2x2) and w3 (3x3) via Kronecker product
            # w2 has entries at (k2, i2) in 2x2
            # w3 has entries at (k3, i3) in 3x3
            # Combined: (k2*3+k3, i2*3+i3) in 6x6
            for (k2, i2), val2 in w2.items():
                for (k3, i3), val3 in w3.items():
                    k = k2 * 3 + k3
                    i = i2 * 3 + i3
                    W[k * n + i, col] += val2 * val3
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
