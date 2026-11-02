import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 2x2 * 2x8 matrix multiplication.
    The decomposition is built by tiling 4 instances of the 2x2x2 Strassen algorithm.
    Rank = 4 * 7 = 28.
    """
    n, m, p = DIMENSIONS
    
    # The 2x2x2 Strassen decomposition
    # We represent each rank-1 term (slice) as (u_slice, v_slice, w_slice)
    # u_slice: vector of length n*m = 4 (indices i*m + j)
    # v_slice: vector of length m*p = 4 (indices j*p + k)
    # w_slice: vector of length p*n = 4 (indices k*n + i)
    
    # Strassen's 7 products for 2x2 matrices:
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = A*B = [[C00, C01], [C10, C11]]
    # P1 = (a + d) * (e + h)
    # P2 = (c + d) * e
    # P3 = a * (f - h)
    # P4 = d * (g - e)
    # P5 = (a + b) * h
    # P6 = (c - a) * (e + f)
    # P7 = (b - d) * (g + h)
    #
    # C00 = P1 - P5 + P4 + P7
    # C01 = P3 + P5
    # C10 = P2 + P6
    # C11 = P1 - P2 + P3 + P6
    
    # We define the coefficients for u, v, w for each of the 7 ranks.
    # Indexing:
    # u: [a, b, c, d] -> indices 0, 1, 2, 3
    # v: [e, f, g, h] -> indices 0, 1, 2, 3
    # w: [C00, C01, C10, C11] -> indices 0, 1, 2, 3 (Note: W[k*n+i] maps C[i,k] to k*n+i? 
    #      Task says W[k*n+i, r]. 
    #      Output C is 2x2. C[i,k]. 
    #      Index in W is k*n + i. 
    #      So W[0, r] corresponds to C[0,0] (k=0, i=0).
    #      W[1, r] corresponds to C[0,1] (k=1, i=0)? No, k=0, i=1 -> 0*2+1=1.
    #      Let's check the trace convention carefully.
    #      "W[k*n+i, r]".
    #      Output matrix C has dimensions n x p? No, A is n x m, B is m x p. C is n x p.
    #      Here n=2, p=2 for the block.
    #      C[i, k] where i is row (0..n-1), k is col (0..p-1).
    #      The problem statement says: "output factor follows the official trace convention W[k*n+i, r]".
    #      This usually means the index in the W vector is $k \cdot n + i$.
    #      So for C[0,0]: k=0, i=0 -> idx 0.
    #      For C[0,1]: k=1, i=0 -> idx 1.
    #      For C[1,0]: k=0, i=1 -> idx 2.
    #      For C[1,1]: k=1, i=1 -> idx 3.
    #      Wait, standard row-major for C[i,k] would be i*p+k.
    #      Here it is k*n+i. This is column-major storage of C, or row-major of C^T.
    #      Let's verify with the seed program's logic.
    #      Seed: W[k_global * n + i][col].
    #      In the seed, for P1 (r0), w_coeffs[0] = [[1, 0], [0, 1]].
    #      i=0, k_local=0 -> val=1. W[0*2+0] = W[0] = 1. (C[0,0] += 1)
    #      i=1, k_local=1 -> val=1. W[1*2+1] = W[3] = 1. (C[1,1] += 1)
    #      This matches C00+=1, C11+=1.
    #      So W index = k*n + i.
    #      My w_slice indices:
    #      C[0,0] -> idx 0
    #      C[0,1] -> idx 1  (k=1, i=0)
    #      C[1,0] -> idx 2  (k=0, i=1)
    #      C[1,1] -> idx 3  (k=1, i=1)
    
    # Let's re-verify the Strassen formulas against this indexing.
    # P1: C00+=1, C11+=1.
    #   w[0]=1, w[3]=1.
    # P2: C10+=1, C11-=-1. (C10 is row 1, col 0. C11 is row 1, col 1).
    #   C[1,0] -> idx k=0, i=1 -> 0*2+1=2.
    #   C[1,1] -> idx k=1, i=1 -> 1*2+1=3.
    #   So w[2]=1, w[3]=-1.
    # P3: C01+=1, C11+=1.
    #   C[0,1] -> idx k=1, i=0 -> 1*2+0=2? No, 1*2+0=2.
    #   C[1,1] -> idx 3.
    #   So w[2]=1, w[3]=1.
    #   Wait, C[0,1] is row 0, col 1. k=1, i=0. Index = 1*2+0 = 2.
    #   C[1,1] is row 1, col 1. k=1, i=1. Index = 1*2+1 = 3.
    #   So P3 adds to idx 2 and 3.
    # P4: C00+=1, C10+=1.
    #   C[0,0] -> idx 0.
    #   C[1,0] -> idx 2.
    #   So w[0]=1, w[2]=1.
    # P5: C00-=-1, C01+=1.
    #   C[0,0] -> idx 0.
    #   C[0,1] -> idx 2.
    #   So w[0]=-1, w[2]=1.
    # P6: C11+=1.
    #   C[1,1] -> idx 3.
    #   So w[3]=1.
    # P7: C00+=1.
    #   C[0,0] -> idx 0.
    #   So w[0]=1.
    
    # Let's construct the slices.
    # u_slice (A coeffs): [a, b, c, d]
    # v_slice (B coeffs): [e, f, g, h]
    
    strassen_slices = []
    
    # P1: (a+d)(e+h)
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1] # C00(0), C11(3)
    strassen_slices.append((u1, v1, w1))
    
    # P2: (c+d)e
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, -1] # C10(2), C11(3)
    strassen_slices.append((u2, v2, w2))
    
    # P3: a(f-h)
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    w3 = [0, 0, 1, 1] # C01(2), C11(3)
    strassen_slices.append((u3, v3, w3))
    
    # P4: d(g-e)
    u4 = [0, 0, 0, 1]
    v4 = [-1, 0, 1, 0]
    w4 = [1, 0, 1, 0] # C00(0), C10(2)
    strassen_slices.append((u4, v4, w4))
    
    # P5: (a+b)h
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [-1, 0, 1, 0] # C00(0), C01(2)
    strassen_slices.append((u5, v5, w5))
    
    # P6: (c-a)(e+f)
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 0, 1] # C11(3)
    strassen_slices.append((u6, v6, w6))
    
    # P7: (b-d)(g+h)
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    w7 = [1, 0, 0, 0] # C00(0)
    strassen_slices.append((u7, v7, w7))
    
    num_blocks = 4
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for block in range(num_blocks):
        for r_idx, (u_slice, v_slice, w_slice) in enumerate(strassen_slices):
            col = block * rank_per_block + r_idx
            
            # Fill U
            for idx, val in enumerate(u_slice):
                if val != 0:
                    U[idx][col] = val
                    
            # Fill V
            # v_slice indices correspond to j*p + k_local.
            # In a 2x2 block, p=2.
            # v_slice is [e, f, g, h] -> indices 0,1,2,3.
            # Global index for V: j * p + k_global.
            # Here, the block is at columns k_global = block*2 ... block*2+1.
            # The slice index idx (0..3) maps to local j, k_local.
            # idx = j * 2 + k_local.
            # Global V index = j * p + (block * 2 + k_local).
            # Note: p is the full width (8), not the block width (2).
            for idx, val in enumerate(v_slice):
                if val != 0:
                    j = idx // 2
                    k_local = idx % 2
                    k_global = block * 2 + k_local
                    V_idx = j * p + k_global
                    V[V_idx][col] = val
                    
            # Fill W
            # w_slice indices correspond to k_local * n + i.
            # n=2.
            # idx = k_local * 2 + i.
            # Global W index: k_global * n + i.
            # k_global = block * 2 + k_local.
            for idx, val in enumerate(w_slice):
                if val != 0:
                    k_local = idx // 2
                    i = idx % 2
                    k_global = block * 2 + k_local
                    W_idx = k_global * n + i
                    W[W_idx][col] = val
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
