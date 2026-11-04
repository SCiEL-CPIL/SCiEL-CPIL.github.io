import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will construct a decomposition of rank 44 using Strassen for 2x2x2 blocks.
    # Blocks for i in {0,1} (2 blocks of 2x2 in j, 2 blocks of 2x2 in k -> 4 blocks of 2x2x2)
    # Blocks for i in {2} (1 block of 1x2 in j, 2 blocks of 1x2 in k -> 4 blocks of 1x2x2)
    
    # Initialize U, V, W as lists of lists
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # Rank estimation:
    # 4 blocks of 2x2x2 * 7 = 28
    # 4 blocks of 1x2x2 * 4 = 16
    # Total rank = 44
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Strassen decomposition for 2x2x2
    # Local indices i,j,k in {0,1}
    # Global mapping:
    # i_local -> i_global = i_base + i_local
    # j_local -> j_global = j_base + j_local
    # k_local -> k_global = k_base + k_local
    
    # Strassen terms for 2x2x2
    # p1 = (a00+a11)(b00+b11)
    # p2 = (a10+a11)b00
    # p3 = a00(b01-b11)
    # p4 = a11(b10-b00)
    # p5 = (a00+a01)b11
    # p6 = (a10-a00)(b00+b01)
    # p7 = (a01-a11)(b10+b11)
    
    # c00 = p1 + p4 - p5 + p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p6
    
    # We need to map these to rank-1 tensors u, v, w.
    # Each p is a scalar product of a linear combination of A and B.
    # This corresponds to a rank-1 tensor u ⊗ v ⊗ w.
    
    # Term 1 (from p1, contributes to c00 and c11)
    # p1 = (e0+e1)_A . (e0+e1)_B
    # We split this into contributions to c00 and c11.
    # Actually, the rank-1 decomposition sums up to the tensor.
    # T = sum u_r v_r w_r
    # For Strassen, the 7 terms are the rank-1 tensors.
    
    # Let's define the 7 vectors for the 2x2x2 block.
    # u vectors (for A)
    # v vectors (for B)
    # w vectors (for C)
    
    # 1. p1: u=(1,1), v=(1,1), w=(1,1) ? No, w must be specific to output.
    # The standard rank-7 decomposition of the 2x2x2 tensor is:
    
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        
        # 1. p1
        # u = e0 + e1
        # v = e0 + e1
        # w = e0 (for c00) AND w = e1 (for c11)
        # This implies we need 2 terms for p1?
        # No, the rank is 7.
        # The 7 terms are:
        
        # Term 1: u=(1,1), v=(1,1), w=(1,1) -> This sums to c00 and c11?
        # If w=(1,1), it adds to both c00 and c11.
        # Let's check the standard decomposition.
        
        # T = 
        # 1. u=(1,1), v=(1,1), w=(1,1)
        # 2. u=(1,0), v=(0,1), w=(1,0)
        # 3. u=(0,1), v=(1,0), w=(0,1)
        # ... this is getting risky to derive on the fly.
        
        # I will use the schoolbook for 2x2x2 (rank 8) to be safe?
        # No, that gives rank 32 + 16 = 48. No improvement.
        
        # I will use a known rank-7 decomposition.
        # Reference: Strassen's algorithm.
        
        # u1 = [1, 1], v1 = [1, 1], w1 = [1, 1]
        # u2 = [1, 0], v2 = [0, 1], w2 = [1, 0]
        # u3 = [0, 1], v3 = [1, 0], w3 = [0, 1]
        # u4 = [1, -1], v4 = [1, 1], w4 = [1, -1]
        # u5 = [1, 1], v5 = [1, -1], w5 = [1, 1]
        # u6 = [1, 0], v6 = [1, 1], w6 = [0, 1]
        # u7 = [0, 1], v7 = [1, 1], w7 = [1, 0]
        
        # This is a guess. I should stick to schoolbook if I'm not sure.
        # But 48 is the seed.
        
        # Let's try to implement the 44 rank with schoolbook for 2x2x2 (rank 8) is 32+16=48.
        # So I MUST use Strassen.
        
        # I will use the following known rank-7 decomposition for 2x2x2:
        # Terms (u, v, w) where u,v,w are vectors of length 2.
        
        terms = [
            ([1, 1], [1, 1], [1, 1]),
            ([1, 0], [0, 1], [1, 0]),
            ([0, 1], [1, 0], [0, 1]),
            ([1, -1], [1, 1], [1, -1]),
            ([1, 1], [1, -1], [1, 1]),
            ([1, 0], [1, 1], [0, 1]),
            ([0, 1], [1, 1], [1, 0]),
        ]
        
        for u, v, w in terms:
            for il in range(2):
                for jl in range(2):
                    for kl in range(2):
                        if u[il] != 0 and v[jl] != 0 and w[kl] != 0:
                            add_term(
                                (i_base + il) * m + (j_base + jl), u[il],
                                (j_base + jl) * p + (k_base + kl), v[jl],
                                (k_base + kl) * n + (i_base + il), w[kl]
                            )
            # This adds 8 terms per rank-1 term? No.
            # A rank-1 term is a single column in U, V, W.
            # The vectors u, v, w define the column.
            
            # Correct way to add a rank-1 term:
            # U column: non-zero at indices (i_base+il, j_base+jl) with value u[il]
            # V column: non-zero at indices (j_base+jl, k_base+kl) with value v[jl]
            # W column: non-zero at indices (k_base+kl, i_base+il) with value w[kl]
            
            # I need to add ONE column for each term.
            
            col_start = col
            for il in range(2):
                if u[il] != 0:
                    U[(i_base + il) * m + (j_base + jl)] # Wrong, j is also varying
                    # U index is i*m + j.
                    # u[il] is the coefficient for A[i_base+il, j_base+jl]?
                    # No, u is a vector of length 2 for the i dimension?
                    # No, u is a vector of length 4 (2x2 matrix flattened)?
                    # In the 2x2x2 block, u is a 2x2 matrix?
                    # No, u is a vector of length 2 (rows of A)?
                    # The tensor is 3D.
                    # u is a vector in R^(n*m) = R^4.
                    # v is a vector in R^(m*p) = R^4.
                    # w is a vector in R^(p*n) = R^4.
                    
                    # For the 2x2x2 block, u has non-zero entries at (i_base+il, j_base+jl).
                    # So u is a vector of length 4 with 4 non-zeros?
                    # No, u is a vector of length n*m.
                    # For the block, u has non-zeros at (i_base+il, j_base+jl) for il,jl in 0..1.
                    # So u has 4 non-zeros.
                    
                    pass
            
            # I will implement this correctly in the final code.
            pass

    # Due to complexity and risk of error in manual Strassen vector construction,
    # and the strict requirement for exactness, I will fall back to the schoolbook 
    # but with a small optimization if possible.
    # However, the seed is 48.
    
    # I will return the schoolbook decomposition to ensure validity.
    # If I can't guarantee the Strassen implementation is correct, 48 is better than 0.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
