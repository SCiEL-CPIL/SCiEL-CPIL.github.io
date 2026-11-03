import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = 4, 4, 5
    
    # We'll build a rank-44 decomposition:
    # - Rank 28 for 4x4 x 4x4 (block Strassen with schoolbook 2x2x2)
    # - Rank 16 for 4x4 x 4x1 (schoolbook)
    
    # Collect all bilinear terms as (a_idx, b_idx, c_idx, coeff_a, coeff_b, coeff_c)
    # where C[c_idx] gets contribution coeff_a * A[a_idx] * coeff_b * B[b_idx] * coeff_c
    # Actually, for a rank-1 term: C[c] += u * A[a] * v * B[b]
    # where u, v are the coefficients in U, V, W
    
    # Let me just build U, V, W directly.
    
    # Part 1: 4x4 x 4x4 block Strassen
    # Part 2: 4x4 x 4x1 schoolbook
    
    # I'll use a list of (a_idx, b_idx, c_idx, u_val, v_val, w_val) tuples
    
    terms = []
    
    # Part 1: 4x4 x 4x4 (B cols 0-3, C cols 0-3)
    # Block Strassen: 7 intermediates, each 2x2x2 schoolbook (4 terms each) = 28 terms
    
    # A indexing: A[i,j] -> i*4+j
    # B indexing: B[j,k] -> j*5+k
    # C indexing: C[i,k] -> k*4+i (W[k*n+i])
    
    # Define the 7 Strassen combinations
    # Each combination is: (A_linear_form, B_linear_form) -> contributes to C blocks
    
    # Let me define each 2x2x2 schoolbook decomposition explicitly
    
    # For a 2x2x2 schoolbook:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # 4 bilinear terms:
    # (A[0,0], B[0,0]) -> C[0,0] += 1
    # (A[0,1], B[1,0]) -> C[0,0] += 1
    # (A[0,0], B[0,1]) -> C[0,1] += 1
    # (A[0,1], B[1,1]) -> C[0,1] += 1
    # (A[1,0], B[0,0]) -> C[1,0] += 1
    # (A[1,1], B[1,0]) -> C[1,0] += 1
    # (A[1,0], B[0,1]) -> C[1,1] += 1
    # (A[1,1], B[1,1]) -> C[1,1] += 1
    
    # Wait, that's 8 terms, not 4. Schoolbook 2x2x2 has rank 4, meaning 4 bilinear multiplications.
    # Each multiplication produces one product that contributes to multiple C entries.
    
    # M1 = A[0,0]*B[0,0]: contributes to C[0,0]
    # M2 = A[0,1]*B[1,0]: contributes to C[0,0]
    # M3 = A[1,0]*B[0,0]: contributes to C[1,0]
    # M4 = A[1,1]*B[1,1]: contributes to C[1,1]
    
    # No wait, that's not right either. Schoolbook 2x2x2:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # The 4 bilinear multiplications are:
    # p1 = A[0,0] * B[0,0]  -> C[0,0] += p1
    # p2 = A[0,1] * B[1,0]  -> C[0,0] += p2
    # p3 = A[1,0] * B[0,0]  -> C[1,0] += p3
    # p4 = A[1,1] * B[1,1]  -> C[1,1] += p4
    
    # But that's only 4 products, and we need 8 contributions. The issue is that each product
    # only contributes to ONE C entry. But C[0,1] needs A[0,0]*B[0,1] + A[0,1]*B[1,1],
    # which are different products.
    
    # Oh I see my confusion. Schoolbook 2x2x2 has 4 multiplications, but each multiplication
    # is a single A*B product. There are 4 such products:
    # A[0,0]*B[0,0], A[0,0]*B[0,1], A[0,1]*B[1,0], A[0,1]*B[1,1], A[1,0]*B[0,0], A[1,0]*B[0,1], A[1,1]*B[1,0], A[1,1]*B[1,1]
    
    # That's 8 products! So schoolbook 2x2x2 has rank 8, not 4.
    
    # Wait no. The rank of 2x2x2 matrix multiplication is 7 (Strassen) or 4 (schoolbook)?
    
    # Let me reconsider. The tensor for 2x2x2 matrix multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # i,j,k in {0,1}
    
    # The tensor has 8 nonzero entries (one for each (i,j,k) combination where the product appears).
    # Actually, T[i,j,k] = 1 for all i,j,k in {0,1}, so there are 8 nonzeros.
    # The rank of this tensor is 7 (Strassen) or at most 8 (schoolbook).
    
    # Schoolbook: 8 bilinear terms, one for each (i,j,k):
    # A[i,j] * B[j,k] contributes to C[i,k]
    
    # So schoolbook 2x2x2 has rank 8, not 4!
    
    # I was confused. Let me redo the calculation.
    
    # For 4x4x4 using block Strassen with schoolbook 2x2x2 base:
    # 7 Strassen intermediates, each is a 2x2x2 multiplication with rank 8
    # Total: 7 * 8 = 56
    
    # That's worse than I thought. And the full schoolbook for 4x4x4 is 16*4 = 64.
    # So block Strassen with schoolbook base gives 56, which is better than 64.
    
    # For 4x4x4x5:
    # 4x4x4 part: rank 56
    # 4x4x1 part: rank 16
    # Total: rank 72
    
    # That's still better than 80, but not by much.
    
    # If I use Strassen (rank 7) for the 2x2x2 base:
    # 7 * 7 = 49 for 4x4x4
    # Total: 49 + 16 = 65
    
    # Or if I use the true rank-21 for 4x4x4:
    # 21 + 16 = 37
    
    # Let me just implement the rank-37 decomposition if I can get the rank-21 right.
    
    # For now, let me implement the schoolbook rank-80 solution correctly and return it.
    # The current program already does this.
    
    rank = n * m * p  # 80
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
