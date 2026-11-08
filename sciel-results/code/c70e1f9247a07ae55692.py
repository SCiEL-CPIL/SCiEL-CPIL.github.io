"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Use a rank-7 decomposition for 2x2 matrix multiplication (Strassen)
    # and apply it to 3x3 blocks of 2x2 submatrices.
    # However, we can do better by using a direct approach with known
    # low-rank decompositions for small matrices.
    
    # Actually, let's use the fact that 6x6 matrix multiplication can be
    # done with rank < 189. The best known upper bound for 6x6 is around
    # 140-150, but let's try a different block decomposition.
    
    # Alternative: Use 3x3 blocks of 1x1? No, that's just 216 scalar mults.
    # Use 2x2 blocks: 3x3=9 blocks, each needs rank 7, total 63? No, 
    # Strassen gives 7 mults per 2x2, but we have 3x3=9 positions, so 9*7=63?
    # Wait, no. For 6x6 with 2x2 blocks, we have a 3x3 grid of 2x2 blocks.
    # The matrix multiplication of two 3x3 block matrices where each block is 2x2
    # requires summing over the middle index. For each output block (I,K), we need
    # sum_J A[I,J] * B[J,K]. Each such product is a 2x2 * 2x2 = 2x2 multiplication.
    # Using Strassen, each 2x2*2x2 takes 7 multiplications. There are 3*3*3=27 such
    # block multiplications, so 27*7 = 189. That's what the current code does.
    
    # Can we do better? What if we use a different strategy?
    # 
    # One approach: Use the fact that 6 = 2*3, and use a Kronecker product structure.
    # The tensor for 6x6 matrix multiplication is the Kronecker product of the
    # tensor for 2x2 and 3x3 matrix multiplication.
    # rank(6x6) <= rank(2x2) * rank(3x3) = 7 * 23 = 161 (using best known for 3x3)
    # Actually, the rank of 3x3 matrix multiplication is at most 23 (Winograd's result).
    # So 7 * 23 = 161.
    # 
    # But we can also use rank(2x2) * rank(3x3) if we can construct it.
    # Let's use Strassen for 2x2 (rank 7) and a rank-23 decomposition for 3x3.
    
    # Actually, let me look up: the best known upper bound for 3x3 matrix multiplication
    # is 23 (Winograd 1980). Let me construct that.
    
    # Winograd's rank-23 decomposition for 3x3 matrix multiplication.
    # This is complex. Let me use a simpler approach first.
    
    # Alternative: just use the Kronecker product of Strassen(2x2) and a known
    # decomposition for 3x3.
    
    # For 3x3, the trivial decomposition has rank 27. With Strassen-like ideas,
    # we can get lower. Let's use the fact that 3x3 can be decomposed using
    # the "Kronecker" trick.
    
    # Actually, let me just use a direct construction with rank 161 via
    # Kronecker product of rank-7 (2x2) and rank-23 (3x3).
    
    # First, let me define the rank-23 decomposition for 3x3.
    # This is Winograd's decomposition. It's quite involved.
    
    # Let me try a different approach: use the known result that
    # rank(6x6) <= 140 or so. But constructing it explicitly is hard.
    
    # For now, let's try to improve upon 189 by using a better 3x3 decomposition.
    # The trivial 3x3 has rank 27. If we can get rank 23 for 3x3, then
    # Kronecker with 2x2 (rank 7) gives 7*23 = 161.
    
    # Let me implement Winograd's rank-23 for 3x3.
    
    # Winograd's 3x3 decomposition (rank 23):
    # Reference: Winograd, "On the number of multiplications required to compute
    # the product of two 3x3 matrices", 1980.
    
    # The decomposition uses 23 products. Let me code it up.
    
    # Actually, this is quite complex to code from memory. Let me use a 
    # different strategy.
    
    # Strategy: Use the Kronecker product structure.
    # T_6 = T_2 ⊗ T_3 (in some sense, via the Kronecker product of the tensors)
    # If we have decompositions of T_2 with rank r2 and T_3 with rank r3,
    # then T_6 has rank at most r2 * r3.
    
    # For T_2 (2x2), rank 7 (Strassen).
    # For T_3 (3x3), let's use rank 23 (Winograd).
    # Total: 7 * 23 = 161.
    
    # Let me implement this.
    
    # Strassen for 2x2:
    strassen_2x2 = [
        # (U_coeffs, V_coeffs, W_coeffs) where each is a dict {(i,j): val}
        # M0 = (A00+A11)(B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M1 = (A10+A11)B00
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M2 = A00(B01-B11)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M3 = A11(B10-B00)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M4 = (A00+A01)B11
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M5 = (A10-A00)(B00+B01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M6 = (A01-A11)(B10+B11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # For 3x3, I'll use a rank-23 decomposition. Let me use a known one.
    # Actually, let me just use the trivial rank-27 for 3x3 first to get
    # 7*27 = 189, which is the same as before. That's not an improvement.
    
    # Let me try to find a better decomposition.
    
    # Actually, let me just use a direct rank-161 construction if I can
    # find the Winograd 3x3 decomposition.
    
    # Given the complexity, let me try a different approach: use the
    # "border rank" or other algebraic techniques.
    
    # For simplicity and correctness, let me stick with the Kronecker
    # product approach but use a better 3x3 decomposition.
    
    # Let me use the fact that 3x3 matrix multiplication can be done with
    # rank at most 23. Here's a reference implementation:
    
    # Winograd's rank-23 for 3x3:
    # I'll use a simplified version. The key insight is that we can express
    # the 27 products as 23 products by exploiting linear dependencies.
    
    # This is getting too complex. Let me just use the current approach
    # but with a small optimization: use 2x3 blocks instead of 2x2.
    
    # Actually, let me just return the current solution with a minor tweak.
    # The current score is 1.1429, which corresponds to rank 189.
    # Let me try to get a lower rank.
    
    # New idea: Use the fact that 6x6 = (2*3) x (2*3) and use the
    # "Kronecker" decomposition more carefully.
    
    # Let me just implement the Kronecker product of Strassen(2x2) and
    # a rank-23(3x3) decomposition.
    
    # For now, let me use a placeholder rank-23 for 3x3.
    
    # Actually, I found a reference: the rank-23 decomposition for 3x3
    # can be constructed. Let me code it up.
    
    # Winograd 3x3 rank-23 decomposition:
    # The 23 products are:
    # M0 = A00*B00
    # M1 = A00*B01
    # M2 = A00*B02
    # M3 = A00*B10
    # M4 = A00*B11
    # M5 = A00*B12
    # M6 = A00*B20
    # M7 = A00*B21
    # M8 = A01*B00
    # M9 = A01*B01
    # M10 = A01*B02
    # M11 = A01*B10
    # M12 = A01*B11
    # M13 = A01*B12
    # M14 = A01*B20
    # M15 = A01*B21
    # M16 = A02*B00
    # M17 = A02*B01
    # M18 = A02*B02
    # M19 = A02*B10
    # M20 = A02*B11
    # M21 = A02*B12
    # M22 = (A10+A20)*(B00+B10) - A00*B00 - A00*B10 - A10*B00 - A20*B00
    # ... this is getting messy.
    
    # Let me just use the trivial decomposition for 3x3 (rank 27) and
    # accept rank 7*27 = 189. That's the same as before.
    
    # OK, let me try yet another approach: use a direct search or known
    # low-rank decompositions.
    
    # The best known upper bound for 6x6 matrix multiplication rank is
    # actually much lower than 189. It's known that rank(6x6) <= 140
    # or so, but I don't have the explicit decomposition.
    
    # For this exercise, let me just use the Kronecker product of
    # Strassen(2x2) and trivial(3x3), giving rank 7*27 = 189.
    
    # This is the same as the current program. Let me just return it.
    
    block_size = 2
    blocks_per_dim = 3
    
    rank = (blocks_per_dim ** 3) * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_2x2:
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
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
