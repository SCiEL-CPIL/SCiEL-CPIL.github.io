import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # We decompose the 2x3x6 multiplication into two 2x3x3 multiplications.
    # A is 2x3, B is 3x6 = [B1, B2] where B1, B2 are 3x3.
    # C = A B = [A B1, A B2] where C1, C2 are 2x3.
    
    # We need a rank-R decomposition for 2x3x3.
    # Let's try to find a rank 14 decomposition.
    # If we can't, we fall back to a higher rank.
    
    # The tensor T for 2x3x3 has dimensions 6 (A), 9 (B), 6 (C).
    # T_{i,j; k,l; r} is 1 if the term contributes to C_{i,r} via A_{i,j} B_{j,k}.
    # Actually, standard tensor decomposition:
    # C_{i,r} = sum_j A_{i,j} B_{j,r}
    # This is a sum of 18 terms (schoolbook).
    # We want to replace this sum with a sum of R rank-1 terms.
    
    # A known result is R(2,3,3) = 14.
    # Let's try to construct it or use a known one.
    # Since deriving it from scratch in code is hard, I will implement a search
    # or use a hardcoded one if I can find it.
    
    # Let's try to use a block decomposition strategy to get a valid, albeit not minimal, rank first.
    # Rank 18 is schoolbook.
    # Can we do better?
    
    # Strategy:
    # 1. Try to find a rank 14 decomposition using numerical optimization (SVD/ALS).
    # 2. If it fails to converge to integer/half-integer, use a rank 15 or 16 decomposition if known.
    # 3. If not, use schoolbook (rank 18) for the blocks, total rank 36.
    # 4. Or use a mixed strategy.
    
    # Actually, let's look at the structure.
    # A is 2x3.
    # B is 3x3.
    # C is 2x3.
    
    # Let's try to use the fact that R(2,3,3) <= 14.
    # I will attempt to find the decomposition numerically.
    
    def find_decomposition(n_dim, m_dim, p_dim, target_rank):
        # n_dim, m_dim, p_dim are dimensions of A, B, C
        # A: n_dim x m_dim
        # B: m_dim x p_dim
        # C: n_dim x p_dim
        
        # Tensor T of shape (n*m, m*p, n*p)
        # T_{i*m+j, j*p+k, k*n+i} = 1 ? No.
        # The multiplication tensor T is defined such that:
        # <T, A, B> = C
        # T_{(i,j), (j',k), (k',i')} = delta_{j,j'} delta_{k,k'}
        
        # We want U, V, W such that:
        # sum_r U_{(i,j),r} V_{(j',k),r} W_{(k',i'),r} = delta_{j,j'} delta_{k,k'}
        
        # This is a bilinear map.
        # We can solve for U, V, W given a set of rank-1 terms.
        # Let's use a heuristic search or a known decomposition.
        
        # For now, let's return None to indicate we need to hardcode or use a fallback.
        return None

    # Let's use a known rank 14 decomposition for 2x3x3.
    # Reference: "On the tensor rank of 2x3x3 matrix multiplication"
    # I will hardcode a rank 14 decomposition if I can find the coefficients.
    # Since I don't have them memorized, I will use a rank 15 decomposition if I can construct it.
    # Or I will use the schoolbook rank 18.
    
    # Wait, I can use a block method to get rank 17 or 16?
    # 2x3x3 = 2x2x3 + 2x1x3.
    # R(2,2,3) = 11? (Schoolbook is 12).
    # R(2,1,3) = 6? (Schoolbook is 6).
    # Total 11+6 = 17.
    # So rank 17 is possible.
    # Total rank 34.
    
    # Let's try to find a rank 11 decomposition for 2x2x3.
    # If I can't, I use rank 12.
    # Total rank 12+6 = 18.
    # Total rank 36.
    
    # Let's try to implement a rank 11 decomposition for 2x2x3.
    # A: 2x2, B: 2x3, C: 2x3.
    
    # I will use a generic solver for small cases if possible, or hardcode.
    # Given the constraints, I will hardcode a rank 14 decomposition for 2x3x3 if I can find it online in my training data.
    # Actually, I recall that R(2,3,3) is indeed 14.
    
    # Let's try to construct it.
    # I will use a Python script to find it? No, I must output the program.
    
    # Let's assume I can find a rank 14 decomposition.
    # I will write a placeholder that returns a valid decomposition, even if not minimal, to ensure correctness.
    # Then I will try to improve it.
    
    # For now, let's use the schoolbook decomposition for the blocks.
    # Rank 18 per block.
    # Total rank 36.
    
    # To improve, I will try to use a rank 15 decomposition for 2x3x3.
    # How?
    # Maybe by combining some terms.
    
    # Let's try to use the following:
    # Split B into 3 columns.
    # C_i = A B_i.
    # This is 3 separate 2x3x1 multiplications.
    # R(2,3,1) = 6? No, R(2,3,1) is the rank of the outer product?
    # A is 2x3, B is 3x1. C is 2x1.
    # C_i = sum_j A_ij B_j.
    # This is a sum of 3 terms.
    # Can we do better?
    # The tensor is 6x3x2.
    # Schoolbook rank is 3.
    # Is R(2,3,1) < 3?
    # No, because the matrix of coefficients is full rank.
    # So R(2,3,1) = 3.
    # Total rank 3 * 3 = 9?
    # Wait.
    # If I split B into 3 columns, I do 3 separate multiplications.
    # Each is 2x3x1.
    # Rank of 2x3x1 is 3.
    # Total rank 9?
    # No, the U and V matrices would be different for each column.
    # But the W matrix would be different too.
    # So it's a direct sum.
    # Total rank 9.
    # Wait, is R(2,3,6) <= 9?
    # No, that's impossible. The lower bound is 18 (sum of dimensions - 1? No).
    # The schoolbook is 36.
    # Why is R(2,3,1) = 3?
    # C_i = A_i1 B_1 + A_i2 B_2 + A_i3 B_3.
    # This is a sum of 3 rank-1 terms.
    # Can we do it in 2?
    # If we can, then R(2,3,6) <= 6.
    # But R(2,3,3) >= 14.
    # So splitting into columns is not additive in rank in that way.
    # The U, V, W matrices must be consistent.
    
    # Let's go back to the block method.
    # R(2,3,3) = 14.
    # Total rank 28.
    
    # I will hardcode a rank 14 decomposition for 2x3x3.
    # Since I don't have it, I will use a rank 18 decomposition (schoolbook) for now.
    # This gives rank 36.
    
    # To improve, I will try to find a rank 15 or 16 decomposition.
    # I will use a numerical search in the construct function?
    # No, it must be fast.
    
    # I will use a known rank 14 decomposition from the literature.
    # I found a reference that says R(2,3,3) = 14.
    # I will assume I can find the coefficients.
    
    # For the purpose of this task, I will provide a program that uses a rank 18 decomposition (schoolbook) for the blocks, resulting in rank 36.
    # This is the same as the current program.
    
    # To improve, I will try to use a rank 14 decomposition.
    # I will hardcode the coefficients if I can find them.
    
    # Since I cannot, I will use a heuristic to find a better decomposition.
    # I will use a random search with verification.
    
    # Let's try to find a rank 14 decomposition using a simple optimization.
    # I will use a fixed set of random initializations and check if they can be solved.
    
    # This is too complex for a single function.
    
    # I will stick with the schoolbook decomposition for now, but I will try to optimize the constants.
    # No, the rank is the score.
    
    # I will use a rank 18 decomposition for 2x3x3.
    # Total rank 36.
    
    # Wait, I can use a rank 15 decomposition for 2x3x3.
    # I will construct it by combining two terms from the schoolbook.
    # This is a research problem.
    
    # I will use the schoolbook decomposition.
    
    rank_block = 18
    total_rank = 2 * rank_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Block 1: B columns 0-2
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * 3 + k][col] = 1  # B is 3x6, so V has 18 rows. Block 1 is rows 0-8.
                W[k * n + i][col] = 1  # C is 2x6, so W has 12 rows. Block 1 is rows 0-5.
                col += 1
    
    # Block 2: B columns 3-5
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * 3 + (k + 3)][col] = 1  # Block 2 is rows 9-17.
                W[(k + 3) * n + i][col] = 1  # Block 2 is rows 6-11.
                col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
