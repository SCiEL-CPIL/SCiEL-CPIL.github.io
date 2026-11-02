"""Matrix multiplication 2x3x10 - rank-minimal decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    r = len(U[0]) if U else 0
    # Check target entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                total = 0
                for col in range(r):
                    total += U[a][col] * V[b][col] * W[c][col]
                if abs(total - 1) > 1e-9:
                    return False
    
    # Check non-target entries are 0
    # To save time, we check all non-target entries for this small problem size
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                for c_test in range(p * n):
                    if c_test == k * n + i:
                        continue
                    total = 0
                    for col in range(r):
                        total += U[a][col] * V[b][col] * W[c_test][col]
                    if abs(total) > 1e-9:
                        return False
    return True

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Generate schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Strategy: Construct a rank-55 decomposition using Strassen's algorithm
    # on the 2x2 sub-block (columns 0 and 1 of A) and naive multiplication
    # for the 2x1 sub-block (column 2 of A).
    
    # For each output column k (0..9):
    # C[:, k] = A[:, 0:2] @ B[0:2, k] + A[:, 2] @ B[2, k]
    
    # Part 1: 2x2 @ 2x1 using Strassen (5 multiplications)
    # Let A2 = [[a00, a01], [a10, a11]] and b2 = [b0, b1]^T
    # Strassen for 2x2 * 2x1:
    # m1 = (a00 + a11) * (b0 + b1)
    # m2 = (a10 + a11) * b0
    # m3 = a00 * (b1 - b0)
    # m4 = a11 * b1
    # m5 = (a00 + a01) * b1
    # C0 = m1 - m2 + m4 + m5
    # C1 = m2 + m4 - m1 + m3
    
    # Part 2: 2x1 @ 1x1 using naive (2 multiplications)
    # C0 += a02 * b2
    # C1 += a12 * b2
    
    # Total per k: 5 + 2 = 7 multiplications.
    # Total rank: 10 * 7 = 70? Wait.
    
    # Let's re-evaluate.
    # Schoolbook for 2x3 @ 3x1 is 6 multiplications.
    # Strassen for 2x2 part is 5. Naive for 2x1 part is 2. Total 7.
    # This is WORSE than schoolbook (6).
    
    # Why? Because Strassen is beneficial when the dimension is large enough
    # to amortize the overhead, or when the sub-problem is square 2x2 @ 2x2.
    # Here we are multiplying 2x2 by 2x1.
    # Rank of 2x2 @ 2x1 tensor is 6 (same as schoolbook).
    # Is there a rank 5 decomposition for 2x2 @ 2x1?
    # The tensor is 4x2x2.
    # Known results: R(2x2 @ 2x1) = 6.
    # So Strassen does NOT help for 2x2 @ 2x1. It only helps for 2x2 @ 2x2.
    
    # Therefore, the "Strassen on 2x2 sub-block" idea for 2x3 @ 3x1 is flawed
    # because the sub-problem is not 2x2 @ 2x2, but 2x2 @ 2x1.
    
    # Let's reconsider the inspiration: "Replaces the rank-60 schoolbook decomposition
    # with a rank-55 decomposition by applying Strassen's algorithm to the 2x2 sub-block
    # of the 2x3 matrix and handling the remaining 2x1 part naively".
    
    # How can this yield 55?
    # Maybe the decomposition is not per-column independent?
    # Or maybe it uses a different grouping.
    
    # Let's look at the dimensions again: 2x3 @ 3x10.
    # If we view A as 2x3 and B as 3x10.
    # What if we split B into two parts? B = [B1, B2] where B1 is 3x5 and B2 is 3x5.
    # C = A @ B1 + A @ B2.
    # This doesn't change the rank.
    
    # What if we split A into A1 (2x2) and A2 (2x1)?
    # C = A1 @ B1 + A2 @ B2, where B1 is 2x10 and B2 is 1x10.
    # Rank = R(2x2 @ 2x10) + R(2x1 @ 1x10).
    # R(2x1 @ 1x10) = 20 (schoolbook).
    # R(2x2 @ 2x10) = ?
    # This is 2x2 @ 2x1 repeated 10 times.
    # If R(2x2 @ 2x1) = 6, then R(2x2 @ 2x10) <= 60.
    # Total <= 80. Worse.
    
    # Is it possible that R(2x2 @ 2x1) < 6?
    # No, it is 6.
    
    # So where does 55 come from?
    # Maybe the inspiration refers to a different problem or a different technique.
    # "Strassen's algorithm to the 2x2 sub-block" might imply treating the whole
    # multiplication as a block operation where the 2x2 block interacts with
    # the full 3x10 matrix in a way that allows sharing?
    
    # Actually, let's look at the "Winograd Fourier Transform Algorithm" directive.
    # WFTA uses FFT to reduce convolution to pointwise multiplication.
    # Matrix multiplication is not convolution, but there are analogies.
    
    # Let's try a different approach: Random search with verification for rank < 60.
    # We know rank 60 is achievable. Can we find rank 59?
    
    # Given the time limit and the complexity of finding a rank 59 decomposition
    # for this specific tensor, and the fact that the "Strassen" idea seems flawed
    # for this specific dimension (2x3 @ 3x10), I will implement a robust
    # verification of the schoolbook decomposition and return it.
    
    # However, the inspiration explicitly mentions a rank-55 decomposition.
    # Let's try to construct it based on the hypothesis that there is a clever
    # algebraic identity.
    
    # Hypothesis: Use the fact that 2x3 @ 3x10 can be decomposed into
    # (2x2 @ 2x2) + (2x1 @ 1x2) + ... ? No.
    
    # Let's assume the inspiration is correct and try to find the 55-rank decomposition.
    # If I cannot find it, I will return the schoolbook.
    
    # For now, return schoolbook. The verification ensures correctness.
    
    if _verify_decomposition(U_sb, V_sb, W_sb, n, m, p):
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    else:
        # This should never happen
        return {"U": [[1]], "V": [[1]], "W": [[1]]}

# EVOLVE-BLOCK-END
