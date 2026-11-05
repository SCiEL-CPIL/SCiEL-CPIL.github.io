"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 7
    # Build a known rank-50 decomposition using Strassen on 2x2x2 blocks and schoolbook for the rest.
    # We split the 4x7 matrix B into two 4x4 and 4x3 blocks? No, better to use block structure.
    # Actually, let's use the known optimal rank for 2x4x7 which is 40? No, 2x4x7 is known to be 40?
    # Let's check: 2x3x3 is 23. 2x4x4 is 32. 2x4x7...
    # A safe bet is to use a rank-56 schoolbook if we can't find a good one quickly, but we want to beat 56.
    # Let's try a block construction.
    # Split A (2x4) into two 2x2 blocks: A1, A2.
    # Split B (4x7) into two 4x7 blocks? No, B is 4x7. The inner dimension is 4.
    # So A = [A1 A2] where A1, A2 are 2x2.
    # B = [B1; B2] where B1, B2 are 2x7.
    # C = A1 B1 + A2 B2.
    # We can use Strassen to multiply 2x2 by 2x7?
    # Strassen requires square matrices usually, but can be adapted.
    # Or we can split B1 into B1a (2x3) and B1b (2x4)?
    # Let's stick to a simpler, known good decomposition.
    # The rank of 2x4x7 is known to be 40?
    # Actually, let's just implement a rank-50 decomposition based on Strassen for 2x2x2 and schoolbook for 2x2x1.
    # We can view 2x4x7 as:
    # Split the 4 dimension into 2+2.
    # C = A1 B1 + A2 B2 where A1, A2 are 2x2, B1, B2 are 2x7.
    # For each term, we multiply 2x2 by 2x7.
    # We can split 2x7 into four 2x2 and one 2x1? No, 2x7 = 2x2 + 2x2 + 2x2 + 2x1? No, 2x7 is 2 rows, 7 cols.
    # We can split the 7 columns into 3+4.
    # So B1 = [B1a B1b] where B1a is 2x3, B1b is 2x4.
    # Then A1 B1 = A1 B1a + A1 B1b.
    # A1 B1b is 2x2 * 2x4. We can use Strassen for 2x2*2x2 twice?
    # This is getting complicated.
    
    # Let's try a different approach: Randomized search for a low rank decomposition.
    # But we need to finish in 120s.
    # Let's use the known rank-56 schoolbook but try to reduce it by removing linear dependencies.
    # The schoolbook tensor has rank 56.
    # We can form the tensor T in R^{56 x 56}? No, T is in R^{(nm) x (mp) x (pn)}.
    # Flattened, it's a vector of length 2*4*7*... no.
    # The tensor is T[i, j, k] = delta_{i, i} delta_{j, j} delta_{k, k}? No.
    # T is the matrix multiplication tensor.
    # T[i, j, k] is 1 if the output index (k, i) depends on A[i, j] and B[j, k].
    # Actually, the tensor M is defined by M_{i, j, k} such that (AB)_{i, k} = sum_j M_{i, j, k} A_{i, j} B_{j, k}.
    # Here M_{i, j, k} = 1.
    # We want to write M = sum_c u_c \otimes v_c \otimes w_c.
    # Where u_c is in R^{nm}, v_c in R^{mp}, w_c in R^{pn}.
    # u_c[i*m + j] = 1 if term c uses A[i, j].
    # v_c[j*p + k] = 1 if term c uses B[j, k].
    # w_c[k*n + i] = 1 if term c contributes to C[k, i] (which is C[i, k] in standard notation? No, C[i, k] = sum_j A[i, j] B[j, k]).
    # The problem statement says W[k*n + i, r]. This implies the output index is (k, i) where k is the column of C and i is the row of C?
    # Standard matrix mult: C[i, k] = sum_j A[i, j] B[j, k].
    # So the output is indexed by (i, k).
    # The problem says W[k*n + i, r]. This is index k*n + i.
    # If n=2, p=7, then k is in 0..6, i is in 0..1.
    # So the output dimension is p*n = 14.
    # This matches C being p x n? No, C is n x p = 2 x 7.
    # Usually C is n x p.
    # If the output is stored as W[k*n + i], then for a fixed i, k varies from 0 to p-1.
    # So W is a vector of length p*n.
    # This corresponds to the tensor component M[i, j, k] contributing to C[i, k].
    # So w_c[k*n + i] = 1 if term c contributes to C[i, k].
    
    # Let's construct the schoolbook decomposition and then try to reduce rank by finding linear dependencies among the columns of the factor matrices.
    # U, V, W are matrices of size (nm x rank), (mp x rank), (pn x rank).
    # If we can find a linear combination of columns that is zero in the tensor sum, we can reduce rank.
    # This is equivalent to finding a low-rank approximation of the tensor.
    
    # Let's try a simple heuristic:
    # 1. Generate schoolbook U, V, W (rank 56).
    # 2. Randomly perturb the columns of U, V, W with small noise.
    # 3. Compute the tensor sum S = sum_c u_c \otimes v_c \otimes w_c.
    # 4. Check if S is close to the original tensor T.
    # 5. If we can reduce the rank by dropping columns that have small norms after perturbation? No, that doesn't work.
    
    # Alternative: Use the known fact that the rank of 2x4x7 is 40?
    # Actually, let's just use the schoolbook rank 56. It's safe and fast.
    # But the task asks to find a rank-minimal decomposition.
    # If we can't find a better one, we return 56.
    
    # Let's try to implement a rank-50 decomposition.
    # We can use Strassen for 2x2x2.
    # Split the 4x7 part into 2x2 + 2x2 + 2x2 + 2x1? No.
    # Split the 4 dimension (j) into two 2s.
    # A = [A1 A2], B = [B1; B2].
    # C = A1 B1 + A2 B2.
    # For A1 B1 (2x2 * 2x7):
    # Split B1 into B1a (2x3) and B1b (2x4).
    # A1 B1a: 2x2 * 2x3. Rank?
    # A1 B1b: 2x2 * 2x4. Rank?
    
    # This is too complex to derive on the fly.
    # Let's stick to the schoolbook rank 56.
    
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
