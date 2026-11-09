"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.linalg import null_space

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 4, 4

    # First, try to find a lower-rank decomposition using a search algorithm
    # The tensor T is defined by T[i,j,k] = 1 for all i,j,k
    # We want to find rank-minimal decomposition: T = sum_r u_r v_r w_r
    # where u_r in R^(n*m), v_r in R^(m*p), w_r in R^(p*n)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

    # Build the tensor in a form suitable for rank decomposition
    # We'll use the following approach:
    # 1. Start with schoolbook rank 32
    # 2. Try to reduce rank by finding linear dependencies among the columns
    # 3. Use numerical linear algebra to find a lower-rank approximation

    # The multiplication tensor can be viewed as a 3-tensor
    # T_{ijk} = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Actually, T_{ijk} is the coefficient of A_{ij} * B_{jk} in C_{ik}
    # So T_{ijk} = 1 for all valid i,j,k

    # Let's think of this differently. The tensor M_{n,m,p} has entries
    # M[i,j,k] = 1 for all i,j,k (it's the all-ones tensor for the multiplication)
    # 
    # A rank-r decomposition means:
    # M[i,j,k] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # Let's use a different approach: represent the tensor as a matrix and find
    # its rank via various unfolding methods.

    # Method: Try known decompositions and search for lower rank
    # Known: Strassen gives rank 28 for 2x4x4
    # Can we do better? Let's try a systematic search.

    # Build the constraint matrix for rank-r decomposition
    # For a fixed rank r, we need to find U, V, W such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # This is a system of n*m*p = 32 equations
    # Unknowns: r*(n*m + m*p + p*n) = r*(8 + 16 + 8) = 32r variables
    
    # For r < 28, this is an underdetermined system, so solutions may exist.
    # But the equations are cubic (trilinear), making this a non-linear problem.

    # Alternative approach: use the fact that the tensor has special structure
    # The all-ones tensor T[i,j,k] = 1 can be decomposed as:
    # T[i,j,k] = (1/2) * sum_{a=0}^{1} sum_{b=0}^{3} 1 * 1 * 1 ... no, that's just schoolbook
    
    # Let me try a different structural approach.
    # Partition the j dimension differently.
    
    # Actually, let me try to use a known result:
    # For 2x2x2, Strassen gives rank 7 (optimal)
    # For 2x4x4, we can partition as:
    #   A is 2x4 = [A0 | A1] where A0, A1 are 2x2
    #   B is 4x4 = [B00 B01; B10 B11] where each is 2x2
    #   C is 2x4 = [C0 | C1] where C0, C1 are 2x2
    #
    # C0 = A0*B00 + A1*B10
    # C1 = A0*B01 + A1*B11
    #
    # This is a 2x2 * 2x2 multiplication for each term, but we have 4 such terms.
    # Using Strassen for each: 4 * 7 = 28
    #
    # Can we share Strassen factors between terms? 
    # C0 = A0*B00 + A1*B10
    # C1 = A0*B01 + A1*B11
    #
    # Notice that A0 appears in both, and A1 appears in both.
    # Also B00, B01, B10, B11 all appear.
    #
    # Let's try to factor more cleverly:
    # C0 + C1 = A0*(B00+B01) + A1*(B10+B11)
    # C0 - C1 = A0*(B00-B01) + A1*(B10-B11)
    #
    # This doesn't immediately help reduce rank.

    # Let me try yet another approach: use a computer search.
    # We'll set up the problem as finding a rank-r decomposition.
    # For small r, we can try random initialization and gradient-free methods.
    # But this is a non-convex problem, so it's tricky.

    # Actually, let me try a more systematic approach.
    # The tensor M_{2,4,4} can be seen as a 2x4x4 tensor with all entries 1.
    # We can try to find its tensor rank by examining various unfoldings.

    # Matrix unfolding: fix one index, view as a matrix
    # T_(i) : (m*p) x (p*n) = 16 x 8 matrix, all entries 1
    # Rank of this matrix is 1.
    # But tensor rank is not determined by matrix unfolding rank alone.

    # Let me try a concrete search approach.
    # We'll parameterize U, V, W and try to find a rank-r solution.
    # Start with r = 24 and work down.

    # Actually, for a first attempt, let me just return the Strassen rank-28 solution
    # but with a different construction that might allow for further optimization.

    # Let me try to use a block Strassen approach more carefully.
    
    # The key insight: we have a 2x4 matrix times 4x4 matrix.
    # We can view this as:
    #   A (2x4) * B (4x4) = C (2x4)
    #   where A = [a0 a1 a2 a3] (2x4, so each aj is a 2x1 column)
    #   B is 4x4, C = [c0 c1 c2 c3] (2x4, each cj is 2x1)
    #   cj = sum_{j=0}^{3} aj * B_{j,c} where B_{j,c} is the j-th row, c-th col of B
    
    # This view doesn't immediately help.

    # Let me try a different partition:
    # A (2x4): split into two 2x2 blocks: A = [A_L | A_R]
    # B (4x4): split into four 2x2 blocks: B = [B_LL B_LR; B_RL B_RR]
    # C (2x4): split into two 2x2 blocks: C = [C_L | C_R]
    #
    # C_L = A_L * B_LL + A_R * B_RL
    # C_R = A_L * B_LR + A_R * B_RR
    #
    # Each of these is a sum of two 2x2 * 2x2 products.
    # Using Strassen: each product takes 7 multiplications.
    # Total: 4 * 7 = 28 multiplications.
    #
    # This is the standard block Strassen approach, giving rank 28.

    # Can we do better by sharing computations?
    # Notice that A_L and A_R are shared between the two equations.
    # Also, the four B blocks are all used.
    
    # Let me try to see if we can reduce the rank below 28.
    # One approach: try to find a rank-24 decomposition by solving a system of equations.
    
    # For a rank-r decomposition, we have:
    # C_{i,k} = sum_{j=0}^{3} A_{i,j} * B_{j,k} = sum_{c=0}^{r-1} U_{i,j,c} * V_{j,k,c} * W_{k,i,c}
    # 
    # Wait, the indices are: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # So: C_{i,k} = sum_{j=0}^{3} A_{i,j} B_{j,k} = sum_{c=0}^{r-1} W_{k*n+i, c} * (sum_j U_{i*m+j, c} * A_{i,j}) * (sum_k V_{j*p+k, c} * B_{j,k})
    # 
    # Hmm, this isn't quite right. Let me re-derive.
    #
    # The tensor decomposition says:
    # M_{i,j,k} = sum_c U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c}
    # where M_{i,j,k} = 1 for all i,j,k.
    #
    # And the matrix multiplication is:
    # C_{i,k} = sum_j A_{i,j} * B_{j,k}
    #
    # The tensor M_{i,j,k} is the coefficient that multiplies A_{i,j} * B_{j,k} to contribute to C_{i,k}.
    # So M_{i,j,k} = 1 means that A_{i,j} * B_{j,k} contributes fully to C_{i,k}.
    #
    # A rank-r decomposition of this tensor means we can write:
    # C_{i,k} = sum_c (sum_j U_{i*m+j, c} * A_{i,j}) * (sum_j V_{j*p+k, c} * B_{j,k}) * W_{k*n+i, c}
    # 
    # Wait, that's not right either. Let me be more careful.
    #
    # The tensor T is defined such that:
    # C_{i,k} = sum_{j} T_{i,j,k} * A_{i,j} * B_{j,k}
    # where T_{i,j,k} = 1 for all i,j,k.
    #
    # A rank-r decomposition of T means:
    # T_{i,j,k} = sum_{c=0}^{r-1} u_c(i,j) * v_c(j,k) * w_c(k,i)
    #
    # Then:
    # C_{i,k} = sum_{j} sum_{c} u_c(i,j) * v_c(j,k) * w_c(k,i) * A_{i,j} * B_{j,k}
    #         = sum_{c} w_c(k,i) * (sum_j u_c(i,j) * A_{i,j}) * (sum_j v_c(j,k) * B_{j,k})
    #
    # So we define:
    # U_c = sum_j u_c(i,j) * A_{i,j}  (this is a scalar for each i,c)
    # V_c = sum_j v_c(j,k) * B_{j,k}  (this is a scalar for each k,c)
    # C_{i,k} = sum_c w_c(k,i) * U_c(i) * V_c(k)
    #
    # In the standard notation:
    # U[i*m+j, c] = u_c(i,j)
    # V[j*p+k, c] = v_c(j,k)  
    # W[k*n+i, c] = w_c(k,i)
    #
    # And the constraint is:
    # sum_c u_c(i,j) * v_c(j,k) * w_c(k,i) = 1  for all i,j,k

    # This is a system of 2*4*4 = 32 equations in 32*r unknowns (for rank r).
    # For r=28, we have 896 unknowns and 32 equations, so solutions exist.
    # For r < 28, we have fewer unknowns, but the problem is still non-linear.

    # Let me try a computational approach: use random initialization and 
    # a local search to find a rank-r solution for r < 28.

    # However, this is computationally expensive and may not converge.
    # For now, let me just return the rank-28 Strassen decomposition.

    # Strassen-based decomposition: rank 28
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # (8, 28)

    # Strassen coefficients for 2x2 * 2x2
    u_coeffs = [
        [[1, 0], [0, 1]],   # M1
        [[0, 0], [1, 1]],   # M2
        [[1, 0], [0, 0]],   # M3
        [[0, 0], [0, 1]],   # M4
        [[1, 1], [0, 0]],   # M5
        [[-1, 0], [1, 0]],  # M6
        [[0, 1], [0, -1]],  # M7
    ]
    v_coeffs = [
        [[1, 0], [0, 1]],   # M1
        [[1, 0], [0, 0]],   # M2
        [[0, 1], [0, -1]],  # M3
        [[-1, 0], [1, 0]],  # M4
        [[0, 0], [0, 1]],   # M5
        [[1, 1], [0, 0]],   # M6
        [[0, 0], [1, 1]],   # M7
    ]
    w_coeffs = [
        [[1, 0], [0, 1]],   # M1
        [[0, 1], [0, -1]],  # M2
        [[0, 0], [1, 1]],   # M3
        [[1, 0], [0, 0]],   # M4
        [[-1, 0], [1, 0]],  # M5
        [[0, 1], [0, 0]],   # M6
        [[-1, 0], [0, 1]],  # M7
    ]

    for s in range(4):
        j_block = s // 2
        k_block = s % 2
        for f in range(7):
            col = s * 7 + f
            for pi in range(2):
                for pj in range(2):
                    val = u_coeffs[f][pi][pj]
                    if val != 0:
                        j = 2 * j_block + pj
                        U[pi * m + j][col] += val
            for pj in range(2):
                for qk in range(2):
                    val = v_coeffs[f][pj][qk]
                    if val != 0:
                        j = 2 * j_block + pj
                        k = 2 * k_block + qk
                        V[j * p + k][col] += val
            for qk in range(2):
                for pi in range(2):
                    val = w_coeffs[f][qk][pi]
                    if val != 0:
                        k = 2 * k_block + qk
                        W[k * n + pi][col] += val

    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
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
