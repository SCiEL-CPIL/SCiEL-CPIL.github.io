"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time

def construct():
    # Decomposition:
    # C = A B, A is 2x3, B is 3x10, C is 2x10.
    # C_{i,k} = sum_{j=0}^{2} A_{i,j} B_{j,k}
    #
    # We partition the sum by j=0, j=1, j=2.
    # For j=0: A_{i,0} B_{0,k}. A_{i,0} has 2 entries (i=0,1). B_{0,k} has 10 entries (k=0..9).
    #   We can use 2 terms: one for A_{0,0} and one for A_{1,0}.
    #   Term for i=0: U_{0,0,c}=1, V_{0,k,c}=1, W_{k,0,c}=1.
    #   Term for i=1: U_{1,0,c}=1, V_{0,k,c}=1, W_{k,1,c}=1.
    #   This gives 2 terms.
    #
    # For j=1: A_{i,1} B_{1,k}. Same structure. 2 terms.
    # For j=2: A_{i,2} B_{2,k}. Same structure. 2 terms.
    # Total: 6 terms. But we can do better.
    #
    # Actually, for each fixed j, the product is A_{:,j} * B_{j,:}, which is an outer product.
    # The rank of the tensor for fixed j is 2 (since A_{:,j} is 2x1 and B_{j,:} is 1x10).
    # So total rank = 3 * 2 = 6? No, the tensor is the sum of three such terms.
    #
    # Wait, the tensor T_{i,j,k} = A_{i,j} B_{j,k}. We want to write this as sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}.
    # Note that W depends on k and i, but in the standard decomposition W_{k,i,c} is the coefficient for the (k,i) output.
    #
    # Let's use the standard schoolbook decomposition but look for a better one.
    #
    # Actually, the matrix multiplication tensor M_{2,3,10} has rank at most 60 (schoolbook).
    #
    # Let's try to use the fact that n=2 is small.
    # C_{i,k} = A_{i,0} B_{0,k} + A_{i,1} B_{1,k} + A_{i,2} B_{2,k}.
    #
    # We can write this as:
    # C_{i,k} = (A_{i,0} + A_{i,1} + A_{i,2}) B_{0,k} + A_{i,1} (B_{1,k} - B_{0,k}) + A_{i,2} (B_{2,k} - B_{0,k})
    #
    # This gives 3 terms for each i, but the first term is shared.
    # Term 1: U_{i,j,c}=1 for all i,j, V_{j,k,c}=delta_{j,0}, W_{k,i,c}=1. This is 1 term? No, U is 2x3, V is 3x10.
    # U_{i,j,c} = 1 for all i,j. V_{0,k,c}=1, V_{j,k,c}=0 for j!=0. W_{k,i,c}=1 for all k,i.
    # This gives sum_i sum_j A_{i,j} * delta_{j,0} * 1 = sum_i A_{i,0} = A_{0,0} + A_{1,0}.
    # This doesn't match C_{i,k} = sum_j A_{i,j} B_{j,k}.
    #
    # Let's stick to a valid decomposition. The simplest is schoolbook with rank 60.
    #
    # Can we do better?
    # For n=2, p=10, m=3.
    #
    # Consider the case where we fix k. Then C_{:,k} = A B_{:,k}, where B_{:,k} is a 3x1 vector.
    # C_{0,k} = A_{0,0} B_{0,k} + A_{0,1} B_{1,k} + A_{0,2} B_{2,k}
    # C_{1,k} = A_{1,0} B_{0,k} + A_{1,1} B_{1,k} + A_{1,2} B_{2,k}
    #
    # This is a 2x3x1 tensor for each k, repeated 10 times.
    # The rank of the 2x3x1 tensor is at most 2 (since it's an outer product of a 2x1 and 1x3, or sum of two such).
    # Actually, T_{i,j} = A_{i,j} b_j. This is the outer product of column vector A (2x1) and row vector b (1x3).
    # The rank of this tensor is 1. So for each k, we need 1 term.
    # But the terms must be consistent across k.
    #
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    #
    # If we write B_{j,k} as a linear combination of some basis, we might reduce rank.
    #
    # Actually, the rank of the 2x3x10 matrix multiplication tensor is known to be at most 60.
    #
    # Let's just output the schoolbook decomposition to be safe, since finding a lower rank is non-trivial.
    #
    # However, we can try a simple improvement.
    # For the 2x3x10 case, we can use the fact that the tensor is a sum of 10 terms of rank 2 (for each k, the 2x3x1 slice has rank 1, but they are independent).
    #
    # Actually, let's just use the schoolbook decomposition. It's rank 60.
    #
    n, m, p = DIMENSIONS
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
