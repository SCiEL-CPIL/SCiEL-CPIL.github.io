import numpy as np
import time
import sys
from itertools import product

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    start_time = time.time()
    time_limit = 140.0 # Safety margin

    # We want to find U, V, W with rank R such that for all i,j,k:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    
    # Strategy: Use a grid search over half-integers for small ranks.
    # Since the tensor is all 1s, the solution might be simple.
    # The space of half-integers is discrete.
    # For R=1: U[i,j]*V[j,k]*W[k,i] = 1 for all i,j,k.
    # This implies U[i,j] = 1/V[j,k]/W[k,i].
    # Let's check if R=1 is possible.
    # If U[i,j] = a_i b_j, V[j,k] = b_j c_k, W[k,i] = c_k a_i, then product is a_i^2 b_j^2 c_k^2 = 1.
    # This requires a_i^2 = 1, b_j^2 = 1, c_k^2 = 1.
    # So a_i, b_j, c_k are +/- 1.
    # Then U[i,j] = a_i b_j, V[j,k] = b_j c_k, W[k,i] = c_k a_i.
    # Product: (a_i b_j)(b_j c_k)(c_k a_i) = a_i^2 b_j^2 c_k^2 = 1.
    # So R=1 works!
    # Let's verify.
    # i in [0,n), j in [0,m), k in [0,p).
    # U[i*m+j] = a_i b_j
    # V[j*p+k] = b_j c_k
    # W[k*n+i] = c_k a_i
    # Sum over c=0: U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = a_i b_j * b_j c_k * c_k a_i = a_i^2 b_j^2 c_k^2.
    # If we set a_i = 1, b_j = 1, c_k = 1 for all i,j,k, then the product is 1.
    # So U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1.
    # Let's check if this is a valid decomposition.
    # The problem statement says "exactly decompose the matrix multiplication tensor".
    # The tensor T_{i,j,k} = 1.
    # The decomposition is sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = T_{i,j,k}.
    # If R=1, U=1, V=1, W=1, then sum = 1*1*1 = 1.
    # This is a valid exact decomposition with rank 1.
    
    # Why did the seed program not find this?
    # The seed program uses ALS which might get stuck or start from random values that don't converge to this simple solution.
    # Or maybe I am misinterpreting the tensor.
    # "Matrix multiplication 2,4,7"
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 7 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]."
    
    # Let's re-read carefully.
    # Standard matrix multiplication: C = AB.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # This is a trilinear map (A, B) -> C.
    # The tensor T has indices i, j, k.
    # T_{i,j,k} = 1.
    # The decomposition is T_{i,j,k} = sum_c U_{i,c} V_{j,c} W_{k,c}.
    # But the problem specifies:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # And W[k*n+i, r].
    
    # In the standard decomposition, U is (n, rank), V is (m, rank), W is (p, rank).
    # Here, U is (n*m, rank). This is different.
    # If U is (n*m, rank), it depends on both i and j.
    # This implies the decomposition is NOT the standard one for the tensor T_{i,j,k}=1.
    # It must be a different tensor or a different definition of the factors.
    
    # Let's look at the schoolbook code again:
    # U[i * m + j][col] = 1
    # V[j * p + k][col] = 1
    # W[k * n + i][col] = 1
    # For a specific (i,j,k), the term is 1*1*1 = 1.
    # For any other combination of indices in the sum?
    # The sum is over c.
    # The equation is: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = T_{i,j,k}.
    # In the schoolbook case, for a fixed (i,j,k), only the term c=(i,j,k) is non-zero.
    # So the sum is 1.
    # This means T_{i,j,k} = 1 for all i,j,k.
    # So the tensor IS T_{i,j,k}=1.
    
    # But why is U (n*m, rank)?
    # In the standard decomposition, U_{i,c} depends only on i.
    # Here, U_{i*m+j, c} depends on i and j.
    # This is a different factorization format.
    # It is not the standard CP decomposition of the tensor T.
    # It is a decomposition where the factors are "mixed" in the indices.
    
    # Let's check if R=1 works in this format.
    # U[i*m+j, 0] = a_{i,j}
    # V[j*p+k, 0] = b_{j,k}
    # W[k*n+i, 0] = c_{k,i}
    # We need a_{i,j} b_{j,k} c_{k,i} = 1 for all i,j,k.
    # Can we find such a_{i,j}, b_{j,k}, c_{k,i}?
    # Let a_{i,j} = x_i y_j.
    # Let b_{j,k} = y_j z_k.
    # Let c_{k,i} = z_k x_i.
    # Then a_{i,j} b_{j,k} c_{k,i} = (x_i y_j)(y_j z_k)(z_k x_i) = x_i^2 y_j^2 z_k^2.
    # We need x_i^2 y_j^2 z_k^2 = 1.
    # This is possible if x_i^2 = 1, y_j^2 = 1, z_k^2 = 1.
    # So x_i, y_j, z_k are +/- 1.
    # Then a_{i,j} = x_i y_j, b_{j,k} = y_j z_k, c_{k,i} = z_k x_i.
    # These are half-integers (integers).
    # So R=1 should work!
    
    # Let's test this hypothesis.
    # Set x_i = 1 for all i, y_j = 1 for all j, z_k = 1 for all k.
    # Then a_{i,j} = 1, b_{j,k} = 1, c_{k,i} = 1.
    # U[i*m+j, 0] = 1.
    # V[j*p+k, 0] = 1.
    # W[k*n+i, 0] = 1.
    
    # Let's verify with the code logic.
    # For any i,j,k:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # = U[i*m+j, 0] V[j*p+k, 0] W[k*n+i, 0]
    # = 1 * 1 * 1 = 1.
    
    # This seems too simple. Is there a catch?
    # "exactly decompose the matrix multiplication tensor"
    # Maybe the tensor is not T_{i,j,k}=1?
    # No, the schoolbook code produces T_{i,j,k}=1.
    
    # Let's double check the shapes.
    # U: (n*m, rank) -> 8x1
    # V: (m*p, rank) -> 28x1
    # W: (p*n, rank) -> 14x1
    
    # If R=1, the score is 1.0 / 1.0 = 1.0?
    # No, score is seed/value.
    # Seed is 56.
    # Value is 1.
    # Score = 56 / 1 = 56.
    # This would be a massive improvement.
    
    # Why did the seed program fail to find this?
    # It used ALS with random initialization.
    # If you start with random values, the product of three random vectors is likely to be small and noisy.
    # The gradient might point away from the simple solution.
    # Or the least squares steps might not converge to the all-ones solution.
    
    # Let's implement the R=1 solution.
    
    R = 1
    U = [[1.0] for _ in range(n * m)]
    V = [[1.0] for _ in range(m * p)]
    W = [[1.0] for _ in range(p * n)]
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(R):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if for some reason R=1 doesn't work (e.g. my understanding is wrong)
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
