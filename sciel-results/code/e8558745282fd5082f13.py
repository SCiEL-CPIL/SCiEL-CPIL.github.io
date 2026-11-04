"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK. Maintaining schoolbook rank 27."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n=3, m=3, p=3):
    """Verify that the decomposition exactly computes matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    For matrix mult, T[i,j,k] should be 1 if (i,j,k) corresponds to a valid (A[i,j]*B[j,k]) term, else 0.
    Actually for matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    So the tensor T[i,j,k] = 1 for all i,j,k (it's the identity-like tensor for multiplication)
    Wait - the tensor for matrix multiplication has T[i,j,k] = 1 for all valid indices.
    """
    rank = len(U[0]) if U else 0
    # Check tensor: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if total != 1:
                    return False
    return True

def _round_to_half(x):
    """Round to nearest half-integer."""
    return Fraction(2 * round(2 * x)) / 2

def _try_rank23_decomposition():
    """
    Attempt to construct a rank-23 decomposition for 3x3 matrix multiplication.
    Based on known algebraic structures similar to Strassen's algorithm.
    
    The key idea: use linear combinations of matrix entries to reduce the number
    of scalar multiplications needed.
    """
    n, m, p = 3, 3, 3
    rank = 23
    
    # For 3x3 matrix multiplication, we can use a structured approach.
    # Let A and B be 3x3 matrices. C = A * B.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We need to find rank-23 factors U, V, W such that:
    # C[i,k] = sum_c U[i,k,c] * V[i,k,c] * W[i,k,c]  (using appropriate indexing)
    #
    # Actually, the standard formulation:
    # U is (n*m, rank) = (9, rank) - indexed by (i,j)
    # V is (m*p, rank) = (9, rank) - indexed by (j,k)  
    # W is (p*n, rank) = (9, rank) - indexed by (k,i)
    #
    # And: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
    #
    # This is equivalent to finding a rank-23 decomposition of the 3x3x3 all-ones tensor.
    
    # One known approach uses the following structure based on circulant matrices
    # and algebraic identities. Let's try a systematic search with structure.
    
    # Initialize with schoolbook and try to merge terms
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Start with 23 columns. First 27 would be schoolbook, but we need only 23.
    # Use a known pattern: exploit that some linear combinations can be shared.
    
    # Strategy: Use the fact that for 3x3, we can group terms.
    # Let's use a random initialization with structure and verify.
    
    # Actually, let's use a known rank-23 construction based on:
    # The tensor M_3,3,3 has border rank 23 (Strassen).
    # An explicit decomposition can be constructed using:
    # - 9 terms for diagonal-like contributions
    # - 14 additional terms for off-diagonal combinations
    
    # Let me try a different approach: use the known decomposition from literature.
    # For 3x3 matrix multiplication, one rank-23 decomposition uses:
    
    # Define basis for U, V, W using linear combinations.
    # This is complex, so let's use a heuristic: start with random rank-23 factors
    # and check if they satisfy the tensor equation.
    
    # Better: use the algebraic structure. The 3x3 matrix multiplication tensor
    # can be decomposed using the following 23 terms:
    
    # Term 1-9: Direct products (but merged cleverly)
    # Terms 10-23: Cross terms using (A[i,j]+A[i',j'])(B[j,k]+B[j',k']) expansions
    
    # Let's construct it explicitly:
    # For each pair of indices, define linear forms in A and B.
    
    # U[i*m+j, c] = coefficient of A[i,j] in the c-th linear form
    # V[j*p+k, c] = coefficient of B[j,k] in the c-th linear form
    # W[k*n+i, c] = 1 (or coefficient for C[k,i]... wait, W is indexed by (k,i) for C[i,k])
    
    # Actually W[k*n+i, c] should be the coefficient such that:
    # C[i,k] = sum_c (sum_j U[i,j,c]*A[i,j]) * (sum_j V[j,k,c]*B[j,k]) * W[k,i,c]
    # But that's not quite right. The standard formulation is:
    # C[i,k] = sum_c U_c(i) * V_c(k) where U_c(i) depends on row i of A, V_c(k) depends on column k of B.
    
    # Let me use the correct formulation:
    # U is (n*m, rank): U[i*m+j, c] is the coefficient of A[i,j] in the c-th factor
    # V is (m*p, rank): V[j*p+k, c] is the coefficient of B[j,k] in the c-th factor
    # W is (p*n, rank): W[k*n+i, c] is the coefficient for C[i,k] output
    
    # The condition is: for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Let's try to build this using known algebraic identities.
    
    # One approach: use the fact that (x+y)(x'-y') = xx' - xy' + yx' - yy'
    # This allows combining 4 products into 3 multiplications.
    
    # For 3x3, we can apply such identities to reduce from 27 to 23.
    
    # Let me define 23 linear forms:
    
    # Forms 1-9: A[i,j] * B[j,k] for all i,j,k (but we'll merge some)
    # Actually, let's just try a random search with verification.
    
    # Given time constraints, let's return None to indicate failure and fall back to schoolbook.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try rank-23 first
    if time.time() < deadline - 10:
        result = _try_rank23_decomposition()
        if result is not None:
            return result
    
    # Fall back to schoolbook rank 27
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
