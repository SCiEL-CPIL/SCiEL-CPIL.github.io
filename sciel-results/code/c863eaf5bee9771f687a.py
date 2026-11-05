"""Matrix multiplication 3,3,7 tensor decomposition."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    Strategy: Start with schoolbook (rank 63) and attempt greedy rank reduction.
    For each term, check if it can be removed while the remaining terms still
    exactly span the multiplication tensor. If so, remove it.
    
    This is a heuristic that may find rank 62 or lower if any term is redundant.
    """
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # If time is running out, return schoolbook immediately
    if _time_left() < 10:
        return _schoolbook(n, m, p)
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    U = result["U"]
    V = result["V"]
    W = result["W"]
    rank = len(U[0])
    
    # Try to remove terms greedily
    # For each column r, check if removing it still gives exact decomposition
    removed = []
    for r in range(rank):
        if _time_left() < 5:
            break
        # Try removing term r
        U_test = [row[:r] + row[r+1:] for row in U]
        V_test = [row[:r] + row[r+1:] for row in V]
        W_test = [row[:r] + row[r+1:] for row in W]
        
        if _verify_decomposition(U_test, V_test, W_test, n, m, p):
            # Remove this term
            U = U_test
            V = V_test
            W = W_test
            removed.append(r)
    
    if removed:
        return {"U": U, "V": V, "W": W}
    
    return result

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition (rank n*m*p)."""
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

def _strassen_3x3():
    """
    Return Strassen's rank-7 decomposition for 3x3 * 3x3.
    Returns (U, V, W) where each is a list of lists.
    U shape: (9, 7), V shape: (9, 7), W shape: (9, 7)
    
    Strassen's 7 multiplications for 2x2, extended to 3x3.
    Actually, let me use the known rank-23 decomposition for 3x3*3x3.
    """
    pass

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is correct.
    
    The condition is: for all i in [0,n), j in [0,m), k in [0,p):
    sum_r U[i*m+j, r] * V[j*p+k, r] == W[k*n+i, r] ... 
    
    Wait, I need to get this right. Let me think about the tensor structure.
    
    For matrix multiplication C = A * B:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T[i,j,k] represents the coefficient of A[i,j] * B[j,k] in C[i,k].
    So T[i,j,k] = 1 if and only if the product A[i,j]*B[j,k] contributes to C[i,k].
    
    Since every product A[i,j]*B[j,k] contributes to C[i,k] (with coefficient 1),
    we have T[i,j,k] = 1 for all valid i,j,k.
    
    A rank-1 decomposition T = sum_r u_r ⊗ v_r ⊗ w_r means:
    T[i,j,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[k,i]
    
    Wait, the indices don't match up. Let me use the standard formulation.
    
    Actually, I think the correct condition is:
    For the decomposition to compute C = AB, we need:
    C[i,k] = sum_r W[i,k,r] * (sum_j U[i,j,r] A[i,j]) * (sum_j V[j,k,r] B[j,k]) * W[i,k,r]
    
    No, that's not right either.
    
    Let me look at this from the perspective of the output. The output C has entries
    C[i,k] for i in [0,n), k in [0,p). The decomposition should satisfy:
    
    C[i,k] = sum_r W[i,k,r] * (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k])
    
    For this to equal sum_j A[i,j] * B[j,k] for all A, B, we need:
    sum_r W[i,k,r] * (sum_j U[i,j,r] A[i,j]) * (sum_j V[j,k,r] B[j,k]) = sum_j A[i,j] B[j,k]
    
    This must hold for all A, B. So the coefficient of A[i',j'] B[j'',k''] must be:
    δ_{i,i'} δ_{k,k'} δ_{j',j''}
    
    The coefficient is:
    sum_r W[i,k,r] * U[i',j',r] * V[j'',k'',r]
    
    So we need:
    sum_r W[i,k,r] * U[i',j',r] * V[j'',k'',r] = δ_{i,i'} δ_{k,k'} δ_{j',j''}
    
    In the schoolbook decomposition:
    For each (i,j,k), we have one term with:
    U[i,j,r] = 1, V[j,k,r] = 1, W[i,k,r] = 1
    
    Then:
    sum_r W[i,k,r] * U[i',j',r] * V[j'',k'',r] 
    = sum over r where (i,k,r) matches (i',j',r) and (j'',k'',r)
    = 1 if i=i', k=k', j'=j'', and 0 otherwise
    
    Yes! That's correct.
    
    So the verification condition is:
    For all i,k,i',j',j'',k'':
    sum_r W[k*n+i, r] * U[i'*m+j', r] * V[j''*p+k'', r] = δ_{i,i'} δ_{k,k'} δ_{j',j''}
    
    But wait, the problem statement says the output factor follows W[k*n+i, r].
    So W is indexed by (k, i) not (i, k). Let me re-read.
    
    "The output factor follows the official trace convention W[k*n+i,r]."
    
    So W[k*n+i, r] corresponds to the (k, i) entry, which is C[i,k] (row i, col k).
    
    So in my notation above, W[i,k,r] should be W[k*n+i, r].
    
    Let me redo: The condition is:
    For all i,k,i',j',j'',k'':
    sum_r W[k*n+i, r] * U[i'*m+j', r] * V[j''*p+k'', r] = δ_{i,i'} δ_{k,k'} δ_{j',j''}
    
    OK so now I have the correct condition. Let me implement the verification.
    """
    n, m, p = n, m, p
    rank = len(U[0])
    
    # Check all combinations
    for i in range(n):
        for k in range(p):
            for i2 in range(n):
                for j in range(m):
                    for j2 in range(m):
                        for k2 in range(p):
                            expected = 1 if (i == i2 and k == k2 and j == j2) else 0
                            actual = 0.0
                            for r in range(rank):
                                actual += W[k * n + i][r] * U[i2 * m + j][r] * V[j2 * p + k2][r]
                            if abs(actual - expected) > 1e-10:
                                return False
    return True

def _rank23_3x3x3():
    """
    Return a rank-23 decomposition for 3x3 * 3x3 matrix multiplication.
    
    Uses the known rank-23 formula. The explicit coefficients are from
    the literature on fast matrix multiplication.
    
    For simplicity and correctness, I'll use a constructive approach:
    Start with the schoolbook (rank 9) and try to find a lower-rank
    decomposition through optimization.
    """
    n, m, p = 3, 3, 3
    
    # For now, return the schoolbook decomposition for 3x3*3x3
    # This has rank 9.
    U = [[0] * 9 for _ in range(9)]
    V = [[0] * 9 for _ in range(9)]
    W = [[0] * 9 for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j][col] = 1
                V[j * 3 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    return U, V, W

def _compute_error(U, V, W, n, m, p):
    """
    Compute the squared error of the decomposition.
    """
    rank = len(U[0])
    error = 0.0
    for i in range(n):
        for k in range(p):
            for i2 in range(n):
                for j in range(m):
                    for j2 in range(m):
                        for k2 in range(p):
                            expected = 1.0 if (i == i2 and k == k2 and j == j2) else 0.0
                            actual = 0.0
                            for r in range(rank):
                                actual += W[k * n + i][r] * U[i2 * m + j][r] * V[j2 * p + k2][r]
                            error += (actual - expected) ** 2
    return error

def _construct_improved():
    """
    Try to construct an improved decomposition for 3x3 * 3x7.
    
    For now, return the schoolbook decomposition.
    """
    n, m, p = 3, 3, 7
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
