"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def _rank1_decomposition(n, m, p):
    """Schoolbook rank n*m*p decomposition for n x m * m * p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0]) if U else 0
    
    # Build the tensor T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    T_computed = np.zeros((n*m, m*p, p*n))
    for c in range(rank):
        u_vec = np.array([U[i][c] for i in range(n*m)])
        v_vec = np.array([V[i][c] for i in range(m*p)])
        w_vec = np.array([W[i][c] for i in range(p*n)])
        T_computed += np.einsum('i,j,k->ijk', u_vec, v_vec, w_vec)
    
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1
    
    diff = np.max(np.abs(T_computed - T_expected))
    return diff < 1e-10

def _construct_2x3x9():
    """
    Construct a valid decomposition for 2x3 * 3x9.
    We start with the schoolbook decomposition (rank 54).
    Then we attempt to reduce the rank using linear algebra (null space elimination).
    """
    n, m, p = 2, 3, 9
    start = time.time()
    deadline = start + 140  # Safety margin before 150s kill
    
    # Start with schoolbook
    U, V, W = _rank1_decomposition(n, m, p)
    
    # Verification is implicit in construction, but let's be safe.
    # if not _verify_decomposition(U, V, W, n, m, p):
    #     return U, V, W
    
    # Try to reduce rank
    # The tensor T is the sum of rank-1 tensors.
    # We can view the columns of U, V, W as vectors.
    # If we have linear dependencies among the columns, we can reduce the rank.
    # Specifically, if sum_c alpha_c * u_c = 0, sum_c alpha_c * v_c = 0, sum_c alpha_c * w_c = 0,
    # then the term sum_c alpha_c * (u_c \otimes v_c \otimes w_c) = 0.
    # So we can remove these components.
    
    # Let's check for linear dependencies in the space of terms.
    # We have R terms. We can form a matrix M of size (n*m + m*p + p*n) x R? No, that's not quite right.
    # The condition is that the tensor T is unchanged if we add a null vector to the coefficients.
    # Let the decomposition be T = sum_{c=1}^R u_c \otimes v_c \otimes w_c.
    # If we have a vector alpha in R^R such that sum_c alpha_c u_c = 0, sum_c alpha_c v_c = 0, sum_c alpha_c w_c = 0,
    # then T = T + sum_c alpha_c (u_c \otimes v_c \otimes w_c) ? No, that's not how it works.
    
    # Actually, the set of all decompositions of a tensor T is an affine space.
    # The tangent space at a decomposition is given by the null space of a certain map.
    # Let's just try to find a smaller rank decomposition by solving a linear system.
    
    # Alternative: Use the fact that M_{2,3,9} can be decomposed into blocks.
    # But let's stick to the null space idea first, as it's a general structural improvement.
    
    # Let's check if the columns of U, V, W are linearly independent in a way that allows reduction.
    # We can stack U, V, W into a single matrix? No.
    
    # Let's try a different structural approach:
    # The tensor M_{2,3,9} is the Kronecker product of M_{2,1,1} and M_{1,3,9}? No.
    # M_{n,m,p} = M_{n,1,1} \otimes M_{1,m,p}?
    # M_{2,3,9} = M_{2,1,1} \otimes M_{1,3,9}.
    # Rank(M_{2,1,1}) = 2.
    # Rank(M_{1,3,9}) = 27 (schoolbook).
    # Rank(M_{2,3,9}) <= 2 * 27 = 54.
    
    # Can we do better?
    # M_{1,3,9} is just a vector of 27 elements. Its rank is 27.
    # So this doesn't help.
    
    # What about M_{2,3,9} = M_{2,3,3} \otimes M_{1,1,3}?
    # Rank(M_{2,3,3}) = 18 (schoolbook).
    # Rank(M_{1,1,3}) = 3.
    # Rank(M_{2,3,9}) <= 18 * 3 = 54.
    
    # It seems 54 is the natural bound from these factorizations.
    # The current score 1.08 implies rank 50.
    # So the previous program found a rank 50 decomposition.
    # Let's try to reproduce that or find something better.
    
    # The previous program used Strassen on 2x2 blocks.
    # Let's implement that logic again, as it was successful.
    
    U, V, W = _construct_2x3x9_strassen()
    
    if _verify_decomposition(U, V, W, n, m, p):
        # Try to reduce further using null space elimination
        U, V, W = _try_reduce_rank_nullspace(U, V, W, n, m, p, deadline)
        return U, V, W
    else:
        # Fallback to schoolbook if Strassen decomposition is incorrect
        return _rank1_decomposition(n, m, p)

def _strassen_2x2x2_terms():
    """
    Return the 7 rank-1 terms for Strassen's M_{2,2,2} decomposition.
    """
    terms = []
    
    # P1: (A11+A22)(B11+B22)
    u1 = {(0,0): 1, (1,1): 1}
    v1 = {(0,0): 1, (1,1): 1}
    w1 = {(0,0): 1, (1,1): 1}
    terms.append((u1, v1, w1))
    
    # P2: (A21+A22)B11
    u2 = {(1,0): 1, (1,1): 1}
    v2 = {(0,0): 1}
    w2 = {(1,0): 1, (1,1): -1}
    terms.append((u2, v2, w2))
    
    # P3: A11(B12-B22)
    u3 = {(0,0): 1}
    v3 = {(0,1): 1, (1,1): -1}
    w3 = {(0,1): 1, (1,1): 1}
    terms.append((u3, v3, w3))
    
    # P4: A22(B21-B11)
    u4 = {(1,1): 1}
    v4 = {(1,0): 1, (0,0): -1}
    w4 = {(0,0): 1, (1,0): 1}
    terms.append((u4, v4, w4))
    
    # P5: (A11+A12)B22
    u5 = {(0,0): 1, (0,1): 1}
    v5 = {(1,1): 1}
    w5 = {(0,0): -1, (0,1): 1}
    terms.append((u5, v5, w5))
    
    # P6: (A21-A11)(B11+B12)
    u6 = {(1,0): 1, (0,0): -1}
    v6 = {(0,0): 1, (0,1): 1}
    w6 = {(1,1): 1}
    terms.append((u6, v6, w6))
    
    # P7: (A12-A22)(B21+B22)
    u7 = {(0,1): 1, (1,1): -1}
    v7 = {(1,0): 1, (1,1): 1}
    w7 = {(0,0): 1}
    terms.append((u7, v7, w7))
    
    return terms

def _construct_2x3x9_strassen():
    """
    Construct decomposition for 2x3 * 3x9 using Strassen on 2x2 blocks.
    Achieves rank 50.
    """
    n, m, p = 2, 3, 9
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Part 1: A_{j<2} * B_{j<2} where j in {0,1}
    k_pairs = [(0,1), (2,3), (4,5), (6,7)]
    k_single = [8]
    
    strassen_terms = _strassen_2x2x2_terms()
    
    for (k1, k2) in k_pairs:
        for (u_dict, v_dict, w_dict) in strassen_terms:
            U_term = {}
            for (li, lj), coeff in u_dict.items():
                U_term[(li, lj)] = coeff
            
            V_term = {}
            for (lj, lk), coeff in v_dict.items():
                V_term[(lj, k1 + lk)] = coeff
            
            W_term = {}
            for (li, lk), coeff in w_dict.items():
                W_term[(li, k1 + lk)] = coeff
            
            U_terms.append(U_term)
            V_terms.append(V_term)
            W_terms.append(W_term)
    
    for k in k_single:
        for i in range(2):
            for j in range(2):
                U_term = {(i, j): 1}
                V_term = {(j, k): 1}
                W_term = {(i, k): 1}
                U_terms.append(U_term)
                V_terms.append(V_term)
                W_terms.append(W_term)
    
    # Part 2: A_{j=2} * B_{j=2}
    for i in range(2):
        for k in range(9):
            U_term = {(i, 2): 1}
            V_term = {(2, k): 1}
            W_term = {(i, k): 1}
            U_terms.append(U_term)
            V_terms.append(V_term)
            W_terms.append(W_term)
    
    rank = len(U_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for (i, j), coeff in U_terms[col].items():
            U[i * m + j][col] = coeff
        for (j, k), coeff in V_terms[col].items():
            V[j * p + k][col] = coeff
        for (i, k), coeff in W_terms[col].items():
            W[k * n + i][col] = coeff
    
    return U, V, W

def _try_reduce_rank_nullspace(U, V, W, n, m, p, deadline):
    """
    Try to reduce rank by finding null space vectors.
    """
    rank = len(U[0]) if U else 0
    current_rank = rank
    
    # We want to find a vector alpha in R^R such that:
    # sum_c alpha_c U[:,c] = 0
    # sum_c alpha_c V[:,c] = 0
    # sum_c alpha_c W[:,c] = 0
    
    # This is a homogeneous system of linear equations.
    # We can stack U, V, W into a single matrix A of size (n*m + m*p + p*n) x R.
    # Then we want to find the null space of A.
    
    A = np.vstack([
        np.array(U).T,
        np.array(V).T,
        np.array(W).T
    ])
    
    # Find null space
    try:
        _, s, vh = np.linalg.svd(A, full_matrices=True)
        # Null space is spanned by rows of vh corresponding to small singular values
        tol = 1e-10
        num_zero = np.sum(s < tol)
        
        if num_zero > 0:
            # We have a null space of dimension num_zero
            # We can remove these dimensions from the decomposition
            # The new decomposition is given by the projection onto the orthogonal complement
            
            # The rows of vh are the right singular vectors.
            # The last num_zero rows span the null space.
            # We want to keep the first R - num_zero rows.
            
            # Actually, we can just drop the columns corresponding to the null space.
            # But we need to be careful: the decomposition is T = sum_c u_c v_c w_c.
            # If we have a null space vector alpha, it means sum_c alpha_c u_c = 0, etc.
            # This doesn't directly mean we can drop a column.
            # It means the decomposition is not unique.
            
            # Let's just return the current decomposition if we don't have a clear reduction strategy.
            # For now, we'll just return the input.
            pass
    except Exception:
        pass
    
    return U, V, W

def construct():
    """
    Returns tensor decomposition for matrix multiplication 2x3 * 3x9.
    """
    start = time.time()
    deadline = start + 145  # Safety margin before 150s kill
    
    try:
        result = _construct_2x3x9()
        return {"U": result[0], "V": result[1], "W": result[2]}
    except Exception as e:
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
