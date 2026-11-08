import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # First, try to find a rank-21 decomposition via systematic search
    if time.time() < deadline - 30:
        result = _try_systematic_rank_21(n, m, p, deadline)
        if result is not None:
            return result
    
    # Fallback to rank-22
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val

    # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

    # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

    # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
    add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
    add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
    add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    # T[i,j,k] should be 1 for all valid (i,j,k)
    # i = row_a * m + col_a, j = row_b * p + col, k = col * n + row_b
    # Wait, let me re-read the convention: W[k*n+i, r]
    # So W index is k*n + i where i is row of A (0..n-1), k is col of B (0..p-1)
    # U index: i*m + j where i is row of A, j is col of A (= row of B)
    # V index: j*p + k where j is row of B, k is col of B
    # 
    # The tensor T[i,j,k] = 1 if the multiplication contributes, which is always 1
    # for the standard matrix multiplication tensor.
    #
    # Actually, the matrix multiplication tensor is T_{(i,j),(j',k),(i',k')} = δ_{j,j'}δ_{i,i'}
    # But in the flattened form: U[i*m+j, r], V[j'*p+k, r], W[k'*n+i', r]
    # The product sum_r U[i*m+j,r]*V[j'*p+k,r]*W[k'*n+i',r] should equal δ_{j,j'}δ_{i,i'}
    
    for i in range(n):
        for j in range(m):
            for j_prime in range(m):
                for k in range(p):
                    for i_prime in range(n):
                        for k_prime in range(p):
                            val = sum(U[i*m+j][r] * V[j_prime*p+k][r] * W[k_prime*n+i_prime][r] for r in range(rank))
                            expected = 1.0 if (j == j_prime and i == i_prime) else 0.0
                            if abs(val - expected) > 1e-9:
                                return False
    return True


def _try_systematic_rank_21(n, m, p, deadline):
    """
    Try to find a rank-21 decomposition using a structured approach.
    
    Key idea: The matrix multiplication tensor for M_{2,3,4} has a specific structure.
    Let's try to find a decomposition where the 8 outer-product terms for the 
    "cross" block can be reduced to 7 by exploiting linear dependencies.
    
    The cross terms are:
    A[0,2]*B[2,0:4] -> 4 terms
    A[1,2]*B[2,0:4] -> 4 terms
    Total: 8 terms
    
    Can we reduce this to 7? Only if there's a linear dependency among these 8 rank-1 tensors.
    But these are all distinct rank-1 tensors (different support patterns), so they're linearly independent.
    
    Alternative: try a different block decomposition entirely.
    
    Let me try: split B into B[:, 0:2] and B[:, 2:4] (each 3x2)
    A[0:2, 0:2] * B[:, 0:2] -> use 7-term Strassen-like
    A[0:2, 2:3] * B[:, 0:2] -> 4 terms (outer product)
    A[0:2, 0:2] * B[:, 2:4] -> use 7-term Strassen-like  
    A[0:2, 2:3] * B[:, 2:4] -> 4 terms (outer product)
    Total: 7+4+7+4 = 22
    
    What if I use a different decomposition for the 2x3 * 3x2 part?
    The tensor for M_{2,3,2} has been studied. Let me check if rank < 11 is possible.
    
    Actually, let me try a completely different approach: use the fact that 
    M_{2,3,4} can be decomposed using the "laser method" or by finding 
    a rank-21 solution through careful algebraic manipulation.
    
    For now, let's try a numerical optimization approach with a time limit.
    """
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)]  # 6 x 21
    V = [[0] * rank for _ in range(m * p)]  # 12 x 21
    W = [[0] * rank for _ in range(p * n)]  # 8 x 21
    
    # Initialize with a structured guess based on the rank-22 solution
    # but try to merge two terms
    
    # Start with the rank-22 solution and see if we can zero out one column
    # by adjusting others
    
    # Actually, let's try a random restart approach with verification
    import random
    random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
        
        # Initialize with random values, then try to refine
        U_init = np.array(U, dtype=float)
        V_init = np.array(V, dtype=float)
        W_init = np.array(W, dtype=float)
        
        # Set up the problem: we need sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
        # for all valid (i,j,k)
        
        # This is a system of polynomial equations. Let's try gradient descent.
        # But it's highly non-convex. Let's try a simpler approach:
        # fix U and V, solve for W linearly, then fix U and W, solve for V, etc.
        
        # Actually, for a given U and V, W can be solved linearly:
        # W[k,r] = T[i,j,k] / (U[i,r]*V[j,r]) for the active terms
        # This doesn't work directly because of the sum.
        
        # Let's try: initialize U, V randomly, then for each k, solve for W[k,:]
        # using the constraint that sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
        
        # This is underdetermined (21 unknowns, up to 24 equations per k)
        # So we can use least squares.
        
        U_arr = np.random.randn(n * m, rank) * 0.1
        V_arr = np.random.randn(m * p, rank) * 0.1
        
        # For each output index k (0..7), solve for W[k,:]
        W_arr = np.zeros((p * n, rank))
        
        for k in range(p * n):
            # k = col_b * n + row_a, where col_b in 0..3, row_a in 0..1
            col_b = k // n
            row_a = k % n
            
            # We need: sum_r U[row_a*m + col_a, r] * V[row_b*p + col_b, r] * W[k, r] = delta(col_a, row_b)
            # for all col_a in 0..2, row_b in 0..2
            
            # This gives us 6 equations for W[k,:] (21 unknowns)
            # Let's set up the linear system: A * W[k,:] = b
            # where A[col_a*m + row_b, r] = U[row_a*m + col_a, r] * V[row_b*p + col_b, r]
            # and b[col_a*m + row_b] = 1 if col_a == row_b else 0
            
            A_k = np.zeros((m * m, rank))
            b_k = np.zeros(m * m)
            for col_a in range(m):
                for row_b in range(m):
                    A_k[col_a * m + row_b, :] = U_arr[row_a * m + col_a, :] * V_arr[row_b * p + col_b, :]
                    b_k[col_a * m + row_b] = 1.0 if col_a == row_b else 0.0
            
            # Solve least squares
            if np.linalg.matrix_rank(A_k) > 0:
                W_k, residuals, rank_k, sv = np.linalg.lstsq(A_k, b_k, rcond=None)
                W_arr[k, :] = W_k
        
        # Now check the error
        max_err = 0
        for i in range(n):
            for j in range(m):
                for j_prime in range(m):
                    for k in range(p):
                        for i_prime in range(n):
                            for k_prime in range(p):
                                val = sum(U_arr[i*m+j, r] * V_arr[j_prime*p+k, r] * W_arr[k_prime*n+i_prime, r] for r in range(rank))
                                expected = 1.0 if (j == j_prime and i == i_prime) else 0.0
                                max_err = max(max_err, abs(val - expected))
        
        if max_err < 1e-6:
            # Found a solution! Round to half-integers
            U_rounded = np.round(U_arr * 2) / 2
            V_rounded = np.round(V_arr * 2) / 2
            W_rounded = np.round(W_arr * 2) / 2
            
            # Verify the rounded solution
            max_err_rounded = 0
            for i in range(n):
                for j in range(m):
                    for j_prime in range(m):
                        for k in range(p):
                            for i_prime in range(n):
                                for k_prime in range(p):
                                    val = sum(U_rounded[i*m+j, r] * V_rounded[j_prime*p+k, r] * W_rounded[k_prime*n+i_prime, r] for r in range(rank))
                                    expected = 1.0 if (j == j_prime and i == i_prime) else 0.0
                                    max_err_rounded = max(max_err_rounded, abs(val - expected))
            
            if max_err_rounded < 1e-9:
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
    
    return None
