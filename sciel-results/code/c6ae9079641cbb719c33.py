import time
import random
import numpy as np
from itertools import product

def verify_decomposition(U, V, W, n=2, m=2, p=7):
    """
    Verify if U, V, W form a valid decomposition of the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    """
    n, m, p = 2, 2, 7
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin before 150s kill

    n, m, p = 2, 2, 7
    schoolbook_rank = n * m * p  # 28

    # Schoolbook construction (rank 28)
    def make_schoolbook():
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W

    best_U, best_V, best_W = make_schoolbook()
    best_rank = schoolbook_rank

    # For 2x2x7, the tensor has a special structure:
    # T[i,j,k] = delta_{i,k} for i in {0,1}, j in {0,1}, k in {0,1,...,6}
    # This means for each k, we have a 2x2x1 tensor (matrix-vector product)
    # The key insight: we can potentially share computations across k values
    # by using the same "intermediate" values.
    
    # Let's try a structured approach:
    # For rank R, we have R terms. Each term r has:
    #   U[0,r], U[1,r], U[2,r], U[3,r]  (for i*m+j where i,j in {0,1})
    #   V[0,r], ..., V[13,r]            (for j*p+k where j in {0,1}, k in {0,...,6})
    #   W[0,r], ..., W[13,r]            (for k*n+i where k in {0,...,6}, i in {0,1})
    
    # The constraint is: for each (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # Let's try to find a rank 27 solution by starting from schoolbook
    # and trying to merge two terms.
    
    # Actually, let's try a more direct approach: use the fact that
    # the tensor is a sum of 7 independent 2x2x1 tensors (one for each k).
    # Each 2x2x1 tensor has rank 4 (schoolbook) or possibly 3?
    # For 2x2x1: T[i,j,k] = delta_{i,k}
    # This is the tensor for multiplying a 2x2 matrix by a 2x1 vector.
    # The rank of this tensor is 4 (it's just the identity embedding).
    # Wait, let me reconsider. The tensor T_{i,j,k} where i,j,k in {0,1}
    # with T[i,j,k] = 1 if i=k, else 0.
    # This can be written as: sum_i sum_j e_i ⊗ e_j ⊗ e_i
    # = e_0⊗e_0⊗e_0 + e_0⊗e_1⊗e_0 + e_1⊗e_0⊗e_1 + e_1⊗e_1⊗e_1
    # This is a sum of 4 rank-1 tensors. Can it be done with 3?
    # The border rank of 2x2x1 is 4, and the rank is also 4.
    # So each k-block requires rank 4.
    # For 7 blocks, that's 28 total.
    
    # But wait, we can potentially share the A-matrix part across k values!
    # If we use the same U values for different k's, we might save rank.
    
    # Let's try: suppose we use rank R terms. For each term r, we have
    # u0_r, u1_r, u2_r, u3_r for the A-matrix.
    # For each k, we need: sum_r u_{i,j,r} * v_{j,k,r} * w_{k,i,r} = delta_{i,k}
    
    # This is a system of 28 equations in 4R + 14R + 14R = 32R variables.
    # For R=27, we have 864 variables and 28 equations. Very underdetermined.
    # The challenge is finding integer/half-integer solutions.
    
    # Let's try a randomized search with a structured initialization.
    # Start from schoolbook and try to perturb it.
    
    def try_rank(R, max_iters=5000):
        """Try to find a rank R decomposition using randomized search."""
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Initialize with schoolbook-like structure
            # Use small random integers
            U = [[random.randint(-2, 2) for _ in range(R)] for _ in range(4)]
            V = [[random.randint(-2, 2) for _ in range(R)] for _ in range(14)]
            W = [[random.randint(-2, 2) for _ in range(R)] for _ in range(14)]
            
            # Verify
            if verify_decomposition(U, V, W):
                return U, V, W
        
        return None
    
    # Try rank 27
    result = try_rank(27, max_iters=1000)
    if result:
        best_U, best_V, best_W = result
        best_rank = 27
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # If random search fails, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
