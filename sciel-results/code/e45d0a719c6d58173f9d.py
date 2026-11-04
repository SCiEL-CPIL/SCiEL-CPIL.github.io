"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]
import time
import random
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    """Find a low-rank decomposition for 2x4 * 4x4 matrix multiplication."""
    n, m, p = 2, 4, 4  # A is n x m, B is m x p, C is n x p
    # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    # W[k*n+i, r] convention
    
    # The tensor T[i,j,k, i',j',k'] = delta_{j,k'} * delta_{j',k}
    # We need: sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = delta_{j,k'} * delta_{j',k}
    
    # For n=2, m=4, p=4:
    # U has 8 rows, V has 16 rows, W has 8 rows
    # We want to minimize rank
    
    # Try rank 24 first, then try to reduce
    # Known: schoolbook is 32. Let's try to find better.
    
    # Strategy: Use a structured decomposition
    # For 2xm * mxm, we can potentially do better than 2m^2
    
    # Let's try a direct construction for rank 24
    # Idea: Group the 4 middle indices and use shared multiplications
    
    # Actually, let me try a computational search for small ranks
    # Start with a known good decomposition
    
    # For 2x4 * 4x4, try rank 24
    # Construction: 
    # C[i][k] = sum_j A[i][j] B[j][k]
    # = A[i][0]B[0][k] + A[i][1]B[1][k] + A[i][2]B[2][k] + A[i][3]B[3][k]
    
    # We can try to share some multiplications between the two rows of A
    # Let's define new variables to reduce the count
    
    # Try rank 24: use 24 bilinear forms
    # Each bilinear form is a product of a linear form in A and a linear form in B
    
    # Let me try a specific construction:
    # For each k (0..3), C[0][k] and C[1][k] share the B[j][k] factors
    # C[0][k] = A[0][0]B[0][k] + A[0][1]B[1][k] + A[0][2]B[2][k] + A[0][3]B[3][k]
    # C[1][k] = A[1][0]B[0][k] + A[1][1]B[1][k] + A[1][2]B[2][k] + A[1][3]B[3][k]
    
    # If we define S[j][k] = B[j][k], we still need 4 mults per row per k
    # That's 8*4 = 32 total. To reduce, we need to share.
    
    # Alternative: Use the fact that we can combine terms
    # Let's try: for each pair of j values, create combined terms
    
    # Actually, let me just do a search for a valid decomposition of rank r
    # for r from 20 to 32, and return the smallest that works
    
    def check_decomposition(U, V, W, rank):
        """Check if the decomposition is valid."""
        n, m, p = 2, 4, 4
        # For all i,i',j,j',k,k':
        # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] should equal delta_{j,k'}*delta_{j',k}
        for i in range(n):
            for ip in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            for kp in range(p):
                                expected = (1 if j == kp and jp == k else 0)
                                actual = 0
                                for c in range(rank):
                                    actual += U[i*m+j][c] * V[jp*p+k][c] * W[kp*n+ip][c]
                                if abs(actual - expected) > 1e-9:
                                    return False
        return True
    
    def try_rank(rank):
        """Try to find a decomposition of given rank using random search."""
        n, m, p = 2, 4, 4
        # Random initialization with 0, 1, -1
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Initialize with random ±1 values
        for _ in range(50):
            U = np.random.choice([-1, 0, 1], size=(n*m, rank))
            V = np.random.choice([-1, 0, 1], size=(m*p, rank))
            W = np.random.choice([-1, 0, 1], size=(p*n, rank))
            
            if check_decomposition(U.tolist(), V.tolist(), W.tolist(), rank):
                return U.tolist(), V.tolist(), W.tolist()
        
        return None
    
    # Try ranks from 20 to 32
    for rank in range(20, 33):
        result = try_rank(rank)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
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
