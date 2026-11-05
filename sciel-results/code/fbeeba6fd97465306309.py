<<<<<<< SEARCH
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    dim_a = n * m  # 6
    dim_b = m * p  # 18
    dim_c = p * n  # 12

    # Build the target tensor T[a, b, c]
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the tensor T."""
        rank = U.shape[1]
        # Use einsum for efficient verification
        result = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.array_equal(result, T)
=======
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = DIMENSIONS
    dim_a = n * m  # 6
    dim_b = m * p  # 18
    dim_c = p * n  # 12

    # Build the target tensor T[a, b, c]
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the tensor T."""
        rank = U.shape[1]
        # Use einsum for efficient verification
        result = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.array_equal(result, T)

    def kron_decomposition():
        """
        Attempt a Kronecker product based decomposition.
        Structure: A (2x3) * B (3x6) = C (2x6)
        
        We can view this as: for each output position (i,k), 
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        Using Kronecker product structure with the middle dimension j=0,1,2:
        We can try to factor using the fact that m=3 is prime.
        
        Strategy: Use a structured decomposition where we group terms by
        the relationship between i and k in the output.
        """
        # Try rank-28 decomposition using structured approach
        # Group the 36 schoolbook terms into structured blocks
        
        # First, let's try a rank-30 approach with some optimization
        rank = 30
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        
        # Use the structure: split B into two 3x3 blocks
        # Block 1: B[:, 0:3], Block 2: B[:, 3:6]
        # For each block, use an optimized 2x3 by 3x3 multiplication
        
        # 2x3 by 3x3: we can use a hybrid approach
        # Split A into A[:,0:2] (2x2) and A[:,2:3] (2x1)
        # Split B into B[0:2,:] (2x3) and B[2:3,:] (1x3)
        
        col = 0
        for block in range(2):
            # Part 1: A[:,0:2] * B[0:2, 3*block:3*block+3]
            # This is 2x2 by 2x3
            # Use 8 multiplications (standard for 2x2 by 2x3)
            for r in range(8):
                if r < 4:
                    # Diagonal terms: A[i,0]*B[0,k] and A[i,1]*B[1,k]
                    i = r // 2
                    k_local = r % 2
                    # U: A[i, 0] for i in {0,1}
                    U[i * m + 0, col] = 1 if r % 2 == 0 else 0
                    U[(i+1) * m + 0, col] = 0 if r % 2 == 0 else 0
                    # V: B[0 or 1, k_local]
                    V[0 * p + 3*block + k_local, col] = 1 if r % 2 == 0 else 0
                    V[1 * p + 3*block + k_local, col] = 0 if r % 2 == 0 else 1
                    # W: C[i, 3*block + k_local]
                    W[(3*block + k_local) * n + i, col] = 1
                else:
                    # Cross terms for the 2x2 by 2x3
                    r2 = r - 4
                    i = r2 // 2
                    k_local = r2 % 2
                    # A[i,0]*B[1,k] and A[i,1]*B[0,k]
                    U[i * m + 0, col] = 1 if r2 % 2 == 0 else 0
                    U[i * m + 1, col] = 0 if r2 % 2 == 0 else 1
                    V[1 * p + 3*block + k_local, col] = 1 if r2 % 2 == 0 else 0
                    V[0 * p + 3*block + k_local, col] = 0 if r2 % 2 == 0 else 1
                    W[(3*block + k_local) * n + i, col] = 1
                col += 1
            
            # Part 2: A[:,2:3] * B[2:3, 3*block:3*block+3]
            # This is 2x1 by 1x3 = 6 multiplications
            for i in range(2):
                for k_local in range(3):
                    U[i * m + 2, col] = 1
                    V[2 * p + 3*block + k_local, col] = 1
                    W[(3*block + k_local) * n + i, col] = 1
                    col += 1
        
        return U, V, W

    def structured_search(rank_target):
        """
        Attempt to find a decomposition of the given rank using structured patterns.
        Uses the symmetry and structure of the problem.
        """
        if time.time() > deadline - 30:
            return None
        
        U = np.zeros((dim_a, rank_target), dtype=np.int64)
        V = np.zeros((dim_b, rank_target), dtype=np.int64)
        W = np.zeros((dim_c, rank_target), dtype=np.int64)
        
        # Try various structured patterns
        # Pattern 1: Group by output column
        for col in range(rank_target):
            if time.time() > deadline - 25:
                return None
            
            # Assign a structured pattern to this column
            # Each column represents one "multiplication" in the algorithm
            # We need to choose which (i,j) pairs from A and (j,k) pairs from B
            # contribute to which (i,k) outputs
            
            # Use a greedy approach: find the most "connected" triple
            # For simplicity, use a round-robin assignment
            
            i = (col * 3) % n
            j = (col * 2) % m
            k = (col * 5) % p
            
            U[i * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + i, col] = 1
        
        if verify_decomposition(U, V, W):
            return U, V, W
        
        return None
>>>>>>> REPLACE

