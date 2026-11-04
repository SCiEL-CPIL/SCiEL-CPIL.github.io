"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Strategy: 
    1. Use a structured block decomposition (Strassen + Standard) as a baseline.
    2. Use a fast verification method based on matrix multiplication.
    3. Attempt to find lower ranks using random search with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety

    n, m, p = DIMENSIONS
    dim_a = n * m  # 6
    dim_b = m * p  # 18
    dim_c = p * n  # 12

    # Build the target tensor T[a, b, c]
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify_decomposition(U, V, W):
        """
        Verify that U, V, W exactly decompose the tensor T.
        U: (dim_a, rank)
        V: (dim_b, rank)
        W: (dim_c, rank)
        Condition: sum_c U[a,c] * V[b,c] * W[c,c] == T[a,b,c]
        This is equivalent to:
        For each c, let u_c = U[:,c], v_c = V[:,c], w_c = W[:,c]
        Then sum_c u_c v_c^T w_c^T == T
        We can check this by iterating over c and accumulating into a 3D array,
        or more efficiently by fixing one index and checking matrix products.
        
        Efficient check:
        For each a, T[a,:,:] should equal sum_c U[a,c] * V[:,c] * W[:,c]^T
        Let M_c = V[:,c] * W[:,c]^T (outer product)
        Then T[a,:,:] = sum_c U[a,c] M_c
        
        We can precompute the list of M_c or just loop.
        Given small sizes (6, 18, 12), even a simple loop is fast.
        """
        rank = U.shape[1]
        # We can use einsum or explicit loops.
        # result[a,b,c_idx] = sum_k U[a,k] V[b,k] W[c_idx,k]
        # Let's use einsum for speed if possible, or just numpy operations.
        
        # Construct the sum: sum_k U[:,k, None] * V[:,k, None] * W[:,k, None]
        # This creates a (dim_a, rank, dim_b, rank, dim_c, rank) array if not careful.
        # Better:
        # For each k, term = np.einsum('a,b,c->abc', U[:,k], V[:,k], W[:,k])
        # This is O(rank * dim_a * dim_b * dim_c).
        # 36 * 6 * 18 * 12 = 46656 operations per verify. Very fast.
        
        result = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
        for k in range(rank):
            # u, v, w are vectors of length dim_a, dim_b, dim_c
            # outer product
            result += np.einsum('a,b,c->abc', U[:, k], V[:, k], W[:, k])
        
        return np.array_equal(result, T)

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W

    def strassen_block_decomposition():
        """
        Return rank-33 decomposition.
        Decompose 2x3 * 3x6 into 3 blocks of 2x3 * 3x2.
        For each block (columns 2b, 2b+1 of B):
          We have A (2x3) and B_block (3x2).
          We can split A into A_left (2x2) and A_right (2x1).
          A_left * B_left (2x2 * 2x2) can be done with Strassen (rank 7).
          A_right * B_left (2x1 * 2x2) is standard (rank 4).
          Total per block: 7 + 4 = 11.
          3 blocks * 11 = 33.
        """
        rank = 33
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        
        # Strassen 2x2x2 factors
        # M = A * B
        # U (left), V (right), W (output)
        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        S_U = [
            [[1, 1], [1, 1]],  # A11+A22 -> rows 0,1 cols 0,1? No, U maps A to M inputs.
            # Let's define U such that M_input = U * A_flat?
            # Actually, the tensor decomposition is sum_c U[a,c] V[b,c] W[c_idx,c]
            # For a 2x2*2x2 block:
            # a indexes A (4 elements), b indexes B (4 elements), c_idx indexes C (4 elements)
            # U is 4x7, V is 4x7, W is 4x7
            
            # Let's use the explicit factors from literature or derive them.
            # Strassen factors for 2x2:
            # A is 2x2, B is 2x2, C is 2x2.
            # U (4x7):
            # Col 0: A11+A22 -> [1,0, 0,1] (if A is row-major: 11,12,21,22)
            # Col 1: A21+A22 -> [0,0, 1,1]
            # Col 2: A11     -> [1,0, 0,0]
            # Col 3: A22     -> [0,0, 0,1]
            # Col 4: A11+A12 -> [1,1, 0,0]
            # Col 5: A21-A11 -> [-1,0, 1,0]
            # Col 6: A12-A22 -> [0,1, 0,-1]
            
            [[1, 0, 0, 1],
             [0, 0, 1, 1],
             [1, 0, 0, 0],
             [0, 0, 0, 1],
             [1, 1, 0, 0],
             [-1, 0, 1, 0],
             [0, 1, 0, -1]]
        ]
        # V (4x7):
        # Col 0: B11+B22 -> [1,0, 0,1]
        # Col 1: B11     -> [1,0, 0,0]
        # Col 2: B21-B22 -> [0,1, 0,-1]
        # Col 3: B12-B11 -> [-1,1, 0,0]
        # Col 4: B22     -> [0,0, 0,1]
        # Col 5: B11+B12 -> [1,1, 0,0]
        # Col 6: B21+B22 -> [0,1, 0,1]
        
        S_V = [
            [[1, 0, 0, 1],
             [1, 0, 0, 0],
             [0, 1, 0, -1],
             [-1, 1, 0, 0],
             [0, 0, 0, 1],
             [1, 1, 0, 0],
             [0, 1, 0, 1]]
        ]
        # W (4x7):
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        # W[c_idx, k] is the coefficient of M_k in C_c_idx
        
        S_W = [
            [[1, 0, 0, 1],   # C11: M1(1) + M4(1) - M5(-1) + M7(1) -> wait, indices
             # Let's map columns to M1..M7
             # Col 0 (M1): C11(1), C12(0), C21(0), C22(1)
             # Col 1 (M2): C11(0), C12(0), C21(1), C22(-1)
             # Col 2 (M3): C11(0), C12(1), C21(0), C22(1)
             # Col 3 (M4): C11(1), C12(0), C21(1), C22(0)
             # Col 4 (M5): C11(-1), C12(1), C21(0), C22(0)
             # Col 5 (M6): C11(0), C12(0), C21(0), C22(1)
             # Col 6 (M7): C11(1), C12(0), C21(0), C22(0)
             ],
            [0, 0, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 0, 0],
            [-1, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0]
        ]
        
        # Standard 2x1 * 1x2 factors (Rank 4)
        # A is 2x1, B is 1x2.
        # C11 = A1*B1, C12 = A1*B2, C21 = A2*B1, C22 = A2*B2
        # U (2x4):
        # Col 0: A1 -> [1, 0]
        # Col 1: A1 -> [1, 0]
        # Col 2: A2 -> [0, 1]
        # Col 3: A2 -> [0, 1]
        Std_U = [
            [[1, 1, 0, 0],
             [0, 0, 1, 1]]
        ]
        # V (2x4):
        # Col 0: B1 -> [1, 0]
        # Col 1: B2 -> [0, 1]
        # Col 2: B1 -> [1, 0]
        # Col 3: B2 -> [0, 1]
        Std_V = [
            [[1, 0, 1, 0],
             [0, 1, 0, 1]]
        ]
        # W (4x4):
        # C11: M0(1), others 0
        # C12: M1(1), others 0
        # C21: M2(1), others 0
        # C22: M3(1), others 0
        Std_W = [
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]
        ]
        
        col = 0
        for b in range(3):
            # Block b handles columns 2b, 2b+1 of B (and C)
            # A is full 2x3.
            # We split A into A_left (cols 0,1) and A_right (col 2).
            # B_block is 3x2.
            # We split B_block into B_left (rows 0,1) and B_right (row 2).
            
            # Part 1: A_left (2x2) * B_left (2x2) using Strassen
            for r in range(7):
                # U part: A_left is 2x2. Global A is 2x3.
                # A_left indices: (0,0), (0,1), (1,0), (1,1) -> global 0,1,3,4
                # S_U is 4x7.
                # Map local A indices (0,1,2,3) to global (0,1,3,4)
                # local 0 -> global 0
                # local 1 -> global 1
                # local 2 -> global 3
                # local 3 -> global 4
                U[0, col] = S_U[0][r][0]
                U[1, col] = S_U[0][r][1]
                U[3, col] = S_U[0][r][2]
                U[4, col] = S_U[0][r][3]
                
                # V part: B_left is 2x2 (rows 0,1 of B_block).
                # B_block cols are 2b, 2b+1.
                # Global B indices for rows 0,1 and cols 2b, 2b+1:
                # (0, 2b), (0, 2b+1), (1, 2b), (1, 2b+1)
                # Global: 0*p + 2b, 0*p + 2b+1, 1*p + 2b, 1*p + 2b+1
                # p=6. So: 2b, 2b+1, 6+2b, 6+2b+1
                V[2*b, col] = S_V[0][r][0]
                V[2*b+1, col] = S_V[0][r][1]
                V[6+2*b, col] = S_V[0][r][2]
                V[6+2*b+1, col] = S_V[0][r][3]
                
                # W part: C_block is 2x2 (rows 2b, 2b+1, cols 0,1).
                # Global C indices: (k, i) -> k*n + i.
                # k in {2b, 2b+1}, i in {0,1}.
                # (2b, 0), (2b, 1), (2b+1, 0), (2b+1, 1)
                # Global: 2b*2+0, 2b*2+1, (2b+1)*2+0, (2b+1)*2+1
                W[2*(2*b)+0, col] = S_W[0][r][0]
                W[2*(2*b)+1, col] = S_W[0][r][1]
                W[2*(2*b+1)+0, col] = S_W[0][r][2]
                W[2*(2*b+1)+1, col] = S_W[0][r][3]
                
                col += 1
            
            # Part 2: A_right (2x1) * B_right (1x2) using Standard
            # A_right is col 2 of A. Global indices: (0,2)->2, (1,2)->5
            # B_right is row 2 of B_block. Global indices: (2, 2b)->12+2b, (2, 2b+1)->12+2b+1
            # C_block is 2x2. Same as above.
            
            for r in range(4):
                # U part: A_right (2x1)
                # Std_U is 2x4.
                # local 0 -> global 2
                # local 1 -> global 5
                U[2, col] = Std_U[0][r][0]
                U[5, col] = Std_U[0][r][1]
                
                # V part: B_right (1x2)
                # Std_V is 2x4.
                # local 0 -> global 12+2b
                # local 1 -> global 12+2b+1
                V[12+2*b, col] = Std_V[0][r][0]
                V[12+2*b+1, col] = Std_V[0][r][1]
                
                # W part: C_block (2x2)
                # Std_W is 4x4.
                W[2*(2*b)+0, col] = Std_W[0][r][0]
                W[2*(2*b)+1, col] = Std_W[0][r][1]
                W[2*(2*b+1)+0, col] = Std_W[0][r][2]
                W[2*(2*b+1)+1, col] = Std_W[0][r][3]
                
                col += 1
        
        return U, V, W

    # Start with the known rank-33 decomposition
    best_U, best_V, best_W = strassen_block_decomposition()
    best_rank = 33
    
    # Verify the baseline
    if not verify_decomposition(best_U, best_V, best_W):
        # Fallback to schoolbook if Strassen construction is wrong
        best_U, best_V, best_W = schoolbook()
        best_rank = n * m * p

    # Compute lower bounds from matricizations
    T1 = T.reshape(dim_a, dim_b * dim_c)
    rank1 = np.linalg.matrix_rank(T1)
    T2 = T.reshape(dim_b, dim_a * dim_c)
    rank2 = np.linalg.matrix_rank(T2)
    T3 = T.reshape(dim_c, dim_a * dim_b)
    rank3 = np.linalg.matrix_rank(T3)
    lower_bound = max(rank1, rank2, rank3)

    # Search for lower ranks
    found_improvement = False
    np.random.seed(42)
    
    # Try ranks from lower_bound to best_rank - 1
    for r in range(lower_bound, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try random search for this rank
        max_attempts = 5000
        attempts = 0
        while not found_improvement and attempts < max_attempts and time.time() < deadline - 10:
            attempts += 1
            
            # Strategy: Random sparse integer vectors
            # Most entries are 0, some are +/-1
            # Density depends on rank
            # For small rank, higher density might be needed?
            # Let's try density 0.2 to 0.5
            
            density = 0.3
            U = np.zeros((dim_a, r), dtype=np.int64)
            V = np.zeros((dim_b, r), dtype=np.int64)
            W = np.zeros((dim_c, r), dtype=np.int64)
            
            for c in range(r):
                # Generate random non-zero positions
                n_nonzero_U = np.random.randint(1, 4)
                n_nonzero_V = np.random.randint(1, 4)
                n_nonzero_W = np.random.randint(1, 4)
                
                idx_U = np.random.choice(dim_a, n_nonzero_U, replace=False)
                idx_V = np.random.choice(dim_b, n_nonzero_V, replace=False)
                idx_W = np.random.choice(dim_c, n_nonzero_W, replace=False)
                
                U[idx_U, c] = np.random.choice([-1, 1], n_nonzero_U)
                V[idx_V, c] = np.random.choice([-1, 1], n_nonzero_V)
                W[idx_W, c] = np.random.choice([-1, 1], n_nonzero_W)
            
            if verify_decomposition(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                found_improvement = True
                break
            
            # Occasionally try denser random
            if attempts % 10 == 0:
                U = np.random.randint(-1, 2, size=(dim_a, r))
                V = np.random.randint(-1, 2, size=(dim_b, r))
                W = np.random.randint(-1, 2, size=(dim_c, r))
                if verify_decomposition(U, V, W):
                    best_U, best_V, best_W = U, V, W
                    best_rank = r
                    found_improvement = True
                    break

    # If we found an improvement, we could try to go even lower, but time is limited.
    # Return the best found.
    
    # Convert to lists for the interface
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
