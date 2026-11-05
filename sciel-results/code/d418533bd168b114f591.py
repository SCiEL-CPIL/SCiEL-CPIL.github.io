"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Strategy: Use Strassen-based block decomposition (rank 33) as baseline,
    then attempt structured improvements.
    """
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin for safety

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
        """Verify that U, V, W exactly decompose the tensor T using matrix mult."""
        rank = U.shape[1]
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Check via matricization: T_1[a, (b,c)] = sum_r U[a,r] * (V[:,r] ⊗ W[:,r])
        # Faster: for each a, check if T[a,:,:].reshape(-1) is in span of {V[:,r]*W[:,r].reshape(-1)}
        # Even faster: reconstruct and compare
        result = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
        for r in range(rank):
            result += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        return np.array_equal(result, T)

    def verify_fast(U, V, W):
        """Faster verification using matricization rank check + spot check."""
        rank = U.shape[1]
        # Reconstruct T from U,V,W
        result = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
        for r in range(rank):
            result += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
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
        Return rank-33 decomposition: 3 blocks of (Strassen 2x2x2 + Standard 2x1x2).
        Each block handles a 2x2 sub-block of the 3x6 B matrix plus the remaining column.
        """
        rank = 33
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        
        # Strassen 2x2x2 factors
        S_U = np.array([
            [[1, 0], [0, 1]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 1], [0, -1]]
        ], dtype=np.int64)
        S_V = np.array([
            [[1, 0], [0, 1]],
            [[1, 0], [0, 0]],
            [[0, 1], [0, -1]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, 0], [1, 1]]
        ], dtype=np.int64)
        S_W = np.array([
            [[1, 0], [0, 1]],
            [[0, 1], [0, -1]],
            [[0, 0], [1, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 0], [0, 0]]
        ], dtype=np.int64)
        
        # Standard 2x1x2 factors
        Std_U = np.array([
            [[1], [0]],
            [[1], [0]],
            [[0], [1]],
            [[0], [1]]
        ], dtype=np.int64)
        Std_V = np.array([
            [[1, 0]],
            [[0, 1]],
            [[1, 0]],
            [[0, 1]]
        ], dtype=np.int64)
        Std_W = np.array([
            [[1, 0], [0, 0]],
            [[0, 0], [1, 0]],
            [[0, 1], [0, 0]],
            [[0, 0], [0, 1]]
        ], dtype=np.int64)
        
        col = 0
        for b in range(3):
            # Strassen part: handles columns 2b, 2b+1 of B (and corresponding C)
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i * m + j, col] = S_U[r][i][j]
                for j in range(2):
                    for kp in range(2):
                        k_global = 2 * b + kp
                        V[j * p + k_global, col] = S_V[r][j][kp]
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = S_W[r][kp][i]
                col += 1
            
            # Standard part: handles column j=2 of A/B, columns 2b, 2b+1 of B
            for r in range(4):
                for i in range(2):
                    U[i * m + 2, col] = Std_U[r][i][0]
                for kp in range(2):
                    k_global = 2 * b + kp
                    V[2 * p + k_global, col] = Std_V[r][0][kp]
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = Std_W[r][kp][i]
                col += 1
        
        return U, V, W

    # Start with the known rank-33 decomposition
    best_U, best_V, best_W = strassen_block_decomposition()
    best_rank = 33
    
    # Verify the baseline
    if not verify_decomposition(best_U, best_V, best_W):
        best_U, best_V, best_W = schoolbook()
        best_rank = n * m * p

    # Compute lower bounds from matricizations
    T1 = T.reshape(dim_a, dim_b * dim_c).astype(np.float64)
    rank1 = np.linalg.matrix_rank(T1)
    T2 = T.reshape(dim_b, dim_a * dim_c).astype(np.float64)
    rank2 = np.linalg.matrix_rank(T2)
    T3 = T.reshape(dim_c, dim_a * dim_b).astype(np.float64)
    rank3 = np.linalg.matrix_rank(T3)
    lower_bound = max(rank1, rank2, rank3)

    # Try to find a better decomposition with structured search
    # For this small problem, try a limited number of structured approaches
    found_improvement = False
    
    if lower_bound < best_rank and time.time() < deadline - 20:
        # Try rank = lower_bound through best_rank-1
        for r in range(lower_bound, best_rank):
            if time.time() > deadline - 15:
                break
            
            # Structured search: try to find a valid decomposition of rank r
            # Use the fact that the tensor is small - try random {-1,0,1} with bias
            max_attempts = 500
            attempts = 0
            while not found_improvement and attempts < max_attempts and time.time() < deadline - 15:
                attempts += 1
                
                # Generate sparse random candidates
                U = np.zeros((dim_a, r), dtype=np.int64)
                V = np.zeros((dim_b, r), dtype=np.int64)
                W = np.zeros((dim_c, r), dtype=np.int64)
                
                for col_idx in range(r):
                    # Each column should have a few non-zeros
                    nz_u = np.random.randint(1, 4)
                    nz_v = np.random.randint(1, 5)
                    nz_w = np.random.randint(1, 4)
                    
                    u_idx = np.random.choice(dim_a, size=nz_u, replace=False)
                    v_idx = np.random.choice(dim_b, size=nz_v, replace=False)
                    w_idx = np.random.choice(dim_c, size=nz_w, replace=False)
                    
                    U[u_idx, col_idx] = np.random.choice([-1, 1], size=nz_u)
                    V[v_idx, col_idx] = np.random.choice([-1, 1], size=nz_v)
                    W[w_idx, col_idx] = np.random.choice([-1, 1], size=nz_w)
                
                if verify_decomposition(U, V, W):
                    best_U, best_V, best_W = U, V, W
                    best_rank = r
                    found_improvement = True
                    break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
