"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Strategy: Use Strassen-based block decomposition (rank 33) as baseline, then search
    for lower-rank decompositions using structured random and greedy methods.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety

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
        result = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            result += np.einsum('i,j,k->ijk', uc, vc, wc)
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
        S_U = [
            [[1, 0], [0, 1]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 1], [0, -1]]
        ]
        S_V = [
            [[1, 0], [0, 1]],
            [[1, 0], [0, 0]],
            [[0, 1], [0, -1]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, 0], [1, 1]]
        ]
        S_W = [
            [[1, 0], [0, 1]],
            [[0, 1], [0, -1]],
            [[0, 0], [1, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 0], [0, 0]]
        ]
        
        # Standard 2x1x2 factors
        Std_U = [
            [[1], [0]],
            [[1], [0]],
            [[0], [1]],
            [[0], [1]]
        ]
        Std_V = [
            [[1, 0]],
            [[0, 1]],
            [[1, 0]],
            [[0, 1]]
        ]
        Std_W = [
            [[1, 0], [0, 0]],
            [[0, 0], [1, 0]],
            [[0, 1], [0, 0]],
            [[0, 0], [0, 1]]
        ]
        
        col = 0
        for b in range(3):
            # Strassen part: handles columns 2b, 2b+1 of B (and corresponding C)
            for r in range(7):
                # Map U (2x2) to global U (rows for i=0,1 and j=0,1)
                for i in range(2):
                    for j in range(2):
                        U[i * m + j, col] = S_U[r][i][j]
                
                # Map V (2x2) to global V (rows for j=0,1 and k=2b, 2b+1)
                for j in range(2):
                    for kp in range(2):
                        k_global = 2 * b + kp
                        V[j * p + k_global, col] = S_V[r][j][kp]
                
                # Map W (2x2) to global W (rows for k=2b, 2b+1 and i=0,1)
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = S_W[r][kp][i]
                
                col += 1
            
            # Standard part: handles column j=2 of A/B, columns 2b, 2b+1 of B
            for r in range(4):
                # Map U (2x1) to global U (rows for i=0,1 and j=2)
                for i in range(2):
                    U[i * m + 2, col] = Std_U[r][i][0]
                
                # Map V (1x2) to global V (row for j=2 and k=2b, 2b+1)
                for kp in range(2):
                    k_global = 2 * b + kp
                    V[2 * p + k_global, col] = Std_V[r][0][kp]
                
                # Map W (2x2) to global W (rows for k=2b, 2b+1 and i=0,1)
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = Std_W[r][kp][i]
                
                col += 1
        
        return U, V, W

    def try_rank_search(max_rank, time_budget):
        """
        Search for a decomposition of rank <= max_rank using random initialization
        and iterative refinement. Returns (U, V, W) if found, else None.
        """
        if time.time() > deadline - 5:
            return None
        
        best_rank = max_rank + 1
        best_U = best_V = best_W = None
        
        for attempt_rank in range(max_rank, 31, -1):  # Try from max_rank down to 32
            if time.time() > deadline - 10:
                break
            
            for trial in range(5):
                if time.time() > deadline - 10:
                    break
                
                # Random sparse initialization
                U = np.zeros((dim_a, attempt_rank), dtype=np.float64)
                V = np.zeros((dim_b, attempt_rank), dtype=np.float64)
                W = np.zeros((dim_c, attempt_rank), dtype=np.float64)
                
                # Initialize with random ±1 values, sparsely
                for r in range(attempt_rank):
                    # Each factor gets a few non-zero entries
                    nz_u = np.random.choice(dim_a, size=min(3, dim_a), replace=False)
                    nz_v = np.random.choice(dim_b, size=min(3, dim_b), replace=False)
                    nz_w = np.random.choice(dim_c, size=min(3, dim_c), replace=False)
                    U[nz_u, r] = np.random.choice([-1, 1], size=len(nz_u))
                    V[nz_v, r] = np.random.choice([-1, 1], size=len(nz_v))
                    W[nz_w, r] = np.random.choice([-1, 1], size=len(nz_w))
                
                # Greedy refinement: try to fix each rank-1 component
                for iter_round in range(50):
                    if time.time() > deadline - 10:
                        break
                    
                    # Compute residual
                    residual = T.astype(np.float64).copy()
                    for r in range(attempt_rank):
                        residual -= np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                    
                    # Check if close to zero
                    if np.allclose(residual, 0, atol=1e-9):
                        # Round to integers
                        U_int = np.round(U).astype(np.int64)
                        V_int = np.round(V).astype(np.int64)
                        W_int = np.round(W).astype(np.int64)
                        if verify_decomposition(U_int, V_int, W_int):
                            return U_int, V_int, W_int
                        break
                    
                    # Try to improve each component
                    for r in range(attempt_rank):
                        if time.time() > deadline - 10:
                            break
                        # Compute contribution of this component
                        contrib = np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                        # The residual should ideally be reduced by adjusting this component
                        # Simple heuristic: nudge towards reducing residual norm
                        grad_U = np.einsum('j,k,ijk->i', V[:, r], W[:, r], residual)
                        grad_V = np.einsum('i,k,ijk->j', U[:, r], W[:, r], residual)
                        grad_W = np.einsum('i,j,ijk->k', U[:, r], V[:, r], residual)
                        
                        # Small step
                        lr = 0.1
                        U[:, r] += lr * grad_U
                        V[:, r] += lr * grad_V
                        W[:, r] += lr * grad_W
                
                # Final check
                residual = T.astype(np.float64).copy()
                for r in range(attempt_rank):
                    residual -= np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                if np.allclose(residual, 0, atol=1e-9):
                    U_int = np.round(U).astype(np.int64)
                    V_int = np.round(V).astype(np.int64)
                    W_int = np.round(W).astype(np.int64)
                    if verify_decomposition(U_int, V_int, W_int):
                        if attempt_rank < best_rank:
                            best_rank = attempt_rank
                            best_U, best_V, best_W = U_int, V_int, W_int
                        return U_int, V_int, W_int
        
        if best_rank < max_rank + 1 and best_U is not None:
            return best_U, best_V, best_W
        return None

    # Start with the known rank-33 decomposition
    best_U, best_V, best_W = strassen_block_decomposition()
    best_rank = 33
    
    # Verify the baseline
    if not verify_decomposition(best_U, best_V, best_W):
        # Fallback to schoolbook if Strassen construction is wrong
        best_U, best_V, best_W = schoolbook()
        best_rank = n * m * p
    else:
        # Try to find a better decomposition
        if time.time() < deadline - 20:
            result = try_rank_search(32, deadline - time.time())
            if result is not None:
                new_U, new_V, new_W = result
                new_rank = new_U.shape[1]
                if new_rank < best_rank:
                    best_U, best_V, best_W = new_U, new_V, new_W
                    best_rank = new_rank

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
