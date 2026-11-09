# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Search for a rank-minimal exact tensor decomposition for 2x3x3 matrix multiplication.
    Uses NumPy for fast verification and searches for rank 17 first, then falls back to 18."""
    deadline = time.time() + 140  # Safety margin
    n, m, p = 2, 3, 3
    
    # Target tensor indices
    # a = i*m + j, b = j*p + k, c = k*n + i
    non_zero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                non_zero_indices.append((a, b, c))
    
    # Total tensor size
    total_size = (n * m) * (m * p) * (p * n)
    # Create a flat array of all indices, or just check the specific ones?
    # It's better to compute the full tensor and check it, as it's small (324 elements).
    
    def verify_np(U, V, W):
        """U, V, W are numpy arrays of shape (dim, rank)."""
        # Compute tensor T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        T = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check non-zero entries
        for a, b, c in non_zero_indices:
            if T[a, b, c] != 1.0:
                return False
        
        # Check zero entries
        # We can check the whole tensor for zeros, excluding the non-zero ones.
        # Mask of non-zero indices
        mask = np.zeros(T.shape, dtype=bool)
        for a, b, c in non_zero_indices:
            mask[a, b, c] = True
            
        # Check if all non-masked entries are 0
        if not np.allclose(T[~mask], 0.0):
            return False
            
        return True

    # Schoolbook rank 18 solution (fallback)
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return U, V, W, rank

    U_sb, V_sb, W_sb, rank_sb = get_schoolbook()
    
    # Search for rank 17
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    if time.time() < deadline - 5:
        # Try rank 17
        target_rank = 17
        # Use random search with integer/half-integer values
        # Values: -1, 0, 1
        # We can also try -2, 2
        
        np.random.seed(42)
        
        # To increase chances, we can try different value sets
        value_sets = [
            [-1, 0, 1],
            [-2, -1, 0, 1, 2],
            [-0.5, 0, 0.5],
            [-1.5, -0.5, 0, 0.5, 1.5]
        ]
        
        for vals in value_sets:
            if time.time() > deadline - 5:
                break
            for trial in range(20000):
                if time.time() > deadline - 5:
                    break
                
                # Generate random matrices
                # Shape: (6, 17), (9, 17), (6, 17)
                U_try = np.random.choice(vals, size=(n * m, target_rank))
                V_try = np.random.choice(vals, size=(m * p, target_rank))
                W_try = np.random.choice(vals, size=(p * n, target_rank))
                
                if verify_np(U_try, V_try, W_try):
                    best_U, best_V, best_W, best_rank = U_try, V_try, W_try, target_rank
                    break
            if best_rank < rank_sb:
                break
    
    # Convert to lists for the required output format
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
