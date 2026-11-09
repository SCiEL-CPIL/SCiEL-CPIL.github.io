"""Optimized search for low-rank exact decomposition of 3x4x6 matrix multiplication."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 6]

def _construct_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _build_tensor(n, m, p):
    """Build the flattened tensor T[i, j, k] = 1 if the product C[i,k] uses A[i,j]B[j,k]."""
    tensor = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i*m + j, j*p + k, k*n + i] = 1
    return tensor

def _is_exact_decomposition(U, V, W, tensor, tol=1e-10):
    """Check if U, V, W exactly decompose the tensor."""
    try:
        T = np.einsum('ac,bc,dc->abd', U, V, W)
    except Exception:
        return False
    diff = np.abs(T - tensor)
    return np.all(diff < tol)

def _find_3x2x2_decomposition(rank_target=8):
    """
    Search for a low-rank decomposition of 3x2x2 matrix multiplication.
    A (3x2) * B (2x2) -> C (3x2).
    """
    n, m, p = 3, 2, 2
    tensor = _build_tensor(n, m, p)
    
    # Aggressive random search with small integers
    for seed in range(3000):
        random.seed(seed)
        np.random.seed(seed)
        
        U = np.random.choice([-1, 0, 1], size=(n*m, rank_target))
        V = np.random.choice([-1, 0, 1], size=(m*p, rank_target))
        W = np.random.choice([-1, 0, 1], size=(p*n, rank_target))
        
        if _is_exact_decomposition(U, V, W, tensor):
            return {"U": U, "V": V, "W": W}
    
    # Try with wider range
    for seed in range(3000):
        random.seed(seed + 50000)
        np.random.seed(seed + 50000)
        
        U = np.random.choice([-2, -1, 0, 1, 2], size=(n*m, rank_target))
        V = np.random.choice([-2, -1, 0, 1, 2], size=(m*p, rank_target))
        W = np.random.choice([-2, -1, 0, 1, 2], size=(p*n, rank_target))
        
        if _is_exact_decomposition(U, V, W, tensor):
            return {"U": U, "V": V, "W": W}
    
    return None

def _construct_rank48_decomposition():
    """
    Construct a rank-48 decomposition for 3x4x6.
    Strategy:
    1. Split p=6 into 3 blocks of 2.
    2. For each block, split m=4 into 2 blocks of 2.
    3. Use a rank-8 decomposition for 3x2x2 blocks.
    Total rank = 3 (k-blocks) * 2 (j-blocks) * 8 = 48.
    """
    n, m, p = 3, 4, 6
    
    # Find a 3x2x2 decomposition
    block_decomp = _find_3x2x2_decomposition(8)
    if block_decomp is None:
        return None

    U_block = block_decomp["U"] # 6x8
    V_block = block_decomp["V"] # 4x8
    W_block = block_decomp["W"] # 6x8
    
    rank = 48
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    offset = 0
    for k_block in range(3):
        k_start = k_block * 2
        for j_block in range(2):
            j_start = j_block * 2
            
            for r in range(8):
                c = offset + r
                
                for i in range(3):
                    for j_local in range(2):
                        j_global = j_start + j_local
                        U[i * 4 + j_global, c] = U_block[i * 2 + j_local, r]
                
                for j_local in range(2):
                    for k_local in range(2):
                        j_global = j_start + j_local
                        k_global = k_start + k_local
                        V[j_global * 6 + k_global, c] = V_block[j_local * 2 + k_local, r]
                
                for k_local in range(2):
                    for i in range(3):
                        k_global = k_start + k_local
                        W[k_global * 3 + i, c] = W_block[k_local * 3 + i, r]
            
            offset += 8
            
    return {"U": U, "V": V, "W": W}

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    schoolbook = _construct_schoolbook()
    best_rank = n * m * p
    best_decomp = schoolbook
    
    # Build the target tensor for verification
    tensor = _build_tensor(n, m, p)
    
    # Try rank-48 construction
    if time.time() < deadline - 20:
        result = _construct_rank48_decomposition()
        if result is not None:
            # Verify
            if _is_exact_decomposition(result["U"], result["V"], result["W"], tensor):
                best_rank = 48
                best_decomp = {"U": result["U"].tolist(), "V": result["V"].tolist(), "W": result["W"].tolist()}
                return best_decomp
    
    # Fallback to schoolbook if no improvement found
    return best_decomp

if __name__ == "__main__":
    res = construct()
    print(res["U"][0])
