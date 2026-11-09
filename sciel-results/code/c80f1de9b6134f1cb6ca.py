import numpy as np
import time
import random

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s limit
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.int64)
        V = np.zeros((m * p, rank), dtype=np.int64)
        W = np.zeros((p * n, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    def check_decomposition(U, V, W, rank):
        """Vectorized check: Sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k"""
        # Reshape to 3D
        U3 = U.reshape(n, m, rank)  # U[i,j,c]
        V3 = V.reshape(m, p, rank)  # V[j,k,c]
        W3 = W.reshape(p, n, rank)  # W[k,i,c]
        
        # For each (i,j,k), compute sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        # We can do this by iterating over c and accumulating, or using einsum
        # result[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        
        # Use einsum: sum over c of U[i,j,c] * V[j,k,c] * W[k,i,c]
        # This is a bit tricky with einsum because the indices are permuted
        # Let's just do it with a loop over c for clarity and correctness
        result = np.zeros((n, m, p), dtype=np.int64)
        for c in range(rank):
            # U3[:, :, c] has shape (n, m)
            # V3[:, :, c] has shape (m, p)  
            # W3[:, :, c] has shape (p, n)
            # We need: for each i,j,k: U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
            # This is an outer-product-like combination
            # result[i,j,k] += U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
            Uc = U3[:, :, c]  # (n, m)
            Vc = V3[:, :, c]  # (m, p)
            Wc = W3[:, :, c]  # (p, n)
            
            # For each i: Uc[i,:] * Wc[:,i] gives (m,) * (p,) -> need outer
            # Actually: for fixed i,j,k: Uc[i,j] * Vc[j,k] * Wc[k,i]
            # Let's compute this as: sum over j of (Uc[i,j] * Vc[j,k]) * Wc[k,i]
            # = (Uc @ Vc)[i,k] * Wc[k,i]... no, that's not right either
            # 
            # Let me just do it directly with broadcasting
            # Uc[i,j] * Vc[j,k] * Wc[k,i]
            # Shape: (n,m) * (m,p) * (p,n) -> need to align indices
            # result[i,j,k] = Uc[i,j] * Vc[j,k] * Wc[k,i]
            
            # Use np.einsum: 'ij,jk,ki->ijk'
            contrib = np.einsum('ij,jk,ki->ijk', Uc, Vc, Wc, optimize=True)
            result += contrib
        
        # Check if all entries equal 1
        return np.all(result == 1)
    
    def random_attempt(rank, seed_offset=0):
        """Try a random integer decomposition."""
        np.random.seed(42 + seed_offset + rank * 1000)
        U = np.random.randint(-2, 3, size=(n * m, rank))
        V = np.random.randint(-2, 3, size=(m * p, rank))
        W = np.random.randint(-2, 3, size=(p * n, rank))
        return U, V, W
    
    def structured_attempt(rank, attempt_idx):
        """Try structured initializations based on known patterns."""
        np.random.seed(123 + attempt_idx * 7919 + rank * 31)
        
        # Strategy 1: Start from a scaled-down version of schoolbook
        # Strategy 2: Use Strassen-like block structure
        # Strategy 3: Sparse random with higher magnitude
        
        if attempt_idx % 3 == 0:
            # Sparse random: most entries 0, some ±1, ±2
            U = np.zeros((n * m, rank), dtype=np.int64)
            V = np.zeros((m * p, rank), dtype=np.int64)
            W = np.zeros((p * n, rank), dtype=np.int64)
            density = 0.3
            for c in range(rank):
                mask_u = np.random.random(n * m) < density
                mask_v = np.random.random(m * p) < density
                mask_w = np.random.random(p * n) < density
                U[mask_u, c] = np.random.choice([-2, -1, 1, 2], size=mask_u.sum())
                V[mask_v, c] = np.random.choice([-2, -1, 1, 2], size=mask_v.sum())
                W[mask_w, c] = np.random.choice([-2, -1, 1, 2], size=mask_w.sum())
        elif attempt_idx % 3 == 1:
            # Denser random
            U = np.random.randint(-1, 2, size=(n * m, rank))
            V = np.random.randint(-1, 2, size=(m * p, rank))
            W = np.random.randint(-1, 2, size=(p * n, rank))
        else:
            # Very sparse
            U = np.zeros((n * m, rank), dtype=np.int64)
            V = np.zeros((m * p, rank), dtype=np.int64)
            W = np.zeros((p * n, rank), dtype=np.int64)
            for c in range(rank):
                idx_u = np.random.choice(n * m, size=max(1, n * m // 5), replace=False)
                idx_v = np.random.choice(m * p, size=max(1, m * p // 5), replace=False)
                idx_w = np.random.choice(p * n, size=max(1, p * n // 5), replace=False)
                U[idx_u, c] = np.random.choice([-1, 1], size=len(idx_u))
                V[idx_v, c] = np.random.choice([-1, 1], size=len(idx_v))
                W[idx_w, c] = np.random.choice([-1, 1], size=len(idx_w))
        
        return U, V, W
    
    # Search from high rank down
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Try a range of ranks. Given the difficulty, focus on ranks close to 75 first
    # where random search has a better chance, then try lower
    ranks_to_try = list(range(74, 60, -1)) + list(range(60, 45, -5))
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        max_attempts = 50 if rank >= 70 else 30 if rank >= 60 else 15
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            if attempt < max_attempts // 2:
                U, V, W = random_attempt(rank, attempt)
            else:
                U, V, W = structured_attempt(rank, attempt)
            
            if check_decomposition(U, V, W, rank):
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break
        
        if best_rank < n * m * p:
            # Found something, try to go lower but don't spend too much time
            if rank <= 55:
                break
    
    if best_rank < n * m * p and best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        U, V, W, _ = schoolbook()
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
