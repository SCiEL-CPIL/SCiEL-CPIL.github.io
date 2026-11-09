"""Find rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [3, 3, 8]

def construct():
    """Find rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Index helpers
    def idx_u(i, j):
        return i * m + j
    
    def idx_v(j, k):
        return j * p + k
    
    def idx_w(k, i):
        return k * n + i
    
    # Precompute index arrays for vectorized operations
    i_idx = np.array([i for i in range(n) for j in range(m) for k in range(p)])
    j_idx = np.array([j for i in range(n) for j in range(m) for k in range(p)])
    k_idx = np.array([k for i in range(n) for j in range(m) for k in range(p)])
    
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    # Schoolbook decomposition as baseline
    schoolbook_rank = n * m * p  # 72
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[idx_u(i, j), col] = 1.0
                V_sb[idx_v(j, k), col] = 1.0
                W_sb[idx_w(k, i), col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = schoolbook_rank
    
    # Helper to compute tensor value from U, V, W
    def compute_tensor(U, V, W):
        """Compute the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]"""
        r = U.shape[1]
        # Reshape for vectorized computation
        T = np.zeros((n, m, p))
        for r_idx in range(r):
            u_r = U[:, r_idx]
            v_r = V[:, r_idx]
            w_r = W[:, r_idx]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
        return T
    
    # Helper to verify decomposition
    def verify(U, V, W, tol=1e-9):
        T = compute_tensor(U, V, W)
        return np.allclose(T, 1.0, atol=tol)
    
    # Entry sets to try
    entry_sets = [
        [-1.0, 0.0, 1.0],
        [-1.0, -0.5, 0.0, 0.5, 1.0],
        [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
    ]
    
    # Try ranks from lower bound up
    lower_bound = max(n * m, m * p, p * n)  # 24
    
    # For each candidate rank, try to find a decomposition
    for candidate_rank in range(lower_bound, min(lower_bound + 30, schoolbook_rank + 1)):
        if time.time() - start_time > deadline - 30:
            break
        
        found = False
        
        # Try different entry sets and random seeds
        for entry_set in entry_sets:
            if time.time() - start_time > deadline - 20:
                break
            
            # Try multiple random restarts
            for seed in range(20):
                if time.time() - start_time > deadline - 15:
                    break
                
                np.random.seed(seed * 1000 + candidate_rank)
                
                U = np.zeros((n * m, candidate_rank))
                V = np.zeros((m * p, candidate_rank))
                W = np.zeros((p * n, candidate_rank))
                
                # Residual tensor
                residual = np.ones((n, m, p))
                
                success = True
                
                for step in range(candidate_rank):
                    if time.time() - start_time > deadline - 10:
                        success = False
                        break
                    
                    # Find best rank-1 component to add
                    best_score = -1
                    best_u, best_v, best_w = None, None, None
                    
                    # Try structured patterns first
                    # Pattern 1: all ones with sign flips
                    # Pattern 2: random from entry set
                    
                    # Generate candidate components
                    candidates = []
                    
                    # Add some structured candidates
                    # All ones
                    candidates.append((np.ones(n*m), np.ones(m*p), np.ones(p*n)))
                    
                    # Random candidates
                    for _ in range(30):
                        u = np.random.choice(entry_set, size=(n * m,))
                        v = np.random.choice(entry_set, size=(m * p,))
                        w = np.random.choice(entry_set, size=(p * n,))
                        candidates.append((u, v, w))
                    
                    # Evaluate each candidate
                    for u, v, w in candidates:
                        # Compute the tensor contribution
                        contrib = np.zeros((n, m, p))
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    contrib[i, j, k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
                        
                        # Score: how well does this reduce the residual?
                        # We want contrib to be close to residual
                        # Use negative L1 distance as score
                        diff = residual - contrib
                        score = -np.sum(np.abs(diff))
                        
                        if score > best_score:
                            best_score = score
                            best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
                    
                    if best_u is None:
                        success = False
                        break
                    
                    # Add this component
                    U[:, step] = best_u
                    V[:, step] = best_v
                    W[:, step] = best_w
                    
                    # Update residual
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                residual[i, j, k] -= best_u[i*m+j] * best_v[j*p+k] * best_w[k*n+i]
                    
                    # Check if we've satisfied all terms
                    if np.allclose(residual, 0, atol=1e-9):
                        if step + 1 < best_rank:
                            best_rank = step + 1
                            best_U = U[:, :step + 1].copy()
                            best_V = V[:, :step + 1].copy()
                            best_W = W[:, :step + 1].copy()
                            found = True
                        break
                    elif step == candidate_rank - 1:
                        # Didn't reach zero residual
                        success = False
                        break
                
                if found:
                    break
            
            if found:
                break
        
        if found:
            break
    
    # Verify the best decomposition found
    if best_rank < schoolbook_rank:
        verified = verify(best_U, best_V, best_W)
        
        if verified:
            # Convert to lists with integer/half-integer entries
            U_list = [[round(x, 1) for x in row] for row in best_U.tolist()]
            V_list = [[round(x, 1) for x in row] for row in best_V.tolist()]
            W_list = [[round(x, 1) for x in row] for row in best_W.tolist()]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return schoolbook decomposition
    U_list = U_sb.tolist()
    V_list = V_sb.tolist()
    W_list = W_sb.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
